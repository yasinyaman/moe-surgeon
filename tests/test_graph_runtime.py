# SPDX-License-Identifier: Apache-2.0
"""Offline coverage for the piecewise CUDA-graph MoERunner substitution.

The load-bearing correctness question -- does late splitting_ops injection actually
let the tier run under piecewise graphs and stay token-identical -- is CUDA-only and
answered on GB10. What is checkable here without vLLM or a GPU is the output
stabilisation arithmetic: that :meth:`_stable_copy` copies into a *persistent* buffer
keyed by (role, width, dtype), reuses it across calls, and refuses (returns the input
untouched) when a shape will not fit the capture-sized buffer. That refusal is the
difference between a correct fallback and a silent write past the buffer.
"""

from __future__ import annotations

import pytest

from vllm_moe_surgeon.compat import graph_runtime

torch = pytest.importorskip("torch")


class _FakeBase:
    """Stands in for vLLM's MoERunner: records construction, returns a sentinel."""

    def __init__(self, *args, **kwargs):
        self.constructed = (args, kwargs)

    def _forward_impl(self, *args, **kwargs):
        return ("stock-output", args, kwargs)


def _make_runner(monkeypatch, *, enabled: bool):
    """Build a SurgeonMoERunner over the fake base, with read_config() stubbed so
    __init__ never reaches vLLM (get_current_vllm_config)."""

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.enabled = enabled
    monkeypatch.setattr(graph_runtime, "read_config", lambda: cfg, raising=False)
    # read_config is imported inside __init__ from .runtime; patch there too.
    from vllm_moe_surgeon.compat import runtime

    monkeypatch.setattr(runtime, "read_config", lambda: cfg)
    runner_cls = graph_runtime.make_surgeon_moe_runner(_FakeBase)
    return runner_cls()


def test_subclass_is_built_over_the_installed_runner():
    runner_cls = graph_runtime.make_surgeon_moe_runner(_FakeBase)
    assert issubclass(runner_cls, _FakeBase)
    assert runner_cls.__name__ == "SurgeonMoERunner"


def test_disabled_tier_leaves_stabilisation_off_and_forwards_untouched(monkeypatch):
    runner = _make_runner(monkeypatch, enabled=False)
    # Tier off -> __init__ returns before importing any vLLM config symbol.
    assert runner._stable_moe_output is False
    assert runner._stable_output_rows == 0
    # _forward_impl must still delegate to the base and, with stabilisation off,
    # return its result verbatim (the early return dodges the vLLM imports too).
    out = runner._forward_impl("a", k=1)
    assert out == ("stock-output", ("a",), {"k": 1})


def test_stable_copy_reuses_one_persistent_buffer_per_key(monkeypatch):
    runner = _make_runner(monkeypatch, enabled=False)
    runner._stable_moe_output = True
    runner._stable_output_rows = 8

    a = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    out1 = runner._stable_copy(a, 0)
    assert torch.equal(out1, a)
    # The returned view must be backed by the persistent buffer, not the input.
    assert out1.data_ptr() != a.data_ptr()
    buf = runner._stable_output_bufs[(0, 3, torch.float32)]
    assert out1.data_ptr() == buf.data_ptr()

    # A second, smaller call with the same key reuses the very same buffer.
    b = torch.full((2, 3), 7.0)
    out2 = runner._stable_copy(b, 0)
    assert runner._stable_output_bufs[(0, 3, torch.float32)] is buf
    assert out2.data_ptr() == buf.data_ptr()
    assert torch.equal(out2, b)


def test_stable_copy_keys_role_and_width_apart(monkeypatch):
    runner = _make_runner(monkeypatch, enabled=False)
    runner._stable_moe_output = True
    runner._stable_output_rows = 8

    t = torch.zeros(2, 3)
    runner._stable_copy(t, 0)
    runner._stable_copy(t, 1)
    runner._stable_copy(torch.zeros(2, 5), 0)
    assert set(runner._stable_output_bufs) == {
        (0, 3, torch.float32),
        (1, 3, torch.float32),
        (0, 5, torch.float32),
    }


def test_stable_copy_refuses_shapes_that_would_not_fit(monkeypatch):
    runner = _make_runner(monkeypatch, enabled=False)
    runner._stable_moe_output = True
    runner._stable_output_rows = 4

    # More rows than the capture-sized buffer: must return the input untouched and
    # allocate nothing, rather than write past a 4-row buffer.
    too_many = torch.zeros(5, 3)
    out = runner._stable_copy(too_many, 0)
    assert out is too_many
    assert runner._stable_output_bufs == {}

    # A non-2D tensor is equally refused (the kernels only ever hand us [rows, dim]).
    not_2d = torch.zeros(2, 3, 4)
    assert runner._stable_copy(not_2d, 0) is not_2d
    assert runner._stable_output_bufs == {}


def _vllm_importable() -> bool:
    import importlib.util

    return importlib.util.find_spec("vllm") is not None


def test_installed_predicate_tracks_the_registry():
    # graph_runtime_installed() must reflect the real registry state, and must not
    # raise when vLLM is absent. It is what keeps validate() demanding
    # --enforce-eager until the substitution is actually in place.
    if not _vllm_importable():
        # Offline: the import guard fires, so the predicate reports not-installed
        # and install_graph() is a tolerant no-op.
        assert graph_runtime.graph_runtime_installed() is False
        assert graph_runtime.install_graph() is False
        return
    # vLLM present: the predicate equals whether MoERunner is registered. Do NOT
    # call install_graph() here -- it mutates the process-global op_registry_oot,
    # which the dedicated GB10 boot test owns; this test only reads.
    from vllm.model_executor.custom_op import op_registry_oot

    assert graph_runtime.graph_runtime_installed() == ("MoERunner" in op_registry_oot)


def test_split_active_requires_the_op_not_just_the_registration():
    """Registration is not the invariant that makes non-eager safe -- the split is.

    ``_inject_moe_splitting`` declines silently when there is no piecewise compilation
    to split into, and in that case the MoE op is captured together with the cache's
    dynamic host code. That configuration produced garbage output (one token id
    repeated) in the GB10 control run, so ``validate`` must refuse it. It can only do
    that by checking the *effect* -- vllm::moe_forward present in splitting_ops -- and
    not merely that the runner class got registered.
    """
    class _CC:
        def __init__(self, ops):
            self.splitting_ops = ops

    class _Cfg:
        def __init__(self, ops):
            self.compilation_config = _CC(ops)

    # No vLLM in the offline env, so registration is False and the answer is False
    # regardless -- which is itself the safe direction.
    if not graph_runtime.graph_runtime_installed():
        assert graph_runtime.moe_split_active(_Cfg(["vllm::moe_forward"])) is False
        return

    assert graph_runtime.moe_split_active(_Cfg(None)) is False
    assert graph_runtime.moe_split_active(_Cfg([])) is False
    assert graph_runtime.moe_split_active(_Cfg(["vllm::unified_attention"])) is False
    assert graph_runtime.moe_split_active(_Cfg(["vllm::moe_forward"])) is True


class _StabilisingBase:
    """A base MoERunner that stabilises its own output, as the disk-tier fork's does.

    The fork computes these attributes for its *in-tree* cache, gated on splitting_ops
    rather than on which cache is enabled. Our subclass is registered unconditionally
    (at plugin time no VllmConfig exists, so it cannot be gated on the tier), so it also
    wraps runs that are not ours -- and must leave them exactly as it found them.
    """

    def __init__(self, *args, **kwargs):
        self._stable_moe_output = True
        self._stable_output_rows = 512
        self._stable_output_bufs = {}

    def _forward_impl(self, *args, **kwargs):
        return "base-output"


def test_disabled_tier_does_not_clobber_a_base_that_stabilises_itself(monkeypatch):
    """The regression that would silently disable the in-tree cache's stabilisation.

    Overwriting the base's flags on the disabled path leaves the *other* cache running
    under piecewise graphs with no capture-stable address -- the configuration the GB10
    control run showed producing garbage (one token id repeated per prompt). Merely
    installing this package must not do that.
    """
    class _Cfg:
        enabled = False

    from vllm_moe_surgeon.compat import runtime

    monkeypatch.setattr(runtime, "read_config", lambda: _Cfg())
    runner = graph_runtime.make_surgeon_moe_runner(_StabilisingBase)()

    assert runner._stable_moe_output is True
    assert runner._stable_output_rows == 512
    assert runner._stable_output_bufs == {}


def test_disabled_tier_supplies_defaults_when_the_base_has_none(monkeypatch):
    """Stock vLLM's MoERunner sets no such attributes, so they must be supplied or the
    overridden _forward_impl raises AttributeError on the first token."""
    class _Cfg:
        enabled = False

    from vllm_moe_surgeon.compat import runtime

    monkeypatch.setattr(runtime, "read_config", lambda: _Cfg())
    runner = graph_runtime.make_surgeon_moe_runner(_FakeBase)()

    assert runner._stable_moe_output is False
    assert runner._forward_impl("x") == ("stock-output", ("x",), {})


def test_injection_declines_when_there_are_no_stock_split_points(monkeypatch):
    """Empty splitting_ops means attention is inside one full graph. Injecting the MoE
    ops there yields PIECEWISE carrying only those ops, which vLLM's CudagraphDispatcher
    asserts on at worker init -- blaming its own compilation settings rather than us.

    Calls the real ``_inject_moe_splitting`` rather than restating its condition, which
    is the difference between a test and a tautology; needs vLLM for CompilationMode.
    """
    if not _vllm_importable():
        pytest.skip("needs vLLM for CompilationMode / CUDAGraphMode")

    from vllm.config import CompilationMode, CUDAGraphMode

    class _MC:
        enforce_eager = False

    class _CC:
        def __init__(self, ops):
            self.splitting_ops = ops
            self.mode = CompilationMode.VLLM_COMPILE
            self.cudagraph_mode = CUDAGraphMode.FULL
            self.max_cudagraph_capture_size = 512

    class _Cfg:
        def __init__(self, ops):
            self.model_config = _MC()
            self.compilation_config = _CC(ops)
            self.additional_config = {"surgeon": {"expert_cache_size": 8}}

    for empty in (None, []):
        cfg = _Cfg(empty)
        graph_runtime._inject_moe_splitting(cfg)
        assert not cfg.compilation_config.splitting_ops, (
            "injected MoE ops into a config with no stock split points"
        )
        assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.FULL, (
            "downgraded to PIECEWISE without carving anything out"
        )

    # With stock split points present the injection does its job.
    cfg = _Cfg(["vllm::unified_attention"])
    graph_runtime._inject_moe_splitting(cfg)
    assert "vllm::moe_forward" in cfg.compilation_config.splitting_ops
    assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
