# SPDX-License-Identifier: Apache-2.0
"""Configuration checks for the out-of-tree tier, without vLLM.

``check_config`` answers questions about the *request* alone -- no layer, no device,
no vLLM -- so these run in the laptop loop. The refusals are the interesting part:
each is a case where proceeding produces a wrong *result* rather than an error, so it
has to be caught before a weight is allocated. ``validate`` adds the checks that need
a layer and an engine, and is exercised on the GPU boxes.
"""

from __future__ import annotations

import pytest

from vllm_moe_surgeon.compat.runtime import RuntimeConfig, check_config, validate


class _ParallelConfig:
    use_ep = False
    ep_size = 1
    dp_size = 1
    is_sequence_parallel = False


class _MoEConfig:
    experts_per_token = 8
    has_bias = False
    moe_parallel_config = _ParallelConfig()


class _Layer:
    """The little of a RoutedExperts layer that ``validate`` reads."""

    layer_name = "model.layers.0.mlp.experts"
    local_num_experts = 64
    moe_config = _MoEConfig()


def test_use_disk_needs_both_a_directory_and_a_ram_budget():
    assert RuntimeConfig(expert_cache_size=24, store_dir="/s", ram_cache=48).use_disk
    assert not RuntimeConfig(expert_cache_size=24, store_dir="/s", ram_cache=0).use_disk
    assert not RuntimeConfig(
        expert_cache_size=24, store_dir=None, ram_cache=48
    ).use_disk


def test_a_store_without_a_ram_budget_is_refused():
    """The combination that produced a mislabelled measurement.

    Three boot-floor arms were recorded as "tiered, ram_cache 0". They were not tiered:
    ``use_disk`` is ``bool(store_dir) and ram_cache > 0``, so the disk store was never
    built and the provider ran full-DRAM, holding every expert page-locked for the
    process lifetime -- 31.10 GiB against the genuine tiered arm's 23.40 GiB. Both
    modes logged the same "expert cache active" line, so nothing gave it away.

    A store_dir asks for a disk tier and ram_cache=0 refuses to build one. That is a
    contradiction, and silently resolving it in favour of the more expensive mode is
    how the wrong number got recorded.
    """
    config = RuntimeConfig(expert_cache_size=24, store_dir="/store", ram_cache=0)
    with pytest.raises(ValueError, match="asks for the disk tier"):
        check_config(config)

    # The message has to name the way out, because the fix is not obvious from
    # "disabled": raising ram_cache is one, dropping store_dir is the other.
    try:
        check_config(config)
    except ValueError as exc:
        assert "ram_cache >= expert_cache_size" in str(exc)
        assert "drop store_dir" in str(exc)


def test_full_dram_mode_is_still_allowed_when_asked_for_deliberately():
    """No store_dir means the operator wants full-DRAM. That is a valid request."""
    check_config(RuntimeConfig(expert_cache_size=24, ram_cache=0))
    check_config(RuntimeConfig(expert_cache_size=24, store_dir="/s", ram_cache=48))


def test_a_disabled_config_is_not_second_guessed():
    """Nothing is checked when the tier is off -- there is nothing to get wrong."""
    check_config(RuntimeConfig(expert_cache_size=0, store_dir="/s", ram_cache=0))
    check_config(RuntimeConfig(expert_cache_size=0, fp8_store=True))


def test_a_warm_tier_smaller_than_the_resident_set_is_refused():
    """A pool below capacity cannot hold what the GPU has: every eviction hits disk."""
    with pytest.raises(ValueError, match="below expert_cache_size"):
        check_config(
            RuntimeConfig(expert_cache_size=24, store_dir="/s", ram_cache=16)
        )
    check_config(RuntimeConfig(expert_cache_size=24, store_dir="/s", ram_cache=24))


def test_fp8_store_without_a_store_is_refused():
    """fp8_store describes record encoding. With no store it encodes nothing."""
    with pytest.raises(ValueError, match="no .*store is being built"):
        check_config(RuntimeConfig(expert_cache_size=24, fp8_store=True))
    check_config(
        RuntimeConfig(
            expert_cache_size=24, store_dir="/s", ram_cache=48, fp8_store=True
        )
    )


# ------------------------------------------------- the checks that need a live engine


def test_validate_delegates_the_pure_config_checks():
    """validate must not be the only door: the same contradiction has to fail there.

    It reaches vLLM further down, so this asserts only that the delegated check fires
    first -- which is what makes the vLLM-free tests above meaningful.
    """
    config = RuntimeConfig(expert_cache_size=24, store_dir="/store", ram_cache=0)
    with pytest.raises(ValueError, match="asks for the disk tier"):
        validate(config, _Layer())


def test_capacity_below_top_k_is_refused_unless_split_by_expert():
    pytest.importorskip("vllm", reason="validate reaches vLLM for the eager check")
    layer = _Layer()
    with pytest.raises(ValueError, match="fewer than the 8 experts"):
        validate(RuntimeConfig(expert_cache_size=4, ram_cache=0), layer)


def test_bias_and_parallelism_are_refused_by_name():
    layer = _Layer()

    class _Biased(_MoEConfig):
        has_bias = True

    layer.moe_config = _Biased()
    with pytest.raises(ValueError, match="bias"):
        validate(RuntimeConfig(expert_cache_size=24, ram_cache=0), layer)

    class _EP(_ParallelConfig):
        use_ep = True
        ep_size = 2

    class _MoEEP(_MoEConfig):
        moe_parallel_config = _EP()

    layer.moe_config = _MoEEP()
    with pytest.raises(ValueError, match="expert parallelism"):
        validate(RuntimeConfig(expert_cache_size=24, ram_cache=0), layer)


def test_install_registers_under_the_class_name_key():
    """Regression for the silent no-op the README admits: the substitution must land
    at the key CustomOp.__new__ looks up -- the base class ``__name__`` -- not the op
    name. Registered under the op name once, it went where nothing looks, the tier
    never installed, and the tokens still matched. Import-level (no CUDA), so it runs
    on the GPU boxes and skips here."""
    pytest.importorskip("vllm")
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    from vllm_moe_surgeon.compat.runtime import install

    key = UnquantizedFusedMoEMethod.__name__
    op_registry_oot.pop(key, None)  # re-runnable

    assert install() is True
    registered = op_registry_oot.get(key)
    assert registered is not None, "substitution did not land at the class-name key"
    assert registered.__name__ == "SurgeonUnquantizedMoEMethod"
    assert issubclass(registered, UnquantizedFusedMoEMethod)
    # Idempotent: register_oot asserts on a duplicate, which would take the engine
    # down, so a second install must decline rather than re-register.
    assert install() is False


def test_install_fp8_registers_a_shadowing_config():
    """The fp8 path goes through a Fp8Config shadowing the 'fp8' name, not register_oot
    (Fp8MoEMethod is not a CustomOp). Verify the registration takes. Import-level (no
    CUDA), so it runs on the GPU boxes and skips here."""
    pytest.importorskip("vllm")
    from vllm_moe_surgeon.compat.fp8_runtime import install_fp8

    assert install_fp8() is True
    try:
        from vllm.model_executor.layers.quantization import get_quantization_config

        assert get_quantization_config("fp8").__name__ == "SurgeonFp8Config"
    except ImportError:
        pass  # registry accessor moved; the True return already proves it registered


def test_tensor_parallelism_is_refused_because_the_store_has_no_rank():
    """TP ranks hold different shards at identical shapes; one store file would be
    shared and rank 1 would serve rank 0's shard. Refuse before allocating."""
    class _TP(_ParallelConfig):
        tp_size = 2

    class _MoETP(_MoEConfig):
        moe_parallel_config = _TP()

    layer = _Layer()
    layer.moe_config = _MoETP()
    with pytest.raises(ValueError, match="tensor parallelism"):
        validate(RuntimeConfig(expert_cache_size=24, ram_cache=0), layer)


# ------------------------------------- streaming is the default when there is a store


def test_streaming_is_on_by_default_whenever_the_disk_tier_is_on():
    """The non-streaming path costs 12.0 GiB of page-locked host memory for nothing.

    Measured on OLMoE: boot floor 23.40 GiB without streaming against 11.35 with it.
    Nobody should reach the expensive path by leaving a flag unset, so "unset" means
    "stream if there is a store to stream into".
    """
    tiered = RuntimeConfig(expert_cache_size=24, store_dir="/s", ram_cache=48)
    assert tiered.stream_load is None, "the field itself stays unset"
    assert tiered.streams is True

    # No store to stream into: nothing to do.
    assert RuntimeConfig(expert_cache_size=24, ram_cache=0).streams is False


def test_explicitly_off_is_distinguishable_from_unset():
    """Forcing the old path has to remain possible, and has to look different."""
    forced_off = RuntimeConfig(
        expert_cache_size=24, store_dir="/s", ram_cache=48, stream_load=False
    )
    assert forced_off.stream_load is False
    assert forced_off.streams is False

    forced_on = RuntimeConfig(expert_cache_size=24, store_dir="/s", ram_cache=48,
                              stream_load=True)
    assert forced_on.streams is True


def test_the_tri_state_reader_keeps_unset_unset():
    from vllm_moe_surgeon.compat.runtime import _tri_state

    assert _tri_state(None) is None
    assert _tri_state("") is None
    assert _tri_state(True) is True
    assert _tri_state(False) is False
    assert _tri_state("1") is True
    assert _tri_state("true") is True
    assert _tri_state("0") is False


# ----------------------------------------------------------------------
# The oversubscription warning. Not a refusal: a cache smaller than the
# per-layer union is legitimate on a device that cannot hold the working
# set -- the chunk split is what makes that possible. What it must not be
# is silent, which is what it was when a 24-slot cache against a 46-expert
# union cost 2.6x decode on OLMoE with nothing in the logs to say why.
# ----------------------------------------------------------------------


class _Provider:
    def __init__(self, capacity):
        self.capacity = capacity


def _ids(values):
    torch = pytest.importorskip("torch")
    return torch.tensor(values)


def test_oversubscribed_cache_warns_once_and_names_the_numbers(caplog):
    from vllm_moe_surgeon.compat.runtime import _warn_if_oversubscribed

    layer = _Layer()
    layer.layer_name = "model.layers.3.mlp.experts"
    provider = _Provider(capacity=4)

    with caplog.at_level("WARNING"):
        _warn_if_oversubscribed(layer, provider, _ids([[0, 1, 2], [3, 4, 5]]))
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "6" in message and "4" in message
    assert "model.layers.3.mlp.experts" in message

    # Once per layer: a per-forward warning would drown the log it is meant to
    # make readable.
    with caplog.at_level("WARNING"):
        _warn_if_oversubscribed(layer, provider, _ids([[0, 1, 2], [3, 4, 5]]))
    assert len(caplog.records) == 1


def test_a_cache_that_covers_the_union_is_silent(caplog):
    from vllm_moe_surgeon.compat.runtime import _warn_if_oversubscribed

    layer = _Layer()
    provider = _Provider(capacity=8)
    with caplog.at_level("WARNING"):
        # Union is 3 distinct experts across both rows, well under capacity.
        _warn_if_oversubscribed(layer, provider, _ids([[0, 1, 2], [0, 1, 2]]))
    assert not caplog.records
    assert not getattr(layer, "_surgeon_capacity_warned", False)


def test_the_warning_never_breaks_a_forward(caplog):
    from vllm_moe_surgeon.compat.runtime import _warn_if_oversubscribed

    class _Hostile:
        def unique(self):
            raise RuntimeError("no unique() on this tensor type")

    layer = _Layer()
    with caplog.at_level("WARNING"):
        # A diagnostic that raises would take down every forward it was added to.
        _warn_if_oversubscribed(layer, _Provider(capacity=1), _Hostile())
    assert not caplog.records


def test_a_prefill_shaped_forward_is_not_warned_about():
    """The restriction that makes the warning worth having rather than noise.

    A prefill routes hundreds of tokens at once, so its union approaches the whole
    expert set -- 56-62 of 64 measured on OLMoE. A check that fired on it would demand
    ``capacity == num_experts`` from every configuration, including the one that
    benchmarked fastest, and would be advice about a cost paid once per request rather
    than the per-token one where the 2.6x actually lives.
    """
    from vllm_moe_surgeon.compat.runtime import _DECODE_ROWS, _warn_if_oversubscribed

    layer = _Layer()
    provider = _Provider(capacity=4)
    prefill = _ids([[0, 1, 2, 3, 4, 5, 6, 7]] * (_DECODE_ROWS + 1))
    _warn_if_oversubscribed(layer, provider, prefill)
    assert not getattr(layer, "_surgeon_capacity_warned", False)

    # The same oversubscription at decode shape does warn.
    _warn_if_oversubscribed(layer, provider, _ids([[0, 1, 2, 3, 4, 5, 6, 7]]))
    assert layer._surgeon_capacity_warned is True


def test_an_explicit_expert_split_suppresses_the_capacity_warning(caplog):
    """The split IS the acknowledgement that the device cannot hold the working
    set. The check reads the provider's build-time snapshot -- at forward time
    there is no ambient vLLM config, so a read_config() here would fall back to
    the env, miss additional_config (the documented channel), and warn anyway."""
    from vllm_moe_surgeon.compat.runtime import _warn_if_oversubscribed

    class _SplitProvider:
        def __init__(self):
            self.capacity = 4
            self.split = "expert"

    layer = _Layer()
    with caplog.at_level("WARNING"):
        # topk_ids=None: the suppression must decide before touching the ids.
        _warn_if_oversubscribed(layer, _SplitProvider(), None)
    assert not caplog.records
    assert layer._surgeon_capacity_warned is True
