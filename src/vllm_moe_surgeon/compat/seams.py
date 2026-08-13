# SPDX-License-Identifier: Apache-2.0
"""The declared contract between this package and vLLM's internals.

Every vLLM name we depend on is listed here as data. Two things consume the
table: :mod:`tests.test_seams`, which resolves each entry against the installed
vLLM, and the version bindings in this package, which do the actual importing.

The point is failure ordering. vLLM guarantees exactly one thing for out-of-tree
code -- that ``ModelRegistry.register_model`` keeps working (``docs/design/
plugin_system.md`` explicitly disclaims the stability of "the interface for the
model/module"). Everything below is therefore borrowed, not owed. When an upgrade
moves one of these, we want a named test failure that says which seam moved and
why we were holding it, not a ``TypeError`` sixty frames deep inside a worker.

Every entry is something the runtime, the profiler or the writer actually touches.
An entry marked ``required=False`` is one we can currently live without -- an
emerging upstream helper we would adopt if present (``_orient_fused_weight``), or a
name held for a path that installs itself only when reached and declines gracefully
when the name has moved (the fp8 method substitution, and the ``MoERunner``
substitution that lifts the ``--enforce-eager`` requirement; see ``DECISIONS.md`` and
:mod:`.graph_runtime`). Those paths degrade rather than crash on a missing name -- fp8
falls back to no fp8 tier, the graph path back to requiring eager -- so their seams are
news, not a broken pin. The table is not the *plan's* wishlist: an
``_apply_quant_method`` override the plan budgeted but the implementation never
needed is not here, because a seam nothing calls trains you to ignore the test. The
``MoERunner`` subclass and ``PluggableLayer`` substitution the plan also budgeted were
unused until the piecewise CUDA-graph path (S5) was built; now that it calls them,
they are here.

Stability tiers, from the survey in the plan:

``documented``
    Covered by vLLM's own docs and CI (plugin groups, ``register_model``,
    ``register_quantization_config``, ``register_model_loader``).
``used_in_tree``
    Real mechanism, exercised in-tree, thinly documented. Mechanism is stable;
    the classes it substitutes are not (``CustomOp.register_oot``).
``internal``
    Private surface. Expected to move. Held anyway because there is no
    alternative -- each of these carries a note saying what breaks without it.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Literal

Kind = Literal["class", "function", "method", "attribute"]
Tier = Literal["documented", "used_in_tree", "internal"]


@dataclass(frozen=True)
class Seam:
    """One vLLM name we hold, and what we hold it for."""

    #: ``"module.path:QualName"``, e.g. ``"vllm.model_executor.custom_op:CustomOp"``
    #: or ``"...:CustomOp.register_oot"``.
    target: str
    kind: Kind
    tier: Tier
    #: What we do with it, and what breaks if it disappears.
    why: str
    #: Parameter names we require to be present on a callable. Deliberately a
    #: subset, not the full signature: requiring an exact match would fail on
    #: every harmless upstream addition, which trains you to ignore the test.
    params: tuple[str, ...] = field(default_factory=tuple)
    #: ``False`` for a seam we would *like* to use but can live without -- one
    #: that upstream has only recently grown, or that we have a fallback for.
    #: Tracking upstream is not only about names disappearing: when upstream
    #: extracts a helper we were duplicating, we want to notice and adopt it.
    #: Optional seams report as informational rather than failing the pin.
    required: bool = True

    @property
    def module(self) -> str:
        return self.target.split(":", 1)[0]

    @property
    def qualname(self) -> str:
        return self.target.split(":", 1)[1]


SEAMS: tuple[Seam, ...] = (
    # ------------------------------------------------------------------
    # Registration mechanism. This is how we get installed at all.
    # ------------------------------------------------------------------
    Seam(
        target="vllm.model_executor.custom_op:CustomOp.register_oot",
        kind="method",
        tier="used_in_tree",
        why=(
            "Called as UnquantizedFusedMoEMethod.register_oot(SurgeonUnquantized...) "
            "to substitute the unquantized MoE method, so the expert cache can "
            "change where expert weights are allocated and what apply() dispatches "
            "to. Without it the unquantized MoE path needs an in-tree patch. "
            "CONTRACT: op_registry_oot is keyed by the *class* name, because "
            "CustomOp.__new__ looks up cls.__name__ -- not by the op name given to "
            "@CustomOp.register. Registering under the op name puts the class where "
            "nothing looks, and the substitution no-ops silently -- with matching "
            "tokens, which is why matching output is not evidence it engaged."
        ),
        params=("name",),
    ),
    Seam(
        target="vllm.model_executor.layers.quantization:register_quantization_config",
        kind="function",
        tier="documented",
        required=False,
        why=(
            "WIRED for the fp8 path (compat/fp8_runtime.py): registers our Fp8Config "
            "under the name 'fp8', shadowing the builtin, so our Fp8MoEMethod is "
            "swapped in -- Fp8MoEMethod is not a CustomOp, so register_oot cannot "
            "reach it. Optional: install_fp8 declines gracefully on absence, so fp8 "
            "moving disables the fp8 tier but never fails the pin or the unquantized "
            "path."
        ),
        params=("quantization",),
    ),
    # ------------------------------------------------------------------
    # Classes. UnquantizedFusedMoEMethod we subclass; RoutedExperts we do NOT
    # subclass -- the layer objects vLLM hands our method are its instances, and
    # we read its attribute surface.
    # ------------------------------------------------------------------
    Seam(
        target="vllm.model_executor.layers.fused_moe.routed_experts:RoutedExperts",
        kind="class",
        tier="internal",
        why=(
            "The layer objects vLLM passes to our quant method are RoutedExperts "
            "instances; we read its attribute surface (w13_weight/w2_weight, "
            "layer_name, local_num_experts, moe_config, activation, "
            "global_num_experts) rather than subclassing it. ~1700 lines; the class "
            "that has already been renamed once (it was FusedMoE)."
        ),
    ),
    Seam(
        target=(
            "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method"
            ":UnquantizedFusedMoEMethod"
        ),
        kind="class",
        tier="internal",
        why="Base class we subclass as SurgeonUnquantizedMoEMethod.",
    ),
    Seam(
        target="vllm.model_executor.layers.quantization.fp8:Fp8Config",
        kind="class",
        tier="internal",
        required=False,
        why="Subclassed (compat/fp8_runtime.py) so get_quant_method hands MoE layers "
        "our Fp8MoEMethod. Optional -- fp8 tier only; install_fp8 declines on absence.",
    ),
    Seam(
        target="vllm.model_executor.layers.quantization.fp8:Fp8MoEMethod",
        kind="class",
        tier="internal",
        required=False,
        why="Subclassed as SurgeonFp8MoEMethod, overriding _setup_kernel and apply. "
        "Optional -- fp8 tier only; install_fp8 declines on absence.",
    ),
    Seam(
        target=(
            "vllm.model_executor.layers.fused_moe.oracle.fp8"
            ":convert_to_fp8_moe_kernel_format"
        ),
        kind="function",
        tier="internal",
        required=False,
        why="Called in the copied _setup_kernel body to shuffle fp8 weights/scales to "
        "runtime format before the cache takes them. Optional -- fp8 tier only.",
    ),
    Seam(
        target=(
            "vllm.model_executor.layers.fused_moe.oracle.fp8:make_fp8_moe_kernel"
        ),
        kind="function",
        tier="internal",
        required=False,
        why="Builds the fp8 MoE kernel in the copied _setup_kernel body. Optional -- "
        "fp8 tier only.",
    ),
    Seam(
        target="vllm.model_executor.layers.fused_moe.oracle.fp8:Fp8MoeBackend",
        kind="class",
        tier="internal",
        required=False,
        why="The AITER-shuffle branch of the copied _setup_kernel body tests against "
        "it. Optional -- fp8 tier only.",
    ),
    # ------------------------------------------------------------------
    # Methods our SurgeonUnquantizedMoEMethod OVERRIDES. A rename here means our
    # override silently stops being called and the stock path runs instead -- the
    # worst failure mode in the package (matching tokens, no cache) and the reason
    # this table exists. All three are defined directly on the base class.
    # ------------------------------------------------------------------
    Seam(
        target=(
            "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method"
            ":UnquantizedFusedMoEMethod.create_weights"
        ),
        kind="method",
        tier="internal",
        why=(
            "Overridden to allocate zero-expert / cpu-pinned parameters and to stamp "
            "the streaming loader onto them. Renamed upstream -> weights allocated "
            "the stock way, the tier never installs, silent."
        ),
        params=("layer", "num_experts", "intermediate_size_per_partition",
                "params_dtype"),
    ),
    Seam(
        target=(
            "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method"
            ":UnquantizedFusedMoEMethod.process_weights_after_loading"
        ),
        kind="method",
        tier="internal",
        why=(
            "Overridden to build the provider and hand-build the MoE kernel instead "
            "of _setup_kernel shuffling the full weights onto the device. Renamed "
            "upstream -> _surgeon_provider is never set, apply() takes its "
            "provider-None fallback to the stock path, silent."
        ),
        params=("layer",),
    ),
    Seam(
        target=(
            "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method"
            ":UnquantizedFusedMoEMethod.apply"
        ),
        kind="method",
        tier="internal",
        why=(
            "Overridden to run the forward through the expert cache. Renamed "
            "upstream -> the stock apply runs on the (released) weights, silent."
        ),
        params=("layer", "topk_ids", "shared_experts"),
    ),
    Seam(
        target=(
            "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method"
            ":make_unquantized_moe_kernel"
        ),
        kind="function",
        tier="internal",
        why=(
            "Imported and called in the overridden process_weights_after_loading to "
            "build the kernel by hand (skipping _setup_kernel's device shuffle). "
            "Gone -> the cache-aware method cannot build its kernel."
        ),
    ),
    Seam(
        target="vllm.model_executor.utils:replace_parameter",
        kind="function",
        tier="internal",
        why=(
            "Releases the full w13/w2 parameters to torch.empty(0) once the provider "
            "has copied what it needs. Gone -> the full expert tensors stay "
            "resident, defeating the tier."
        ),
        params=("layer", "param_name", "new_data"),
    ),
    Seam(
        target="vllm.model_executor.utils:set_weight_attrs",
        kind="function",
        tier="internal",
        why=(
            "Stamps the (possibly stream-wrapped) weight_loader and shard attrs onto "
            "each parameter in the overridden create_weights."
        ),
        params=("weight", "weight_attrs"),
    ),
    Seam(
        target=(
            "vllm.model_executor.layers.fused_moe.routed_experts"
            ":RoutedExperts._orient_fused_weight"
        ),
        kind="method",
        tier="internal",
        required=False,
        why=(
            "Upstream extracted the fused-weight orientation logic that used to "
            "be an inline branch inside weight_loader. Where it exists we should "
            "call it instead of duplicating the transpose rule; where it does not, "
            "our code carries its own copy. Absent at the 0.26.1 merge base, present "
            "~300 commits later -- kept optional so the pinned version still passes."
        ),
    ),
    # ------------------------------------------------------------------
    # Telemetry rides vLLM's own routed-experts capture, so this section
    # holds a CLI flag and an output field instead of a routing hook.
    #
    # The plan budgeted three internal seams here -- a MoERunner subclass,
    # _apply_quant_method, and FusedMoERouter.select_experts -- to read
    # topk_ids out of the forward pass. None are needed:
    # --enable-return-routed-experts already captures per-token, per-layer
    # expert ids and returns them per request. Both are public API.
    # ------------------------------------------------------------------
    Seam(
        target="vllm.config.model:ModelConfig.enable_return_routed_experts",
        kind="attribute",
        tier="documented",
        why=(
            "The flag that turns on per-token routed-expert capture, exposed as "
            "--enable-return-routed-experts. This is the entire telemetry "
            "mechanism; without it we would be back to subclassing MoERunner and "
            "wrapping router.select_experts."
        ),
    ),
    Seam(
        target="vllm.outputs:CompletionOutput.routed_experts",
        kind="attribute",
        tier="documented",
        why=(
            "Where the capture surfaces: [seq_len, num_layers, top_k] expert ids "
            "per finished request, in both the offline LLM path and (base64 "
            "np.save encoded) the OpenAI-compatible response. The read side of "
            "the profiler is ordinary numpy over this field."
        ),
    ),
    Seam(
        target="vllm.model_executor.layers.quantization.fp8:Fp8MoEMethod._setup_kernel",
        kind="method",
        tier="internal",
        required=False,
        why=(
            "RESERVED for the fp8 path (DECISIONS.md, not yet wired). The cache must "
            "be installed here, BEFORE get_fused_moe_quant_config captures the scale "
            "tensors. Getting this order wrong scales every expert by whichever one "
            "occupies its slot -- wrong output from the first token, no exception. "
            "See notes/fp8-duzeltmesi.md."
        ),
    ),
    # ------------------------------------------------------------------
    # Piecewise CUDA-graph path (compat/graph_runtime.py). All optional: install_graph
    # declines on any missing name and the splitting-op injection is wrapped in a
    # try/except, so a rename here costs the non-eager speedup (validate() falls back
    # to requiring --enforce-eager), not correctness. Two pieces: a config-time
    # injection that carves the MoE op out of the captured region (a wrap of
    # VllmConfig.__post_init__, because a plugin cannot edit the config class or add
    # the fork's offload_config field), and a MoERunner substitution that stabilises
    # the eager MoE op's output address. The whole point of the path is to remove the
    # eager requirement, so we still want to see any of these move.
    # ------------------------------------------------------------------
    Seam(
        # Definition module, not the vllm.config re-export: the static check descends
        # into .__post_init__, which needs the ClassDef, not the re-export node.
        target="vllm.config.vllm:VllmConfig.__post_init__",
        kind="method",
        tier="internal",
        required=False,
        why=(
            "Wrapped so that, after the stock post-init defaults splitting_ops, we "
            "append vllm::moe_forward(_shared) when the surgeon tier is on and the "
            "model is not eager -- the fork does this inline, gated on its own "
            "offload_config. Renamed -> our wrap never runs, the MoE op is captured "
            "into the graph, and the tier is stuck on --enforce-eager (the fallback "
            "validate() enforces). Injecting later (from the runner __init__) is too "
            "late: the split points are fixed before the runner is built."
        ),
    ),
    Seam(
        target="vllm.config.compilation:CompilationConfig.mode",
        kind="attribute",
        tier="internal",
        required=False,
        why=(
            "Read to confirm piecewise compilation (VLLM_COMPILE) is actually in play "
            "before appending the MoE splitting op; a non-eager run in another mode "
            "cannot carve the op out, so we leave it for validate() to refuse."
        ),
    ),
    Seam(
        target="vllm.config.compilation:CompilationMode",
        kind="class",
        tier="internal",
        required=False,
        why=(
            "Its VLLM_COMPILE member is the value CompilationConfig.mode is compared "
            "against to decide whether piecewise splitting can carry the MoE op."
        ),
    ),
    Seam(
        target=(
            "vllm.model_executor.layers.fused_moe.runner.moe_runner:MoERunner"
        ),
        kind="class",
        tier="internal",
        required=False,
        why=(
            "Subclassed as SurgeonMoERunner and installed with register_oot so the "
            "MoE op runs eager (with the cache) as a piecewise-graph splitting point "
            "and its output is copied to a capture-stable address. Gone -> the tier "
            "keeps requiring --enforce-eager, no non-eager path."
        ),
    ),
    Seam(
        target=(
            "vllm.model_executor.layers.fused_moe.runner.moe_runner"
            ":MoERunner._forward_impl"
        ),
        kind="method",
        tier="internal",
        required=False,
        why=(
            "Overridden to route the MoE output through _maybe_stabilize_output. The "
            "stock _moe_forward custom op calls layer._forward_impl with no "
            "stabilisation wrap, so the wrap must live in this override. Renamed "
            "upstream -> our stabilisation stops being called and a non-eager run is "
            "a silent capture-address bug, which is why the fallback refuses eager "
            "unless this installs."
        ),
    ),
    Seam(
        target="vllm.model_executor.custom_op:PluggableLayer.register_oot",
        kind="method",
        tier="used_in_tree",
        required=False,
        why=(
            "Called as MoERunner.register_oot(SurgeonMoERunner) to substitute the "
            "runner. This is PluggableLayer's register_oot, a DIFFERENT method from "
            "CustomOp.register_oot above -- both write op_registry_oot, keyed by "
            "class name, but a rename of one is invisible to the other's seam."
        ),
        params=("name",),
    ),
    Seam(
        # Targeted at the definition module, not the vllm.config re-export: we read
        # the field off a CompilationConfig instance (compilation_config), and only
        # the defining module lets the static check see the field declaration.
        target="vllm.config.compilation:CompilationConfig.splitting_ops",
        kind="attribute",
        tier="internal",
        required=False,
        why=(
            "The config-time injection (the VllmConfig.__post_init__ wrap) appends "
            "vllm::moe_forward(_shared) here so the MoE op becomes a graph split "
            "point. Gone or renamed -> the op is captured into the graph and the "
            "cache runs inside it, the exact case the port cannot do."
        ),
    ),
    Seam(
        target="vllm.config.compilation:CompilationConfig.max_cudagraph_capture_size",
        kind="attribute",
        tier="internal",
        required=False,
        why=(
            "Read to size the persistent output buffer (its row count bounds the "
            "shapes we can stabilise). Gone -> no bound, and stabilisation is "
            "skipped for every shape (falls back to returning the workspace view)."
        ),
    ),
    Seam(
        target="vllm.config:CUDAGraphMode",
        kind="class",
        tier="internal",
        required=False,
        why=(
            "PIECEWISE and NONE members gate both the install (skip when NONE) and "
            "the per-pass stabilisation (only on a PIECEWISE runtime pass)."
        ),
    ),
    Seam(
        target="vllm.forward_context:get_forward_context",
        kind="function",
        tier="internal",
        required=False,
        why=(
            "Reads cudagraph_runtime_mode for the current pass, so stabilisation "
            "runs only when the pass actually replays a piecewise graph."
        ),
    ),
    Seam(
        target="vllm.forward_context:is_forward_context_available",
        kind="function",
        tier="internal",
        required=False,
        why=(
            "Guards the get_forward_context read: outside a forward pass (setup, "
            "profiling) there is no context and stabilisation is a no-op."
        ),
    ),
    Seam(
        target="vllm.forward_context:ForwardContext.cudagraph_runtime_mode",
        kind="attribute",
        tier="internal",
        required=False,
        why=(
            "The per-pass field compared against CUDAGraphMode.PIECEWISE to decide "
            "whether the output needs a capture-stable address this pass."
        ),
    ),
    # ------------------------------------------------------------------
    # Config plumbing.
    # ------------------------------------------------------------------
    Seam(
        # Targeted at the import path the code actually uses (``from vllm.config
        # import get_current_vllm_config``), a re-export of vllm.config.vllm's
        # definition, not the definition module -- so the check tracks the name we
        # import, which is the one that can move.
        target="vllm.config:get_current_vllm_config",
        kind="function",
        tier="internal",
        why="Reads additional_config from inside layers that get no config passed.",
    ),
    Seam(
        target="vllm.config:VllmConfig",
        kind="class",
        tier="documented",
        why=(
            "Carries additional_config, our configuration channel. The in-code "
            "comment on that field names out-of-tree config as its purpose."
        ),
    ),
)


#: Behavioural couplings that live OUTSIDE ``compat/`` and import no vLLM, yet still
#: depend on a vLLM format, layout or naming convention as *data*. The layering test
#: proves those modules do not import vLLM; it cannot prove they do not assume things
#: about it, and these are the assumptions. None is machine-checkable by :func:`check`
#: -- the presence of a symbol does not verify a byte layout or a key order -- so they
#: are recorded here for a human to re-check on an upgrade. This is the honest bound on
#: the claim that a vLLM upgrade is confined to ``compat/``: import cost is; these are
#: tracked separately.
DATA_COUPLINGS: tuple[tuple[str, str, str], ...] = (
    (
        "telemetry/transport.py",
        "the base64 + numpy.save wire encoding of CompletionOutput.routed_experts "
        "in vLLM's OpenAI-compatible response",
        "a change to the encoding decodes to garbage, not an error",
    ),
    (
        "store/expert_cpu_exec.py",
        "layer.activation string semantics -- the CPU co-execution path "
        "re-implements 'silu' act-and-mul exactly as the fused kernel computes "
        "it, and assumes topk_weights are NOT folded into the input "
        "(apply_router_weight_on_input=False); validate() refuses both "
        "deviations by name",
        "a changed activation meaning or a new weight-folding mode diverges "
        "host-computed expert outputs from the kernel's, as wrong output "
        "rather than an error",
    ),
    (
        "telemetry/layers.py",
        "RoutedExptsCapturer sizes its buffer by num_hidden_layers and zeroes it each "
        "step, so dense-layer rows read as 'expert 0 chosen by every token'; and the "
        "MoE-layer resolver mirrors vLLM's config key order",
        "a change to the buffer's zero-fill or layer indexing silently mis-aggregates",
    ),
    (
        "surgery/tiered.py",
        "the store key derives from RoutedExperts.layer_name -- the factory prefix "
        "'model.layers.{L}.mlp' plus '.experts'",
        "a rename of the layer-name scheme mis-keys the store; a mismatched store is "
        "then rebuilt rather than reused, or (worse) reused across checkpoints",
    ),
    (
        "hot_experts.py",
        "residency priors are keyed by the same layer_name the provider looks them up "
        "under",
        "a key-scheme change makes every prior a miss -- cold start, not wrong output",
    ),
    (
        "surgery/descriptors.py",
        "the per-expert checkpoint tensor naming "
        "'model.layers.{L}.mlp.experts.{e}.{gate,up,down}_proj.weight' and the stacked "
        "(Granite) layout, which are HF export conventions vLLM also assumes",
        "an unrecognised naming scheme raises KeyError (loud) rather than mis-reads",
    ),
)


@dataclass(frozen=True)
class SeamProblem:
    seam: Seam
    detail: str


def resolve(seam: Seam) -> Any:
    """Import and walk to the seam's target. Raises on failure."""
    obj: Any = importlib.import_module(seam.module)
    for part in seam.qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _has_field(owner: Any, name: str) -> bool:
    """Does ``owner`` declare a field called ``name``?

    Plain ``getattr`` is not enough. vLLM's configs are pydantic dataclasses,
    which move declared fields off the class body, so a field can exist while
    ``getattr(cls, name)`` raises. Check every place a field can hide.
    """
    import dataclasses

    if hasattr(owner, name):
        return True
    if name in getattr(owner, "__annotations__", {}):
        return True
    # pydantic BaseModel / pydantic dataclass
    model_fields = getattr(owner, "model_fields", None)
    if isinstance(model_fields, dict) and name in model_fields:
        return True
    if dataclasses.is_dataclass(owner):
        if any(f.name == name for f in dataclasses.fields(owner)):
            return True
    # Annotations on base classes, for a subclassed config.
    for base in getattr(owner, "__mro__", ())[1:]:
        if name in getattr(base, "__annotations__", {}):
            return True
    return False


def check(seam: Seam) -> SeamProblem | None:
    """Verify one seam resolves and, for callables, carries the params we need."""
    if seam.kind == "attribute":
        # Resolve the owner, then look the field up properly.
        *owner_path, field_name = seam.qualname.split(".")
        if not owner_path:
            return SeamProblem(seam, "attribute seam needs 'Module:Owner.field'")
        owner_seam = Seam(
            target=f"{seam.module}:{'.'.join(owner_path)}",
            kind="class",
            tier=seam.tier,
            why=seam.why,
        )
        try:
            owner = resolve(owner_seam)
        except ModuleNotFoundError as exc:
            return SeamProblem(seam, f"module missing: {exc}")
        except AttributeError as exc:
            return SeamProblem(seam, f"owner missing: {exc}")
        if not _has_field(owner, field_name):
            return SeamProblem(
                seam, f"{'.'.join(owner_path)} has no field {field_name!r}"
            )
        return None

    try:
        obj = resolve(seam)
    except ModuleNotFoundError as exc:
        return SeamProblem(seam, f"module missing: {exc}")
    except AttributeError as exc:
        return SeamProblem(seam, f"name missing: {exc}")

    if seam.kind == "class" and not inspect.isclass(obj):
        return SeamProblem(seam, f"expected a class, got {type(obj).__name__}")
    if seam.kind in ("function", "method") and not callable(obj):
        return SeamProblem(seam, f"expected a callable, got {type(obj).__name__}")

    if seam.params:
        # For a class, the params we care about are its constructor's.
        target = obj.__init__ if inspect.isclass(obj) else obj
        try:
            available = set(inspect.signature(target).parameters)
        except (TypeError, ValueError) as exc:
            return SeamProblem(seam, f"signature unavailable: {exc}")
        missing = [p for p in seam.params if p not in available]
        if missing:
            return SeamProblem(seam, f"parameters gone: {missing}")

    return None


def check_all(include_optional: bool = False) -> list[SeamProblem]:
    """Every required seam that no longer holds.

    An empty list means the installed vLLM is safe to run against. Optional
    seams are excluded by default: a missing optional seam is news, not a break.
    """
    return [
        problem
        for seam in SEAMS
        if (include_optional or seam.required) and (problem := check(seam)) is not None
    ]


# ----------------------------------------------------------------------
# Static checking, for hosts with no vLLM installed.
#
# Importing vLLM drags in torch, CUDA probing and a compiled extension, so on a
# development laptop the import-based check above cannot run at all. Parsing the
# source tree instead catches the failure this table is most likely to contain
# today -- a typo'd or stale target -- and it catches it in the editor rather
# than after a sync to a GPU box.
# ----------------------------------------------------------------------


def _module_path(source_root: str, module: str) -> str | None:
    import os

    rel = module.replace(".", os.sep)
    for candidate in (
        os.path.join(source_root, rel + ".py"),
        os.path.join(source_root, rel, "__init__.py"),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def _toplevel_names(tree: Any) -> dict[str, Any]:
    import ast

    names: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names[node.target.id] = node
        elif isinstance(node, ast.ImportFrom):
            # Re-exports count: register_quantization_config is defined in a
            # submodule and surfaced from the package __init__.
            for alias in node.names:
                names[alias.asname or alias.name] = node
    return names


def _declares_field(class_node: Any, name: str) -> bool:
    """Does this ClassDef body declare ``name`` as an attribute?"""
    import ast

    for child in class_node.body:
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            if child.target.id == name:
                return True
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return True
    return False


def check_static(seam: Seam, source_root: str) -> SeamProblem | None:
    """Verify a seam against a vLLM *source tree*, without importing it.

    ``source_root`` is the directory containing the ``vllm`` package, e.g.
    a vLLM checkout. Weaker than :func:`check` -- it cannot see
    signatures on inherited members or names produced by decorators -- so a pass
    here is necessary, not sufficient.
    """
    import ast

    path = _module_path(source_root, seam.module)
    if path is None:
        return SeamProblem(seam, f"module file not found under {source_root}")

    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)

    parts = seam.qualname.split(".")
    names = _toplevel_names(tree)
    node = names.get(parts[0])
    if node is None:
        return SeamProblem(seam, f"{parts[0]} not defined in {seam.module}")

    _Def = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef

    for part in parts[1:]:
        if not isinstance(node, ast.ClassDef):
            return SeamProblem(seam, f"cannot look up .{part} on a non-class")

        child_def = next(
            (c for c in node.body if isinstance(c, _Def) and c.name == part), None
        )
        if child_def is not None:
            node = child_def
            continue

        # A declared field -- a dataclass or pydantic attribute, which appears
        # as an annotated assignment rather than a def. It is terminal: there is
        # nothing to descend into, so the remaining path (if any) is bogus.
        if _declares_field(node, part):
            node = None
            continue

        # Inherited members are legitimately invisible to a single-file parse;
        # say so rather than reporting a false break.
        bases = [ast.unparse(b) for b in node.bases]
        return SeamProblem(
            seam,
            f"{part} not defined directly on {node.name} "
            f"(bases: {bases or 'none'}) -- inherited, or gone",
        )

    if node is None:
        # Terminated on a field. Nothing further to verify statically.
        return None

    if seam.params and isinstance(node, ast.ClassDef):
        # Mirror check(): for a class, the params we require are its
        # constructor's. Without this the static level would silently pass a
        # seam the import level checks, which is worse than not checking.
        init = next(
            (
                child
                for child in node.body
                if isinstance(child, ast.FunctionDef) and child.name == "__init__"
            ),
            None,
        )
        if init is None:
            return SeamProblem(
                seam,
                f"{node.name} defines no __init__ directly, so params "
                f"{list(seam.params)} cannot be checked statically",
            )
        node = init

    if seam.params and isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        args = node.args
        available = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
        if args.vararg:
            available.add(args.vararg.arg)
        if args.kwarg:
            # **kwargs swallows anything, so param checks cannot fail here.
            available |= set(seam.params)
        missing = [p for p in seam.params if p not in available]
        if missing:
            return SeamProblem(seam, f"parameters gone: {missing}")

    return None


def check_all_static(
    source_root: str, include_optional: bool = False
) -> list[SeamProblem]:
    """Every required seam that fails the source-tree check."""
    return [
        problem
        for seam in SEAMS
        if (include_optional or seam.required)
        and (problem := check_static(seam, source_root)) is not None
    ]
