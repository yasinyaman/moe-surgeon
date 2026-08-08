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

Stability tiers, from the survey in the plan:

``documented``
    Covered by vLLM's own docs and CI (plugin groups, ``register_model``,
    ``register_quantization_config``, ``register_model_loader``).
``used_in_tree``
    Real mechanism, exercised in-tree, thinly documented. Mechanism is stable;
    the classes it substitutes are not (``PluggableLayer.register_oot``).
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
    # Registration mechanisms. These are how we get installed at all.
    # ------------------------------------------------------------------
    Seam(
        target="vllm.model_executor.custom_op:CustomOp.register_oot",
        kind="method",
        tier="used_in_tree",
        why=(
            "Substitutes UnquantizedFusedMoEMethod so the expert cache can change "
            "where expert weights are allocated and what apply() dispatches to. "
            "Without it the unquantized MoE path needs an in-tree patch. "
            "CONTRACT: op_registry_oot is keyed by the *class* name, because "
            "CustomOp.__new__ looks up cls.__name__ -- not by the op name given to "
            "@CustomOp.register. Registering under the op name puts the class where "
            "nothing looks, and the substitution no-ops silently."
        ),
        params=("name",),
    ),
    Seam(
        target="vllm.model_executor.custom_op:PluggableLayer.register_oot",
        kind="method",
        tier="used_in_tree",
        why=(
            "Substitutes RoutedExperts (weight loading, cache install) and "
            "MoERunner (routing telemetry, output stabilization). The single "
            "most load-bearing seam in the package."
        ),
        params=("name",),
    ),
    Seam(
        target="vllm.model_executor.layers.quantization:register_quantization_config",
        kind="function",
        tier="documented",
        why=(
            "Registers a config under the name 'fp8', shadowing the builtin, so "
            "Fp8MoEMethod can be swapped. Fp8MoEMethod is not a CustomOp, so "
            "register_oot cannot reach it; custom configs override same-named "
            "builtins, which is the only route in."
        ),
        params=("quantization",),
    ),
    # ------------------------------------------------------------------
    # Classes we subclass.
    # ------------------------------------------------------------------
    Seam(
        target="vllm.model_executor.layers.fused_moe.routed_experts:RoutedExperts",
        kind="class",
        tier="internal",
        why=(
            "Base class for our MoE layer. Owns the expert parameters and the "
            "weight loader. ~1700 lines with ~20 constructor kwargs; the class "
            "that has already been renamed once (it was FusedMoE)."
        ),
    ),
    Seam(
        target="vllm.model_executor.layers.fused_moe.runner.moe_runner:MoERunner",
        kind="class",
        tier="internal",
        why=(
            "Base class for the telemetry runner. Note it carries no "
            "@PluggableLayer.register decorator of its own; interception works "
            "because PluggableLayer.__new__ keys on cls.__name__."
        ),
        params=("router",),
    ),
    Seam(
        target=(
            "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method"
            ":UnquantizedFusedMoEMethod"
        ),
        kind="class",
        tier="internal",
        why="Base class for the cache-aware unquantized MoE method.",
    ),
    Seam(
        target="vllm.model_executor.layers.quantization.fp8:Fp8Config",
        kind="class",
        tier="internal",
        why="Subclassed so get_quant_method can return our Fp8MoEMethod.",
    ),
    Seam(
        target="vllm.model_executor.layers.quantization.fp8:Fp8MoEMethod",
        kind="class",
        tier="internal",
        why="Base class for the cache-aware fp8 MoE method.",
    ),
    Seam(
        target=(
            "vllm.model_executor.layers.fused_moe.fused_moe_method_base"
            ":FusedMoEMethodBase"
        ),
        kind="class",
        tier="documented",
        why=(
            "The contract a MoE quant method must satisfy. We only need it for "
            "isinstance checks; note that supports_expert_lru_cache was added by "
            "the in-tree prototype and does NOT exist upstream, so the package "
            "keeps its own backend allowlist instead of reading that property."
        ),
    ),
    # ------------------------------------------------------------------
    # Methods we override. Each of these is a behaviour we replace, so a
    # rename here means our override silently stops being called -- the worst
    # failure mode in the package and the reason this table exists.
    # ------------------------------------------------------------------
    Seam(
        target=(
            "vllm.model_executor.layers.fused_moe.routed_experts"
            ":RoutedExperts.weight_loader"
        ),
        kind="method",
        tier="internal",
        why=(
            "Intercepted to stream expert shards straight to the disk store so "
            "the [num_experts, ...] tensor is never materialized. A rename means "
            "checkpoints load normally and the store is never written -- silent."
        ),
    ),
    Seam(
        target=(
            "vllm.model_executor.layers.fused_moe.routed_experts"
            ":RoutedExperts._needs_intermediate_size_param"
        ),
        kind="method",
        tier="internal",
        why=(
            "Compares in-tree quant method CLASS NAMES as strings, so our "
            "subclass is invisible to it. Overridden to re-add ourselves. If "
            "this method moves, our method gets the wrong parameter set."
        ),
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
            "call it instead of duplicating the transpose rule in our override; "
            "where it does not, our override carries its own copy. Absent at the "
            "0.26.1 merge base, present ~300 commits later -- kept optional so "
            "the pinned version still passes."
        ),
    ),
    Seam(
        target=(
            "vllm.model_executor.layers.fused_moe.runner.moe_runner"
            ":MoERunner._forward_impl"
        ),
        kind="method",
        tier="internal",
        why=(
            "Overridden to stabilize the MoE op's output address under piecewise "
            "CUDA graphs. This replaces an in-tree patch; the underlying bug is "
            "upstream's and should be proposed separately."
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
        why=(
            "The cache must be installed here, BEFORE get_fused_moe_quant_config "
            "captures the scale tensors. Getting this order wrong scales every "
            "expert by whichever one happens to occupy its slot -- wrong output "
            "from the first token, no exception. See notes/fp8-duzeltmesi.md."
        ),
    ),
    # ------------------------------------------------------------------
    # Config plumbing.
    # ------------------------------------------------------------------
    Seam(
        target="vllm.config.vllm:get_current_vllm_config",
        kind="function",
        tier="internal",
        why="Reads additional_config from inside layers that get no config passed.",
    ),
    Seam(
        target="vllm.config.vllm:VllmConfig",
        kind="class",
        tier="documented",
        why=(
            "Carries additional_config, our configuration channel. The in-code "
            "comment on that field names out-of-tree config as its purpose."
        ),
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
    ``/Users/korns/vllm/repos/vllm``. Weaker than :func:`check` -- it cannot see
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
