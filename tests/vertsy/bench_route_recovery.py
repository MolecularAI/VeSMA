"""Route recovery benchmark (pytest-benchmark).

Run a real depth-2 retro search from a PaRoutes target using the full committed
``used14k`` template library (``used14k-reactions.tsv.gz``), extract every solved route, and check whether the
reference PaRoutes route is among them under the three route comparators
(:mod:`vertsy.routes`). Unlike route *replay* (which feeds each route its own minimal
templates), recovery searches against the whole library, so rule ids on the generated and
reference sides come from one shared numbering (``route_recovery_rule_index``): the matcher
is built from the library templates in order, and the reference route's reactions are mapped
to the same indices.

The benchmark records, per expansion layer, the frontier size, the number of (mol, rule)
pairs considered, the positive matches (reaction applications), the precursors produced, and
the layer wall time, plus per-comparator recovery counts — all in ``benchmark.extra_info`` for
the paper tables. Only the rdkit path is exercised on the slow reference machine; cpu, cuda,
and metal backends are wired so the same bench can run elsewhere via the Makefile / env vars
``BENCH_ROUTE_RECOVERY_BACKENDS`` and ``BENCH_ROUTE_RECOVERY_TARGETS``.
"""

from __future__ import annotations

import gc
import gzip
import json
import os
import resource
import sys
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pytest
from rdkit import Chem

from tests.datadirs import external_data_root
from tests.vertsy.paper_provenance import source_artifact
from tests.vertsy.paroutes_reference import (
    RouteRecoveryRuleIndex,
    raw68k_rule_index,
    route_recovery_rule_index,
    uspto42k_rule_index,
)
from vertsy.expansions import Expansions
from vertsy.interner import DictInterner
from vertsy.interop import route_from_aizynth_dict
from vertsy.metrics import (
    library,
    loose,
    min_route_depth,
    precursor_route_count_bounded,
    precursors_recovered,
    prepare_graph,
    route_solved,
    solved_route_count_bounded,
    starting_material_recovered,
    starting_material_route_count_bounded,
    strict,
)
from vertsy.routes import Route, encode_route, leaves
from vertsy.search import MoleculeBatch, expansion, object_mol_array, step
from vertsy.stock import load_stock
from vertsy.store.numpy_store import NumpyExpansionStore
from vesma.interop import cmol_from_mol
from vesma.keyer import (
    Keyer,
    cmol_rdkit_smiles_keyer,
    rdkit_mol_canonical_smiles,
    rdkit_smiles_keyer,
    short_ring_wl_keyer,
)
from vesma.matcher import HitArray
from vesma.rdkit_matcher import (
    RdkitReactionMatcher,
    rdkit_reactions_from_smarts,
)
from vesma.reaction import ReactionPattern, reaction_patterns_from_smarts

pytestmark = [pytest.mark.perf, pytest.mark.paper_benchmark]

PAROUTES_DIR = external_data_root() / "external" / "paroutes" / "processed"
ROUTE_TEMPLATES_FILE = "used14k-reactions.tsv.gz"
RAW68K_TEMPLATES_FILE = "raw68k-templates.tsv.gz"
TARGETS_FILE = "n1-d2-f0.smi.gz"
STOCK_FILE = "n1-stock.smi.gz"
ROUTES_FILE = "n1-routes.json.gz"

MAX_DEPTH = 2
# Rule id for a reference reaction absent from the library: no generated hit ever carries
# a negative rule id, so such a reference route is simply never recovered (rather than a
# hard error). See reference_rule_resolver.
MISSING_RULE = -1

DEFAULT_BACKENDS = ("rdkit",)
DEFAULT_TARGET_COUNTS = (1,)
DEFAULT_RULE_SETS = ("used14k",)
SUPPORTED_BACKENDS = {
    "rdkit",
    "cpu",
    "cuda-host",
    "cuda-device",
    "cuda-host-prefilter",
    "cuda-device-prefilter",
    # Both fold identity in per delivered piece -- `matcher.keyed_match_folding` over
    # `match_batches` -- rather than keying a joined precursor set after the fact. That is the
    # default because the search collapses to distinct molecules anyway, so the rows in between
    # were pure footprint: at depth 1 over 100 targets the folded arms hold 508,506 precursor
    # rows where the joined ones held 4,167,360, for an identical next frontier and identical
    # per-target route counts.
    "metal",
    "metal-prefilter",
    # The joined delivery the two above used to use, kept as the paired ablation: `match`, then
    # key the whole batch. Worth keeping a row for, because the comparison only means anything
    # inside one run -- the same construction kernel on the same workload drifts by a quarter
    # between measurement sessions on the Metal host, so a recorded number is not a baseline.
    "metal-joined",
    "metal-prefilter-joined",
}
# used14k: n1+n5-used subset of the raw library. raw68k: the full raw library, with the
# reference reaction hashes aligned through the committed used14k reaction rows. Both support
# all comparators. uspto42k has unrelated ids, so only reaction-agnostic comparators apply.
SUPPORTED_RULE_SETS = {"used14k", "raw68k", "uspto42k"}
DEFAULT_SOLVED_ROUTE_COUNT_CAP = 100_000

Key = Any
Mol = Any

# A 1-arg match callable matching ``vertsy.search.Match``: frontier mols ->
# (precursor mols, precursor keys, hits).
MatchFn = Callable[[MoleculeBatch], tuple[MoleculeBatch, Sequence[Key], HitArray]]


@dataclass(frozen=True)
class LayerStats:
    """One expansion layer's instrumentation."""

    depth: int
    frontier_size: int
    pairs: int  # frontier_size * rule_count (every mol x every rule considered)
    applications: int  # distinct (mol_idx, rule, app) hits == positive matches
    precursors: int  # precursor molecules produced
    # Precursor atom-count summary (0 when no precursors) — supports the "precursors shrink
    # with depth, lowering later fanout" narrative. Best-effort across backend batch types.
    precursor_atoms_mean: float
    precursor_atoms_median: float
    precursor_atoms_max: int
    next_frontier_size: int
    match_split_seconds: float  # wall time of backend match + retro split
    step_seconds: float  # full wall time of the layer's `step`
    cuda_memory: dict[str, int | float]
    # Metal per-stage retro breakdown for this layer (empty for non-metal backends):
    # match / assignment-collect / retro-construct-kernel / wl-kernel / precursor-gather
    # seconds, plus candidate_pairs/prefilter_seconds for the prefiltered variant.
    metal_retro: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetRecoveryResult:
    target_index: int
    target_smiles: str
    layers: tuple[LayerStats, ...]
    # Rule-aware comparators are None when the rule ids do not align with the reference
    # (e.g. the uspto42k rule set); only the reference-agnostic ones below apply then.
    strict: int | None
    loose: int | None
    library: int | None
    any_solved: int
    starting_materials_recovered: int
    precursors_recovered: int
    solved_route_count_depth_bounded_capped: int
    precursor_route_count_depth_bounded_capped: int
    starting_material_route_count_depth_bounded_capped: int
    solved_route_count_cap: int
    min_route_depth: int | None
    search_seconds: float
    metric_seconds: float
    full_solve_seconds: float


@dataclass(frozen=True)
class RecoveryResult:
    targets: tuple[TargetRecoveryResult, ...]

    @property
    def num_targets(self) -> int:
        return len(self.targets)

    @property
    def strict(self) -> int | None:
        return _sum_optional(target.strict for target in self.targets)

    @property
    def loose(self) -> int | None:
        return _sum_optional(target.loose for target in self.targets)

    @property
    def library(self) -> int | None:
        return _sum_optional(target.library for target in self.targets)

    @property
    def any_solved(self) -> int:
        return sum(target.any_solved for target in self.targets)

    @property
    def starting_materials_recovered(self) -> int:
        return sum(target.starting_materials_recovered for target in self.targets)

    @property
    def precursors_recovered(self) -> int:
        return sum(target.precursors_recovered for target in self.targets)

    @property
    def search_seconds(self) -> float:
        return sum(target.search_seconds for target in self.targets)

    @property
    def metric_seconds(self) -> float:
        return sum(target.metric_seconds for target in self.targets)

    @property
    def full_solve_seconds(self) -> float:
        return sum(target.full_solve_seconds for target in self.targets)


def _sum_optional(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


#: Where identity is folded into the precursors a layer delivers. The distinction a record has to
#: carry is not "deduplicated or not" -- every arm's search ends up with distinct molecules -- but
#: *when*, because that is what decides the peak the layer holds.
#:
#: `joined` materialises every application's precursor into one batch and keys it afterwards.
#: `folded` keys each delivered piece and collapses it against the keys already seen, so only
#: distinct molecules are ever held; this is `matcher.keyed_match_folding` over `match_batches`,
#: composed at the call site and available to any backend with a batched delivery.
#: `folded-in-backend` is the same collapse done inside a matcher's own match, against its own
#: running map, where the generic seam cannot see it -- `cuda_matcher.match_keyed` and its
#: `key_to_physical`. It also places the host/device frontier, so it remains CUDA's native
#: search delivery; the generic fold provides a separately composable identity collapse.
DELIVERY_JOINED = "joined"
DELIVERY_FOLDED = "folded"
DELIVERY_FOLDED_IN_BACKEND = "folded-in-backend"


@dataclass(frozen=True)
class RecoveryBackend:
    name: str
    codec: Keyer[Any, Key]
    rule_count: int
    stock: frozenset[Key]
    compile_seconds: float
    from_smiles_mol: Callable[[str], Mol]
    from_smiles_key: Callable[[str], Key]
    stack_mols: Callable[[Sequence[Mol]], MoleculeBatch]
    match: MatchFn
    #: How the precursors reached the search, one of the DELIVERY_* constants above.
    #:
    #: Required, with no default, because a default is what let this go wrong: it read "joined",
    #: and the CUDA arms -- which fold identity in as parts arrive, inside `match_keyed` -- took
    #: it silently and would have published rows claiming a delivery they do not use. An arm
    #: that declares its own is an arm that cannot be mislabelled by someone else's default.
    precursor_delivery: str
    cuda_diagnostics: list[Any] | None = None
    metal_diagnostics: list[dict[str, float]] | None = None


# --------------------------------------------------------------------------- backends


def build_recovery_backend(
    backend: str, rule_index: RouteRecoveryRuleIndex
) -> RecoveryBackend:
    """Compile deduplicated SMARTS (rule_id == position) into a backend."""
    smarts = rule_index.smarts
    rule_count = len(smarts)

    if backend == "rdkit":
        started = perf_counter()
        matcher = RdkitReactionMatcher(rdkit_reactions_from_smarts(smarts))
        compile_seconds = perf_counter() - started

        def match(mols: MoleculeBatch) -> tuple[MoleculeBatch, Sequence[Key], HitArray]:
            precursors, hits = matcher.match(mols)
            return object_mol_array(precursors), keys_rdkit(precursors), hits

        return RecoveryBackend(
            name=backend,
            codec=rdkit_smiles_keyer,
            rule_count=rule_count,
            stock=load_recovery_stock(rdkit_smiles_keyer),
            compile_seconds=compile_seconds,
            from_smiles_mol=Chem.MolFromSmiles,
            from_smiles_key=rdkit_smiles_keyer.from_smiles,
            stack_mols=object_mol_array,
            match=match,
            # `RunReactants` per application into one object array, canonical SMILES after.
            precursor_delivery=DELIVERY_JOINED,
        )

    if backend == "cpu":
        from vesma.cpu_matcher import CpuReactionMatcher

        codec = cmol_rdkit_smiles_keyer
        patterns = route_recovery_patterns(smarts)
        started = perf_counter()
        matcher = CpuReactionMatcher(patterns)
        compile_seconds = perf_counter() - started

        def match(mols: MoleculeBatch) -> tuple[MoleculeBatch, Sequence[Key], HitArray]:
            precursors, hits = matcher.match(mols)
            return (
                object_mol_array(precursors),
                [codec.encode(m) for m in precursors],
                hits,
            )

        return RecoveryBackend(
            name=backend,
            codec=codec,
            rule_count=rule_count,
            stock=load_recovery_stock(codec),
            compile_seconds=compile_seconds,
            from_smiles_mol=lambda smiles: cmol_from_mol(Chem.MolFromSmiles(smiles)),
            from_smiles_key=codec.from_smiles,
            stack_mols=object_mol_array,
            match=match,
            # `RunReactants` per application, collected into one object array, keyed after.
            precursor_delivery=DELIVERY_JOINED,
        )

    if backend in {
        "cuda-host",
        "cuda-device",
        "cuda-host-prefilter",
        "cuda-device-prefilter",
    }:
        cp = pytest.importorskip("cupy")
        if not cp.cuda.is_available():
            pytest.skip("CuPy cannot access CUDA on this machine.")
        from vesma.cuda_matcher import (
            CudaFoldedPrefilteredReactionMatcher,
            CudaReactionMatcher,
            cuda_stack_mols,
        )

        frontier_mode = "host" if backend.startswith("cuda-host") else "device"
        codec = short_ring_wl_keyer
        patterns = route_recovery_patterns(smarts)
        started = perf_counter()
        matcher: Any = CudaReactionMatcher(patterns, frontier_mode=frontier_mode)
        if backend.endswith("-prefilter"):
            matcher = CudaFoldedPrefilteredReactionMatcher(matcher)
        cp.cuda.Stream.null.synchronize()
        compile_seconds = perf_counter() - started
        cuda_diagnostics: list[Any] = []

        def match(mols: MoleculeBatch) -> tuple[MoleculeBatch, Sequence[Key], HitArray]:
            precursors, precursor_keys, hits = matcher.match_keyed(mols)
            cp.cuda.Stream.null.synchronize()
            diagnostic = matcher.last_diagnostics
            cuda_diagnostics.append(
                cuda_memory_snapshot(
                    cp,
                    diagnostic,
                    getattr(matcher, "last_prefilter_diagnostics", None),
                )
            )
            return precursors, precursor_keys, hits

        return RecoveryBackend(
            name=backend,
            codec=codec,
            rule_count=rule_count,
            stock=load_recovery_stock(codec),
            compile_seconds=compile_seconds,
            from_smiles_mol=lambda smiles: cmol_from_mol(Chem.MolFromSmiles(smiles)),
            from_smiles_key=codec.from_smiles,
            stack_mols=cuda_stack_mols,
            match=match,
            # `match_keyed` runs the WL pass on the device and folds each part against a running
            # `key_to_physical` map, retaining one physical molecule per distinct key. That is a
            # fold, not a join -- the batch it returns is a logical view over deduplicated
            # storage -- so these rows must not inherit a "joined" default.
            precursor_delivery=DELIVERY_FOLDED_IN_BACKEND,
            cuda_diagnostics=cuda_diagnostics,
        )

    if backend in {
        "metal",
        "metal-prefilter",
        "metal-joined",
        "metal-prefilter-joined",
    }:
        require_metal()
        from vesma.matcher import all_vs_all_batches, keyed_match_folding
        from vesma.metal_reaction_matcher import (
            MetalFoldedPrefilteredReactionMatcher,
            MetalReactionMatcher,
            metal_concat_batches,
            metal_short_ring_wl_key,
            metal_stack_mols,
        )

        # metal-prefilter wraps the raw reaction matcher in the folded path-fingerprint
        # prefilter, so the matcher kernel only sees candidate (mol, rule) pairs instead of
        # the full frontier x all-rules cross product.
        codec = short_ring_wl_keyer
        patterns = route_recovery_patterns(smarts)
        started = perf_counter()
        matcher = MetalReactionMatcher(patterns)
        if backend.startswith("metal-prefilter"):
            matcher = MetalFoldedPrefilteredReactionMatcher(matcher)
        compile_seconds = perf_counter() - started
        metal_diagnostics: list[dict[str, float]] = []

        if not backend.endswith("-joined"):
            # Identity folded in as each construction dispatch lands, so what the layer holds is
            # one row per distinct molecule rather than one per application. The search collapses
            # to distinct molecules anyway -- `step` interns the keys and keeps one representative
            # per sid -- so the precursors in between were pure footprint, and on this corpus
            # there are several times more of them than there are molecules.
            folded = keyed_match_folding(
                all_vs_all_batches(matcher),
                metal_short_ring_wl_key,
                metal_concat_batches,
            )

            def match(
                mols: MoleculeBatch,
            ) -> tuple[MoleculeBatch, Sequence[Key], HitArray]:
                precursors, keys, hits = folded(mols)
                metal_diagnostics.append(metal_retro_snapshot(matcher))
                return precursors, keys, hits

        else:

            def match(
                mols: MoleculeBatch,
            ) -> tuple[MoleculeBatch, Sequence[Key], HitArray]:
                precursors, hits = matcher.match(mols)
                metal_diagnostics.append(metal_retro_snapshot(matcher))
                return precursors, metal_short_ring_wl_key(precursors), hits

        return RecoveryBackend(
            name=backend,
            precursor_delivery=(
                DELIVERY_JOINED if backend.endswith("-joined") else DELIVERY_FOLDED
            ),
            codec=codec,
            rule_count=rule_count,
            stock=load_recovery_stock(codec),
            compile_seconds=compile_seconds,
            from_smiles_mol=lambda smiles: cmol_from_mol(Chem.MolFromSmiles(smiles)),
            from_smiles_key=codec.from_smiles,
            stack_mols=metal_stack_mols,
            match=match,
            metal_diagnostics=metal_diagnostics,
        )

    raise ValueError(f"Unsupported BENCH_ROUTE_RECOVERY_BACKENDS entry: {backend}")


def route_recovery_patterns(smarts: Sequence[str]) -> Sequence[ReactionPattern]:
    return reaction_patterns_from_smarts(smarts, cache=True)


def keys_rdkit(precursors: Sequence[Mol]) -> list[Key]:
    return [rdkit_mol_canonical_smiles(mol) for mol in precursors]


def require_metal() -> None:
    if sys.platform != "darwin":
        pytest.skip("Metal backend is only available on macOS.")
    try:
        from vesma.metal_matcher import GpuUnavailableError, probe_metal
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Metal backend unavailable: {exc}")
    try:
        probe_metal()
    except (GpuUnavailableError, RuntimeError) as exc:  # pragma: no cover
        pytest.skip(f"Metal backend not available: {exc}")


# ------------------------------------------------------------------- data loading


def load_recovery_stock(codec: Keyer[Any, Key]) -> frozenset[Key]:
    path = PAROUTES_DIR / STOCK_FILE
    if not path.exists():
        pytest.skip(f"Missing benchmark stock file: {path}")
    return frozenset(load_stock(path, codec))


def load_target_smiles(num_targets: int, filename: str = TARGETS_FILE) -> list[str]:
    path = PAROUTES_DIR / filename
    if not path.exists():
        pytest.skip(f"Missing benchmark target file: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        smiles = [line.strip().split()[0] for line in handle if line.strip()]
    if num_targets > len(smiles):
        pytest.skip(
            f"Requested {num_targets} targets but only {len(smiles)} available."
        )
    return smiles[:num_targets]


def load_reference_nodes(target_smiles_list: Sequence[str]) -> list[dict[str, Any]]:
    """Map each target to its reference PaRoutes route node (root mol with one reaction)."""
    path = PAROUTES_DIR / ROUTES_FILE
    if not path.exists():
        pytest.skip(f"Missing benchmark route file: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = json.load(handle)

    exact: dict[str, dict[str, Any]] = {}
    for row in rows:
        exact.setdefault(row["smiles"], row)
    canonical: dict[str, dict[str, Any]] | None = None

    nodes: list[dict[str, Any]] = []
    for smiles in target_smiles_list:
        row = exact.get(smiles)
        if row is None:
            if canonical is None:
                canonical = {}
                for candidate in rows:
                    canonical.setdefault(
                        rdkit_smiles_keyer.from_smiles(candidate["smiles"]), candidate
                    )
            row = canonical.get(rdkit_smiles_keyer.from_smiles(smiles))
        if row is None:
            pytest.skip(f"No reference route for target {smiles!r}")
        children = list(row.get("children", []))
        nodes.append({"type": "mol", "smiles": row["smiles"], "children": children[:1]})
    return nodes


def reference_rule_resolver(
    rule_id_by_reaction_hash: Mapping[str, int],
) -> Callable[[dict[str, Any]], int]:
    def resolve(metadata: dict[str, Any]) -> int:
        reaction_hash = str(metadata["reaction_hash"])
        rule_id = rule_id_by_reaction_hash.get(reaction_hash)
        if rule_id is None:
            warnings.warn(
                f"Reference reaction {reaction_hash!r} is absent from the route-recovery "
                f"library; this route can never be recovered (sentinel rule "
                f"{MISSING_RULE}).",
                stacklevel=2,
            )
            return MISSING_RULE
        return rule_id

    return resolve


@dataclass(frozen=True)
class Reference:
    """Reference route reduced to what each comparator needs.

    ``rule_aligned_route`` is present only when the search rule ids match the reference's
    reactions (the used14k rule set); otherwise the rule-aware comparators are skipped and
    only ``leaves`` / ``precursors`` (the reaction-agnostic comparators) are used.
    """

    target_key: Key
    leaves: frozenset[Key]
    precursors: frozenset[Key]
    rule_aligned_route: Route[Key] | None


def route_molecule_keys(route: Route[Key]) -> frozenset[Key]:
    keys = {route.product}
    for child in route.precursors:
        keys |= route_molecule_keys(child)
    return frozenset(keys)


def build_references(
    nodes: Sequence[Mapping[str, Any]],
    rule_index: RouteRecoveryRuleIndex,
    codec: Keyer[Any, Key],
    *,
    rule_aligned: bool,
) -> list[Reference]:
    if rule_aligned:
        resolve = reference_rule_resolver(rule_index.rule_id_by_reaction_hash)
    else:
        # uspto42k rules do not align with route reaction hashes; the reference is used only
        # for its molecules (leaves + precursors), so give every reaction a sentinel rule.
        def resolve(metadata: dict[str, Any]) -> int:
            return MISSING_RULE

    references: list[Reference] = []
    for node in nodes:
        route = encode_route(route_from_aizynth_dict(node, resolve), codec)
        references.append(
            Reference(
                target_key=route.product,
                leaves=leaves(route),
                precursors=route_molecule_keys(route) - {route.product},
                rule_aligned_route=route if rule_aligned else None,
            )
        )
    return references


# ------------------------------------------------------------- instrumented search


def distinct_applications(hits: HitArray) -> int:
    """Count disconnections: the application key is the first three hit columns."""
    rows = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
    if rows.shape[0] == 0:
        return 0
    return int(np.unique(rows[:, :3], axis=0).shape[0])


def precursor_atom_counts(batch: MoleculeBatch) -> np.ndarray:
    """Per-precursor atom counts across every backend batch type.

    Handles a ``CudaMoleculeBatch`` (``atom_counts_device``), a
    ``CudaMoleculeDeviceLogicalBatch`` (``physical`` + ``logical_indices``), a
    ``CudaMoleculeHostBatch`` (``parts`` of host parts), a ``MetalMoleculeBatch``
    (``atom_counts``), and object arrays of rdkit mols (``GetNumAtoms``) or cmols
    (``atom_count``).

    Fail-fast: an unrecognized molecule type raises rather than being silently recorded as a
    zero atom count (which would corrupt the paper's precursor-size stats). Only a genuinely
    empty batch yields an empty array.
    """
    device = getattr(batch, "atom_counts_device", None)
    if device is not None:  # CudaMoleculeBatch
        host = device.get() if hasattr(device, "get") else device
        return np.asarray(host, dtype=np.int64)
    physical = getattr(batch, "physical", None)
    logical_indices = getattr(batch, "logical_indices", None)
    if physical is not None and logical_indices is not None:  # device logical view
        host = physical.atom_counts_device
        host = host.get() if hasattr(host, "get") else host
        return np.asarray(host, dtype=np.int64)[np.asarray(logical_indices)]
    parts = getattr(batch, "parts", None)
    if parts is not None:  # CudaMoleculeHostBatch (parts hold unique physical mols)
        if not parts:
            return np.empty(0, dtype=np.int64)
        physical = np.concatenate(
            [np.asarray(part.atom_counts, dtype=np.int64) for part in parts]
        )
        # logical_indices maps the produced (with-multiplicity) precursor sequence onto the
        # deduplicated physical parts; without it we'd average the unique mols, not the output.
        logical_indices = getattr(batch, "logical_indices", None)
        if logical_indices is None:
            return physical
        return physical[np.asarray(logical_indices)]
    counts = getattr(batch, "atom_counts", None)
    if counts is not None:  # MetalMoleculeBatch
        return np.asarray(counts, dtype=np.int64)
    values: list[int] = []
    for mol in batch:  # object array of rdkit mols / cmols
        if hasattr(mol, "atom_count"):
            values.append(int(mol.atom_count))
        elif hasattr(mol, "GetNumAtoms"):
            values.append(int(mol.GetNumAtoms()))
        else:
            raise TypeError(
                "precursor_atom_counts: unsupported molecule type "
                f"{type(mol).__name__!r} (no atom_count / GetNumAtoms)"
            )
    return np.asarray(values, dtype=np.int64)


def atom_count_stats(counts: np.ndarray) -> tuple[float, float, int]:
    """(mean, median, max) precursor atom count; zeros when there are no precursors."""
    if counts.size == 0:
        return (0.0, 0.0, 0)
    return (float(counts.mean()), float(np.median(counts)), int(counts.max()))


def run_recovery_target(
    backend: RecoveryBackend,
    target_index: int,
    target_smiles: str,
    reference: Reference,
) -> TargetRecoveryResult:
    """Depth-2 search for one target, then compute graph-native route metrics.

    Mirrors :func:`vertsy.search.search`'s stock pre-marking, then folds :func:`step`
    layer by layer so each layer can be timed and its match output counted.
    """
    full_started = perf_counter()
    key = backend.from_smiles_key(target_smiles)
    mols = backend.stack_mols([backend.from_smiles_mol(target_smiles)])

    expansions: Expansions[Key] = Expansions(NumpyExpansionStore(), DictInterner())
    sids = expansions.intern([key])
    in_stock = np.fromiter((key in backend.stock,), dtype=np.bool_, count=1)
    expansions.mark_expanded(sids[in_stock])
    frontier_sids = sids[~in_stock]
    frontier_mols = mols[~in_stock]

    layers: list[LayerStats] = []
    search_started = perf_counter()
    for depth in range(MAX_DEPTH):
        if not len(frontier_sids):
            break
        frontier_size = int(len(frontier_sids))
        captured: dict[str, Any] = {}

        def counting_match(
            batch: MoleculeBatch, _captured: dict[str, Any] = captured
        ) -> tuple[MoleculeBatch, Sequence[Key], HitArray]:
            started = perf_counter()
            precursor_mols, precursor_keys, hits = backend.match(batch)
            _captured["match_split_seconds"] = perf_counter() - started
            _captured["hits"] = hits
            _captured["precursors"] = len(precursor_keys)
            mean_atoms, median_atoms, max_atoms = atom_count_stats(
                precursor_atom_counts(precursor_mols)
            )
            _captured["precursor_atoms_mean"] = mean_atoms
            _captured["precursor_atoms_median"] = median_atoms
            _captured["precursor_atoms_max"] = max_atoms
            _captured["cuda_memory"] = cuda_memory_extra(last_cuda_diagnostic(backend))
            _captured["metal_retro"] = last_metal_retro(backend)
            return precursor_mols, precursor_keys, hits

        started = perf_counter()
        layer = step(
            frontier_sids,
            frontier_mols,
            expansions,
            backend.stock,
            match=expansion(counting_match),
        )
        frontier_sids, frontier_mols = layer.sids, layer.mols
        step_seconds = perf_counter() - started

        layers.append(
            LayerStats(
                depth=depth,
                frontier_size=frontier_size,
                pairs=frontier_size * backend.rule_count,
                applications=distinct_applications(captured["hits"]),
                precursors=int(captured["precursors"]),
                precursor_atoms_mean=float(captured["precursor_atoms_mean"]),
                precursor_atoms_median=float(captured["precursor_atoms_median"]),
                precursor_atoms_max=int(captured["precursor_atoms_max"]),
                next_frontier_size=int(len(frontier_sids)),
                match_split_seconds=float(captured["match_split_seconds"]),
                step_seconds=step_seconds,
                cuda_memory=captured["cuda_memory"],
                metal_retro=captured.get("metal_retro", {}),
            )
        )
    search_seconds = perf_counter() - search_started

    count_cap = solved_route_count_cap()
    metric_started = perf_counter()
    state = prepare_graph(expansions, backend.stock)
    # Reaction-agnostic comparators work for every rule set.
    any_solved_value = route_solved(key)(state)
    sm_recovered_value = starting_material_recovered(key, reference.leaves, MAX_DEPTH)(
        state
    )
    prec_recovered_value = precursors_recovered(key, reference.precursors, MAX_DEPTH)(
        state
    )
    # Rule-aware comparators only when the reference reactions align with the search rules.
    if reference.rule_aligned_route is not None:
        aligned = reference.rule_aligned_route
        strict_value: int | None = strict(aligned)(state)
        loose_value: int | None = loose(aligned)(state)
        library_value: int | None = library(aligned)(state)
    else:
        strict_value = loose_value = library_value = None
    bounded_solved_count = solved_route_count_bounded(key, count_cap, MAX_DEPTH)(state)
    bounded_precursor_count = precursor_route_count_bounded(key, count_cap, MAX_DEPTH)(
        state
    )
    bounded_starting_material_count = starting_material_route_count_bounded(
        key, count_cap, MAX_DEPTH
    )(state)
    depth = min_route_depth(key)(state)
    metric_seconds = perf_counter() - metric_started

    return TargetRecoveryResult(
        target_index=target_index,
        target_smiles=target_smiles,
        layers=tuple(layers),
        strict=strict_value,
        loose=loose_value,
        library=library_value,
        any_solved=any_solved_value,
        starting_materials_recovered=sm_recovered_value,
        precursors_recovered=prec_recovered_value,
        solved_route_count_depth_bounded_capped=bounded_solved_count,
        precursor_route_count_depth_bounded_capped=bounded_precursor_count,
        starting_material_route_count_depth_bounded_capped=(
            bounded_starting_material_count
        ),
        solved_route_count_cap=count_cap,
        min_route_depth=depth,
        search_seconds=search_seconds,
        metric_seconds=metric_seconds,
        full_solve_seconds=perf_counter() - full_started,
    )


def run_recovery(
    backend: RecoveryBackend,
    target_smiles_list: Sequence[str],
    references: Sequence[Reference],
) -> RecoveryResult:
    results: list[TargetRecoveryResult] = []
    for index, (smiles, reference) in enumerate(
        zip(target_smiles_list, references, strict=True)
    ):
        try:
            results.append(run_recovery_target(backend, index, smiles, reference))
        finally:
            if backend.name.startswith("cuda"):
                cleanup_cuda_memory()
    return RecoveryResult(tuple(results))


def solved_route_count_cap() -> int:
    value = int(
        os.environ.get(
            "BENCH_ROUTE_RECOVERY_ROUTE_COUNT_CAP",
            str(DEFAULT_SOLVED_ROUTE_COUNT_CAP),
        )
    )
    if value <= 0:
        raise ValueError("BENCH_ROUTE_RECOVERY_ROUTE_COUNT_CAP must be positive.")
    return value


def recovery_extra_info(
    result: RecoveryResult,
    *,
    backend: str,
    rule_set: str,
    template_source: str,
    rules: int,
    stock_size: int,
    compile_seconds: float,
    precursor_delivery: str,
) -> dict[str, Any]:
    return {
        "experiment": "route-recovery",
        "backend": backend,
        "rule_set": rule_set,
        "reaction_comparators_feasible": rule_set in {"used14k", "raw68k"},
        "num_targets": result.num_targets,
        "route_set": "paroutes-n1-d2-f0",
        "template_source": template_source,
        "stock_policy": "paroutes-n1-stock",
        "rules": rules,
        "stock_size": stock_size,
        "max_depth": MAX_DEPTH,
        "compile_seconds": compile_seconds,
        "targets": [target_extra_info(target) for target in result.targets],
        "comparator_results_per_target": [
            [target.strict, target.loose, target.library] for target in result.targets
        ],
        "strict_per_target": [target.strict for target in result.targets],
        "loose_per_target": [target.loose for target in result.targets],
        "library_per_target": [target.library for target in result.targets],
        "any_solved_per_target": [target.any_solved for target in result.targets],
        "starting_materials_recovered_per_target": [
            target.starting_materials_recovered for target in result.targets
        ],
        "precursors_recovered_per_target": [
            target.precursors_recovered for target in result.targets
        ],
        "solved_route_count_depth_bounded_capped_per_target": [
            target.solved_route_count_depth_bounded_capped for target in result.targets
        ],
        "precursor_route_count_depth_bounded_capped_per_target": [
            target.precursor_route_count_depth_bounded_capped
            for target in result.targets
        ],
        "starting_material_route_count_depth_bounded_capped_per_target": [
            target.starting_material_route_count_depth_bounded_capped
            for target in result.targets
        ],
        "solved_route_count_cap": solved_route_count_cap(),
        "min_route_depth_per_target": [
            target.min_route_depth for target in result.targets
        ],
        "total_search_seconds": result.search_seconds,
        "total_metric_seconds": result.metric_seconds,
        "total_full_solve_seconds": result.full_solve_seconds,
        "strict_reference_recoveries": result.strict,
        "loose_reference_recoveries": result.loose,
        "library_reference_recoveries": result.library,
        "any_solved_recoveries": result.any_solved,
        "starting_materials_recoveries": result.starting_materials_recovered,
        "precursors_recoveries": result.precursors_recovered,
        "cuda_memory": aggregate_cuda_memory(
            layer.cuda_memory for target in result.targets for layer in target.layers
        ),
        # Peak resident for the whole arm. It used to be recorded only inside the CUDA memory
        # snapshot, so every non-CUDA arm reported no memory at all -- which is awkward for a
        # delivery whose main effect is memory: the folded arms halve peak resident at 100
        # targets, and the record could not show it. One number per process, which is why each
        # arm gets its own process.
        "peak_process_rss_bytes": process_rss_bytes(),
        "precursor_delivery": precursor_delivery,
        "source_artifacts": source_artifacts(template_source),
    }


def target_extra_info(target: TargetRecoveryResult) -> dict[str, Any]:
    return {
        "target_index": target.target_index,
        "target_smiles": target.target_smiles,
        "comparators": [target.strict, target.loose, target.library],
        "strict": target.strict,
        "loose": target.loose,
        "library": target.library,
        "any_solved": target.any_solved,
        "starting_materials_recovered": target.starting_materials_recovered,
        "precursors_recovered": target.precursors_recovered,
        "solved_route_count_depth_bounded_capped": (
            target.solved_route_count_depth_bounded_capped
        ),
        "precursor_route_count_depth_bounded_capped": (
            target.precursor_route_count_depth_bounded_capped
        ),
        "starting_material_route_count_depth_bounded_capped": (
            target.starting_material_route_count_depth_bounded_capped
        ),
        "solved_route_count_cap": target.solved_route_count_cap,
        "min_route_depth": target.min_route_depth,
        "search_seconds": target.search_seconds,
        "metric_seconds": target.metric_seconds,
        "full_solve_seconds": target.full_solve_seconds,
        "cuda_memory": aggregate_cuda_memory(
            layer.cuda_memory for layer in target.layers
        ),
        "layers": [layer_extra_info(layer) for layer in target.layers],
    }


def layer_extra_info(layer: LayerStats) -> dict[str, Any]:
    return {
        "depth": layer.depth,
        "frontier_size": layer.frontier_size,
        "pairs": layer.pairs,
        "applications": layer.applications,
        "precursors": layer.precursors,
        "precursor_atoms_mean": layer.precursor_atoms_mean,
        "precursor_atoms_median": layer.precursor_atoms_median,
        "precursor_atoms_max": layer.precursor_atoms_max,
        "next_frontier_size": layer.next_frontier_size,
        "match_split_seconds": layer.match_split_seconds,
        "step_seconds": layer.step_seconds,
        "cuda_memory": layer.cuda_memory,
        "metal_retro": layer.metal_retro,
    }


def reset_cuda_diagnostics(backend: RecoveryBackend) -> None:
    if backend.cuda_diagnostics is not None:
        backend.cuda_diagnostics.clear()


def cleanup_cuda_memory() -> None:
    gc.collect()
    cupy = sys.modules.get("cupy")
    if cupy is None:
        return
    cupy.cuda.Stream.null.synchronize()
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()


def last_cuda_diagnostic(backend: RecoveryBackend) -> tuple[Any, ...]:
    if not backend.cuda_diagnostics:
        return ()
    return (backend.cuda_diagnostics[-1],)


_METAL_STAGE_FIELDS = (
    "match_seconds",
    "assignment_collect_seconds",
    "retro_seconds",
    "retro_construct_kernel_seconds",
    "wl_kernel_seconds",
    "precursor_gather_seconds",
    "precursor_concat_seconds",
    "consumer_seconds",
    "total_seconds",
)


def metal_retro_snapshot(matcher: Any) -> dict[str, float]:
    """Per-stage retro breakdown of the most recent `match` on a Metal reaction matcher
    (raw or prefiltered wrapper). Empty if no diagnostics were recorded."""
    diagnostic = getattr(matcher, "last_diagnostics", None)
    snapshot: dict[str, float] = {}
    if diagnostic is not None:
        snapshot = {
            field_name: float(getattr(diagnostic, field_name))
            for field_name in _METAL_STAGE_FIELDS
        }
    # The prefiltered wrapper also exposes how many candidate pairs survived the prune and
    # what the prune itself cost -- the whole point of the A/B against the raw matcher.
    if hasattr(matcher, "last_candidate_pair_count"):
        snapshot["candidate_pairs"] = int(matcher.last_candidate_pair_count)
        snapshot["prefilter_seconds"] = float(matcher.last_prefilter_seconds)
    return snapshot


def last_metal_retro(backend: RecoveryBackend) -> dict[str, float]:
    if not backend.metal_diagnostics:
        return {}
    return backend.metal_diagnostics[-1]


def reset_metal_diagnostics(backend: RecoveryBackend) -> None:
    if backend.metal_diagnostics is not None:
        backend.metal_diagnostics.clear()


def cuda_memory_snapshot(
    cupy: Any,
    diagnostic: Any | None,
    prefilter: Any | None,
) -> dict[str, int | float]:
    free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
    pool = cupy.get_default_memory_pool()
    snapshot: dict[str, int | float] = {
        "match_calls": 1,
        "peak_driver_used_bytes": int(total_bytes - free_bytes),
        "peak_cupy_pool_used_bytes": int(pool.used_bytes()),
        "peak_cupy_pool_total_bytes": int(pool.total_bytes()),
        "peak_process_rss_bytes": process_rss_bytes(),
        "resident_rule_bytes": (
            0 if diagnostic is None else int(diagnostic.resident_rule_bytes)
        ),
        "peak_estimated_workspace_bytes": (
            0 if diagnostic is None else int(diagnostic.peak_estimated_workspace_bytes)
        ),
        "usable_device_bytes_min": (
            int(free_bytes)
            if diagnostic is None
            else int(diagnostic.usable_device_bytes)
        ),
    }
    if prefilter is not None:
        snapshot.update(
            {
                "dense_pair_count": int(prefilter.dense_pair_count),
                "candidate_pair_count": int(prefilter.candidate_pair_count),
                "pruned_pair_count": int(prefilter.pruned_pair_count),
                "prefilter_seconds": float(prefilter.prefilter_seconds),
            }
        )
    return snapshot


def process_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def cuda_memory_extra(
    diagnostics: Sequence[dict[str, int | float]],
) -> dict[str, int | float]:
    if not diagnostics:
        return {}
    result: dict[str, int | float] = {
        "match_calls": sum(d["match_calls"] for d in diagnostics),
        "peak_driver_used_bytes": max(d["peak_driver_used_bytes"] for d in diagnostics),
        "peak_cupy_pool_used_bytes": max(
            d["peak_cupy_pool_used_bytes"] for d in diagnostics
        ),
        "peak_cupy_pool_total_bytes": max(
            d["peak_cupy_pool_total_bytes"] for d in diagnostics
        ),
        "peak_process_rss_bytes": max(d["peak_process_rss_bytes"] for d in diagnostics),
        "resident_rule_bytes": max(d["resident_rule_bytes"] for d in diagnostics),
        "peak_estimated_workspace_bytes": max(
            d["peak_estimated_workspace_bytes"] for d in diagnostics
        ),
        "usable_device_bytes_min": min(
            d["usable_device_bytes_min"] for d in diagnostics
        ),
    }
    prefiltered = [d for d in diagnostics if "dense_pair_count" in d]
    if prefiltered:
        dense = int(sum(d["dense_pair_count"] for d in prefiltered))
        candidates = int(sum(d["candidate_pair_count"] for d in prefiltered))
        result.update(
            {
                "dense_pair_count": dense,
                "candidate_pair_count": candidates,
                "pruned_pair_count": dense - candidates,
                "prune_ratio": 0.0 if dense == 0 else 1.0 - candidates / dense,
                "prefilter_seconds": float(
                    sum(d["prefilter_seconds"] for d in prefiltered)
                ),
            }
        )
    return result


def aggregate_cuda_memory(
    items: Iterable[dict[str, int | float]],
) -> dict[str, int | float]:
    entries = tuple(item for item in items if item)
    if not entries:
        return {}
    result: dict[str, int | float] = {
        "match_calls": sum(entry["match_calls"] for entry in entries),
        "peak_driver_used_bytes": max(
            entry["peak_driver_used_bytes"] for entry in entries
        ),
        "peak_cupy_pool_used_bytes": max(
            entry["peak_cupy_pool_used_bytes"] for entry in entries
        ),
        "peak_cupy_pool_total_bytes": max(
            entry["peak_cupy_pool_total_bytes"] for entry in entries
        ),
        "peak_process_rss_bytes": max(
            entry["peak_process_rss_bytes"] for entry in entries
        ),
        "resident_rule_bytes": max(entry["resident_rule_bytes"] for entry in entries),
        "peak_estimated_workspace_bytes": max(
            entry["peak_estimated_workspace_bytes"] for entry in entries
        ),
        "usable_device_bytes_min": min(
            entry["usable_device_bytes_min"] for entry in entries
        ),
    }
    prefiltered = [entry for entry in entries if "dense_pair_count" in entry]
    if prefiltered:
        dense = int(sum(entry["dense_pair_count"] for entry in prefiltered))
        candidates = int(sum(entry["candidate_pair_count"] for entry in prefiltered))
        result.update(
            {
                "dense_pair_count": dense,
                "candidate_pair_count": candidates,
                "pruned_pair_count": dense - candidates,
                "prune_ratio": 0.0 if dense == 0 else 1.0 - candidates / dense,
                "prefilter_seconds": float(
                    sum(entry["prefilter_seconds"] for entry in prefiltered)
                ),
            }
        )
    return result


def template_artifact_path(template_source: str) -> Path:
    if template_source == "used14k-reactions":
        return PAROUTES_DIR / ROUTE_TEMPLATES_FILE
    if template_source == "raw68k-templates":
        return PAROUTES_DIR / RAW68K_TEMPLATES_FILE
    if template_source == "uspto42k":
        return PAROUTES_DIR / "uspto_unique_templates.csv.gz"
    raise ValueError(f"Unsupported template source: {template_source}")


def source_artifacts(
    template_source: str, targets_file: str = TARGETS_FILE
) -> dict[str, Any]:
    artifacts = {
        "targets": source_artifact(PAROUTES_DIR / targets_file),
        "stock": source_artifact(PAROUTES_DIR / STOCK_FILE),
        "templates": source_artifact(template_artifact_path(template_source)),
        "routes": source_artifact(PAROUTES_DIR / ROUTES_FILE),
    }
    if template_source == "raw68k-templates":
        artifacts["reference_reactions"] = source_artifact(
            PAROUTES_DIR / ROUTE_TEMPLATES_FILE
        )
    return artifacts


# ----------------------------------------------------------------------- the bench


def test_bench_route_recovery_n1_d2(
    benchmark: Any, route_recovery_backend: str, rule_set: str, num_targets: int
) -> None:
    cleanup_cuda_memory()
    backend: RecoveryBackend | None = None
    target: Callable[[], RecoveryResult] | None = None
    try:
        rule_index, template_source, rule_aligned = load_rule_set(rule_set)
        if not rule_index.smarts:
            pytest.skip(
                f"Route recovery library {rule_set!r} has no compatible templates."
            )

        backend = build_recovery_backend(route_recovery_backend, rule_index)
        compile_seconds = backend.compile_seconds
        stock_size = len(backend.stock)
        target_smiles = load_target_smiles(num_targets)
        nodes = load_reference_nodes(target_smiles)
        search_smiles = [node["smiles"] for node in nodes]
        references = build_references(
            nodes, rule_index, backend.codec, rule_aligned=rule_aligned
        )

        def target() -> RecoveryResult:
            if backend is None:
                raise RuntimeError("route recovery backend was released")
            reset_cuda_diagnostics(backend)
            reset_metal_diagnostics(backend)
            try:
                return run_recovery(backend, search_smiles, references)
            finally:
                cleanup_cuda_memory()

        result = benchmark.pedantic(
            target,
            rounds=rounds(),
            warmup_rounds=warmup_rounds(),
            iterations=1,
        )
        benchmark.extra_info.update(
            recovery_extra_info(
                result,
                backend=route_recovery_backend,
                rule_set=rule_set,
                template_source=template_source,
                rules=len(rule_index.smarts),
                stock_size=stock_size,
                compile_seconds=compile_seconds,
                precursor_delivery=backend.precursor_delivery,
            )
        )
    finally:
        target = None
        backend = None
        cleanup_cuda_memory()


def load_rule_set(rule_set: str) -> tuple[RouteRecoveryRuleIndex, str, bool]:
    """Return (rule index, template-source label, whether rule ids align with references)."""
    if rule_set == "used14k":
        templates_path = PAROUTES_DIR / ROUTE_TEMPLATES_FILE
        if not templates_path.exists():
            pytest.skip(f"Missing benchmark template file: {templates_path}")
        return route_recovery_rule_index(templates_path), "used14k-reactions", True
    if rule_set == "raw68k":
        templates_path = PAROUTES_DIR / RAW68K_TEMPLATES_FILE
        if not templates_path.exists():
            pytest.skip(f"Missing benchmark template file: {templates_path}")
        return (
            raw68k_rule_index(
                templates_path,
                PAROUTES_DIR / ROUTE_TEMPLATES_FILE,
            ),
            "raw68k-templates",
            True,
        )
    if rule_set == "uspto42k":
        return uspto42k_rule_index(), "uspto42k", False
    raise ValueError(f"Unsupported BENCH_ROUTE_RECOVERY_RULE_SETS entry: {rule_set}")


# ------------------------------------------------------------------ parametrization


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "route_recovery_backend" in metafunc.fixturenames:
        metafunc.parametrize("route_recovery_backend", selected_backends())
    if "rule_set" in metafunc.fixturenames:
        metafunc.parametrize("rule_set", selected_rule_sets())
    if "num_targets" in metafunc.fixturenames:
        metafunc.parametrize("num_targets", selected_target_counts())


def selected_rule_sets() -> tuple[str, ...]:
    raw = os.environ.get("BENCH_ROUTE_RECOVERY_RULE_SETS")
    values = (
        DEFAULT_RULE_SETS
        if not raw
        else tuple(value.strip() for value in raw.replace(",", " ").split())
    )
    unsupported = [value for value in values if value not in SUPPORTED_RULE_SETS]
    if unsupported:
        raise ValueError(
            f"BENCH_ROUTE_RECOVERY_RULE_SETS has unsupported entries: {unsupported}"
        )
    return tuple(value for value in values if value)


def selected_backends() -> tuple[str, ...]:
    raw = os.environ.get("BENCH_ROUTE_RECOVERY_BACKENDS")
    values = (
        DEFAULT_BACKENDS
        if not raw
        else tuple(value.strip() for value in raw.replace(",", " ").split())
    )
    unsupported = [value for value in values if value not in SUPPORTED_BACKENDS]
    if unsupported:
        raise ValueError(
            f"BENCH_ROUTE_RECOVERY_BACKENDS has unsupported entries: {unsupported}"
        )
    return tuple(value for value in values if value)


def selected_target_counts() -> tuple[int, ...]:
    raw = os.environ.get("BENCH_ROUTE_RECOVERY_TARGETS")
    if not raw:
        return DEFAULT_TARGET_COUNTS
    return tuple(int(value) for value in raw.replace(",", " ").split() if value.strip())


def rounds() -> int:
    value = int(os.environ.get("BENCH_ROUNDS", "1"))
    if value <= 0:
        raise ValueError("BENCH_ROUNDS must be positive.")
    return value


def warmup_rounds() -> int:
    value = int(os.environ.get("BENCH_WARMUP_ROUNDS", "0"))
    if value < 0:
        raise ValueError("BENCH_WARMUP_ROUNDS must be non-negative.")
    return value
