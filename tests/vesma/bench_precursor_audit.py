"""Application-level precursor audit: vesma CPU vs RDKit vs RDChiral.

This is a **correctness artifact, not a timing one**. It runs under pytest-benchmark only
to reuse its provenance capture and JSON writer; ``rounds=1, warmup_rounds=0,
iterations=1`` and the wall time it records is context, never a paper table. The timed
comparisons live in ``bench_match_scaling.py`` and emit no audit keys.

What it is for
--------------
Vesma's chemistry model refuses applications that RDKit and RDChiral answer, and answers
some they refuse. The point of the audit is to find out whether each divergence is a
consequence of the model as designed, or a defect in it. So the artifact is built to be
read by a chemist afterwards: every divergence and every refusal is written out in full,
with the molecule and the template that produced it. Nothing is capped, sampled or
summarized away, because the interesting cases are exactly the rare ones.

Neither comparison backend is ground truth. Divergence counts order the review; they do
not decide it.

Why the unit is the application, not the pair
--------------------------------------------
One (molecule, rule) pair is many atom assignments, and they do not share a fate.
Ethylbenzene against ``[*:1]~[*:2]>>[*:1].[*:2]`` yields four applications vesma accepts
and twelve it refuses. A pair-level status has no truthful value to take here, and a pair
whose surviving applications happen to agree with RDKit would report as clean agreement
while concealing twelve refusals. Accepted applications come from the returned hits and
refused ones from `CpuReactionMatcher.last_rejected_applications`; together they account
for every application attempted.

Comparison still happens at pair level, because ``app_idx`` is not comparable across
backends -- each enumerates assignments in its own order, and RDChiral's index counts
*distinct outcomes* rather than assignments. What is compared is the pair's set of
outcomes; what is *accounted* is the application.

Fixed denominators
------------------
Every pair stays in the denominator under an explicit status. A backend that raised, or
returned a precursor with no key in the shared space, is recorded as such rather than
dropped -- otherwise the agreement rate improves whenever a backend fails, which is
exactly backwards. Agreement among jointly-successful pairs is reported as well, never
alone.

Two identities, both audit-specific
-----------------------------------
- **Cross-tool**: isotope-preserving canonical RDKit SMILES. `vesma.keyer` deliberately
  passes ``isomericSmiles=False`` so that isotopologues share a search key; an audit that
  inherited that would score ``[13CH4]`` and ``C`` as agreement.
- **Vesma-native**: the exact ``(Z, M, C, H, BondKind)`` graph under a canonical atom
  order. A precursor RDKit cannot sanitize has no cross-tool key at all and would vanish
  from the audit; this gives it an identity anyway. It is also the identity a Metal or
  CUDA port will have to reproduce atom for atom.

InChI is deliberately not used: it disconnects organometallic bonds, so ``[CH3][Mg]Br``
and ``[CH3].[Mg]Br`` collapse to one key -- concealing precisely the bonding the metal
configurations were added to express.

Crash resilience
----------------
The full sweep is 768,015 pairs over about half a day, and nothing reaches disk until the
whole audit returns, so the fold is checkpointed after every chunk and resumes from the
last complete one. A checkpoint is refused unless its token matches the sweep exactly --
size, chunking, and both chemistry versions -- because splicing two half-runs into one
artifact would be reported as a single measurement and would not look like a failure.

Knobs (from the environment, set by the Makefile):
- ``BENCH_AUDIT_SIZES``  molecule counts to sweep (default ``100``)
- ``BENCH_AUDIT_CHUNK``  molecules per streaming chunk (default 50)
- ``BENCH_AUDIT_CHECKPOINT_DIR``  where to checkpoint; empty disables it
"""

from __future__ import annotations

import os
import pickle
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.datasets import chemistry_fingerprint
from tests.vesma.bench_matcher import (
    MatchCorpus,
    load_match_corpus_by_indices,
    load_positive_pairs,
    load_rdchiral_reactions,
    molecule_indices,
    template_smarts,
)
from vesma.canonicalization import canonicalish_atom_order, reorder_cmol
from vesma.cmol import CMol
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.interop import mol_from_cmol
from vesma.keyer import rdkit_version, vesma_version
from vesma.matcher import HitArray
from vesma.rdkit_matcher import RdkitReactionMatcher, rdkit_reactions_from_smarts
from vesma.reaction import (
    RHS_ANNOTATIONS,
    AnnotationCounts,
    merged_annotations,
    no_annotation_mismatches,
)
from vesma.rejection import (
    RejectionCounts,
    RejectionReason,
    by_name,
    counted,
    merged,
    no_rejections,
)

pytestmark = [pytest.mark.perf, pytest.mark.paper_benchmark]

DEFAULT_AUDIT_SIZES = (100,)
DEFAULT_CHUNK_MOLECULES = 50
DEFAULT_CHECKPOINT_DIR = "output/benchmarks/precursor-audit/checkpoints"

VESMA = "vesma-cpu"
RDKIT = "rdkit"
RDCHIRAL = "rdchiral"
BACKENDS = (VESMA, RDKIT, RDCHIRAL)

Pair = tuple[int, int]
Application = tuple[int, int, int]
# One pair's outcomes; each outcome is the set of precursor keys one application produced.
OutcomeSets = frozenset[frozenset[str]]

EMPTY: OutcomeSets = frozenset()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def test_bench_precursor_audit(benchmark: Any, size: int) -> None:
    pytest.importorskip("rdchiral")
    report = benchmark.pedantic(
        lambda: audit_precursors(size),
        rounds=1,
        warmup_rounds=0,
        iterations=1,
    )
    benchmark.extra_info.update(
        {
            "experiment": "paper-validation",
            "backend": "cpu",
            "validation": "precursor-audit",
            "size": size,
            "frontier": "n1-d2-f0",
            "template_source": "uspto42k-compat",
            "positive_pair_source": "paroutes-n1-d2-match-uspto42k",
            "timing_is_context_only": True,
            **report,
        }
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "size" in metafunc.fixturenames:
        metafunc.parametrize("size", audit_sizes())


def audit_sizes() -> tuple[int, ...]:
    raw = os.environ.get("BENCH_AUDIT_SIZES")
    if not raw:
        return DEFAULT_AUDIT_SIZES
    return tuple(int(value) for value in raw.replace(",", " ").split() if value.strip())


def chunk_molecules() -> int:
    return int(os.environ.get("BENCH_AUDIT_CHUNK", DEFAULT_CHUNK_MOLECULES))


# --------------------------------------------------------------------------- #
# Identity codecs, audit-local
# --------------------------------------------------------------------------- #
AUDIT_RDKIT_CODEC = "rdkit-canonical-smiles-isotope-preserving"
AUDIT_NATIVE_CODEC = "vesma-native-zmch-bondkind-v1"


def audit_smiles(molecule: Any) -> str:
    """Canonical RDKit SMILES with atom maps and stereo dropped, isotopes kept.

    Stereo is removed before writing, so ``isomericSmiles`` here controls the isotope
    label alone -- and the audit keeps it. Vesma carries an exact mass number per atom
    and a key space that erased it would report an isotopic relabelling as agreement.
    """
    from rdkit import Chem

    copied = Chem.Mol(molecule)
    for atom in copied.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(copied)
    return Chem.MolToSmiles(copied, canonical=True, isomericSmiles=True)


def audit_key_from_cmol(molecule: CMol) -> str:
    return audit_smiles(mol_from_cmol(molecule))


def audit_key_from_rdmol(molecule: Any) -> str:
    return audit_smiles(molecule)


def audit_key_from_smiles(smiles: str) -> str:
    from rdkit import Chem

    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return audit_smiles(parsed)


def native_key(molecule: CMol) -> str:
    """The exact vesma graph under a canonical atom order.

    Total where the cross-tool codec is partial: it never consults RDKit, so a precursor
    whose delocalized system RDKit refuses to kekulize still gets an identity instead of
    disappearing from the audit. It is also the identity a Metal or CUDA port has to
    reproduce atom for atom, which is why it encodes the state exactly rather than
    hashing it.
    """
    if molecule.atom_count == 0:
        return ""
    ordered = reorder_cmol(molecule, canonicalish_atom_order(molecule))
    atoms = ",".join(
        f"{z}.{m}.{c}.{h}"
        for z, m, c, h in zip(
            ordered.Z.tolist(),
            ordered.M.tolist(),
            ordered.C.tolist(),
            ordered.H.tolist(),
            strict=True,
        )
    )
    rows, cols = np.nonzero(np.triu(ordered.bonds, 1))
    bonds = ",".join(
        f"{int(i)}.{int(j)}.{int(ordered.bonds[i, j])}"
        for i, j in zip(rows.tolist(), cols.tolist(), strict=True)
    )
    return f"{atoms}|{bonds}"


# --------------------------------------------------------------------------- #
# One backend's account of one pair
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairResult:
    """Everything one backend did with one (molecule, rule) pair.

    ``outcomes`` is the comparable part: the applications that both succeeded and could
    be keyed in the shared space. The counters beside it are what keeps the comparison
    honest -- they say how much of the pair the outcome set does *not* represent.
    """

    outcomes: OutcomeSets = EMPTY
    accepted: int = 0
    unkeyable: int = 0
    rejections: RejectionCounts = ()
    failed: bool = False

    @property
    def refused(self) -> int:
        return sum(self.rejections)

    @property
    def attempted(self) -> int:
        return self.accepted + self.unkeyable + self.refused

    @property
    def comparable(self) -> bool:
        """Whether this pair's outcome set is a complete account of what the backend did."""
        return not self.failed and not self.unkeyable


BackendResults = dict[Pair, PairResult]


def result_for(results: BackendResults, pair: Pair) -> PairResult:
    return results.get(pair, PairResult())


# --------------------------------------------------------------------------- #
# Accumulators
# --------------------------------------------------------------------------- #
@dataclass
class PairingCounters:
    """Set-agreement tallies for one ordered backend pairing, over a fixed denominator."""

    pairs_compared: int = 0
    pairs_both_empty: int = 0
    pairs_incomparable: int = 0
    equal: int = 0
    equal_nonempty: int = 0
    equal_comparable: int = 0
    comparable: int = 0
    intersection_total: int = 0
    union_total: int = 0
    jaccard_sum: float = 0.0
    outcomes_left_only: int = 0
    outcomes_right_only: int = 0
    outcomes_shared: int = 0


@dataclass
class BackendCounters:
    """Output-volume and status tallies for one backend."""

    pairs: int = 0
    applications_accepted: int = 0
    applications_refused: int = 0
    applications_unkeyable: int = 0
    pairs_failed: int = 0
    distinct_outcomes: int = 0
    outcome_deduplicated_precursors: int = 0
    rejections: RejectionCounts = field(default_factory=no_rejections)
    distinct_precursor_keys: set[str] = field(default_factory=set)


@dataclass
class Accumulator:
    pairings: dict[tuple[str, str], PairingCounters]
    backends: dict[str, BackendCounters]
    divergences: list[dict[str, Any]] = field(default_factory=list)
    refusals: list[dict[str, Any]] = field(default_factory=list)
    unkeyable: list[dict[str, Any]] = field(default_factory=list)
    # One row per outcome RDKit built and threw away, and one per component vesma
    # dropped as undescribed. Both are things a backend did that its return value does
    # not show, so a total alone cannot be audited chemically -- and a total alone is
    # also easy to misread: 42 components over 27 applications was written up as 22,
    # which was the count of divergent *pairs* from the same template.
    rdkit_discards: list[dict[str, Any]] = field(default_factory=list)
    undescribed: list[dict[str, Any]] = field(default_factory=list)
    hydrogen_completion_atoms: int = 0
    hydrogen_completion_applications: int = 0
    # Applications vesma built whose RHS spelled a derived property the result does not
    # have. These are accepted outcomes, not refusals -- an RHS is a rewrite program and
    # these fields are consequences of running it. Kept because they say something about
    # the *templates*, and losing them would make a notation problem look like agreement.
    annotation_mismatches: AnnotationCounts = field(
        default_factory=no_annotation_mismatches
    )
    annotation_mismatch_applications: int = 0
    # Components the edit disconnected that carry no RHS atom. Not precursors -- the
    # template never wrote them -- but a fact about the template worth surfacing.
    undescribed_components: int = 0
    undescribed_component_applications: int = 0
    # What vesma's matcher enumerated, so its accounting can be checked rather than
    # trusted. Unlike RDKit's outcome total this is not comparable across backends --
    # the two enumerate different things -- but each must close against its own.
    vesma_applications_attempted: int = 0
    vesma_pairs_declined: int = 0
    vesma_pairs_search_incomplete: int = 0
    rdkit_outcomes_total: int = 0
    rdkit_outcomes_unsanitizable: int = 0


def new_accumulator() -> Accumulator:
    return Accumulator(
        pairings={pairing: PairingCounters() for pairing in combinations(BACKENDS, 2)},
        backends={backend: BackendCounters() for backend in BACKENDS},
    )


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Crash resilience
# --------------------------------------------------------------------------- #
# A checkpoint is transient local state, not a published artifact, so it is pickled
# rather than given a schema: it is read only by the process that resumes it, and the
# audit's own JSON remains the thing anyone else reads.
CHECKPOINT_FORMAT = "vesma-precursor-audit-checkpoint-v1"


@dataclass(frozen=True)
class CheckpointPlan:
    """What a checkpoint must match before it may be resumed.

    Resuming a checkpoint written by a different sweep, or by different chemistry, would
    splice two incompatible halves into one artifact and report the result as a single
    measurement -- a far worse failure than losing the run, because it would not look
    like a failure. Anything that does not match is discarded rather than adapted.
    """

    path: Path
    token: tuple[str, ...]


def checkpoint_plan(size: int, chunk: int, molecules: int) -> CheckpointPlan | None:
    root = os.environ.get("BENCH_AUDIT_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)
    if not root:
        return None
    return CheckpointPlan(
        path=Path(root) / f"audit-{size}.pickle",
        token=(
            CHECKPOINT_FORMAT,
            str(size),
            str(chunk),
            str(molecules),
            vesma_version(),
            chemistry_fingerprint(),
            rdkit_version(),
        ),
    )


def load_checkpoint(plan: CheckpointPlan | None) -> tuple[int, Accumulator]:
    """The next chunk to run and the state to run it into; chunk 0 and a fresh state if
    there is nothing usable to resume."""
    if plan is None or not plan.path.exists():
        return 0, new_accumulator()
    try:
        token, done, accumulator = pickle.loads(plan.path.read_bytes())
    except Exception as error:  # a truncated or unreadable checkpoint is not a failure
        warnings.warn(
            f"Ignoring unreadable audit checkpoint {plan.path}: {error}", stacklevel=2
        )
        return 0, new_accumulator()
    if tuple(token) != plan.token:
        warnings.warn(
            f"Ignoring audit checkpoint {plan.path}: it was written for a different "
            "sweep or a different version, and resuming it would merge two runs.",
            stacklevel=2,
        )
        return 0, new_accumulator()
    print(f"\nResuming audit from {plan.path} at molecule {done}.")
    return int(done), accumulator


def save_checkpoint(plan: CheckpointPlan | None, done: int, state: Accumulator) -> None:
    """Write through a temporary file, so a crash mid-write leaves the last one intact."""
    if plan is None:
        return
    plan.path.parent.mkdir(parents=True, exist_ok=True)
    temporary = plan.path.with_suffix(".pickle.partial")
    temporary.write_bytes(pickle.dumps((plan.token, done, state)))
    os.replace(temporary, plan.path)


def discard_checkpoint(plan: CheckpointPlan | None) -> None:
    """Once the artifact is complete the checkpoint is a trap, not a safety net."""
    if plan is not None:
        plan.path.unlink(missing_ok=True)


def audit_precursors(size: int) -> dict[str, Any]:
    from vesma.rdchiral_matcher import rdchiral_version

    all_indices = molecule_indices(size)
    chunk = chunk_molecules()
    plan = checkpoint_plan(size, chunk, len(all_indices))
    resumed, accumulator = load_checkpoint(plan)

    # Streaming in molecule chunks keeps peak memory at one chunk: materializing three
    # full outcome maps over the largest workload would run to gigabytes.
    #
    # The fold is also checkpointed after every chunk. Nothing reaches disk until the
    # whole audit returns, and the full sweep is 768,015 pairs over about half a day, so
    # without this a crash in the last hour costs the entire run.
    for start in range(resumed, len(all_indices), chunk):
        fold_chunk(accumulator, all_indices[start : start + chunk])
        save_checkpoint(plan, start + chunk, accumulator)
    discard_checkpoint(plan)

    return {
        "pairs_compared": accumulator.backends[VESMA].pairs,
        "agreement_records": [
            pairing_record(left, right, counters)
            for (left, right), counters in accumulator.pairings.items()
        ],
        "backend_records": [
            backend_record(backend, accumulator.backends[backend])
            for backend in BACKENDS
        ],
        "vesma_rejections": by_name(accumulator.backends[VESMA].rejections),
        "rdkit_dropped_outcomes": {
            "outcomes_total": accumulator.rdkit_outcomes_total,
            "outcomes_unsanitizable": accumulator.rdkit_outcomes_unsanitizable,
        },
        "hydrogen_completion": {
            "atoms": accumulator.hydrogen_completion_atoms,
            "applications": accumulator.hydrogen_completion_applications,
        },
        "rhs_annotation_mismatches": {
            "by_property": {
                name: count
                for name, count in zip(
                    RHS_ANNOTATIONS, accumulator.annotation_mismatches, strict=True
                )
                if count
            },
            "applications": accumulator.annotation_mismatch_applications,
        },
        "undescribed_components": {
            "components": accumulator.undescribed_components,
            "applications": accumulator.undescribed_component_applications,
            "records": accumulator.undescribed,
        },
        "rdkit_discards": accumulator.rdkit_discards,
        "vesma_enumeration": {
            "applications_attempted": accumulator.vesma_applications_attempted,
            "pairs_declined": accumulator.vesma_pairs_declined,
            "pairs_search_incomplete": accumulator.vesma_pairs_search_incomplete,
        },
        # Uncapped, all three: the rare cases are the ones worth reading.
        "divergences": accumulator.divergences,
        "refusals": accumulator.refusals,
        "unkeyable_precursors": accumulator.unkeyable,
        "cross_tool_codec": AUDIT_RDKIT_CODEC,
        "native_codec": AUDIT_NATIVE_CODEC,
        "vesma_version": vesma_version(),
        # The packaged version does not move between commits, so it cannot say which
        # chemistry produced this artifact. The fingerprint can, and the checkpoint
        # token carries the same value: a sweep cannot resume across a chemistry change.
        "chemistry_fingerprint": chemistry_fingerprint(),
        "rdkit_version": rdkit_version(),
        "rdchiral_version": rdchiral_version(),
        "rdchiral_combine_enantiomers": False,
        "vesma_processing": "default",
        "chunk_molecules": chunk,
    }


def fold_chunk(accumulator: Accumulator, indices: tuple[int, ...]) -> None:
    corpus = load_match_corpus_by_indices(indices)
    pairs = load_positive_pairs(corpus)
    if not pairs:
        return

    results = {
        VESMA: vesma_run(accumulator, corpus, pairs),
        RDKIT: rdkit_run(accumulator, corpus, pairs),
        RDCHIRAL: rdchiral_run(corpus, pairs),
    }

    for backend, backend_results in results.items():
        fold_backend(accumulator.backends[backend], backend_results, pairs)
    for left, right in accumulator.pairings:
        fold_pairing(
            accumulator.pairings[(left, right)], results[left], results[right], pairs
        )
    for pair in pairs:
        fold_divergence(accumulator, corpus, pair, results)


def fold_backend(
    counters: BackendCounters, results: BackendResults, pairs: Sequence[Pair]
) -> None:
    counters.pairs += len(pairs)
    for pair in pairs:
        result = result_for(results, pair)
        counters.applications_accepted += result.accepted
        counters.applications_refused += result.refused
        counters.applications_unkeyable += result.unkeyable
        counters.pairs_failed += int(result.failed)
        if result.rejections:
            counters.rejections = merged(counters.rejections, result.rejections)
        counters.distinct_outcomes += len(result.outcomes)
        for outcome in result.outcomes:
            # One outcome contributes one precursor per distinct component key; summing
            # over deduplicated outcomes is the count RDChiral's API already returns.
            counters.outcome_deduplicated_precursors += len(outcome)
            counters.distinct_precursor_keys.update(outcome)


def fold_pairing(
    counters: PairingCounters,
    left_results: BackendResults,
    right_results: BackendResults,
    pairs: Sequence[Pair],
) -> None:
    for pair in pairs:
        left = result_for(left_results, pair)
        right = result_for(right_results, pair)
        shared = len(left.outcomes & right.outcomes)
        union = len(left.outcomes | right.outcomes)
        comparable = left.comparable and right.comparable

        counters.pairs_compared += 1
        counters.outcomes_shared += shared
        counters.outcomes_left_only += len(left.outcomes) - shared
        counters.outcomes_right_only += len(right.outcomes) - shared
        counters.intersection_total += shared
        counters.union_total += union
        counters.jaccard_sum += 1.0 if union == 0 else shared / union
        counters.pairs_both_empty += int(not union)
        counters.comparable += int(comparable)
        counters.pairs_incomparable += int(not comparable)
        if left.outcomes == right.outcomes:
            counters.equal += 1
            counters.equal_nonempty += int(bool(union))
            counters.equal_comparable += int(comparable)


# --------------------------------------------------------------------------- #
# Divergence records
# --------------------------------------------------------------------------- #
def fold_divergence(
    accumulator: Accumulator,
    corpus: MatchCorpus,
    pair: Pair,
    results: dict[str, BackendResults],
) -> None:
    vesma = result_for(results[VESMA], pair)
    rdkit = result_for(results[RDKIT], pair)
    rdchiral = result_for(results[RDCHIRAL], pair)
    if vesma.outcomes == rdkit.outcomes == rdchiral.outcomes:
        return

    mol_idx, rule_id = pair
    template = corpus.templates[rule_id]
    accumulator.divergences.append(
        {
            # The frontier index, not the chunk-local one: pairs are keyed against the
            # chunk's corpus, which is re-indexed from zero per chunk.
            "molecule_index": corpus.source_molecule_indices[mol_idx],
            "smiles": corpus.smiles[mol_idx],
            "rule_id": rule_id,
            "template_id": template.id,
            "template": template.smarts,
            "partition": partition(vesma.outcomes, rdkit.outcomes, rdchiral.outcomes),
            "vesma_vs_rdkit": mechanism(vesma.outcomes, rdkit.outcomes),
            "vesma_vs_rdchiral": mechanism(vesma.outcomes, rdchiral.outcomes),
            "rdkit_duplicates_atoms": duplicates_atoms(vesma.outcomes, rdkit.outcomes),
            "rhs_declares_multiple_components": declares_multiple_components(
                template.smarts
            ),
            "vesma": serialize(vesma.outcomes),
            "rdkit": serialize(rdkit.outcomes),
            "rdchiral": serialize(rdchiral.outcomes),
            "vesma_rejections": by_name(vesma.rejections or no_rejections()),
            "statuses": {
                VESMA: status_of(vesma),
                RDKIT: status_of(rdkit),
                RDCHIRAL: status_of(rdchiral),
            },
        }
    )


def status_of(result: PairResult) -> str:
    """Whether the outcome set is the whole story for this pair, and if not, why not."""
    if result.failed:
        return "backend_failed"
    if result.unkeyable and not result.accepted:
        # Nothing survived to compare. Worth its own name because this is the case that
        # reads as a clean empty result from the return value alone -- at size 100 all
        # 101 pairs where RDKit "produced nothing" are in fact this.
        return "all_applications_unkeyable"
    if result.unkeyable:
        return "partly_unkeyable"
    if result.refused and not result.accepted:
        return "all_applications_refused"
    if result.refused:
        return "partly_refused"
    return "complete"


def partition(vesma: OutcomeSets, rdkit: OutcomeSets, rdchiral: OutcomeSets) -> str:
    """Which backend stands apart. Orthogonal to *why*, which `mechanism` answers."""
    if vesma == rdchiral:
        return "rdkit-alone"
    if vesma == rdkit:
        return "rdchiral-alone"
    if rdkit == rdchiral:
        return "vesma-alone"
    return "all-three-differ"


def mechanism(vesma: OutcomeSets, other: OutcomeSets) -> str:
    """How vesma's outcome set differs from one comparison backend's.

    Classification is by molecular formula, which separates "same atoms, drawn
    differently" from "different atoms" without needing a graph isomorphism.
    """
    if vesma == other:
        return "equal"
    if not vesma:
        return "vesma-empty"
    if not other:
        return "other-empty"
    if formula_signature(vesma) == formula_signature(other):
        return "same-formula"
    if heavy_signature(vesma) == heavy_signature(other):
        return (
            "hydrogen-only-vesma-gained"
            if hydrogen_total(vesma) > hydrogen_total(other)
            else "hydrogen-only-vesma-lost"
        )
    return "different-heavy-atoms"


def duplicates_atoms(vesma: OutcomeSets, rdkit: OutcomeSets) -> bool:
    """True when RDKit built a heavier outcome than vesma did from the same pair.

    On a disconnection template whose matched bond lies on a ring, RDKit follows the
    declared multi-component topology and copies the shared ring atoms into *both*
    products, while vesma applies the bond edit to the graph and returns the single
    ring-opened component. RDKit's outcome then carries more heavy atoms than vesma's.

    The comparison is against vesma's outcome for the same pair rather than against the
    input molecule: a retro template legitimately introduces atoms the product never had
    (leaving groups such as Br or OH), so "heavier than the input" fires on ordinary
    templates and does not isolate the duplication case.
    """
    if not vesma or not rdkit:
        return False
    return max(map(outcome_heavy_atoms, rdkit)) > max(map(outcome_heavy_atoms, vesma))


def declares_multiple_components(smarts: str) -> bool:
    """Whether the template's RHS asks for more than one precursor.

    This is what actually separates vesma from RDKit on the whole ``rdkit-alone`` class,
    and ``duplicates_atoms`` only sees part of it. When such a template matches around a
    ring, the two declared components are one molecule: vesma applies the bond edits and
    returns the single ring-opened precursor, while RDKit follows the declared topology
    and splits. Where the declared components share atoms RDKit copies them into both,
    which is the case ``duplicates_atoms`` catches; where they do not, RDKit severs a
    bond the template never asked to break and the outcome is no heavier than vesma's,
    so that test reads ``False`` on the very same phenomenon.

    Read off the SMARTS rather than the parsed pattern because it is a property of what
    the template *said*, and a dot in the RHS says it unambiguously.
    """
    return "." in smarts.split(">>", 1)[-1]


def serialize(outcomes: OutcomeSets) -> list[list[str]]:
    return sorted(sorted(outcome) for outcome in outcomes)


def formula_signature(outcomes: OutcomeSets) -> frozenset[tuple[str, ...]]:
    return frozenset(
        tuple(sorted(molecular_formula(key) for key in outcome)) for outcome in outcomes
    )


def heavy_signature(outcomes: OutcomeSets) -> frozenset[tuple[str, ...]]:
    return frozenset(
        tuple(sorted(heavy_formula(key) for key in outcome)) for outcome in outcomes
    )


def hydrogen_total(outcomes: OutcomeSets) -> int:
    if not outcomes:
        return 0
    return max(sum(hydrogen_count(key) for key in outcome) for outcome in outcomes)


def outcome_heavy_atoms(outcome: frozenset[str]) -> int:
    return sum(heavy_atom_count(key) for key in outcome)


@lru_cache(maxsize=200_000)
def parsed_molecule(smiles: str) -> Any:
    from rdkit import Chem

    return Chem.MolFromSmiles(smiles)


@lru_cache(maxsize=200_000)
def molecular_formula(smiles: str) -> str:
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    molecule = parsed_molecule(smiles)
    return "" if molecule is None else CalcMolFormula(molecule)


@lru_cache(maxsize=200_000)
def heavy_formula(smiles: str) -> str:
    """The molecular formula with its hydrogen term removed."""
    import re

    return re.sub(r"H\d*", "", molecular_formula(smiles))


@lru_cache(maxsize=200_000)
def hydrogen_count(smiles: str) -> int:
    molecule = parsed_molecule(smiles)
    if molecule is None:
        return 0
    return sum(
        atom.GetTotalNumHs() + (atom.GetAtomicNum() == 1)
        for atom in molecule.GetAtoms()
    )


@lru_cache(maxsize=200_000)
def heavy_atom_count(smiles: str) -> int:
    molecule = parsed_molecule(smiles)
    return 0 if molecule is None else molecule.GetNumHeavyAtoms()


# --------------------------------------------------------------------------- #
# Backend runs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KeyedApplication:
    """One application's precursors in the shared key space, or why they are not in it."""

    keys: frozenset[str]
    error: str
    precursor_index: int

    @property
    def keyed(self) -> bool:
        return not self.error


def keyed_applications(
    precursors: np.ndarray, hits: HitArray, key: Callable[[Any], str]
) -> dict[Application, KeyedApplication]:
    """Key every hit once, grouped by application.

    Keys are read through each hit's precursor column rather than by row position: a
    backend may return a deduplicated precursor batch shorter than the hit array.

    A single unkeyable component invalidates the whole application, the same treatment
    `rdkit_matcher.split_outcome` gives an outcome RDKit cannot sanitize -- a partial
    precursor set would compare as a different outcome rather than as a missing one.
    ``precursor_index`` points at the offending component when there is one, so the
    failure can be identified afterwards rather than merely counted.
    """
    keys: dict[Application, set[str]] = {}
    errors: dict[Application, str] = {}
    sample: dict[Application, int] = {}
    for mol_idx, rule_id, app_idx, precursor_idx in hits.tolist():
        application = (mol_idx, rule_id, app_idx)
        sample.setdefault(application, precursor_idx)
        try:
            encoded = key(precursors[precursor_idx])
        except Exception as exc:
            if application not in errors:
                errors[application] = f"{type(exc).__name__}: {exc}"
                sample[application] = precursor_idx
            continue
        keys.setdefault(application, set()).add(encoded)

    return {
        application: KeyedApplication(
            keys=(
                frozenset()
                if application in errors
                else frozenset(keys.get(application, ()))
            ),
            error=errors.get(application, ""),
            precursor_index=index,
        )
        for application, index in sample.items()
    }


def vesma_run(
    accumulator: Accumulator, corpus: MatchCorpus, pairs: Sequence[Pair]
) -> BackendResults:
    """Default CPU processing: no precursor atom cap, full ring descriptors.

    The question here is semantic agreement, so the GPU capacity cap is deliberately not
    applied -- it would drop precursors for reasons unrelated to how a template applies.
    """
    matcher = CpuReactionMatcher([template.pattern for template in corpus.templates])
    precursors, hits = matcher.match(corpus.cmols, pairs, n_jobs=1)

    keyed = keyed_applications(precursors, hits, audit_key_from_cmol)
    record_unkeyable(accumulator, corpus, VESMA, keyed, precursors, native=True)
    results = fold_vesma_refusals(
        accumulator,
        corpus,
        pair_results(keyed, pairs),
        matcher.last_rejected_applications,
    )

    diagnostics = matcher.last_diagnostics
    if diagnostics is not None:
        accumulator.hydrogen_completion_atoms += diagnostics.hydrogen_completion_atoms
        accumulator.hydrogen_completion_applications += (
            diagnostics.hydrogen_completion_applications
        )
        accumulator.annotation_mismatches = merged_annotations(
            accumulator.annotation_mismatches, diagnostics.annotation_mismatches
        )
        accumulator.annotation_mismatch_applications += (
            diagnostics.annotation_mismatch_applications
        )
        accumulator.vesma_applications_attempted += diagnostics.applications_attempted
        accumulator.vesma_pairs_declined += diagnostics.pairs_declined
        accumulator.vesma_pairs_search_incomplete += diagnostics.pairs_search_incomplete
        accumulator.undescribed_components += diagnostics.undescribed_components
        accumulator.undescribed_component_applications += (
            diagnostics.undescribed_component_applications
        )
        for mol_idx, rule_id, app_idx, formula in diagnostics.undescribed_rows:
            accumulator.undescribed.append(
                {
                    "molecule_index": corpus.source_molecule_indices[mol_idx],
                    "smiles": corpus.smiles[mol_idx],
                    "rule_id": rule_id,
                    "template_id": corpus.templates[rule_id].id,
                    "template": corpus.templates[rule_id].smarts,
                    "app_idx": app_idx,
                    "formula": formula,
                }
            )
    return results


def fold_vesma_refusals(
    accumulator: Accumulator,
    corpus: MatchCorpus,
    results: BackendResults,
    refused_rows: np.ndarray,
) -> BackendResults:
    """Attach each refused application to its pair and write it out in full.

    Every refusal is recorded, including those on pairs whose surviving applications
    agree with both comparison backends. An agreement reached after refusing most of the
    pair is still a fact about the model worth reading.
    """
    for mol_idx, rule_id, app_idx, reason_value in refused_rows.tolist():
        pair = (mol_idx, rule_id)
        reason = RejectionReason(reason_value)
        previous = result_for(results, pair)
        results[pair] = PairResult(
            outcomes=previous.outcomes,
            accepted=previous.accepted,
            unkeyable=previous.unkeyable,
            rejections=merged(previous.rejections or no_rejections(), counted(reason)),
            failed=previous.failed,
        )
        template = corpus.templates[rule_id]
        accumulator.refusals.append(
            {
                "molecule_index": corpus.source_molecule_indices[mol_idx],
                "smiles": corpus.smiles[mol_idx],
                "rule_id": rule_id,
                "template_id": template.id,
                "template": template.smarts,
                "app_idx": app_idx,
                "reason": reason.name.lower(),
            }
        )
    return results


def rdkit_run(
    accumulator: Accumulator, corpus: MatchCorpus, pairs: Sequence[Pair]
) -> BackendResults:
    matcher = RdkitReactionMatcher(
        rdkit_reactions_from_smarts(template_smarts(corpus)), n_jobs=1
    )
    precursors, hits = matcher.match(corpus.rd_mols, pairs)
    diagnostics = matcher.last_diagnostics
    discarded: dict[Pair, int] = {}
    if diagnostics is not None:
        accumulator.rdkit_outcomes_total += diagnostics.outcomes_total
        accumulator.rdkit_outcomes_unsanitizable += diagnostics.outcomes_unsanitizable
        for mol_idx, rule_id, app_idx, error in diagnostics.discarded:
            discarded[(mol_idx, rule_id)] = discarded.get((mol_idx, rule_id), 0) + 1
            accumulator.rdkit_discards.append(
                {
                    "molecule_index": corpus.source_molecule_indices[mol_idx],
                    "smiles": corpus.smiles[mol_idx],
                    "rule_id": rule_id,
                    "template_id": corpus.templates[rule_id].id,
                    "template": corpus.templates[rule_id].smarts,
                    "app_idx": app_idx,
                    "error": error,
                }
            )

    keyed = keyed_applications(precursors, hits, audit_key_from_rdmol)
    record_unkeyable(accumulator, corpus, RDKIT, keyed, precursors, native=False)
    return pair_results(keyed, pairs, discarded=discarded)


def rdchiral_run(corpus: MatchCorpus, pairs: Sequence[Pair]) -> BackendResults:
    """RDChiral with ``combine_enantiomers=False``.

    Vesma is stereo-blind and the audit codec strips stereochemistry, so letting the
    codec collapse enantiomers is more faithful than RDChiral's regex merge of
    ``@``/``@@`` SMILES pairs, which can synthesize an outcome no other backend had the
    chance to produce.
    """
    from vesma.rdchiral_matcher import (
        RdchiralReactionMatcher,
        rdchiral_reactants_from_smiles,
    )

    matcher = RdchiralReactionMatcher(
        load_rdchiral_reactions(template_smarts(corpus)),
        combine_enantiomers=False,
    )
    reactants = rdchiral_reactants_from_smiles(corpus.smiles)
    precursors, hits = matcher.match(reactants, pairs)
    diagnostics = matcher.last_diagnostics
    failed = set(diagnostics.failed_pairs) if diagnostics is not None else set()
    keyed = keyed_applications(precursors, hits, audit_key_from_smiles)
    return pair_results(keyed, pairs, failed=failed)


def pair_results(
    keyed: dict[Application, KeyedApplication],
    pairs: Sequence[Pair],
    *,
    failed: set[Pair] | None = None,
    discarded: Mapping[Pair, int] | None = None,
) -> BackendResults:
    """Collect keyed applications into one `PairResult` per pair.

    Every pair in ``pairs`` gets an entry even when the backend produced nothing, so the
    denominator is fixed by the pair list rather than by what happened to succeed.

    ``discarded`` counts applications the backend built and then threw away -- RDKit's
    unsanitizable outcomes. They belong with the unkeyable ones: in both cases the
    backend had something and the audit cannot see it, so the pair's outcome set is not
    the whole story and must not be compared as though it were. Counting them only in
    the run-wide total, as this did at first, leaves the pair reading `complete`, and a
    discard then scores as a genuine empty result.
    """
    outcomes: dict[Pair, set[frozenset[str]]] = {}
    accepted: dict[Pair, int] = {}
    excluded: dict[Pair, int] = dict(discarded or {})
    for (mol_idx, rule_id, _), application in keyed.items():
        pair = (mol_idx, rule_id)
        if application.keyed:
            outcomes.setdefault(pair, set()).add(application.keys)
            accepted[pair] = accepted.get(pair, 0) + 1
        else:
            excluded[pair] = excluded.get(pair, 0) + 1

    failed_pairs = failed or set()
    return {
        pair: PairResult(
            outcomes=frozenset(outcomes.get(pair, ())),
            accepted=accepted.get(pair, 0),
            unkeyable=excluded.get(pair, 0),
            rejections=no_rejections(),
            failed=pair in failed_pairs,
        )
        for pair in set(pairs) | failed_pairs
    }


def record_unkeyable(
    accumulator: Accumulator,
    corpus: MatchCorpus,
    backend: str,
    keyed: dict[Application, KeyedApplication],
    precursors: np.ndarray,
    *,
    native: bool,
) -> None:
    """Write out every precursor the shared key space cannot express.

    For vesma these carry the native key as well, so a precursor RDKit refuses to
    serialize is still identified rather than merely counted -- that is the whole reason
    the native codec exists.
    """
    for (mol_idx, rule_id, app_idx), application in sorted(keyed.items()):
        if application.keyed:
            continue
        template = corpus.templates[rule_id]
        accumulator.unkeyable.append(
            {
                "backend": backend,
                "molecule_index": corpus.source_molecule_indices[mol_idx],
                "smiles": corpus.smiles[mol_idx],
                "rule_id": rule_id,
                "template_id": template.id,
                "template": template.smarts,
                "app_idx": app_idx,
                "error": application.error,
                "native_key": (
                    native_key(precursors[application.precursor_index])
                    if native
                    else ""
                ),
            }
        )


# --------------------------------------------------------------------------- #
# Report records
# --------------------------------------------------------------------------- #
def pairing_record(left: str, right: str, counters: PairingCounters) -> dict[str, Any]:
    compared = counters.pairs_compared
    nonempty = compared - counters.pairs_both_empty
    return {
        "left": left,
        "right": right,
        "pairs_compared": compared,
        "pairs_both_empty": counters.pairs_both_empty,
        "pairs_incomparable": counters.pairs_incomparable,
        "equal": counters.equal,
        "equal_nonempty": counters.equal_nonempty,
        "equal_comparable": counters.equal_comparable,
        # The fixed denominator is every pair attempted. The other two rates are
        # published beside it, never instead of it: an agreement rate that rises when a
        # backend fails is measuring the wrong thing.
        "equal_rate": ratio(counters.equal, compared),
        "equal_rate_nonempty": ratio(counters.equal_nonempty, nonempty),
        "equal_rate_comparable": ratio(counters.equal_comparable, counters.comparable),
        "jaccard_mean": ratio(counters.jaccard_sum, compared),
        "jaccard_micro": ratio(counters.intersection_total, counters.union_total),
        "outcomes_shared": counters.outcomes_shared,
        "outcomes_left_only": counters.outcomes_left_only,
        "outcomes_right_only": counters.outcomes_right_only,
    }


def backend_record(backend: str, counters: BackendCounters) -> dict[str, Any]:
    return {
        "backend": backend,
        "pairs": counters.pairs,
        "applications_accepted": counters.applications_accepted,
        "applications_refused": counters.applications_refused,
        "applications_unkeyable": counters.applications_unkeyable,
        "pairs_failed": counters.pairs_failed,
        "distinct_outcomes": counters.distinct_outcomes,
        "outcome_deduplicated_precursors": counters.outcome_deduplicated_precursors,
        "distinct_precursor_keys": len(counters.distinct_precursor_keys),
        "rejections": by_name(counters.rejections),
    }


def ratio(numerator: float, denominator: float) -> float:
    return 0.0 if not denominator else numerator / denominator
