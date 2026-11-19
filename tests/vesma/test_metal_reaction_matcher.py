"""Metal high-level reaction-matcher orchestration parity against the CPU path.

Validates that ``MetalReactionMatcher`` (matcher + retro + decode + keying) produces
the same precursor WL tokens as ``CpuReactionMatcher`` + ``short_ring_wl_molecule_id``,
that the folded-fingerprint prefilter drops no true match, and that the molecule batch
satisfies the search ``MoleculeBatch`` protocol.
"""

from __future__ import annotations

import shutil
import sys
from collections import Counter

import numpy as np
import pytest
from rdkit import Chem

from tests.dedup_matcher import DEDUP_MODES, DedupingReactionMatcher, DedupMode
from vesma import gmol_from_cmol
from vesma.adjacency import bond_kind_matrix
from vesma.bonds import BOND_TOPOLOGY_ANY, BOND_TOPOLOGY_MASK
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.identity import short_ring_wl_molecule_id
from vesma.interop import cmol_from_mol
from vesma.reaction import parse_reaction_smarts
from vesma.rings import annotate_bond_topology, ring_info_from_bonds

pytestmark = [pytest.mark.gpu, pytest.mark.metal]

RULES_SMARTS = [
    "[c:1][c:2]>>[c:1].[c:2]",
    "[C:1][C:2]>>[C:1].[C:2]",
    "[C:1][O:2]>>[C:1].[O:2]",
    "[C:1][N:2]>>[C:1].[N:2]",
]
MOL_SMILES = ["c1ccccc1C", "CCO", "c1ccncc1", "CC(=O)Oc1ccccc1", "CCN", "Cc1ccc(N)cc1"]


def skip_if_no_metal_toolchain() -> None:
    if sys.platform != "darwin":
        pytest.skip("Metal backend is only available on macOS.")
    if shutil.which("clang++") is None:
        pytest.skip("clang++ is required for the Metal backend.")
    from vesma.metal_matcher import GpuUnavailableError, probe_metal

    try:
        probe_metal()
    except (GpuUnavailableError, RuntimeError) as exc:  # pragma: no cover
        pytest.skip(f"Metal backend not available: {exc}")


@pytest.mark.parametrize(
    "overflow_field", ("overflow", "atom_overflow", "edited_atom_overflow")
)
def test_precursor_gather_rejects_only_incomplete_assignments(
    overflow_field: str,
) -> None:
    from vesma.metal_reaction_matcher import _precursor_batch
    from vesma.metal_retro import MetalRetroOutput, RetroStatus

    status = (RetroStatus * 2)()
    status[0].component_count = 1
    setattr(status[0], overflow_field, 1)
    status[0].output_base = 0
    status[1].component_count = 1
    status[1].output_base = 1
    # Two emitted rows, of which the first belongs to the overflowed assignment. Built as the three
    # arrays the kernel now writes; this used to hand-build `GpuMolStruct` records and set their
    # hash fields, which the gather no longer reads because identity is not its business.
    produced = MetalRetroOutput(
        status=status,
        atoms=np.zeros((2, 14 * 64), dtype=np.uint32),
        bonds=np.zeros((2, 64 * 64), dtype=np.uint8),
        atom_counts=np.array([0, 1], dtype=np.uint32),
    )

    batch, hits = _precursor_batch(produced, np.array([[0, 0], [7, 3]], dtype=np.int64))

    # Only the complete assignment's row survives, and the hit points at it.
    assert len(batch) == 1
    assert batch.atom_counts.tolist() == [1]
    assert hits.tolist() == [[3, 7, 1, 0]]


def _rules_and_mols():
    rules = [parse_reaction_smarts(s) for s in RULES_SMARTS]
    cmols = [cmol_from_mol(Chem.MolFromSmiles(s)) for s in MOL_SMILES]
    return rules, cmols


def _cpu_keyed_hits(rules, cmols) -> Counter:
    precursors, hits = CpuReactionMatcher(rules).match(cmols)
    return Counter(
        (int(hit[0]), int(hit[1]), short_ring_wl_molecule_id(precursors[hit[3]]))
        for hit in hits
    )


def test_metal_reaction_matcher_precursor_keys_match_cpu() -> None:
    skip_if_no_metal_toolchain()
    from vesma.metal_reaction_matcher import (
        MetalReactionMatcher,
        metal_short_ring_wl_key,
        metal_stack_mols,
    )

    rules, cmols = _rules_and_mols()
    expected = _cpu_keyed_hits(rules, cmols)

    batch, hits = MetalReactionMatcher(rules).match(metal_stack_mols(cmols))
    keys = metal_short_ring_wl_key(batch)
    observed = Counter((int(hit[0]), int(hit[1]), keys[hit[3]]) for hit in hits)

    assert observed == expected
    assert sum(observed.values()) > 0


def test_metal_assignment_probe_retries_saturated_pairs() -> None:
    skip_if_no_metal_toolchain()
    from vesma.metal_reaction_matcher import (
        MetalReactionMatcher,
        metal_short_ring_wl_key,
        metal_stack_mols,
    )

    rule = parse_reaction_smarts("[C:1][C:2]>>[C:1].[C:2]")
    molecule = cmol_from_mol(Chem.MolFromSmiles("CCCCCCCCCC"))
    batch = metal_stack_mols([molecule])
    full = MetalReactionMatcher([rule], max_assignments=32, assignment_probe_cap=32)
    adaptive = MetalReactionMatcher([rule], max_assignments=32, assignment_probe_cap=4)

    full_mols, full_hits = full.match(batch)
    adaptive_mols, adaptive_hits = adaptive.match(batch)
    full_keys = metal_short_ring_wl_key(full_mols)
    adaptive_keys = metal_short_ring_wl_key(adaptive_mols)
    assert Counter(full_keys[row[3]] for row in full_hits) == Counter(
        adaptive_keys[row[3]] for row in adaptive_hits
    )
    assert adaptive.last_diagnostics is not None
    assert full.last_diagnostics is not None
    assert adaptive.last_diagnostics.match_retry_pairs == 1
    assert (
        adaptive.last_diagnostics.match_assignment_count
        == full.last_diagnostics.match_assignment_count
    )


@pytest.mark.parametrize("mode", DEDUP_MODES)
def test_metal_deduplicated_precursor_batch_keeps_the_same_hits(
    mode: DedupMode,
) -> None:
    """A packed Metal batch survives the deduplicated half of the matcher contract.

    Metal itself never collapses its precursor batch, so the wrapper stands in for a
    backend that does: the packed batch is gathered down to the surviving rows and the
    hits are renumbered against it. The keyed hit multiset must not move.
    """
    skip_if_no_metal_toolchain()
    from vesma.metal_reaction_matcher import (
        MetalReactionMatcher,
        metal_short_ring_wl_key,
        metal_stack_mols,
    )

    rules, cmols = _rules_and_mols()
    matcher = DedupingReactionMatcher(
        MetalReactionMatcher(rules), metal_short_ring_wl_key, mode
    )

    batch, hits = matcher.match(metal_stack_mols(cmols))
    keys = metal_short_ring_wl_key(batch)
    observed = Counter((int(hit[0]), int(hit[1]), keys[hit[3]]) for hit in hits)

    assert observed == _cpu_keyed_hits(*_rules_and_mols())
    if mode == "none":
        assert len(batch) == len(hits)
    elif mode == "full":
        assert len(batch) == len(set(keys)) < len(hits)
    else:
        assert len(set(keys)) <= len(batch) <= len(hits)


# Ring shapes a bounded small-ring detector gets wrong, plus the shapes a naive one does.
# Macrocycles are past the 3-8 window the WL identity flexer uses; biphenyl and the
# ring-ring single bond are bridges whose atoms all survive a 2-core; spiro and fused
# systems share an atom or an edge between rings.
RING_TOPOLOGY_SMILES = [
    "CC1CCCCC1",  # ring plus one exocyclic chain bond
    "CCCCCC",  # no ring at all
    "c1ccccc1-c1ccccc1",  # biphenyl: the bond joining two rings is a bridge
    "C1CCC(CC1)C1CCCCC1",  # the aliphatic twin of biphenyl
    "C1CCCCCCCCC1",  # 9-membered
    "C1CCCCCCCCCCC1",  # 12-membered
    "N1CCCCCCCCCCCCCCCCC1",  # 18-membered, past any bounded ring search
    "C1CC2CCC1CC2",  # bridged bicyclic
    "C1CCC2(CC1)CCCCC2",  # spiro
    "c1ccc2ccccc2c1",  # fused aromatic
    "C1=CC2=CC=CC3=C2C(=C1)C=C3",  # acenaphthylene: fused with a five-membered ring
    "CC(C)(C)c1ccc2OCOc2c1",  # fused hetero ring with a bulky chain substituent
    "OC(=O)CCc1ccc(cc1)N1CCOCC1",  # two rings joined by a bridge, plus a chain tail
]


@pytest.mark.parametrize("smiles", RING_TOPOLOGY_SMILES)
def test_metal_perceives_ring_topology_like_the_cpu(smiles: str) -> None:
    """Device ring-or-chain topology equals ``rings.annotate_bond_topology``, per edge.

    A bond is a ring bond iff it is not a bridge, so this is exact for any ring size --
    unlike the 3-8 detector the WL identity token uses, which these macrocycles defeat.
    """
    skip_if_no_metal_toolchain()
    from vesma.metal_reaction_matcher import MetalReactionMatcher, metal_stack_mols

    molecule = cmol_from_mol(Chem.MolFromSmiles(smiles))
    neutral = parse_reaction_smarts("[*:1]>>[*:1]")
    generated, _ = MetalReactionMatcher([neutral]).match(metal_stack_mols([molecule]))
    assert len(generated) > 0

    atom_count = molecule.Z.size
    reference = gmol_from_cmol(molecule).bonds
    expected = annotate_bond_topology(reference) & np.uint8(BOND_TOPOLOGY_MASK)
    ring_info = ring_info_from_bonds(reference)
    for index in range(len(generated)):
        device = generated.bonds[index].reshape(64, 64)[:atom_count, :atom_count]
        assert np.array_equal(device & np.uint8(BOND_TOPOLOGY_MASK), expected)
        # x falls out of the same non-bridge set, so it is exact wherever topology is.
        words = generated.atoms[index].reshape(64, 14)[:atom_count]
        assert words[:, 13].tolist() == [
            1 << int(value) for value in ring_info.ring_connectivity
        ]


RING_MEMBERSHIP_SMILES = [
    *RING_TOPOLOGY_SMILES,
    "C12C3C4C1C3C24",  # K3,3: nine relevant 4-rings, six through each atom, tied paths
    "C12C3C4C1C5C2C3C45",  # cubane: six 4-rings, three through each atom
    "C12C3C1C4C2C34",  # prismane
    "C1C2CC3CC1CC(C2)C3",  # adamantane
    "C1CC2CCC1C2",  # norbornane
    "C1C2CC1C2",  # bicyclo[1.1.1]pentane
    "C1CC2CC3CC1CC2C3",  # twistane
    "c1cc2ccc3cccc4ccc(c1)c2c34",  # pyrene
]


@pytest.mark.parametrize("smiles", RING_MEMBERSHIP_SMILES)
def test_metal_perceives_ring_membership_like_the_cpu(smiles: str) -> None:
    """Device `R` and `r` -- relevant cycles -- equal the host's, atom by atom."""
    skip_if_no_metal_toolchain()
    from vesma.metal_reaction_matcher import MetalReactionMatcher, metal_stack_mols

    molecule = cmol_from_mol(Chem.MolFromSmiles(smiles))
    neutral = parse_reaction_smarts("[*:1]>>[*:1]")
    generated, _ = MetalReactionMatcher([neutral]).match(metal_stack_mols([molecule]))
    assert len(generated) > 0
    atom_count = molecule.Z.size
    for index in range(len(generated)):
        words = generated.atoms[index].reshape(64, 14)[:atom_count]
        assert words[:, 11].tolist() == [1 << int(value) for value in molecule.R]
        assert words[:, 12].tolist() == [1 << int(value) for value in molecule.r]


def test_metal_generated_frontier_carries_exact_bond_topology() -> None:
    """A GPU-built precursor answers `@`/`!@` exactly, as a host-packed one does.

    retro_construct perceives ring-or-chain topology per edge, so the answer no longer
    depends on which retro layer the molecule came from. Methylcyclohexane is the case
    that separates the two: six ring bonds and one chain bond to the methyl, where an
    unknown-topology precursor would accept every C-C bond for both queries.
    """
    skip_if_no_metal_toolchain()
    from vesma.metal_reaction_matcher import MetalReactionMatcher, metal_stack_mols

    molecule = cmol_from_mol(Chem.MolFromSmiles("CC1CCCCC1"))
    neutral = parse_reaction_smarts("[C:1]-[C:2]>>[C:1]-[C:2]")
    topology_rules = [
        parse_reaction_smarts("[C:1]@[C:2]>>[C:1].[C:2]"),
        parse_reaction_smarts("[C:1]!@[C:2]>>[C:1].[C:2]"),
    ]

    host = metal_stack_mols([molecule])
    _, host_hits = MetalReactionMatcher(topology_rules).match(host)
    assert set(host_hits[:, 1].tolist()) == {0, 1}

    generated, _ = MetalReactionMatcher([neutral]).match(host)
    classes = bond_kind_matrix(generated.bonds)
    present = classes != 0
    topology = generated.bonds & np.uint8(BOND_TOPOLOGY_MASK)
    assert np.all(topology[present] != BOND_TOPOLOGY_ANY)
    assert np.all(generated.bonds[~present] == 0)

    # A neutral template keeps the product's atom order, so the device topology compares
    # position by position against the CPU annotation of the same molecule.
    atom_count = molecule.Z.size
    expected = annotate_bond_topology(gmol_from_cmol(molecule).bonds) & np.uint8(
        BOND_TOPOLOGY_MASK
    )
    for index in range(len(generated)):
        device = generated.bonds[index].reshape(64, 64)[:atom_count, :atom_count]
        assert np.array_equal(device & np.uint8(BOND_TOPOLOGY_MASK), expected)

    _, generated_hits = MetalReactionMatcher(topology_rules).match(generated)
    _, cpu_hits = CpuReactionMatcher(topology_rules).match([molecule])
    assert set(generated_hits[:, 1].tolist()) == set(cpu_hits[:, 1].tolist()) == {0, 1}


def test_metal_generated_frontier_refuses_chain_query_on_a_ring_bond() -> None:
    """The reported divergence: `-!@` found nothing on CPU and every ring bond on Metal.

    Layer one already agreed, because a host-packed frontier carries exact topology.
    Only the second layer, whose molecules the device built, disagreed.
    """
    skip_if_no_metal_toolchain()
    from vesma.metal_reaction_matcher import MetalReactionMatcher, metal_stack_mols

    molecule = cmol_from_mol(Chem.MolFromSmiles("CC1CCCCC1"))
    rules = [parse_reaction_smarts("[C:1]-!@[C:2]>>[C:1].[C:2]")]

    metal = MetalReactionMatcher(rules)
    batch, first_hits = metal.match(metal_stack_mols([molecule]))
    _, second_hits = metal.match(batch)

    cpu = CpuReactionMatcher(rules)
    cpu_precursors, cpu_first_hits = cpu.match([molecule])
    _, cpu_second_hits = cpu.match(list(cpu_precursors))

    assert len(first_hits) == len(cpu_first_hits)
    # Cyclohexane and methane: no chain carbon-carbon bond is left to disconnect.
    assert len(second_hits) == len(cpu_second_hits) == 0


def test_metal_ring_triple_identity_is_stable_across_generated_layer() -> None:
    skip_if_no_metal_toolchain()
    from vesma.metal_reaction_matcher import (
        MetalReactionMatcher,
        metal_short_ring_wl_key,
        metal_stack_mols,
    )

    molecule = cmol_from_mol(Chem.MolFromSmiles("C1CCCCCC#C1"))
    expected = short_ring_wl_molecule_id(molecule)
    neutral = parse_reaction_smarts("[C:1]#[C:2]>>[C:1]#[C:2]")
    matcher = MetalReactionMatcher([neutral])

    first, _ = matcher.match(metal_stack_mols([molecule]))
    assert len(first)
    assert set(metal_short_ring_wl_key(first)) == {expected}

    second, _ = matcher.match(first)
    assert len(second)
    assert set(metal_short_ring_wl_key(second)) == {expected}


def test_metal_folded_prefilter_keeps_all_matches() -> None:
    skip_if_no_metal_toolchain()
    from vesma.matcher import all_vs_all, keyed_match
    from vesma.metal_reaction_matcher import (
        MetalFoldedPrefilteredReactionMatcher,
        MetalReactionMatcher,
        metal_short_ring_wl_key,
        metal_stack_mols,
    )

    rules, cmols = _rules_and_mols()
    inner = MetalReactionMatcher(rules)
    dense_batch, dense_hits = inner.match(metal_stack_mols(cmols))
    dense_keys = metal_short_ring_wl_key(dense_batch)
    dense = Counter(
        (int(hit[0]), int(hit[1]), dense_keys[hit[3]]) for hit in dense_hits
    )

    pre = MetalFoldedPrefilteredReactionMatcher(inner)
    batch, keys, hits = keyed_match(all_vs_all(pre), metal_short_ring_wl_key)(
        metal_stack_mols(cmols)
    )
    filtered = Counter((int(hit[0]), int(hit[1]), keys[hit[3]]) for hit in hits)

    assert filtered == dense
    # The composed lift answers exactly what the keyer answers for that batch.
    assert keys == metal_short_ring_wl_key(batch)


def test_metal_reaction_diagnostics_stage_timing() -> None:
    skip_if_no_metal_toolchain()
    from vesma.metal_reaction_matcher import MetalReactionMatcher, metal_stack_mols

    rules, cmols = _rules_and_mols()
    matcher = MetalReactionMatcher(rules)
    batch, _ = matcher.match(metal_stack_mols(cmols))
    assert len(batch) > 0  # this corpus produces precursors, so the retro path ran

    diag = matcher.last_diagnostics
    assert diag is not None
    stages = (
        diag.match_seconds,
        diag.assignment_collect_seconds,
        diag.retro_seconds,
        diag.retro_construct_kernel_seconds,
        diag.wl_kernel_seconds,
        diag.precursor_gather_seconds,
        diag.precursor_concat_seconds,
    )
    assert all(s >= 0.0 for s in stages)
    assert 0.0 < diag.match_kernel_seconds <= diag.match_seconds
    assert diag.match_buffer_seconds + diag.match_copyback_seconds <= diag.match_seconds
    assert diag.match_template_upload_bytes > 0
    assert diag.match_assignment_count > 0
    assert diag.match_positive_pairs <= diag.match_assignment_count
    assert diag.match_output_copyback_bytes == (
        (len(cmols) * len(rules) + diag.match_retry_pairs) * 4 * 4
        + (diag.match_assignment_count + diag.match_retry_pairs * 8) * 64 * 4
    )
    matcher.match(metal_stack_mols(cmols))
    repeated = matcher.last_diagnostics
    assert repeated is not None
    assert repeated.match_template_upload_bytes == 0
    # Construct registers GPU time once precursors exist, bounded by the retro dispatch wall
    # time. `wl_kernel_seconds` is **zero**, and that is the claim: the matcher keys nothing, so
    # no WL work happens in this dispatch. Identity is `metal_short_ring_wl_key`'s business, and
    # the cost of it arrives on `MetalWlHash.seconds` at whatever call asked for keys. The field
    # stays rather than going missing, so a reader sees "no WL work here" instead of a gap.
    assert diag.retro_construct_kernel_seconds > 0.0
    assert diag.wl_kernel_seconds == 0.0
    assert diag.retro_construct_kernel_seconds <= diag.retro_seconds + 1e-6
    # Per-stage wall times account for the bulk of the call (allow slack for stacking +
    # pair building outside the timed stages) and never exceed the total.
    stage_wall = (
        diag.match_seconds
        + diag.assignment_collect_seconds
        + diag.retro_seconds
        + diag.precursor_gather_seconds
        + diag.precursor_concat_seconds
    )
    assert stage_wall <= diag.total_seconds + 1e-6
    # The join has to land inside the total, which is the whole reason it is counted: it used to
    # be stamped after `total_seconds` and so went missing, and at 2,881 molecules the phase it
    # hid was the largest line in the budget.
    assert diag.precursor_concat_seconds > 0.0


def test_metal_prefilter_reports_candidate_pairs() -> None:
    skip_if_no_metal_toolchain()
    from vesma.metal_reaction_matcher import (
        MetalFoldedPrefilteredReactionMatcher,
        MetalReactionMatcher,
        metal_stack_mols,
    )

    rules, cmols = _rules_and_mols()
    pre = MetalFoldedPrefilteredReactionMatcher(MetalReactionMatcher(rules))
    pre.match(metal_stack_mols(cmols))

    # The prefilter can only keep pairs that exist, never invent them, and its diagnostics
    # proxy the wrapped matcher's per-stage timing.
    assert 0 <= pre.last_candidate_pair_count <= len(rules) * len(cmols)
    assert pre.last_prefilter_seconds >= 0.0
    assert pre.last_diagnostics is not None
    assert pre.last_diagnostics.total_seconds >= 0.0


def test_metal_reaction_matcher_empty_pairs_refresh_diagnostics() -> None:
    from vesma.metal_reaction_matcher import (
        MetalMoleculeBatch,
        MetalReactionDiagnostics,
        MetalReactionMatcher,
    )

    matcher = MetalReactionMatcher.__new__(MetalReactionMatcher)
    matcher._rules = [object()]
    matcher._n_patterns = 1
    matcher.last_diagnostics = MetalReactionDiagnostics(total_seconds=123.0)
    batch = MetalMoleculeBatch(
        atom_counts=np.array([1], dtype=np.uint32),
        atoms=np.zeros((1, 16), dtype=np.uint32),
        bonds=np.zeros((1, 64), dtype=np.uint8),
        rows=np.zeros((1, 8), dtype=np.uint64),
    )

    precursors, hits = matcher.match(batch, np.empty((0, 2), dtype=np.int64))

    assert len(precursors) == 0
    assert hits.shape == (0, 4)
    assert matcher.last_diagnostics is not None
    assert matcher.last_diagnostics.total_seconds < 123.0


def test_metal_molecule_batch_getitem() -> None:
    skip_if_no_metal_toolchain()
    from vesma.metal_reaction_matcher import (
        MetalReactionMatcher,
        metal_short_ring_wl_key,
        metal_stack_mols,
    )

    rules, cmols = _rules_and_mols()
    batch, _ = MetalReactionMatcher(rules).match(metal_stack_mols(cmols))
    n = len(batch)
    assert n > 3

    # Integer index array (order preserved), bool mask, slice.
    sub = batch[np.array([2, 0])]
    assert len(sub) == 2
    keys = metal_short_ring_wl_key(batch)
    # Keying a gather equals gathering the keys -- which is the property a stored `keys` field used
    # to provide by construction and the keyer now has to earn.
    assert metal_short_ring_wl_key(sub) == [keys[2], keys[0]]

    mask = np.zeros(n, dtype=bool)
    mask[[1, 3]] = True
    assert metal_short_ring_wl_key(batch[mask]) == [keys[1], keys[3]]

    assert metal_short_ring_wl_key(batch[1:3]) == keys[1:3]

    # Scalar int must be rejected to preserve the batch axis.
    with pytest.raises(TypeError, match="preserve the batch axis"):
        batch[0]


def test_a_frontier_batch_can_be_keyed_on_device() -> None:
    """One keyer, total over any batch this arm can hold, agreeing with the CPU codec.

    This was a `strict=True` xfail stating the goal, and before that its opposite -- a test pinning
    that a frontier *refused* to be keyed, because keys were a field `retro_construct`'s fused pass
    filled and a frontier had no such record. The refusal was the honest answer to a partial `key`
    block; the block is total now, so the refusal has nothing left to refuse.

    What makes one keyer enough is that the WL token ignores bond topology annotation
    (`tests/vesma/test_metal_retro.py`), so host-packed and device-built bonds give the same token
    and the keyer never has to ask which kind of batch it holds.
    """
    from vesma import cmol_from_smiles
    from vesma.keyer import short_ring_wl_molecule_id
    from vesma.metal_reaction_matcher import metal_short_ring_wl_key, metal_stack_mols

    mols = [
        cmol_from_smiles(text)[0]
        for text in ("C1CCNCC1", "CCNC", "O=C(CN1CCCC1)Nc1ccc(Br)cc1")
    ]
    frontier = metal_stack_mols(mols)
    assert metal_short_ring_wl_key(frontier) == [
        short_ring_wl_molecule_id(one) for one in mols
    ]


@pytest.mark.metal
def test_chunking_the_keyer_changes_no_token() -> None:
    """The keyer slices large batches, and a slice boundary must not be visible in a token.

    Exact rather than approximate, because the WL kernel is independent per molecule: one
    threadgroup reads one molecule's rows and writes one molecule's hash, and nothing crosses a
    row. So this is the property that lets the keyer stay *total* over any batch instead of
    raising once the upload would outgrow a single device buffer -- which is the same partiality
    the `keys` field had, just at a different boundary.

    Driven by lowering the chunk size rather than by building a batch big enough to need the real
    one: the production slice is 64k rows and no batch here comes close, so the only way to
    exercise the seam is to move it.
    """
    skip_if_no_metal_toolchain()
    import vesma.metal_reaction_matcher as module

    rules, cmols = _rules_and_mols()
    batch, _ = module.MetalReactionMatcher(rules).match(module.metal_stack_mols(cmols))
    assert len(batch) > 3, "too few rows to slice"

    whole = module.metal_short_ring_wl_key(batch)
    original = module._WL_KEY_ROWS_PER_DISPATCH
    try:
        module._WL_KEY_ROWS_PER_DISPATCH = 2
        sliced = module.metal_short_ring_wl_key(batch)
    finally:
        module._WL_KEY_ROWS_PER_DISPATCH = original

    assert sliced == whole
    assert len(whole) == len(batch), "one token per row, whatever the slicing"


@pytest.mark.metal
def test_screening_explicit_pairs_loses_no_outcome() -> None:
    """The forward probe's path: explicit pairs, screened, must answer as the dense matcher does.

    This is the one the suite was missing. The probe hands this matcher a sparse, deliberately
    chosen pair set, and until recently the prefilter ignored a given pair set entirely and sent
    every pair to the kernel. Screening them is sound only because the fingerprint test is
    subset-sound -- it can let through more pairs than a sharper test would, never fewer -- and
    `test_prefilter_soundness.py` holds that up over the shipped library. What this adds is the
    end-to-end consequence on *this* path: same outcomes, not merely same surviving pairs.

    Asserted on the keyed outcome multiset rather than on hit rows, because the screen changes
    which pairs the matcher is asked about and therefore the order and the `app_idx` numbering --
    neither of which is an answer. What must not change is the set of molecules built, and which
    (molecule, rule) built each one.
    """
    skip_if_no_metal_toolchain()
    from vesma.metal_reaction_matcher import (
        MetalFoldedPrefilteredReactionMatcher,
        MetalReactionMatcher,
        metal_short_ring_wl_key,
        metal_stack_mols,
    )

    rules, cmols = _rules_and_mols()
    pairs = [(mol, rule) for mol in range(len(cmols)) for rule in range(len(rules))]

    inner = MetalReactionMatcher(rules)
    batch = metal_stack_mols(cmols)
    dense_precursors, dense_hits = inner.match(batch, pairs)
    dense_keys = metal_short_ring_wl_key(dense_precursors)
    dense = Counter(
        (int(hit[0]), int(hit[1]), dense_keys[hit[3]]) for hit in dense_hits
    )

    pre = MetalFoldedPrefilteredReactionMatcher(inner)
    screened_precursors, screened_hits = pre.match(batch, pairs)
    screened_keys = metal_short_ring_wl_key(screened_precursors)
    screened = Counter(
        (int(hit[0]), int(hit[1]), screened_keys[hit[3]]) for hit in screened_hits
    )

    assert screened == dense
    # And the screen has to have done something, or this proves only that it is harmless: with
    # the screen disabled this count equals the pair count exactly.
    assert pre.last_candidate_pair_count < len(pairs)
    assert pre.last_prefilter_seconds > 0.0


def test_batched_precursors_are_the_materialised_ones() -> None:
    """`match_batches` delivers exactly what `match` joins, in the same order.

    Only the words inside each precursor's `atom_count` are compared. The arrays past it carry
    whatever the previous dispatch left in the reused output mirror, so two plain `match` calls
    are not byte-equal to each other either -- the live data is what either caller may read.
    """
    skip_if_no_metal_toolchain()
    from vesma.metal_reaction_matcher import (
        MetalReactionMatcher,
        concat_hits,
        metal_concat_batches,
    )

    def live(batch):
        valid = np.arange(64)[None, :] < batch.atom_counts[:, None]
        bond = valid[:, :, None] & valid[:, None, :]
        return (
            batch.atom_counts,
            batch.atoms.reshape(-1, 64, 14)[valid],
            batch.bonds.reshape(-1, 64, 64)[bond],
        )

    rules, cmols = _rules_and_mols()
    matcher = MetalReactionMatcher(rules)

    expected_batch, expected_hits = matcher.match(cmols)
    expected = live(expected_batch)
    expected_hits = expected_hits.copy()
    expected_rejections = matcher.last_diagnostics.rejections

    parts = []
    hit_parts = []
    for batch, hits in matcher.match_batches(cmols):
        parts.append(batch)
        hit_parts.append(hits)
    assert parts, "the generator yielded no precursors at all"

    joined = metal_concat_batches(parts)
    joined_hits = concat_hits(hit_parts, [len(part) for part in parts])
    assert len(joined) == len(expected_batch)
    for name, want, got in zip(("counts", "atoms", "bonds"), expected, live(joined)):
        assert np.array_equal(want, got), f"delivered {name} differ"
    assert np.array_equal(joined_hits, expected_hits)
    assert matcher.last_diagnostics.rejections == expected_rejections


def test_match_counts_the_join_and_batches_do_not() -> None:
    """The join is `match`'s phase, so only `match` reports time against it."""
    skip_if_no_metal_toolchain()
    from vesma.metal_reaction_matcher import MetalReactionMatcher

    rules, cmols = _rules_and_mols()
    matcher = MetalReactionMatcher(rules)

    matcher.match(cmols)
    joined = matcher.last_diagnostics
    assert joined is not None
    assert joined.precursor_concat_seconds > 0.0
    # The join is inside the total it reports, which is the whole reason it is counted: it used
    # to be stamped after `total_seconds` and so went missing.
    assert joined.precursor_concat_seconds < joined.total_seconds

    for _batch, _hits in matcher.match_batches(cmols):
        pass
    streamed = matcher.last_diagnostics
    assert streamed is not None
    assert streamed.precursor_concat_seconds == 0.0
    # Same kernels either way, so the device-side phases agree; only the host join differs.
    assert streamed.retro_construct_kernel_seconds > 0.0


def test_consumer_time_between_pulls_is_not_billed_to_the_matcher() -> None:
    """A generator's clock runs while it is suspended; the matcher must not charge for that.

    `total_seconds` was stamped as elapsed-since-entry, which for a pull-based caller meant every
    interval the generator sat parked at a `yield` counted as matcher time. A folded 100-target
    route recovery reported 39.6 s of matcher time whose own stages summed to 28.8 s, and the
    missing 10.8 s was the consumer's keyer. Sleeping between pulls is the same shape of lie,
    large enough to measure: the sleep has to land in `consumer_seconds` and nowhere else.
    """
    skip_if_no_metal_toolchain()
    from time import sleep

    from vesma.metal_reaction_matcher import MetalReactionMatcher

    rules, cmols = _rules_and_mols()
    matcher = MetalReactionMatcher(rules)

    pulls = 0
    for _batch, _hits in matcher.match_batches(cmols):
        pulls += 1
        sleep(0.05)
    assert pulls > 0

    diag = matcher.last_diagnostics
    assert diag is not None
    slept = 0.05 * pulls
    # The sleep is accounted as the consumer's, within a round of scheduler slack.
    assert diag.consumer_seconds >= slept * 0.9
    # And excluded from the matcher's own total, which is what the stage sums are compared to.
    stage_wall = (
        diag.match_seconds
        + diag.assignment_collect_seconds
        + diag.retro_seconds
        + diag.precursor_gather_seconds
    )
    assert stage_wall <= diag.total_seconds + 1e-6
    # The sharp end: without the fix `total_seconds` would exceed the stages by the whole sleep.
    assert diag.total_seconds < slept


def test_joined_match_charges_almost_nothing_to_its_consumer() -> None:
    """`match` is its own consumer and only appends, so the parked time is negligible."""
    skip_if_no_metal_toolchain()
    from vesma.metal_reaction_matcher import MetalReactionMatcher

    rules, cmols = _rules_and_mols()
    matcher = MetalReactionMatcher(rules)
    matcher.match(cmols)

    diag = matcher.last_diagnostics
    assert diag is not None
    assert diag.consumer_seconds >= 0.0
    assert diag.consumer_seconds < 0.5 * diag.total_seconds


def test_abandoning_the_batch_generator_still_reports_what_it_ran() -> None:
    """A consumer that stops pulling leaves work undone, and that call is still described.

    Worth pinning because it is a real behaviour change from a function that always ran to
    completion: the diagnostics are stamped in a `finally`, so an abandoned call reports the
    pieces that were pulled rather than vanishing or reporting the previous call's numbers.
    """
    skip_if_no_metal_toolchain()
    from vesma.metal_reaction_matcher import MetalReactionMatcher

    rules, cmols = _rules_and_mols()
    matcher = MetalReactionMatcher(rules)

    full_batch, _ = matcher.match(cmols)
    full = matcher.last_diagnostics
    assert full is not None
    full_assignments = full.match_assignment_count

    batches = matcher.match_batches(cmols)
    first_batch, _first_hits = next(batches)
    batches.close()  # GeneratorExit through the `finally`

    partial = matcher.last_diagnostics
    assert partial is not None
    # It ran, so it reports work; it stopped early, so it reports less than the whole call.
    assert 0 < partial.match_assignment_count <= full_assignments
    assert partial.retro_construct_kernel_seconds > 0.0
    assert len(first_batch) <= len(full_batch)


def test_folding_keyed_match_over_metal_batches_agrees_with_the_joined_batch() -> None:
    """The generic fold, on the real backend: same molecules, same hit provenance, deduplicated.

    `keyed_match_folding` collapses by key as pieces arrive, so its batch holds one row per
    distinct molecule where `match` holds one per application. The claim is that nothing is lost
    in the collapse: every (molecule, rule, key) triple the joined batch supports is supported by
    the folded one, and every row it returns is named by a hit.
    """
    skip_if_no_metal_toolchain()
    from vesma.matcher import keyed_match_folding
    from vesma.metal_reaction_matcher import (
        MetalReactionMatcher,
        metal_concat_batches,
        metal_short_ring_wl_key,
    )

    rules, cmols = _rules_and_mols()
    matcher = MetalReactionMatcher(rules)

    joined_batch, joined_hits = matcher.match(cmols)
    joined_keys = metal_short_ring_wl_key(joined_batch)

    keyed = keyed_match_folding(
        lambda mols: matcher.match_batches(mols),
        metal_short_ring_wl_key,
        metal_concat_batches,
    )
    folded_batch, folded_keys, folded_hits = keyed(cmols)

    # One row per distinct molecule, and the keys are parallel to the batch.
    assert len(folded_keys) == len(folded_batch)
    assert len(set(folded_keys)) == len(folded_keys)
    assert set(folded_keys) == set(joined_keys)
    assert len(folded_batch) <= len(joined_batch)
    # No orphans: the contract a consumer indexes against.
    assert set(folded_hits[:, 3].tolist()) == set(range(len(folded_batch)))
    # Same provenance multiset, which is what parity compares and what the store records.
    folded_triples = Counter(
        (int(row[0]), int(row[1]), folded_keys[int(row[3])]) for row in folded_hits
    )
    joined_triples = Counter(
        (int(row[0]), int(row[1]), joined_keys[int(row[3])]) for row in joined_hits
    )
    assert folded_triples == joined_triples
