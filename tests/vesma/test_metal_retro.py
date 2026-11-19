"""Metal retrosynthesis kernel parity against the CPU reference.

Drives the same matcher assignments into both the CPU bond-edit path
(``precursors_from_mapping`` + ``short_ring_wl_molecule_id``) and the Metal
``retro_construct`` kernel, and asserts the per-assignment short-ring WL token
sets agree. This isolates the kernel (perception, property/H refresh, WL hash)
from the matcher, which is covered by ``test_metal_matcher.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vesma import cmol_from_smiles, gmol_from_cmol
from vesma.cpu_matcher import CpuMatcher
from vesma.identity import short_ring_wl_molecule_id
from vesma.reaction import parse_reaction_smarts, precursors_from_mapping
from vesma.rejection import ChemistryRejection, RejectionReason
from vesma.rings import RingDescriptors

pytestmark = [pytest.mark.gpu, pytest.mark.metal]


def skip_if_no_metal_toolchain() -> None:
    from vesma.metal_matcher import GpuUnavailableError, probe_metal

    try:
        probe_metal()
    except (GpuUnavailableError, RuntimeError) as exc:
        pytest.skip(f"Metal backend not available: {exc}")


def _cpu_assignments(product, reaction):
    matcher = CpuMatcher()
    result = matcher.find_matches(reaction.lhs, gmol_from_cmol(product))
    return np.asarray(result.assignments)[:, : reaction.lhs.atom_count]


def _token(produced: Any, row: int) -> tuple[int, int]:
    """One emitted row's WL identity, through the keyer the arm actually uses.

    These used to read `mols[row].hash_lo/.hash_hi` -- hash fields the construct kernel filled on
    an interleaved `GpuMol`. The kernel writes three parallel arrays now and keys nothing, so the
    token comes from `metal_wl_hash`, which means these tests exercise the production keyer rather
    than a kernel side effect.
    """
    from vesma.metal_retro import metal_wl_hash

    keyed = metal_wl_hash(
        produced.atoms[row : row + 1],
        produced.bonds[row : row + 1],
        produced.atom_counts[row : row + 1],
    )
    return int(keyed.hash_lo[0]), int(keyed.hash_hi[0])


def _cpu_outcomes(product, reaction, assignments):
    """Per mapping, either the precursor token set or the reason it was refused.

    A refusal is an outcome, not a missing one, so it belongs in the comparison. Keeping
    them apart -- comparing only the applications that succeeded -- would let a backend
    agree by refusing everything it found hard.
    """
    outcomes = []
    for mapping in assignments:
        try:
            precursors = precursors_from_mapping(
                product=product,
                reaction=reaction,
                mapping=mapping,
                ring_descriptors=RingDescriptors.CONNECTIVITY,
            )
        except ChemistryRejection as refusal:
            outcomes.append(refusal.reason)
            continue
        outcomes.append(
            frozenset(short_ring_wl_molecule_id(precursor) for precursor in precursors)
        )
    return outcomes


def _metal_outcomes(product, reaction, assignments):
    from vesma.metal_retro import (
        REJECTION_NONE,
        MetalRetroConstructor,
        reaction_data_from_pattern,
    )

    n = assignments.shape[0]
    if n == 0:
        return []
    a64 = np.full((n, 64), -1, dtype=np.int32)
    a64[:, : assignments.shape[1]] = assignments
    rdata = reaction_data_from_pattern(reaction)
    produced = MetalRetroConstructor().construct_batch(
        [gmol_from_cmol(product)], [rdata], a64, [(0, 0)] * n
    )
    # WL tokens live on the emitted precursors (output_mols), packed at status.output_base
    # -- the same place the production _precursor_batch reads them from.
    outcomes = []
    for i in range(n):
        if produced.status[i].rejection_reason != REJECTION_NONE:
            outcomes.append(RejectionReason(produced.status[i].rejection_reason))
            continue
        cc = produced.status[i].component_count
        base = produced.status[i].output_base
        outcomes.append(frozenset(_token(produced, base + c) for c in range(cc)))
    return outcomes


CASES = [
    # Heavy-atom valence options (the CPU/CUDA parity grid).
    ("CC", "[C:1][C:2]>>[C:1].[C:2]"),
    ("COC", "[C:1][O][C:2]>>[C:1]-[C:2]"),
    ("C", "[C:1]>>[C:1]-O"),
    ("C", "[C:1]>>[C:1].[O-]"),
    ("C", "[C:1]>>[OH-]"),
    ("N", "[N:1]>>[NH2:1]"),
    ("C1=CC1", "[C:1]1[C:2][C:3]1>>[C:1]1=[C:2][C:3]1"),
    ("C1=CCC1", "[C:1]1[C:2][C:3][C:4]1>>[C:1]1=[C:2][C:3][C:4]1"),
    ("C", "C>>O"),
    ("CS", "[C:1][S:2]>>[C:1].[S:2]"),
    ("CP", "[C:1][P:2]>>[C:1].[P:2]"),
    ("CB", "[C:1][B:2]>>[C:1].[B:2]"),
    ("CCl", "[C:1][Cl:2]>>[C:1].[Cl:2]"),
    ("CF", "[C:1][F:2]>>[C:1].[F:2]"),
    ("CS(C)(=O)=O", "[C:1][S:2]>>[C:1].[S:2]"),
    ("OP(O)O", "[O:1][P:2]>>[O:1].[P:2]"),
    # Aromaticity re-perception: breaking aromatic/heteroaromatic/fused rings,
    # and detaching substituents while a ring stays aromatic.
    ("c1ccccc1", "[c:1][c:2]>>[c:1].[c:2]"),
    ("c1ccncc1", "[c:1][c:2]>>[c:1].[c:2]"),
    ("c1cc[nH]c1", "[c:1][c:2]>>[c:1].[c:2]"),
    ("c1ccoc1", "[c:1][c:2]>>[c:1].[c:2]"),
    ("c1ccc2ccccc2c1", "[c:1][c:2]>>[c:1].[c:2]"),
    ("c1ccccc1C", "[c:1][C:2]>>[c:1].[C:2]"),
    ("Cc1ccccc1", "[C:1][c:2]>>[C:1].[c:2]"),
    ("c1ccccc1-c1ccccc1", "[c:1]-[c:2]>>[c:1].[c:2]"),
    ("Cc1ccc(N)cc1", "[c:1][N:2]>>[c:1].[N:2]"),
    ("O=Cc1ccccc1", "[C:1][c:2]>>[C:1].[c:2]"),
    # Keep the precursor ring intact to exercise the 8-atom identity cutoff,
    # fused/bridged alternatives, and acyclic paths after deleting a test edge.
    ("C1CCCCC1", "[C:1]>>[C:1]"),
    ("C1CCCCCCC1", "[C:1]>>[C:1]"),
    ("C1CCCCCCCC1", "[C:1]>>[C:1]"),
    ("C1CCC2CCCCC2C1", "[C:1]>>[C:1]"),
    ("C1CC2CCC1C2", "[C:1]>>[C:1]"),
    # An amine bridging a biaryl's meta positions, and one atom longer, para. These are the
    # shape where the atom mask and the accepted circuit's edges came apart on both arms: the
    # fused component's joint count passes, the dependent outer cycle is chordless so it is
    # counted as an elementary face, and every edge at the nitrogen then lies on two faces
    # instead of one. The host refused the molecule outright and the device wrote it into the
    # atom word and read it back two ways. An identity edit on the nitrogen re-perceives the
    # whole graph on both arms, so the two must agree that the nitrogen is aliphatic.
    ("N1c2cccc(c2)-c2ccccc21", "[N:1]>>[N:1]"),
    ("N1c2ccc(cc2)-c2ccccc21", "[N:1]>>[N:1]"),
    ("N1c2ccccc2-c2ccccc21", "[n:1]>>[n:1]"),
]


@pytest.mark.parametrize(("product_smiles", "reaction_smarts"), CASES)
def test_metal_retro_matches_cpu_short_ring_wl_tokens(
    product_smiles: str, reaction_smarts: str
) -> None:
    skip_if_no_metal_toolchain()
    try:
        import vesma.metal_retro  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"Metal backend unavailable: {exc}")

    product = cmol_from_smiles(product_smiles)[0]
    reaction = parse_reaction_smarts(reaction_smarts)
    assignments = _cpu_assignments(product, reaction)

    expected = _cpu_outcomes(product, reaction, assignments)
    observed = _metal_outcomes(product, reaction, assignments)
    assert observed == expected


def test_metal_retro_profile_phases_preserve_full_output() -> None:
    skip_if_no_metal_toolchain()
    from vesma.metal_retro import MetalRetroConstructor, reaction_data_from_pattern

    product = cmol_from_smiles("CC")[0]
    reaction = parse_reaction_smarts("[C:1][C:2]>>[C:1].[C:2]")
    assignments = _cpu_assignments(product, reaction)
    padded = np.full((len(assignments), 64), -1, dtype=np.int32)
    padded[:, : assignments.shape[1]] = assignments
    constructor = MetalRetroConstructor()
    args = (
        [gmol_from_cmol(product)],
        [reaction_data_from_pattern(reaction)],
        padded,
        [(0, 0)] * len(assignments),
    )

    baseline = constructor.construct_batch(*args)
    baseline_tokens = [
        frozenset(
            (*_token(baseline, row.output_base + i),)
            for i in range(row.component_count)
        )
        for row in baseline.status[: len(assignments)]
    ]
    for mode in (1, 2, 5, 4, 3, 6, 7, 8, 9):
        produced = constructor.construct_batch(*args, profile_mode=mode)
        for index, row in enumerate(produced.status[: len(assignments)]):
            assert row.profile_stage == (3 if mode >= 6 else mode)
            assert row.profile_edited_atoms == 2
            if mode in (2, 3, 4, 5, 6, 7, 8, 9):
                assert row.profile_kept_components == 2
            if mode in (3, 6, 7, 8, 9):
                assert row.profile_perceived_atoms == 2
                assert row.component_count == baseline.status[index].component_count
            if mode == 3:
                observed = frozenset(
                    (*_token(produced, row.output_base + i),)
                    for i in range(row.component_count)
                )
                assert observed == baseline_tokens[index]


def test_metal_retro_overflow_keeps_heaviest_and_flags() -> None:
    """An assignment that splits into more than MAX_COMPONENTS_PER_ASSIGNMENT pieces is
    truncated to the heaviest components, clamps component_count to the cap, and sets
    the overflow flag instead of crashing."""
    skip_if_no_metal_toolchain()
    from vesma.metal_retro import (
        MAX_COMPONENTS_PER_ASSIGNMENT,
        MetalRetroConstructor,
        reaction_data_from_pattern,
    )

    n_atoms = MAX_COMPONENTS_PER_ASSIGNMENT + 2  # forces overflow by 2 components
    chain = "C" * n_atoms  # an n-carbon alkane chain
    lhs = "".join(f"[C:{i + 1}]" for i in range(n_atoms))
    rhs = ".".join(f"[C:{i + 1}]" for i in range(n_atoms))  # break every backbone bond
    product = cmol_from_smiles(chain)[0]
    reaction = parse_reaction_smarts(f"{lhs}>>{rhs}")
    assignments = _cpu_assignments(product, reaction)
    assert assignments.shape[0] >= 1

    a64 = np.full((assignments.shape[0], 64), -1, dtype=np.int32)
    a64[:, : assignments.shape[1]] = assignments
    rdata = reaction_data_from_pattern(reaction)
    constructor = MetalRetroConstructor()
    with pytest.warns(UserWarning, match="more than"):
        produced = constructor.construct_batch(
            [gmol_from_cmol(product)], [rdata], a64, [(0, 0)] * assignments.shape[0]
        )

    assert constructor.last_overflow_count >= 1
    for i in range(assignments.shape[0]):
        # Splitting an n-carbon chain into singletons yields n > cap components.
        assert produced.status[i].overflow != 0
        assert produced.status[i].component_count == MAX_COMPONENTS_PER_ASSIGNMENT

    from vesma.metal_reaction_matcher import _precursor_batch

    batch, hits = _precursor_batch(
        produced, np.zeros((len(produced.status), 2), dtype=np.int64)
    )
    assert len(batch) == len(hits) == 0


def test_metal_retro_drops_oversize_precursor_component() -> None:
    skip_if_no_metal_toolchain()
    from vesma.metal_retro import (
        MetalRetroConstructor,
        reaction_data_from_pattern,
    )

    product = cmol_from_smiles("C")[0]
    large_component = "[C:1]" + "C" * 64
    reaction = parse_reaction_smarts(f"[C:1]>>{large_component}.O")
    assignments = _cpu_assignments(product, reaction)
    assert assignments.shape[0] == 1

    a64 = np.full((1, 64), -1, dtype=np.int32)
    a64[:, : assignments.shape[1]] = assignments
    rdata = reaction_data_from_pattern(reaction)
    constructor = MetalRetroConstructor()

    with pytest.warns(UserWarning, match="over 64 atoms"):
        produced = constructor.construct_batch(
            [gmol_from_cmol(product)], [rdata], a64, [(0, 0)]
        )

    assert constructor.last_atom_overflow_component_count == 1
    assert constructor.last_atom_overflow_count == 1
    assert produced.status[0].atom_overflow == 1
    assert produced.status[0].component_count == 1
    base = produced.status[0].output_base
    oxygen = short_ring_wl_molecule_id(cmol_from_smiles("O")[0])
    assert _token(produced, base) == oxygen

    from vesma.metal_reaction_matcher import _precursor_batch

    batch, hits = _precursor_batch(produced, np.zeros((1, 2), dtype=np.int64))
    assert len(batch) == len(hits) == 0


@pytest.mark.parametrize(
    "smiles,aromatic_nitrogen",
    [
        ("N1c2cccc(c2)-c2ccccc21", False),  # amine bridging a biaryl's meta positions
        ("N1c2ccc(cc2)-c2ccccc21", False),  # one atom longer, para
        ("N1c2ccccc2-c2ccccc21", True),  # carbazole, a genuine five-ring closure
    ],
)
def test_metal_perception_agrees_with_the_host_on_a_bridged_ring_system(
    smiles: str, aromatic_nitrogen: bool
) -> None:
    """The device's atom aromaticity word, compared against the host's, atom by atom.

    ``test_metal_retro_matches_cpu_short_ring_wl_tokens`` cannot see this: the WL seed reads
    the total hydrogen count and the bond classes, never the aromaticity word, so an arm that
    disagrees about a flag still produces the same token. It diverged anyway. Before the atom
    mask was derived from the accepted circuit's edges, the host refused this molecule
    outright -- an aromatic nitrogen with two single bonds is not a molecule -- while the
    device wrote the flag into word 4 and then read it back two ways in the same loop, taking
    the aliphatic configuration from the bonds for valence and the aromatic branch from the
    word for the hydrogen split. That flag is carried into the *next* edit's hydrogen law, so
    it does not stay cosmetic.
    """
    skip_if_no_metal_toolchain()

    from vesma.gmol_atoms import AROMATIC
    from vesma.metal_reaction_matcher import MetalReactionMatcher, metal_stack_mols
    from vesma.reaction import reaction_patterns_from_smarts

    aromaticity_word, words_per_atom = 4, 14

    molecule = cmol_from_smiles(smiles)[0]
    nitrogen = int(np.flatnonzero(molecule.Z == 7)[0])
    assert bool(molecule.A[nitrogen] == AROMATIC) is aromatic_nitrogen

    # An identity edit on the nitrogen keeps the graph whole, so the device re-perceives the
    # whole of it and its answer is comparable to the host's for every atom.
    identity = "[n:1]>>[n:1]" if aromatic_nitrogen else "[N:1]>>[N:1]"
    matcher = MetalReactionMatcher(
        reaction_patterns_from_smarts((identity,), cache=True)
    )
    batch, hits = matcher.match(metal_stack_mols([molecule]))
    assert len(hits) >= 1, "the identity rule should match the nitrogen"

    device_words = batch.atoms[0].reshape(-1, words_per_atom)
    device_a = device_words[: molecule.atom_count, aromaticity_word]
    assert np.array_equal(device_a.astype(np.uint8), molecule.A)


# The corpus CUDA already keys a host-built batch over in `test_cuda_retro.py`: size boundaries
# either side of the 64-atom bucket, the ring sizes the short-ring identity cuts at, two charged
# aromatics, and a multi-component molecule.
WL_BATCH_CASES = (
    "C",
    "CCO",
    "C1CC1",
    "C1CCCCCCC1",
    "c1ccccc1",
    "c1cc[nH]c1",
    "C" * 30,
    "C" * 46,
    "C" * 64,
    "[CH-]1C=CC=C1",
    "c1ccc[cH+]cc1",
    "O=C(CN1CCCC1)Nc1ccc(Br)cc1",
    "CC.CCC.c1ccccc1",
)


@pytest.mark.parametrize("smiles", WL_BATCH_CASES)
def test_the_device_keys_a_host_built_batch_as_the_cpu_codec_does(smiles: str) -> None:
    """The keyer over a batch, which is what makes the arm's `key` block total.

    The kernel this replaced, `wl_hash_precursors`, wrote `hash_lo`/`hash_hi` in place on a
    `GpuMol`, so it could only key what `retro_construct` had built; a frontier packed from CMols
    had no such record and could not be keyed at all. `wl_hash_molecules` takes the batch layout instead -- CUDA's signature, and CUDA proves
    the shape works by keying a host-built batch bit-exact against the CPU codec
    (`tests/vesma/test_cuda_retro.py`).

    Asserted against `short_ring_wl_molecule_id` rather than against a recorded token, for the
    reason the parity files give: a golden file records what the device did on the day it was
    written, and comparing to the model records whether the device still agrees with it.
    """
    from vesma import cmol_from_smiles
    from vesma.keyer import short_ring_wl_molecule_id
    from vesma.metal_reaction_matcher import metal_stack_mols
    from vesma.metal_retro import metal_wl_hash

    molecule, _info = cmol_from_smiles(smiles)
    batch = metal_stack_mols([molecule])
    keyed = metal_wl_hash(batch.atoms, batch.bonds, batch.atom_counts)
    assert (int(keyed.hash_lo[0]), int(keyed.hash_hi[0])) == short_ring_wl_molecule_id(
        molecule
    )


def test_the_token_does_not_depend_on_bond_topology_annotation() -> None:
    """One keyer can serve every batch here only because this holds.

    A host-packed molecule carries CPU-perceived ring topology on its bond bytes; a device-built
    precursor carries the device's own; a retro product has the annotation stripped. If the WL token
    read any of that, a single keyer would need to know which kind of batch it had.

    It does not: `identity_bond_class` masks the bond to its class and re-derives short-ring
    membership from the adjacency itself. Pinned here because the rest of the design rests on it --
    `metal_wl_hash` is handed `batch.bonds` with no question about where they came from.
    """
    import numpy as np

    from vesma import cmol_from_smiles
    from vesma.cmol_builder import gmol_from_cmol
    from vesma.metal_matcher import _pack_atom_masks, _pack_bond_matrix
    from vesma.metal_retro import metal_wl_hash

    for smiles in ("C1CCCCC1", "c1ccc2ccccc2c1", "C1CC2CCC1CC2", "O=C1CCC(=O)N1"):
        graph = gmol_from_cmol(cmol_from_smiles(smiles)[0])
        atoms = np.stack([_pack_atom_masks(graph, 64)])
        counts = np.array([graph.atom_count], dtype=np.uint32)
        annotated = np.stack([_pack_bond_matrix(graph, 64, annotate_topology=True)])
        plain = np.stack([_pack_bond_matrix(graph, 64, annotate_topology=False)])
        # The inputs really do differ, or the assertion below would be vacuous.
        assert not np.array_equal(annotated, plain), smiles

        with_topology = metal_wl_hash(atoms, annotated, counts)
        without = metal_wl_hash(atoms, plain, counts)
        assert with_topology.hash_lo.tolist() == without.hash_lo.tolist(), smiles
        assert with_topology.hash_hi.tolist() == without.hash_hi.tolist(), smiles
