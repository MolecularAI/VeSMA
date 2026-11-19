from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from vesma import cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.bonds import (
    BOND_KIND_MASK,
    BOND_TOPOLOGY_ANY,
    BOND_TOPOLOGY_MASK,
)
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.cuda_matcher import (
    CudaFoldedPrefilteredMatcher,
    CudaMatcher,
    CudaReactionMatcher,
    cuda_short_ring_wl_key,
    cuda_stack_mols,
)
from vesma.identity import short_ring_wl_molecule_id
from vesma.reaction import parse_reaction_smarts
from vesma.rejection import RejectionReason
from vesma.rings import annotate_bond_topology, ring_info_from_bonds


def _skip_if_no_cuda() -> None:
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CuPy cannot access CUDA on this machine.")


def pattern(smarts: str):
    return gmol_from_smarts(smarts)[0]


def molecule(smiles: str):
    return gmol_from_cmol(cmol_from_smiles(smiles)[0])


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_matcher_finds_assignments() -> None:
    _skip_if_no_cuda()

    matcher = CudaMatcher()
    result = matcher.find_matches(pattern("CC"), molecule("CCC"), max_assignments=8)

    assert result.status == "match"
    assert {tuple(row) for row in result.assignments.tolist()} == {
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
    }


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_matcher_has_match_many() -> None:
    _skip_if_no_cuda()

    matcher = CudaMatcher()
    observed = matcher.has_match_many(
        [pattern("CO"), pattern("CN"), pattern("CC")],
        [molecule("CCO"), molecule("CCC"), molecule("N")],
    )

    expected = np.array(
        [[True, False, False], [False, False, False], [True, True, False]]
    )
    assert np.array_equal(observed, expected)


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_folded_prefiltered_matcher_matches_raw_matcher() -> None:
    _skip_if_no_cuda()

    patterns = [pattern("CO"), pattern("CN"), pattern("CC")]
    molecules = [molecule("CCO"), molecule("CCC"), molecule("N")]

    raw = CudaMatcher().has_match_many(patterns, molecules)
    folded = CudaFoldedPrefilteredMatcher().has_match_many(patterns, molecules)

    assert np.array_equal(folded, raw)


# Ring shapes a bounded small-ring detector gets wrong, plus the shapes a naive one does.
# Kept in step with RING_TOPOLOGY_SMILES in test_metal_reaction_matcher.py.
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


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize("smiles", RING_TOPOLOGY_SMILES)
def test_cuda_perceives_ring_topology_like_the_cpu(smiles: str) -> None:
    """Device ring-or-chain topology equals ``rings.annotate_bond_topology``, per edge.

    A bond is a ring bond iff it is not a bridge, so this is exact for any ring size --
    unlike the 3-8 detector the WL identity token uses, which these macrocycles defeat.
    """
    _skip_if_no_cuda()

    product = cmol_from_smiles(smiles)[0]
    neutral = parse_reaction_smarts("[*:1]>>[*:1]")
    generated, _ = CudaReactionMatcher([neutral]).match(cuda_stack_mols([product]))
    assert generated.count > 0

    atom_count = product.Z.size
    reference = gmol_from_cmol(product).bonds
    expected = annotate_bond_topology(reference) & np.uint8(BOND_TOPOLOGY_MASK)
    ring_info = ring_info_from_bonds(reference)
    bonds = generated.bonds_device.get().reshape(-1, 64, 64)
    atoms = generated.atoms_device.get().reshape(-1, 64, 14)
    for index in range(bonds.shape[0]):
        device = bonds[index][:atom_count, :atom_count]
        assert np.array_equal(device & np.uint8(BOND_TOPOLOGY_MASK), expected)
        # x falls out of the same non-bridge set, so it is exact wherever topology is.
        assert atoms[index][:atom_count, 13].tolist() == [
            1 << int(value) for value in ring_info.ring_connectivity
        ]


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_generated_frontier_carries_exact_bond_topology() -> None:
    """A GPU-built precursor answers `@`/`!@` exactly, as a host-packed one does.

    perceive_component_molecules writes ring-or-chain topology onto each bond byte, so
    the answer no longer depends on which retro layer the molecule came from.
    Methylcyclohexane is the case that separates the two: six ring bonds and one chain
    bond to the methyl, where an unknown-topology precursor accepted every C-C bond for
    both queries. Mirrors the Metal twin in test_metal_reaction_matcher.py.
    """
    _skip_if_no_cuda()

    product = cmol_from_smiles("CC1CCCCC1")[0]
    neutral = parse_reaction_smarts("[C:1]-[C:2]>>[C:1]-[C:2]")
    topology_rules = [
        parse_reaction_smarts("[C:1]@[C:2]>>[C:1].[C:2]"),
        parse_reaction_smarts("[C:1]!@[C:2]>>[C:1].[C:2]"),
    ]

    host = cuda_stack_mols([product])
    _, host_hits = CudaReactionMatcher(topology_rules).match(host)
    assert set(host_hits[:, 1].tolist()) == {0, 1}

    generated, _ = CudaReactionMatcher([neutral]).match(host)
    bonds = generated.bonds_device.get()
    classes = bonds & np.uint8(BOND_KIND_MASK)
    present = classes != 0
    topology = bonds & np.uint8(BOND_TOPOLOGY_MASK)
    assert np.all(topology[present] != BOND_TOPOLOGY_ANY)
    assert np.all(bonds[~present] == 0)

    # A neutral template keeps the product's atom order, so the device topology compares
    # position by position against the CPU annotation of the same molecule.
    atom_count = product.Z.size
    expected = annotate_bond_topology(gmol_from_cmol(product).bonds) & np.uint8(
        BOND_TOPOLOGY_MASK
    )
    for index in range(bonds.shape[0]):
        device = bonds[index].reshape(64, 64)[:atom_count, :atom_count]
        assert np.array_equal(device & np.uint8(BOND_TOPOLOGY_MASK), expected)

    _, generated_hits = CudaReactionMatcher(topology_rules).match(generated)
    _, cpu_hits = CpuReactionMatcher(topology_rules).match([product])
    assert set(generated_hits[:, 1].tolist()) == set(cpu_hits[:, 1].tolist()) == {0, 1}


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_generated_frontier_refuses_chain_query_on_a_ring_bond() -> None:
    """The reported divergence: `-!@` found nothing on CPU and every ring bond on CUDA.

    Layer one already agreed, because a host-packed frontier carries exact topology.
    Only the second layer, whose molecules the device built, disagreed.
    """
    _skip_if_no_cuda()

    product = cmol_from_smiles("CC1CCCCC1")[0]
    rules = [parse_reaction_smarts("[C:1]-!@[C:2]>>[C:1].[C:2]")]

    cuda = CudaReactionMatcher(rules)
    batch, first_hits = cuda.match(cuda_stack_mols([product]))
    _, second_hits = cuda.match(batch)

    cpu = CpuReactionMatcher(rules)
    cpu_precursors, cpu_first_hits = cpu.match([product])
    _, cpu_second_hits = cpu.match(list(cpu_precursors))

    assert len(first_hits) == len(cpu_first_hits)
    # Cyclohexane and methane: no chain carbon-carbon bond is left to disconnect.
    assert len(second_hits) == len(cpu_second_hits) == 0


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_ring_triple_identity_is_stable_across_generated_layer() -> None:
    _skip_if_no_cuda()

    product = cmol_from_smiles("C1CCCCCC#C1")[0]
    expected = short_ring_wl_molecule_id(product)
    neutral = parse_reaction_smarts("[C:1]#[C:2]>>[C:1]#[C:2]")
    matcher = CudaReactionMatcher([neutral])

    first, _ = matcher.match(cuda_stack_mols([product]))
    assert set(cuda_short_ring_wl_key(first)) == {expected}

    second, _ = matcher.match(first)
    assert set(cuda_short_ring_wl_key(second)) == {expected}


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_rejection_rolls_back_the_whole_application_and_is_counted() -> None:
    _skip_if_no_cuda()

    product = cmol_from_smiles("C")[0]
    rule = parse_reaction_smarts("[C:1]>>[C:1].[NH2]")
    matcher = CudaReactionMatcher([rule])

    precursors, hits = matcher.match([product])
    diagnostics = matcher.last_diagnostics

    assert len(precursors) == 0
    assert hits.shape == (0, 4)
    assert diagnostics is not None
    assert diagnostics.rejections[RejectionReason.CONFIGURATION_INCOMPATIBLE] == 1
    assert sum(diagnostics.rejections) == 1


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_refuses_an_application_that_orphans_a_component() -> None:
    """CUDA refuses the same incomplete disconnection the reference refuses.

    This rule hydrolyses an ester but names only the acid on its RHS, so applying it leaves
    the ethyl fragment in no precursor the template describes. Returning the acid alone would
    present "benzoic acid" as a complete answer to a step that also needs ethanol -- worst
    exactly where it is least visible, in a multi-layer search where the missing piece is the
    molecule that failed to reach stock -- so the whole application is refused rather than
    silently trimmed. `tests/vesma/test_reaction.py` pins the policy on this very molecule and
    rule; this is the CUDA kernel held to it, since `cuda_retro.py` counts `orphaned_components`
    itself and must reach the same verdict as the host reference, not merely a similar one.

    It used to assert the opposite -- one precursor, the acid, with the fragment dropped and
    counted -- which is what the engine did before the refusal landed. The policy commit updated
    the reference and the CPU path and could not run this one, because asserting it needs a CUDA
    device. So it sat wrong until there was a box to run it on, which is the argument for keeping
    a GPU gate in reach rather than in principle.
    """
    _skip_if_no_cuda()

    product = cmol_from_smiles("CCOC(=O)c1ccccc1")[0]
    rule = parse_reaction_smarts(
        "[C:4]-[O;H0;D2;+0:3]-[C:2]=[O;D1;H0:1]>>[O;D1;H0:1]=[C:2]-[OH;D1;+0:3]"
    )
    matcher = CudaReactionMatcher([rule])

    precursors, hits = matcher.match([product])
    diagnostics = matcher.last_diagnostics

    # Refused whole: no precursor survives, so there is no hit to point at one.
    assert len(precursors) == 0
    assert hits.shape == (0, 4)
    assert diagnostics is not None
    # The match itself succeeded -- this is a refusal after a hit, not a failure to match.
    assert diagnostics.status_yes == 1
    assert diagnostics.undescribed_components == 1


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_keeps_an_application_whose_deletion_orphans_nothing() -> None:
    """The shape the corpus actually has, which the refusal must not touch.

    Same rule, one carbon less: the deleted atom is pinned to a single neighbour, so nothing is
    left attached to nothing and the disconnection is complete. The acid comes back with its WL
    key, and no component is undescribed. Without this half the test above would be satisfied by
    a backend that refused everything.
    """
    _skip_if_no_cuda()

    product = cmol_from_smiles("COC(=O)c1ccccc1")[0]
    rule = parse_reaction_smarts(
        "[C:4]-[O;H0;D2;+0:3]-[C:2]=[O;D1;H0:1]>>[O;D1;H0:1]=[C:2]-[OH;D1;+0:3]"
    )
    matcher = CudaReactionMatcher([rule])

    precursors, hits = matcher.match([product])
    diagnostics = matcher.last_diagnostics

    assert len(precursors) == 1
    assert hits.shape == (1, 4)
    assert set(cuda_short_ring_wl_key(precursors)) == {
        short_ring_wl_molecule_id(cmol_from_smiles("O=C(O)c1ccccc1")[0])
    }
    assert diagnostics is not None
    assert diagnostics.undescribed_components == 0


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize("frontier_mode", ["device", "host"])
def test_the_generic_fold_reproduces_cudas_own_dedup(frontier_mode: str) -> None:
    """`keyed_match_folding` and `match_keyed` collapse identity to the same answer.

    CUDA folds identity in itself, against a running `key_to_physical` map, retaining one physical
    molecule per distinct WL key. `matcher.keyed_match_folding` over `match_batches` does the same
    collapse from outside the matcher, composed by the caller and portable across backends. They
    have to agree exactly, and this says so on both frontier modes: same distinct keys, same
    (molecule, rule, key) multiset, one row per distinct key, no row unreferenced.

    Pinned because the two are measured against each other, and because the measurement decided
    *not* to retire the bespoke one: at 100 molecules the generic fold matched it to the row and to
    within 0.3% on time while holding 3.2x the device memory in device mode, since `match_keyed`
    retains into a preallocated frontier where the fold concatenates parts. The two therefore
    coexist, and coexisting paths drift unless something compares them.
    """
    _skip_if_no_cuda()
    from vesma.cuda_matcher import cuda_concat_mols
    from vesma.matcher import keyed_match_folding

    rules = [
        parse_reaction_smarts(smarts)
        for smarts in (
            "[c:1][c:2]>>[c:1].[c:2]",
            "[C:1][C:2]>>[C:1].[C:2]",
            "[C:1][O:2]>>[C:1].[O:2]",
        )
    ]
    mols = [
        cmol_from_smiles(smiles)[0]
        for smiles in ("c1ccccc1C", "CCO", "CC(=O)Oc1ccccc1", "Cc1ccc(N)cc1")
    ]
    matcher = CudaReactionMatcher(rules, frontier_mode=frontier_mode)

    bespoke_batch, bespoke_keys, bespoke_hits = matcher.match_keyed(mols)
    bespoke_keys = list(bespoke_keys)
    folded_batch, folded_keys, folded_hits = keyed_match_folding(
        lambda m: matcher.match_batches(m),
        cuda_short_ring_wl_key,
        cuda_concat_mols,
    )(mols)
    folded_keys = list(folded_keys)

    def provenance(hits, keys) -> Counter:
        """(molecule, rule, key): names no precursor row, so it survives either assembly."""
        return Counter((int(r[0]), int(r[1]), keys[int(r[3])]) for r in hits)

    assert set(folded_keys) == set(bespoke_keys)
    assert len(folded_keys) == len(set(folded_keys)) == len(folded_batch)
    assert len(bespoke_keys) == len(set(bespoke_keys))
    assert provenance(folded_hits, folded_keys) == provenance(
        bespoke_hits, bespoke_keys
    )
    assert set(folded_hits[:, 3].tolist()) == set(range(len(folded_batch)))
