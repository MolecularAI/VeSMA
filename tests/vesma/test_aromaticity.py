from __future__ import annotations

import numpy as np
import pytest

from vesma import (
    PiDonor,
    bond_delocalization,
    cmol_from_smiles,
    simple_cycles,
)
from vesma.adjacency import bond_presence
from vesma.aromaticity import pi_donor
from vesma.bonds import BondKind
from vesma.gmol_atoms import ALIPHATIC, AROMATIC
from vesma.identity import short_ring_wl_molecule_id
from vesma.parsing.errors import SmartsError


def molecule(smiles: str):
    return cmol_from_smiles(smiles)[0]


@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccccc1",
        "c1ccncc1",
        "c1cc[nH]c1",
        "c1ccoc1",
        "c1ccsc1",
        "c1ccc2ccccc2c1",
        "c1ccc2[nH]ccc2c1",
        "[CH-]1C=CC=C1",
        "O=C1C=CC=CN1",
        "C1=CC2=CC=CC=CC2=C1",  # azulene (5-7 fused), needs 7-ring perception
        "[cH+]1cccccc1",  # tropylium (7-ring aromatic cation)
        "[o+]1ccccc1",  # pyrylium (O+ localizes as N)
        "[s+]1ccccc1",  # thiopyrylium (S+ localizes as P)
        "O=c1cccccc1",  # tropone
        "[CH+]1C=C1",  # cyclopropenyl cation (2pi, Hueckel n=0)
    ],
)
def test_simple_aromaticity_accepts_supported_ring_systems(smiles: str) -> None:
    graph = molecule(smiles)

    assert np.any(graph.A == AROMATIC)
    assert np.any(bond_delocalization(graph.bonds))


@pytest.mark.parametrize(
    "smiles",
    [
        "C=C1C=CC=C1",
        "O=C1C=CC=C1",
        "C1=CC=C1",
        "C1CC1",  # cyclopropane (0 pi electrons)
        "C1=CC=CC=CC1",  # cycloheptatriene (neutral 7-ring, not aromatic)
        "C1=CC=CC=CC=C1",  # cyclooctatetraene (8 pi electrons, anti-aromatic)
    ],
)
def test_simple_aromaticity_rejects_nonaromatic_cycles(smiles: str) -> None:
    graph = molecule(smiles)

    assert np.all(graph.A == ALIPHATIC)
    assert not np.any(bond_delocalization(graph.bonds))


@pytest.mark.parametrize(
    ("acceptor_z", "expected"),
    [
        # Huckel bookkeeping is relative to the circuit: a sole exocyclic pi bond
        # engages the atom's p orbital wherever it points, so the partner element does
        # not change the contribution.
        pytest.param(6, PiDonor.ZERO, id="carbon"),
        pytest.param(7, PiDonor.ZERO, id="nitrogen"),
        pytest.param(8, PiDonor.ZERO, id="oxygen"),
    ],
)
def test_pi_donor_handles_exocyclic_double_bond(
    acceptor_z: int,
    expected: PiDonor,
) -> None:
    cyclic_edges = np.zeros((4, 4), dtype=bool)
    cyclic_edges[0, 1] = cyclic_edges[1, 0] = True
    cyclic_edges[0, 2] = cyclic_edges[2, 0] = True

    bond_orders = np.zeros((4, 4), dtype=np.uint8)
    # The ring bonds carry their orders too: selecting a configuration reads the whole
    # local state, where the old degree-based rule looked only at the exocyclic bond.
    bond_orders[0, 1] = bond_orders[1, 0] = BondKind.SINGLE
    bond_orders[0, 2] = bond_orders[2, 0] = BondKind.SINGLE
    bond_orders[0, 3] = bond_orders[3, 0] = BondKind.DOUBLE

    assert (
        pi_donor(
            atom=0,
            cyclic_edges=cyclic_edges,
            Z=np.array([6, 6, 6, acceptor_z], dtype=np.int16),
            C=np.zeros(4, dtype=np.int8),
            total_h=np.zeros(4, dtype=np.int16),
            bond_orders=bond_orders,
            heavy_degrees=np.array([3, 1, 1, 1], dtype=np.int16),
            delocalized_atoms=np.zeros(4, dtype=bool),
        )
        == expected
    )


def test_bond_between_two_aromatic_rings_is_not_aromatic() -> None:
    graph = molecule("c1ccccc1-c2ccccc2")
    bridge = np.argwhere(
        np.triu(
            (graph.A[:, None] == AROMATIC)
            & (graph.A[None, :] == AROMATIC)
            & ~bond_delocalization(graph.bonds)
            & bond_presence(graph.bonds),
            k=1,
        )
    )

    assert bridge.shape == (1, 2)


def test_fused_thiazole_ring_with_two_lone_pair_donors_is_aromatic() -> None:
    """A documented, accepted divergence from RDKit. The simple electron-count rule
    marks this 2-imino-thiazole ring (S and a bridgehead N each donate a lone pair,
    the imine carbon donates 0, the C=C donates 2) aromatic with 6 pi electrons;
    RDKit keeps it aliphatic via its hybridisation/ring-perception heuristics. We
    prefer the simpler, consistent Hueckel rule over a special case to suppress this
    single fused-bridgehead pattern, which would also (wrongly) reject uracil and
    tetrazolone rings that RDKit aromatises. The phenyl ring is aromatic either way."""
    graph = molecule("S1C2=N[C@H](c3ccccc3)CN2C=C1")

    assert tuple(np.flatnonzero(graph.A == AROMATIC)) == (
        0,
        1,
        4,
        5,
        6,
        7,
        8,
        9,
        11,
        12,
        13,
    )


def test_fused_amidine_ring_over_aromatized_vs_rdkit() -> None:
    """Documented, accepted divergence (the deferred case from the bond-type-query
    story). The amidine ring N1-C=N fused onto naphthalene forms a 14-pi fused
    system under the delocalized count, so the amine donates a lone pair and the
    whole system reads aromatic; RDKit keeps the amidine ring aliphatic. The amine
    here is structurally identical to a uracil/tetrazolone ring nitrogen (two sp2
    ring neighbours plus one acyclic sp3), which RDKit *does* aromatize, so no simple
    consistent rule separates them. We accept the over-aromatization rather than add
    a special case; it is a triage false-positive, which the model tolerates."""
    graph = molecule("COC(=O)CN1C=Nc2cccc3cccc1c23")

    assert tuple(np.flatnonzero(graph.A == AROMATIC)) == (
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
    )


@pytest.mark.parametrize(
    "smiles",
    [
        "Fc1ccccc1-n1nnnc1-c1ccccc1I",
        "Clc1cnc2nc(Cl)c(Cl)nc2c1",
    ],
)
def test_aromatic_heteroatoms_do_not_gain_implicit_hydrogens(
    smiles: str,
) -> None:
    graph = molecule(smiles)
    nitrogens = graph.Z == 7

    assert np.all(graph.A[nitrogens] == AROMATIC)
    assert np.all(graph.H[nitrogens] == 0)
    assert np.all(graph.h[nitrogens] == 0)


def test_an_authored_aromatic_hydrogen_count_is_not_overwritten() -> None:
    """Authored hydrogen is molecular state, so `[cH0]` is a different claim from `c`.

    Perception used to replace an aromatic carbon's count with `3 - degree` outright,
    which made `[cH0]1ccccc1`, `[cH2]1ccccc1` and `c1ccccc1` all come back as benzene.
    A written count is now answered rather than rewritten: the sigma frame it describes
    is not one an aromatic carbon can occupy, no configuration contributes to the circuit,
    and the ring is rejected instead of quietly becoming a different molecule.
    """
    from vesma.bonds import BondKind
    from vesma.cmol_builder import cmol_from_seed_data
    from vesma.configurations import ConfigurationError

    n = 6
    bonds = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        j = (i + 1) % n
        bonds[i, j] = bonds[j, i] = np.uint8(BondKind.DELOCALIZED)

    def benzene_with_authored_h(count: int):
        return cmol_from_seed_data(
            Z=np.full(n, 6, dtype=np.int16),
            C=np.zeros(n, dtype=np.int8),
            explicit_h=np.full(n, count, dtype=np.int16),
            implicit_h=np.zeros(n, dtype=np.int16),
            atom_map=np.zeros(n, dtype=np.int32),
            bonds=bonds,
        )

    assert benzene_with_authored_h(1).H.tolist() == [1] * n
    assert all(int(a) == AROMATIC for a in benzene_with_authored_h(1).A)
    for wrong in (0, 2):
        with pytest.raises(ConfigurationError):
            benzene_with_authored_h(wrong)


def test_an_unwritten_aromatic_hydrogen_is_derived_from_the_sigma_frame() -> None:
    from vesma.bonds import BondKind
    from vesma.cmol_builder import cmol_from_seed_data

    n = 6
    bonds = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        j = (i + 1) % n
        bonds[i, j] = bonds[j, i] = np.uint8(BondKind.DELOCALIZED)

    benzene = cmol_from_seed_data(
        Z=np.full(n, 6, dtype=np.int16),
        C=np.zeros(n, dtype=np.int8),
        explicit_h=np.zeros(n, dtype=np.int16),
        implicit_h=np.full(n, -1, dtype=np.int16),
        atom_map=np.zeros(n, dtype=np.int32),
        bonds=bonds,
    )

    assert all(int(a) == AROMATIC for a in benzene.A)
    assert benzene.H.tolist() == [1] * n


def test_cycle_enumeration_is_canonical_for_fused_rings() -> None:
    graph = molecule("c1ccc2ccccc2c1")
    cycles = simple_cycles(bond_presence(graph.bonds), min_size=5, max_size=8)

    assert np.count_nonzero(np.triu(bond_delocalization(graph.bonds), k=1)) == 11
    assert [cycle.size for cycle in cycles] == [6, 6]
    for cycle in cycles:
        assert int(cycle[0]) == int(cycle.min())
        assert int(cycle[1]) < int(cycle[-1])


def test_isoelectronic_valence_is_nonlinear_in_charge() -> None:
    from vesma.aromaticity import isoelectronic_valence

    # An anion drops one valence (O- behaves like F); N+ gains one (like C); but the
    # tropylium C+ also drops (like B) -- valence peaks at carbon, so a flat base+charge
    # would wrongly give C+ valence 5.
    assert isoelectronic_valence(8, -1, fallback=2) == 1  # O-
    assert isoelectronic_valence(16, -1, fallback=2) == 1  # S-
    assert isoelectronic_valence(7, -1, fallback=3) == 2  # N- (azide terminus)
    assert isoelectronic_valence(7, 1, fallback=3) == 4  # N+ (nitro / ammonium)
    assert isoelectronic_valence(6, 1, fallback=4) == 3  # C+ (tropylium), not 5


def test_pi_donor_and_hydrogen_law_agree_on_anion_valence() -> None:
    from vesma.aromaticity import isoelectronic_valence
    from vesma.hydrogens import hydrogens_for_written_state

    # Drift guard: the aromatic (pi_donor) and the hydrogen paths go through the same
    # isoelectronic rule, so an anionic atom resolves to one standard valence on both --
    # a sigma frame filling that valence needs no hydrogen, one short needs exactly one.
    # This is the invariant the spurious-H-on-anions bug broke; it now holds because both
    # read the same configuration table rather than two parallel valence lists.
    for atomic_number, base in [(8, 2), (16, 2), (7, 3)]:
        valence = isoelectronic_valence(atomic_number, -1, fallback=base)
        for bond_order_sum, expected in ((valence, 0), (valence - 1, 1)):
            assert (
                hydrogens_for_written_state(
                    z=atomic_number,
                    charge=-1,
                    sigma_degree=bond_order_sum,
                    bond_order_sum=bond_order_sum,
                    delocalized=False,
                )
                == expected
            )


def test_cycle_enumeration_handles_more_than_64_atoms() -> None:
    # The CPU enumerator masks are arbitrary-precision Python ints, so it has no 64-atom
    # limit. Large connected components occur in route replay (a heavily protected retro
    # intermediate before the connected-component split), and must be enumerated, not
    # rejected.
    adjacency = np.zeros((70, 70), dtype=bool)
    ring = [60, 61, 62, 63, 64, 65]  # a 6-ring whose atoms sit past index 64
    for left, right in zip(ring, ring[1:] + ring[:1], strict=True):
        adjacency[left, right] = adjacency[right, left] = True

    cycles = simple_cycles(adjacency, min_size=5, max_size=8)

    assert len(cycles) == 1
    assert sorted(int(atom) for atom in cycles[0]) == ring


def test_perceived_aromaticity_agrees_with_rdkit_on_stock_corpus() -> None:
    """Atom-level aromaticity agreement vs RDKit on PaRoutes stock. This is the
    guardrail for aromaticity-model changes: keep agreement at least at the
    measured baseline (atom-level >= 99.9%, molecule-level >= 99.8%). RDKit is a
    reference, not a target, so exact parity is not required, but a regression in
    this number means a model change quietly dropped or over-marked rings. The
    delocalized electron-count model agrees with RDKit on every molecule in the
    first 1000 of this corpus; the few divergences across the full file are
    fused-amine bridgeheads and boroxine rings (see the documented cases above)."""
    import gzip

    from tests.datadirs import external_data_root

    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    stock_path = (
        external_data_root() / "external" / "paroutes" / "processed" / "n1-stock.smi.gz"
    )

    molecules = 0
    atoms = 0
    atom_disagreements = 0
    molecule_disagreements = 0
    bond_molecule_disagreements = 0
    with gzip.open(stock_path, "rt") as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            parsed = Chem.MolFromSmiles(fields[0])
            if parsed is None:
                continue
            smiles = Chem.MolToSmiles(parsed, isomericSmiles=False)
            reference = Chem.MolFromSmiles(smiles)
            if reference is None:
                continue
            try:
                graph = cmol_from_smiles(smiles)[0]
            except Exception:
                continue
            if int(graph.Z.size) != reference.GetNumAtoms():
                continue
            rdkit_aromatic = [atom.GetIsAromatic() for atom in reference.GetAtoms()]
            vesma_aromatic = [bool(graph.A[i] == AROMATIC) for i in range(graph.Z.size)]
            disagree = sum(
                1 for a, b in zip(rdkit_aromatic, vesma_aromatic, strict=True) if a != b
            )
            # Bonds are counted separately, because a molecule can agree atom for atom
            # and still disagree on an edge -- and the edge is what a pattern matches on.
            # Four corpus molecules did exactly that until the circuit-boundary rule
            # landed, and this check, being atom-only, could not see any of them.
            bond_disagree = sum(
                1
                for bond in reference.GetBonds()
                if (
                    int(graph.bonds[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                    == int(BondKind.DELOCALIZED)
                )
                != bond.GetIsAromatic()
            )
            molecules += 1
            atoms += len(rdkit_aromatic)
            atom_disagreements += disagree
            molecule_disagreements += int(disagree > 0)
            bond_molecule_disagreements += int(bond_disagree > 0)
            if molecules >= 1000:
                break

    assert molecules >= 900
    atom_agreement = (atoms - atom_disagreements) / atoms
    molecule_agreement = (molecules - molecule_disagreements) / molecules
    bond_agreement = (molecules - bond_molecule_disagreements) / molecules
    assert atom_agreement >= 0.999, (atom_agreement, atom_disagreements, atoms)
    assert molecule_agreement >= 0.998, (molecule_agreement, molecule_disagreements)
    assert bond_agreement >= 0.998, (bond_agreement, bond_molecule_disagreements)
    # No molecule may agree on every atom and still disagree on a bond: that is the
    # shape the union-marking defect took, and it is the shape this test missed.
    assert bond_molecule_disagreements <= molecule_disagreements, (
        bond_molecule_disagreements,
        molecule_disagreements,
    )


@pytest.mark.parametrize(
    ("delocalized_spelling", "localized_spelling"),
    [
        ("c1ccccc1", "C1=CC=CC=C1"),
        ("c1ccncc1", "C1=CC=NC=C1"),
        ("O=c1cccc[nH]1", "O=C1C=CC=CN1"),
        ("c1c[nH]c(=O)[nH]c1=O", "O=C1NC(=O)C=CN1"),
    ],
)
def test_spelling_does_not_change_perception(
    delocalized_spelling: str, localized_spelling: str
) -> None:
    """Inside the automatic domain a molecule perceives the same however it is written.

    Contribution is decided relative to the candidate circuit rather than from the input
    notation, so a delocalized spelling and its localized form agree on atom
    aromaticity, on canonical bonds, and therefore on which patterns they match. Input
    notation is not chemical authority.
    """
    written_delocalized = molecule(delocalized_spelling)
    written_localized = molecule(localized_spelling)

    # The two spellings may enumerate atoms in different orders, so the invariant is
    # order-independent: the same molecule identity, and the same number of atoms
    # perceived delocalized.
    assert short_ring_wl_molecule_id(written_delocalized) == short_ring_wl_molecule_id(
        written_localized
    )
    assert int((written_delocalized.A == AROMATIC).sum()) == int(
        (written_localized.A == AROMATIC).sum()
    )


def test_an_exocyclic_pi_bond_is_counted_out_of_the_circuit() -> None:
    """Tropone reaches six and is delocalized; fulvene reaches four and is not."""
    assert AROMATIC in molecule("O=C1C=CC=CC=C1").A
    assert AROMATIC not in molecule("C=C1C=CC=C1").A


@pytest.mark.parametrize(
    "smiles, aromatic_atoms, role",
    [
        (
            "Cc1nnc2c3c(c(N4CCOCC4)nn12)C1CCC3C1",
            {1, 2, 3, 4, 5, 6, 7, 14, 15},
            "triazolo-pyridazine fused to a saturated bicycle",
        ),
        (
            "c1ccc2c(c1)CCCC2",
            {0, 1, 2, 3, 4, 5},
            "tetralin: benzene sharing a bond with a saturated ring",
        ),
        (
            "C1CC2=CC=CC=C2C1",
            {2, 3, 4, 5, 6, 7},
            "indane, written Kekule",
        ),
    ],
)
def test_a_saturated_ring_sharing_a_bond_does_not_break_the_aromatic_system(
    smiles: str, aromatic_atoms: set[int], role: str
) -> None:
    """A ring that cannot delocalize is not a candidate circuit, so it cannot be fused
    into one.

    Cycles used to be grouped into fused systems purely by shared edges, before anything
    asked whether their atoms could belong to a circuit at all. A saturated ring sharing
    a bond with an aromatic one was therefore pulled into the same system; the joint
    Huckel count failed on the sp3 atoms, and the per-cycle fallback then measured each
    aromatic ring alone -- where the shared fusion bond is exocyclic and its electron
    goes uncounted. The triazolo-pyridazine counts ten as one system and five as either
    ring by itself, so it was rejected outright and was the one molecule in PaRoutes
    that could not be read.
    """
    perceived = molecule(smiles)

    assert {
        index for index, value in enumerate(perceived.A) if int(value) == AROMATIC
    } == aromatic_atoms, role


@pytest.mark.parametrize(
    "smiles, aromatic, role",
    [
        ("c1ccccccccc1", True, "[10]annulene: 10 electrons, 4n+2 with n=2"),
        ("c1ccccccccccccc1", True, "[14]annulene: 14 electrons, n=3"),
        ("c1ccccc1", True, "benzene still comes through the automatic domain"),
    ],
)
def test_an_asserted_circuit_is_validated_at_any_size(
    smiles: str, aromatic: bool, role: str
) -> None:
    """A circuit the input asserted is a claim to check, not a candidate to discover.

    Automatic perception stops at the 3-8 domain because enumeration has to be bounded,
    but that bound is a property of the search and not of chemistry. An asserted circuit
    is read off by walking the delocalized edges, so its size costs a walk rather than an
    enumeration and a written ring is answered at whatever size it was written.
    """
    perceived = molecule(smiles)

    assert np.all((perceived.A == AROMATIC) == aromatic), role


def test_an_asserted_circuit_that_fails_huckel_is_still_rejected() -> None:
    """Being asserted makes a circuit checkable, not true. Cyclooctatetraene carries
    eight electrons, so the assertion is refused rather than honoured -- and refused
    outright rather than downgraded to single bonds, since resolving it would mean
    inventing bond orders the input never gave."""
    from vesma.aromaticity import DelocalizedSystemInvalidError

    with pytest.raises(DelocalizedSystemInvalidError):
        molecule("c1ccccccc1")


@pytest.mark.parametrize(
    "atoms, outcome, role",
    [
        (6, "aromatic", "benzene: inside the domain, perceived"),
        (8, "aliphatic", "cyclooctatetraene: inside the domain, examined and refused"),
        (10, "declined", "[10]annulene: outside the domain"),
        (14, "declined", "[14]annulene: outside the domain, and genuinely aromatic"),
        (18, "declined", "[18]annulene: outside the domain, and genuinely aromatic"),
    ],
)
def test_a_ring_outside_the_automatic_domain_asks_to_be_annotated(
    atoms: int, outcome: str, role: str
) -> None:
    """ "We did not look" is a different answer from "not aromatic".

    Perception enumerates cycles of 3 to 8 atoms because the search has to be bounded.
    Returning "aliphatic" for everything larger made the model spelling-dependent: a
    localized [14]annulene came back aliphatic while the same molecule written
    delocalized came back aromatic, and for [14] and [18] the localized answer is simply
    wrong. Declining keeps the one property the model is built on.

    Deciding this needs only that such a ring *exists*, never which one it is, and
    existence is a cyclomatic count over the atoms that could delocalize -- so it costs a
    pass, not an enumeration.
    """
    from vesma.aromaticity import RequiresDelocalizedAnnotation

    kekule = "C1" + "".join("=C" if i % 2 else "C" for i in range(1, atoms)) + "1"

    if outcome == "declined":
        with pytest.raises((RequiresDelocalizedAnnotation, SmartsError)):
            molecule(kekule)
        # Written delocalized, the same ring is a claim that can be checked.
        assert np.all(molecule("c1" + "c" * (atoms - 1) + "1").A == AROMATIC), role
        return

    perceived = molecule(kekule)
    expected = AROMATIC if outcome == "aromatic" else ALIPHATIC
    assert np.all(perceived.A == expected), role


@pytest.mark.parametrize(
    "smiles, role",
    [
        (
            "C1CCCCCCCCCCCCC1",
            "a saturated macrocycle has no atom that could delocalize",
        ),
        (
            "O=C1CCCCCCC=CCCCCO1",
            "a macrolide's alkene is an acyclic fragment once the "
            "sp3 carbons are excluded",
        ),
        ("c1ccc2ccccc2c1", "naphthalene is covered by enumerated cycles"),
        ("C1=CC2=CC=CC=CC2=C1", "azulene's 5 and 7 rings are both inside the domain"),
    ],
)
def test_ordinary_large_rings_are_not_refused(smiles: str, role: str) -> None:
    """The refusal is narrow by construction, and both filters are doing work.

    Restricting to atoms that could belong to a circuit is what keeps macrocycles quiet;
    dropping edges already on an enumerated cycle is what keeps small rings quiet whether
    they were accepted or refused.
    """
    assert molecule(smiles) is not None, role


def test_a_large_cycle_sharing_an_edge_with_a_small_one_is_not_missed() -> None:
    """The hole that deleting enumerated edges left open.

    A large eligible cycle fused to a small one passes through the shared bond. Removing
    that bond along with the small cycle broke the large one, so nothing was left to
    detect and the molecule came back silently aliphatic -- in exactly the fused
    arrangement the check exists to catch.

    Comparing cycle-space dimensions instead is exact: the eligible subgraph's rank counts
    every independent cycle, the examined cycles span a subspace, and a shortfall means
    one was never looked at. No edge is deleted, so nothing can be lost with it.
    """
    from vesma.aromaticity import RequiresDelocalizedAnnotation

    with pytest.raises((RequiresDelocalizedAnnotation, SmartsError)):
        molecule("C1=CC=CC2=CC=CC=CC=CC=C12")


def test_gf2_rank_counts_independent_cycles() -> None:
    """Addition over GF(2) is exclusive or, so elimination is pivots and xors."""
    from vesma.aromaticity import gf2_rank

    assert gf2_rank([]) == 0
    assert gf2_rank([0b011, 0b110]) == 2
    # The third vector is the sum of the first two, so it adds no dimension.
    assert gf2_rank([0b011, 0b110, 0b101]) == 2
    assert gf2_rank([0b011, 0b011]) == 1


def test_a_cycle_through_an_ineligible_atom_is_not_a_basis_vector() -> None:
    """The rank basis must contain cycles, not fragments of them.

    A ten-membered eligible ring fused to a triangle whose third atom is an sp3 CH2. The
    triangle is small enough to enumerate, but it leaves the eligible subgraph, and
    restricting it to that subgraph turns it into a single edge -- still a nonzero vector,
    so it raised the measured rank by one and exactly cancelled the unexamined ten-ring.
    The molecule came back silently aliphatic.

    Only cycles wholly inside the eligible subgraph are basis vectors now, and each
    accepted asserted circuit is kept separately rather than folded into a union, since
    an OR of overlapping circuits need not be a cycle-space vector at all.
    """
    from vesma.aromaticity import RequiresDelocalizedAnnotation

    with pytest.raises((RequiresDelocalizedAnnotation, SmartsError)):
        molecule("C12=CC=CC=CC=CC=C1C2")


def test_restricting_a_cycle_to_the_eligible_subgraph_is_refused_not_silent() -> None:
    """The guard that makes the defect above impossible to reintroduce quietly."""
    import numpy as np

    from vesma.aromaticity import edge_vector

    cycle = np.zeros((3, 3), dtype=bool)
    for left, right in ((0, 1), (1, 2), (0, 2)):
        cycle[left, right] = cycle[right, left] = True

    # A basis holding only one of the triangle's three edges.
    with pytest.raises(ValueError, match="outside the eligible subgraph"):
        edge_vector(cycle, {(0, 1): 0})


@pytest.mark.parametrize(
    ("name", "smiles", "delocalized_bonds"),
    [
        # Each ring carries an acceptable count alone, so the bond they share lies on two
        # accepted circuits and is delocalized: 11 bonds, not 10.
        ("naphthalene", "c1ccc2ccccc2c1", 11),
        ("indole", "c1ccc2[nH]ccc2c1", 10),
        ("quinoline", "c1ccc2ncccc2c1", 11),
        ("anthracene", "c1ccc2cc3ccccc3cc2c1", 16),
        ("pyrene", "c1cc2ccc3cccc4ccc(c1)c2c34", 19),
        # Neither ring is acceptable alone -- five and seven electrons -- and only the
        # ten-electron perimeter is. The transannular bond is interior to that circuit,
        # so it is not delocalized: 10 bonds, not 11.
        ("azulene", "c1ccc2cccc-2cc1", 10),
        # Rings joined by a bond that lies on no circuit of either.
        ("biphenyl", "c1ccccc1-c1ccccc1", 12),
        ("biphenylene", "c1ccc2c(c1)-c1ccccc1-2", 12),
    ],
)
def test_delocalized_bonds_are_circuit_edges_not_fused_system_edges(
    name: str, smiles: str, delocalized_bonds: int
) -> None:
    """A delocalized edge lies on an accepted circuit, not merely inside an aromatic
    system.

    Perception used to mark the union of every ring edge once a fused system passed as a
    whole, which made the bond between two rings delocalized even where the circuit that
    justified them ran around the outside of both. Naphthalene and azulene are the two
    halves of the distinction: both are ten-electron bicyclics, and only one of them has
    a fusion bond on an accepted circuit.
    """
    molecule = cmol_from_smiles(smiles)[0]

    count = int(
        np.count_nonzero(np.triu(molecule.bonds, 1) == int(BondKind.DELOCALIZED))
    )
    assert count == delocalized_bonds, name


@pytest.mark.parametrize(
    "smiles",
    [
        # The four corpus molecules whose atom aromaticity agreed with RDKit exactly
        # while one ring-fusion bond did not. Between them they blocked 324 template
        # applications in the full sweep, because bond class is what a pattern matches.
        "CCCNC(=O)c1c[nH]cc2c(=O)n(-c3ccccc3)nc1-2",
        "CN(C)CCCNC(=O)c1ccc(-n2nc3c4ccccc4[nH]c(C4CC4)c-3c2=O)cc1",
        "CC(C)c1[nH]n(-c2c(Cl)cc(Cl)cc2Cl)c2nc(Cc3ccc(NC(=O)CN4CCNCC4)cc3)nc(=O)c1-2",
        "CC1(C)OB(c2ccc(COc3ccc4c(-c5ccccc5C(=O)O)c5ccc(=O)cc-5oc4c3)cc2)OC1(C)C",
    ],
)
def test_fusion_chord_agrees_with_rdkit_on_the_audited_molecules(smiles: str) -> None:
    Chem = pytest.importorskip("rdkit.Chem")

    molecule = cmol_from_smiles(smiles)[0]
    reference = Chem.MolFromSmiles(smiles)

    for bond in reference.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        delocalized = int(molecule.bonds[left, right]) == int(BondKind.DELOCALIZED)
        assert delocalized == bond.GetIsAromatic(), (left, right)


# Spelling invariance is the property the whole model exists to provide, and until
# recently nothing could measure it: the paired tests above are four hand-picked
# symmetric molecules, none of which can expose a fused system whose rings disagree.
SPELLING_SAMPLE_MOLECULES = 400
SPELLING_VARIANTS_PER_MOLECULE = 3


def localized_variant(chem, molecule, rng):
    """One randomly renumbered, localized spelling, with the exact atom correspondence.

    Two details decide whether this measures anything.

    The molecule is renumbered *before* it is kekulized: ``doRandom`` only varies the
    traversal the writer takes, not which bonds it makes double, so kekulizing first
    would compare one Kekule assignment against itself in a different order.

    And the correspondence is carried explicitly, by composing the renumbering with the
    writer's own output order. Canonical ranks look like they would serve, and do not:
    RDKit ranks atoms partly by bond order, so a delocalized graph and a localized one
    can rank chemically equivalent atoms differently and the comparison then reports
    differences between atoms that were never meant to line up. That mistake produced a
    confident measurement of 5 atom and 16 bond divergences that were not there.
    """
    order = list(range(molecule.GetNumAtoms()))
    rng.shuffle(order)
    renumbered = chem.RenumberAtoms(molecule, order)
    chem.Kekulize(renumbered, clearAromaticFlags=True)
    smiles = chem.MolToSmiles(renumbered, kekuleSmiles=True, canonical=False)
    written = renumbered.GetProp("_smilesAtomOutputOrder")
    output = [int(index) for index in written[1:-1].rstrip(",").split(",")]
    # vesma parses in the order the SMILES was written, so its atom k is the original
    # molecule's atom order[output[k]].
    return smiles, [order[index] for index in output]


@pytest.mark.parametrize(
    ("name", "smiles"),
    [
        # These are the molecules that actually exercise the defect, and none of them is
        # in the frontier corpus: sampling 400 frontier molecules passes against the
        # chemistry that got azulene wrong. A corpus sweep measures how much a defect
        # costs on this data; it does not decide whether the property holds.
        ("azulene", "c1ccc2cccc-2cc1"),
        ("biphenylene", "c1ccc2c(c1)-c1ccccc1-2"),
        ("pyrene", "c1cc2ccc3cccc4ccc(c1)c2c34"),
        ("pentalene", "C1=CC2=CC=CC2=C1"),
        ("naphthalene", "c1ccc2ccccc2c1"),
        ("indole", "c1ccc2[nH]ccc2c1"),
        ("tropone", "O=c1cccccc1"),
        ("thiophene", "c1ccsc1"),
        ("frontier 748", "CCCNC(=O)c1c[nH]cc2c(=O)n(-c3ccccc3)nc1-2"),
        ("frontier 1386", "COCCOCCOc1cc2c(cc1C#N)C(c1ccccc1Cl)=NC1=C(C)NNC1=N2"),
        ("frontier 2284", "NC(=O)CN1CCN(c2ccc3c(c2)N2CCN=C2C(c2ccccc2)=[N+]3[O-])CC1"),
    ],
)
def test_the_discriminating_ring_systems_perceive_the_same_however_spelled(
    name: str, smiles: str
) -> None:
    """Fused systems whose rings disagree with each other, one spelling at a time.

    Verified to fail on the chemistry that preceded the circuit selector: azulene,
    biphenylene and 748 differed in bond classes there, and pyrene in atom aromaticity.
    """
    import random

    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    reference = Chem.MolFromSmiles(smiles)
    delocalized = cmol_from_smiles(smiles)[0]
    rng = random.Random(20260815)

    for _ in range(6):
        spelling, to_original = localized_variant(Chem, Chem.Mol(reference), rng)
        localized = cmol_from_smiles(spelling)[0]
        index = np.array(to_original)
        assert np.array_equal(localized.H, delocalized.H[index]), (name, spelling)
        assert np.array_equal(localized.A, delocalized.A[index]), (name, spelling)
        assert np.array_equal(
            localized.bonds, delocalized.bonds[np.ix_(index, index)]
        ), (name, spelling)


def test_perception_does_not_depend_on_localized_bond_spelling() -> None:
    """One molecule, many localized spellings, one perception -- hydrogens included.

    The contract is narrow and deliberately so: perception is invariant under
    localized/delocalized bond spelling **when atomic number, isotope, charge, total
    hydrogen and connectivity are unchanged**. It says nothing about tautomers, proton
    transfers, or charge-separated resonance forms, which are different molecules and
    should perceive differently.

    Hydrogen is compared too, because the two spellings are decoded by different routes:
    a localized spelling reconstructs an omitted count from graph valence, a delocalized
    one from the sigma frame. Those are two decoders for two encodings of one state, and
    this is what holds them to the same answer.

    Molecules are parsed directly rather than through the benchmark corpus loader, whose
    cache can hold graphs built by earlier chemistry.
    """
    import gzip
    import random

    from tests.datadirs import external_data_root

    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    path = (
        external_data_root() / "external" / "paroutes" / "processed" / "n1-d2-f0.smi.gz"
    )
    rng = random.Random(20260815)
    molecules = comparisons = 0
    hydrogen = atoms = bonds = 0
    with gzip.open(path, "rt") as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            reference = Chem.MolFromSmiles(fields[0])
            if reference is None:
                continue
            try:
                delocalized = cmol_from_smiles(fields[0])[0]
            except Exception:
                continue
            molecules += 1
            for _ in range(SPELLING_VARIANTS_PER_MOLECULE):
                try:
                    spelling, to_original = localized_variant(
                        Chem, Chem.Mol(reference), rng
                    )
                    localized = cmol_from_smiles(spelling)[0]
                except Exception:
                    continue
                if localized.atom_count != delocalized.atom_count:
                    continue
                comparisons += 1
                index = np.array(to_original)
                hydrogen += not np.array_equal(localized.H, delocalized.H[index])
                atoms += not np.array_equal(localized.A, delocalized.A[index])
                bonds += not np.array_equal(
                    localized.bonds, delocalized.bonds[np.ix_(index, index)]
                )
            if molecules >= SPELLING_SAMPLE_MOLECULES:
                break

    assert comparisons >= SPELLING_SAMPLE_MOLECULES
    assert hydrogen == 0, hydrogen
    assert atoms == 0, atoms
    assert bonds == 0, bonds


@pytest.mark.parametrize(
    ("name", "smiles", "aromatic_atoms", "delocalized_bonds"),
    [
        # Ten electrons either way, and the fusion bond is the whole distinction:
        # naphthalene's rings each carry six alone, so the bond they share lies on two
        # accepted circuits; azulene's carry five and seven, so only the perimeter is
        # accepted and the transannular bond is interior to it.
        ("naphthalene", "c1ccc2ccccc2c1", 10, 11),
        ("azulene", "c1ccc2cccc-2cc1", 10, 10),
        ("biphenylene", "c1ccc2c(c1)-c1ccccc1-2", 12, 12),
        ("pyrene", "c1cc2ccc3cccc4ccc(c1)c2c34", 16, 19),
        ("indole", "c1ccc2[nH]ccc2c1", 9, 10),
        # An eight-electron perimeter, so nothing is accepted. Its circuit is still
        # *evaluated*: chordless-ness selects elementary faces for boundary
        # construction, and must not filter which circuits may be considered, or a
        # charged ten-electron pentalene could never be found.
        ("pentalene", "C1=CC2=CC=CC2=C1", 0, 0),
        # The exocyclic pair the circuit-relative count exists to separate.
        ("tropone", "O=c1cccccc1", 7, 7),
        ("fulvene", "C=C1C=CC=C1", 0, 0),
        # Hypervalent controls. A ring sulfur that can join a circuit does; a sulfoxide
        # or sulfone sulfur cannot, so it keeps the concrete selector and the bond-order
        # discrimination that separates it from a sulfide.
        ("thiophene", "c1ccsc1", 5, 5),
        ("thiopyrylium", "[s+]1ccccc1", 6, 6),
        ("phosphole", "C1=CC=CP1", 5, 5),
        ("cyclic sulfide", "C1CCSC1", 0, 0),
        ("cyclic sulfoxide", "O=S1CCCC1", 0, 0),
        ("cyclic sulfone", "O=S1(=O)CCCC1", 0, 0),
    ],
)
def test_perception_of_the_discriminating_ring_systems(
    name: str, smiles: str, aromatic_atoms: int, delocalized_bonds: int
) -> None:
    molecule = cmol_from_smiles(smiles)[0]

    assert int(np.count_nonzero(molecule.A == AROMATIC)) == aromatic_atoms, name
    assert (
        int(np.count_nonzero(np.triu(molecule.bonds, 1) == int(BondKind.DELOCALIZED)))
        == delocalized_bonds
    ), name


def test_the_two_frontier_molecules_that_disagreed_with_rdkit() -> None:
    """One was a defect and is now fixed; the other is a divergence we keep.

    1386's fused seven- and five-membered rings were accepted because a bond into the
    adjacent ring was called exocyclic separately from each face. It is not exocyclic to
    the conjugated component, and counting it that way is what over-aromatized them.

    2284 keeps its divergence for a reason that survives the change: its adjacent
    five-ring contains saturated atoms and is excluded from the eligible component, so
    the bond really is external to the circuit and the fused-bridgehead count stands.
    """
    Chem = pytest.importorskip("rdkit.Chem")

    fixed = cmol_from_smiles("COCCOCCOc1cc2c(cc1C#N)C(c1ccccc1Cl)=NC1=C(C)NNC1=N2")[0]
    reference = Chem.MolFromSmiles(
        "COCCOCCOc1cc2c(cc1C#N)C(c1ccccc1Cl)=NC1=C(C)NNC1=N2"
    )
    assert int(np.count_nonzero(fixed.A == AROMATIC)) == sum(
        atom.GetIsAromatic() for atom in reference.GetAtoms()
    )

    smiles = "NC(=O)CN1CCN(c2ccc3c(c2)N2CCN=C2C(c2ccccc2)=[N+]3[O-])CC1"
    kept = cmol_from_smiles(smiles)[0]
    assert int(np.count_nonzero(kept.A == AROMATIC)) == 16

    # The one frontier molecule whose perception differs from RDKit's, pinned directly so
    # the count in the divergences document cannot go stale. It must keep diverging, and
    # it must diverge in the stated direction: vesma seeing more aromaticity, never less.
    kept_reference = Chem.MolFromSmiles(smiles)
    rdkit_aromatic = sum(atom.GetIsAromatic() for atom in kept_reference.GetAtoms())
    assert int(np.count_nonzero(kept.A == AROMATIC)) > rdkit_aromatic
    assert any(
        (
            int(kept.bonds[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            == BondKind.DELOCALIZED
        )
        != bond.GetIsAromatic()
        for bond in kept_reference.GetBonds()
    )


@pytest.mark.parametrize(
    "substrate,bridge_ring,aromatic_bridge",
    [
        # Carbazole. The nitrogen closes a five-ring sharing one bond with each benzene --
        # ordinary fusion -- and is genuinely aromatic.
        ("Nc1ccccc1-c1ccccc1Cl", 5, True),
        # The same amination bridging the *meta* positions. The nitrogen's ring shares a
        # two-bond path with one benzene, so it bridges rather than fuses, and no circuit
        # through the nitrogen is accepted. It must come out aliphatic.
        ("Nc1cccc(-c2ccccc2Cl)c1", 6, False),
        ("Nc1ccc(-c2ccccc2Cl)cc1", 7, False),
    ],
)
def test_a_ring_closing_n_arylation_never_flags_an_atom_off_its_own_circuit(
    substrate: str, bridge_ring: int, aromatic_bridge: bool
) -> None:
    """Perception may not mark an atom aromatic unless an accepted circuit reaches it.

    The bridged cases raised `CMol aromatic atoms must belong to aromatic bonds` before the
    atom flags were derived from the circuit's own edges, and the raise was a bare
    `ValueError` from a constructor, so one nonsensical forward probe ended a 150-target
    sweep. The chemistry is nonsense either way -- bridging a biaryl's meta positions with
    one N-H is not a Buchwald product -- but refusing it is the search's job, and building
    a molecule that is not a molecule is nobody's.

    The six- and seven-membered closures are the ones that go wrong, because both are
    inside the 3-8 enumeration window *and* the long way round the same bridge is too: all
    four cycles come back chordless, `circuit_boundary` counts the dependent outer one as a
    face, and every edge at the nitrogen then lies in two faces rather than one.
    """
    from vesma.cpu_matcher import CpuReactionMatcher
    from vesma.reaction import reaction_patterns_from_smarts

    buchwald = "[c:1]-[Cl,Br,I].[NH2;D1:2]-[c:3]>>[c:1]-[N:2]-[c:3]"
    matcher = CpuReactionMatcher(reaction_patterns_from_smarts((buchwald,), cache=True))

    precursors, hits = matcher.match([molecule(substrate)], n_jobs=1)
    assert len(hits) == 1, f"the {bridge_ring}-ring closure should match once"

    product = precursors[0]
    nitrogen = int(np.flatnonzero(product.Z == 7)[0])
    assert bool(product.A[nitrogen] == AROMATIC) is aromatic_bridge

    delocalized = (product.bonds & np.uint8(BondKind.DELOCALIZED)) != 0
    stranded = (product.A == AROMATIC) & ~delocalized.any(axis=1)
    not_a_molecule = "an aromatic atom off every aromatic bond is not a molecule"
    assert not np.any(stranded), not_a_molecule


@pytest.mark.parametrize(
    "smiles,aromatic_atoms,aromatic_bonds,role",
    [
        (
            "N1c2cccc(c2)-c2ccccc21",
            12,
            12,
            "a six-ring N bridge across a biaryl meta pair",
        ),
        ("N1c2ccc(cc2)-c2ccccc21", 12, 12, "a seven-ring N bridge, para pair"),
        ("N1C2=CC=CC(=C2)C2=CC=CC=C21", 12, 12, "the meta bridge spelled Kekule"),
        (
            "N1c2ccccc2-c2ccccc21",
            13,
            15,
            "carbazole, where the closure really is aromatic",
        ),
    ],
)
def test_an_atom_is_aromatic_only_where_an_accepted_circuit_edge_is(
    smiles: str, aromatic_atoms: int, aromatic_bonds: int, role: str
) -> None:
    """The flag and the bonds are one statement, not two that happen to agree.

    The joint-acceptance branch used to take its atoms from the union of the component's
    cycles and its edges from the perimeter, which are different sets. A nitrogen bridging
    a biaryl's *meta* positions is where they came apart: the component holds both benzenes
    and two cycles through the nitrogen, the dependent outer one is chordless too -- its
    would-be chord is a two-bond path, not an edge, so `is_chordless` cannot see it -- and
    every edge at the nitrogen then lies on two faces instead of one. The nitrogen was
    flagged aromatic with two single bonds, which is not a molecule, and the first three
    rows here could not be read at all.

    The counts are pinned, not just the equivalence, because they separate this fix from a
    tempting alternative. Excluding the dependent cycle from the faces also makes the model
    consistent -- and gives 13 aromatic atoms over 15 bonds, a fully delocalized tricycle.
    That tricycle is bridged rather than ortho-fused, so it cannot be planar and cannot be
    aromatic; RDKit says 12 and 12, and so does this.
    """
    perceived = molecule(smiles)
    delocalized = (perceived.bonds & np.uint8(BondKind.DELOCALIZED)) != 0

    assert np.array_equal(perceived.A == AROMATIC, delocalized.any(axis=1)), role
    assert int(np.count_nonzero(perceived.A == AROMATIC)) == aromatic_atoms
    assert int(np.count_nonzero(delocalized)) // 2 == aromatic_bonds
