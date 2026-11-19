from __future__ import annotations

from enum import IntEnum

import numpy as np

from .adjacency import BondMatrix, BoolBondMatrix, bond_presence
from .bonds import (
    BOND_CONCRETE_KINDS,
    BOND_ORDER_BY_KIND,
    BondKind,
)
from .configurations import (
    ConfigurationError,
    ConfigurationFailure,
    PiContribution,
    select_circuit_configuration,
)
from .gmol_atoms import ALIPHATIC, AROMATIC, Aromaticity, AtomVector, Charge
from .hydrogens import total_hydrogens
from .rejection import ChemistryRejection, RejectionReason
from .rings import (
    AROMATIC_CYCLE_MAX_SIZE,
    AROMATIC_CYCLE_MIN_SIZE,
    simple_cycles,
)
from .symbols import Z_TO_ELEMENT

AROMATIC_BIT = int(BondKind.DELOCALIZED)


__all__ = [
    "PiDonor",
    "perceive_aromaticity",
    "total_hydrogens",
]

DEFAULT_VALENCE_BY_Z = {
    4: 2,
    5: 3,
    6: 4,
    7: 3,
    8: 2,
    9: 1,
    14: 4,
    15: 3,
    16: 2,
    17: 1,
    32: 4,
    33: 3,
    34: 2,
    35: 1,
    50: 4,
    51: 3,
    52: 2,
    53: 1,
}
OUTER_ELECTRONS_BY_Z = {
    5: 3,
    6: 4,
    7: 5,
    8: 6,
    15: 5,
    16: 6,
    33: 5,
    34: 6,
    52: 6,
}
ELECTRONEGATIVITY_BY_Z = {
    5: 2.04,
    6: 2.55,
    7: 3.04,
    8: 3.44,
    9: 3.98,
    14: 1.90,
    15: 2.19,
    16: 2.58,
    17: 3.16,
    33: 2.18,
    34: 2.55,
    35: 2.96,
    52: 2.10,
    53: 2.66,
}


class PiDonor(IntEnum):
    INVALID = -1
    AMBIGUOUS = -2
    ZERO = 0
    ONE = 1
    TWO = 2


def isoelectronic_valence(atomic_number: int, charge: int, *, fallback: int) -> int:
    """The standard valence of an atom adjusted for formal charge, isoelectronically.

    A charge of ``q`` makes the atom share the valence-electron count of the neutral
    element ``q`` places earlier in the periodic table, so its standard valence is that
    of element ``Z - q``: O- behaves like F (valence 1), N+ like C (4), the tropylium
    C+ like B (3). This is non-linear -- valence peaks at carbon -- which is why a flat
    ``base + charge`` is wrong for cations (it would give C+ valence 5). Both the
    aromatic pi-donor reconstruction and (in its linear-on-occupancy form) the aliphatic
    implicit-hydrogen inference route their charge handling through this one rule, so the
    two cannot drift apart and reintroduce the spurious-hydrogen-on-anions bug.
    """
    return DEFAULT_VALENCE_BY_Z.get(atomic_number - charge, fallback)


def perceive_aromaticity(
    *,
    Z: AtomVector,
    C: Charge,
    explicit_h: AtomVector,
    implicit_h: AtomVector,
    bonds: BondMatrix,
    pinned_total_h: AtomVector | None = None,
) -> tuple[Aromaticity, BondMatrix, AtomVector]:
    """Perceive aromaticity directly on the input bonds, with no Kekule
    localization. Aromaticity is a delocalized property: each ring atom's pi
    contribution is reconstructed from its (charge-adjusted) standard valence and
    its concrete exocyclic bonds (see ``pi_donor``), so single/double/aromatic ring
    bonds all perceive identically. Returns the atom aromatic mask, bonds in the
    canonical pure-aromatic form (perceived-aromatic edges carry ``BOND_AROMATIC``
    with no order; every other present edge carries its concrete order), and the
    per-atom total hydrogen count that drove perception (the authoritative H, so
    callers never recompute it from the order-stripped canonical bonds).

    ``pinned_total_h`` supplies counts the caller has already decided; perception takes
    those verbatim instead of inferring them. Reaction editing pins every surviving
    atom, so a precursor's hydrogen comes from the hydrogen law."""
    adjacency = bond_presence(bonds)
    concrete = (bonds & np.uint8(BOND_CONCRETE_KINDS)).astype(np.uint8)
    source_aromatic = ((bonds & np.uint8(AROMATIC_BIT)) != 0).any(axis=1)
    bond_orders = BOND_ORDER_BY_KIND[bonds & np.uint8(BOND_CONCRETE_KINDS)]
    heavy_degrees = adjacency.sum(axis=1).astype(np.int16)
    order_sums = bond_orders.sum(axis=1).astype(np.int16)
    total_h = total_hydrogens(
        Z=Z,
        C=C,
        explicit_h=explicit_h,
        implicit_h=implicit_h,
        order_sums=order_sums,
        heavy_degrees=heavy_degrees,
        bond_orders_omitted=source_aromatic,
        pinned_total_h=pinned_total_h,
    )

    if adjacency.shape[0] > 64:
        # A graph wider than the 64-bit cycle-mask convention. If it is acyclic there are
        # no rings to perceive, so short-circuit to all-aliphatic and skip enumeration (a
        # cheap fast-path for big protected chains). A cyclic graph this large -- e.g. a
        # heavily protected retro intermediate before the connected-component split --
        # falls through to the normal cycle-based perception below, whose per-atom masks
        # are arbitrary-precision Python ints and so handle any atom count.
        active = np.any(adjacency, axis=1)
        component_count = len(components_from_adjacency(adjacency)) + int(
            np.count_nonzero(~active)
        )
        edge_count = int(np.count_nonzero(np.triu(adjacency, k=1)))
        if edge_count == adjacency.shape[0] - component_count:
            return (
                np.full(Z.shape[0], ALIPHATIC, dtype=np.uint8),
                canonical_bonds(concrete, adjacency, np.zeros_like(adjacency)),
                total_h,
            )

    aromatic_atoms = np.zeros(Z.shape[0], dtype=bool)
    aromatic_edges = np.zeros_like(adjacency)

    cycles = tuple(
        simple_cycles(
            adjacency,
            min_size=AROMATIC_CYCLE_MIN_SIZE,
            max_size=AROMATIC_CYCLE_MAX_SIZE,
        )
    )
    # Whether an atom can belong to a circuit at all is a property of its configuration,
    # not of any candidate circuit, so it is settled before cycles are grouped. Grouping
    # first would let a saturated ring that merely shares a bond with an aromatic one be
    # fused into the same system: the joint count then fails on the sp3 atoms, and the
    # per-cycle fallback tests each aromatic ring alone, where the shared fusion bond is
    # exocyclic and its electron goes uncounted. Real fused heteroaromatics are rejected
    # that way -- a triazolo-pyridazine counts ten electrons as one system and five as
    # either ring by itself.
    no_circuit = np.zeros_like(adjacency)
    can_delocalize = np.array(
        [
            pi_donor(
                atom=atom,
                cyclic_edges=no_circuit,
                Z=Z,
                C=C,
                total_h=total_h,
                bond_orders=bond_orders,
                heavy_degrees=heavy_degrees,
                delocalized_atoms=source_aromatic,
            )
            is not PiDonor.INVALID
            for atom in range(Z.shape[0])
        ],
        dtype=bool,
    )
    # An atom carrying an asserted delocalized bond whose configuration cannot join a
    # circuit is a configuration failure, and saying so here is the difference between a
    # true cause and a downstream symptom: left alone it surfaces later as an orphan edge,
    # which reports that the bond is not cyclic when the ring is plainly cyclic.
    unfit = source_aromatic & ~can_delocalize
    if unfit.any():
        atom = int(np.flatnonzero(unfit)[0])
        raise ConfigurationError(
            f"Atom {atom} ({Z_TO_ELEMENT.get(int(Z[atom]), '*')}) carries a delocalized "
            f"bond but its written state occupies no configuration that can belong to a "
            f"circuit.",
            failure=ConfigurationFailure.INCOMPATIBLE,
        )
    cycles = tuple(cycle for cycle in cycles if bool(can_delocalize[cycle].all()))
    cycle_edge_matrices = tuple(
        cycle_edge_matrix(cycle, Z.shape[0]) for cycle in cycles
    )
    for component in fused_cycle_components(cycle_edge_matrices):
        component_cycles = tuple(cycles[index] for index in component)
        component_edges = tuple(cycle_edge_matrices[index] for index in component)
        # Each ring is asked on its own first. A ring that carries an acceptable count by
        # itself is an accepted circuit, and a bond two such rings share belongs to both
        # of them -- naphthalene's fusion bond is delocalized because it lies on two
        # circuits, not because the pair of rings together came to a total.
        # The exocyclic classification is made against the whole component, not against
        # the single face being counted: a bond into the neighbouring ring is not
        # exocyclic to the conjugated system, and calling it so made the count depend on
        # where a Kekule spelling happened to put its double bonds.
        component_span = np.logical_or.reduce(component_edges)
        accepted = tuple(
            cycle_system_is_aromatic(
                cycles=(cycle,),
                cyclic_edges=component_span,
                Z=Z,
                C=C,
                total_h=total_h,
                bond_orders=bond_orders,
                heavy_degrees=heavy_degrees,
                delocalized_atoms=source_aromatic,
            )
            for cycle, cycle_edges in zip(
                component_cycles, component_edges, strict=True
            )
        )
        for cycle, cycle_edges, ok in zip(
            component_cycles, component_edges, accepted, strict=True
        ):
            if ok:
                aromatic_atoms[cycle] = True
                aromatic_edges |= cycle_edges

        # Only where a ring cannot stand alone is the fused system asked as a whole. Then
        # the accepted circuit is the system's perimeter, and the bonds between rings are
        # interior to it rather than on it -- see ``circuit_boundary``.
        if (
            all(accepted)
            or len(component) <= 1
            or not cycle_system_is_aromatic(
                cycles=component_cycles,
                cyclic_edges=component_span,
                Z=Z,
                C=C,
                total_h=total_h,
                bond_orders=bond_orders,
                heavy_degrees=heavy_degrees,
                delocalized_atoms=source_aromatic,
            )
        ):
            continue
        # The atoms come from the circuit's own edges, not from the cycles that proposed it.
        # Taking the union of the component's cycles claims more than the circuit supports: the
        # perimeter is what was accepted, so an atom the perimeter does not touch delocalizes
        # across nothing, and saying it is aromatic anyway builds a molecule that is not one.
        #
        # An amine bridging the *meta* positions of a biaryl is the case that showed this. The
        # bridged component holds four cycles -- both benzenes, a six-ring through the nitrogen
        # the short way round, and an eight-ring the long way -- and all four are chordless, so
        # `circuit_boundary` counts the dependent eight-ring as a face and every edge at the
        # nitrogen falls in exactly two faces rather than one. The benzenes were already accepted
        # ring by ring above, so the perimeter adds nothing at the nitrogen, and the nitrogen was
        # nonetheless flagged aromatic with two single bonds -- which `CMol` then refused, fatally,
        # in the middle of a sweep.
        #
        # Deriving the flags from the edges makes that unreachable rather than merely unlikely:
        # every branch that marks an atom aromatic now supplies an edge for it in the same
        # statement, so "an aromatic atom lies on an aromatic bond" holds by construction instead
        # of being checked at the end. It is also what the module already says a delocalized atom
        # is -- see `circuit_boundary`, where azulene's transannular bond is not on the circuit.
        boundary = circuit_boundary(component_cycles, Z.shape[0], adjacency)
        aromatic_edges |= boundary
        aromatic_atoms |= np.any(boundary, axis=1)

    # Automatic perception stops at the 3-8 domain, but a circuit the input *asserted*
    # is a claim to check rather than a candidate to discover, so its size is not
    # bounded by what enumeration can afford to search. Any asserted circuit left over
    # is validated here against the same Huckel bookkeeping.
    asserted_circuit_edges: list[BoolBondMatrix] = []
    for circuit in asserted_circuits(adjacency, concrete, aromatic_edges):
        circuit_edges = cycle_edge_matrix(circuit, Z.shape[0])
        if not cycle_system_is_aromatic(
            cycles=(circuit,),
            cyclic_edges=circuit_edges,
            Z=Z,
            C=C,
            total_h=total_h,
            bond_orders=bond_orders,
            heavy_degrees=heavy_degrees,
            delocalized_atoms=source_aromatic,
        ):
            # The circuit was found by walking the asserted edges, so it is cyclic by
            # construction; what failed is the electron count. Falling through would
            # leave its edges orphaned and report the opposite of what is true.
            raise DelocalizedSystemInvalidError(
                f"A delocalized circuit of {circuit.size} atoms was asserted, but its "
                f"electron count does not satisfy Huckel."
            )
        aromatic_atoms[circuit] = True
        aromatic_edges |= circuit_edges
        asserted_circuit_edges.append(circuit_edges)

    require_annotation_outside_the_automatic_domain(
        adjacency=adjacency,
        can_delocalize=can_delocalize,
        examined_cycles=(*cycle_edge_matrices, *asserted_circuit_edges),
    )

    perceived_A = np.where(aromatic_atoms, AROMATIC, ALIPHATIC).astype(np.uint8)
    return (
        perceived_A,
        canonical_bonds(concrete, adjacency, aromatic_edges),
        total_h,
    )


class DelocalizedSystemInvalidError(ChemistryRejection):
    """An asserted circuit that is genuinely a circuit, but not a delocalized one.

    Distinct from an orphan edge, and the distinction is the whole point of a typed
    reason: an orphan bond claims delocalization outside any circuit, while this claims it
    inside a circuit whose electron count refuses it. Cyclooctatetraene is the second, not
    the first, and reporting it as the first states the opposite of what is true.
    """

    reason = RejectionReason.DELOCALIZED_SYSTEM_INVALID


class RequiresDelocalizedAnnotation(ChemistryRejection):
    """A ring outside the automatic domain that only an annotation can settle.

    Perception enumerates cycles of 3 to 8 atoms. Above that the search is not affordable,
    but "we did not look" is not the same answer as "not aromatic", and returning the
    second for the first is how a localized [14]annulene came back aliphatic while the
    same molecule written delocalized came back aromatic. The model does not get to be
    spelling-dependent, so it declines instead: write the ring delocalized and the
    assertion is checked at whatever size it was written.
    """

    reason = RejectionReason.REQUIRES_DELOCALIZED_ANNOTATION


def require_annotation_outside_the_automatic_domain(
    *,
    adjacency: BoolBondMatrix,
    can_delocalize: np.ndarray[tuple[int], np.dtype[np.bool_]],
    examined_cycles: tuple[BoolBondMatrix, ...],
) -> None:
    """Refuse a ring that could delocalize but is larger than perception looks.

    Only the *existence* of such a ring has to be decided, never which one it is, and
    that is a question about the cycle space. Restrict to the subgraph of atoms that
    could belong to a circuit at all -- this is what keeps ordinary macrocycles quiet,
    since a macrolide's sp3 carbons cannot join one and its alkenes are left as acyclic
    fragments. That subgraph's cycle rank, ``E - V + components``, is the number of
    independent cycles it contains. The cycles perception already examined span a
    subspace of that space, and its dimension is their rank over GF(2). If the whole
    space is larger, some independent cycle was never examined.

    Comparing ranks rather than deleting edges is what makes this exact. Deleting the
    edges of small cycles loses a large cycle that merely shares one with them, and that
    is the common fused case -- the very shape the check exists to catch.
    """
    eligible = adjacency & can_delocalize[:, None] & can_delocalize[None, :]
    if not eligible.any():
        return
    edges = [
        (int(left), int(right))
        for left, right in np.argwhere(np.triu(eligible, k=1)).tolist()
    ]
    nodes = int(np.count_nonzero(eligible.any(axis=1)))
    components = len(components_from_adjacency(eligible))
    cycle_rank = len(edges) - nodes + components
    if cycle_rank <= 0:
        return

    index = {edge: position for position, edge in enumerate(edges)}
    examined = [edge_vector(cycle, index) for cycle in examined_cycles if cycle.any()]
    if gf2_rank(examined) >= cycle_rank:
        return
    raise RequiresDelocalizedAnnotation(
        f"A ring outside the {AROMATIC_CYCLE_MIN_SIZE}-{AROMATIC_CYCLE_MAX_SIZE} atom "
        f"domain perception searches has atoms that could delocalize, and nothing has "
        f"examined it. Write the ring delocalized to assert it."
    )


def edge_vector(cycle: BoolBondMatrix, index: dict[tuple[int, int], int]) -> int:
    """One cycle as a bitmask over the eligible-edge basis, for GF(2) arithmetic.

    Every edge must be in the basis. Quietly skipping the ones that are not would turn a
    cycle passing through an ineligible atom into an open path -- still a nonzero vector,
    so it would raise the measured rank and hide the very ring the caller is looking for.
    Callers pass cycles wholly inside the eligible subgraph, and this refuses anything
    else rather than silently degrading it.
    """
    vector = 0
    for left, right in np.argwhere(np.triu(cycle, k=1)).tolist():
        position = index.get((int(left), int(right)))
        if position is None:
            raise ValueError(
                "A cycle outside the eligible subgraph cannot be a basis vector."
            )
        vector |= 1 << position
    return vector


def gf2_rank(vectors: list[int]) -> int:
    """Rank of a set of GF(2) vectors, by elimination on bitmasks.

    Addition over GF(2) is exclusive or, so Gaussian elimination is a loop of pivots and
    xors with no arithmetic to go wrong and no tolerance to choose.
    """
    pivots: list[int] = []
    for vector in vectors:
        for pivot in pivots:
            vector = min(vector, vector ^ pivot)
        if vector:
            pivots.append(vector)
    return len(pivots)


def asserted_circuits(
    adjacency: BoolBondMatrix,
    concrete: BondMatrix,
    aromatic_edges: BoolBondMatrix,
) -> list[AtomVector]:
    """Circuits the input asserted that perception has not already accounted for.

    An asserted delocalized edge carries no bond order, so a connected run of them is
    either a circuit or a claim that cannot be one. Only a component in which every atom
    has exactly two asserted neighbours is returned: that is a simple cycle, and reading
    it off costs a walk rather than an enumeration, so an authored ring is validated at
    whatever size it was written. A component of any other shape is left alone and
    reported as an orphan, because resolving it would mean searching for which of its
    cycles the author meant.
    """
    asserted = adjacency & (concrete == 0) & ~aromatic_edges
    if not asserted.any():
        return []
    degrees = asserted.sum(axis=1)
    circuits: list[AtomVector] = []
    for component in components_from_adjacency(asserted):
        if component.size < AROMATIC_CYCLE_MIN_SIZE:
            continue
        if not np.all(degrees[component] == 2):
            continue
        start = int(component.min())
        walk = [start]
        previous, current = start, int(np.flatnonzero(asserted[start])[0])
        while current != start:
            walk.append(current)
            neighbours = np.flatnonzero(asserted[current])
            step = [int(atom) for atom in neighbours if atom != previous]
            previous, current = current, step[0]
        circuits.append(np.array(walk, dtype=np.int32))
    return circuits


class OrphanDelocalizedBondError(ChemistryRejection):
    """A delocalized edge that belongs to no accepted circuit.

    Delocalization is a property of a circuit, so an edge asserting it outside one
    asserts something false. The edge is not quietly resolved to a single bond: that
    would invent a bond order the input never gave and hand back a different molecule
    than the one asked for. A template that destroys a circuit must author every
    concrete bond it releases.
    """

    reason = RejectionReason.DELOCALIZED_EDGE_NOT_CYCLIC


def canonical_bonds(
    concrete: BondMatrix,
    adjacency: BoolBondMatrix,
    aromatic_edges: BoolBondMatrix,
) -> BondMatrix:
    """Canonical bond matrix: delocalized edges carry pure ``DELOCALIZED``; every other
    present edge carries its concrete order."""
    orphan = adjacency & (concrete == 0) & ~aromatic_edges
    if orphan.any():
        raise OrphanDelocalizedBondError(
            "Delocalized bond belongs to no accepted circuit; a released bond must be "
            "authored, never inferred."
        )
    result = concrete.copy()
    result[aromatic_edges] = np.uint8(AROMATIC_BIT)
    return result


_PI_DONOR_BY_CONTRIBUTION = {
    PiContribution.INVALID: PiDonor.INVALID,
    PiContribution.AMBIGUOUS: PiDonor.AMBIGUOUS,
    PiContribution.NONE: PiDonor.ZERO,
    PiContribution.ONE: PiDonor.ONE,
    PiContribution.TWO: PiDonor.TWO,
}


def pi_donor(
    *,
    atom: int,
    cyclic_edges: BoolBondMatrix,
    Z: AtomVector,
    C: Charge,
    total_h: AtomVector,
    bond_orders: BondMatrix,
    heavy_degrees: AtomVector,
    delocalized_atoms: np.ndarray[tuple[int], np.dtype[np.bool_]],
) -> PiDonor:
    """Pi electrons an atom brings to a candidate circuit.

    What the atom *can* bring is a property of its configuration, so it is read from the
    configuration table rather than re-derived here. What remains is the one question the
    table cannot answer, because it is about the circuit rather than the atom: when the
    atom has a single pi bond, does that bond lie inside this circuit?

    This is Huckel bookkeeping relative to the candidate circuit, not a claim about where
    electron density physically sits. An endocyclic pi bond puts its electron into the
    circuit; a sole exocyclic one engages the atom's p orbital elsewhere and puts nothing
    in. Which way the bond points is a topological fact, so carbon, nitrogen and oxygen
    partners are treated alike and no electronegativity comparison is needed -- tropone
    counts six and is delocalized, fulvene counts four and is not.
    """
    configuration = select_circuit_configuration(
        z=int(Z[atom]),
        charge=int(C[atom]),
        sigma_degree=int(heavy_degrees[atom]),
        total_h=int(total_h[atom]),
    )
    if isinstance(configuration, ConfigurationFailure):
        return PiDonor.INVALID
    if configuration.pi is PiContribution.INVALID:
        return PiDonor.INVALID
    if configuration.pi is PiContribution.ONE:
        # Relative to the whole eligible system, not to one face of it. A concrete double
        # bond drawn *inside* the system is a Kekule spelling of bonds the delocalized
        # spelling does not write at all, so letting its placement decide the count makes
        # perception depend on notation. A double bond leaving the system is authored
        # chemistry that survives every spelling -- tropone's carbonyl, fulvene's
        # exocyclic methylene -- and still engages the p orbital elsewhere.
        exocyclic_double = (bond_orders[atom] >= 2) & ~cyclic_edges[atom]
        if exocyclic_double.any():
            return PiDonor.ZERO
        return PiDonor.ONE
    # Converted by name, not by value: the two enums number their non-numeric states
    # differently, so an integer conversion silently turned an unresolved contribution
    # into an invalid one.
    return _PI_DONOR_BY_CONTRIBUTION[configuration.pi]


def cycle_system_is_aromatic(
    *,
    cycles: tuple[AtomVector, ...],
    cyclic_edges: BoolBondMatrix,
    Z: AtomVector,
    C: Charge,
    total_h: AtomVector,
    bond_orders: BondMatrix,
    heavy_degrees: AtomVector,
    delocalized_atoms: np.ndarray[tuple[int], np.dtype[np.bool_]],
) -> bool:
    """Whether a circuit is delocalized.

    An atom that cannot take part rules the circuit out outright; otherwise the electron
    count must satisfy Huckel. Because every contribution is decided relative to this
    circuit, a molecule perceives identically however its bonds were spelled.
    """
    atoms = np.unique(np.concatenate(cycles))
    donors = tuple(
        pi_donor(
            atom=int(atom),
            cyclic_edges=cyclic_edges,
            Z=Z,
            C=C,
            total_h=total_h,
            bond_orders=bond_orders,
            heavy_degrees=heavy_degrees,
            delocalized_atoms=delocalized_atoms,
        )
        for atom in atoms
    )
    if PiDonor.INVALID in donors:
        return False
    electron_count = sum(int(donor) for donor in donors)
    return electron_count >= 2 and (electron_count - 2) % 4 == 0


def is_chordless(cycle: AtomVector, adjacency: BoolBondMatrix) -> bool:
    """Whether the cycle has no edge joining two atoms that are not its neighbours.

    A cycle carrying a chord is not an elementary face of the ring system; it is two
    smaller cycles seen from outside. Two fused five-rings, for instance, also enumerate
    as an eight-ring whose chord is the bond they share.
    """
    atoms = [int(atom) for atom in cycle]
    size = len(atoms)
    for left in range(size):
        for right in range(left + 1, size):
            if right - left == 1 or (left == 0 and right == size - 1):
                continue
            if adjacency[atoms[left], atoms[right]]:
                return False
    return True


def circuit_boundary(
    cycles: tuple[AtomVector, ...],
    atom_count: int,
    adjacency: BoolBondMatrix,
) -> BoolBondMatrix:
    """The edges of a jointly-aromatic ring system that lie on its circuit.

    A delocalized edge is one that belongs to an accepted circuit -- not one that merely
    joins two atoms which participate somewhere in the same aromatic system. Where a
    fused system is accepted *as a whole*, because no constituent ring carries an
    acceptable count on its own, the accepted circuit is the system's perimeter and the
    bonds shared between rings are not part of it. Azulene is the textbook case: five and
    seven electrons separately, ten around the rim, and a transannular bond that measures
    like an ordinary single bond rather than an aromatic one.

    The boundary is computed as the edges belonging to exactly one elementary face. Only
    chordless cycles count as faces: ``simple_cycles`` returns every simple cycle in the
    3-8 window, so a pair of fused five-rings also yields their eight-ring perimeter, and
    including that dependent cycle would cancel the very edges it is made of.

    This applies *only* to the jointly-accepted case. A system whose rings each carry
    their own acceptable count -- naphthalene -- is accepted ring by ring elsewhere, and
    the bond they share belongs to two accepted circuits and stays delocalized.
    """
    faces = tuple(cycle for cycle in cycles if is_chordless(cycle, adjacency))
    if not faces:
        # Nothing elementary to bound; keep the whole system rather than emit no circuit
        # at all, which would read as "these atoms delocalize across no bond".
        return np.logical_or.reduce(
            tuple(cycle_edge_matrix(cycle, atom_count) for cycle in cycles)
        )
    incidence = np.zeros((atom_count, atom_count), dtype=np.int16)
    for face in faces:
        incidence += cycle_edge_matrix(face, atom_count).astype(np.int16)
    return incidence == 1


def cycle_edge_matrix(cycle: AtomVector, atom_count: int) -> BoolBondMatrix:
    edges = np.zeros((atom_count, atom_count), dtype=bool)
    for index, left in enumerate(cycle):
        right = cycle[(index + 1) % cycle.size]
        edges[left, right] = True
        edges[right, left] = True
    return edges


def fused_cycle_components(
    cycle_edges: tuple[BoolBondMatrix, ...],
) -> list[tuple[int, ...]]:
    if not cycle_edges:
        return []
    adjacency = np.zeros((len(cycle_edges), len(cycle_edges)), dtype=bool)
    for left in range(len(cycle_edges)):
        for right in range(left + 1, len(cycle_edges)):
            if np.any(cycle_edges[left] & cycle_edges[right]):
                adjacency[left, right] = True
                adjacency[right, left] = True

    components: list[tuple[int, ...]] = []
    visited = np.zeros(len(cycle_edges), dtype=bool)
    for root in range(len(cycle_edges)):
        if visited[root]:
            continue
        stack = [root]
        visited[root] = True
        component: list[int] = []
        while stack:
            cycle = stack.pop()
            component.append(cycle)
            for neighbor in np.flatnonzero(adjacency[cycle]):
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                stack.append(int(neighbor))
        components.append(tuple(sorted(component)))
    return components


def electrons_stolen(*, donor_z: int, acceptor_z: int) -> bool:
    donor = ELECTRONEGATIVITY_BY_Z.get(donor_z)
    acceptor = ELECTRONEGATIVITY_BY_Z.get(acceptor_z)
    return donor is not None and acceptor is not None and acceptor > donor


def components_from_adjacency(adjacency: BoolBondMatrix) -> list[AtomVector]:
    visited = np.zeros(adjacency.shape[0], dtype=bool)
    components: list[AtomVector] = []
    for root in np.flatnonzero(np.any(adjacency, axis=1)):
        if visited[root]:
            continue
        stack = [int(root)]
        visited[root] = True
        component: list[int] = []
        while stack:
            atom = stack.pop()
            component.append(atom)
            for neighbor in np.flatnonzero(adjacency[atom]):
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                stack.append(int(neighbor))
        components.append(np.asarray(sorted(component), dtype=np.int16))
    return components
