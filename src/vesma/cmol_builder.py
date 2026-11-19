from __future__ import annotations

import numpy as np

from .adjacency import (
    BondMatrix,
    bond_order_matrix,
    bond_presence,
    bonds_with_unknown_topology,
)
from .aromaticity import perceive_aromaticity
from .bonds import (
    BondKind,
    exact_bond_kind,
    resolve_default_bond_flags,
)
from .cmol import CMol
from .configurations import (
    ConfigurationError,
    ConfigurationFailure,
    select_configuration,
)
from .gmol import GMol
from .gmol_atoms import (
    ALIPHATIC,
    ALL_A_MASK,
    AROMATIC,
    MASS_ANY,
    MASS_UNLABELED,
    Aromaticity,
    AtomData,
    AtomMasks,
    AtomVector,
    Charge,
    atom_masks_from_props,
    bucket_mask_exact,
    mask_for_charge,
    z_mask_for_numbers,
)
from .gmol_builder import (
    SmartsConversionInfo,
    gmol_from_normalized_smarts,
    gmol_from_smarts,
    gmol_from_smiles,
)
from .hydrogens import (
    HydrogenFailure,
    UnwrittenHydrogenError,
    hydrogens_for_written_state,
)
from .parsing.errors import SmartsError
from .parsing.ir import NormalizedSmartsMol
from .rejection import ChemistryRejection
from .rings import RingDescriptors, ring_info_from_bonds
from .symbols import Z_TO_ELEMENT

AtomMapVector = np.ndarray[tuple[int], np.dtype[np.int32]]


def perceived_cmol(
    graph: GMol, info: SmartsConversionInfo
) -> tuple[CMol, SmartsConversionInfo]:
    """Perceive a parsed graph into a molecule, keeping a refusal's own reason.

    A ``ChemistryRejection`` already names why the model declined, and re-wrapping it as
    a bare ``SmartsError`` would discard exactly the information a batch caller counts by.
    """
    try:
        return cmol_from_gmol(graph, perceive_aromaticity=True), info
    except SmartsError, ChemistryRejection:
        raise
    except ValueError as exc:
        raise SmartsError(str(exc)) from exc


def cmol_from_smiles(smiles: str) -> tuple[CMol, SmartsConversionInfo]:
    """Build a molecule from SMILES text.

    The SMILES reading runs on the IR before any graph is built, so a bracket atom that
    wrote no hydrogen count arrives here carrying an exact ``H0`` -- which is what
    ``[C]`` and ``[CH0]`` being the same molecule means. Composing the *query* reading
    with ``cmol_from_gmol`` cannot do this: by then the bracket is gone and there is
    nothing left to concretize, so the hydrogen count gets derived instead.
    """
    return perceived_cmol(*gmol_from_smiles(smiles))


def cmol_from_smarts(smarts: str) -> tuple[CMol, SmartsConversionInfo]:
    """Concretize a *query* into a molecule, for a SMARTS that happens to be exact.

    This is not the way to read SMILES -- use ``cmol_from_smiles``. Exactness is checked
    only for the element, the aromaticity and the bond kinds; charge and hydrogen fall
    back to defaults, so a query that constrains them loosely still yields a molecule.
    """
    return perceived_cmol(*gmol_from_smarts(smarts))


def cmol_from_normalized_smarts(
    smarts_mol: NormalizedSmartsMol,
) -> tuple[CMol, SmartsConversionInfo]:
    return perceived_cmol(*gmol_from_normalized_smarts(smarts_mol))


def resolved_mass(atom_masks: AtomMasks) -> AtomVector:
    """The nucleus a query graph denotes once it becomes a molecule.

    A query that says nothing about the nucleus yields natural abundance, the same way
    an omitted charge or hydrogen count resolves to its unlabelled default.
    """
    return np.where(atom_masks.M == MASS_ANY, MASS_UNLABELED, atom_masks.M).astype(
        np.int16
    )


def cmol_from_gmol(
    graph: GMol,
    *,
    perceive_aromaticity: bool = False,
    ring_descriptors: RingDescriptors = RingDescriptors.ALL,
) -> CMol:
    atom_masks = graph.atom_masks
    atom_count = atom_masks.size
    # Atom aromatic intent, used to resolve an omitted/default query bond
    # (QBOND_DEFAULT) into a concrete seed bond when a SMARTS-parsed pattern is
    # converted into a molecule. Only the query path (perceive_aromaticity=True) needs
    # this; a concrete graph (perceive_aromaticity=False) already carries single-class
    # bonds that pass through untouched.
    a_intent = atom_masks.exact_a()
    bonds = np.zeros((atom_count, atom_count), dtype=np.uint8)
    for left in range(atom_count):
        for right in range(atom_count):
            flags = int(graph.bonds[left, right])
            if flags == 0:
                continue
            if perceive_aromaticity:
                flags = resolve_default_bond_flags(
                    flags,
                    left_aromatic=int(a_intent[left]) == AROMATIC,
                    right_aromatic=int(a_intent[right]) == AROMATIC,
                )
            concrete_flags = exact_bond_kind(flags)
            if concrete_flags is None:
                raise ValueError(
                    "Concrete molecule conversion requires exact bond types."
                )
            bonds[left, right] = np.uint8(concrete_flags)

    Z = atom_masks.exact_z()
    if np.any(Z == -1):
        raise ValueError(
            "Concrete molecule conversion requires single-bit atomic number mask."
        )

    C = atom_masks.exact_c()

    aromatic_edges = ((bonds & np.uint8(BondKind.DELOCALIZED)) != 0).any(axis=1)
    a_masks = atom_masks.exact_a()
    A = np.where(
        a_masks == ALL_A_MASK, np.where(aromatic_edges, AROMATIC, ALIPHATIC), a_masks
    ).astype(np.uint8)

    h_total = atom_masks.exact_h_total()
    h_implicit = atom_masks.exact_h_implicit()

    explicit_h = np.zeros(atom_count, dtype=np.int16)
    implicit_h = np.full(atom_count, -1, dtype=np.int16)

    h_total_valid = h_total != -1
    h_implicit_valid = h_implicit != -1

    # Case 1: Total is specified, implicit is NOT specified
    mask1 = h_total_valid & ~h_implicit_valid
    explicit_h[mask1] = h_total[mask1]
    implicit_h[mask1] = 0

    # Case 2: Implicit IS specified
    # explicit_h = max(h_total - h_implicit, 0)
    mask2 = h_implicit_valid
    h_tot = np.where(
        h_total_valid, h_total, 0
    )  # Fallback to 0 for math if not specified, but usually it should be specified
    explicit_h[mask2] = np.maximum(h_tot[mask2] - h_implicit[mask2], 0).astype(np.int16)
    implicit_h[mask2] = h_implicit[mask2]

    if perceive_aromaticity:
        return cmol_from_seed_data(
            Z=Z,
            M=resolved_mass(atom_masks),
            C=C,
            explicit_h=explicit_h,
            implicit_h=implicit_h,
            atom_map=graph.atom_map,
            bonds=bonds,
            ring_descriptors=ring_descriptors,
        )

    return _cmol_from_exact_seed_data(
        atom_masks=atom_masks,
        Z=Z,
        A=A,
        C=C,
        explicit_h=explicit_h,
        implicit_h=implicit_h,
        atom_map=graph.atom_map,
        bonds=bonds,
        ring_descriptors=ring_descriptors,
    )


def cmol_from_exact_arrays(
    *,
    Z: AtomVector,
    A: Aromaticity,
    M: AtomVector | None = None,
    C: Charge,
    D: AtomVector,
    H: AtomVector,
    h: AtomVector,
    v: AtomVector,
    X: AtomVector,
    R: AtomVector,
    r: AtomVector,
    x: AtomVector,
    explicit_h: AtomVector,
    implicit_h: AtomVector,
    atom_map: AtomMapVector,
    bonds: BondMatrix,
    ring_descriptors: RingDescriptors = RingDescriptors.ALL,
) -> CMol:
    atom_count = Z.shape[0]
    M = np.zeros_like(Z) if M is None else M
    bond_graph = bond_presence(bonds)
    bond_orders = bond_order_matrix(bonds)
    ring_info = ring_info_from_bonds(bonds) if ring_descriptors else None
    membership = ring_info if ring_descriptors & RingDescriptors.MEMBERSHIP else None
    connectivity = (
        ring_info if ring_descriptors & RingDescriptors.CONNECTIVITY else None
    )
    aromatic_atoms = A == AROMATIC

    # Vectorized computation of degrees and bond order sums
    degrees = np.asarray(bond_graph.sum(axis=1), dtype=np.int16)
    bond_order_sums = np.asarray(bond_orders.sum(axis=1), dtype=np.int16)

    D = np.where(D < 0, degrees, D)

    for idx in range(atom_count):
        degree = int(degrees[idx])
        explicit = max(int(explicit_h[idx]), 0)
        raw_implicit = int(implicit_h[idx])
        if h[idx] >= 0:
            inferred_implicit = int(h[idx])
        elif raw_implicit >= 0:
            inferred_implicit = raw_implicit
        elif int(C[idx]) != 0:
            # A charge is only writable inside brackets, and a bracket atom carries no
            # implicit hydrogen. See total_hydrogens for the same rule on the other path.
            inferred_implicit = 0
        else:
            # Nothing was written, so the count is the fewest the atom's declared
            # configurations allow -- the same law molecule literals and added RHS atoms
            # use, so all three agree on what a bare N or a lowercase n means.
            resolved = hydrogens_for_written_state(
                z=int(Z[idx]),
                charge=int(C[idx]),
                sigma_degree=degree,
                bond_order_sum=int(bond_order_sums[idx]),
                delocalized=bool(aromatic_atoms[idx]),
            )
            if isinstance(resolved, HydrogenFailure):
                symbol = Z_TO_ELEMENT.get(int(Z[idx]), "*")
                raise UnwrittenHydrogenError(
                    f"Atom {idx} ({symbol}) has no determined hydrogen count: "
                    f"{resolved.name.lower()}.",
                    failure=resolved,
                )
            inferred_implicit = resolved - explicit

        total_h = explicit + inferred_implicit
        if H[idx] < 0:
            H[idx] = np.int16(total_h)
        if h[idx] < 0:
            h[idx] = np.int16(inferred_implicit)
        if v[idx] < 0:
            v[idx] = np.int16(
                selected_valence(
                    z=int(Z[idx]),
                    charge=int(C[idx]),
                    sigma_degree=degree,
                    bond_order_sum=int(bond_order_sums[idx]),
                    total_h=int(H[idx]),
                    delocalized=bool(aromatic_atoms[idx]),
                    atom=idx,
                )
            )
        if X[idx] < 0:
            X[idx] = np.int16(degree + int(H[idx]))
        if R[idx] < 0:
            R[idx] = (
                np.int16(int(membership.membership_count[idx]))
                if membership is not None
                else np.int16(-1)
            )
        if r[idx] < 0:
            r[idx] = (
                np.int16(int(membership.smallest_ring_size[idx]))
                if membership is not None
                else np.int16(-1)
            )
        if x[idx] < 0:
            x[idx] = (
                np.int16(int(connectivity.ring_connectivity[idx]))
                if connectivity is not None
                else np.int16(-1)
            )

    return CMol(
        Z=Z,
        M=M,
        A=A,
        C=C,
        D=D,
        H=H,
        h=h,
        v=v,
        X=X,
        R=R,
        r=r,
        x=x,
        bonds=bonds,
        atom_map=atom_map,
    )


def gmol_from_cmol(molecule: CMol) -> GMol:
    def exact_or_wildcard(val: int) -> int:
        return 0xFF if val == -1 else bucket_mask_exact(val)

    atom_data: list[AtomData] = [
        {
            "Z": z_mask_for_numbers([int(molecule.Z[idx])]),
            "A": int(molecule.A[idx]),
            "C": mask_for_charge(int(molecule.C[idx])),
            "D": exact_or_wildcard(int(molecule.D[idx])),
            "H": exact_or_wildcard(int(molecule.H[idx])),
            "h": exact_or_wildcard(int(molecule.h[idx])),
            "v": exact_or_wildcard(int(molecule.v[idx])),
            "X": exact_or_wildcard(int(molecule.X[idx])),
            "R": exact_or_wildcard(int(molecule.R[idx])),
            "r": exact_or_wildcard(int(molecule.r[idx])),
            "x": exact_or_wildcard(int(molecule.x[idx])),
            "M": int(molecule.M[idx]),
        }
        for idx in range(molecule.atom_count)
    ]
    return GMol(
        atom_masks=atom_masks_from_props(atom_data),
        bonds=bonds_with_unknown_topology(molecule.bonds),
        atom_map=molecule.atom_map,
        # A molecule carries no grouping. Grouping is what a *query* says about which molecules its
        # parts come from, and this is already one molecule -- disconnected or not, it is the thing
        # being matched against rather than the thing making the claim.
        atom_group=np.zeros(molecule.atom_count, dtype=np.int32),
        atom_predicate=np.full(molecule.atom_count, -1, dtype=np.int32),
        predicates=(),
    )


def _cmol_from_exact_seed_data(
    *,
    atom_masks: AtomMasks,
    Z: AtomVector,
    A: Aromaticity,
    C: Charge,
    explicit_h: AtomVector,
    implicit_h: AtomVector,
    atom_map: AtomMapVector,
    bonds: BondMatrix,
    ring_descriptors: RingDescriptors = RingDescriptors.ALL,
) -> CMol:
    D = atom_masks.exact_property("D")
    H = atom_masks.exact_property("H")
    h = atom_masks.exact_property("h")
    v = atom_masks.exact_property("v")
    X = atom_masks.exact_property("X")
    R = atom_masks.exact_property("R")
    r = atom_masks.exact_property("r")
    x = atom_masks.exact_property("x")

    return cmol_from_exact_arrays(
        Z=Z,
        M=resolved_mass(atom_masks),
        A=A,
        C=C,
        D=D,
        H=H,
        h=h,
        v=v,
        X=X,
        R=R,
        r=r,
        x=x,
        explicit_h=explicit_h,
        implicit_h=implicit_h,
        atom_map=atom_map,
        bonds=bonds,
        ring_descriptors=ring_descriptors,
    )


def cmol_from_seed_data(
    *,
    Z: AtomVector,
    C: Charge,
    M: AtomVector | None = None,
    explicit_h: AtomVector,
    implicit_h: AtomVector,
    atom_map: AtomMapVector,
    bonds: BondMatrix,
    ring_descriptors: RingDescriptors = RingDescriptors.ALL,
    pinned_total_h: AtomVector | None = None,
) -> CMol:
    M = np.zeros_like(Z) if M is None else M
    perceived_A, bonds, perceived_total_h = perceive_aromaticity(
        Z=Z,
        C=C,
        explicit_h=explicit_h,
        implicit_h=implicit_h,
        bonds=bonds,
        pinned_total_h=pinned_total_h,
    )
    aromatic_atoms = perceived_A == AROMATIC

    atom_count = Z.shape[0]
    D = np.zeros(atom_count, dtype=np.int16)
    H = np.zeros(atom_count, dtype=np.int16)
    h = np.zeros(atom_count, dtype=np.int16)
    v = np.zeros(atom_count, dtype=np.int16)
    X = np.zeros(atom_count, dtype=np.int16)

    bond_graph = bond_presence(bonds)
    bond_orders = bond_order_matrix(bonds)
    ring_info = ring_info_from_bonds(bonds) if ring_descriptors else None
    membership = ring_info if ring_descriptors & RingDescriptors.MEMBERSHIP else None
    connectivity = (
        ring_info if ring_descriptors & RingDescriptors.CONNECTIVITY else None
    )

    for idx in range(atom_count):
        explicit = max(int(explicit_h[idx]), 0)
        symbol = Z_TO_ELEMENT.get(int(Z[idx]), "*")
        is_aromatic = bool(aromatic_atoms[idx])
        degree = int(bond_graph[idx].sum())
        order_sum = int(bond_orders[idx].sum())
        # Perception is the single source of truth for total H (it cannot be
        # recovered from the order-stripped canonical bonds). For an aromatic
        # heteroatom the hydrogens are stored as explicit ([nH]-style) so the
        # molecule round-trips through the identity seed; aromatic B/C store all
        # H as implicit (it is degree-derived, never explicit); aliphatic atoms
        # carry the remainder as implicit.
        total_h = int(perceived_total_h[idx])
        if is_aromatic and symbol not in {"B", "C"}:
            inferred_implicit = 0
        elif is_aromatic:
            # Aromatic B/C hydrogen is fully derived from the ring (3 - degree); it is
            # never an independent degree of freedom, so store all of it as implicit.
            # This keeps H - h == 0 for every aromatic carbon regardless of the seed's
            # explicit_h, so a later survivor carry (reaction.py) cannot re-inject an
            # explicit H that perception would then double-count and de-aromatize.
            inferred_implicit = total_h
        else:
            inferred_implicit = total_h - explicit
        D[idx] = np.int16(degree)
        H[idx] = np.int16(total_h)
        h[idx] = np.int16(inferred_implicit)
        v[idx] = np.int16(
            selected_valence(
                z=int(Z[idx]),
                charge=int(C[idx]),
                sigma_degree=degree,
                bond_order_sum=order_sum,
                total_h=total_h,
                delocalized=is_aromatic,
                atom=idx,
            )
        )
        X[idx] = np.int16(degree + total_h)

    return CMol(
        Z=Z,
        M=M,
        A=perceived_A,
        C=C,
        D=D,
        H=H,
        h=h,
        v=v,
        X=X,
        R=(
            np.asarray(membership.membership_count, dtype=np.int16)
            if membership is not None
            else np.full(atom_count, -1, dtype=np.int16)
        ),
        r=(
            np.asarray(membership.smallest_ring_size, dtype=np.int16)
            if membership is not None
            else np.full(atom_count, -1, dtype=np.int16)
        ),
        x=(
            np.asarray(connectivity.ring_connectivity, dtype=np.int16)
            if connectivity is not None
            else np.full(atom_count, -1, dtype=np.int16)
        ),
        bonds=bonds,
        atom_map=atom_map,
    )


def selected_valence(
    *,
    z: int,
    charge: int,
    sigma_degree: int,
    bond_order_sum: int,
    total_h: int,
    delocalized: bool,
    atom: int,
) -> int:
    """The graph valence of the configuration this atom's complete state occupies.

    Public ``v`` is read off the configuration rather than computed beside it, so an atom
    whose state no configuration allows is reported here instead of being handed a valence
    that merely adds up. Aromatic ions need no special case: the isoelectronic shift is
    already in the table, so pyridinium N+ selects valence 4 and tropylium C+ selects 3.
    """
    configuration = select_configuration(
        z=z,
        charge=charge,
        sigma_degree=sigma_degree,
        bond_order_sum=bond_order_sum,
        total_h=total_h,
        delocalized=delocalized,
    )
    if isinstance(configuration, ConfigurationFailure):
        symbol = Z_TO_ELEMENT.get(z, "*")
        raise ConfigurationError(
            f"Atom {atom} ({symbol}, charge {charge}) occupies no declared "
            f"configuration: {configuration.name.lower()}.",
            failure=configuration,
        )
    return configuration.graph_valence


__all__ = [
    "cmol_from_exact_arrays",
    "cmol_from_gmol",
    "cmol_from_normalized_smarts",
    "cmol_from_seed_data",
    "perceived_cmol",
    "cmol_from_smarts",
    "cmol_from_smiles",
    "gmol_from_cmol",
]
