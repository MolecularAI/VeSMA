"""The single hydrogen-count model shared by aromaticity perception and WL identity.

Both perception (``perceive_aromaticity`` in :mod:`vesma.aromaticity`) and molecule
identity (``total_hydrogens_for_identity`` in :mod:`vesma.identity`) need the same
per-atom total hydrogen count, derived so that Kekule and aromatic encodings of the
same molecule agree atom-for-atom. Keeping one model here means the two callers cannot
drift apart -- the divergence that let charged aromatic ring carbons carry a different H
count in identity than in perception.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from .configurations import (
    METAL_Z,
    Configuration,
    ConfigurationFailure,
    select_configuration,
    unwritten_hydrogen_configuration,
    unwritten_hydrogens,
)
from .gmol_atoms import AtomVector, Charge
from .rejection import ChemistryRejection, RejectionReason
from .symbols import Z_TO_ELEMENT

__all__ = [
    "HYDROGEN_UNPINNED",
    "HydrogenFailure",
    "UnwrittenHydrogenError",
    "complete_hydrogens",
    "hydrogens_after_edit",
    "hydrogens_for_written_state",
    "total_hydrogens",
]

# A total-hydrogen seed that the caller has not fixed, and which perception resolves.
HYDROGEN_UNPINNED = -1


class HydrogenFailure(IntEnum):
    """Why an edited atom's hydrogen count could not be determined."""

    CHARGE_CHANGED_WITH_H_OMITTED = 0
    NEGATIVE = 1
    INCOMPATIBLE_CONFIGURATION = 2
    AMBIGUOUS_CONFIGURATION = 3
    UNDECLARED_CONFIGURATION = 4


# Total by construction, so a configuration failure added later cannot fall through a
# chain of member tests and be handed onward as if it were a configuration.
HYDROGEN_FAILURE_BY_CONFIGURATION_FAILURE = {
    ConfigurationFailure.INCOMPATIBLE: 2,
    ConfigurationFailure.AMBIGUOUS: 3,
    ConfigurationFailure.UNDECLARED: 4,
}


def complete_hydrogens(
    configuration: Configuration,
    *,
    bond_order_sum: int,
    sigma_degree: int,
    delocalized: bool,
) -> int | HydrogenFailure:
    """The hydrogen count an atom must carry to stay in its configuration.

    The configuration is copied from the matched atom, so there is no choice among
    valence options here and nothing to search: the edited local graph plus one declared
    configuration determine exactly one count. A delocalized atom is measured against
    its sigma capacity, since a delocalized bond carries no order to subtract.
    """
    if configuration.z in METAL_Z:
        return 0
    completed = (
        configuration.sigma_capacity - sigma_degree
        if delocalized
        else configuration.graph_valence - bond_order_sum
    )
    if completed < 0:
        return HydrogenFailure.NEGATIVE
    return completed


def hydrogens_for_written_state(
    *,
    z: int,
    charge: int,
    sigma_degree: int,
    bond_order_sum: int,
    delocalized: bool,
) -> int | HydrogenFailure:
    """Hydrogen for an atom that has no configuration to copy, from its written state.

    An atom a template adds exists only as what the RHS wrote, so its hydrogen comes from
    ``unwritten_hydrogen_configuration``: the fewest its declared configurations allow
    given that bonding. This is the same law molecule literals use, so ``[C:1]-[P]`` and
    the literal ``P`` in a SMILES agree on being a phosphine.
    """
    configuration = unwritten_hydrogen_configuration(
        z=z,
        charge=charge,
        sigma_degree=sigma_degree,
        bond_order_sum=bond_order_sum,
        delocalized=delocalized,
    )
    if isinstance(configuration, ConfigurationFailure):
        return HydrogenFailure(HYDROGEN_FAILURE_BY_CONFIGURATION_FAILURE[configuration])
    return unwritten_hydrogens(
        configuration,
        sigma_degree=sigma_degree,
        bond_order_sum=bond_order_sum,
        delocalized=delocalized,
    )


def hydrogens_after_edit(
    *,
    z: int,
    charge_before: int,
    charge_after: int,
    sigma_degree_before: int,
    bond_order_sum_before: int,
    total_h_before: int,
    delocalized_before: bool,
    sigma_degree_after: int,
    bond_order_sum_after: int,
    delocalized_after: bool,
    bonding_changed: bool,
) -> int | HydrogenFailure:
    """Hydrogen on a surviving atom whose RHS did not assign one.

    An atom whose charge and incident bonds are both untouched keeps its hydrogen
    exactly; splitting a molecule into components is not a bond change, so context
    atoms come through unaltered. An atom whose bonding changed is completed from the
    configuration it was already in. A charge change with the hydrogen left unwritten is
    rejected rather than completed: configurations are keyed by charge, so the matched
    atom's configuration says nothing about the atom after the change, and inventing one
    would be a search for a target state.
    """
    if charge_after != charge_before:
        return HydrogenFailure.CHARGE_CHANGED_WITH_H_OMITTED
    if not bonding_changed:
        return total_h_before
    configuration = select_configuration(
        z=z,
        charge=charge_before,
        sigma_degree=sigma_degree_before,
        bond_order_sum=bond_order_sum_before,
        total_h=total_h_before,
        delocalized=delocalized_before,
    )
    if isinstance(configuration, ConfigurationFailure):
        return HydrogenFailure(HYDROGEN_FAILURE_BY_CONFIGURATION_FAILURE[configuration])
    return complete_hydrogens(
        configuration,
        bond_order_sum=bond_order_sum_after,
        sigma_degree=sigma_degree_after,
        delocalized=delocalized_after,
    )


class UnwrittenHydrogenError(ChemistryRejection):
    """An atom left its hydrogen count unwritten and no configuration supplies one.

    The atom is in a state the model does not declare, so there is nothing to complete it
    to. Reporting that is the point: silently handing back zero hydrogens would make up a
    molecule the input never described.
    """

    def __init__(self, detail: str, *, failure: HydrogenFailure) -> None:
        super().__init__(detail, reason=REASON_BY_HYDROGEN_FAILURE[failure])


REASON_BY_HYDROGEN_FAILURE = {
    HydrogenFailure.CHARGE_CHANGED_WITH_H_OMITTED: (
        RejectionReason.HYDROGEN_OMITTED_WITH_CHARGE_CHANGE
    ),
    HydrogenFailure.NEGATIVE: RejectionReason.HYDROGEN_COMPLETION_NEGATIVE,
    HydrogenFailure.INCOMPATIBLE_CONFIGURATION: (
        RejectionReason.HYDROGEN_COMPLETION_INCOMPATIBLE
    ),
    HydrogenFailure.AMBIGUOUS_CONFIGURATION: (
        RejectionReason.HYDROGEN_COMPLETION_AMBIGUOUS
    ),
    HydrogenFailure.UNDECLARED_CONFIGURATION: RejectionReason.UNDECLARED_CONFIGURATION,
}


def total_hydrogens(
    *,
    Z: AtomVector,
    C: Charge,
    explicit_h: AtomVector,
    implicit_h: AtomVector,
    order_sums: AtomVector,
    heavy_degrees: AtomVector,
    bond_orders_omitted: np.ndarray[tuple[int], np.dtype[np.bool_]],
    pinned_total_h: AtomVector | None = None,
) -> AtomVector:
    """Total hydrogens, from three sources in decreasing authority.

    A count the caller has already decided is pinned and taken verbatim -- reaction
    editing pins every atom it produces, so a precursor's hydrogen comes from the
    hydrogen law and is never re-derived here. A count the input wrote is taken exactly,
    because it is molecular state rather than a hint. Only a count nobody wrote is
    derived, by the same law molecule literals and added atoms use.

    Kekule and delocalized encodings of one molecule still agree atom-for-atom, since a
    delocalized atom is measured against its sigma capacity and a concrete one against
    its bond orders, and both land on the same configuration."""
    result = np.zeros(Z.shape[0], dtype=np.int16)
    for atom in range(Z.shape[0]):
        # A caller that already knows the count -- reaction editing, where the hydrogen
        # law decided it -- pins it, and perception does not get to re-derive it.
        if (
            pinned_total_h is not None
            and int(pinned_total_h[atom]) != HYDROGEN_UNPINNED
        ):
            result[atom] = np.int16(pinned_total_h[atom])
            continue
        explicit = max(int(explicit_h[atom]), 0)
        implicit = int(implicit_h[atom])
        if implicit < 0 and int(C[atom]) != 0:
            # A formal charge can only be written inside brackets, and a bracket atom
            # carries no implicit hydrogen -- that is what the notation means. So a
            # charged atom that wrote no count has none: `[O-]` is an oxide and not a
            # hydroxide.
            #
            # Text read as SMILES no longer needs this: `vesma.parsing.reading` records
            # which atoms were bracketed and writes the exact `H0` upstream, so every
            # bracket atom arrives here with a count and the uncharged case is covered
            # too. It still guards seeds from producers that do not go through that
            # reading -- notably `cmol_from_smarts`, which concretizes a *query* and so
            # has no bracket to consult.
            implicit = 0
        if implicit >= 0:
            # The count was written. It is molecular state, not a hint, so perception
            # takes it exactly -- including where it disagrees with what perception would
            # have derived. `[cH0]1ccccc1` is a different claim from `c1ccccc1` and must
            # be answered as such rather than quietly rewritten into it.
            result[atom] = np.int16(explicit + implicit)
            continue
        resolved = hydrogens_for_written_state(
            z=int(Z[atom]),
            charge=int(C[atom]),
            sigma_degree=int(heavy_degrees[atom]),
            bond_order_sum=int(order_sums[atom]),
            # Syntax provenance, not perceived chemistry: it selects which *decoder*
            # reads an omitted hydrogen count, because a localized spelling supplies
            # bond orders to reconstruct from and a delocalized one deliberately does
            # not. Two decoders for two encodings of one state; that they agree exactly
            # is checked over the corpus against randomized localized spellings.
            delocalized=bool(bond_orders_omitted[atom]),
        )
        if isinstance(resolved, HydrogenFailure):
            symbol = Z_TO_ELEMENT.get(int(Z[atom]), "*")
            raise UnwrittenHydrogenError(
                f"Atom {atom} ({symbol}) has no determined hydrogen count: "
                f"{resolved.name.lower()}.",
                failure=resolved,
            )
        result[atom] = np.int16(resolved)
    return result
