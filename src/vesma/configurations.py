"""Closed-shell atom configurations.

An atom's chemistry is fixed by which closed-shell configuration it is in, and this
module is the one place that says which configurations exist. A configuration answers
both questions the rest of the model asks of an atom:

- how many hydrogens it carries, via ``sigma_capacity`` (heavy sigma neighbours plus
  hydrogens) and ``graph_valence`` (bond-order sum plus hydrogens);
- what it may contribute to a delocalized circuit, via ``pi``.

Selection is a lookup, never a search: an atom's local state either picks out exactly
one configuration, no configuration, or several. The last two are reported rather than
resolved, so nothing in the model ever chooses "the first valence that fits".

Main-group configurations are enumerated from two element facts -- the standard valence
and the outer-electron count -- rather than typed out per row, so the table cannot drift
from the rules it encodes. Metals are declared explicitly: they do not follow the
main-group pattern and their entries come from the reaction corpora that use them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import cache

import numpy as np

from .rejection import ChemistryRejection, RejectionReason
from .symbols import ELEMENT_TO_Z

__all__ = [
    "Configuration",
    "ConfigurationError",
    "ConfigurationFailure",
    "CONFIG_CHARGE_MAX",
    "CONFIG_CHARGE_MIN",
    "CONFIG_CHARGE_SPAN",
    "CONFIG_PI_BIAS",
    "CONFIG_Z_SPAN",
    "METAL_CONFIGURATIONS",
    "PiContribution",
    "configurations_for",
    "configuration_table",
    "select_circuit_configuration",
    "select_configuration",
    "unwritten_hydrogen_configuration",
    "unwritten_hydrogens",
]

# The packed device domain is exactly the domain expressible by AtomMask charge and
# element words. Both GPU backends consume this same generated table.
CONFIG_CHARGE_MIN = -3
CONFIG_CHARGE_MAX = 4
CONFIG_CHARGE_SPAN = CONFIG_CHARGE_MAX - CONFIG_CHARGE_MIN + 1
CONFIG_Z_SPAN = 128
CONFIG_PI_BIAS = 2


class PiContribution(IntEnum):
    """What an atom contributes to a delocalized circuit it belongs to."""

    INVALID = -2  # cannot belong to one at all
    AMBIGUOUS = -1  # belongs, but the count is not fixed by local state
    NONE = 0
    ONE = 1
    TWO = 2


class ConfigurationFailure(IntEnum):
    """Why an atom's local state did not pick out one configuration."""

    INCOMPATIBLE = 0  # configurations exist for this element and charge; none matches
    AMBIGUOUS = 1  # several do
    UNDECLARED = 2  # the element and charge have no configurations at all


class ConfigurationError(ChemistryRejection):
    """An atom's complete written state occupies no single declared configuration.

    Raised where the state was fully specified and simply is not one the model declares,
    which is a different thing from a count left unwritten: there is nothing to derive
    here, only a claim to reject.
    """

    def __init__(self, detail: str, *, failure: ConfigurationFailure) -> None:
        super().__init__(detail, reason=REASON_BY_CONFIGURATION_FAILURE[failure])


REASON_BY_CONFIGURATION_FAILURE = {
    ConfigurationFailure.INCOMPATIBLE: RejectionReason.CONFIGURATION_INCOMPATIBLE,
    ConfigurationFailure.AMBIGUOUS: RejectionReason.CONFIGURATION_AMBIGUOUS,
    ConfigurationFailure.UNDECLARED: RejectionReason.UNDECLARED_CONFIGURATION,
}


@dataclass(frozen=True)
class Configuration:
    """One closed-shell state an atom of this element and charge may occupy."""

    z: int
    charge: int
    sigma_capacity: int
    graph_valence: int
    pi: PiContribution


# Standard valence of the neutral element. A charge of q makes an atom share the
# valence-electron count of the element q places earlier, so its standard valence is
# that of Z - q: O- behaves like F, N+ like C, tropylium C+ like B. The relation is
# non-linear -- it peaks at carbon -- which is why a flat base+charge is wrong.
DEFAULT_VALENCE_BY_Z: dict[int, int] = {
    2: 0, 4: 2, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 10: 0,
    14: 4, 15: 3, 16: 2, 17: 1, 18: 0,
    32: 4, 33: 3, 34: 2, 35: 1, 36: 0,
    50: 4, 51: 3, 52: 2, 53: 1, 54: 0,
}  # fmt: skip
# The noble-gas rows above are not there to describe argon. They are what the
# isoelectronic shift lands on for a closed-shell anion: chloride reaches argon's count
# and so has valence 0, as do oxide and nitride. Without them a bare halide ion -- a
# counter-ion in a great many reactions -- would occupy no declared configuration.

# Valence-shell electrons, used to decide whether a fully sigma-bonded atom still holds
# a lone pair to donate. Absent for elements that never join a delocalized circuit.
OUTER_ELECTRONS_BY_Z: dict[int, int] = {
    5: 3, 6: 4, 7: 5, 8: 6,
    15: 5, 16: 6, 33: 5, 34: 6, 52: 6,
}  # fmt: skip

# Additional valences an element reaches by expanding its octet, beyond the standard one
# above. Iodine's rows are the hypervalent iodanes of synthesis: lambda-3 (PhI(OAc)2,
# PhICl2), lambda-5 (IBX, the Dess-Martin periodinane, iodyl chloride) and lambda-7
# (periodate). Chlorine and bromine expand in the same way (chlorate, perchlorate) but no
# corpus template bonds to such an atom covalently, so those rows stay undeclared until
# something needs them.
EXPANDED_VALENCES_BY_Z: dict[int, tuple[int, ...]] = {
    # Nitrogen is deliberately absent. A neutral graph-valence-5 nitrogen would admit
    # every local N(V) tuple as ordinary chemistry, and the states that motivate it are
    # not drawn that way: nitro is [N+](=O)[O-], azide is [N-]=[N+]=N, an N-oxide is
    # [N+][O-]. Charge-separated forms are already covered, because N+ is isoelectronic
    # with carbon and reaches valence 4 without any expansion. Measured: no atom in the
    # PaRoutes corpora occupies a neutral valence-5 nitrogen. The rare specially
    # stabilized hypercoordinate nitrogen does not justify opening the general case;
    # such a state is reported as undeclared rather than quietly accepted.
    15: (5,),
    16: (4, 6),
    33: (5,),
    34: (4, 6),
    53: (3, 5, 7),
}

# Whether an element can exceed its standard valence at all. This is a property of the
# atom itself rather than of its isoelectronic partner, so an O+ does not inherit
# nitrogen's higher valence however the charge shifts the electron count.
#
# The name is conventional, not mechanistic. Hypervalent bonding is not central-atom d
# orbital participation -- that account is obsolete -- but three-centre four-electron
# bonding with substantial ligand and nonbonding character, as IUPAC describes it. What
# this table encodes is only which elements are observed to reach these graph valences,
# which is a fact about the corpus and the literature rather than an orbital claim.
EXPANDS_OCTET = frozenset(EXPANDED_VALENCES_BY_Z)

# Metals do not follow the main-group pattern, so their configurations are declared
# from the reaction corpora that use them: Grignard, Stille, Negishi, Ullmann,
# organolithium, permanganate and osmium tetroxide fragments. Every one carries no
# hydrogen and never donates to a circuit. Bonds to a metal are authored template graph
# bonds; nothing here infers an oxidation state or a coordination geometry.
METAL_CONFIGURATIONS: dict[str, tuple[tuple[int, int, int], ...]] = {
    # An alkali metal gives up its one valence electron to a closed shell, so the bare
    # cation is its second configuration whether or not a corpus template happens to
    # write it. Potassium's pair came from template 21730, which carries both forms.
    "Li": ((0, 1, 1), (1, 0, 0)),
    "Na": ((0, 1, 1), (1, 0, 0)),
    "K": ((0, 1, 1), (1, 0, 0)),
    "Mg": ((0, 0, 0), (0, 2, 2), (1, 1, 1)),
    # The zero-coordinate entry is the metal itself, declared for the reason magnesium's is: a
    # Grignard names its turnings and a Negishi names its zinc dust, both reagents a search has to
    # source. Zinc lacked it only because no template had needed it, which is the same omission
    # aluminium had and not -- as this file was briefly read to say -- a statement that bare metals
    # are not molecules. `Mg: ((0, 0, 0), ...)` had said otherwise all along.
    "Zn": ((0, 0, 0), (0, 2, 2), (1, 1, 1), (2, 1, 1)),
    # Aluminium was *undeclared* rather than refused, which is a different thing: neutral Al fell
    # through to the main-group path, where three bonds is a sextet and so not an octet, and only
    # `[AlH4-]` survived because charge -1 with four bonds is one. Nothing had needed the neutral
    # form until the hydride reductions wanted to name what their aluminium becomes. The trivalent
    # entry is the Lewis-acidic sextet -- AlCl3, Al(OH)3 -- which is exactly the state the octet
    # model cannot hold and this table exists to declare; the four-coordinate anion is the
    # aluminate, and it is declared here because a metal in this table no longer consults the
    # main-group path at all.
    "Al": ((0, 3, 3), (-1, 4, 4)),
    "Cu": ((0, 1, 1), (0, 2, 2)),
    "Sn": ((0, 1, 1), (0, 2, 2), (0, 3, 3), (0, 4, 4)),
    "Mn": ((0, 4, 7),),
    "Os": ((0, 4, 8),),
}

METAL_Z = frozenset(ELEMENT_TO_Z[symbol] for symbol in METAL_CONFIGURATIONS)


def pi_contribution(
    *, pi_bonds: int, outer_electrons: int | None, sigma: int, charge: int
) -> PiContribution:
    """What an atom in this configuration brings to a circuit.

    A circuit member needs a perpendicular orbital, so its sigma frame may occupy at
    most three directions. Beyond that the atom is saturated and has none left. An atom
    with two or more pi bonds is an sp centre and cannot be a ring member; one pi bond
    puts a single electron into the circuit; none leaves the perpendicular orbital free,
    which donates a lone pair when one is left over and nothing when the orbital is
    empty. An odd leftover would be a radical, which is out of scope.
    """
    if outer_electrons is None:
        return PiContribution.INVALID
    if sigma > 3:
        return PiContribution.INVALID
    if pi_bonds >= 2:
        return PiContribution.INVALID
    if pi_bonds == 1:
        return PiContribution.ONE
    nonbonding = outer_electrons - charge - sigma
    if nonbonding >= 2:
        return PiContribution.TWO
    if nonbonding <= 0:
        return PiContribution.NONE
    return PiContribution.INVALID


@cache
def configurations_for(z: int, charge: int) -> tuple[Configuration, ...]:
    """Every closed-shell configuration an atom of this element and charge may occupy."""
    if z in METAL_Z:
        symbol = next(
            s
            for s, value in ELEMENT_TO_Z.items()
            if value == z and s in METAL_CONFIGURATIONS
        )
        return tuple(
            Configuration(
                z=z,
                charge=metal_charge,
                sigma_capacity=degree,
                graph_valence=valence,
                pi=PiContribution.INVALID,
            )
            for metal_charge, degree, valence in METAL_CONFIGURATIONS[symbol]
            if metal_charge == charge
        )

    # The isoelectronic shift governs the whole valence set, expanded rows included: a
    # sulfoxonium S+ shares phosphorus's electron count and so reaches phosphorus's
    # expanded valence 5, which is what the Corey-Chaykovsky reagent [Me3S=O]+ occupies.
    # Whether the shift is available at all is still the atom's own question, so an O+
    # does not inherit nitrogen's expansion.
    standard = DEFAULT_VALENCE_BY_Z.get(z - charge)
    if standard is None:
        return ()
    expanded = EXPANDED_VALENCES_BY_Z.get(z - charge, ()) if z in EXPANDS_OCTET else ()
    outer_electrons = OUTER_ELECTRONS_BY_Z.get(z)
    return tuple(
        Configuration(
            z=z,
            charge=charge,
            sigma_capacity=sigma,
            graph_valence=valence,
            # Octet expansion and ring delocalization are different regimes: a
            # hypervalent centre is pyramidal or higher-coordinate rather than
            # planar sp2, so an expanded valence never joins a circuit. A sulfoxide
            # sulfur has a pi bond and a lone pair, but not a ring p-orbital.
            pi=(
                pi_contribution(
                    pi_bonds=valence - sigma,
                    outer_electrons=outer_electrons,
                    sigma=sigma,
                    charge=charge,
                )
                if valence == standard
                else PiContribution.INVALID
            ),
        )
        for valence in (standard, *expanded)
        for sigma in range(valence + 1)
    )


@cache
def configuration_table() -> tuple[np.ndarray, np.ndarray]:
    """Pack the declared configurations for consumption by device kernels.

    The index maps ``(z, charge)`` to ``offset | (count << 16)`` in the row array.
    Each row stores sigma capacity, graph valence, and biased pi contribution in three
    bytes. Generating this from :func:`configurations_for` gives CPU, Metal, and CUDA a
    single chemistry source of truth.
    """
    index = np.zeros(CONFIG_Z_SPAN * CONFIG_CHARGE_SPAN, dtype=np.uint32)
    rows: list[int] = []
    for z in range(CONFIG_Z_SPAN):
        for charge in range(CONFIG_CHARGE_MIN, CONFIG_CHARGE_MAX + 1):
            configurations = configurations_for(z, charge)
            offset = len(rows)
            if offset > 0xFFFF or len(configurations) > 0xFFFF:
                raise RuntimeError("Configuration table outgrew its packed index.")
            rows.extend(
                int(configuration.sigma_capacity)
                | (int(configuration.graph_valence) << 8)
                | ((int(configuration.pi) + CONFIG_PI_BIAS) << 16)
                for configuration in configurations
            )
            slot = z * CONFIG_CHARGE_SPAN + (charge - CONFIG_CHARGE_MIN)
            index[slot] = offset | (len(configurations) << 16)
    return index, np.array(rows, dtype=np.uint32)


def select_configuration(
    *,
    z: int,
    charge: int,
    sigma_degree: int,
    bond_order_sum: int,
    total_h: int,
    delocalized: bool,
) -> Configuration | ConfigurationFailure:
    """The one configuration an atom's local state occupies.

    A delocalized atom has no bond orders to read -- that is what delocalization means --
    so its sigma frame constrains it, together with the fact that it demonstrably belongs
    to a circuit: a configuration that cannot join one is not a candidate for an atom
    that already has. A concrete atom is constrained instead by its sigma frame and its
    bond-order sum, which is what separates an sp3 centre from an sp2 one of the same
    element.
    """
    sigma = sigma_degree + total_h
    consistent = [
        configuration
        for configuration in configurations_for(z, charge)
        if configuration.sigma_capacity == sigma
        and (
            configuration.pi != PiContribution.INVALID
            if delocalized
            else configuration.graph_valence == bond_order_sum + total_h
        )
    ]
    if not consistent:
        # An element and charge the table says nothing about is a different failure from
        # one whose declared states simply do not fit: the first is a gap in the model,
        # the second is a claim about the molecule.
        return (
            ConfigurationFailure.INCOMPATIBLE
            if configurations_for(z, charge)
            else ConfigurationFailure.UNDECLARED
        )
    if len(consistent) > 1:
        return ConfigurationFailure.AMBIGUOUS
    return consistent[0]


def select_circuit_configuration(
    *,
    z: int,
    charge: int,
    sigma_degree: int,
    total_h: int,
) -> Configuration | ConfigurationFailure:
    """The configuration an atom occupies when asked whether it can join a pi circuit.

    Deliberately blind to bond orders. Concrete orders are a *spelling*: the same ring
    written delocalized has none at all, so letting them select the configuration makes
    perception depend on notation -- which is the one thing this model exists to prevent.
    What constrains a circuit member is its sigma frame and the requirement that it have
    a defined pi contribution at all.

    This is well posed rather than merely convenient. Across the whole table, once the
    configurations that cannot join a circuit are excluded, every (element, charge, sigma
    frame) admits exactly one pi contribution -- 71 frames, no exceptions. Bond order is
    still needed to tell a sulfide from a sulfoxide from a sulfone, but never to decide
    what an atom brings to a circuit, because those hypervalent states cannot join one.
    """
    sigma = sigma_degree + total_h
    consistent = [
        configuration
        for configuration in configurations_for(z, charge)
        if configuration.sigma_capacity == sigma
        and configuration.pi != PiContribution.INVALID
    ]
    if not consistent:
        return (
            ConfigurationFailure.INCOMPATIBLE
            if configurations_for(z, charge)
            else ConfigurationFailure.UNDECLARED
        )
    if len(consistent) > 1:
        return ConfigurationFailure.AMBIGUOUS
    return consistent[0]


def unwritten_hydrogen_configuration(
    *,
    z: int,
    charge: int,
    sigma_degree: int,
    bond_order_sum: int,
    delocalized: bool,
) -> Configuration | ConfigurationFailure:
    """The configuration of an atom whose hydrogen count was never written.

    An omitted hydrogen is not a free variable to solve for. It means the fewest hydrogens
    the atom's declared configurations allow given the bonding that *was* written: a bare
    ``N`` is an amine, a bare ``S`` on two single bonds is a thioether, and a lowercase
    ``n`` with two ring neighbours is pyridine-like rather than pyrrole-like. That is what
    SMILES itself means by those atoms, so a molecule or template gets what its author
    wrote rather than what a search found.

    The consequence worth naming is that octet expansion must be evidenced by bonding.
    Sulfone sulfur reaches valence 4 because four bond orders are written; phosphonic
    phosphorus reaches 5 because four written orders leave no lower declared valence; but
    neither is ever reached by adding hydrogens, so ``[C:1]-[P]`` is a phosphine and not a
    phosphorane.

    This is the only place in the model that decides a hydrogen from written state alone,
    and it runs only where there is no configuration to copy -- a molecule literal, and an
    atom a template adds. A surviving mapped atom copies its configuration instead (see
    ``hydrogens_after_edit``), so no edit can re-select an atom into a different oxidation
    state behind the caller's back.
    """
    reachable: list[tuple[int, Configuration]] = []
    for configuration in configurations_for(z, charge):
        if delocalized:
            # The atom demonstrably belongs to a circuit, so a configuration that cannot
            # join one is not a candidate, and its sigma frame is what constrains it --
            # a delocalized bond carries no order to subtract.
            if configuration.pi == PiContribution.INVALID:
                continue
            hydrogens = configuration.sigma_capacity - sigma_degree
        else:
            hydrogens = configuration.graph_valence - bond_order_sum
            if configuration.sigma_capacity != sigma_degree + hydrogens:
                continue
        if hydrogens >= 0:
            reachable.append((hydrogens, configuration))
    if not reachable:
        return (
            ConfigurationFailure.INCOMPATIBLE
            if configurations_for(z, charge)
            else ConfigurationFailure.UNDECLARED
        )
    fewest = min(hydrogens for hydrogens, _ in reachable)
    chosen = [
        configuration for hydrogens, configuration in reachable if hydrogens == fewest
    ]
    if len(chosen) > 1:
        return ConfigurationFailure.AMBIGUOUS
    return chosen[0]


def unwritten_hydrogens(
    configuration: Configuration,
    *,
    sigma_degree: int,
    bond_order_sum: int,
    delocalized: bool,
) -> int:
    """The hydrogen count implied by a configuration and the bonding that was written."""
    return (
        configuration.sigma_capacity - sigma_degree
        if delocalized
        else configuration.graph_valence - bond_order_sum
    )
