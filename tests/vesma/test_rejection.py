"""The closed set of reasons the model refuses to build a molecule."""

from __future__ import annotations

import numpy as np
import pytest

from tests.vesma.bench_precursor_audit import audit_smiles
from vesma.bonds import BondKind
from vesma.cmol_builder import cmol_from_seed_data, cmol_from_smiles, gmol_from_cmol
from vesma.cpu_matcher import CpuMatcher, CpuReactionMatcher, generate_precursors
from vesma.interop import mol_from_cmol
from vesma.reaction import parse_reaction_smarts, precursors_from_mapping_result
from vesma.rejection import (
    ChemistryRejection,
    RejectionReason,
    by_name,
    counted,
    merged,
    no_rejections,
)


def molecule(smiles: str):
    return cmol_from_smiles(smiles)[0]


def first_application(smiles: str, smarts: str):
    """One application of a template, with the diagnostics it produced.

    ``generate_precursors`` returns outcomes alone, so the annotation counts a successful
    application carries are only reachable one mapping at a time.
    """
    product = molecule(smiles)
    reaction = parse_reaction_smarts(smarts)
    matches = CpuMatcher().find_matches(reaction.lhs, gmol_from_cmol(product))
    assert matches.assignments.size, f"{smarts} does not match {smiles}"
    return precursors_from_mapping_result(
        product=product, reaction=reaction, mapping=matches.assignments[0]
    )


def written(result) -> list[str]:
    return [audit_smiles(mol_from_cmol(precursor)) for precursor in result.precursors]


def test_counts_partition_the_rejected_applications() -> None:
    """Counts are per reason and merge elementwise, so a run's totals add up.

    A tuple rather than a mapping, because merging two chunks is then plain addition and
    the layout is the array of counters a device kernel keeps.
    """
    left = counted(RejectionReason.DELOCALIZED_EDGE_NOT_CYCLIC)
    right = counted(RejectionReason.CONFIGURATION_INCOMPATIBLE)

    both = merged(left, right)

    assert sum(both) == 2
    assert by_name(both) == {
        "configuration_incompatible": 1,
        "delocalized_edge_not_cyclic": 1,
    }
    assert merged(no_rejections(), both) == both


def test_every_reason_has_a_distinct_slot() -> None:
    assert len(no_rejections()) == len(RejectionReason)
    assert len({reason.value for reason in RejectionReason}) == len(RejectionReason)


@pytest.mark.parametrize(
    "smarts, reason, role",
    [
        (
            "[C:1]>>[N:1]",
            RejectionReason.NUCLEAR_CONSERVATION_VIOLATED,
            "a mapped atom may assert its nucleus but never transmute it",
        ),
        (
            "[C:1]>>[C:1]-[B+2]",
            RejectionReason.UNDECLARED_CONFIGURATION,
            "an added atom whose element and charge the table declares nothing for",
        ),
    ],
)
def test_a_refusal_names_its_reason(
    smarts: str, reason: RejectionReason, role: str
) -> None:
    with pytest.raises(ChemistryRejection) as raised:
        generate_precursors(molecule("CCO"), parse_reaction_smarts(smarts))

    assert raised.value.reason is reason, role


def test_an_orphan_delocalized_bond_names_its_reason() -> None:
    """An acyclic delocalized edge. A delocalized *ring* that fails Huckel is a different
    reason, because it is a circuit -- just not a delocalized one."""
    n = 2
    bonds = np.zeros((n, n), dtype=np.uint8)
    bonds[0, 1] = bonds[1, 0] = np.uint8(BondKind.DELOCALIZED)

    with pytest.raises(ChemistryRejection) as raised:
        cmol_from_seed_data(
            Z=np.full(n, 6, dtype=np.int16),
            C=np.zeros(n, dtype=np.int8),
            explicit_h=np.zeros(n, dtype=np.int16),
            implicit_h=np.full(n, -1, dtype=np.int16),
            atom_map=np.zeros(n, dtype=np.int32),
            bonds=bonds,
        )

    assert raised.value.reason is RejectionReason.DELOCALIZED_EDGE_NOT_CYCLIC


def test_a_cyclic_delocalized_system_that_fails_huckel_names_its_own_reason() -> None:
    """Cyclooctatetraene is a circuit whose electron count refuses it, not an orphan.

    Reporting it as `delocalized_edge_not_cyclic` stated the opposite of what is true, and
    an audit whose whole output is a cause breakdown cannot afford that.
    """
    from vesma.aromaticity import DelocalizedSystemInvalidError

    with pytest.raises(DelocalizedSystemInvalidError) as raised:
        molecule("c1ccccccc1")

    assert raised.value.reason is RejectionReason.DELOCALIZED_SYSTEM_INVALID


def test_an_atom_that_cannot_join_a_circuit_names_the_configuration() -> None:
    """`[cH0]` benzene is a configuration failure, not an orphan bond.

    The ring is plainly cyclic; what fails is that a carbon with two sigma neighbours and
    no hydrogen occupies no configuration able to belong to a circuit.
    """
    from vesma.configurations import ConfigurationError

    n = 6
    bonds = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        j = (i + 1) % n
        bonds[i, j] = bonds[j, i] = np.uint8(BondKind.DELOCALIZED)

    with pytest.raises(ConfigurationError) as raised:
        cmol_from_seed_data(
            Z=np.full(n, 6, dtype=np.int16),
            C=np.zeros(n, dtype=np.int8),
            explicit_h=np.zeros(n, dtype=np.int16),
            implicit_h=np.zeros(n, dtype=np.int16),
            atom_map=np.zeros(n, dtype=np.int32),
            bonds=bonds,
        )

    assert raised.value.reason is RejectionReason.CONFIGURATION_INCOMPATIBLE


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_a_batch_counts_refusals_by_reason_and_keeps_the_rest(n_jobs: int) -> None:
    """One refused application costs only itself, and is counted under its own name.

    The serial and forked paths have to agree exactly: they are the same run split
    differently, so a difference here would mean the rejection bookkeeping depends on
    how the work was chunked.
    """
    molecules = [molecule("CCC"), molecule("c1ccccc1")]
    patterns = (
        parse_reaction_smarts("[C:1][C:2]>>[C:1].[C:2]"),
        parse_reaction_smarts("[c:1][c:2]>>[c:1].[c:2]"),
        parse_reaction_smarts("[C:1]>>[N:1]"),
    )
    matcher = CpuReactionMatcher(patterns)

    precursors, _ = matcher.match(molecules, n_jobs=n_jobs)
    diagnostics = matcher.last_diagnostics

    assert len(precursors) == 8
    assert diagnostics.rejected_applications == sum(diagnostics.rejections)
    # Three transmutations (propane's carbons; benzene's are aromatic, so the aliphatic
    # `[C:1]` does not match them) and twelve ring openings that release delocalized
    # bonds the template never authors.
    assert diagnostics.rejections_by_reason == {
        "nuclear_conservation_violated": 3,
        "delocalized_edge_not_cyclic": 12,
    }


@pytest.mark.parametrize(
    "smarts, mismatch, role",
    [
        ("[C:1][O:2]>>[C:1].[O:2]", {}, "writes no derived property"),
        ("[C:1][O:2]>>[C;D1:1].[O:2]", {}, "writes the degree the edit delivers"),
        ("[C:1][O:2]>>[C;D3:1].[O:2]", {"D": 1}, "writes a degree the edit will not"),
        ("[C:1][O:2]>>[C;X4:1].[O:2]", {}, "writes the connectivity delivered"),
        ("[C:1][O:2]>>[C;X9:1].[O:2]", {"X": 1}, "writes impossible connectivity"),
    ],
)
def test_a_derived_rhs_property_is_recomputed_not_obeyed(
    smarts: str, mismatch: dict[str, int], role: str
) -> None:
    """An RHS is a rewrite program, not an assertion set.

    Its operative vocabulary is atom presence, nucleus, charge, total hydrogen and bond
    kind. ``D`` and ``X`` are consequences of applying that program, so vesma computes
    them from the result and records where the template's spelling disagreed -- it
    neither obeys the spelling nor refuses the application over it.

    The reason is provenance. These fields arrive from whichever extractor emitted the
    templates, and enforcing them would make vesma's output a function of that
    extractor's notation conventions rather than of the chemistry.
    """
    result = first_application("CCO", smarts)

    assert result.precursors, role
    assert result.diagnostics.annotation_mismatches_by_property == mismatch, role


@pytest.mark.parametrize(
    "template, product, expected, role",
    [
        ("[c:1]>>[C:1]", "c1ccccc1", "c1ccccc1", "rewrites no bond, so benzene stands"),
        ("[C:1]>>[c:1]", "CC", "CC", "rewrites no bond, so ethane stands"),
    ],
)
def test_aromaticity_follows_the_graph_rather_than_the_rhs_atom_case(
    template: str, product: str, expected: str, role: str
) -> None:
    """Aromaticity is a consequence of the resulting electronic graph, not a writable bit.

    Neither template changes a bond, a hydrogen count or a charge, so neither can change
    what its atom perceives as. Reading the case as an instruction would let a template
    declare benzene aliphatic; reading it as an assertion would refuse an application
    that did exactly what it was told. Both were tried. The rewrite program says the
    result is unchanged, and perception agrees.
    """
    result = first_application(product, template)

    assert written(result) == [expected], role
    assert result.diagnostics.annotation_mismatches_by_property == {"A": 1}, role


def test_rhs_atom_case_still_resolves_an_omitted_bond() -> None:
    """Not obeying a property is not the same as discarding it.

    An omitted RHS bond between two lower-case atoms resolves to delocalized, and between
    two upper-case atoms to single. That is construction, not assertion: drop the case
    there and every rebuilt aromatic ring quietly becomes a ring of single bonds.
    """
    kept = first_application("c1ccccc1O", "[c:1][c:2][OH:3]>>[c:1][c:2][OH:3]")

    assert written(kept) == ["Oc1ccccc1"]


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_hydrogen_completion_is_counted_as_a_successful_operation(n_jobs: int) -> None:
    """Neutral telemetry, not a failure count.

    Completion is the one hydrogen decision that is neither copied verbatim from the
    matched atom nor written by the template, so how often the model has to make it is
    worth reporting on its own. It merges across chunks the same way rejections do, so
    the figure cannot depend on how the work was split.
    """
    molecules = [molecule("CCO"), molecule("CCC")]
    rules = (
        parse_reaction_smarts("[C:1][O:2]>>[C:1].[O:2]"),
        parse_reaction_smarts("[C:1][C:2]>>[C:1].[C:2]"),
    )
    matcher = CpuReactionMatcher(rules)

    matcher.match(molecules, n_jobs=n_jobs)
    diagnostics = matcher.last_diagnostics

    assert diagnostics.hydrogen_completion_atoms > 0
    assert diagnostics.hydrogen_completion_applications > 0
    assert (
        diagnostics.hydrogen_completion_applications
        <= diagnostics.hydrogen_completion_atoms
    )


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_a_pair_can_contain_both_accepted_and_refused_applications(n_jobs: int) -> None:
    """Provenance is per application, because a pair is not uniformly one or the other.

    Ethylbenzene against `[*:1]~[*:2]>>[*:1].[*:2]` breaks each bond in turn: the two
    aliphatic bonds give precursors, the six aromatic ones release delocalized bonds the
    template never authors and are refused. That is one (molecule, rule) pair carrying
    both outcomes, so a schema with a single status and reason per pair cannot describe
    it -- refusals are recorded as rows keyed by app_idx, the same shape as a hit.
    """
    from vesma.rejection import RejectionReason

    product = molecule("CCc1ccccc1")
    matcher = CpuReactionMatcher((parse_reaction_smarts("[*:1]~[*:2]>>[*:1].[*:2]"),))

    _, hits = matcher.match([product], n_jobs=n_jobs)
    refused = matcher.last_rejected_applications

    accepted = {(int(h[0]), int(h[1]), int(h[2])) for h in hits}
    assert accepted, "the aliphatic disconnections should survive"
    assert refused.shape[0] > 0, "the aromatic ones should not"
    assert refused.shape[1] == 4

    # Every refusal names its own application and its own reason.
    refused_apps = {(int(r[0]), int(r[1]), int(r[2])) for r in refused}
    assert refused_apps.isdisjoint(accepted)
    assert {RejectionReason(int(r[3])) for r in refused} == {
        RejectionReason.DELOCALIZED_EDGE_NOT_CYCLIC
    }
    # The rows and the aggregate counters are two views of one fact.
    assert refused.shape[0] == matcher.last_diagnostics.rejected_applications
