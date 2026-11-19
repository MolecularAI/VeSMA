from __future__ import annotations

import warnings
from unittest.mock import Mock

import numpy as np
import pytest

from vesma import (
    GPU_PARITY_REACTION_PROCESSING,
    MoleculeProcessing,
    ReactionProcessing,
    SmartsError,
    cmol_from_smiles,
    gmol_from_cmol,
)
from vesma.cmol import cmol_union
from vesma.cpu_matcher import CpuReactionMatcher, generate_precursors, has_match
from vesma.gmol_atoms import AROMATIC
from vesma.interop import mol_from_cmol
from vesma.keyer import rdkit_canonical_smiles
from vesma.reaction import (
    UnsupportedReactionError,
    parse_reaction_smarts,
    precursors_from_mapping,
)
from vesma.rings import RingDescriptors


def _molecule(smiles: str):
    return cmol_from_smiles(smiles)[0]


def test_parse_reaction_smarts_extracts_mapped_both_sides() -> None:
    reaction = parse_reaction_smarts("[C:1][O:2]>>[C:1].[O:2]")
    assert reaction.edit.mapped_rhs_map.tolist() == [1, 2]


def test_parse_reaction_smarts_builds_minimal_edit_plan() -> None:
    reaction = parse_reaction_smarts("[C:1][O][C:2]>>[C:1]-[C:2].[N:3]")
    edit = reaction.edit

    assert edit.mapped_rhs_map.tolist() == [1, 2]
    assert edit.deleted_lhs_indices.tolist() == [1]
    assert edit.removed_lhs_edges.tolist() == [[0, 1], [1, 2]]
    assert edit.added_rhs_edges.tolist() == [[0, 1]]


def test_parse_reaction_smarts_allows_rhs_only_mapped_atoms() -> None:
    reaction = parse_reaction_smarts("[C:1]>>[C:1]-[O:2]")
    assert reaction.edit.mapped_rhs_map.tolist() == [1]
    assert int(reaction.rhs.atom_map[1]) == 2


def test_parse_reaction_smarts_allows_no_common_maps() -> None:
    reaction = parse_reaction_smarts("C>>O")

    assert reaction.edit.mapped_rhs_map.tolist() == []
    assert reaction.edit.deleted_lhs_indices.tolist() == [0]


def test_parse_reaction_smarts_allows_disjoint_positive_maps() -> None:
    reaction = parse_reaction_smarts("[C:1]>>[O:2]")

    assert reaction.edit.mapped_rhs_map.tolist() == []
    assert reaction.edit.deleted_lhs_indices.tolist() == [0]


def test_parse_reaction_smarts_treats_zero_map_as_unmapped() -> None:
    reaction = parse_reaction_smarts("[C:0]>>[O:0]")

    assert reaction.lhs.atom_map.tolist() == [0]
    assert reaction.rhs.atom_map.tolist() == [0]
    assert reaction.edit.mapped_rhs_map.tolist() == []
    assert reaction.edit.deleted_lhs_indices.tolist() == [0]


def test_parse_reaction_smarts_negative_map_bubbles_as_smarts_error() -> None:
    with pytest.raises(SmartsError):
        parse_reaction_smarts("[C:-1]>>O")


def test_parse_reaction_smarts_rejects_duplicate_positive_maps() -> None:
    with pytest.raises(UnsupportedReactionError, match="Duplicate atom map number: 1"):
        parse_reaction_smarts("[C:1][O:1]>>[C:1]")


def test_parse_reaction_smarts_requires_three_reaction_fields() -> None:
    with pytest.raises(UnsupportedReactionError, match="lhs>agents>rhs"):
        parse_reaction_smarts("[C:1]>[O:1]")

    with pytest.raises(UnsupportedReactionError, match="lhs>agents>rhs"):
        parse_reaction_smarts("[C:1]>>>[O:1]")


def test_generate_precursors_splits_ester_like_bond() -> None:
    product = _molecule("CC(=O)OC")
    reaction = parse_reaction_smarts(
        "[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])-[O:3].[C:4]"
    )

    outcomes = generate_precursors(product, reaction)
    assert outcomes
    sizes = sorted(prec.atom_count for prec in outcomes[0].precursors)
    assert sizes == [1, 4]


def test_cpu_reaction_matcher_parallel_matches_serial() -> None:
    product = _molecule("CCC")
    rules = (parse_reaction_smarts("[C:1][C:2]>>[C:1].[C:2]"),)
    matcher = CpuReactionMatcher(rules)

    serial_precursors, serial_hits = matcher.match([product], n_jobs=1)
    parallel_precursors, parallel_hits = matcher.match([product], n_jobs=2)

    assert np.array_equal(parallel_hits, serial_hits)
    assert [precursor.atom_count for precursor in parallel_precursors] == [
        precursor.atom_count for precursor in serial_precursors
    ]


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_cpu_reaction_processing_can_omit_post_split_ring_membership(
    n_jobs: int,
) -> None:
    """A caller can request only connectivity while leaving R/r unperceived."""
    product = _molecule("c1ccccc1OC")
    reaction = parse_reaction_smarts("[c:1][O:2][C:3]>>[c:1][O:2].[C:3]")
    matcher = CpuReactionMatcher(
        (reaction,),
        processing=ReactionProcessing(
            after_split=MoleculeProcessing.AROMATICITY
            | MoleculeProcessing.RING_CONNECTIVITY
        ),
    )

    precursors, _ = matcher.match([product], n_jobs=n_jobs)

    ring = next(precursor for precursor in precursors if precursor.atom_count > 1)
    assert AROMATIC in ring.A
    assert ring.R.tolist() == [-1] * ring.atom_count
    assert ring.r.tolist() == [-1] * ring.atom_count
    # Phenol: six ring carbons with two ring bonds each, and a hydroxyl oxygen with none.
    assert sorted(ring.x.tolist()) == [0] + [2] * 6


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("smiles", ["c1ccccc1OC", "C12C3C4C1C5C2C3C45"])
def test_gpu_parity_processing_perceives_relevant_rings(
    n_jobs: int, smiles: str
) -> None:
    product = _molecule(smiles)
    matcher = CpuReactionMatcher(
        [parse_reaction_smarts("[*:1]>>[*:1]")],
        processing=GPU_PARITY_REACTION_PROCESSING,
    )
    precursors, _ = matcher.match([product], n_jobs=n_jobs)
    assert len(precursors) > 0
    for precursor in precursors:
        np.testing.assert_array_equal(precursor.R, product.R)
        np.testing.assert_array_equal(precursor.r, product.r)
        np.testing.assert_array_equal(precursor.x, product.x)


@pytest.mark.parametrize(
    "processing, message",
    [
        (
            ReactionProcessing(before_match=MoleculeProcessing.AROMATICITY),
            "before_match is not implemented",
        ),
        (ReactionProcessing(after_split=MoleculeProcessing.NONE), "must include"),
    ],
)
def test_cpu_reaction_processing_rejects_unimplemented_combinations(
    processing: ReactionProcessing,
    message: str,
) -> None:
    reaction = parse_reaction_smarts("[C:1]>>[C:1]")

    with pytest.raises(ValueError, match=message):
        CpuReactionMatcher((reaction,), processing=processing)


def test_generate_precursors_refuses_application_with_oversize_precursor() -> None:
    """One precursor over the cap refuses the whole application, not just that piece.

    A reaction is defined by all of its named precursors, so returning the methane and
    silently dropping the 65-carbon chain would present an impossible disconnection as a
    complete one. CUDA and Metal apply the same rule.

    The split caps both fragments with hydrogen. Writing `[CH3:1].[CH2:2]` instead
    would hand back a free CH2 carbene, which is not a closed-shell state and is now
    rejected before the atom cap is ever consulted.
    """
    product = _molecule("C" * 66)
    reaction = parse_reaction_smarts("[CH3:1][CH2:2]>>[CH4:1].[CH3:2]")

    with pytest.warns(UserWarning, match="over 64 atoms"):
        outcomes = generate_precursors(
            product,
            reaction,
            max_mappings=1,
            max_precursor_atoms=64,
        )

    assert outcomes == []

    # Without the cap the same application returns both precursors.
    uncapped = generate_precursors(product, reaction, max_mappings=1)
    assert sorted(precursor.atom_count for precursor in uncapped[0].precursors) == [
        1,
        65,
    ]


def test_cpu_reaction_matcher_reports_precursor_atom_overflow() -> None:
    product = _molecule("C" * 66)
    reaction = parse_reaction_smarts("[CH3:1][CH2:2]>>[CH4:1].[CH3:2]")
    matcher = CpuReactionMatcher((reaction,), max_assignments=1, max_precursor_atoms=64)

    with pytest.warns(UserWarning, match="over 64 atoms"):
        precursors, hits = matcher.match([product])

    # The capped application is refused whole, so it contributes no hit row at all.
    assert list(precursors) == []
    assert hits.tolist() == []
    assert matcher.last_diagnostics is not None
    assert matcher.last_diagnostics.atom_overflow_components == 1
    assert matcher.last_diagnostics.atom_overflow_assignments == 1


def test_precursor_atom_cap_must_be_positive() -> None:
    product = _molecule("CC")
    reaction = parse_reaction_smarts("[C:1][C:2]>>[C:1].[C:2]")
    mapping = np.array([0, 1], dtype=np.int32)

    with pytest.raises(ValueError, match="max_precursor_atoms must be positive"):
        precursors_from_mapping(
            product=product,
            reaction=reaction,
            mapping=mapping,
            max_precursor_atoms=0,
        )


def test_generate_precursors_enumerates_multiple_break_sites() -> None:
    product = _molecule("CCC")
    reaction = parse_reaction_smarts("[C:1][C:2]>>[C:1].[C:2]")

    outcomes = generate_precursors(product, reaction)
    assert len(outcomes) == 2
    for outcome in outcomes:
        sizes = sorted(prec.atom_count for prec in outcome.precursors)
        assert sizes == [1, 2]


def test_parse_reaction_smarts_rejects_rhs_query_bonds() -> None:
    with pytest.raises(
        UnsupportedReactionError, match="RHS bond edits require a concrete bond type"
    ):
        parse_reaction_smarts("[C:1].[C:2]>>[C:1]~[C:2]")


def test_unsupported_reaction_error_is_a_smarts_error() -> None:
    assert issubclass(UnsupportedReactionError, SmartsError)


def test_parse_reaction_smarts_invalid_rhs_bubbles_as_smarts_error() -> None:
    with pytest.raises(SmartsError):
        parse_reaction_smarts("[C:1]>>[C")


def test_generate_precursors_removes_unmapped_lhs_atoms() -> None:
    product = _molecule("COC")
    reaction = parse_reaction_smarts("[C:1][O][C:2]>>[C:1]-[C:2]")

    outcomes = generate_precursors(product, reaction, max_mappings=4)
    assert outcomes
    precursor = outcomes[0].precursors[0]
    assert precursor.atom_count == 2
    assert precursor.bonds[0, 1] != 0


def _first_application(smiles: str, smarts: str):
    from vesma.cmol_builder import gmol_from_cmol
    from vesma.cpu_matcher import CpuMatcher
    from vesma.reaction import precursors_from_mapping_result

    product = _molecule(smiles)
    reaction = parse_reaction_smarts(smarts)
    matches = CpuMatcher().find_matches(reaction.lhs, gmol_from_cmol(product))
    assert matches.assignments.size, f"{smarts} does not match {smiles}"
    return precursors_from_mapping_result(
        product=product, reaction=reaction, mapping=matches.assignments[0]
    )


def _first_application_on(product, smarts: str):
    """:func:`_first_application` for a molecule already in hand -- a `cmol_union` pot."""
    from vesma.cpu_matcher import CpuMatcher
    from vesma.reaction import precursors_from_mapping_result

    reaction = parse_reaction_smarts(smarts)
    matches = CpuMatcher().find_matches(reaction.lhs, gmol_from_cmol(product))
    assert matches.assignments.size, f"{smarts} does not match the given molecule"
    return precursors_from_mapping_result(
        product=product, reaction=reaction, mapping=matches.assignments[0]
    )


def _written(result) -> list[str]:
    return sorted(rdkit_canonical_smiles(p) for p in result.precursors)


def test_an_application_that_orphans_a_component_is_refused_whole() -> None:
    """The RHS enumerates the precursors, so what it never wrote is not one of them --
    and a set missing one of them is not a disconnection.

    This ester template maps the alcohol's first carbon and then omits it from the RHS,
    which deletes it. The carbon sits between the ester oxygen and the rest of the
    alcohol, so deleting it disconnects that rest, and what falls out is the leaving group
    with an atom amputated and the wound capped with hydrogen. An ethyl ester yields
    methane where the real co-reactant is ethanol; a longer one yields a different real,
    purchasable compound -- on a 30-atom ester, seventeen carbons of it.

    **The acid alone is returned no longer.** Keeping it presents "ethyl benzoate comes
    from benzoic acid" as a complete answer when the step also needs ethanol, and that is
    the same error the atom cap already refuses, for the reason stated beside it: a
    partial set is worst exactly where it is least visible, in a multi-layer search where
    the missing piece would have been the molecule that failed to reach stock.

    The two cases are not distinguishable here without deciding how large a fragment
    stops being debris, so both are refused. What is *not* affected is the shape the
    corpus actually has -- a leaving atom pinned to one neighbour, which orphans
    nothing; see the next test.
    """
    ester = "[C:4]-[O;H0;D2;+0:3]-[C:2]=[O;D1;H0:1]>>[O;D1;H0:1]=[C:2]-[OH;D1;+0:3]"

    result = _first_application("CCOC(=O)c1ccccc1", ester)

    assert _written(result) == []
    assert result.diagnostics.undescribed_components == 1
    assert result.diagnostics.undescribed_component_applications == 1
    assert result.diagnostics.orphaned_components == 1
    # The fragment's formula survives on the diagnostics, because the fix is to the
    # template and a report has to be able to name what the template dropped.
    assert result.diagnostics.undescribed_formulas == ("C",)


def test_deleting_a_leaving_atom_orphans_nothing_and_is_untouched() -> None:
    """The shape deletion actually has in the corpus, which the rule must not disturb.

    Of 42,554 uspto42k templates only 41 delete a mapped atom, and all but one delete a
    leaving atom pinned to a single neighbour -- here a bromine, ``D1`` -- so there is
    nothing on its far side to orphan. Only the ester template above orphans anything.
    """
    debromination = (
        "[Br;D1;H0:4]-[CH2;D2;+0:1]-[C:2]=[O;D1;H0:3]>>[CH3;D1;+0:1]-[C:2]=[O;D1;H0:3]"
    )

    result = _first_application("O=C(CBr)c1ccccc1", debromination)

    assert _written(result) == ["CC(=O)c1ccccc1"]
    assert result.diagnostics.undescribed_components == 0


def test_a_plain_disconnection_keeps_both_halves() -> None:
    """Both components carry an RHS atom, so the rule is silent."""
    result = _first_application("CCO", "[C:1][O:2]>>[C:1].[O:2]")

    assert _written(result) == ["CC", "O"]
    assert result.diagnostics.undescribed_components == 0


def test_generate_precursors_adds_rhs_atoms_from_template() -> None:
    product = _molecule("C")
    reaction = parse_reaction_smarts("[C:1]>>[C:1]-O")

    outcomes = generate_precursors(product, reaction, max_mappings=1)
    assert outcomes
    precursor = outcomes[0].precursors[0]
    expected = _molecule("CO")
    assert has_match(gmol_from_cmol(expected), gmol_from_cmol(precursor))
    assert has_match(gmol_from_cmol(precursor), gmol_from_cmol(expected))


def test_generate_precursors_allows_no_common_maps() -> None:
    product = _molecule("C")
    reaction = parse_reaction_smarts("C>>O")

    outcomes = generate_precursors(product, reaction, max_mappings=1)

    assert outcomes
    precursor = outcomes[0].precursors[0]
    expected = _molecule("O")
    assert has_match(gmol_from_cmol(expected), gmol_from_cmol(precursor))
    assert has_match(gmol_from_cmol(precursor), gmol_from_cmol(expected))


def test_parse_reaction_smarts_accepts_open_aromatic_rhs_fragments() -> None:
    reaction = parse_reaction_smarts(
        "[#7;a:4]:[c:3]:[c;H0;D3;+0:1](-[I;H0;D1;+0:5]):[c:2]>>"
        "Br-[c;H0;D3;+0:1](:[c:2]):[c:3]:[#7;a:4].[IH;D0;+0:5]"
    )
    assert reaction.rhs.atom_count == 6


def test_generate_precursors_rebuilds_aromatic_rhs_fragments() -> None:
    product = _molecule("Ic1cnccc1")
    reaction = parse_reaction_smarts(
        "[#7;a:4]:[c:3]:[c;H0;D3;+0:1](-[I;H0;D1;+0:5]):[c:2]>>"
        "Br-[c;H0;D3;+0:1](:[c:2]):[c:3]:[#7;a:4].[IH;D0;+0:5]"
    )

    outcomes = generate_precursors(product, reaction, max_mappings=4)
    assert outcomes
    sizes = sorted(prec.atom_count for prec in outcomes[0].precursors)
    assert sizes == [1, 7]


def test_generate_precursors_inherits_unspecified_mapped_charge() -> None:
    product = _molecule("[NH4+]")
    reaction = parse_reaction_smarts("[N+:1]>>[N:1]")

    precursor = generate_precursors(product, reaction, max_mappings=1)[0].precursors[0]

    assert precursor.C.tolist() == [1]


def test_generate_precursors_applies_explicit_neutral_mapped_charge() -> None:
    product = _molecule("[NH4+]")
    reaction = parse_reaction_smarts("[N+:1]>>[N;H3;+0:1]")

    precursor = generate_precursors(product, reaction, max_mappings=1)[0].precursors[0]

    assert precursor.C.tolist() == [0]
    assert precursor.H.tolist() == [3]


def test_a_charge_change_that_leaves_hydrogen_unwritten_is_rejected() -> None:
    """Configurations are keyed by charge, so the matched atom's configuration says
    nothing about the atom after the change; completing one would be a target search."""
    product = _molecule("[NH4+]")
    reaction = parse_reaction_smarts("[N+:1]>>[N;+0:1]")

    with pytest.raises(UnsupportedReactionError, match="charge_changed_with_h_omitted"):
        generate_precursors(product, reaction, max_mappings=1)


def test_generate_precursors_ignores_chiral_markers() -> None:
    product = _molecule("C[C@H](O)Cl")
    reaction = parse_reaction_smarts("[C:1][C@:2](O)Cl>>[C:1][C:2](O)Cl")

    outcomes = generate_precursors(product, reaction, max_mappings=1)
    assert outcomes
    precursor = outcomes[0].precursors[0]
    expected = _molecule("CC(O)Cl")
    assert has_match(gmol_from_cmol(expected), gmol_from_cmol(precursor))
    assert has_match(gmol_from_cmol(precursor), gmol_from_cmol(expected))


def test_generate_precursors_canonicalizes_methoxy_cleavage_precursor() -> None:
    chem = pytest.importorskip("rdkit.Chem")

    product = _molecule("CCN(CC)CCN/C(C)=C1\\C(=O)NC(=O)N(c2ccccc2OC)C1=O")
    reaction = parse_reaction_smarts("[C:1][O:2]>>[C:1].[O:2]")

    outcomes = generate_precursors(product, reaction)
    precursor_smiles = {
        chem.MolToSmiles(mol_from_cmol(precursor))
        for outcome in outcomes
        for precursor in outcome.precursors
        if precursor.atom_count > 1
    }
    expected = chem.MolToSmiles(
        chem.MolFromSmiles("CCN(CC)CCNC(C)=C1C(=O)NC(=O)N(c2ccccc2[OH:2])C1=O")
    )

    assert expected in precursor_smiles


def test_aromatic_carbon_rhs_h_remains_perception_seed_across_steps() -> None:
    bromide = _molecule("COc1cc(Br)cc([N+](=O)[O-])c1N")
    debrominate = parse_reaction_smarts(
        "[Br;H0;D1;+0:1]-[c;H0;D3;+0:3](:[c:2]):[c:4]>>"
        "Br-[Br;H0;D1;+0:1].[c:2]:[cH;D2;+0:3]:[c:4]"
    )
    demethylate = parse_reaction_smarts(
        "[CH3;D1;+0:1]-[O;H0;D2;+0:2]-[c:3]>>I-[CH3;D1;+0:1].[OH;D1;+0:2]-[c:3]"
    )

    intermediate = next(
        precursor
        for outcome in generate_precursors(bromide, debrominate)
        for precursor in outcome.precursors
        if rdkit_canonical_smiles(precursor) == "COc1cccc([N+](=O)[O-])c1N"
    )
    precursor_smiles = {
        rdkit_canonical_smiles(precursor)
        for outcome in generate_precursors(intermediate, demethylate)
        for precursor in outcome.precursors
    }

    assert "Nc1c(O)cccc1[N+](=O)[O-]" in precursor_smiles
    assert "NC1C(O)CCCC1[N+](=O)[O-]" not in precursor_smiles


def test_bracketed_aromatic_carbon_stores_hydrogen_as_implicit() -> None:
    # A bracketed [cH] forces the parser's explicit-H path (cmol_builder.py Case 1). Aromatic
    # carbon hydrogen is degree-derived, never an independent degree of freedom, so it
    # must be stored as implicit (h == H). Otherwise a later survivor carry (H - h) would
    # re-inject an explicit H that perception double-counts and de-aromatizes.
    benzene = _molecule("[cH]1[cH][cH][cH][cH][cH]1")

    assert all(int(a) == AROMATIC for a in benzene.A)
    assert benzene.H.tolist() == [1] * 6
    assert benzene.h.tolist() == [1] * 6


def test_bracketed_aromatic_ring_survives_split_without_dearomatizing() -> None:
    # End-to-end guard for the parser-origin poison: a bracketed-[cH] anisole demethylated
    # leaves an aromatic phenol fragment whose ring carbons are pure survivors. Before the
    # fix those carbons carried explicit_h = H - h = 1 into re-perception, double-counting
    # against the degree-derived aromatic H and saturating the ring.
    anisole = _molecule("CO[c]1[cH][cH][cH][cH][cH]1")
    demethylate = parse_reaction_smarts(
        "[CH3;D1;+0:1]-[O;H0;D2;+0:2]-[c:3]>>I-[CH3;D1;+0:1].[OH;D1;+0:2]-[c:3]"
    )

    precursor_smiles = {
        rdkit_canonical_smiles(precursor)
        for outcome in generate_precursors(anisole, demethylate)
        for precursor in outcome.precursors
    }

    assert "Oc1ccccc1" in precursor_smiles
    assert "OC1CCCCC1" not in precursor_smiles


def test_aromatic_heteroatom_hydrogen_remains_explicit() -> None:
    # Regression for the storage rule: aromatic heteroatom H is a real degree of freedom
    # (pyrrole [nH] donates two pi electrons, pyridine n donates one), so it must stay
    # explicit (h == 0) and not be folded into implicit like aromatic carbon.
    imidazole = _molecule("c1c[nH]cn1")

    assert all(int(a) == AROMATIC for a in imidazole.A)
    nitrogen = [index for index, z in enumerate(imidazole.Z.tolist()) if z == 7]
    pyrrole, pyridine = sorted(nitrogen, key=lambda index: -int(imidazole.H[index]))
    assert int(imidazole.H[pyrrole]) == 1 and int(imidazole.h[pyrrole]) == 0
    assert int(imidazole.H[pyridine]) == 0 and int(imidazole.h[pyridine]) == 0


def test_anionic_oxygen_infers_zero_hydrogens() -> None:
    # An anionic atom whose hydrogen is inferred (e.g. a nitro [O-]) must get zero H,
    # not a spurious one: charge shifts bonding capacity isoelectronically (O- behaves
    # like F, valence 1). Otherwise templates constraining [O;-;H0] never match and
    # nitro/azide/N-oxide disconnections silently produce no precursors.
    nitroarene = _molecule("Cc1ccccc1[N+](=O)[O-]")
    oxygens = [index for index, z in enumerate(nitroarene.Z.tolist()) if z == 8]
    assert [int(nitroarene.H[index]) for index in oxygens] == [0, 0]

    denitrate = parse_reaction_smarts(
        "[O;-;D1;H0:2]-[N+;H0;D3:1](=[O;H0;D1;+0:3])-[c;H0;D3;+0:5](:[c:4]):[c:6]>>"
        "O=[N+;H0;D3:1](-[O;-;D1;H0:2])-[OH;D1;+0:3].[c:4]:[cH;D2;+0:5]:[c:6]"
    )
    precursor_smiles = {
        rdkit_canonical_smiles(precursor)
        for outcome in generate_precursors(nitroarene, denitrate)
        for precursor in outcome.precursors
    }

    assert "Cc1ccccc1" in precursor_smiles
    assert "O=[N+]([O-])O" in precursor_smiles


def test_generate_precursors_without_ring_descriptors() -> None:
    product = _molecule("c1ccccc1OC")
    reaction = parse_reaction_smarts("[C:1][O:2]>>[C:1].[O:2]")

    outcomes = generate_precursors(
        product, reaction, ring_descriptors=RingDescriptors.NONE
    )
    assert outcomes
    for outcome in outcomes:
        for prec in outcome.precursors:
            assert prec.R.tolist() == [-1] * prec.atom_count
            assert prec.r.tolist() == [-1] * prec.atom_count
            assert prec.x.tolist() == [-1] * prec.atom_count


def test_generate_precursors_with_ring_connectivity_only() -> None:
    """The GPU-parity selection: x is exact, R and r stay at the sentinel.

    x counts incident ring bonds, and a ring bond is any non-bridge, so the value does
    not depend on which spanning forest found it -- which is what lets the Metal and
    CUDA perception kernels produce it. R and r read that forest's fundamental cycles.
    """
    product = _molecule("c1ccccc1OC")
    reaction = parse_reaction_smarts("[C:1][O:2]>>[C:1].[O:2]")

    outcomes = generate_precursors(
        product, reaction, ring_descriptors=RingDescriptors.CONNECTIVITY
    )
    assert outcomes
    ring_seen = False
    for outcome in outcomes:
        for prec in outcome.precursors:
            assert prec.R.tolist() == [-1] * prec.atom_count
            assert prec.r.tolist() == [-1] * prec.atom_count
            assert all(value >= 0 for value in prec.x.tolist())
            if prec.atom_count > 1:
                # Phenol: six ring carbons with two ring bonds each, the oxygen with none.
                assert sorted(prec.x.tolist()) == [0] + [2] * 6
                ring_seen = True
    assert ring_seen


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_a_rejected_application_costs_only_itself(n_jobs: int) -> None:
    """Chemistry the model cannot express rejects one application, not the run.

    A ring-opening template releases delocalized bonds it never authors, so every one of
    its applications is rejected. The alkane disconnection in the same batch must still
    return its precursors, and the rejections must be counted rather than silently lost.
    """
    molecules = [_molecule("CCC"), _molecule("c1ccccc1")]
    rules = (
        parse_reaction_smarts("[C:1][C:2]>>[C:1].[C:2]"),
        parse_reaction_smarts("[c:1][c:2]>>[c:1].[c:2]"),
    )
    matcher = CpuReactionMatcher(rules)

    precursors, hits = matcher.match(molecules, n_jobs=n_jobs)

    assert len(precursors) == 8
    assert len(hits) == 8
    assert matcher.last_diagnostics is not None
    assert matcher.last_diagnostics.rejected_applications == 12


def test_a_single_application_still_reports_why_it_was_rejected() -> None:
    """The batch keeps going; the one-shot entry point diagnoses instead."""
    from vesma.aromaticity import OrphanDelocalizedBondError

    with pytest.raises(OrphanDelocalizedBondError):
        generate_precursors(
            _molecule("c1ccccc1"), parse_reaction_smarts("[c:1][c:2]>>[c:1].[c:2]")
        )


def test_the_orphan_record_is_attributed_identically_in_parallel() -> None:
    """A diagnostic only the serial path filled in was a diagnostic nobody could read.

    `stamped_undescribed` moves each dropped fragment's formula into `undescribed_rows`
    together with the application that produced it, and the merge carries
    `undescribed_rows` but not `undescribed_formulas`. Only the serial path stamped, so an
    unstamped worker's formulas were discarded and the attribution came back empty for
    every `n_jobs != 1` -- which is the default, `resolve_n_jobs(None)` being parallel.
    It survived because the one consumer, `bench_precursor_audit`, pins `n_jobs=1`, and
    because nothing asserted on the rows at all.

    The count was summed either way, so this is specifically about *which* molecule, rule
    and application orphaned *what*.
    """
    from vesma.reaction import reaction_patterns_from_smarts

    ester = "[C:4]-[O;H0;D2;+0:3]-[C:2]=[O;D1;H0:1]>>[O;D1;H0:1]=[C:2]-[OH;D1;+0:3]"
    matcher = CpuReactionMatcher(reaction_patterns_from_smarts((ester,), cache=True))
    molecules = [
        _molecule("CCOC(=O)c1ccccc1"),
        _molecule("CCCCOC(=O)c1ccc(Cl)cc1"),
    ]

    attributed = {}
    for jobs in (1, 2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matcher.match(molecules, n_jobs=jobs)
        diagnostics = matcher.last_diagnostics
        assert diagnostics is not None
        attributed[jobs] = sorted(diagnostics.undescribed_rows)
        assert diagnostics.undescribed_components == 2

    assert attributed[1] == attributed[2]
    assert attributed[1], "the rows must be populated, not merely equal"
    assert all(len(row) == 4 for row in attributed[1])


def test_an_untouched_component_of_a_pot_is_a_spectator_and_not_an_orphan() -> None:
    """The distinction that keeps the refusal from refusing every selectivity probe.

    A probe builds one molecule per application with `cmol_union` and runs forward rules
    against it, which is what "the pot" means here. Most forward rules are unimolecular,
    so they match one component and leave the other alone -- and the RHS, naming only what
    it edited, never mentions the other. By the RHS test alone that is indistinguishable
    from an orphan, and refusing it would refuse the forward probe wholesale.

    It is distinguishable by asking the *product's* bonds rather than the edited ones: an
    orphan was attached to something before the edit and is not now, while a spectator was
    always its own molecule. Here TMSCN sits beside the glycosyl acetate it will go on to
    attack, and the C-glycosylation that consumes the acetate must still produce its
    product.
    """
    glycosylation = (
        "[CH1:1](-[O:2])(-[C:3])-[O]-[C](-[CH3])=[O]>>[CH1:1](-[O:2])(-[C:3])-[C]#[N]"
    )
    pot = cmol_union([_molecule("C[Si](C)(C)C#N"), _molecule("CC(=O)OC1CCCCO1")])

    result = _first_application_on(pot, glycosylation)

    assert _written(result) == ["N#CC1CCCCO1"]
    # Counted as undescribed, because the RHS really does not name it...
    assert result.diagnostics.undescribed_components == 1
    assert result.diagnostics.undescribed_formulas == ("C4NSi",)
    # ...but not as orphaned, which is what the refusal keys on.
    assert result.diagnostics.orphaned_components == 0


def test_the_refusal_warning_names_the_orphan_and_not_a_spectator() -> None:
    """A diagnostic that blames the wrong molecule is worse than one that says nothing.

    `undescribed_components` counts every component the RHS does not name, which in a probe's pot
    is every unreacted reagent -- a `cmol_union` of an application's precursors hands a
    unimolecular forward rule a spectator every single time. `orphaned_components` counts only the
    severed subset, and only that refuses. The warning quoted the second count while listing rows
    from the first, so it reported a refusal and then named a reagent that had nothing to do with
    it: forty-five undescribed spectators, zero orphans, no refusal, and a message pointing at
    carbon tetrabromide.
    """
    from vesma.reaction import reaction_patterns_from_smarts

    oxidation = "[OX2H1:1]-[CH2:2]>>[O:1]=[C:2]"
    matcher = CpuReactionMatcher(
        reaction_patterns_from_smarts((oxidation,), cache=True)
    )
    # an alcohol to oxidise, and a reagent for some other reaction sitting in the same pot
    pot = cmol_union([_molecule("OCc1ccccc1"), _molecule("BrC(Br)(Br)Br")])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _precursors, hits = matcher.match([pot], None, n_jobs=1)

    diagnostics = matcher.last_diagnostics
    assert diagnostics is not None
    # The spectator is undescribed, as it must be: no RHS atom names it.
    assert diagnostics.undescribed_components >= 1
    # It is not severed, so it is not an orphan, so nothing is refused and nothing is warned.
    assert diagnostics.orphaned_components == 0
    assert diagnostics.orphaned_rows == ()
    assert np.asarray(hits).reshape(-1, 4).shape[0] > 0
    assert not [w for w in caught if "orphaned" in str(w.message)]


def test_parallel_matching_builds_each_referenced_molecule_graph_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vesma import cpu_matcher

    molecules = [_molecule(smiles) for smiles in ("CC", "CCC", "CCCC")]
    rules = [
        parse_reaction_smarts("[C:1][C:2]>>[C:1].[C:2]"),
        parse_reaction_smarts("[C:1]>>[C:1]"),
    ]
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    matcher = CpuReactionMatcher(rules)
    serial, serial_hits = matcher.match(molecules, pairs, n_jobs=1)
    build_graph = Mock(wraps=cpu_matcher.gmol_from_cmol)
    monkeypatch.setattr(cpu_matcher, "gmol_from_cmol", build_graph)

    parallel, parallel_hits = matcher.match(molecules, pairs, n_jobs=2)

    assert build_graph.call_count == 2
    np.testing.assert_array_equal(parallel_hits, serial_hits)
    assert [mol.atom_count for mol in parallel] == [mol.atom_count for mol in serial]
