"""RDChiral comparison backend: contract, reproducibility, and the ring-split divergence."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdchiral")

from vesma import cmol_from_smiles  # noqa: E402
from vesma.cpu_matcher import CpuReactionMatcher  # noqa: E402
from vesma.keyer import (  # noqa: E402
    cmol_rdkit_smiles_keyer,
    rdkit_smiles_keyer,
    rdkit_token_from_smiles,
)
from vesma.rdchiral_matcher import (  # noqa: E402
    RdchiralReactionMatcher,
    rdchiral_reactants_from_smiles,
    rdchiral_reactions_from_smarts,
    rdchiral_version,
)
from vesma.rdkit_matcher import (  # noqa: E402
    RdkitReactionMatcher,
    rdkit_reactions_from_smarts,
)
from vesma.reaction import parse_reaction_smarts  # noqa: E402

# The disconnection template from the supplementary information: a two-component
# right-hand side whose matched bond lies on a ring.
RING_TEMPLATE = (
    "[C:2]-[CH2;D2;+0:1]-[CH2;D2;+0:3]-[C:4]"
    ">>Br-[CH2;D2;+0:1]-[C:2].Br-[CH2;D2;+0:3]-[C:4]"
)
RING_TARGET = "C1CCCCC1"

# An acyclic amide disconnection, where every backend is expected to agree.
AMIDE_TEMPLATE = "[C:1](=[O:2])-[NH1:3]-[C:4]>>[C:1](=[O:2])-[OH].[NH2:3]-[C:4]"
AMIDE_TARGET = "CCCC(=O)NCc1ccccc1"


def rdchiral_matcher(template: str, target: str, **kwargs):
    matcher = RdchiralReactionMatcher(
        rdchiral_reactions_from_smarts([template]), **kwargs
    )
    return matcher, rdchiral_reactants_from_smiles([target])


def outcome_key_sets(precursors, hits, key) -> frozenset[frozenset[str]]:
    """Group hits into one key set per (mol_idx, rule_id, app_idx) application."""
    applications: dict[tuple[int, int, int], set[str]] = {}
    for mol_idx, rule_id, app_idx, precursor_idx in hits.tolist():
        applications.setdefault((mol_idx, rule_id, app_idx), set()).add(
            key(precursors[precursor_idx])
        )
    return frozenset(frozenset(keys) for keys in applications.values())


def test_hits_satisfy_the_reaction_matcher_invariants() -> None:
    matcher, reactants = rdchiral_matcher(AMIDE_TEMPLATE, AMIDE_TARGET)

    precursors, hits = matcher.match(reactants, [(0, 0)])

    assert hits.shape[1] == 4
    assert hits.dtype == np.int64
    referenced = set(hits[:, 3].tolist())
    assert referenced <= set(range(len(precursors))), "hit points outside the batch"
    assert referenced == set(range(len(precursors))), "orphan precursor rows"


def test_amide_disconnection_agrees_with_rdkit_and_vesma() -> None:
    matcher, reactants = rdchiral_matcher(AMIDE_TEMPLATE, AMIDE_TARGET)
    rdchiral_sets = outcome_key_sets(
        *matcher.match(reactants, [(0, 0)]), rdkit_token_from_smiles
    )

    rdkit_matcher = RdkitReactionMatcher(
        rdkit_reactions_from_smarts([AMIDE_TEMPLATE]), n_jobs=1
    )
    rdkit_sets = outcome_key_sets(
        *rdkit_matcher.match([mol_from_smiles(AMIDE_TARGET)], [(0, 0)]),
        rdkit_smiles_keyer.encode,
    )
    cpu_matcher = CpuReactionMatcher([parse_reaction_smarts(AMIDE_TEMPLATE)])
    cmol = cmol_from_smiles(AMIDE_TARGET)[0]
    vesma_sets = outcome_key_sets(
        *cpu_matcher.match([cmol], [(0, 0)], n_jobs=1),
        cmol_rdkit_smiles_keyer.encode,
    )

    expected = frozenset(
        {
            frozenset(
                {
                    rdkit_token_from_smiles("CCCC(=O)O"),
                    rdkit_token_from_smiles("NCc1ccccc1"),
                }
            )
        }
    )
    assert rdchiral_sets == expected
    assert rdkit_sets == expected
    assert vesma_sets == expected


def test_ring_disconnection_stitches_to_the_single_vesma_precursor() -> None:
    """RDChiral resolves the ring case the way vesma does, not the way RDKit does.

    The supplementary information states this from the literature; this pins it.
    """
    matcher, reactants = rdchiral_matcher(RING_TEMPLATE, RING_TARGET)
    precursors, hits = matcher.match(reactants, [(0, 0)])
    rdchiral_sets = outcome_key_sets(precursors, hits, rdkit_token_from_smiles)

    cpu_matcher = CpuReactionMatcher([parse_reaction_smarts(RING_TEMPLATE)])
    cmol = cmol_from_smiles(RING_TARGET)[0]
    vesma_sets = outcome_key_sets(
        *cpu_matcher.match([cmol], [(0, 0)], n_jobs=1),
        cmol_rdkit_smiles_keyer.encode,
    )

    # One ring-opened precursor, all six carbons conserved.
    assert rdchiral_sets == frozenset(
        {frozenset({rdkit_token_from_smiles("BrCCCCCCBr")})}
    )
    assert rdchiral_sets == vesma_sets

    rdkit_matcher = RdkitReactionMatcher(
        rdkit_reactions_from_smarts([RING_TEMPLATE]), n_jobs=1
    )
    rdkit_sets = outcome_key_sets(
        *rdkit_matcher.match([mol_from_smiles(RING_TARGET)], [(0, 0)]),
        rdkit_smiles_keyer.encode,
    )
    # RDKit follows the declared two-component topology and duplicates the shared ring
    # atoms: two 1-bromobutanes, eight carbons out of a six-carbon ring.
    assert rdkit_sets == frozenset({frozenset({rdkit_token_from_smiles("CCCCBr")})})
    assert rdkit_sets != vesma_sets


def test_app_index_is_reproducible_across_runs() -> None:
    """rdchiralRun returns a set-derived list; app_idx must not depend on its order."""
    matcher, reactants = rdchiral_matcher(AMIDE_TEMPLATE, "CCCC(=O)NCc1ccccc1CC(=O)NC")

    first_precursors, first_hits = matcher.match(reactants, [(0, 0)])
    second_precursors, second_hits = matcher.match(reactants, [(0, 0)])

    assert first_hits.tolist() == second_hits.tolist()
    assert list(first_precursors) == list(second_precursors)


def test_reactions_reject_an_invalid_template() -> None:
    """Template compilation fails fast; it does not silently skip the rule."""
    with pytest.raises(ValueError):
        rdchiral_reactions_from_smarts(["[C:1][OH:2]>>[C:1][Br:2]"])


def test_diagnostics_record_the_deduplicated_outcome_semantics() -> None:
    matcher, reactants = rdchiral_matcher(
        RING_TEMPLATE, RING_TARGET, combine_enantiomers=False
    )

    matcher.match(reactants, [(0, 0)])

    diagnostics = matcher.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.rules_total == 1
    assert diagnostics.pairs_evaluated == 1
    assert diagnostics.pairs_failed == 0
    assert diagnostics.outcome_semantics == "distinct-deduplicated"
    assert diagnostics.precursor_representation == "smiles"
    assert diagnostics.combine_enantiomers is False
    assert rdchiral_version() != "unknown"


def mol_from_smiles(smiles: str):
    from rdkit import Chem

    return Chem.MolFromSmiles(smiles)


def test_rdkit_discards_are_recorded_per_application() -> None:
    """A run-wide total cannot say which pair lost an outcome, and the pair is compared.

    RDKit splits benzene into two aromatic fragments and then cannot kekulize either, so
    it returns nothing -- indistinguishable, from the return value alone, from a template
    that simply did not apply. The rows carry the application and RDKit's own message, so
    an audit can mark that pair incomparable instead of scoring the discard as a genuine
    empty result and improving its agreement rate because a backend failed.
    """
    from rdkit import Chem

    matcher = RdkitReactionMatcher(
        rdkit_reactions_from_smarts(["[c:1][c:2]>>[c:1].[c:2]"]), n_jobs=1
    )

    _, hits = matcher.match([Chem.MolFromSmiles("c1ccccc1")], [(0, 0)])

    diagnostics = matcher.last_diagnostics
    assert diagnostics is not None
    assert hits.size == 0
    assert diagnostics.outcomes_unsanitizable == diagnostics.outcomes_total
    assert len(diagnostics.discarded) == diagnostics.outcomes_unsanitizable
    assert all(row[:2] == (0, 0) for row in diagnostics.discarded)
    assert "Kekulize" in diagnostics.discarded[0][3]


def test_application_indices_are_raw_outcome_indices() -> None:
    """So a discard row names the application it belongs to.

    Numbering only the survivors slides every later application down by one per discard,
    and the row then points at a different application than the one that failed.
    """
    from rdkit import Chem

    matcher = RdkitReactionMatcher(
        rdkit_reactions_from_smarts(["[C:1][O:2]>>[C:1].[O:2]"]), n_jobs=1
    )

    _, hits = matcher.match([Chem.MolFromSmiles("CCO")], [(0, 0)])

    diagnostics = matcher.last_diagnostics
    assert diagnostics is not None
    assert not diagnostics.discarded
    assert sorted({int(hit[2]) for hit in hits}) == list(
        range(diagnostics.outcomes_total)
    )
