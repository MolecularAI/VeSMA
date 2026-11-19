"""The precursor audit's own machinery: identity codecs, accounting, classification.

The audit is what Phase 3 reads to decide whether a divergence is a model consequence or
a model defect, so a defect in the audit itself would be read as chemistry. These tests
pin the properties the artifact's conclusions rest on: that the two identities separate
what they claim to separate, that no application escapes the denominator, and that the
divergence classifier says what it means.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.vesma.bench_precursor_audit import (
    CheckpointPlan,
    PairResult,
    audit_key_from_cmol,
    audit_key_from_smiles,
    declares_multiple_components,
    discard_checkpoint,
    keyed_applications,
    load_checkpoint,
    mechanism,
    native_key,
    new_accumulator,
    pair_results,
    partition,
    save_checkpoint,
    status_of,
)
from vesma.cmol_builder import cmol_from_smiles
from vesma.keyer import rdkit_token_from_smiles
from vesma.matcher import hit_array

pytest.importorskip("rdkit")


def cmol(smiles: str):
    molecule, _ = cmol_from_smiles(smiles)
    return molecule


# --------------------------------------------------------------------------- #
# Cross-tool identity
# --------------------------------------------------------------------------- #
def test_audit_key_preserves_isotopes_where_the_search_keyer_erases_them() -> None:
    """The reason the audit needs its own codec rather than reusing `vesma.keyer`.

    The search keyer groups isotopologues on purpose, so a stock lookup finds a molecule
    however it was labelled. An audit inheriting that would score an isotopic relabelling
    as agreement between two backends that in fact produced different molecules.
    """
    assert rdkit_token_from_smiles("[13CH4]") == rdkit_token_from_smiles("C")
    assert audit_key_from_smiles("[13CH4]") != audit_key_from_smiles("C")


def test_audit_key_strips_stereochemistry_and_atom_maps() -> None:
    """Vesma is stereo-blind, so keeping stereo would report a divergence it cannot have."""
    assert audit_key_from_smiles("N[C@@H](C)C(=O)O") == audit_key_from_smiles(
        "N[C@H](C)C(=O)O"
    )
    assert audit_key_from_smiles("[CH3:1][OH:2]") == audit_key_from_smiles("CO")


@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccccc1O",
        # Vesma's side of the audit goes CMol -> mol_from_cmol -> SMILES, which declares
        # every hydrogen explicitly, while RDKit's own outcomes never take that path. If
        # the two produced different SMILES for one molecule, every vesma outcome would
        # score as a divergence. These are where the two writers could part company: a
        # metal bond RDKit cannot assign a default valence to, a hypervalent phosphorus,
        # a charge, an isotope. Checked across all 2,881 PaRoutes molecules: none differ.
        "O=[Os](=O)(=O)=O",
        "[CH3][Mg]Br",
        "O=[PH2]O",
        "CC(=O)[O-]",
        "[13CH4]",
    ],
)
def test_audit_key_agrees_across_the_cmol_and_smiles_entry_points(smiles: str) -> None:
    """One key space: a precursor keys identically whichever backend built it."""
    assert audit_key_from_cmol(cmol(smiles)) == audit_key_from_smiles(smiles)


# --------------------------------------------------------------------------- #
# Vesma-native identity
# --------------------------------------------------------------------------- #
def test_native_key_is_spelling_invariant() -> None:
    """The property the whole model exists to provide, read through the audit's codec."""
    assert native_key(cmol("c1ccccc1")) == native_key(cmol("C1=CC=CC=C1"))


def test_native_key_separates_isotopes_charges_and_bond_kinds() -> None:
    assert native_key(cmol("C")) != native_key(cmol("[13CH4]"))
    assert native_key(cmol("CC(=O)[O-]")) != native_key(cmol("CC(=O)O"))
    assert native_key(cmol("c1ccccc1")) != native_key(cmol("C1CCCCC1"))


def test_native_key_is_total_where_the_rdkit_codec_is_partial() -> None:
    """A key for every molecule vesma can build, with no RDKit round trip to fail.

    This is what keeps a precursor RDKit cannot serialize inside the audit instead of
    silently absent from it.
    """
    assert native_key(cmol("[CH3][Mg]Br"))
    assert native_key(cmol("c1ccccc1")) != ""


def test_native_key_does_not_disconnect_metals() -> None:
    """Why InChI is not used as the audit identity.

    InChI disconnects organometallic bonds, so a Grignard and its dissociated salt share
    one key -- concealing exactly the bonding the metal configurations were added to
    express. The native codec keeps the edge.
    """
    assert native_key(cmol("[CH3][Mg]Br")) != native_key(cmol("[CH3-].[Mg+]Br"))


# --------------------------------------------------------------------------- #
# Fixed denominators and application accounting
# --------------------------------------------------------------------------- #
def test_every_pair_stays_in_the_denominator_even_with_no_hits() -> None:
    """An agreement rate that improves when a backend produces nothing is meaningless."""
    results = pair_results({}, [(0, 0), (0, 1)])
    assert set(results) == {(0, 0), (0, 1)}
    assert all(result.outcomes == frozenset() for result in results.values())


def test_a_failed_pair_is_recorded_rather_than_dropped() -> None:
    results = pair_results({}, [(0, 0)], failed={(0, 0)})
    assert results[(0, 0)].failed
    assert not results[(0, 0)].comparable
    assert status_of(results[(0, 0)]) == "backend_failed"


def test_an_unkeyable_application_is_excluded_from_the_outcome_set_and_counted() -> (
    None
):
    """Excluded, because a partial outcome would compare as a *different* outcome.

    Counted, because the pair's outcome set is then no longer the whole story and the
    comparison has to say so.
    """
    precursors = np.array(["CO", "!!! not a molecule"], dtype=object)
    hits = hit_array([(0, 0, 0, 0), (0, 0, 1, 1)])
    keyed = keyed_applications(precursors, hits, audit_key_from_smiles)

    assert keyed[(0, 0, 0)].keyed
    assert not keyed[(0, 0, 1)].keyed
    assert keyed[(0, 0, 1)].error

    results = pair_results(keyed, [(0, 0)])
    assert results[(0, 0)].accepted == 1
    assert results[(0, 0)].unkeyable == 1
    assert not results[(0, 0)].comparable
    assert status_of(results[(0, 0)]) == "partly_unkeyable"


def test_one_unkeyable_component_invalidates_its_whole_application() -> None:
    """The same treatment `rdkit_matcher.split_outcome` gives an unsanitizable outcome."""
    precursors = np.array(["CO", "!!! not a molecule"], dtype=object)
    hits = hit_array([(0, 0, 0, 0), (0, 0, 0, 1)])
    keyed = keyed_applications(precursors, hits, audit_key_from_smiles)

    assert not keyed[(0, 0, 0)].keyed
    assert pair_results(keyed, [(0, 0)])[(0, 0)].outcomes == frozenset()


def test_one_pair_carries_both_accepted_and_refused_applications() -> None:
    """The observation that makes the application, not the pair, the unit of accounting.

    A pair-level status has no truthful value to take here, and a pair reported as clean
    agreement can be concealing refusals.
    """
    result = PairResult(
        outcomes=frozenset({frozenset({"CO"})}),
        accepted=4,
        rejections=(0,) * 9 + (12,) + (0,) * 5,
    )
    assert result.accepted == 4
    assert result.refused == 12
    assert result.attempted == 16
    assert status_of(result) == "partly_refused"
    # Refusals do not make the surviving outcome set incomparable: those applications
    # produced nothing to compare, and vesma said why.
    assert result.comparable


# --------------------------------------------------------------------------- #
# Divergence classification
# --------------------------------------------------------------------------- #
def sets(*outcomes: tuple[str, ...]) -> frozenset[frozenset[str]]:
    return frozenset(frozenset(outcome) for outcome in outcomes)


def test_partition_names_the_backend_that_stands_apart() -> None:
    a, b = sets(("CO",)), sets(("CN",))
    assert partition(a, b, a) == "rdkit-alone"
    assert partition(a, a, b) == "rdchiral-alone"
    assert partition(b, a, a) == "vesma-alone"
    assert partition(a, b, sets(("CC",))) == "all-three-differ"


def test_mechanism_is_orthogonal_to_partition() -> None:
    """*Which* backend differs and *how* it differs are separate questions.

    Collapsing them into one label, as the previous harness did, made
    "rdkit-duplicates-atoms" and "rdkit-alone" mutually exclusive when they are not.
    """
    assert mechanism(sets(("CO",)), sets(("CO",))) == "equal"
    assert mechanism(frozenset(), sets(("CO",))) == "vesma-empty"
    assert mechanism(sets(("CO",)), frozenset()) == "other-empty"
    assert mechanism(sets(("CO",)), sets(("CCO",))) == "different-heavy-atoms"
    # Ethanol against dimethyl ether: the same atoms, connected differently. Worth its
    # own class because it is the signature of a bond placed in the wrong position, not
    # of a miscounted valence.
    assert mechanism(sets(("CCO",)), sets(("COC",))) == "same-formula"


def test_mechanism_separates_a_hydrogen_difference_from_a_skeleton_difference() -> None:
    """Same heavy atoms, different hydrogen count: the aromaticity-loss signature."""
    assert mechanism(sets(("C1CCCCC1",)), sets(("c1ccccc1",))) == (
        "hydrogen-only-vesma-gained"
    )
    assert mechanism(sets(("c1ccccc1",)), sets(("C1CCCCC1",))) == (
        "hydrogen-only-vesma-lost"
    )


def test_multi_component_rhs_is_recorded_because_duplication_alone_misses_the_class() -> (
    None
):
    """The property that separates vesma from RDKit across the whole rdkit-alone class.

    At size 100 every one of the 1,282 ``rdkit-alone`` divergences comes from a template
    whose RHS declares two or more precursors, and vesma agrees with RDChiral in every
    one. But ``rdkit_duplicates_atoms`` is False in 197 of them: where the declared
    components share no atoms, RDKit severs a bond the template never asked to break
    rather than copying anything, and its outcome is no heavier than vesma's. Recording
    the declaration keeps those 197 with the class they belong to.
    """
    assert declares_multiple_components("[C:1][O:2]>>[C:1].[O:2]")
    assert not declares_multiple_components("[C:1][O:2]>>[C:1][O:2]")
    # A dot on the left-hand side is a different statement and must not count.
    assert not declares_multiple_components("[C:1].[O:2]>>[C:1][O:2]")


def test_an_rdkit_discard_makes_its_pair_incomparable() -> None:
    """RDKit builds an outcome, cannot sanitize it, and returns nothing for it.

    Counting those only in a run-wide total -- which is where they landed at first --
    leaves the pair reading `complete`, so the discard scores as a genuine empty result
    and agreement improves because a backend failed. That is precisely what fixed
    denominators exist to prevent, and it applies to RDKit exactly as it does to vesma's
    unkeyable precursors and RDChiral's failed pairs.
    """
    results = pair_results({}, [(0, 0)], discarded={(0, 0): 1})

    assert results[(0, 0)].unkeyable == 1
    assert not results[(0, 0)].comparable
    # Nothing survived, so this is not "partly": it is the case that reads as a clean
    # empty result from RDKit's return value alone, and it needs its own name.
    assert status_of(results[(0, 0)]) == "all_applications_unkeyable"


def test_a_discard_beside_a_survivor_still_marks_the_pair() -> None:
    """One pair holds many applications, and they do not share a fate."""
    precursors = np.array(["CO"], dtype=object)
    keyed = keyed_applications(
        precursors, hit_array([(0, 0, 1, 0)]), audit_key_from_smiles
    )

    results = pair_results(keyed, [(0, 0)], discarded={(0, 0): 1})

    assert results[(0, 0)].accepted == 1
    assert results[(0, 0)].unkeyable == 1
    assert not results[(0, 0)].comparable
    assert status_of(results[(0, 0)]) == "partly_unkeyable"


# --------------------------------------------------------------------------- #
# Crash resilience
# --------------------------------------------------------------------------- #
def test_a_checkpoint_round_trips_the_accumulated_state(tmp_path) -> None:
    """The full sweep is half a day, and nothing reaches disk until it returns."""
    plan = CheckpointPlan(path=tmp_path / "audit.pickle", token=("v1", "3"))
    state = new_accumulator()
    state.rdkit_outcomes_total = 7
    state.refusals.append({"template_id": "40639"})

    save_checkpoint(plan, done=50, state=state)
    resumed, restored = load_checkpoint(plan)

    assert resumed == 50
    assert restored.rdkit_outcomes_total == 7
    assert restored.refusals == [{"template_id": "40639"}]


def test_a_checkpoint_from_a_different_sweep_is_refused(tmp_path) -> None:
    """Splicing two runs into one artifact is worse than losing one of them.

    A resumed half-run would be reported as a single measurement and would not look like
    a failure, so anything whose token does not match exactly is discarded rather than
    adapted -- a different size, a different chunking, a different chemistry version.
    """
    written = CheckpointPlan(path=tmp_path / "audit.pickle", token=("v1", "100"))
    save_checkpoint(written, done=50, state=new_accumulator())

    other = CheckpointPlan(path=written.path, token=("v1", "2881"))
    with pytest.warns(UserWarning, match="different"):
        resumed, restored = load_checkpoint(other)

    assert resumed == 0
    assert restored.rdkit_outcomes_total == 0


def test_an_unreadable_checkpoint_starts_over_rather_than_failing(tmp_path) -> None:
    """A crash during the write must cost the resume, not the run."""
    plan = CheckpointPlan(path=tmp_path / "audit.pickle", token=("v1",))
    plan.path.write_bytes(b"truncated garbage")

    with pytest.warns(UserWarning, match="unreadable"):
        resumed, _ = load_checkpoint(plan)

    assert resumed == 0


def test_the_checkpoint_is_removed_once_the_artifact_is_complete(tmp_path) -> None:
    """Left behind, it would be resumed by the next sweep of the same shape."""
    plan = CheckpointPlan(path=tmp_path / "audit.pickle", token=("v1",))
    save_checkpoint(plan, done=50, state=new_accumulator())

    discard_checkpoint(plan)

    assert not plan.path.exists()
    assert load_checkpoint(plan)[0] == 0


def test_native_key_does_not_depend_on_how_hydrogens_were_spelled() -> None:
    """The property the model exists to provide, on the field that used to break it.

    ``h`` counts the hydrogens the input left implicit -- provenance, not molecular
    state. The canonical atom order ranked on it, so one molecule written two ways could
    order two ways and key two ways. It survived a benzene test because benzene is
    symmetric enough that ``h`` never broke a tie; an organotin with three equivalent
    butyl chains is not, and reaction-built precursors carry an ``h`` their re-read form
    does not. That is a false divergence in the audit's own identity.
    """
    explicit = cmol("CCC[CH2][Sn]([CH2]CCC)([CH2]CCC)[c]1nc2ccccc2s1")
    implicit = cmol("CCCC[Sn](CCCC)(CCCC)c1nc2ccccc2s1")

    assert native_key(explicit) == native_key(implicit)
    assert audit_key_from_cmol(explicit) == audit_key_from_cmol(implicit)


def test_native_key_separates_isotopes_that_the_ordering_used_to_tie() -> None:
    """Isotope is molecular state, and the ranking did not consider it.

    Two atoms differing only in mass number were tied, so the tie-break decided their
    order arbitrarily -- a key that could disagree with itself on the same molecule.
    """
    assert native_key(cmol("[13CH3]CO")) != native_key(cmol("C[13CH2]O"))
