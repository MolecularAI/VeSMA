from collections import Counter

import pytest
from rdkit import Chem

from vesma import cmol_from_smiles
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.keyer import rdkit_canonical_smiles
from vesma.rdkit_matcher import RdkitReactionMatcher, rdkit_reactions_from_smarts
from vesma.reaction import parse_reaction_smarts

RULES = ("[C:1][O:2]>>[C:1].[O:2]", "[C:1][C:2]>>[C:1].[C:2]")
SMILES = ("CCO", "CCC", "c1ccccc1CO", "[13CH3]O")


@pytest.mark.parametrize("backend,cap", (("cpu", None), ("cpu", 2), ("rdkit", None)))
@pytest.mark.parametrize("jobs", (1, 2))
@pytest.mark.parametrize("batch_size", (1, 3))
def test_streamed_reactions_equal_joined(
    backend: str, cap: int | None, jobs: int, batch_size: int
) -> None:
    if backend == "cpu":
        matcher = CpuReactionMatcher(
            [parse_reaction_smarts(rule) for rule in RULES], max_precursor_atoms=cap
        )
        mols = [cmol_from_smiles(smiles)[0] for smiles in SMILES]
        key = rdkit_canonical_smiles
        kwargs = {"n_jobs": jobs}
    else:
        matcher = RdkitReactionMatcher(rdkit_reactions_from_smarts(RULES), n_jobs=jobs)
        mols = [Chem.MolFromSmiles(smiles) for smiles in SMILES]
        key = Chem.MolToSmiles
        kwargs = {}
    try:
        precursors, hits = matcher.match(mols, **kwargs)
        expected = Counter((*row[:3], key(precursors[row[3]])) for row in hits)
        expected_diagnostics = matcher.last_diagnostics
        assert expected_diagnostics is not None
        if backend == "cpu":
            expected_rejected = matcher.last_rejected_applications.copy()
        actual = Counter()
        for precursors, hits in matcher.match_batches(
            mols, pairs_per_batch=batch_size, **kwargs
        ):
            actual.update((*row[:3], key(precursors[row[3]])) for row in hits)
        assert actual == expected
        assert matcher.last_diagnostics == expected_diagnostics
        if backend == "cpu":
            assert (matcher.last_rejected_applications == expected_rejected).all()
        assert list(matcher.match_batches(mols, [], **kwargs)) == []
        with pytest.raises(ValueError, match="positive"):
            list(matcher.match_batches(mols, pairs_per_batch=0, **kwargs))
        iterator = matcher.match_batches(mols, pairs_per_batch=1, **kwargs)
        next(iterator)
        iterator.close()
        assert matcher.last_diagnostics is not None
        if backend == "cpu":
            assert (
                0
                < matcher.last_diagnostics.applications_attempted
                < expected_diagnostics.applications_attempted
            )
        else:
            assert matcher.last_diagnostics.pairs_evaluated == 1
    finally:
        if backend == "rdkit":
            matcher.close()


def test_unkeyed_rdkit_stream_does_not_canonicalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mols = [Chem.MolFromSmiles("CCO")]
    matcher = RdkitReactionMatcher(rdkit_reactions_from_smarts(RULES), n_jobs=1)

    def unexpected_key(*args: object, **kwargs: object) -> str:
        pytest.fail("Unkeyed application must not compute canonical SMILES")

    monkeypatch.setattr(Chem, "MolToSmiles", unexpected_key)
    with matcher:
        batches = list(matcher.match_batches(mols, pairs_per_batch=1))
    assert sum(len(hits) for _, hits in batches) > 0
