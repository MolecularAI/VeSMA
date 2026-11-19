from __future__ import annotations

import pytest

from tests.datasets import load_retro_benchmark_corpus
from vesma import find_matches

KNOWN_HIT_CLASS_EXCEPTIONS = frozenset({"5"})


def test_retro_benchmark_corpus_is_parseable_and_balanced() -> None:
    corpus = load_retro_benchmark_corpus()

    assert len(corpus.patterns) == 16
    assert len(corpus.molecules) == 128
    assert sum(pattern.hit_class == "positive" for pattern in corpus.patterns) == 8
    assert sum(pattern.hit_class == "negative" for pattern in corpus.patterns) == 8

    saw_match = False
    saw_no_match = False
    for pattern in corpus.patterns:
        assert pattern.retro_template.partition(">>")[0].strip() == pattern.lhs_smarts
        assert pattern.lhs_atom_count == pattern.query.atom_count
        assert pattern.lhs_atom_count <= 64

        hit_count = 0
        for molecule in corpus.molecules:
            assert molecule.atom_count == molecule.molecule.atom_count
            assert molecule.atom_count <= 64

            matched = (
                find_matches(pattern.query, molecule.molecule, return_first=True).status
                == "match"
            )
            hit_count += int(matched)
            saw_match = saw_match or matched
            saw_no_match = saw_no_match or (not matched)

        if pattern.hit_class == "positive":
            assert hit_count > 0
        else:
            assert hit_count == 0 or pattern.template_code in KNOWN_HIT_CLASS_EXCEPTIONS

    assert saw_match
    assert saw_no_match


@pytest.mark.parametrize(
    ("template_code", "chembl_id"),
    [
        ("10", "399328"),
        ("10", "1408234"),
        ("5", "153180"),
    ],
)
def test_retro_benchmark_known_rdkit_positives_match_after_aromatic_fix(
    template_code: str, chembl_id: str
) -> None:
    chem = pytest.importorskip("rdkit.Chem")
    corpus = load_retro_benchmark_corpus()

    pattern = next(
        entry for entry in corpus.patterns if entry.template_code == template_code
    )
    molecule = next(entry for entry in corpus.molecules if entry.chembl_id == chembl_id)
    rd_query = chem.MolFromSmarts(pattern.lhs_smarts)
    rd_mol = chem.MolFromSmiles(molecule.smiles)

    assert rd_query is not None
    assert rd_mol is not None
    assert rd_mol.HasSubstructMatch(rd_query)
    assert (
        find_matches(pattern.query, molecule.molecule, return_first=True).status
        == "match"
    )
