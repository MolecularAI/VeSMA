"""Recursive SMARTS ``$(...)``: parsing, compilation into atom predicates, and CPU matching.

The corpus test holds the CPU matcher to RDKit exactly on every molecule whose aromaticity the
two toolkits perceive identically. Where they perceive it differently the answers may differ
for reasons recursion has nothing to do with -- the same pattern written flat disagrees the same
way -- so those molecules are left out rather than budgeted for.
"""

from __future__ import annotations

import random
from functools import cache
from pathlib import Path

import pytest

from vesma import cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.cpu_matcher import has_match
from vesma.gmol import GMol
from vesma.gmol_atoms import AROMATIC
from vesma.parsing import SmartsError
from vesma.pattern_identity import lhs_atom_key
from vesma.predicates import require_no_predicates
from vesma.reaction import UnsupportedReactionError, parse_reaction_smarts
from vesma.rejection import ChemistryRejection

Chem = pytest.importorskip("rdkit.Chem")
pytest.importorskip("rdkit.RDLogger").DisableLog("rdApp.*")

BENCHMARK_DIR = (
    Path(__file__).parents[1]
    / "data/external/zbh-smarts/processed/smarts-benchmark-v2.0/substructure_search_set"
)
CORPUS_FILES = (
    "literature_noHs_rec.smarts.lead-like.benchmarkset",
    "literature_noHs_rec.smarts.everything.benchmarkset",
)
SHARED_SAMPLE_SIZE = 40


def molecule(smiles: str) -> GMol:
    return gmol_from_cmol(cmol_from_smiles(smiles)[0])


def pattern(smarts: str) -> GMol:
    return gmol_from_smarts(smarts)[0]


def rdkit_matches(smarts: str, smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles).HasSubstructMatch(Chem.MolFromSmarts(smarts))


@pytest.mark.parametrize(
    ("smarts", "smiles"),
    [
        ("[C;$(C=O)]", "CC(=O)O"),
        ("[C;$(C=O)]", "CCO"),
        ("[N;!$(N=O);$(N[#6])]", "CN=O"),
        ("[N;!$(N=O);$(N[#6])]", "CNC"),
        ("[C,$(N=O)]", "ON=O"),
        ("[C,$(N=O)]", "NO"),
        ("[O;$(O[$(C=O)])]", "CC(=O)OC"),
        ("[O;$(O[$(C=O)])]", "CCOC"),
        ("[$([NX3,NX4+][CX4H]([*])[CX3](=[OX1])[O,N])]", "NC(C)C(=O)O"),
        ("[$([NX3H,NX4H2+])]", "CNC"),
        ("[$([NX3H,NX4H2+])]", "CN(C)C"),
        ("[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]", "CCO"),
        ("[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]", "CCCl"),
        ("C1CC[$(N(C)C)]CC1", "CN1CCCCC1"),
        ("C1CC[$(N(C)C)]CC1", "C1CCNCC1"),
        ("[$([C@H](N)C(=O)O)]", "N[C@@H](C)C(=O)O"),
        ("[C;$(C[$(C[$(C[$(CO)])])])]", "CCCCO"),
        ("[C;$(C[$(C[$(C[$(CO)])])])]", "CCCO"),
    ],
)
def test_recursive_smarts_matches_rdkit(smarts: str, smiles: str) -> None:
    assert has_match(pattern(smarts), molecule(smiles)) == rdkit_matches(smarts, smiles)


def test_single_atom_body_is_inlined_into_masks() -> None:
    inlined = pattern("[$([#7X3H])]")
    flat = pattern("[#7X3H]")
    assert inlined.predicates == ()
    assert lhs_atom_key(inlined, 0) == lhs_atom_key(flat, 0)


def test_element_by_number_keeps_hydrogen_a_count() -> None:
    assert has_match(pattern("[#7H]"), molecule("CNC"))
    assert not has_match(pattern("[#7H]"), molecule("CN(C)C"))


def test_nesting_depth_is_the_predicate_level() -> None:
    graph = pattern("[C;$(C[$(C[$(CO)])])]")
    (predicate,) = graph.predicates
    assert predicate.level == 3
    assert graph.atom_predicate.tolist() == [0]


def test_equal_predicates_are_shared_within_a_pattern() -> None:
    graph = pattern("[C;$(C=O)][C;$(C=O)]")
    assert len(graph.predicates) == 1
    assert graph.atom_predicate.tolist() == [0, 0]


def test_identity_tells_apart_rules_differing_only_in_recursion() -> None:
    assert lhs_atom_key(pattern("[C;$(C=O)]"), 0) != lhs_atom_key(
        pattern("[C;$(C#N)]"), 0
    )
    assert lhs_atom_key(pattern("[C;$(C=O)]"), 0) != lhs_atom_key(pattern("[C]"), 0)


def test_unclosed_recursion_is_a_syntax_error() -> None:
    with pytest.raises(SmartsError):
        pattern("[C;$(C=O]")


def test_recursion_on_the_built_side_of_a_reaction_is_refused() -> None:
    with pytest.raises(UnsupportedReactionError, match="right-hand side"):
        parse_reaction_smarts("[C:1](=O)[OH:2]>>[C;$(C=O):1][OH:2]")


def test_recursion_on_the_matched_side_of_a_reaction_parses() -> None:
    reaction = parse_reaction_smarts("[C;$(C=O):1][OH:2]>>[C:1][Cl:2]")
    assert len(reaction.lhs.predicates) == 1


def test_device_backends_refuse_recursion_rather_than_ignore_it() -> None:
    require_no_predicates(pattern("CC"), backend="Metal")
    with pytest.raises(SmartsError, match="Metal"):
        require_no_predicates(pattern("[C;$(C=O)]"), backend="Metal")


@cache
def corpus_rows() -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for name in CORPUS_FILES:
        for line in (BENCHMARK_DIR / name).read_text().splitlines():
            smarts, smiles = line.split()[:2]
            rows.append((smarts, smiles))
    return tuple(rows)


@cache
def shared_molecules() -> tuple[str, ...]:
    smiles = sorted({row[1] for row in corpus_rows()})
    return tuple(random.Random(1).sample(smiles, SHARED_SAMPLE_SIZE))


@cache
def comparable_molecule(smiles: str) -> GMol | None:
    """The molecule, if vesma reads it and perceives aromaticity exactly as RDKit does."""
    try:
        cmol = cmol_from_smiles(smiles)[0]
    except SmartsError, ChemistryRejection:
        return None
    rdkit = Chem.MolFromSmiles(smiles)
    ours = (cmol.A == AROMATIC).tolist()
    theirs = [atom.GetIsAromatic() for atom in rdkit.GetAtoms()]
    return gmol_from_cmol(cmol) if ours == theirs else None


def corpus_patterns() -> list[str]:
    return sorted({row[0] for row in corpus_rows()})


@pytest.mark.slow
@pytest.mark.parametrize("smarts", corpus_patterns())
def test_zbh_recursive_corpus_matches_rdkit(smarts: str) -> None:
    try:
        graph = pattern(smarts)
    except SmartsError as exc:
        pytest.skip(f"outside vesma's SMARTS grammar, unrelated to recursion: {exc}")
    query = Chem.MolFromSmarts(smarts)
    own = {row[1] for row in corpus_rows() if row[0] == smarts}
    mismatched = [
        smiles
        for smiles in sorted(own | set(shared_molecules()))
        if (target := comparable_molecule(smiles)) is not None
        and has_match(graph, target)
        != Chem.MolFromSmiles(smiles).HasSubstructMatch(query)
    ]
    assert mismatched == []
