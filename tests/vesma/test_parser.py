from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from vesma import (
    SmartsError,
    cmol_from_gmol,
    cmol_from_normalized_smarts,
    cmol_from_smiles,
    gmol_from_cmol,
    gmol_from_smarts,
    gmol_from_smiles,
)
from vesma.adjacency import bond_presence
from vesma.bonds import (
    BOND_KIND_MASK,
    BOND_TOPOLOGY_ANY,
    BOND_TOPOLOGY_CHAIN,
    BOND_TOPOLOGY_RING,
    QBOND_DEFAULT,
    BondKind,
    exact_bond_kind,
)
from vesma.configurations import ChemistryRejection
from vesma.cpu_matcher import has_match
from vesma.gmol_atoms import AROMATIC, MASS_ANY, bucket_mask_exact
from vesma.parsing import parse_smarts
from vesma.parsing.errors import NotMolecularSmilesError
from vesma.parsing.ir import NormalizedSmartsMol, PropVal
from vesma.symbols import ELEMENT_TO_Z


def _gmol(smarts: str):
    return gmol_from_smarts(smarts)[0]


def _gmol_with_info(smarts: str):
    return gmol_from_smarts(smarts)


def _cmol(smiles: str):
    return cmol_from_smiles(smiles)[0]


def _molecule(smiles: str):
    return gmol_from_cmol(_cmol(smiles))


def _smiles_gmol(smiles: str):
    """The query graph of the SMILES *reading*, before any chemistry runs.

    Needed because an open-shell literal refuses at the molecule stage, so its hydrogen
    count cannot be read off a finished CMol.
    """
    return gmol_from_smiles(smiles)[0]


def test_parse_smarts_returns_normalized_ir() -> None:
    parsed = parse_smarts("CCO")
    assert isinstance(parsed, NormalizedSmartsMol)
    assert parsed.atom_count == 3
    assert parsed.bonds.shape == (3, 3)
    assert parsed.smarts_string == "CCO"
    molecule, info = cmol_from_normalized_smarts(parsed)
    assert molecule.atom_count == 3
    assert info.smarts_string == "CCO"


def test_parse_smiles_linear_chain() -> None:
    molecule = _cmol("CCO")
    assert molecule.atom_count == 3
    assert molecule.bonds.shape == (3, 3)
    assert np.array_equal(
        molecule.bonds.astype(int), np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    )


def test_cmol_stores_exact_scalar_properties() -> None:
    molecule = _cmol("C")
    assert molecule.Z.tolist() == [6]
    assert molecule.A.tolist() == [1]
    assert molecule.C.tolist() == [0]
    assert molecule.D.tolist() == [0]
    assert molecule.H.tolist() == [4]
    assert molecule.h.tolist() == [4]
    assert molecule.v.tolist() == [4]
    assert molecule.X.tolist() == [4]


def test_parse_smiles_ring() -> None:
    graph = _molecule("C1CC1")
    assert graph.atom_masks.size == 3
    assert int(bond_presence(graph.bonds).sum()) == 6


def test_parse_smiles_unbracket_atom_infers_implicit_h() -> None:
    graph = _molecule("C")
    assert int(graph.atom_masks.D[0]) == bucket_mask_exact(0)
    assert int(graph.atom_masks.H[0]) == bucket_mask_exact(4)
    assert int(graph.atom_masks.h[0]) == bucket_mask_exact(4)
    assert int(graph.atom_masks.v[0]) == bucket_mask_exact(4)
    assert int(graph.atom_masks.X[0]) == bucket_mask_exact(4)


def test_parse_smiles_bracket_atom_without_h_carries_no_hydrogen() -> None:
    """OpenSMILES 1.0 section 3.1: "``[C]`` and ``[CH0]`` are identical"."""
    assert int(_smiles_gmol("[C]").atom_masks.H[0]) == bucket_mask_exact(0)
    assert int(_smiles_gmol("[CH0]").atom_masks.H[0]) == bucket_mask_exact(0)
    assert int(_smiles_gmol("[CH]").atom_masks.H[0]) == bucket_mask_exact(1)
    assert int(_smiles_gmol("[CH1]").atom_masks.H[0]) == bucket_mask_exact(1)


def test_parse_smiles_unbracketed_atom_is_not_its_bracketed_spelling() -> None:
    """The bracket is the whole signal: ``N`` is ammonia, ``[N]`` is atomic nitrogen."""
    assert int(_smiles_gmol("N").atom_masks.H[0]) == 0xFF
    assert int(_smiles_gmol("[N]").atom_masks.H[0]) == bucket_mask_exact(0)


@pytest.mark.parametrize("smiles", ["[C]", "[CH0]", "[N]", "[O]"])
def test_bracket_atom_without_hydrogen_is_open_shell_and_refused(smiles: str) -> None:
    """Atomic nitrogen is not ammonia, and the model declines to invent one."""
    with pytest.raises(ChemistryRejection):
        cmol_from_smiles(smiles)


def test_smarts_leaves_an_omitted_hydrogen_count_unconstrained() -> None:
    """The same text, read as a query, must not gain the molecule's ``H0``.

    Daylight: "in SMARTS, unspecified properties are not defined to be part of the
    pattern." So ``[N]`` still matches ammonia while ``[NH0]`` does not.
    """
    assert int(_gmol("[N]").atom_masks.H[0]) == 0xFF
    assert int(_gmol("[NH0]").atom_masks.H[0]) == bucket_mask_exact(0)


@pytest.mark.parametrize("smarts", ["[C;!H0]", "[C;H1,H2]", "[C;R]", "[C;D2]", "[C,N]"])
def test_smiles_reading_refuses_a_query(smarts: str) -> None:
    """A query cannot be concretized by default.

    ``exact_h_total`` answers -1 for a mask that is not a single bit and ``exact_c``
    answers 0 for any mask including neutral, so without this check each of these
    silently became methane.
    """
    with pytest.raises(NotMolecularSmilesError):
        cmol_from_smiles(smarts)


@pytest.mark.parametrize("smiles", ["[0S]", "[1000S]"])
def test_smiles_reading_refuses_an_impossible_isotope(smiles: str) -> None:
    with pytest.raises(NotMolecularSmilesError):
        cmol_from_smiles(smiles)


def test_parse_smiles_degree_excludes_explicit_h() -> None:
    graph = _molecule("[CH3]C")
    assert int(graph.atom_masks.D[0]) == bucket_mask_exact(1)
    assert int(graph.atom_masks.H[0]) == bucket_mask_exact(3)
    assert int(graph.atom_masks.h[0]) == bucket_mask_exact(0)
    assert int(graph.atom_masks.v[0]) == bucket_mask_exact(4)
    assert int(graph.atom_masks.X[0]) == bucket_mask_exact(4)


@pytest.mark.parametrize("bond_token", ["~", "@"])
def test_parse_smiles_rejects_any_bond_tokens(bond_token: str) -> None:
    with pytest.raises(SmartsError, match="exact bond types"):
        _molecule(f"C{bond_token}C")


@pytest.mark.parametrize("bond_token", ["/?", "\\?"])
def test_parse_smiles_accepts_directional_optional_bond_tokens(bond_token: str) -> None:
    graph = _molecule(f"C{bond_token}C")
    assert graph.bonds[0, 1] == (int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY)


def test_parse_smiles_rejects_invalid_bond_token() -> None:
    with pytest.raises(SmartsError):
        _gmol("C!C")


def test_parse_smarts_any_bond_maps_to_any() -> None:
    graph = _gmol("C~C")
    assert int(graph.bonds[0, 1]) == (BOND_KIND_MASK | BOND_TOPOLOGY_ANY)


def test_parse_smarts_ring_bond_maps_to_any() -> None:
    graph = _gmol("C@C")
    assert int(graph.bonds[0, 1]) == (BOND_KIND_MASK | BOND_TOPOLOGY_RING)


@pytest.mark.parametrize(
    ("pattern", "expected_flags", "expected_type"),
    [
        (
            "C-!@C",
            int(BondKind.SINGLE) | BOND_TOPOLOGY_CHAIN,
            None,
        ),
        ("C=@C", int(BondKind.DOUBLE) | BOND_TOPOLOGY_RING, None),
        (
            "[C]-;!@[C]",
            int(BondKind.SINGLE) | BOND_TOPOLOGY_CHAIN,
            None,
        ),
        (
            "C=,:C",
            int(BondKind.DOUBLE) | int(BondKind.DELOCALIZED) | BOND_TOPOLOGY_ANY,
            None,
        ),
        (
            "C:!@C",
            int(BondKind.DELOCALIZED) | BOND_TOPOLOGY_CHAIN,
            None,
        ),
        (
            "*-!:aa-!:*",
            int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY,
            BondKind.SINGLE,
        ),
        (
            "C!@C",
            BOND_KIND_MASK | BOND_TOPOLOGY_CHAIN,
            None,
        ),
    ],
)
def test_parse_smarts_preserves_bond_query_flags(
    pattern: str,
    expected_flags: int,
    expected_type: BondKind | None,
) -> None:
    graph = _gmol(pattern)
    assert int(graph.bonds[0, 1]) == expected_flags
    assert exact_bond_kind(int(graph.bonds[0, 1])) == expected_type


def test_parse_smarts_implicit_bond_maps_to_default() -> None:
    # An omitted/default SMARTS bond accepts single OR aromatic (Daylight semantics),
    # encoded as the QBOND_DEFAULT accepted-class mask (distinct from explicit ``-``).
    graph = _gmol("CC")
    assert graph.bonds[0, 1] == QBOND_DEFAULT


def test_parse_smarts_implicit_aromatic_bond_maps_to_default() -> None:
    graph = _gmol("cc")
    assert graph.bonds[0, 1] == QBOND_DEFAULT


def test_parse_smarts_explicit_aromatic_single_bond_stays_single() -> None:
    graph = _gmol("c-c")
    assert graph.bonds[0, 1] == (int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY)


@pytest.mark.parametrize("smarts", ["a1aa1", "a-[Cl]", "[ND1H2]a"])
def test_parse_smarts_accepts_common_aromatic_patterns(smarts: str) -> None:
    graph = _gmol(smarts)
    assert graph.atom_count >= 2


@pytest.mark.parametrize(
    "smarts",
    [
        "C[C@](F)(Cl)Br",
        "C[C@@H](F)Cl",
        "C[C@?](F)(Cl)Br",
        "C[C@?H](Cl)Br",
    ],
)
def test_parse_smarts_ignores_atom_chirality(smarts: str) -> None:
    graph = _gmol(smarts)
    assert graph.atom_count >= 3


@pytest.mark.parametrize("bond_token", ["/", "\\", "/?", "\\?"])
def test_parse_smarts_directional_bonds_map_to_single(bond_token: str) -> None:
    graph = _gmol(f"C{bond_token}C")
    assert graph.bonds[0, 1] == (int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY)


def test_parse_smarts_rejects_exclamation_bond_token() -> None:
    with pytest.raises(SmartsError):
        _gmol("C!C")


@pytest.mark.parametrize("smarts", ["C=&@C", "C-&!@C"])
def test_parse_smarts_rejects_rdkit_alias_bond_tokens(smarts: str) -> None:
    with pytest.raises(SmartsError):
        _gmol(smarts)


def test_parse_smarts_rejects_bare_question_bond_token() -> None:
    with pytest.raises(SmartsError):
        _gmol("C?C")


def test_parse_smarts_rejects_leading_bond() -> None:
    with pytest.raises(SmartsError):
        _gmol("=C")


def test_parse_smarts_rejects_dangling_bond() -> None:
    with pytest.raises(SmartsError):
        _gmol("C=")


def test_parse_smarts_dot_keeps_ring_open_state() -> None:
    graph = _gmol("C1.O2.C12")
    assert graph.atom_masks.size == 3
    # Default (no-symbol) ring-closure bonds are accepted-class masks: single or aromatic.
    assert graph.bonds[2, 0] == QBOND_DEFAULT
    assert graph.bonds[2, 1] == QBOND_DEFAULT


def test_parse_smarts_rejects_conflicting_ring_closure_bonds() -> None:
    with pytest.raises(SmartsError):
        _gmol("C=1CCCCC-1")


@pytest.mark.parametrize("pattern", ["C.1CCCCC=1", "C=1CCCCC.1"])
def test_parse_smarts_rejects_conflicting_ring_none_and_bond(pattern: str) -> None:
    with pytest.raises(SmartsError):
        _gmol(pattern)


def test_parse_smarts_simple() -> None:
    graph = _gmol("[CH2][OH]")
    assert graph.atom_masks.size == 2
    assert int(graph.bonds[0, 1]) != 0


def test_cmol_from_gmol_rejects_non_single_atomic_number_masks() -> None:
    with pytest.raises(ValueError, match="single-bit atomic number mask"):
        cmol_from_gmol(_gmol("[C,N]"))


def test_parse_smarts_logical_operators() -> None:
    graph = _gmol("[c,n;H1]")
    assert graph.atom_masks.size == 1


def test_parse_smarts_accepts_periodic_table_tokens() -> None:
    graph = _gmol("[Se]")
    assert graph.atom_masks.size == 1


def test_parse_smarts_accepts_bracketed_aromatic_selenium() -> None:
    graph = _gmol("[se]")
    assert graph.atom_masks.size == 1
    assert int(graph.atom_masks.Z_lo[0]) & (1 << ELEMENT_TO_Z["Se"])
    assert int(graph.atom_masks.A[0]) == AROMATIC


def test_parse_smarts_accepts_paroutes_selenium_target() -> None:
    graph = _gmol("CCCCCc1ccc(-c2ccc(-c3ccc(CC)cc3)cc2F)[se]1")
    assert graph.atom_masks.size == 25


def test_parse_smarts_encodes_mass_primitive() -> None:
    graph, info = _gmol_with_info("[C13]")
    assert graph.atom_masks.size == 1
    assert int(graph.atom_masks.Z_lo[0]) & (1 << 6)
    assert int(graph.atom_masks.M[0]) == 13
    assert info.smarts_string == "[C13]"
    assert info.messages == []


def test_parse_smarts_leaves_an_unlabelled_atom_unconstrained() -> None:
    graph, _ = _gmol_with_info("C")
    assert int(graph.atom_masks.M[0]) == MASS_ANY


def test_parse_smarts_reads_a_negated_mass_as_a_predicate() -> None:
    # `!` binds the mass alone (Daylight precedence): a carbon that is not carbon-13. Isotope is
    # exact-or-any, so "not 13" has no mask and is a predicate; the carbon stays in the masks.
    graph = _gmol("[!13C]")
    assert graph.predicates
    assert int(graph.atom_masks.Z_lo[0]) == 1 << 6
    assert has_match(graph, _molecule("C"))
    assert not has_match(graph, _molecule("[13CH4]"))


@pytest.mark.parametrize(
    "token", ["+", "++", "-", "--", "+0", "+9", "-0", "-9", "+10", "-25"]
)
def test_parse_smarts_supported_charge_primitives(token: str) -> None:
    graph = _gmol(f"[{token}]")
    assert graph.atom_masks.size == 1


def test_parse_smarts_rejects_unknown_element_token() -> None:
    with pytest.raises(SmartsError):
        _gmol("[Q]")


def test_parse_smarts_rejects_lowercase_d_bucket() -> None:
    with pytest.raises(SmartsError):
        _gmol("[d1]")


def test_parse_smarts_ties_properties_across_alternatives() -> None:
    # Aliphatic carbon or aromatic nitrogen: one clause of two boxes, not four literal clauses.
    graph = _gmol("[C,n]")
    (predicate,) = graph.predicates
    (clause,) = predicate.expression
    assert len(clause) == 2
    # The hull still narrows the masks: carbon or nitrogen, of either aromaticity.
    assert int(graph.atom_masks.Z_lo[0]) == (1 << 6) | (1 << 7)
    assert has_match(graph, _molecule("CC"))
    assert has_match(graph, _molecule("c1ccncc1"))
    assert not has_match(graph, _molecule("c1ccccc1"))
    assert not has_match(graph, _molecule("N"))


def test_parse_smarts_default_ring_primitives() -> None:
    graph = _gmol("[Rrxh]")
    assert int(graph.atom_masks.R[0]) & (1 << 1)
    assert int(graph.atom_masks.r[0]) & (1 << 1)
    assert int(graph.atom_masks.x[0]) & (1 << 1)
    assert int(graph.atom_masks.h[0]) & (1 << 1)


def test_parse_smarts_negation_keeps_other_fields_permissive() -> None:
    graph = _gmol("[!#1]")
    assert int(graph.atom_masks.Z_lo[0]) & (1 << 6)
    assert not (int(graph.atom_masks.Z_lo[0]) & (1 << 1))
    assert int(graph.atom_masks.A[0]) == 0b11
    assert int(graph.atom_masks.C[0]) == 0xFF


def test_parse_smarts_recursive_kept_as_literal() -> None:
    (clause,) = parse_smarts("[$([CX3]=[OX1])]").atoms[0].expression
    assert clause == frozenset({frozenset({PropVal("$", "[CX3]=[OX1]")})})


def test_parse_smarts_unsatisfiable_expression_rejected() -> None:
    with pytest.raises(SmartsError):
        _gmol("[C;N]")


def test_parse_smarts_inlined_recursion_is_unsatisfiable() -> None:
    with pytest.raises(SmartsError):
        _gmol("[C;$(N)]")


def test_parse_smarts_reaction_rejected() -> None:
    with pytest.raises(SmartsError):
        parse_smarts("[CH3:1][OH:2]>>[CH2:1]=[O:2]")


def test_parse_smarts_reaction_missing_lhs_rejected() -> None:
    with pytest.raises(SmartsError):
        parse_smarts(">>[#6]")


def test_parse_smiles_preserves_atom_maps() -> None:
    molecule = _cmol("[CH3:7][OH:8]")
    assert molecule.atom_map.tolist() == [7, 8]


def _iter_patterns_from_tar(path: Path):
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".smarts"):
                continue
            payload = archive.extractfile(member)
            if payload is None:
                continue
            for raw in payload:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("#"):
                    continue
                yield line.split()[0]


def _iter_patterns_from_zip(path: Path):
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".smarts"):
                continue
            payload = archive.read(member).decode("utf-8", errors="ignore")
            for line in payload.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                yield stripped.split()[0]


@pytest.mark.slow
@pytest.mark.parametrize(
    "dataset_name",
    [
        "daylight-smarts.tar.gz",
        "pains-smarts.tar.gz",
        "smarts-benchmark-v2.0.zip",
    ],
)
def test_subset_parser_accepts_real_corpus_examples(
    dataset_name: str, smarts_dataset_dir: Path
) -> None:
    dataset = smarts_dataset_dir / "raw" / dataset_name
    patterns = (
        _iter_patterns_from_zip(dataset)
        if dataset.suffix == ".zip"
        else _iter_patterns_from_tar(dataset)
    )

    parsed = 0
    scanned = 0
    for pattern in patterns:
        scanned += 1
        if scanned > 5000 or parsed >= 100:
            break
        try:
            parse_smarts(pattern)
        except SmartsError:
            continue
        parsed += 1

    assert parsed >= 20
