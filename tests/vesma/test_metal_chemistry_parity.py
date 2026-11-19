"""Device chemistry measured against the CPU model it is a port of.

The CPU chemistry is frozen precisely so the GPU ports have a fixed target, so these tests
ask the device what it thinks and compare that to ``vesma`` itself rather than to a golden
file. A golden file would record what the device did on the day it was written; comparing
against the model records whether the device still agrees with it.
"""

from __future__ import annotations

import shutil
import sys

import numpy as np
import pytest

from vesma.configurations import (
    CONFIG_CHARGE_MAX,
    CONFIG_CHARGE_MIN,
    CONFIG_CHARGE_SPAN,
    CONFIG_PI_BIAS,
    CONFIG_Z_SPAN,
    Configuration,
    ConfigurationFailure,
    configuration_table,
    configurations_for,
    select_circuit_configuration,
    select_configuration,
    unwritten_hydrogen_configuration,
    unwritten_hydrogens,
)

pytestmark = [pytest.mark.gpu, pytest.mark.metal]


def skip_if_no_metal_toolchain() -> None:
    if sys.platform != "darwin":
        pytest.skip("Metal backend requires macOS.")
    if shutil.which("clang++") is None:
        pytest.skip("Metal backend requires clang++.")
    try:
        from vesma.metal_matcher import probe_metal

        probe_metal()
    except Exception as exc:  # pragma: no cover - depends on host GPU
        pytest.skip(f"Metal backend unavailable: {exc}")


# Every element the configuration table declares, plus states it declares nothing about --
# an undeclared element must come back as a *different* failure from a declared element
# whose states do not fit, and only exercising both proves the device keeps them apart.
DECLARED_Z = (5, 6, 7, 8, 9, 14, 15, 16, 17, 32, 33, 34, 35, 50, 51, 52, 53)
METAL_Z = (3, 11, 12, 19, 25, 29, 30, 50, 76)
UNDECLARED_Z = (0, 1, 2, 13, 26, 47, 99)
PROBE_Z = tuple(sorted(set(DECLARED_Z + METAL_Z + UNDECLARED_Z)))


def _probe_domain() -> np.ndarray:
    """Written atom states spanning the domain the packed atom words can express."""
    grids = np.meshgrid(
        np.array(PROBE_Z, dtype=np.int32),
        np.arange(-3, 5, dtype=np.int32),  # the mask_for_charge clamp domain
        np.arange(0, 6, dtype=np.int32),  # sigma degree
        np.arange(0, 7, dtype=np.int32),  # bond-order sum
        np.arange(0, 5, dtype=np.int32),  # total hydrogen
        np.arange(0, 2, dtype=np.int32),  # delocalized
        indexing="ij",
    )
    return np.stack([grid.reshape(-1) for grid in grids], axis=1).astype(np.int32)


def _expected_status(outcome: Configuration | ConfigurationFailure) -> int:
    """The device status word: zero for a selection, otherwise the failure, shifted."""
    if isinstance(outcome, ConfigurationFailure):
        return int(outcome) + 1
    return 0


def _expected_row(
    outcome: Configuration | ConfigurationFailure,
) -> tuple[int, int, int]:
    if isinstance(outcome, ConfigurationFailure):
        return (0, 0, 0)
    return (
        int(outcome.sigma_capacity),
        int(outcome.graph_valence),
        int(outcome.pi) + 2,
    )


def test_device_configuration_table_matches_the_model() -> None:
    """The packed table reproduces ``configurations_for`` entry for entry.

    Packing is where a table stops being the thing it was generated from, so this checks
    the bytes rather than trusting the generator.
    """
    index, rows = configuration_table()
    for z in range(CONFIG_Z_SPAN):
        for charge in range(CONFIG_CHARGE_MIN, CONFIG_CHARGE_MAX + 1):
            slot = int(index[z * CONFIG_CHARGE_SPAN + (charge - CONFIG_CHARGE_MIN)])
            offset, count = slot & 0xFFFF, slot >> 16
            expected = configurations_for(z, charge)
            assert count == len(expected), (z, charge)
            for position, configuration in enumerate(expected):
                word = int(rows[offset + position])
                assert (
                    word & 0xFF,
                    (word >> 8) & 0xFF,
                    ((word >> 16) & 0xFF) - CONFIG_PI_BIAS,
                ) == (
                    configuration.sigma_capacity,
                    configuration.graph_valence,
                    int(configuration.pi),
                )


def test_device_selectors_agree_with_the_configuration_model() -> None:
    """All three selectors, on device, over the whole expressible domain.

    Both what was selected *and* which failure was reported: an undeclared element and a
    declared one whose states do not fit are different answers, and collapsing them would
    turn a gap in the model into a claim about the molecule.
    """
    skip_if_no_metal_toolchain()
    from vesma.metal_retro import probe_configurations

    queries = _probe_domain()
    observed = probe_configurations(queries)

    for query, row in zip(queries.tolist(), observed.tolist(), strict=True):
        z, charge, sigma_degree, bond_order_sum, total_h, delocalized = query
        context = (
            f"z={z} charge={charge} d={sigma_degree} b={bond_order_sum} h={total_h}"
        )

        selected = select_configuration(
            z=z,
            charge=charge,
            sigma_degree=sigma_degree,
            bond_order_sum=bond_order_sum,
            total_h=total_h,
            delocalized=bool(delocalized),
        )
        expected = _expected_row(selected)
        assert row[0] == _expected_status(selected), f"select_configuration {context}"
        assert tuple(row[1:4]) == expected, f"select_configuration {context}"

        circuit = select_circuit_configuration(
            z=z, charge=charge, sigma_degree=sigma_degree, total_h=total_h
        )
        expected = _expected_row(circuit)
        assert row[4] == _expected_status(circuit), f"select_circuit {context}"
        assert tuple(row[5:8]) == expected, f"select_circuit {context}"

        unwritten = unwritten_hydrogen_configuration(
            z=z,
            charge=charge,
            sigma_degree=sigma_degree,
            bond_order_sum=bond_order_sum,
            delocalized=bool(delocalized),
        )
        expected = _expected_row(unwritten)
        assert row[8] == _expected_status(unwritten), f"unwritten {context}"
        assert tuple(row[9:12]) == expected, f"unwritten {context}"
        if not isinstance(unwritten, ConfigurationFailure):
            hydrogens = unwritten_hydrogens(
                unwritten,
                sigma_degree=sigma_degree,
                bond_order_sum=bond_order_sum,
                delocalized=bool(delocalized),
            )
            assert row[12] == hydrogens, f"unwritten hydrogens {context}"


def test_the_isoelectronic_shift_survives_the_bake() -> None:
    """An O+ does not inherit nitrogen's octet expansion.

    The shift governs valence (looked up at ``z - charge``) but not expansion eligibility
    or outer electrons (the atom's own, at ``z``). A device that re-derived the rule could
    apply it uniformly and quietly gain a hypervalent oxygen; one that reads the baked
    answer cannot. Worth pinning explicitly because the two halves look symmetric.
    """
    skip_if_no_metal_toolchain()
    from vesma.metal_retro import probe_configurations

    # O+ shares nitrogen's electron count, so it reaches valence 3 -- but never nitrogen's
    # absent expansion, and never phosphorus's valence 5.
    assert {c.graph_valence for c in configurations_for(8, 1)} == {3}
    # S+ shares phosphorus's count and *does* reach phosphorus's expanded valence 5, which
    # is the sulfoxonium of the Corey-Chaykovsky reagent.
    assert 5 in {c.graph_valence for c in configurations_for(16, 1)}

    queries = np.array(
        [
            [8, 1, 3, 5, 0, 0],  # a hypervalent O+ that must not exist
            [16, 1, 3, 5, 0, 0],  # the sulfoxonium S+ that must
        ],
        dtype=np.int32,
    )
    observed = probe_configurations(queries)
    assert observed[0][0] != 0, "device accepted a valence-5 oxygen cation"
    assert observed[1][0] == 0, "device refused the sulfoxonium sulfur"


def test_the_device_screen_agrees_with_the_host_on_the_shipped_veto_table() -> None:
    """A veto has to mean the same thing on both arms, or it removes different chemistry on each.

    The arms enumerate differently -- that is documented and expected -- so the parity that matters
    is not how many applications each offers but whether the *same* pattern on the *same* molecule
    gets the same answer. Every shipped row against molecules chosen to make each of them fire and
    to make each of them stay quiet, which is the only comparison that isolates the screen.
    """
    skip_if_no_metal_toolchain()

    from pathlib import Path

    from vertsy.filters import host_screen, read_vetoes
    from vesma import cmol_from_smiles
    from vesma.metal_reaction_matcher import metal_screen, metal_stack_mols

    vetoes = read_vetoes(
        Path(__file__).resolve().parents[1]
        / "data"
        / "templates"
        / "templates-vetoes.csv"
    )
    smiles = (
        "CC(=O)Cl",  # B 6: an acyl chloride
        "c1ccccc1CBr",  # B 5: a benzylic halide
        "OB(O)c1ccccc1",  # B 1: boron
        "CCOC(C)=O",  # B 2 / B 3: an ester
        "CC(=O)NC",  # B 4: an amide
        "OB(O)c1ccc(CBr)cc1",  # B 1 and B 5 at once
        "COC(=O)CCC(=O)NC",  # B 2 / B 3 and B 4 at once
        "ClC(=O)c1ccccc1N",  # B 6 beside an aniline
        "c1ccccc1",  # nothing
        "CC#N",  # a nitrile, which no row names
    )
    molecules = [cmol_from_smiles(text)[0] for text in smiles]
    pairs = [
        (veto, row) for veto in range(len(vetoes)) for row in range(len(molecules))
    ]

    host = np.asarray(host_screen(vetoes)(molecules, pairs), dtype=bool)
    device = np.asarray(
        metal_screen([veto.pattern for veto in vetoes])(
            metal_stack_mols(molecules), pairs
        ),
        dtype=bool,
    )

    assert host.tolist() == device.tolist()
    # Guard against the test passing because neither arm found anything -- stated as a floor on
    # how many distinct *rows* fired rather than a total, so adding a veto cannot rot it.
    fired = np.asarray(host, dtype=bool).reshape(len(vetoes), len(molecules))
    assert int(fired.any(axis=1).sum()) >= 5, "too few rows exercised to prove parity"


def test_a_second_ring_oxo_is_a_known_rdkit_boundary_not_a_silent_loss() -> None:
    """VeSMA writes a ring RDKit will not kekulise, and both answers are on the record.

    `MC 82b` writes a chloroazine back to its ring-oxo tautomer, which is the ordinary way to
    write a 2-pyridone. On a ring that *already* carries an exocyclic oxo it produces a six-ring
    with two carbonyls, one N-H and one bridging nitrogen: VeSMA perceives a circuit and writes
    it, RDKit finds no Kekule form and declines. The species is a real 2,4-dione; it is simply not
    an aromatic ring. See `docs/RDKit-Divergences.md`.

    Pinned so that a change in aromaticity perception is noticed rather than discovered by a
    report losing a molecule's name.
    """
    from vertsy.search import object_mol_array
    from vesma import cmol_from_smiles, reaction_patterns_from_smarts
    from vesma.cpu_matcher import CpuReactionMatcher
    from vesma.keyer import cmol_rdkit_smiles_keyer, cmol_smiles_keyer

    pattern = reaction_patterns_from_smarts(
        ("[c:1](-[Cl]):[n;H0:2]>>[c:1](=[O]):[nH:2]",)
    )
    matcher = CpuReactionMatcher(pattern)
    substrate = cmol_from_smiles("Cc1cccc2nc(Cl)c(I)c(=O)n12")[0]
    precursors, _hits = matcher.match(object_mol_array([substrate]), [(0, 0)])

    (product,) = list(precursors)
    assert "c(=O)" in cmol_smiles_keyer.encode(product), "VeSMA writes the ring"
    with pytest.raises(Exception, match="kekulize"):
        cmol_rdkit_smiles_keyer.encode(product)


GROUPED_MATCH_CASES = (
    ("CCC.NNN", "NNNCCC"),
    ("(CCC).(NNN)", "NNNCCC"),
    ("(CCC).(NNN)", "NNNOCCC"),
    ("(CCC).(NNN)", "CCC.NNN"),
    ("(CCC.NNN)", "NNNCCC"),
    ("(CCC.NNN)", "CCC.NNN"),
    ("C1.C12.C2", "CCC"),
    ("(C).(N)", "CN"),
    ("(C).(N)", "C.N"),
    ("(C.N)", "CN"),
    ("(C.N)", "C.N"),
    ("C.(N)", "CN"),
    ("([C:1]-[Cl]).([N:2])", "ClCCN"),
    ("([C:1]-[Cl]).([N:2])", "ClCC.N"),
    # The one valid placement is the cross-component one: the same `CCC` also sits inside
    # the first component, where the pattern forbids it. A device that only filtered
    # complete assignments after collecting them could lose this to its assignment cap.
    ("(CCC).(NNN)", "NNNCCC.CCC"),
    ("(CCC).(NNN)", "NNNCCC.CC"),
)


@pytest.mark.parametrize(("pattern", "target"), GROUPED_MATCH_CASES)
def test_the_device_reads_zero_level_grouping_as_the_host_does(
    pattern: str, target: str
) -> None:
    """A group is relational, so it could not ride in the atom masks the device already has.

    `atom_masks_overlap` ANDs every word from the fourth upward against the molecule's own
    value for that atom, and a group id has no molecule counterpart -- it would be ANDed
    against zero and nothing would match. So it travels as its own pattern buffer, and the
    target's components are derived on device from the adjacency already resident there.
    """
    skip_if_no_metal_toolchain()
    from vesma import gmol_from_smarts, gmol_from_smiles
    from vesma.cpu_matcher import CpuMatcher
    from vesma.metal_matcher import MetalMatcher

    query, _info = gmol_from_smarts(pattern)
    molecule, _info = gmol_from_smiles(target)

    assert MetalMatcher().has_match(query, molecule) == CpuMatcher().has_match(
        query, molecule
    )


@pytest.mark.parametrize(
    "smarts",
    [
        "[C:1]-[N:2]>>[C:1]-[Cl].[N:2]",
        "[C:1]-[N:2]>>([C:1]-[Cl]).([N:2])",
        "[C:1]-[N:2]>>([C:1]-[Cl].[N:2])",
    ],
)
def test_the_device_refuses_a_broken_rhs_grouping_as_the_host_does(smarts: str) -> None:
    """One disconnection, three spellings, two targets -- a ring and a chain.

    The ungrouped form does both readings; `(B).(C)` keeps only the genuine bimolecular cut
    and refuses the ring opening; `(B.C)` keeps only the ring opening. The device decides
    this itself, in the kernel, beside the orphaned-component test it already runs, because
    the host never sees the assignments it refused.
    """
    skip_if_no_metal_toolchain()
    from collections import Counter

    from vesma import cmol_from_smiles, parse_reaction_smarts
    from vesma.cpu_matcher import CpuReactionMatcher
    from vesma.keyer import short_ring_wl_molecule_id
    from vesma.metal_reaction_matcher import (
        MetalReactionMatcher,
        metal_short_ring_wl_key,
        metal_stack_mols,
    )

    rules = [parse_reaction_smarts(smarts)]
    cmols = [cmol_from_smiles(text)[0] for text in ("C1CCNCC1", "CCNC")]

    precursors, hits = CpuReactionMatcher(rules).match(cmols)
    expected = Counter(
        (int(hit[0]), int(hit[1]), short_ring_wl_molecule_id(precursors[hit[3]]))
        for hit in hits
    )

    batch, device_hits = MetalReactionMatcher(rules).match(metal_stack_mols(cmols))
    keys = metal_short_ring_wl_key(batch)
    observed = Counter((int(hit[0]), int(hit[1]), keys[hit[3]]) for hit in device_hits)

    assert observed == expected
    assert sum(observed.values()) > 0


def test_the_device_screen_reads_grouping_as_the_host_screen_does() -> None:
    """The screen is its own entry point, with its own pattern packing and its own upload token.

    `metal_screen` packs a pattern table once and uploads it under a stable token, so a missed
    group array there would not be a one-off wrong answer -- it would be served from cache for
    the rest of the screen's life. The pattern this checks is one a veto row could want: two
    reagents that must not be in a pot together, which a bare dot cannot say.
    """
    skip_if_no_metal_toolchain()

    from vertsy.filters import Veto, host_screen
    from vesma import cmol_from_smiles
    from vesma.metal_reaction_matcher import metal_screen, metal_stack_mols

    patterns = (
        "([C:1](=O)-[Cl]).([N:2])",  # an acyl chloride and an amine, in a pot
        "([C:1](=O)-[Cl].[N:2])",  # the same two groups on one molecule
        "[C:1](=O)-[Cl].[N:2]",  # unstated, which is what the library says today
    )
    smiles = (
        "CC(=O)Cl.CCN",  # two molecules
        "NCCCCC(=O)Cl",  # one molecule carrying both
        "CC(=O)Cl",  # the acyl chloride alone
        "CCN",  # the amine alone
    )
    molecules = [cmol_from_smiles(text)[0] for text in smiles]
    pairs = [
        (pattern, row)
        for pattern in range(len(patterns))
        for row in range(len(molecules))
    ]

    # The host screen reads veto rows and the device screen reads SMARTS, which is the
    # difference the two signatures already carry: nothing in `vesma` knows what a veto is.
    rows = tuple(
        Veto(
            code=f"T {index}",
            cost="pattern",
            scope=frozenset(),
            on="precursors",
            force="refuse",
            sense="forbid",
            pattern=text,
            rationale="a test row",
        )
        for index, text in enumerate(patterns)
    )

    host = np.asarray(host_screen(rows)(molecules, pairs), dtype=bool)
    device = np.asarray(
        metal_screen(patterns)(metal_stack_mols(molecules), pairs), dtype=bool
    )

    assert host.tolist() == device.tolist()
    # And the grouping is doing something, or the comparison above proves nothing.
    answers = host.reshape(len(patterns), len(molecules))
    assert answers[0].tolist() == [True, False, False, False], "two molecules"
    assert answers[1].tolist() == [False, True, False, False], "one molecule"
    assert answers[2].tolist() == [True, True, False, False], "either, unstated"
