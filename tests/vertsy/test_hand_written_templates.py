"""The hand-written selenium and macrocycle rule sets, and the selenium stock.

These rows were written here rather than vendored, so nothing upstream will catch a
regression in them. What is locked below is what was actually hard to get right.
"""

from __future__ import annotations

import csv
import re
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from tests.datadirs import external_data_root
from vertsy.template_lint import duplicates, lost_atom_maps
from vesma import (
    cmol_from_smiles,
    generate_precursors,
    parse_reaction_smarts,
    smiles_from_cmol,
)

TEMPLATES = external_data_root() / "templates"
RULE_SETS = (
    "templates-selenium",
    "templates-protecting-groups",
    "templates-carbohydrate",
    "templates-medchem",
)

# A cyclic selenonium salt, and the reason the selenium set exists.
TARGET_15 = (
    "O[C@H]1[C@H](O)C[Se@@+](C[C@H]([C@@H]2O[C@H](C3=CC=CC=C3)OC[C@H]2O)O)[C@@H]1CO"
)
# A DOTA mono-amide conjugate.
TARGET_5 = (
    "O=C(N[C@H](C(O)=O)CCC(NCC1=CC=C(CNC2=NC=CC=N2)C=C1)=O)"
    "CN3CCN(CC(O)=O)CCN(CC(O)=O)CCN(CC(O)=O)CC3"
)
# A PROTAC with an MDM2 spiro-oxindole half.
TARGET_2 = (
    "O=C1N(C2C(NC(CC2)=O)=O)CC(C1=CC=C3)=C3C#CC(CC4)CCN4C[C@@H]5CC[C@@H]"
    "(NC([C@H]6[C@H](C7=C(F)C(Cl)=CC=C7)[C@@]8(C(C=CC(Cl)=C9)=C9NC8=O)"
    "C%10(CCCCC%10)N6)=O)CC5"
)
# A 17-membered macrolactam.
TARGET_12 = "O=C(CCCNC1=O)NCCOC2=NN3C(C=C2C4=CN(C)N=C4)=NC=C3C5=CC1=CN=C5"


def rows(name: str) -> list[dict[str, str]]:
    with (TEMPLATES / f"{name}.csv").open() as handle:
        return list(csv.DictReader(handle))


def rule(code: str) -> str:
    for name in (*RULE_SETS, "templates-hartenfeller"):
        for row in rows(name):
            if row["template_code"] == code:
                return row["retro_template"]
    raise AssertionError(f"no rule {code!r}")


def precursors(code: str, smiles: str) -> set[str]:
    molecule, _ = cmol_from_smiles(smiles)
    return {
        " + ".join(sorted(smiles_from_cmol(p) for p in outcome.precursors))
        for outcome in generate_precursors(molecule, parse_reaction_smarts(rule(code)))
    }


@pytest.mark.parametrize("name", RULE_SETS)
def test_every_row_parses_and_is_balanced(name: str) -> None:
    table = rows(name)
    smarts = [row["retro_template"] for row in table]

    for row in table:
        parse_reaction_smarts(row["retro_template"])
        assert row["classification"], f"{row['template_code']} has no name"

    assert not [smarts[i] for i in duplicates(smarts)]
    assert not [s for s in smarts if lost_atom_maps(s)]


@pytest.mark.parametrize("name", RULE_SETS)
def test_no_rule_needs_sssr_ring_perception(name: str) -> None:
    """`;R2`/`;r5` mean one thing on the host and another on Metal and CUDA, which leave
    ring membership unknown. Bond-level `@`/`!@` is fine on every backend and at every
    retro layer: perception decides ring or chain per edge, on the host when a CMol is
    packed and in the retro kernels when the device builds a precursor.
    """
    for row in rows(name):
        assert not any(
            token in row["retro_template"] for token in (";R", ";r", "[R", "[r")
        ), row["template_code"]


def test_a_selenonium_salt_comes_from_a_selenide_and_an_epoxide() -> None:
    """The salacinol disconnection, and the whole reason for the selenium set.

    Target 15 must give back the selenolane diol and the protected sugar epoxide -- not a
    ring-opened mess, which is what happens without `!@` on the bond being broken.
    """
    got = precursors("Se 1", TARGET_15)

    assert len(got) == 1, "the exocyclic arm is the only bond that may break"
    only = next(iter(got))
    assert "[Se:1]" in only, "the precursor selenium is neutral, not [SeH+]"
    assert "[O:4]" in only and "[CH2:2]" in only, "an epoxide comes off"


def test_the_precursor_selenium_needs_both_its_charge_and_its_hydrogen_count() -> None:
    """Only one of the three spellings complains, and it is not the one you reach for first.

    Stating neither is silently wrong: the product's `+1` carries over and the "selenide"
    comes back as `[SeH+]`. Stating the charge alone raises. Both together are correct.
    """
    molecule, _ = cmol_from_smiles(TARGET_15)

    def selenium_precursor(right: str) -> str:
        template = rule("Se 1").replace("[Se;H0;+0:1]", right)
        outcomes = generate_precursors(molecule, parse_reaction_smarts(template))
        return next(
            smiles_from_cmol(p)
            for outcome in outcomes
            for p in outcome.precursors
            if "Se" in smiles_from_cmol(p)
        )

    unstated = selenium_precursor("[Se:1]")
    assert "[SeH+:1]" in unstated, "the silent trap this rule avoids"

    with pytest.raises(Exception, match="charge_changed_with_h_omitted"):
        selenium_precursor("[Se+0:1]")

    assert "[Se:1]" in selenium_precursor("[Se;H0;+0:1]")


def test_ring_closing_rules_refuse_an_acyclic_bond() -> None:
    """`@` is what separates these from the intermolecular rules they mirror."""
    assert precursors("Mac 1", "C1CCCCCCCCC=CCCCC1"), "a ring alkene is an RCM product"
    assert not precursors("Mac 1", "C=CCCCCCC=C"), "an acyclic diene is not"
    biphenyl = precursors("Mac 3b", "c1ccc(-c2ccccc2)cc1")
    assert not biphenyl, "biphenyl is Suzuki 31, not Mac 3"


def test_a_macrolactam_gives_back_a_free_acid_rather_than_an_acid_chloride() -> None:
    """Hartenfeller 47 offers an acid chloride; the free acid is the better-stocked precursor."""
    got = precursors("Mac 5", TARGET_12)

    assert got
    acid_chloride = [one for one in got if "Cl" in one]
    assert not acid_chloride, "an acid chloride is Hartenfeller 47's answer"
    assert all("[C:1](-O)=O" in one or "[C:1](=O)(-O)" in one for one in got), got


@pytest.mark.parametrize("name", ["stock-selenium", "stock-chelators"])
def test_the_stock_is_all_readable(name: str) -> None:
    """A stock row that Vesma cannot read is a silent hole in the catalogue."""
    path = TEMPLATES / f"{name}.smi"
    lines = [line for line in path.read_text().splitlines() if line.strip()]

    assert len(lines) >= 10
    for line in lines:
        smiles = line.split()[0]
        molecule, _ = cmol_from_smiles(smiles)
        if name.startswith("stock-selenium"):
            # Checked on the parsed molecule: aromatic selenium is written `[se]`.
            assert "Se" in smiles_from_cmol(molecule).replace("[se", "[Se"), smiles


def test_alkene_and_alkyne_reduction_exist_and_do_not_flood_the_search() -> None:
    """Hydrogenation is as routine as chemistry gets and had no rule in any set here.

    It is left out of template collections often enough to look deliberate, because a retro
    hydrogenation reads every saturated C-C bond as a possible alkene and that sounds like it
    would flood a search. Measured, it does not: the engine counts *distinct products*, so the
    many equivalent bonds of a symmetric molecule collapse to one. On a steroid core and on
    decalin this rule contributes a single application each, against a whole-library total of
    seven and zero -- which is why it is in `medchem` and not held back.

    `MC 80` wants a disubstituted alkene, since semi-hydrogenation of a terminal alkyne gives a
    terminal alkene and that is a different rule from the one worth having here.
    """
    from rdkit import Chem

    for code, product, expected in (
        ("MC 79", "CCCc1ccccc1", "C=CCc1ccccc1"),
        ("MC 79", "C1CCCCC1", "C1=CCCCC1"),
        ("MC 80", "C(=C/c1ccccc1)\\C", "CC#Cc1ccccc1"),
    ):
        got = {
            without_maps(p)
            for one in precursors(code, product)
            for p in one.split(" + ")
        }
        assert Chem.CanonSmiles(expected) in got, f"{code}: {sorted(got)}"

    assert not precursors("MC 80", "C=Cc1ccccc1")

    # the ring bond matters: the final step of a fused-polycyclic route is usually a ring alkene
    assert precursors("MC 79", "C1CCCCC1")


def test_an_azole_nitrogen_can_be_acylated_as_well_as_alkylated() -> None:
    """`MC 27` alkylates an azole nitrogen; nothing acylated one, for any ring.

    `Cpl 1` cannot: it writes `[N:2]`, which is aliphatic nitrogen, so an N-acyl indole or
    N-acyl pyrazole was not an amide to any rule here. The three variants split the same way
    `MC 27` does -- by what flanks the nitrogen -- because a flat SMARTS has to name each ring
    type rather than ask for "an azole".

    The 1,2,3-triazole is the one to get right, and the naming is not the obvious one: acyl on
    **N1** sits between a carbon and a nitrogen, so it is `MC 78b`, and only the N2 isomer has a
    nitrogen on both sides.
    """
    from rdkit import Chem

    # The freed ring comes back in the tautomer the removed acyl group leaves behind, which is
    # not always the one a chemist would write first: N1-acyl-1,2,3-triazole gives the 2H form.
    for code, product, ring in (
        ("MC 78a", "CC(=O)n1ccc2ccccc21", "c1ccc2[nH]ccc2c1"),
        ("MC 78b", "CC(=O)n1cccn1", "c1cn[nH]c1"),
        ("MC 78b", "CC(=O)n1ccnn1", "c1c[nH]nn1"),
        ("MC 78c", "CC(=O)n1nccn1", "c1cn[nH]n1"),
    ):
        got = {
            without_maps(p)
            for one in precursors(code, product)
            for p in one.split(" + ")
        }
        assert Chem.CanonSmiles(ring) in got, f"{code}: {sorted(got)}"
        assert Chem.CanonSmiles("CC(=O)O") in got, f"{code}: {sorted(got)}"

    # an aliphatic amide stays `Cpl 1`'s business
    assert not precursors("MC 78a", "CC(=O)NCC")


def test_the_protac_conjugation_chemistries_are_covered() -> None:
    """The five couplings a heterobifunctional degrader is assembled from.

    A PROTAC is a target ligand, an E3 recruiter and a linker, joined by whatever conjugation is
    highest-yielding -- so the library is judged by whether it can take those joins apart. Four of
    the five were already here between Hartenfeller and medchem; what was missing was the SNAr
    that derivatises the cereblon recruiters.

    `Hartenfeller 56` activates SNAr with a *para nitro* group only, and thalidomide, lenalidomide
    and pomalidomide carry no nitro: their fluorine sits on the phthalimide benzene, activated by
    the ring-fused imide carbonyl beside it. `MC 76` adds carbonyl activation, ortho and para, so
    `4-fluorothalidomide + amine` is reachable -- the standard way a linker is attached to that
    whole E3 ligand family.
    """
    from rdkit import Chem

    pomalidomide_protac = "O=C1CCC(N2C(=O)c3cccc(NCCOC)c3C2=O)C(=O)N1"

    for code, product, expected in (
        ("MC 76a", pomalidomide_protac, "COCCN"),
        ("MC 77a", "COCCNCc1ccccc1", "COCCOS(C)(=O)=O"),
        ("MC 77b", "COCCNCc1ccccc1", "COCCOS(=O)(=O)c1ccc(C)cc1"),
    ):
        got = {
            without_maps(p)
            for one in precursors(code, product)
            for p in one.split(" + ")
        }
        assert Chem.CanonSmiles(expected) in got, f"{code}: {sorted(got)}"

    # the activating carbonyl is the point: a plain aniline is not an SNAr product
    assert not precursors("MC 76a", "COCCNc1ccccc1")

    # The two hand-written joins, so a regression shows up here. Sonogashira
    # (`Hartenfeller 46b`), the Williamson aryl ether (`29e`) and Buchwald-Hartwig (`51b`) are
    # Hartenfeller's and are covered by that table's own tests -- `rule()` resolves only the
    # hand-written sets, which is why they are named here and not asserted.
    for code, product, expected in (
        ("Cpl 1", "CCNC(=O)CC", "CCC(=O)O"),
        ("MC 72", "CCn1cc(-c2ccccc2)nn1", "C#Cc1ccccc1"),
    ):
        got = {
            without_maps(p)
            for one in precursors(code, product)
            for p in one.split(" + ")
        }
        assert Chem.CanonSmiles(expected) in got, f"{code}: {sorted(got)}"


def test_the_macrocyclisation_rules_carry_a_minimum_ring_size() -> None:
    """Written as an explicit `-@` chain, because `;r5` needs perception the GPUs run without.

    Bond-level ring membership is per-edge and both GPU paths compute it; SSSR ring *size* is not
    available to them, so a floor has to be spelled as "the ring must supply these atoms". That
    measures reachable ring atoms rather than ring size, which is the same thing only when the
    matched atoms belong to one ring -- true for these three, and why `Mac 3`/`Mac 4` have no
    floor at all.

    The beta-lactam is the case worth pinning. It does not form by simple cyclisation, and the
    floor has to reject it fused as well as isolated, since the drugs that matter are fused.
    """
    for code, smiles, fires, why in (
        ("Mac 1", "C1=CCC1", False, "cyclobutene is strained"),
        ("Mac 1", "C1=CCCC1", True, "cyclopentene is routine RCM"),
        ("Mac 1", "C1=CCCCCCCCCCCCC1", True, "a real macrocycle"),
        ("Mac 2", "C1#CCCCCC1", False, "cycloheptyne is not isolable"),
        ("Mac 2", "C1#CCCCCCC1", True, "cyclooctyne is the smallest that is"),
        ("Mac 5", "O=C1CCN1", False, "isolated beta-lactam"),
        (
            "Mac 5",
            "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
            False,
            "penicillin G, a fused beta-lactam",
        ),
        ("Mac 5", "O=C1CCCN1", True, "gamma-lactam closes readily"),
        ("Mac 5", "O=C1CCCCCCCCCCCN1", True, "13-ring macrolactam"),
    ):
        assert bool(precursors(code, smiles)) is fires, f"{code}: {why}"


def test_a_two_component_rule_closes_a_ring_without_a_second_rule() -> None:
    """The reason there is no separate ring-closing click, and no separate macrocycle rule set.

    When the broken bond lies in a cycle the two right-hand components stay connected through the
    rest of the ring, so separation returns one precursor carrying both halves.
    """
    acyclic = precursors("MC 72", "CCn1cc(-c2ccccc2)nn1")
    macrocyclic = precursors("MC 72", "C1CCCCn2cc(nn2)CCCC1")

    assert acyclic and all(" + " in one for one in acyclic), acyclic
    assert macrocyclic and all(" + " not in one for one in macrocyclic), macrocyclic


def test_the_rule_sets_are_registered_under_the_names_the_readme_documents() -> None:
    """`AOSTAR_HOST_RULE_SET=hartenfeller,selenium,carbohydrate` has to resolve.

    `macrocycles` is deliberately absent: its nine rules moved into `medchem`, because a
    two-component retro template *already* closes a ring when the bond it breaks is in one --
    the components stay connected through the rest of the cycle and separation returns a single
    precursor. They were never a different kind of chemistry, only the `-@` spelling of couplings
    medchem already carried with `-!@`.
    """
    from tests.vertsy.bench_host_aostar import STOCK_FILES, TEMPLATE_FILES

    for name in ("selenium", "carbohydrate", "medchem"):
        assert TEMPLATE_FILES[name].exists(), name

    # A template set earns its own file when its rules cannot terminate on a general catalogue --
    # which is a different question from whether some *stock* is opt-in. `selenium` qualifies:
    # Enamine carries no selenium at all, so those rules are unusable without
    # `stock-selenium.smi` and would be pure fanout in an ordinary run.
    #
    # `sulfonium` and `ammonium` do not. Enamine holds over twelve thousand dialkyl sulfides, so
    # `S 1` and `S 2` terminate normally, and `S 3`'s bare `[SH2]` is an ordinary commodity that
    # now sits in `stock-common-reagents.smi` with the rest. Menshutkin quaternisation asks only
    # for a secondary amine and an alkyl halide, and no `stock-ammonium.smi` was ever written.
    # Both live in `medchem`, and selenium is the only element left needing its own anything.
    for merged in ("macrocycles", "ammonium", "sulfonium"):
        assert merged not in TEMPLATE_FILES, merged
    assert "ammonium" not in STOCK_FILES
    assert "sulfur" not in STOCK_FILES
    assert STOCK_FILES["selenium"].exists()


def test_every_confirmed_stock_row_names_where_it_can_be_bought() -> None:
    """The point of splitting the two files: this one is checkable, the other is not.

    CAS numbers used to be checked here too and are gone from every stock file. They were the one
    identifier with a rights holder that has asserted claims over redistributing its registry, and
    they were redundant for us: the SMILES *is* the structure and the catalogue number is how the
    molecule is bought. Vendor names stay -- naming a supplier to identify its own product is what
    a trademark permits, and a bare catalogue number nobody can attribute is useless.

    The two in-situ rows are the exception the note carries rather than a hole in the rule: hydrogen
    selenide and sodium hydroselenide are generated in the flask, and what is *bought* is the
    selenium powder the note names.
    """
    for line in (TEMPLATES / "stock-selenium.smi").read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        assert len(parts) == 3, line
        _smiles, name, source = parts
        buyable = any(v in source for v in ("Sigma", "TCI"))

        assert buyable, f"{name}: {source!r}"


def test_no_stock_file_ships_a_cas_number() -> None:
    """Dropped deliberately, so a well-meaning re-add should fail rather than pass review."""
    looks_like_cas = re.compile(r"\d{2,7}-\d{2}-\d")
    offenders = [
        f"{path.name}: {line}"
        for path in sorted(TEMPLATES.glob("stock-*.smi"))
        for line in path.read_text().splitlines()
        for field in line.split("\t")[1:]
        if looks_like_cas.fullmatch(field.strip())
    ]

    assert not offenders, offenders


def test_the_readme_records_that_zinc_carries_no_selenium() -> None:
    """The measurement is why this stock is hand-written; losing it invites redoing the search."""
    readme = (TEMPLATES / "README.md").read_text()

    assert "0 hits out of 1.64 billion" in readme
    assert Path(TEMPLATES / "stock-selenium.smi").exists()


def without_maps(smiles: str) -> str:
    """Vesma writes atom maps into precursor SMILES; stock comparison needs them gone."""
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None, smiles
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule)


def test_no_vendored_rule_couples_an_amide_from_a_free_acid() -> None:
    """The gap `Cpl 1` exists to fill. If Hartenfeller ever grows one, this file is redundant."""
    hartenfeller = external_data_root() / "templates" / "templates-hartenfeller.csv"
    with hartenfeller.open() as handle:
        amide_products = [
            row["retro_template"].split(">>")[1]
            for row in csv.DictReader(handle)
            # Carboxamides only: this leading shape excludes the sulfonamide
            # rule 48 and the urea rule 57, whose left-hand sides start elsewhere.
            if row["retro_template"].startswith("[C:1](=O)-[N")
        ]

    assert amide_products, "the amide rules should still be there"
    free_acid = [
        rhs for rhs in amide_products if "[Cl]" not in rhs and "[Br]" not in rhs
    ]
    assert not free_acid, "every vendored amide disconnection yields an acid halide"


def test_a_dota_conjugate_reaches_dota_in_one_step() -> None:
    """Targets 5 and 9 are DOTA conjugates, which is why they need stock and not a macrocycle rule."""
    from rdkit import Chem

    dota = Chem.CanonSmiles("OC(=O)CN1CCN(CC(O)=O)CCN(CC(O)=O)CCN(CC(O)=O)CC1")
    stock = {
        Chem.CanonSmiles(line.split()[0])
        for line in (TEMPLATES / "stock-chelators.smi").read_text().splitlines()
        if line.strip()
    }
    assert dota in stock, "DOTA has to be in the chelator stock for this to matter"

    molecule, _ = cmol_from_smiles(TARGET_5)
    reached = {
        without_maps(smiles_from_cmol(p))
        for outcome in generate_precursors(
            molecule, parse_reaction_smarts(rule("Cpl 1"))
        )
        for p in outcome.precursors
    }

    assert dota in reached, sorted(reached)


def test_the_spiro_oxindole_of_target_2_opens_under_the_cycloaddition() -> None:
    """Nothing else in any rule set can build that ring system."""
    acid = "O=C(O)C1NC2(CCCCC2)C2(C(=O)Nc3cc(Cl)ccc32)C1c1cccc(Cl)c1F"

    got = precursors("Cyc 1", acid)

    assert len(got) == 1
    fragments = next(iter(got)).split(" + ")
    assert len(fragments) == 3, "a three-component cycloaddition"


def test_the_cycloaddition_does_not_fire_on_any_old_pyrrolidine() -> None:
    """The oxindole is spelled out precisely so this stays narrow."""
    assert not precursors("Cyc 1", "C1CCNC1"), "plain pyrrolidine"
    assert not precursors("Cyc 1", "O=C(O)C1CCCN1"), "proline"


def test_a_cyclic_selenide_opens_to_a_dihalide_and_a_selenide_source() -> None:
    """`Se 6` is the route to the selenosugar; `[SeH2]` stands in for Na2Se and must be stock."""
    got = precursors("Se 6b", "OC1C(O)C[Se]C1CO")

    assert got
    assert all("[SeH2:2]" in one for one in got), got
    # H2Se is generated in situ from selenium powder, and lives in the *verified* file with a
    # note saying so. Filing it under "availability not confirmed" said the wrong thing about a
    # real reagent -- what is bought is selenium powder, and the generation is standard practice.
    #
    # An earlier version of this comment blamed a parser limit, claiming Vesma could not read the
    # bare `[Se]` the catalogue sells. Both halves were wrong. `[Se]` is refused because neutral
    # selenium is declared at graph valence 2, 4 or 6 and never 0 -- valence 0 exists only at
    # charge -2, where the octet closes -- so atomic Se is an open-shell species the closed-shell
    # model is right to reject. And `[Se]` was never the structure anyway: elemental selenium is
    # Se8 rings or helical chains, every atom two-coordinate, which this model reads happily.
    stock = (TEMPLATES / "stock-selenium.smi").read_text()

    assert "[SeH2]" in stock, "Se 6 needs a selenide source somewhere in stock"
    assert "Se powder itself is Sigma 229865" in stock


def test_the_chelator_stock_has_the_right_molecules() -> None:
    """Checked by formula: a mistyped macrocycle is easy to make and invisible by eye."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    expected = {
        "cyclen (1,4,7,10-tetraazacyclododecane)": "C8H20N4",
        "cyclam (1,4,8,11-tetraazacyclotetradecane)": "C10H24N4",
        "DOTA": "C16H28N4O8",
        "DOTA tris(t-Bu ester)": "C28H52N4O8",
        "NOTA": "C12H21N3O6",
        "TETA": "C18H32N4O8",
        "DTPA": "C14H23N3O10",
        "EDTA": "C10H16N2O8",
    }
    formulas = {}
    for line in (TEMPLATES / "stock-chelators.smi").read_text().splitlines():
        if not line.strip():
            continue
        smiles, name, coverage = line.split("\t")
        assert coverage in ("also in Enamine", "not in Enamine"), line
        molecule = Chem.MolFromSmiles(smiles)
        assert molecule is not None, name
        cmol_from_smiles(smiles)
        formulas[name] = rdMolDescriptors.CalcMolFormula(molecule)

    for name, formula in expected.items():
        assert formulas[name] == formula, f"{name}: {formulas[name]} != {formula}"


#: The four rules that deliberately go the other way, and the only ones allowed to. A code ending
#: in ``r`` adds its group in the retro direction, so the *last* step of the synthesis is a
#: deprotection -- which is the only way a protect/react/deprotect detour can exist in the graph at
#: all, and without it "the price never lengthened a route" is guaranteed by the data rather than
#: measured. They are safe only because a 1,2- or 1,3-diol is a rare condition: 0.28% and 0.20% of
#: the PaRoutes products, against 11.09% for any hydroxyl and 27.51% for any N-H amine.
ADDING_PROTECTING_GROUPS = frozenset(
    {
        # The diol-conditioned acetals, narrow by construction.
        "PG 9r",
        "PG 10r",
        "PG 11r",
        "PG 12r",
        # The broad ones, allowed deliberately since 2026-10-03 -- see the test below.
        "PG 1r",
        "PG 2r",
        "PG 3r",
        "PG 4r",
        "PG 5r",
        "PG 6r",
        "PG 7r",
        "PG 8r",
        "PG 13r",
        "PG 14r",
        "PG 15r",
        "PG 16r",
        "PG 17r",
        "PG 18r",
    }
)


def test_no_protecting_group_rule_adds_a_group_going_backwards() -> None:
    """Every rule declares its direction, and the `r` codes are the ones that add.

    An adding rule -- "this alcohol was once a TBS ether" -- fires on every hydroxyl in every
    molecule, where a removing rule fires only where the group already is. Counting atoms is a
    blunt but sufficient check that each rule does what its code says.

    **The file used to allow only four adders, all conditioned on a diol, and that changed.**
    The reason it changed is worth keeping, because the old reasoning was not wrong about the
    cost -- it was wrong about the alternative. Removal-only means a route whose last step is a
    deprotection cannot be built at all, since nothing can propose the protected precursor; and
    the protect/react/deprotect detour the selectivity price exists for ends in a deprotection
    whenever the target carries no protecting group of its own. Measured on 500 PaRoutes n1
    routes, 230 of the 501 unimolecular reference steps the collection could not reproduce were
    deprotections, so the gap was the largest single one in the set.

    Against that, the fanout argument is weaker than it looks. The default heuristic prices a
    molecule by `|u| x (1 - T)` -- small and close to stock -- so an adder's successors are
    scored worse than the product they came from and are ranked below it rather than expanded.
    They cost one match pass and a fingerprint each, not a subtree. And this file is optional:
    a run that does not want them does not load it.
    """
    from rdkit import Chem

    from vertsy.template_lint import top_level_pieces

    def bare(piece: str) -> str:
        """A zero-level group with its parentheses off, for a reader that cannot take them.

        RDKit parses a reaction side but not a standalone `(A).(B)`, and these rows carry that
        grouping now that their molecularity is stated. The parentheses are a claim about
        molecularity and nothing this test asks about, so they come off here rather than being
        left out of the table.
        """
        return piece[1:-1] if piece.startswith("(") and piece.endswith(")") else piece

    for row in rows("templates-protecting-groups"):
        left, right = row["retro_template"].split(">>")
        lhs = Chem.MolFromSmarts(bare(left))
        pieces = [bare(one) for one in top_level_pieces(right)]
        assert lhs is not None, row["template_code"]
        assert all(Chem.MolFromSmarts(one) is not None for one in pieces), row[
            "template_code"
        ]
        code = row["template_code"]
        # Against the **substrate** component of the right-hand side, not the whole of it. A
        # remover now names the reagent that installed the group -- `PG 13` hands back acetic
        # anhydride beside the alcohol -- so the total right-hand side is larger than the left on
        # both kinds of rule and the blunt count no longer separates them. The component carrying
        # map `:1` is the substrate by convention throughout this table, and comparing *that*
        # against the left is the direction claim: a remover hands back less of the molecule than
        # it matched, an adder more. Checked to hold on all 33 rows.
        # Split on *zero-level* dots, so a grouped piece holding its own dot -- the
        # intramolecular shape `(A.B)` -- stays one piece instead of becoming two.
        carriers = [one for one in pieces if ":1]" in one]
        substrate = Chem.MolFromSmarts(carriers[0]).GetNumAtoms()
        if code in ADDING_PROTECTING_GROUPS:
            assert substrate > lhs.GetNumAtoms(), f"{code} is declared an adder"
            continue
        assert substrate < lhs.GetNumAtoms(), f"{code} adds atoms"


@pytest.mark.parametrize(
    ("adder", "remover"),
    [("PG 9r", "PG 9"), ("PG 10r", "PG 10"), ("PG 11r", "PG 11"), ("PG 12r", "PG 12")],
)
def test_an_adder_and_its_remover_are_inverses_on_a_polyol(
    adder: str, remover: str
) -> None:
    """Protect then deprotect gets the diol back, which is what makes the detour a detour.

    Run on the chloro-tetraol this repo's own target-15 narrative produces, because that is the
    molecule the detour was described for: five hydroxyls, of which an acetal blocks two.
    """
    from rdkit import Chem

    protected = precursors(adder, "OCC(O)C(O)C(O)CCl")
    assert protected, f"{adder} did not protect a polyol"
    freed = {
        without_maps(part)
        for each in protected
        for back in precursors(remover, without_maps(each))
        # The remover names the reagent that installed the group -- acetone, for an acetonide --
        # so its precursor side has two components and only one of them is the molecule.
        for part in back.split(" + ")
    }
    wanted = Chem.CanonSmiles("OCC(O)C(O)C(O)CCl")
    assert wanted in freed, f"{remover} does not undo {adder}"


def test_a_protecting_group_rule_ignores_the_unprotected_molecule() -> None:
    """Which is why the branching factor stays low."""
    free_amine = precursors("PG 1", "NCc1ccccc1")
    free_alcohol = precursors("PG 5", "OCc1ccccc1")
    assert free_amine == set(), "a free amine is not a Boc carbamate"
    assert free_alcohol == set(), "a free alcohol is not a silyl ether"
    assert precursors("PG 1", "CC(C)(C)OC(=O)NCc1ccccc1"), "the Boc carbamate is"


def test_the_benzylidene_comes_off_the_target_15_sugar_epoxide() -> None:
    """`Se 1` hands back a protected sugar; without this the chain stops there."""
    from rdkit import Chem

    got = precursors("PG 9", "OC1COC(c2ccccc2)OC1C1CO1")

    assert len(got) == 1
    triol = without_maps(next(iter(got)))
    assert triol == Chem.CanonSmiles("OCC(O)C(O)C1CO1"), triol


def test_the_halide_fgi_refuses_where_the_reaction_does_not_work() -> None:
    """Spelled to the substitution that actually reacts, not to every halide.

    An aryl chloride has no Appel route, a tertiary halide eliminates, and an alpha-halo
    ketone does not come from the alpha-hydroxy ketone -- it gets its own rules.
    """
    assert precursors("FGI 1a", "CCCCCl"), "a primary alkyl chloride does"
    assert not precursors("FGI 1a", "Clc1ccccc1"), "aryl chloride"
    assert not precursors("FGI 1a", "CC(=O)CCl"), "alpha-chloro ketone"
    assert not precursors("FGI 4b", "CC(C)(C)Br"), "tertiary bromide"
    assert precursors("FGI 5b", "CC(=O)CBr"), "the alpha-halo ketone has its own rule"


def test_both_halves_of_target_15_reach_a_commercial_pentitol() -> None:
    """The chain the selenium, protecting-group and FGI files exist to complete."""
    from rdkit import Chem

    xylitol = Chem.CanonSmiles("OCC(O)C(O)C(O)CO")

    def only(code: str, smiles: str) -> str:
        """The substrate-derived precursor, told apart by carrying atom maps.

        Not "the one without selenium" any more: these rules now emit their reagent as a precursor
        too -- `FGI 1a` names SOCl2, `FGI 1b` names CBr4 and PPh3 -- so several pieces lack
        selenium and two of them outweigh the pentitol. What distinguishes the substrate is that
        its atoms came *through* the rule and so carry its map numbers, while a reagent the
        right-hand side invents carries none.
        """
        pieces = precursors(code, smiles).pop().split(" + ")
        carried = [one for one in pieces if ":" in one and "Se" not in one]
        assert len(carried) == 1, f"{code}: {pieces}"
        return without_maps(carried[0])

    # electrophile half: benzylidene off, epoxide opened, halide back to the alcohol
    epoxide = only("PG 9", "OC1COC(c2ccccc2)OC1C1CO1")
    assert Chem.CanonSmiles(epoxide) == Chem.CanonSmiles("OCC(O)C(O)C1CO1")
    assert Chem.CanonSmiles(only("FGI 1a", "OCC(O)C(O)C(O)CCl")) == xylitol

    # selenium half: selenolane opened to the dibromide, then both halides back
    dibromide = only("Se 6b", "OCC1[Se]CC(O)C1O")
    assert Chem.CanonSmiles(dibromide) == Chem.CanonSmiles("OCC(Br)C(O)C(O)CBr")
    mono = only("FGI 1b", dibromide)
    assert Chem.CanonSmiles(only("FGI 4b", mono)) == xylitol


def test_stock_coverage_must_be_judged_with_the_keyer_not_string_equality() -> None:
    """A separate sugar stock was written and deleted once this was measured properly.

    The chain ends on a flat pentitol and the Enamine catalogue writes its sugars with
    stereochemistry, so comparing canonical SMILES as strings says "absent". The search does
    not compare that way: both keyers drop stereochemistry, so the catalogue already covers it.
    """
    from rdkit import Chem

    from vesma.keyer import cmol_rdkit_smiles_keyer, short_ring_wl_keyer

    flat = "OCC(O)C(O)C(O)CO"
    catalogue = "OC[C@@H](O)[C@H](O)[C@@H](O)CO"  # a real Enamine row

    assert Chem.CanonSmiles(flat) != Chem.CanonSmiles(catalogue), "strings differ"
    chain, _ = cmol_from_smiles(flat)
    stocked, _ = cmol_from_smiles(catalogue)
    assert short_ring_wl_keyer(chain) == short_ring_wl_keyer(stocked)
    assert cmol_rdkit_smiles_keyer(chain) == cmol_rdkit_smiles_keyer(stocked)


def test_dota_is_the_row_that_makes_the_chelator_file_load_bearing() -> None:
    """Five of the eleven chelators are already in Enamine; DOTA is not, and 5 and 9 need it.

    The coverage column exists because the sugar file was added on a string comparison and
    deleted once the same question was asked with the keyer.
    """
    rows = [
        line.split("\t")
        for line in (TEMPLATES / "stock-chelators.smi").read_text().splitlines()
        if line.strip()
    ]
    coverage = {name: state for _, name, state in rows}

    assert coverage["DOTA"] == "not in Enamine"
    assert coverage["cyclen (1,4,7,10-tetraazacyclododecane)"] == "also in Enamine"
    assert sum(s == "not in Enamine" for s in coverage.values()) == 6


HARTENFELLER = external_data_root() / "templates" / "templates-hartenfeller.csv"


def hartenfeller_rule(code: str) -> str:
    with HARTENFELLER.open() as handle:
        for row in csv.DictReader(handle):
            if row["template_code"] == code:
                return row["retro_template"]
    raise AssertionError(code)


def heavy_atom_balance(template: str, smiles: str) -> tuple[int, int]:
    """Heavy atoms in the product, and in the precursors of its first outcome.

    A retro step adds atoms -- a halide, a leaving group -- so precursors should never carry
    *fewer* heavy atoms than the product unless something was deliberately removed.
    """
    from rdkit import Chem

    molecule, _ = cmol_from_smiles(smiles)
    outcomes = generate_precursors(molecule, parse_reaction_smarts(template))
    assert outcomes, f"no match: {smiles}"
    total = sum(
        Chem.MolFromSmiles(smiles_from_cmol(p)).GetNumHeavyAtoms()
        for p in outcomes[0].precursors
    )
    return Chem.MolFromSmiles(smiles).GetNumHeavyAtoms(), total


def test_an_unmapped_skeleton_atom_does_not_delete_its_substituent() -> None:
    """Found by a route that "solved" the PROTAC above while dropping half the molecule.

    An unmapped left-hand atom is deleted with everything attached to it. That is what a
    leaving group or a protecting group wants; it is a bug when the atom is skeleton and can
    carry an arbitrary substituent, because the loss then scales with the substituent.
    """
    internal = heavy_atom_balance(
        hartenfeller_rule("Hartenfeller 46b"), "CCCCC#Cc1ccccc1"
    )
    assert internal[1] >= internal[0], "Sonogashira dropped the alkyne substituent"

    terminal = heavy_atom_balance(hartenfeller_rule("Hartenfeller 46b"), "C#Cc1ccccc1")
    assert terminal[1] >= terminal[0], "and still works on a terminal alkyne"

    nitro = heavy_atom_balance(
        hartenfeller_rule("Hartenfeller 55a"), "CNc1ccc(C)cc1[N+](=O)[O-]"
    )
    assert nitro[1] >= nitro[0], "nitro SNAr dropped a ring substituent"


def test_the_triazole_rule_only_claims_the_nh_tautomer_it_can_make() -> None:
    """Its two ring nitrogens come from hydrazine and are unmapped, so an N-substituted
    triazole would have silently lost its substituent. Constrained to D2 instead."""
    template = hartenfeller_rule("Hartenfeller 16a")
    nh, _ = cmol_from_smiles("Cc1nnc(-c2ccccc2)[nH]1")
    n_methyl, _ = cmol_from_smiles("Cc1nn(C)c(-c2ccccc2)n1")

    assert generate_precursors(nh, parse_reaction_smarts(template))
    assert not generate_precursors(n_methyl, parse_reaction_smarts(template))


def test_de_aromatizing_an_nh_heteroatom_resolves_its_hydrogen_split() -> None:
    """A triazole ring-opening makes a nitrile nitrogen, and it must not inherit the N-H.

    An aromatic `[nH]` is stored with its hydrogen *explicit* -- `H=1, h=0` -- so that it writes
    back as `[nH]` and not `n`. `reaction.py`'s survivor carry used to hand that explicit count to
    whatever the right-hand side made of the atom, so `Hartenfeller 15`/`16a`/`16b` on an N-H
    triazole gave the nitrile nitrogen `explicit=1` against an edited total of 0, and the builder
    stored `h = -1`: total right, explicit/implicit impossible.

    Nothing caught it near the cause. The device arms clamp the same subtraction to zero, so a
    metal search accepted the application; `smiles.py` reads `H - h` as the explicit count, so the
    molecule *serialised* as `[NH]` with no error; and only `identity.py`'s seed refused it, as a
    bare `ValueError` out of guided replay that ended the whole run rather than one target.

    So this asserts the chemistry rather than the absence of a crash: every precursor the rule
    makes is a molecule whose hydrogen split is determined, and the nitrile really has none.
    """
    for code in ("Hartenfeller 15", "Hartenfeller 16a", "Hartenfeller 16b"):
        reaction = parse_reaction_smarts(hartenfeller_rule(code))
        molecule, _ = cmol_from_smiles("Cc1nnc(-c2ccccc2)[nH]1")
        outcomes = generate_precursors(molecule, reaction)

        assert outcomes, f"{code} must still fire on the N-H tautomer"
        for outcome in outcomes:
            for precursor in outcome.precursors:
                where = f"{code}: {smiles_from_cmol(precursor)}"
                assert not np.any(precursor.h < 0), where
                assert not np.any(precursor.H < 0), where

    # And the nitrile nitrogen specifically carries no hydrogen, which is the chemistry.
    reaction = parse_reaction_smarts(hartenfeller_rule("Hartenfeller 16a"))
    molecule, _ = cmol_from_smiles("Cc1nnc[nH]1")
    nitriles = [
        precursor
        for outcome in generate_precursors(molecule, reaction)
        for precursor in outcome.precursors
        if "#" in smiles_from_cmol(precursor)
    ]
    assert nitriles, "the ring-opening makes a nitrile"
    for precursor in nitriles:
        nitrogen = [index for index, z in enumerate(precursor.Z) if z == 7]
        assert all(int(precursor.H[index]) == 0 for index in nitrogen)


def test_the_sulfonium_rules_mirror_the_selenium_ones() -> None:
    """Salacinol is the sulfur twin of the selenonium salt above, and nothing anywhere
    disconnects an S+ -- it was the only unsolved target in the 43k-rule run at the time.
    """
    from rdkit import Chem

    target_14 = (
        "O[C@H]1[C@H](O)C[S@@+](C[C@H]([C@@H]2O[C@H](C3=C(Br)C=CC=C3)OC[C@H]2O)O)"
        "[C@@H]1CO"
    )
    got = precursors("S 1", target_14)

    assert len(got) == 1
    parts = {without_maps(p) for p in next(iter(got)).split(" + ")}
    assert Chem.CanonSmiles("OCC1SCC(O)C1O") in parts, "the thiolane triol"
    assert any("C1CO1" in p for p in parts), "and a sugar epoxide"


def test_the_sulfide_precursor_is_neutral_like_the_selenide() -> None:
    """Same trap as Se 1: stating neither charge nor hydrogen count silently keeps the +1."""
    molecule, _ = cmol_from_smiles("C[S+](C)CCO")
    template = rule("S 2b").replace("[S;H0;+0:1]", "[S:1]")

    kept = {
        smiles_from_cmol(p)
        for outcome in generate_precursors(molecule, parse_reaction_smarts(template))
        for p in outcome.precursors
        if "S" in smiles_from_cmol(p)
    }
    assert any("[SH+" in k for k in kept), "the trap the +0 and H0 avoid"


def test_the_ammonium_rules_complete_the_salacinol_family() -> None:
    """Targets 14, 15 and 16 are one scaffold with S+, Se+ and N+ at the centre.

    The nitrogen member reduces to an iminosugar that the Enamine catalogue already carries,
    which is why it needs a rule and no stock, where its S and Se twins need both.
    """
    from rdkit import Chem

    target_16 = (
        "O[C@H]1[C@H](O)C[N@+](C[C@H]([C@@H]2O[C@H](C3=C([N+]([O-])=O)C=CC=C3)OC[C@H]2O)O)"
        "([H])[C@@H]1CO"
    )
    got = precursors("N 1", target_16)

    assert len(got) == 1
    parts = {without_maps(p) for p in next(iter(got)).split(" + ")}
    iminosugar = Chem.CanonSmiles("OCC1NCC(O)C1O")
    assert iminosugar in parts, "1,4-dideoxy-1,4-imino-arabinitol"


def test_the_menshutkin_rule_gives_a_tertiary_amine_and_a_halide() -> None:
    """The quaternary member: D4 with no hydrogen, so it needs its own rule."""
    from rdkit import Chem

    got = precursors("N 3b", "C[N+](C)(C)Cc1ccccc1")

    assert got
    parts = {without_maps(p) for one in got for p in one.split(" + ")}
    assert Chem.CanonSmiles("CN(C)C") in parts, "trimethylamine"
    assert Chem.CanonSmiles("BrCc1ccccc1") in parts, "benzyl bromide"


def test_every_coupling_reagent_can_itself_be_made() -> None:
    """The dead-end analysis found the library consuming reagents nothing could make.

    Suzuki consumes boronic acids, Stille consumes stannanes, and Hartenfeller 47 and iSCB 68
    consume acid chlorides -- and every one of those turned up as a fragment that was not in
    stock and that no rule could take apart. Each coupling route terminated at its own reagent.
    """
    from rdkit import Chem

    cases = [
        ("MC 16", "O=C(Cl)c1cc(F)cc(F)c1", "O=C(O)c1cc(F)cc(F)c1"),
        ("MC 18b", "O=C(O)c1cncc(B(O)O)c1", "O=C(O)c1cncc(Br)c1"),
        ("MC 19b", "C[Sn](C)(C)c1cccc(N)c1", "Nc1cccc(Br)c1"),
        ("MC 20", "CS(=O)(=O)Cl", "CS(=O)(=O)O"),
    ]
    for code, product, expected in cases:
        got = {
            without_maps(p)
            for one in precursors(code, product)
            for p in one.split(" + ")
        }
        assert Chem.CanonSmiles(expected) in got, f"{code}: {sorted(got)}"


def test_the_medchem_set_covers_what_hartenfeller_omits() -> None:
    """Spot checks on the families added while working through the campaign targets."""
    from rdkit import Chem

    # an aniline comes from a nitroarene, not from nowhere
    got = {
        without_maps(p)
        for one in precursors("MC 3", "Nc1ccc(Cl)cc1")
        for p in one.split(" + ")
    }
    assert Chem.CanonSmiles("O=[N+]([O-])c1ccc(Cl)cc1") in got

    # Balz-Schiemann: the aryl fluoride asked for at the start of this work
    got = {
        without_maps(p)
        for one in precursors("MC 10", "Fc1ccc(C)cc1")
        for p in one.split(" + ")
    }
    assert Chem.CanonSmiles("Cc1ccc(N)cc1") in got

    # SNAr on an azine with an alcohol, which rule 54 could only do with nitrogen
    assert precursors("MC 1a", "COc1ncccc1")
    # a pyrimidine ring, present in seven of the nineteen unsolved targets
    assert precursors("MC 7", "Cc1ccnc(N)n1")


def test_the_thioether_rules_leave_sulfones_alone() -> None:
    """`iSCB 64` broke S-C bonds on sulfones and left the sulfur holding a hydrogen.

    A bare `[S:2]` matches sulfur at any oxidation state, so the Williamson-like thioether
    disconnection also fired on sulfones and sulfoxides, emitting `C[SH](=O)=O` and
    `C[SH]=O`. Pinning the sulfur divalent confines it to the sulfide the reaction needs.
    """
    for code in ("iSCB 64a", "iSCB 64b", "iSCB 64c"):
        pattern = parse_reaction_smarts(hartenfeller_rule(code))
        for oxidised in ("CS(=O)(=O)C", "CS(C)=O", "CCS(=O)(=O)C"):
            molecule, _ = cmol_from_smiles(oxidised)
            assert not generate_precursors(molecule, pattern), f"{code} on {oxidised}"
        # the sulfide it is actually for still works
        sulfide, _ = cmol_from_smiles("CSC")
        assert generate_precursors(sulfide, pattern), code


def test_common_reagents_are_the_fragments_the_catalogue_omits() -> None:
    """The reagents that killed AND-nodes on their trivial partner.

    Enamine is a screening-compound catalogue: nothing below benzyl-sized is in it, so a
    methylation dies on `CI` and a reductive amination on `CC=O` however good the other
    precursor is. Each row carries a CAS number rather than an invented catalogue code.

    Two labels, because the file outgrew its name rather than its purpose. A reagent is
    consumed to make a bond -- iodomethane, acetic anhydride, TMSCN. A building block *is* part
    of the product: D-galactose is bought, not made, for exactly the reason the reagents are,
    but calling a sugar a reagent would be wrong. It stays in this one file because the test of
    membership is "small, bulk, and absent from a screening catalogue", which is as true of a
    sugar as of methylamine. `stock-selenium.smi` is separate because organoselenium is genuinely
    exotic and only a handful of targets ever want it; sugars are not that.

    Stereochemistry is written out even though the keyer drops it, so the file reads correctly
    to a chemist. A mutarotating sugar is one purchase whichever anomer is drawn.
    """
    path = TEMPLATES / "stock-common-reagents.smi"
    rows = [line.split("\t") for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) >= 50, len(rows)
    for smiles, name, source in rows:
        cmol_from_smiles(smiles)  # parses, or this raises
        assert name.strip(), f"every row names its molecule: {smiles}"
        # A row may append its catalogue entry after the label -- "commodity reagent, Sigma
        # 295442" -- which is where a report gets the number a chemist orders by. The label
        # still has to come first, because that is what claims the row was verified at all.
        #
        # "in situ from" is the third accepted label, and it claims the same thing by a longer
        # route: the reagent is not bottled, so the row names the catalogued precursor it is
        # made from instead. Nitrous acid is the case here -- nobody buys it, everybody makes it
        # from sodium nitrite, which is two rows up with its own number. `stock-selenium.smi`
        # uses the convention for `[SeH2]` and the selenols and the README describes it; what is
        # not acceptable is a row whose availability nothing accounts for.
        verified = source.startswith(
            ("commodity reagent", "commodity building block", "in situ from")
        )

        assert verified, f"{name}: {source!r}"
    present = {smiles for smiles, _, _ in rows}
    # the specific fragments the hard-7 leaf census found dead
    for fragment in ("CI", "CBr", "CCl", "CO", "CN", "N", "CC=O", "CC#N", "CC(C)N"):
        assert fragment in present, fragment
    # and the sugar the one unsolved hard-22 target bottoms out on, without which its route
    # terminates on a specialty azide the paper's own authors did not buy either
    assert any(name == "D-galactose" for _, name, _ in rows)


def test_the_hexoses_in_stock_share_one_key_and_that_is_a_limit_worth_naming() -> None:
    """D-glucose, D-galactose and D-mannose are one molecule to the keyer. Pinned, not fixed.

    `vesma.keyer` drops stereochemistry, which everywhere else in this repository is a fair
    trade: two stereoisomers of a drug-like scaffold are made the same way. On sugars it is not,
    because the stereochemistry *is* the chemistry -- glucose and galactose differ at one centre
    and behave differently for it.

    The consequence to keep in view: a carbohydrate route that terminates on "D-galactose" was
    matched against a key three purchasable hexoses share, so its stock claim is sound up to
    constitution and no further. Every one of the three really is buyable, so the route is not
    *false*; it may simply have bought the wrong isomer. Nothing downstream repairs that.

    This is why all three sit in the stock file even though one would satisfy the matcher. The
    file is also read by chemists, and if the keyer ever learns stereochemistry the rows are
    already correct.
    """
    from vesma.keyer import rdkit_token_from_smiles

    rows = [
        line.split("\t")
        for line in (TEMPLATES / "stock-common-reagents.smi").read_text().splitlines()
        if line.strip()
    ]
    keyed: dict[str, list[str]] = {}
    for smiles, name, _source in rows:
        keyed.setdefault(rdkit_token_from_smiles(smiles), []).append(name)

    collisions = {key: sorted(names) for key, names in keyed.items() if len(names) > 1}
    shared = list(collisions.values())

    assert shared == [["D-galactose", "D-glucose", "D-mannose"]], collisions

    # the pentoses are not caught by it: they differ in constitution, not only configuration
    assert rdkit_token_from_smiles(
        "O[C@@H]1CO[C@@H](O)[C@H](O)[C@H]1O"
    ) != rdkit_token_from_smiles("OC[C@H]1O[C@@H](O)[C@H](O)[C@H]1O")


def test_reductive_amination_covers_aldehydes_not_just_ketones() -> None:
    """Hartenfeller's rule matches `[CH1:1]` only, so it sees ketones and no aldehydes.

    `R-CH2-NH-R'` is the aldehyde product and the commonest amine disconnection there is;
    it had no rule at all. MC 25 and MC 26 add it for secondary, aryl and tertiary amines.
    """
    from rdkit import Chem

    # both carbons are candidates for the aldehyde, so both disconnections are offered
    got = {
        without_maps(p)
        for one in precursors("MC 25a", "CCCNCC1CCCCC1")
        for p in one.split(" + ")
    }
    assert Chem.CanonSmiles("CCC=O") in got
    assert Chem.CanonSmiles("NCC1CCCCC1") in got

    # an aniline nitrogen, which the alkyl rule's [CX4:5] excludes
    got = {
        without_maps(p)
        for one in precursors("MC 25b", "CCCNc1ccccc1")
        for p in one.split(" + ")
    }
    assert Chem.CanonSmiles("Nc1ccccc1") in got

    # a tertiary amine loses its aldehyde and keeps both other substituents
    got = {
        without_maps(p)
        for one in precursors("MC 26", "CCCN(C)C1CCCCC1")
        for p in one.split(" + ")
    }
    assert Chem.CanonSmiles("CNC1CCCCC1") in got

    # the ketone case stays with Hartenfeller: MC 25a must not claim it
    assert not precursors("MC 25a", "CC(C)NC1CCCCC1")


def test_azole_n_alkylation_reaches_pyrazoles_and_fused_rings() -> None:
    """MC 13 arylates an azole nitrogen; nothing alkylated one.

    The nitrogen is written with `[a:2]`/`[a:3]` neighbours rather than `[c]` because a
    pyrazole N1 sits next to N2, which a carbon-only pattern misses.
    """
    from rdkit import Chem

    for code, product, expected in (
        ("MC 27b", "Cn1cncc1", "CBr"),  # imidazole: N between two carbons
        (
            "MC 27e",
            "Cc1nn(CC(F)(F)F)cc1N",
            "FC(F)(F)CBr",
        ),  # pyrazole: carbon and nitrogen
        (
            "MC 27b",
            "Cn1cnc2c(C)cncc21",
            "CBr",
        ),  # fused imidazopyridine, hard-7 target 7
        ("MC 27h", "Cn1nnnn1", "CBr"),  # 2H-tetrazole: N between two nitrogens
    ):
        got = {
            without_maps(p)
            for one in precursors(code, product)
            for p in one.split(" + ")
        }
        assert Chem.CanonSmiles(expected) in got, f"{product}: {sorted(got)}"

    # the three ring types partition: each variant declines the others' substrates
    assert not precursors("MC 27b", "Cc1nn(CC(F)(F)F)cc1N")
    assert not precursors("MC 27e", "Cn1cncc1")


def test_esters_can_come_from_an_alkyl_halide_not_only_an_alcohol() -> None:
    """hard-7 target 4 needs 2-ethyldecan-1-ol, which Enamine does not carry.

    It does carry the corresponding bromide, and alkylating a carboxylate with a primary
    halide is ordinary K2CO3/DMF chemistry, so MC 29 offers that disconnection alongside
    the acid-plus-alcohol one. Secondary esters are excluded: `[CH2;D2:3]` is primary.
    """
    from rdkit import Chem

    got = {
        without_maps(p)
        for one in precursors("MC 29b", "CCCCCCCC(CC)COC(=O)CCC")
        for p in one.split(" + ")
    }
    assert Chem.CanonSmiles("CCCCCCCC(CC)CBr") in got
    assert Chem.CanonSmiles("CCCC(=O)O") in got

    # a secondary ester is not made this way, and the rule must not claim it
    assert not precursors("MC 29b", "CC(=O)OC(C)C")


def test_the_ring_constructions_only_fire_on_fused_systems() -> None:
    """A quinoline pattern matches plain pyridine too, and breaking that ring is invalid.

    Written as a bare six-ring, MC 30 and MC 31 matched pyridine and 2-hydroxypyrimidine and
    raised OrphanDelocalizedBondError -- the two broken bonds leave aromatic atoms in no
    circuit. The benzo ring is therefore written out, as Hartenfeller's indole and
    benzothiazole rules do, so a monocycle simply does not match.
    """
    from rdkit import Chem

    for code, monocycle in (
        ("MC 30a", "c1ccncc1"),
        ("MC 30b", "Cc1ccncc1"),
        ("MC 31", "Oc1ncccn1"),
    ):
        assert not precursors(code, monocycle), f"{code} matched {monocycle}"

    # Friedlaender: quinoline back to 2-aminobenzaldehyde and a methyl ketone
    got = {
        without_maps(p)
        for one in precursors("MC 30a", "c1ccc2ncccc2c1")
        for p in one.split(" + ")
    }
    assert Chem.CanonSmiles("Nc1ccccc1C=O") in got
    assert Chem.CanonSmiles("CC=O") in got

    # the C3-substituted variant takes the higher ketone and leaves C3-H to MC 30a
    got = {
        without_maps(p)
        for one in precursors("MC 30b", "Cc1cc2ccccc2nc1C")
        for p in one.split(" + ")
    }
    assert Chem.CanonSmiles("CCC(C)=O") in got
    assert not precursors("MC 30b", "c1ccc2ncccc2c1")

    # quinazolin-2-ol back to 2-aminobenzaldehyde and urea
    got = {
        without_maps(p)
        for one in precursors("MC 31", "Oc1ncc2ccccc2n1")
        for p in one.split(" + ")
    }
    assert Chem.CanonSmiles("NC(N)=O") in got
    assert Chem.CanonSmiles("Nc1ccccc1C=O") in got


def test_the_reinvent_checklist_gaps_are_scoped_to_what_they_are_for() -> None:
    """Probing REINVENT4's 34 reaction types left three with no rule; two were worth adding.

    Wurtz is deliberately absent: coupling two alkyl halides over sodium gives statistical
    mixtures, and REINVENT4 carries it to enumerate libraries rather than because anyone runs
    it. The other two are real named reactions, and each must not claim its neighbours.
    """
    from rdkit import Chem

    got = {
        without_maps(p)
        for one in precursors("MC 32", "CCOC(=N)C")
        for p in one.split(" + ")
    }
    assert Chem.CanonSmiles("CC#N") in got
    assert Chem.CanonSmiles("CCO") in got
    # an ester is not an imidate
    assert not precursors("MC 32", "CCOC(C)=O")

    got = {
        without_maps(p)
        for one in precursors("MC 33", "CC[N+](=O)[O-]")
        for p in one.split(" + ")
    }
    assert Chem.CanonSmiles("CCBr") in got
    # a nitroarene comes from nitration, not from displacing a halide with nitrite
    assert not precursors("MC 33", "O=[N+]([O-])c1ccccc1")


def test_aryl_rings_can_gain_a_nitrile_acid_or_aldehyde() -> None:
    """Nothing in the library made an aryl aldehyde or an aryl carboxylic acid.

    That is load-bearing rather than cosmetic: every amide route ends at an acid, so with no
    way to make one, a route could only terminate on an acid the catalogue happened to carry.
    MC 34 to MC 37 are the palladium functionalisations of an aryl halide plus the methyl
    oxidation; MC 38 and MC 39 are the alcohol oxidations.
    """
    from rdkit import Chem

    for code, product, expected in (
        ("MC 34b", "N#Cc1ccccc1", "Brc1ccccc1"),
        ("MC 35b", "OC(=O)c1ccccc1", "Brc1ccccc1"),
        ("MC 36", "OC(=O)c1ccccc1", "Cc1ccccc1"),
        ("MC 37b", "O=Cc1ccccc1", "Brc1ccccc1"),
        ("MC 38", "O=Cc1ccccc1", "OCc1ccccc1"),
        ("MC 39", "CC(=O)C", "CC(C)O"),
    ):
        got = {
            without_maps(p)
            for one in precursors(code, product)
            for p in one.split(" + ")
        }
        assert Chem.CanonSmiles(expected) in got, f"{code}: {sorted(got)}"

    # each must stop at what it is for: a benzyl nitrile is not an aryl nitrile, an ester is
    # not an acid, an aryl ketone is not an aldehyde, and an acid is not a ketone
    assert not precursors("MC 34b", "N#CCc1ccccc1")
    assert not precursors("MC 35b", "COC(=O)c1ccccc1")
    assert not precursors("MC 37b", "CC(=O)c1ccccc1")
    assert not precursors("MC 39", "CC(=O)O")


def test_the_oxindole_rule_needs_a_fused_ring() -> None:
    """hard-7 target 3 blocks on a 2-alkoxy-azaindole; strip the ether and it is an oxindole.

    The disconnection is the reductive cyclisation of a (2-aminoaryl)acetic acid, which breaks
    one bond -- N1 to C2 -- so unlike the quinoline rules the fused ring survives untouched.
    It still has to be *written*, generically as `[a]` atoms rather than as a benzo ring, or
    the rule covers indoles but not the azaindole the target actually carries.
    """
    from rdkit import Chem

    for code, product, expected in (
        ("MC 40a", "Oc1cc2ccccc2[nH]1", "Nc1ccccc1CC(=O)O"),
        ("MC 40b", "Oc1cc2ncccc2[nH]1", "Nc1cccnc1CC(=O)O"),
    ):
        got = {
            without_maps(p)
            for one in precursors(code, product)
            for p in one.split(" + ")
        }
        assert Chem.CanonSmiles(expected) in got, f"{product}: {sorted(got)}"

    # one flat variant per aza position rather than one pattern with [a] neighbours: the
    # matcher is built for flat patterns, and between them the five cover the same ground
    fired = {
        code
        for code in ("MC 40a", "MC 40b", "MC 40c", "MC 40d", "MC 40e")
        if precursors(code, "Oc1cc2nccc(O)c2[nH]1")
    }
    assert fired == {"MC 40b"}, fired

    # a 2-hydroxypyrrole with nothing fused to it would leave aromatic atoms in no circuit
    assert not precursors("MC 40a", "Oc1ccc[nH]1")
    assert not precursors("MC 40a", "Cc1cc(O)[nH]c1")
    # and the oxygen analogue is a benzofuranone, made another way
    assert not precursors("MC 40a", "Oc1cc2ccccc2o1")


def test_aromatic_rings_can_gain_a_halide_or_a_nitro_group() -> None:
    """Aryl halides could only come from an aniline, and nitroarenes from nowhere at all.

    With no electrophilic halogenation and no nitration, every aromatic route had to begin
    from a ring the catalogue already carried substituted. hard-7 target 3 stops exactly
    there: its naphthalene half is a bromide nothing could put a bromine on.
    """
    from rdkit import Chem

    for code, product, expected in (
        ("MC 41b", "Brc1ccccc1", "c1ccccc1"),
        ("MC 42", "O=[N+]([O-])c1ccccc1", "c1ccccc1"),
    ):
        got = {
            without_maps(p)
            for one in precursors(code, product)
            for p in one.split(" + ")
        }
        assert Chem.CanonSmiles(expected) in got, f"{code}: {sorted(got)}"

    # a benzylic halide is not an aryl halide, and a nitroalkane is not a nitroarene
    assert not precursors("MC 41b", "BrCc1ccccc1")
    assert not precursors("MC 42", "C[N+](=O)[O-]")


def test_the_interconversions_from_the_reaction_class_checklist() -> None:
    """Gaps found by probing a ranked list of reaction classes, authored flat here.

    The list gives names and a quality rank, no SMARTS and no licence, so it is read as a
    checklist of what a medicinal chemist actually runs and the patterns are our own. Two
    entries were left out deliberately: decarboxylation and deamination reduce in retro to
    "any alkane" and "any arene", which is fanout without chemistry.
    """
    from rdkit import Chem

    for code, product, expected in (
        ("MC 43a", "CCC(=O)O", "CCC(=O)OC"),  # saponification
        ("MC 44", "Oc1ccccc1", "COc1ccccc1"),  # BBr3 demethylation
        ("MC 45", "CCCN", "CCCN=[N+]=[N-]"),  # Staudinger
        ("MC 46b", "CCCN=[N+]=[N-]", "CCCBr"),  # azide displacement
        ("MC 47b", "CCCC#N", "CCCBr"),  # cyanide displacement
        ("MC 48", "CCC(N)=N", "CCC#N"),  # amidine
        ("MC 49", "CCC(N)=NO", "CCC#N"),  # amidoxime, the oxadiazole precursor
        ("MC 50", "CC(C)=NO", "CC(C)=O"),  # oxime
        ("MC 51", "CCC(N)=S", "CCC(N)=O"),  # Lawesson
        ("MC 54", "CCCF", "CCCO"),  # deoxyfluorination
        ("MC 55b", "CC1(C)OB(c2ccccc2)OC1(C)C", "Brc1ccccc1"),  # Miyaura to Bpin
        ("MC 56", "CCCS", "CCCBr"),
    ):
        got = {
            without_maps(p)
            for one in precursors(code, product)
            for p in one.split(" + ")
        }
        assert Chem.CanonSmiles(expected) in got, f"{code}: {sorted(got)}"

    # each declines its nearest neighbour
    assert not precursors("MC 43a", "CCC(=O)OC")  # an ester is not the acid
    assert not precursors("MC 44", "OCc1ccccc1")  # a benzyl alcohol is not a phenol
    assert not precursors("MC 47a", "N#Cc1ccccc1")  # aryl nitriles go by MC 34
    assert not precursors("MC 48", "CCC(N)=O")  # an amide is not an amidine
    assert not precursors("MC 54", "Fc1ccccc1")  # an aryl fluoride is Balz-Schiemann


def test_methyl_ethers_and_glycosides_have_a_route() -> None:
    """The Williamson rule needs `[CH2]`, so a methyl ether on a ring CH matched nothing.

    hard-22 target 13 stops there: its methyl glycoside was neither stocked nor makeable.
    MC 58 is the O-methylation the Williamson pattern excluded; MC 59 is the Fischer
    glycosidation, which is how an anomeric methyl ether is actually made.
    """
    from rdkit import Chem

    glucoside = "COC1OC(CO)C(O)C(O)C1O"
    glucose = "OCC1OC(O)C(O)C(O)C1O"

    got = {
        without_maps(p)
        for one in precursors("MC 58c", glucoside)
        for p in one.split(" + ")
    }
    assert Chem.CanonSmiles(glucose) in got, sorted(got)
    assert Chem.CanonSmiles("CI") in got

    # Fischer needs the anomeric carbon -- two oxygens on it -- not any methyl ether
    got = {
        without_maps(p)
        for one in precursors("MC 59", glucoside)
        for p in one.split(" + ")
    }
    assert Chem.CanonSmiles(glucose) in got, sorted(got)
    assert not precursors("MC 59", "CCOC")

    # an aryl methyl ether is demethylated by MC 44, not alkylated by MC 58
    assert not precursors("MC 58c", "COc1ccccc1")


def test_the_remaining_ring_formations_and_the_sulfinamide() -> None:
    """Probing the heterocycle families left three real gaps once probes were substituted.

    Most apparent gaps were artefacts: Hartenfeller's benzimidazole, thiazole and pyrazole
    rules want a substituent, so probing with the bare parent ring matched nothing. With
    substituted probes only benzimidazolone, benzotriazole and alkene epoxidation were
    genuinely missing, plus Ellman's sulfinamide from the protection family.
    """
    from rdkit import Chem

    diamine = Chem.CanonSmiles("Nc1ccccc1N")
    for code, product in (
        ("MC 60a", "O=c1[nH]c2ccccc2[nH]1"),
        ("MC 61a", "c1ccc2[nH]nnc2c1"),
    ):
        got = {
            without_maps(p)
            for one in precursors(code, product)
            for p in one.split(" + ")
        }
        assert diamine in got, f"{code}: {sorted(got)}"

    got = {
        without_maps(p)
        for one in precursors("MC 62", "C1CC2OC2CC1")
        for p in one.split(" + ")
    }
    assert Chem.CanonSmiles("C1=CCCCC1") in got, sorted(got)

    got = {
        without_maps(p)
        for one in precursors("MC 63", "CC(C)(C)S(=O)NC(C)C")
        for p in one.split(" + ")
    }
    assert Chem.CanonSmiles("CC(C)N") in got, sorted(got)

    # only three aza variants each: benzimidazolone and benzotriazole are symmetric, so a
    # nitrogen at C4 and at C7 describe the same molecule and a fourth pattern is dead weight
    assert {
        c
        for c in ("MC 60a", "MC 60b", "MC 60c")
        if precursors(c, "O=c1[nH]c2ncccc2[nH]1")
    } == {"MC 60b"}
    assert {
        c for c in ("MC 61a", "MC 61b", "MC 61c") if precursors(c, "c1cc2[nH]nnc2cn1")
    } == {"MC 61c"}


def test_the_last_gaps_from_the_checklist() -> None:
    """Probing the remaining families left eight gaps in forty-five; seven were worth adding.

    Alkene hydrogenation was not: in retro it is "any alkane from any alkene at any
    position", the same fanout-without-chemistry as decarboxylation. A triazolopyridine rule
    was drafted and dropped -- its [4,3-a] and [1,5-a] isomers come from a hydrazinopyridine
    and an aminopyridine respectively, and a rule that cannot tell them apart would invent a
    precursor.
    """
    from rdkit import Chem

    for code, product, expected in (
        ("MC 64b", "CCSc1ccccc1", "CCS"),  # S-arylation
        ("MC 65b", "CCSc1ccccc1", "Sc1ccccc1"),  # S-alkylation of the thiophenol
        ("MC 66b", "CC(O)(c1ccccc1)c1ccccc1", "CC(=O)c1ccccc1"),  # aryl metal addition
        ("MC 67", "C[N+](C)([O-])C", "CN(C)C"),  # N-oxide
        ("MC 68", "CCOP(=O)(OCC)CC", "CCBr"),  # Arbuzov
        ("MC 69", "CCOP(=O)(OCC)OCC", "CCO"),
        ("MC 70a", "O=c1[nH]c(=O)c2ccccc2[nH]1", "Nc1ccccc1C(=O)O"),  # anthranilic acid
    ):
        got = {
            without_maps(p)
            for one in precursors(code, product)
            for p in one.split(" + ")
        }
        assert Chem.CanonSmiles(expected) in got, f"{code}: {sorted(got)}"

    # the N-oxide states both charge and hydrogen count on the precursor nitrogen; stating
    # neither leaves the charge on, and stating only the charge raises
    assert not precursors("MC 67", "C[N+](C)(C)C")
    # a dialkyl sulfide is not an aryl thioether
    assert not precursors("MC 64b", "CCSCC")

    # quinazolinedione's benzo ring is not symmetric, so all five variants earn their place
    fired = {
        code: bool(precursors(code, "O=c1[nH]c(=O)c2ccncc2[nH]1"))
        for code in ("MC 70a", "MC 70b", "MC 70c", "MC 70d", "MC 70e")
    }
    assert [c for c, hit in fired.items() if hit] == ["MC 70d"], fired


# --------------------------------------------------------------------------- #
# Library coverage
# --------------------------------------------------------------------------- #


# One representative product per named transformation, drawn from two published checklists
# of reaction classes -- REINVENT4's fragmentation definitions and a list of 348 classes
# ranked by how reliably they work. Neither supplied usable SMARTS, so these are the probes
# that were used to find the gaps, kept as a regression guard.
#
# The assertion is deliberately weak: *some* rule in the merged library disconnects this
# product. It does not check that the right rule fires for the right reason -- the per-rule
# tests above do that. What it catches is a class going uncovered again after a refactor,
# which no per-rule test can, because it does not care which rule supplies it.
def test_the_published_route_to_the_one_unsolved_hard22_target_is_now_expressible() -> (
    None
):
    """Target 13 is compound 7 of ChemMedChem 2025, 20, e202401012, and its route is in the SI.

    Walking that route step by step left three consecutive gaps, all in the front end: the
    anomeric C-C bond, the CuAAC *from the azide*, and nitrile hydrolysis to the acid. Everything
    from acid 2 onward was already covered. These are the three, checked on the real
    intermediates rather than on look-alikes.

    Hartenfeller carries a Huisgen already, but spelled from the alkyl halide, which offers a
    3-chloro sugar where the route uses 3-azido-3-deoxy galactose. That is why MC 72 is ours and
    is not an edit to the Hartenfeller table, which is a citation rather than a working list.
    """
    from rdkit import Chem

    for code, product, expected in (
        # I-1 + TMSCN/BF3: the glycosyl cyanide, on a plain pyranose so the test does not
        # depend on the campaign's protecting groups
        ("Gly 1a", "N#CC1CCCCO1", "CC(=O)OC1CCCCO1"),
        # I-2 + aryl alkyne, CuI: the azide is the precursor, not a halide
        ("MC 72", "CCn1cc(-c2ccccc2)nn1", "C#Cc1ccccc1"),
        ("MC 72", "CCn1cc(-c2ccccc2)nn1", "CC[N]=[N+]=[N-]"),
        # HCl reflux: nitrile to acid
        ("MC 71", "CCC(=O)O", "CCC#N"),
    ):
        got = {
            without_maps(p)
            for one in precursors(code, product)
            for p in one.split(" + ")
        }
        assert Chem.CanonSmiles(expected) in got, f"{code}: {sorted(got)}"

    # and each stops at what it is for
    # an ester is not an acid, so nitrile hydrolysis must not reach for it
    assert not precursors("MC 71", "CCC(=O)OC")
    # a benzotriazole has no H on C5, so the cycloaddition must not claim it
    assert not precursors("MC 72", "CCn1nnc2ccccc21")
    # 1,5-disubstituted is the ruthenium regiochemistry, not this one
    assert not precursors("MC 72", "CCn1c(-c2ccccc2)cnn1")
    # a nitrile needs an anomeric carbon to be a C-glycoside; a plain one is not
    assert not precursors("Gly 1a", "N#CC1CCCCC1")


def test_a_sugar_can_be_reached_from_a_bulk_one_without_buying_the_azide() -> None:
    """The leg below the campaign intermediate, and it needed no carbohydrate chemistry at all.

    3-Azido-3-deoxy-tetra-O-acetylgalactose is not a commodity -- the paper's own SI makes it
    from a literature procedure rather than buying it. Reaching it from a bulk sugar wants an
    alcohol activated as a sulfonate, displaced by azide, and the polyol acetylated. All three
    are ordinary, which is why they are here and not in the carbohydrate set: the only genuinely
    sugar-specific step in this whole route is the anomeric C-C bond.

    The esterification is the one that surprises. `MC 29` made an ester from an acid and an
    *alkyl halide*; nothing made one from an acid and an *alcohol*, which is the mirror of
    `Cpl 1` and is how every acetate on a sugar is put on.
    """
    from rdkit import Chem

    for code, product, expected in (
        # One sulfonate rule now, the sulfonyl group riding a mapped star, so the same row
        # answers for the triflate and the mesylate where there used to be three.
        ("MC 73a", "CCOS(=O)(=O)C(F)(F)F", "CCO"),
        ("MC 73a", "CCOS(C)(=O)=O", "CCO"),
        ("MC 74c", "CCN=[N+]=[N-]", "CCOS(=O)(=O)C(F)(F)F"),
        ("MC 74a", "CCN=[N+]=[N-]", "CCOS(C)(=O)=O"),
        ("MC 75", "CCOC(C)=O", "CC(=O)O"),
        ("MC 75", "CCOC(C)=O", "CCO"),
    ):
        got = {
            without_maps(p)
            for one in precursors(code, product)
            for p in one.split(" + ")
        }
        assert Chem.CanonSmiles(expected) in got, f"{code}: {sorted(got)}"

    # an amide is not an ester, and a lactone is a ring closure rather than this disconnection
    assert not precursors("MC 75", "CCNC(C)=O")
    assert not precursors("MC 75", "O=C1CCCO1")
    # the sulfonate rules want a sulfonate ester, not a sulfone or a sulfonamide
    assert not precursors("MC 73a", "CCS(C)(=O)=O")
    assert not precursors("MC 73a", "CCNS(C)(=O)=O")


COVERED_TRANSFORMATIONS = (
    ("amide from acid and amine", "CCNC(=O)CC"),
    ("CuAAC 1,4-triazole", "CCn1cc(-c2ccccc2)nn1"),
    ("C-glycosyl nitrile", "N#CC1CCCCO1"),
    ("nitrile hydrolysis to an acid", "CCC(=O)O"),
    ("ester from an acid and an alcohol", "CCOC(C)=O"),
    ("sulfonate ester from an alcohol", "CCOS(C)(=O)=O"),
    ("amide N-arylation", "CNc1ccccc1"),
    ("Buchwald-Hartwig", "CN(C)c1ccccc1"),
    ("SNAr on an azine", "COc1ccccn1"),
    ("Suzuki biaryl", "c1ccc(-c2ccccn2)cc1"),
    ("Stille biaryl", "c1ccc(-c2ccccc2)cc1"),
    ("Sonogashira", "C#Cc1ccccc1"),
    ("Heck", "C=Cc1ccccc1"),
    ("Ullmann biaryl", "c1ccc(-c2ccccc2)cc1"),
    ("reductive amination, aldehyde", "CCCNCC"),
    ("reductive amination, ketone", "CC(C)NC1CCCCC1"),
    ("amine alkylation", "CCNCC"),
    ("azole N-alkylation", "Cn1cncc1"),
    ("Williamson ether", "CCOCC"),
    ("O-methylation", "COC1OC(CO)C(O)C(O)C1O"),
    ("aryl ether", "CCOc1ccccc1"),
    ("ester from acid and alcohol", "CCOC(=O)CC"),
    ("ester by O-alkylation", "CCCCCCCC(CC)COC(=O)CCC"),
    ("ester hydrolysis", "CCC(=O)O"),
    ("phenol demethylation", "Oc1ccccc1"),
    ("sulfonamide", "CCNS(=O)(=O)c1ccccc1"),
    ("urea", "CCNC(=O)NCC"),
    ("thiourea", "CCNC(=S)NCC"),
    ("carbamate", "CCNC(=O)OCC"),
    ("guanidine", "N=C(N)NCC"),
    ("aryl halogenation", "Brc1ccccc1"),
    ("aromatic nitration", "O=[N+]([O-])c1ccccc1"),
    ("nitro to aniline", "Nc1ccc(Cl)cc1"),
    ("Sandmeyer nitrile", "N#Cc1ccccc1"),
    ("aryl carboxylic acid", "OC(=O)c1ccccc1"),
    ("aryl aldehyde", "O=Cc1ccccc1"),
    ("boronic acid", "OB(O)c1ccccc1"),
    ("pinacol boronate", "CC1(C)OB(c2ccccc2)OC1(C)C"),
    ("aryl stannane", "C[Sn](C)(C)c1ccccc1"),
    ("acid chloride", "CC(=O)Cl"),
    ("alkyl halide from alcohol", "CCCBr"),
    ("alkyl nitrile from halide", "CCCC#N"),
    ("alkyl azide", "CCCN=[N+]=[N-]"),
    ("Staudinger to amine", "CCCN"),
    ("deoxyfluorination", "CCCF"),
    ("thiol from halide", "CCCS"),
    ("thioether", "CCSCC"),
    ("aryl thioether", "CCSc1ccccc1"),
    ("sulfoxide", "CS(C)=O"),
    ("sulfone", "CS(C)(=O)=O"),
    ("amine N-oxide", "C[N+](C)([O-])C"),
    ("amidine from nitrile", "CCC(N)=N"),
    ("amidoxime from nitrile", "CCC(N)=NO"),
    ("oxime", "CC(C)=NO"),
    ("thioamide", "CCC(N)=S"),
    ("hydrazide", "CCC(=O)NN"),
    ("hydroxamic acid", "CCC(=O)NO"),
    ("primary amide from nitrile", "CCC(N)=O"),
    ("alcohol from aldehyde", "CCCO"),
    ("aldehyde from alcohol", "CCC=O"),
    ("ketone from alcohol", "CC(=O)C"),
    ("epoxide from alkene", "C1CC2OC2CC1"),
    ("phosphonate, Arbuzov", "CCOP(=O)(OCC)CC"),
    ("phosphate ester", "CCOP(=O)(OCC)OCC"),
    ("quinoline, Friedlaender", "c1ccc2ncccc2c1"),
    ("quinazolin-2-ol", "Oc1ncc2ccccc2n1"),
    ("quinazolin-4-one", "O=c1[nH]cnc2ccccc12"),
    ("quinazoline-2,4-dione", "O=c1[nH]c(=O)c2ccccc2[nH]1"),
    ("oxindole", "Oc1cc2ccccc2[nH]1"),
    ("benzimidazole", "Cc1nc2ccccc2[nH]1"),
    ("benzimidazolone", "O=c1[nH]c2ccccc2[nH]1"),
    ("benzotriazole", "c1ccc2[nH]nnc2c1"),
    ("benzothiazole", "Cc1nc2ccccc2s1"),
    ("benzoxazole", "Cc1nc2ccccc2o1"),
    ("indazole", "Cc1n[nH]c2ccccc12"),
    ("indole", "c1ccc2[nH]ccc2c1"),
    ("pyrazole", "Cc1cc(C)[nH]n1"),
    ("imidazole", "Cc1cnc(C)[nH]1"),
    ("thiazole", "Cc1csc(C)n1"),
    ("isoxazole", "c1cc(C)no1"),
    ("tetrazole", "Cc1nnn[nH]1"),
    ("triazole, Huisgen", "Cn1cc(C)nn1"),
    ("1,2,4-oxadiazole", "Cc1nc(C)no1"),
    ("1,3,4-oxadiazole", "Cc1nnc(C)o1"),
    ("pyrimidine", "c1cncnc1"),
    ("Pictet-Spengler", "C1Cc2ccccc2CN1"),
    ("hydantoin, Bucherer-Bergs", "O=C1NC(=O)NC1(C)C"),
    ("aldol", "CC(O)CC(C)=O"),
    ("Knoevenagel", "CCOC(=O)C(=Cc1ccccc1)C(=O)OCC"),
    ("Horner-Wadsworth-Emmons", "CC=Cc1ccccc1"),
    ("Diels-Alder", "C1=CCCCC1"),
    ("Henry", "OC(C[N+](=O)[O-])c1ccccc1"),
    ("Strecker", "CC(N)C#N"),
    ("Grignard alcohol", "CC(O)c1ccccc1"),
    ("aryl metal addition", "CC(O)(c1ccccc1)c1ccccc1"),
    ("Friedel-Crafts acylation", "CC(=O)c1ccccc1"),
    ("N-Boc protection", "CCNC(=O)OC(C)(C)C"),
    ("N-Cbz protection", "CCNC(=O)OCc1ccccc1"),
    ("N-Fmoc protection", "CCNC(=O)OCC1c2ccccc2-c2ccccc21"),
    ("O-TBS protection", "CCO[Si](C)(C)C(C)(C)C"),
    ("O-THP protection", "CCOC1CCCCO1"),
    ("acetonide", "CC1(C)OCC(C)O1"),
    ("Ellman sulfinamide", "CC(C)(C)S(=O)NC(C)C"),
    ("Fischer glycosidation", "COC1OC(CO)C(O)C(O)C1O"),
)


@cache
def merged_library_patterns() -> tuple:
    """Every rule the curated sets contribute, compiled once."""
    from tests.vertsy.bench_host_aostar import TEMPLATE_FILES, load_merged_library

    names = (
        "hartenfeller",
        "medchem",
        "selenium",
        "protecting-groups",
    )
    library = load_merged_library(tuple((n, TEMPLATE_FILES[n]) for n in names), ())
    return tuple(parse_reaction_smarts(smarts) for smarts in library.smarts)


@pytest.mark.parametrize("label,product", COVERED_TRANSFORMATIONS, ids=lambda v: None)
def test_the_library_still_covers_each_named_transformation(
    label: str, product: str
) -> None:
    """Every class the checklists named, and that the library covered when it was probed."""
    molecule, _ = cmol_from_smiles(product)

    for pattern in merged_library_patterns():
        # a rule that refuses this particular product is not a failure of the library
        try:
            if generate_precursors(molecule, pattern):
                return
        except Exception:  # noqa: BLE001
            continue

    raise AssertionError(f"no rule disconnects {label}: {product}")


def test_the_anomeric_position_can_carry_a_halide() -> None:
    """Every halide-from-alcohol rule required carbon neighbours, so sugars were excluded.

    `FGI 4` asks for `[CH1](-[CX4])(-[CX4])`, two sp3 carbons. An anomeric carbon has one
    carbon and the ring oxygen, so no rule in the library could put a halide there -- and a
    glycosyl halide is how essentially every glycosyl donor is made. Found by walking hard-22's
    last unsolved target back by hand: the free sugar was reachable, the glycosyl chloride one
    step above it was not.
    """
    from rdkit import Chem

    got = {
        without_maps(p)
        for one in precursors("FGI 7a", "ClC1OC(CO)C(O)C(O)C1O")
        for p in one.split(" + ")
    }
    assert Chem.CanonSmiles("OCC1OC(O)C(O)C(O)C1O") in got, sorted(got)

    # an alpha-alkoxy halide is the same chemistry and is covered by the same rule
    assert precursors("FGI 7a", "CCOC(C)Cl")

    # a halide with no oxygen on its carbon belongs to FGI 4, and MOM-Cl is a CH2 case
    assert not precursors("FGI 7a", "CC(Cl)C")
    assert not precursors("FGI 7a", "ClCOC")


# Each pair is one disconnection written twice: the parent asserts two precursors, the sibling
# asserts one molecule in two fragments. The molecules are chosen so each spelling has exactly
# one reading available -- a lactam whose nitrogen carries an aryl rather than a second alkyl,
# say, so the only `[CX4]` the rule can cut is the ring one.
# The last field is how many *reagents* the pair's rules name, which the molecularity claim has to
# look past: a reductive amination names its hydride and the water that protonates the alkoxide, so
# its precursor count is two higher than the number of starting materials it is making a claim
# about. The claim itself is unchanged -- parent two substrates, sibling one molecule in two
# fragments -- and writing the reagent count down is what keeps it readable as that claim.
GROUPED_PAIRS = (
    ("Hartenfeller 30c", "MC 99", "CN1CCCCC1", "CCCN(C)C", 2),
    ("Hartenfeller 29a", "MC 100a", "C1CCOCC1", "CCOCC", 0),
    ("MC 8a", "MC 101a", "O=C1CCCN1c1ccccc1", "CC(=O)N(C)c1ccccc1", 0),
    ("Hartenfeller 47", "MC 102", "O=C1CCCCCN1", "CC(=O)NC", 0),
    ("MC 26", "MC 103", "CN1CCCCC1", "CCCN(C)C", 2),
)


@pytest.mark.parametrize(
    ("parent", "sibling", "cyclic", "acyclic", "reagents"),
    GROUPED_PAIRS,
    ids=[pair[0] for pair in GROUPED_PAIRS],
)
def test_a_grouped_rule_and_its_sibling_divide_the_two_readings(
    parent: str, sibling: str, cyclic: str, acyclic: str, reagents: int
) -> None:
    """What `test_a_two_component_rule_closes_a_ring_without_a_second_rule` describes, split.

    That test states the position these rules used to occupy: a two-component rule *already*
    closes a ring, because the halves stay connected through the rest of the cycle and
    separation hands back one precursor. True, and the reason the macrocycle set was merged
    away -- but it means one rule gave two different answers about its own molecularity, and
    nothing in the template could say which was meant. That is the defect the search exploits,
    and `MC 72` still sits on the old side of it.

    These pairs say it instead. The parent is `>>(B).(C)`: two starting materials, so a ring
    substrate is refused rather than silently answered with one precursor. The sibling is
    `>>(B.C)`: one molecule in two fragments, which is the ring closure, and which refuses the
    acyclic substrate it was never about. Across 150 PaRoutes targets the split is exact --
    410 outcomes left the parents and the same 410 arrived at the siblings -- so nothing was
    lost, and every one of them now carries a claim about its own molecularity that is true.
    """
    for code, molecule, expected in (
        (parent, acyclic, 2 + reagents),
        (parent, cyclic, None),
        (sibling, cyclic, 1 + reagents),
        (sibling, acyclic, None),
    ):
        outcomes = precursors(code, molecule)
        if expected is None:
            assert not outcomes, f"{code} should not fire on {molecule}"
        else:
            molecule_counts = {one.count(" + ") + 1 for one in outcomes}
            assert molecule_counts == {expected}, f"{code} on {molecule}: {outcomes}"


def test_the_esterification_rules_refuse_a_lactone_and_a_sibling_takes_it() -> None:
    """The defect the subsumption reading found, and the chemistry it was hiding.

    `iSCB 70` and `MC 75` were both "ester from an acid and an alcohol", and `iSCB 70` was broader
    by exactly one thing: it omitted `MC 75`'s `-!@` on the ester C-O. That breadth was a bug. On
    gamma-butyrolactone it cut the ring bond and handed back **one** precursor, the ring-opened
    hydroxy acid, while its text claimed two -- the molecularity defect, in a rule that had never
    been migrated.

    A rule broader than its near-twin is broader *somewhere*, and that somewhere is where to look.
    `iSCB 70` has since been deleted as a duplicate of `MC 75`; see
    `test_the_duplicate_esterification_row_is_gone`.
    """
    for code in ("MC 75",):
        assert precursors(code, "CCOC(C)=O"), f"{code} still makes an ester"
        assert not precursors(code, "O=C1CCCO1"), f"{code} must refuse a lactone"
        assert not precursors(code, "O=C1CCCCO1"), f"{code} must refuse a lactone"

    # And the reaction is real, so it has a home rather than being refused outright.
    made = precursors("MC 104", "O=C1CCCO1")
    assert {without_maps(one) for one in made} == {"O=C(O)CCCO"}
    # One molecule, which is the whole claim a `(B.C)` right-hand side makes.
    assert all(" + " not in one for one in made)
    assert not precursors("MC 104", "CCOC(C)=O"), "lactonisation is not esterification"


def test_the_duplicate_esterification_row_is_gone() -> None:
    """`iSCB 70` was `MC 75` character for character, so only one of them ever compiled.

    Which one was an accident of load order: `templates-hartenfeller.csv` is read first, so the
    code that reached a report was `iSCB 70` and the row doing the work lived in the medchem table
    under another name. The duplicate is deleted and `MC 75` is the rule.

    Deleting it is a no-op for what a search compiles -- that is the whole argument for doing it.
    The row was already dropped at load; it only ever looked like a rule.
    """
    with pytest.raises(AssertionError):
        rule("iSCB 70")
    assert rule("MC 75"), "and the surviving twin is the one that does the chemistry"
