from __future__ import annotations

from typing import Any

import pytest

from vertsy.replay import (
    matching_application_rows,
    step_precursor_names,
    template_replay_routes,
)
from vertsy.routes import Route
from vertsy.search import MoleculeBatch, object_mol_array
from vesma.matcher import hit_array


def keyed_match(
    outcomes: dict[tuple[str, int], tuple[tuple[tuple[str, str], ...], ...]],
):
    def match(
        mols: MoleculeBatch[str], pairs: list[tuple[int, int]]
    ) -> tuple[MoleculeBatch[str], list[str], Any]:
        precursors: list[str] = []
        keys: list[str] = []
        hits: list[tuple[int, int, int, int]] = []
        for mol_idx, rule in pairs:
            for app_idx, application in enumerate(
                outcomes.get((mols[mol_idx], rule), ())
            ):
                for molecule, key in application:
                    hits.append((mol_idx, rule, app_idx, len(precursors)))
                    precursors.append(molecule)
                    keys.append(key)
        return object_mol_array(precursors), keys, hit_array(hits)

    return match


def test_template_replay_translates_an_opaque_key_route_to_readable_keys() -> None:
    reference = Route(
        "hash-target",
        7,
        (
            Route("hash-stock", None),
            Route("hash-mid", 9, (Route("hash-leaf", None),)),
        ),
    )
    match = keyed_match(
        {
            ("target molecule", 7): (
                (("wrong molecule", "wrong-hash"),),
                (
                    ("stock molecule", "hash-stock"),
                    ("middle molecule", "hash-mid"),
                ),
            ),
            ("middle molecule", 9): ((("leaf molecule", "hash-leaf"),),),
        }
    )

    assert template_replay_routes(
        reference,
        "target molecule",
        match=match,
        stack_mols=object_mol_array,
        output_key=str.upper,
    ) == (
        Route(
            "TARGET MOLECULE",
            7,
            (
                Route("STOCK MOLECULE", None),
                Route("MIDDLE MOLECULE", 9, (Route("LEAF MOLECULE", None),)),
            ),
        ),
    )


def test_template_replay_requires_the_reference_precursor_multiset() -> None:
    reference = Route("target", 3, (Route("same", None), Route("same", None)))
    match = keyed_match({("T", 3): ((("A", "same"), ("B", "other")),)})

    assert not template_replay_routes(
        reference,
        "T",
        match=match,
        stack_mols=object_mol_array,
        output_key=str,
    )


def test_template_replay_rejects_nonpositive_cap() -> None:
    with pytest.raises(ValueError, match="cap"):
        template_replay_routes(
            Route("target", None),
            "T",
            match=keyed_match({}),
            stack_mols=object_mol_array,
            output_key=str,
            cap=0,
        )


def test_step_precursor_names_names_what_each_rule_would_make() -> None:
    """One matcher call per step, each slot in the batch carrying its own rule.

    What a report needs and a replayed route cannot give it: replay reproduces the route that was
    *chosen*, so the alternatives beside it are molecules nothing ever named.
    """
    match = keyed_match(
        {
            ("product", 1): ((("alcohol", "k-alcohol"), ("chloride", "k-chloride")),),
            ("product", 2): ((("amine", "k-amine"),),),
        }
    )

    names = step_precursor_names(
        "product",
        [1, 2],
        match=match,
        stack_mols=object_mol_array,
        output_key=lambda molecule: f"{molecule}!",
    )

    assert names == {
        "k-alcohol": "alcohol!",
        "k-chloride": "chloride!",
        "k-amine": "amine!",
    }


def test_step_precursor_names_asks_nothing_when_no_rule_applies() -> None:
    """A molecule with no disconnection in the graph must not cost a matcher call."""
    calls: list[Any] = []

    def match(mols: Any, pairs: Any) -> Any:
        calls.append(pairs)
        raise AssertionError("no rules, no call")

    assert (
        step_precursor_names(
            "product",
            (),
            match=match,
            stack_mols=object_mol_array,
            output_key=str,
        )
        == {}
    )
    assert calls == []


def test_a_name_is_keyed_by_the_search_key_not_by_position() -> None:
    """So a name can only ever land on the molecule the search knew by that key.

    Two rules reaching the same precursor agree about it, and the mapping says so once rather than
    twice -- which is also why a report can look an alternative up by key alone.
    """
    match = keyed_match(
        {
            ("product", 1): ((("shared", "k-shared"), ("left", "k-left")),),
            ("product", 2): ((("shared", "k-shared"), ("right", "k-right")),),
        }
    )

    names = step_precursor_names(
        "product",
        [1, 2],
        match=match,
        stack_mols=object_mol_array,
        output_key=str,
    )

    assert names == {"k-shared": "shared", "k-left": "left", "k-right": "right"}


def test_a_molecule_the_keyer_cannot_express_costs_one_name_not_the_run() -> None:
    """A report is an add-on: it may lose a name, and it may not take down the run.

    This names every disconnection of every molecule on a reported route, rules the search would
    never take included, so it reaches products no replay had to stand behind -- and canonical
    SMILES goes through RDKit, which refuses to kekulise some of them. One such molecule ended a
    twenty-target sweep at target 8 before this.
    """
    match = keyed_match(
        {
            ("product", 1): ((("fine", "k-fine"), ("unkekulizable", "k-bad")),),
            ("product", 2): ((("also-fine", "k-also"),),),
        }
    )

    def output_key(molecule: str) -> str:
        if molecule == "unkekulizable":
            raise ValueError("Can't kekulize mol")
        return molecule

    names = step_precursor_names(
        "product",
        [1, 2],
        match=match,
        stack_mols=object_mol_array,
        output_key=output_key,
    )

    assert names == {"k-fine": "fine", "k-also": "also-fine"}
    assert "k-bad" not in names, "and the report simply shows that one as its key"


def rows_for(
    applications: list[tuple[str, ...]],
) -> tuple[list[str], list[str], Any]:
    """One frontier product, one rule, one application per precursor tuple given."""
    precursors: list[str] = []
    hits: list[tuple[int, int, int, int]] = []
    for app_idx, application in enumerate(applications):
        for molecule in application:
            hits.append((0, 0, app_idx, len(precursors)))
            precursors.append(molecule)
    return precursors, precursors, hit_array(hits)


def test_an_exact_comparator_refuses_an_application_that_names_its_reagent() -> None:
    """The default reading: does the library draw the edge in the reference's own vocabulary."""
    _mols, keys, hits = rows_for([("amine", "hydride")])
    kept = matching_application_rows(
        ["amide"], keys, hits, {"amide": frozenset({frozenset({"amine"})})}
    )
    assert not len(kept)


def test_an_ignored_reagent_does_not_stop_the_edge_from_matching() -> None:
    """A recorded route names substrates; a curated template increasingly names the hydride too.

    That difference is not a disconnection, so `ignored` drops it from the comparison -- and the
    *whole* application survives, both rows, because the reagent is still a precursor the graph
    has to carry.
    """
    _mols, keys, hits = rows_for([("amine", "hydride")])
    kept = matching_application_rows(
        ["amide"],
        keys,
        hits,
        {"amide": frozenset({frozenset({"amine"})})},
        ignored={"hydride"},
    )
    assert kept.tolist() == [0, 1]


def test_a_reagent_the_reference_names_and_the_rule_does_not_also_matches() -> None:
    """Stripping is symmetric, so neither side has to be the one that spells the reagent out."""
    _mols, keys, hits = rows_for([("amine",)])
    kept = matching_application_rows(
        ["amide"],
        keys,
        hits,
        {"amide": frozenset({frozenset({"amine", "hydride"})})},
        ignored={"hydride"},
    )
    assert kept.tolist() == [0]


def test_ignoring_reagents_does_not_admit_a_different_substrate() -> None:
    """The loosening is bounded: only the named reagents go, never a substrate."""
    _mols, keys, hits = rows_for([("aniline", "hydride")])
    kept = matching_application_rows(
        ["amide"],
        keys,
        hits,
        {"amide": frozenset({frozenset({"amine"})})},
        ignored={"hydride"},
    )
    assert not len(kept)
