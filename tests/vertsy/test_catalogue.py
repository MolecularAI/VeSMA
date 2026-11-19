"""The catalogue monoid: one composition law, two strengths, and the laws that say so."""

from __future__ import annotations

import pytest

from .catalogue import NOTHING, Catalogue, Catalogues, merge, notes_for

CURATED = Catalogue(
    "curated.smi",
    keys=frozenset({"acid", "amine"}),
    notes={"acid": ("curated.smi: acetic acid, Sigma A6283",)},
)
BULK = Catalogue("bulk.smi", keys=frozenset({"amine", "ether"}))


def test_a_catalogue_is_a_mapping_and_membership_is_derived_from_it() -> None:
    """`contains = nonempty . notes`, which is why merge is written once and not twice."""
    assert CURATED["acid"] == ("curated.smi: acetic acid, Sigma A6283",)
    # an unannotated row still answers: the file name is all it has to say
    assert CURATED["amine"] == ("curated.smi",)
    assert "amine" in CURATED
    assert "ether" not in CURATED
    with pytest.raises(KeyError):
        CURATED["ether"]


def test_merging_concatenates_notes_rather_than_picking_one() -> None:
    """The point of the composite: three suppliers carrying a molecule should say so three times.

    A merged index cannot, because `stock_slices` deduplicates before anything is prepared and so
    only ever knows which file listed a molecule first.
    """
    both = merge(CURATED, BULK)

    assert both["amine"] == ("curated.smi", "bulk.smi")
    assert both["acid"] == ("curated.smi: acetic acid, Sigma A6283",)
    assert both["ether"] == ("bulk.smi",)
    assert set(both) == {"acid", "amine", "ether"}
    assert len(both) == 3


def test_merge_is_a_monoid() -> None:
    """Associative, with an identity, and flat -- so assembly order cannot change the value."""
    assert merge(NOTHING, CURATED).parts == (CURATED,)
    assert merge(CURATED, NOTHING).parts == (CURATED,)

    left = merge(merge(CURATED, BULK), NOTHING)
    right = merge(CURATED, merge(BULK, NOTHING))

    # equal values, not merely equivalent behaviour: nesting is flattened away
    assert left == right
    assert left.parts == (CURATED, BULK)


def test_the_empty_catalogue_buys_nothing() -> None:
    assert "acid" not in NOTHING
    assert not len(NOTHING)
    with pytest.raises(KeyError):
        NOTHING["acid"]


def test_notes_for_skips_what_is_not_stocked() -> None:
    """A leaf nothing carries is absent rather than reported with an empty list."""
    found = notes_for(
        merge(CURATED, BULK), {"CC(=O)O": "acid", "CCO": "ether", "c1ccccc1": "arene"}
    )

    assert found == {
        "CC(=O)O": ["curated.smi: acetic acid, Sigma A6283"],
        "CCO": ["bulk.smi"],
    }


def test_a_bulk_catalogue_costs_nothing_per_key_to_annotate() -> None:
    """The reason `keys` and `notes` are separate fields rather than one mapping.

    A downloaded catalogue is a million bare SMILES whose name is the whole of what it can tell
    you. Mapping every key to that one string measured 212 bytes an entry, which is ~394 MB to
    store a fact membership already implies.
    """
    bulk = Catalogue("huge.smi", keys=frozenset(range(1000)))

    assert not bulk.notes
    assert bulk[500] == ("huge.smi",)
    assert len(bulk) == 1000


def test_catalogues_holds_its_parts_rather_than_copying_them_together() -> None:
    """Merging is a tuple, so the composite cannot go stale against what it was built from."""
    both = merge(CURATED, BULK)

    assert isinstance(both, Catalogues)
    assert both.parts == (CURATED, BULK)
    assert both.parts[0] is CURATED
