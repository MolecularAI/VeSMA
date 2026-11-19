"""Where a molecule can be bought, as a monoid.

A search needs one question answered about stock -- ``key in stock`` -- and should pay for
nothing else. A report needs a different one -- *which catalogues is this in, and what do they
say* -- and can afford more, because by then the search has finished. These are the same
structure at two strengths, which is what lets one composition law serve both.

The algebra
-----------
A catalogue is a function from a molecule key to what is known about it:

* as a **Collection**, ``Key -> Bool``, merged with ``or``;
* as a **Mapping**, ``Key -> tuple[Note, ...]``, merged by concatenation.

Both are monoids: merging is associative and :data:`NOTHING` is the identity. They are not two
algebras but one, because ``contains = nonempty . notes`` is a monoid homomorphism from
``(tuple[Note, ...], +, ())`` onto ``(Bool, or, False)``. So membership is *derived* from the
annotation rather than implemented beside it, and :func:`merge` is written once.

**Not a semiring, and the reason is worth keeping.** A semiring wants two operations with a
distributive law, and a catalogue has one natural operation: union. The only candidate for the
second is intersection -- "carried by both suppliers" -- and while sets under union and
intersection do form a distributive lattice, the enriched version does not lift: concatenating
notes does not distribute over intersection. Naming this a semiring would promise a second
operation with nothing for it to do, and this repository already spends that vocabulary on the
search fold, where the two operations are real.

Memory
------
:class:`Catalogue` keeps ``keys`` and ``notes`` apart on purpose. A downloaded catalogue is a
million bare SMILES: its name is the whole of what it can tell you, so mapping every key to that
one string is a million entries -- measured at 212 bytes each -- to store a fact already implied
by membership. A curated file is a few hundred rows that each say something different, and those
go in ``notes``. ``__getitem__`` reads the second and falls back to the first.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import chain
from typing import Any

__all__ = ["NOTHING", "Catalogue", "Catalogues", "merge"]


@dataclass(frozen=True)
class Catalogue(Mapping[Any, "tuple[str, ...]"]):
    """One stock file: the keys it carries, and what it says about the ones it annotates."""

    name: str
    keys: Collection[Any] = frozenset()
    notes: Mapping[Any, tuple[str, ...]] = field(default_factory=dict)
    """Only the rows that say something. A file of bare SMILES leaves this empty."""

    def __getitem__(self, key: Any) -> tuple[str, ...]:
        found = self.notes.get(key)
        if found:
            return found
        if key in self.keys:
            return (self.name,)
        raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        return key in self.keys or key in self.notes

    def __iter__(self) -> Iterator[Any]:
        seen = set()
        for key in chain(self.keys, self.notes):
            if key not in seen:
                seen.add(key)
                yield key

    def __len__(self) -> int:
        return sum(1 for _ in self)


@dataclass(frozen=True)
class Catalogues(Mapping[Any, "tuple[str, ...]"]):
    """Several catalogues as one, holding them rather than copying them together.

    The parts keep their own keys, so merging costs a tuple. That is also what makes the answer
    complete: a molecule three suppliers carry is reported three times, where a merged index
    deduplicated before anything was prepared and could only ever name the first.
    """

    parts: tuple[Catalogue, ...] = ()

    def __getitem__(self, key: Any) -> tuple[str, ...]:
        found = tuple(
            chain.from_iterable(part[key] for part in self.parts if key in part)
        )
        if not found:
            raise KeyError(key)
        return found

    def __contains__(self, key: Any) -> bool:
        return any(key in part for part in self.parts)

    def __iter__(self) -> Iterator[Any]:
        seen = set()
        for part in self.parts:
            for key in part:
                if key not in seen:
                    seen.add(key)
                    yield key

    def __len__(self) -> int:
        return sum(1 for _ in self)


NOTHING = Catalogues()
"""The empty catalogue: buys nothing, knows nothing, and is the identity of :func:`merge`."""


def merge(*catalogues: Catalogue | Catalogues) -> Catalogues:
    """Associative, with :data:`NOTHING` as its identity. Flattens, so nesting cannot change it.

    ``merge(a, merge(b, c))`` and ``merge(merge(a, b), c)`` are the same value and not merely
    equivalent, which is the property that lets a caller assemble a run's stock in whatever order
    the configuration arrives in.
    """
    parts: list[Catalogue] = []
    for catalogue in catalogues:
        if isinstance(catalogue, Catalogues):
            parts.extend(catalogue.parts)
        else:
            parts.append(catalogue)
    return Catalogues(tuple(parts))


def notes_for(
    catalogues: Mapping[Any, Sequence[str]], keys: Mapping[str, Any]
) -> dict[str, list[str]]:
    """``{smiles: [note, ...]}`` for the molecules ``keys`` names, skipping what is not stocked.

    Takes the keying already done rather than doing it, so this module needs no chemistry.
    """
    found: dict[str, list[str]] = {}
    for smiles, key in keys.items():
        notes = catalogues.get(key)
        if notes:
            found[smiles] = list(notes)
    return found
