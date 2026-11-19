from __future__ import annotations

from dataclasses import dataclass

from vertsy import solved_targets
from vertsy.expansions import expansions_of


@dataclass(frozen=True)
class Token:
    name: str


def token(name: str) -> Token:
    return Token(name)


def test_solved_targets_closes_over_expansion_steps() -> None:
    target = token("target")
    mid = token("mid")
    stock = token("stock")
    dead = token("dead")
    expansions = expansions_of(
        {
            target: frozenset({(0, frozenset({mid})), (1, frozenset({dead}))}),
            mid: frozenset({(2, frozenset({stock}))}),
            dead: frozenset(),
        }
    )

    solved = solved_targets({target, mid, dead}, expansions, {stock})

    assert solved == {mid, target}
