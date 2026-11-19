"""Why the model refused to build a molecule.

Rejection is an ordinary outcome here, not an error condition: a template applied to a
molecule may describe a state the chemistry does not allow, and saying so precisely is
more useful than either guessing a repair or returning nothing. This module declares the
closed set of things that can be said, so every refusal in the codebase names one of them
and a run can be summarised without parsing messages.

The set is closed and ordered. An application reports exactly one reason -- the first that
applies, reading down -- so counts across a corpus partition the rejected applications
rather than overlapping. The order runs from the most basic question to the most derived:
whether the atom is one we know, then whether its state is one we declare, then its
hydrogen, then the circuits it belongs to, then what the template asserted about the
result, and last the size limits of the machine rather than of the chemistry.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = [
    "ChemistryRejection",
    "RejectionCounts",
    "RejectionReason",
    "no_rejections",
]


class RejectionReason(IntEnum):
    """The closed set of reasons one template application produced no precursors."""

    UNSUPPORTED_ELEMENT = 0
    UNDECLARED_CONFIGURATION = 1
    NUCLEAR_CONSERVATION_VIOLATED = 2
    CONFIGURATION_INCOMPATIBLE = 3
    CONFIGURATION_AMBIGUOUS = 4
    HYDROGEN_COMPLETION_NEGATIVE = 5
    HYDROGEN_COMPLETION_INCOMPATIBLE = 6
    HYDROGEN_COMPLETION_AMBIGUOUS = 7
    HYDROGEN_OMITTED_WITH_CHARGE_CHANGE = 8
    DELOCALIZED_EDGE_NOT_CYCLIC = 9
    DELOCALIZED_SYSTEM_INVALID = 10
    REQUIRES_DELOCALIZED_ANNOTATION = 11
    # Reserved, no longer raised. An RHS is a rewrite program, not an assertion set, so a
    # derived property that disagrees with the result is counted as an annotation
    # mismatch and the application is returned -- see ``reaction.RHS_ANNOTATIONS``. The
    # member stays so the counters keep their positions and runs recorded before the
    # change remain comparable with runs recorded after it.
    RHS_POSTCONDITION_FAILED = 12
    PRECURSOR_CAPACITY_EXCEEDED = 13
    UNSUPPORTED_TEMPLATE = 14


class ChemistryRejection(ValueError):
    """A refusal that names its reason.

    Carrying the reason on the exception is what lets a batch caller count refusals by
    kind without inspecting messages, and lets a single-application caller be told what
    was wrong. The subclasses exist to be caught individually where that reads better;
    the reason is what gets counted.
    """

    reason: RejectionReason = RejectionReason.UNSUPPORTED_TEMPLATE

    def __init__(self, detail: str, *, reason: RejectionReason | None = None) -> None:
        super().__init__(detail)
        if reason is not None:
            self.reason = reason


# One counter per reason. A tuple rather than a mapping so that merging is elementwise
# addition and the layout is the same array of counters a device kernel would keep.
RejectionCounts = tuple[int, ...]


def no_rejections() -> RejectionCounts:
    return (0,) * len(RejectionReason)


def counted(reason: RejectionReason) -> RejectionCounts:
    """Counts recording one application rejected for this reason."""
    return tuple(int(index == reason) for index in range(len(RejectionReason)))


def merged(left: RejectionCounts, right: RejectionCounts) -> RejectionCounts:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def by_name(counts: RejectionCounts) -> dict[str, int]:
    """The non-zero counts, keyed by reason name, for reporting."""
    return {
        reason.name.lower(): counts[reason]
        for reason in RejectionReason
        if counts[reason]
    }
