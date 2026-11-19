from __future__ import annotations

from vesma.matcher import Hit, HitArray, K, M

# The domain vocabulary `M` (molecule repr) and `K` (key / identity token) is defined once in
# `vesma.matcher` and imported here so vertsy shares the exact same type variables. This module
# re-exports them (plus the matcher's `Hit`/`HitArray`) as vertsy's single vocabulary hub.

# Rule ids are concrete ints. `vesma.matcher` documents why rule-id is not (yet) shared vocabulary.
RuleId = int
# One materialised step in key space: (rule_id, precursor keys). The product and
# precursor keys share K throughout an Expansions[K].
Step = tuple[RuleId, frozenset[K]]


__all__ = [
    "Hit",
    "HitArray",
    "K",
    "M",
    "RuleId",
    "Step",
]
