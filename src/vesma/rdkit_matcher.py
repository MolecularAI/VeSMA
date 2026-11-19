from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from .matcher import Hit, HitArray, Pair, concat_hits, hit_array
from .parallel import (
    ResidentForkPool,
    chunked,
    effective_n_jobs,
    parallel_chunks,
    resolve_n_jobs,
)

if TYPE_CHECKING:
    from rdkit import Chem
    from rdkit.Chem.rdChemReactions import ChemicalReaction

__all__ = [
    "RdkitDiagnostics",
    "RdkitFilterCatalogReactionMatcher",
    "RdkitReactionMatcher",
    "RdkitReactionWork",
    "collapse_by_key",
    "rdkit_reactions_from_smarts",
    "split_outcome",
]


@dataclass(frozen=True)
class RdkitDiagnostics:
    """RDKit outcomes, including applications it built but could not sanitize."""

    rules_total: int
    pairs_evaluated: int
    outcomes_total: int
    outcomes_unsanitizable: int
    # (mol_idx, rule_id, raw application index, RDKit's message) per discard.
    discarded: tuple[tuple[int, int, int, str], ...] = ()


def merged_rdkit_diagnostics(
    parts: Sequence[RdkitDiagnostics],
) -> RdkitDiagnostics:
    return RdkitDiagnostics(
        rules_total=parts[0].rules_total if parts else 0,
        pairs_evaluated=sum(part.pairs_evaluated for part in parts),
        outcomes_total=sum(part.outcomes_total for part in parts),
        outcomes_unsanitizable=sum(part.outcomes_unsanitizable for part in parts),
        discarded=tuple(row for part in parts for row in part.discarded),
    )


class RdkitReactionMatcher:
    """Reaction matcher over `rdkit.Chem.Mol` molecules.

    Compiles reaction SMARTS to RDKit reactions; `match` runs `RunReactants` for each
    `(mol_idx, rule_id)` pair and splits each outcome into connected-component
    precursor molecules. No deduplication — identical steps collapse downstream in
    key space. Pairs with `rdkit_smiles_keyer` (K=str).

    `n_jobs` is a constructor argument rather than a `match` argument because the workers hold
    the compiled rules: they are part of receiving the rule set, not part of a call. They are
    forked on first use, inherit the rules by copy-on-write, and stay resident and idle between
    calls — so `match` pays for its own molecules and nothing else. `RunReactants` holds the
    GIL, which is why these are processes: a thread pool measured *slower* than serial (676
    against 851 pairs/s on PaRoutes n1 against raw68k).

    Call `close()` when done with a matcher that owns workers, or use it as a context manager.
    """

    def __init__(
        self,
        reactions: Sequence[ChemicalReaction],
        *,
        n_jobs: int | None = None,
        key: Callable[[Sequence[Chem.Mol]], Sequence[Hashable]] | None = None,
    ) -> None:
        self.reactions = tuple(reactions)
        self.key = key
        self.last_diagnostics: RdkitDiagnostics | None = None
        self.workers = ResidentForkPool(
            initializer=init_rdkit_reaction_globals,
            initargs=(self.reactions, key),
            n_jobs=n_jobs,
        )

    def close(self) -> None:
        self.workers.close()

    def __enter__(self) -> RdkitReactionMatcher:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def match(
        self,
        mols: Sequence[Chem.Mol],
        pairs: Sequence[Pair] | None = None,
    ) -> tuple[np.ndarray, HitArray]:
        precursors, _, hits = self.match_keyed(mols, pairs)
        return precursors, hits

    def match_batches(
        self,
        mols: Sequence[Chem.Mol],
        pairs: Sequence[Pair] | None = None,
        *,
        pairs_per_batch: int = 4096,
    ) -> Iterator[tuple[np.ndarray, HitArray]]:
        """Sanitized precursor batches, retaining at most one worker window.

        With the default ``key=None``, this computes neither SMILES nor identity keys.
        Each batch owns its molecules and its hit rows use batch-local precursor indices.
        """
        if pairs_per_batch <= 0:
            raise ValueError("pairs_per_batch must be positive")
        if pairs is None:
            pairs = [
                (index, rule)
                for index in range(len(mols))
                for rule in range(len(self.reactions))
            ]
        frontier = tuple(mols)
        diagnostics = RdkitDiagnostics(len(self.reactions), 0, 0, 0)
        try:
            for window in chunked(
                chunked(pairs, pairs_per_batch), self.workers.n_workers
            ):
                results = self.workers.map(
                    match_rdkit_reaction_chunk,
                    tuple(RdkitReactionWork(frontier, part) for part in window),
                )
                for precursors, _, hits, partial in results:
                    diagnostics = merged_rdkit_diagnostics((diagnostics, partial))
                    if len(hits):
                        yield precursors, hits
                    del precursors, hits, partial
                del results
        finally:
            self.last_diagnostics = diagnostics

    def match_keyed(
        self,
        mols: Sequence[Chem.Mol],
        pairs: Sequence[Pair] | None = None,
    ) -> tuple[np.ndarray, Sequence[Hashable], HitArray]:
        """`match`, plus the identity key of each precursor row — computed in the workers.

        Keying is where an expansion's time goes once the reactions themselves are parallel:
        `MolToSmiles` over a batch's precursors measured **1,235 ms** against 850 ms for the
        whole parallel match, and it ran serially in the parent because `keyed_match` composes
        match-then-key there. Doing it in the worker parallelises it and shrinks what comes
        home, since a worker that knows a key can drop duplicates before anything is pickled —
        on PaRoutes n1 against raw68k, 6,026 precursor rows carry only 1,383 distinct keys.

        Returns an empty key sequence when no `key` was given, which is the plain `match` path.
        """
        if pairs is None:
            pairs = [
                (mol_idx, rule_id)
                for mol_idx in range(len(mols))
                for rule_id in range(len(self.reactions))
            ]

        # One chunk per worker would leave the pool idling on whichever chunk drew the slowest
        # rules; several apiece lets the queue even that out.
        chunks = parallel_chunks(pairs, n_jobs=self.workers.n_workers)
        frontier = tuple(mols)
        results = self.workers.map(
            match_rdkit_reaction_chunk,
            tuple(RdkitReactionWork(frontier, chunk) for chunk in chunks),
        )
        precursors = concat_object_arrays(tuple(part for part, _, _, _ in results))
        hits = concat_hits(
            tuple(hit for _, _, hit, _ in results),
            tuple(len(part) for part, _, _, _ in results),
        )
        keys: list[Hashable] = []
        for _, part_keys, _, _ in results:
            keys.extend(part_keys)
        self.last_diagnostics = replace(
            merged_rdkit_diagnostics(
                tuple(diagnostics for _, _, _, diagnostics in results)
            ),
            rules_total=len(self.reactions),
        )
        return precursors, keys, hits


class RdkitReactionWork(NamedTuple):
    """One worker's share of a `match` call: the frontier, and the pairs to run against it.

    The rules are absent on purpose — they are already in the worker, inherited once. Only this
    travels per call, and the frontier is the whole batch rather than a slice so a pair's
    `mol_idx` still indexes it.
    """

    mols: tuple[Any, ...]
    pairs: tuple[Pair, ...]


_G_RDKIT_REACTIONS: tuple[Any, ...] | None = None
_G_RDKIT_KEY: Callable[[Sequence[Any]], Sequence[Hashable]] | None = None


def init_rdkit_reaction_globals(
    reactions: tuple[Any, ...],
    key: Callable[[Sequence[Any]], Sequence[Hashable]] | None = None,
) -> None:
    global _G_RDKIT_REACTIONS, _G_RDKIT_KEY
    _G_RDKIT_REACTIONS = reactions
    _G_RDKIT_KEY = key


def match_rdkit_reaction_chunk(
    work: RdkitReactionWork,
) -> tuple[np.ndarray, Sequence[Hashable], HitArray, RdkitDiagnostics]:
    from rdkit import Chem

    if _G_RDKIT_REACTIONS is None:
        raise RuntimeError(
            "rdkit reaction worker globals unset; run init_rdkit_reaction_globals first."
        )

    precursors: list[Any] = []
    hits: list[Hit] = []
    outcomes_total = 0
    outcomes_unsanitizable = 0
    discarded: list[tuple[int, int, int, str]] = []
    for mol_idx, rule_id in work.pairs:
        reaction = _G_RDKIT_REACTIONS[rule_id]
        for app_idx, outcome in enumerate(reaction.RunReactants((work.mols[mol_idx],))):
            outcomes_total += 1
            fragments, error = split_outcome_or_error(outcome, Chem)
            if fragments is None:
                outcomes_unsanitizable += 1
                discarded.append((mol_idx, rule_id, app_idx, error))
                continue
            for fragment in fragments:
                hits.append((mol_idx, rule_id, app_idx, len(precursors)))
                precursors.append(fragment)

    diagnostics = RdkitDiagnostics(
        rules_total=len(_G_RDKIT_REACTIONS),
        pairs_evaluated=len(work.pairs),
        outcomes_total=outcomes_total,
        outcomes_unsanitizable=outcomes_unsanitizable,
        discarded=tuple(discarded),
    )
    if _G_RDKIT_KEY is None:
        return np.array(precursors, dtype=object), (), hit_array(hits), diagnostics
    collapsed, keys, collapsed_hits = collapse_by_key(
        precursors, hits, _G_RDKIT_KEY(precursors)
    )
    return collapsed, keys, collapsed_hits, diagnostics


def collapse_by_key(
    precursors: Sequence[Any],
    hits: Sequence[Hit],
    keys: Sequence[Hashable],
) -> tuple[np.ndarray, Sequence[Hashable], HitArray]:
    """Keep one precursor row per distinct key, repointing every hit at the row that survived.

    The 4-column `HitArray` is what permits this: a hit names its precursor by index rather
    than by position, so rows may be shared. Rule libraries carry many near-synonymous
    templates, so the duplication is large — 6,026 rows collapse to 1,383 on a five-molecule
    PaRoutes frontier — and collapsing here rather than in the caller means the duplicates are
    never pickled out of the worker at all.

    Only within this chunk: a key seen by two workers still crosses twice, which the contract
    allows ("not at all, partially within a batch, or fully") and which downstream key-space
    interning collapses anyway.
    """
    kept: dict[Hashable, int] = {}
    unique: list[Any] = []
    unique_keys: list[Hashable] = []
    remap: list[int] = []
    for index, key in enumerate(keys):
        row = kept.get(key)
        if row is None:
            row = kept[key] = len(unique)
            unique.append(precursors[index])
            unique_keys.append(key)
        remap.append(row)

    collapsed = [
        (mol_idx, rule_id, app_idx, remap[precursor_idx])
        for mol_idx, rule_id, app_idx, precursor_idx in hits
    ]
    return np.array(unique, dtype=object), unique_keys, hit_array(collapsed)


def concat_object_arrays(parts: Sequence[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.array([], dtype=object)
    if not any(len(part) for part in parts):
        return np.array([], dtype=object)
    return np.concatenate(parts).astype(object, copy=False)


def split_outcome(
    outcome: Sequence[Chem.Mol],
    chem: Any,
) -> list[Chem.Mol] | None:
    """Split one RunReactants outcome into sanitized connected-component fragments.

    Returns None if any product cannot be sanitized, so the caller can drop the
    whole outcome rather than emit a partial precursor set.
    """
    return split_outcome_or_error(outcome, chem)[0]


def split_outcome_or_error(
    outcome: Sequence[Chem.Mol],
    chem: Any,
) -> tuple[list[Chem.Mol] | None, str]:
    """As :func:`split_outcome`, together with the sanitization failure message."""
    fragments: list[Any] = []
    for product in outcome:
        try:
            for fragment in chem.GetMolFrags(product, asMols=True, sanitizeFrags=True):
                fragments.append(fragment)
        except Exception as error:
            return None, f"{type(error).__name__}: {error}"
    return fragments, ""


def rdkit_reactions_from_smarts(
    smarts: Sequence[str],
) -> Sequence[ChemicalReaction]:
    from rdkit.Chem import rdChemReactions

    reactions: list[ChemicalReaction] = []
    for text in smarts:
        try:
            reaction = rdChemReactions.ReactionFromSmarts(text)
        except Exception as exc:
            raise ValueError(f"RDKit failed to parse reaction SMARTS: {text}") from exc
        if reaction is None:
            raise ValueError(f"RDKit failed to parse reaction SMARTS: {text}")
        reactions.append(reaction)
    return tuple(reactions)


class RdkitFilterCatalogReactionMatcher:
    """Reaction matcher that prefilters `(mol, rule)` pairs through an RDKit FilterCatalog.

    Running `RunReactants` for every molecule against every template dominates matching, and
    most of that work is thrown away because the template's left-hand side does not occur in
    the molecule at all. A FilterCatalog answers "which left-hand sides occur here?" as one
    substructure screen, so only pairs that can produce something reach the real matcher.
    Same hits as `RdkitReactionMatcher`, fewer reaction runs.
    """

    def __init__(
        self,
        reactions: Sequence[ChemicalReaction],
        lhs_smarts: Sequence[str],
        *,
        n_jobs: int | None = None,
        key: Callable[[Sequence[Chem.Mol]], Sequence[Hashable]] | None = None,
    ) -> None:
        from rdkit.Chem import FilterCatalog, MolFromSmarts

        if len(reactions) != len(lhs_smarts):
            raise ValueError(
                "reactions and lhs_smarts must be parallel: "
                f"{len(reactions)} != {len(lhs_smarts)}"
            )
        self.matcher = RdkitReactionMatcher(reactions, n_jobs=n_jobs, key=key)
        # The catalog threads inside C++ rather than through a pool, so it needs the width as a
        # number at call time rather than a set of workers.
        self.screen_threads = effective_n_jobs(resolve_n_jobs(n_jobs))

        params = FilterCatalog.FilterCatalogParams()
        self.catalog = FilterCatalog.FilterCatalog(params)
        for idx, smarts in enumerate(lhs_smarts):
            query = MolFromSmarts(smarts)
            if query is None:
                raise ValueError(f"RDKit FilterCatalog rejected query {idx}.")
            smarts_matcher = FilterCatalog.SmartsMatcher(f"{idx}", query)
            if not smarts_matcher.IsValid():
                raise ValueError(
                    f"RDKit FilterCatalog rejected query {idx} (invalid matcher)."
                )
            self.catalog.AddEntry(
                FilterCatalog.FilterCatalogEntry(f"{idx}", smarts_matcher)
            )

    def close(self) -> None:
        self.matcher.close()

    def __enter__(self) -> RdkitFilterCatalogReactionMatcher:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def match(
        self,
        mols: Sequence[Chem.Mol],
        pairs: Sequence[Pair] | None = None,
    ) -> tuple[np.ndarray, HitArray]:
        precursors, _, hits = self.match_keyed(mols, pairs)
        return precursors, hits

    def match_keyed(
        self,
        mols: Sequence[Chem.Mol],
        pairs: Sequence[Pair] | None = None,
    ) -> tuple[np.ndarray, Sequence[Hashable], HitArray]:
        candidates = self.screen(mols)

        if pairs is not None:
            candidates.intersection_update(pairs)

        # Sorted, not just listed: set iteration order varies between runs, and pair order
        # decides the app index each hit is given, which routes are keyed by.
        return self.matcher.match_keyed(mols, pairs=sorted(candidates))

    def screen(self, mols: Sequence[Chem.Mol]) -> set[Pair]:
        """The `(mol, rule)` pairs whose left-hand side occurs in the molecule.

        `RunFilterCatalog` rather than a Python loop over `catalog.GetMatches(mol)`: it is the
        same C++ screen, but it runs the whole batch and threads it over molecules, and it
        skips constructing a `FilterCatalogEntry` per *test* rather than per hit. On PaRoutes
        n1 against raw68k the two agree pair-for-pair at every batch size, and the batched form
        is 3.9x faster on a five-molecule frontier (1.02 s -> 0.26 s, 8 cores).

        Threading is over *molecules*, so a one-molecule batch gets none of it: measured
        0.30 M pair-tests/s at one molecule against 1.30 M/s at five. A caller that expands a
        narrow frontier pays roughly four times as much per molecule as one that expands a
        wide one, which is a reason to batch expansions that has nothing to do with search
        quality.
        """
        from rdkit import Chem
        from rdkit.Chem import FilterCatalog

        # RunFilterCatalog takes SMILES, not Mol. The round trip is ~50 us per molecule
        # against a ~50 ms screen, so it stays under a tenth of a percent and keeps this a
        # drop-in for the Mol-based `Matcher` protocol.
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        # present at runtime, absent from the rdkit stubs
        rows = FilterCatalog.RunFilterCatalog(  # ty: ignore[unresolved-attribute]
            self.catalog, smiles, self.screen_threads
        )

        candidates: set[Pair] = set()
        for mol_idx, row in enumerate(rows):
            for entry in row:
                # An entry's description is the only id RDKit carries back out of a catalog,
                # so the rule index round-trips through the string it was registered under.
                description = entry.GetDescription()
                if description == "Bad smiles":
                    # RunFilterCatalog reports a string it cannot read this way instead of
                    # raising, so this is its contract rather than a case reachable from any
                    # molecule above: `MolToSmiles` raises first on anything it will not write.
                    # It stays because screening a molecule out silently would lose every route
                    # through it with no trace.
                    raise ValueError(
                        f"RDKit could not parse molecule {mol_idx} for the filter catalog: "
                        f"{smiles[mol_idx]!r}"
                    )
                candidates.add((mol_idx, int(description)))
        return candidates
