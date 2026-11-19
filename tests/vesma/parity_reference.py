"""A saved CPU reference, so device parity can be checked without CPU on the same box.

No machine has both CUDA and Metal, so each device is compared against CPU and never
against the other. That works, but it made the CPU leg something every machine recomputes
for every run -- and at the full corpus the CPU side is the expensive half. Computing it
once and comparing many times turns a twenty-minute check into a load and a diff.

What the reference holds is what agreement is defined as: the multiset of
``(molecule, rule, precursor key)`` outcomes, and the per-reason refusal histogram. Both
are small -- a few tens of megabytes over the whole corpus -- and both are exactly what a
device run produces, so the comparison is the same one either way.

What it deliberately does *not* hold is the capped-application count. That is a property
of the device that produced it, not of the chemistry, so it has to come from the run being
checked rather than from the reference.

A stale reference is the failure mode worth designing against: comparing a device against
chemistry it was not built from would report divergence that is really just drift. So the
artifact records the chemistry fingerprint, the corpus it covers and the processing it was
produced under, and loading refuses rather than warns when any of them disagree.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tests.datasets import chemistry_fingerprint

REFERENCE_FORMAT = 1


@dataclass(frozen=True)
class ParityReference:
    """The CPU outcome of one corpus, and the identity of the run that produced it."""

    size: int
    molecule_indices: tuple[int, ...]
    template_count: int
    chemistry_fingerprint: str
    processing: str
    #: The precursor atom cap the CPU leg was asked to respect, or `None` for uncapped. Part
    #: of the question, not of the answer, which is why it is recorded and checked rather than
    #: assumed.
    max_precursor_atoms: int | None
    precursor_count: int
    keyed: Counter
    refusals: dict[str, int]

    def provenance(self) -> str:
        return (
            f"size={self.size} templates={self.template_count} "
            f"chemistry={self.chemistry_fingerprint} processing={self.processing} "
            f"max_precursor_atoms={self.max_precursor_atoms}"
        )


def keyed_counter(hits: Any, keys: Sequence[Any]) -> Counter:
    """The ``(molecule, rule, key)`` multiset, built the same way on every backend."""
    return Counter(
        (int(hit[0]), int(hit[1]), key) for hit, key in zip(hits, keys, strict=True)
    )


def default_path(size: int, *, max_precursor_atoms: int | None) -> Path:
    """One file per question, not one per corpus.

    The cap is part of the question, not merely metadata. Both GPU parity gates now
    use the 64-atom reference; uncapped comparisons retain a separate cache file.
    """
    cap = "uncapped" if max_precursor_atoms is None else str(max_precursor_atoms)
    return Path("output/parity-reference") / f"cpu-{size}-{cap}.npz"


def build(
    size: int,
    *,
    processing: Any,
    processing_name: str,
    max_precursor_atoms: int | None,
    n_jobs: int | None,
) -> ParityReference:
    """Run CPU over the corpus and capture what agreement is defined as.

    Kept here rather than in either sweep so the two devices are compared against one
    reference produced one way, instead of two CPU runs that could quietly differ.

    `max_precursor_atoms` is not optional and is recorded, because it decides *which
    question* the reference answers. The devices refuse an application whose precursor
    exceeds their cap; a reference computed without that cap holds outcomes no device can
    produce, and every one of them would read as a device shortfall. So it joins the
    chemistry fingerprint and the processing mode in what `require_current` refuses on.
    """
    from tests.vesma.bench_matcher import load_match_corpus, load_positive_pairs
    from vesma.cpu_matcher import CpuReactionMatcher
    from vesma.rejection import RejectionReason

    corpus = load_match_corpus(size)
    pairs = load_positive_pairs(corpus)
    rules = [template.pattern for template in corpus.templates]
    matcher = CpuReactionMatcher(
        rules, max_precursor_atoms=max_precursor_atoms, processing=processing
    )
    precursors, hits = matcher.match(corpus.cmols, pairs, n_jobs=n_jobs)
    keys = parallel_keys(precursors, hits, n_jobs=n_jobs)
    refusals = Counter(
        RejectionReason(int(row[3])).name.lower()
        for row in matcher.last_rejected_applications
    )
    return ParityReference(
        size=size,
        molecule_indices=tuple(int(i) for i in corpus.source_molecule_indices),
        template_count=len(rules),
        chemistry_fingerprint=chemistry_fingerprint(),
        processing=processing_name,
        max_precursor_atoms=max_precursor_atoms,
        precursor_count=len(precursors),
        keyed=keyed_counter(hits, keys),
        refusals=dict(refusals),
    )


def _key_chunk(precursors: Sequence[Any]) -> list[Any]:
    from vesma.identity import short_ring_wl_molecule_id

    return [short_ring_wl_molecule_id(precursor) for precursor in precursors]


def parallel_keys(precursors: Any, hits: Any, *, n_jobs: int | None) -> list[Any]:
    """WL keys for every hit's precursor, spread across processes unless told not to.

    Keying dominates the CPU leg -- it walks rings and mixes 64-bit words per molecule in
    Python -- and it is perfectly independent per precursor. Forked workers inherit the
    precursors, so only the tokens come back.

    `n_jobs=1` is not a tuning choice but a correctness one for one caller: forking after
    CuPy has initialised CUDA is unsafe on Linux, so the CUDA bench must build its reference
    serially. The keys are identical either way -- this only decides who computes them --
    which is what lets both benches share one reference despite that.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    ordered = [precursors[int(hit[3])] for hit in hits]
    workers = 1 if n_jobs == 1 else max(1, (os.cpu_count() or 2) - 1)
    if len(ordered) < 4096 or workers == 1:
        return _key_chunk(ordered)
    size = (len(ordered) + workers - 1) // workers
    chunks = [ordered[i : i + size] for i in range(0, len(ordered), size)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [key for chunk in pool.map(_key_chunk, chunks) for key in chunk]


def load_or_build(
    size: int,
    *,
    processing: Any,
    processing_name: str,
    max_precursor_atoms: int | None,
    n_jobs: int | None,
) -> ParityReference:
    """The saved reference for this corpus, computing and saving it if absent.

    A cached reference is reused only when it answers the same question: same chemistry,
    same processing mode, same precursor cap. Any of those differing rebuilds rather than
    raises -- `require_current` is what raises, on the run's own facts, and a cache whose
    key moved on is a cache miss rather than an error.
    """
    path = default_path(size, max_precursor_atoms=max_precursor_atoms)
    if path.exists():
        reference = load(path)
        if (
            reference.chemistry_fingerprint == chemistry_fingerprint()
            and reference.processing == processing_name
            and reference.max_precursor_atoms == max_precursor_atoms
        ):
            return reference
    reference = build(
        size,
        processing=processing,
        processing_name=processing_name,
        max_precursor_atoms=max_precursor_atoms,
        n_jobs=n_jobs,
    )
    save(reference, path)
    return reference


def save(reference: ParityReference, path: Path) -> Path:
    """Write the reference as arrays plus a metadata header.

    Stored column-wise rather than as pickled tuples: the tokens are two 64-bit words, so
    the whole corpus is four narrow columns and loads without executing anything.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    triples = sorted(reference.keyed.elements())
    molecules = np.array([t[0] for t in triples], dtype=np.int32)
    rules = np.array([t[1] for t in triples], dtype=np.int32)
    lo = np.array([t[2][0] for t in triples], dtype=np.uint64)
    hi = np.array([t[2][1] for t in triples], dtype=np.uint64)
    header = json.dumps(
        {
            "format": REFERENCE_FORMAT,
            "size": reference.size,
            "molecule_indices": list(reference.molecule_indices),
            "template_count": reference.template_count,
            "chemistry_fingerprint": reference.chemistry_fingerprint,
            "processing": reference.processing,
            "max_precursor_atoms": reference.max_precursor_atoms,
            "precursor_count": reference.precursor_count,
            "refusals": reference.refusals,
        }
    )
    np.savez_compressed(
        path, header=np.array(header), molecule=molecules, rule=rules, lo=lo, hi=hi
    )
    return path


def load(path: Path) -> ParityReference:
    with np.load(path, allow_pickle=False) as data:
        header = json.loads(str(data["header"]))
        if header.get("format") != REFERENCE_FORMAT:
            raise ValueError(
                f"{path} is format {header.get('format')}, not {REFERENCE_FORMAT}."
            )
        keyed = Counter(
            zip(
                data["molecule"].tolist(),
                data["rule"].tolist(),
                zip(data["lo"].tolist(), data["hi"].tolist(), strict=True),
                strict=True,
            )
        )
    return ParityReference(
        size=header["size"],
        molecule_indices=tuple(header["molecule_indices"]),
        template_count=header["template_count"],
        chemistry_fingerprint=header["chemistry_fingerprint"],
        processing=header["processing"],
        max_precursor_atoms=header["max_precursor_atoms"],
        precursor_count=header["precursor_count"],
        keyed=keyed,
        refusals=dict(header["refusals"]),
    )


def require_current(
    reference: ParityReference,
    *,
    molecule_indices: Sequence[int],
    template_count: int,
    processing: str,
    max_precursor_atoms: int | None,
) -> None:
    """Refuse a reference that does not describe the run being checked.

    Each of these would otherwise show up as a chemical divergence that is really an
    accounting one, which is the single way a saved reference can be worse than no
    reference at all.
    """
    fingerprint = chemistry_fingerprint()
    if reference.chemistry_fingerprint != fingerprint:
        raise ValueError(
            f"Reference was built from chemistry {reference.chemistry_fingerprint}, "
            f"this tree is {fingerprint}. Rebuild it rather than comparing across a "
            f"chemistry change."
        )
    if tuple(reference.molecule_indices) != tuple(molecule_indices):
        raise ValueError(
            f"Reference covers {len(reference.molecule_indices)} molecules, this run "
            f"has {len(molecule_indices)}, or they are different molecules."
        )
    if reference.template_count != template_count:
        raise ValueError(
            f"Reference used {reference.template_count} templates, this run "
            f"{template_count}."
        )
    if reference.processing != processing:
        raise ValueError(
            f"Reference was produced under processing {reference.processing!r}, this run "
            f"asks for {processing!r}. The two CPU runs are not the same reference."
        )
    if reference.max_precursor_atoms != max_precursor_atoms:
        raise ValueError(
            f"Reference was built with a {reference.max_precursor_atoms}-atom precursor "
            f"cap, this run asks for {max_precursor_atoms}. An uncapped reference holds "
            f"outcomes no capped device can produce, and each would read as a shortfall."
        )


def compare(
    reference: ParityReference,
    *,
    backend: str,
    device_keyed: Counter,
    device_refusals: dict[str, int],
    capped: int,
    components_per_capped_application: int | None,
) -> str:
    """Assert device agreement with the reference, and return a line describing the run.

    Refusals are compared whatever the caps did: a capacity limit stops a precursor being
    emitted, it never invents or suppresses a chemical refusal.

    The two directions are not symmetric and are asserted separately. An outcome the
    device built and CPU did not is a molecule nobody asked for, which is the failure the
    model exists to prevent; an outcome CPU built and the device did not is a stated
    capacity limit, so it is bounded by what the caps can account for rather than waved
    through.

    Two bounds, and the weaker one is always applied. A capped application is a single
    (molecule, rule, application), so every outcome it loses carries that molecule and rule:
    the missing outcomes can touch at most `capped` distinct (molecule, rule) pairs, whatever
    each application held. That needs no device constant, so it holds on every backend.

    `components_per_capped_application` adds the tighter bound where a device limits components
    per application -- then the *count* of missing outcomes is bounded too. Metal passes its
    6-component cap; CUDA has no component cap, so it passes `None` and gets the spread bound
    alone, which the printed line records.

    `None` used to mean the shortfall went entirely unchecked, which made a single capped
    application switch the check off and let an arbitrary divergence through. Measured on an
    RTX 5080 at 100 molecules, where 8 capped applications lost 8 outcomes spanning 5
    (molecule, rule) pairs, the spread bound is satisfied with room and would still have caught
    a divergence scattered across more pairs than were capped.
    """
    # Messages hoisted so each assert fits on one line. Left inline they need wrapping, and
    # there black wraps the condition where ruff wraps the message -- a standoff neither can
    # win, on a repo whose [tool.ruff.format] note requires the two to agree on every file.
    detail = f"refusals {device_refusals} against reference {reference.refusals}"
    assert device_refusals == reference.refusals, f"{backend} {detail}"

    invented = device_keyed - reference.keyed
    unasked = f"{sum(invented.values())} {backend} outcomes"
    assert not invented, f"{unasked} the CPU reference does not contain"

    missing = reference.keyed - device_keyed
    if capped == 0:
        # No cap fired, so there is nothing for a shortfall to hide behind, whatever the
        # backend bounds. This is the branch that actually runs on both devices today.
        assert not missing, (
            f"{sum(missing.values())} reference outcomes missing from {backend} with no "
            f"cap in play"
        )
    else:
        # A capped application is one (molecule, rule, application), so every outcome it loses
        # carries that application's molecule and rule. The missing outcomes can therefore touch
        # at most `capped` distinct (molecule, rule) pairs however many components each lost --
        # which bounds the *spread* of a shortfall without needing to know how many outcomes one
        # application can hold. That matters because the component count is a device constant
        # Metal has and CUDA does not, and the arm without one used to assert nothing whatsoever:
        # a single capped application switched the check off, so an arbitrary divergence passed
        # as long as one application had also been capped. A real divergence is spread across
        # many pairs, which is exactly what this catches.
        span = {(molecule, rule) for molecule, rule, _key in missing}
        assert len(span) <= capped, (
            f"{sum(missing.values())} missing {backend} outcomes span {len(span)} "
            f"(molecule, rule) pairs, more than the {capped} that capped applications can "
            f"account for"
        )
        if components_per_capped_application is not None:
            # Tighter where the device bounds components per application: then the *count* of
            # missing outcomes is bounded too, not only how far they spread.
            allowed = capped * components_per_capped_application
            assert sum(missing.values()) <= allowed, (
                f"{sum(missing.values())} missing outcomes exceed the {allowed} that {capped} "
                f"capped applications can account for"
            )
    # Says which bound actually applied, rather than whether any did. `SHORTFALL-UNCHECKED` used
    # to mean "nothing was asserted here"; now the spread bound always is, and this records when
    # the tighter count bound was not available to go with it.
    unbounded = (
        ""
        if components_per_capped_application is not None or capped == 0
        else " shortfall-bounded-by-spread-only"
    )
    return (
        f"{backend} vs reference [{reference.provenance()}] "
        f"precursors={sum(device_keyed.values())}/{reference.precursor_count} "
        f"capped={capped} refusals={device_refusals}{unbounded}"
    )
