from __future__ import annotations

import math
import multiprocessing as mp
import os
import weakref
from collections.abc import Callable, Sequence
from typing import Any

import joblib

__all__ = [
    "ResidentForkPool",
    "chunked",
    "effective_n_jobs",
    "parallel_chunks",
    "parallel_map",
    "parallel_map_fork",
    "resolve_n_jobs",
    "validate_n_jobs",
]


def resolve_n_jobs(n_jobs: int | None) -> int:
    if n_jobs is not None:
        return validate_n_jobs(n_jobs)

    value = os.environ.get("VESMA_N_JOBS")
    if value is None:
        return -1
    return validate_n_jobs(int(value))


def validate_n_jobs(n_jobs: int) -> int:
    if n_jobs == 0:
        raise ValueError("n_jobs must not be 0; use -1 for all available workers.")
    return n_jobs


def effective_n_jobs(n_jobs: int) -> int:
    if n_jobs > 0:
        return n_jobs
    return max(1, joblib.cpu_count() + 1 + n_jobs)


def chunked[T](items: Sequence[T], chunk_size: int) -> tuple[tuple[T, ...], ...]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return tuple(
        tuple(items[start : start + chunk_size])
        for start in range(0, len(items), chunk_size)
    )


def parallel_chunks[T](
    items: Sequence[T],
    *,
    n_jobs: int | None = None,
    chunks_per_worker: int = 4,
) -> tuple[tuple[T, ...], ...]:
    if not items:
        return ()
    jobs = resolve_n_jobs(n_jobs)
    if jobs == 1:
        return (tuple(items),)
    chunk_count = min(len(items), effective_n_jobs(jobs) * chunks_per_worker)
    chunk_size = math.ceil(len(items) / chunk_count)
    return chunked(items, chunk_size)


def parallel_map[T, U](
    function: Callable[[T], U],
    items: Sequence[T],
    *,
    n_jobs: int | None = None,
) -> tuple[U, ...]:
    jobs = resolve_n_jobs(n_jobs)
    if jobs == 1:
        return tuple(function(item) for item in items)

    with joblib.parallel_config(prefer="threads"):
        return tuple(
            joblib.Parallel(n_jobs=jobs, batch_size="auto")(
                joblib.delayed(function)(item) for item in items
            )
        )


def parallel_map_fork[T, U](
    function: Callable[[T], U],
    items: Sequence[T],
    *,
    initializer: Callable[..., None],
    initargs: tuple[Any, ...] = (),
    n_jobs: int | None = None,
) -> tuple[U, ...]:
    """Map over items using a multiprocessing fork pool to inherit read-only memory.

    Provides a zero-pickling overhead way to pass large global state to workers
    (like the entire dataset) by initializing globals in the forked process via
    `initializer(*initargs)`. The `items` passed to the `function` should be
    small identifiers (e.g. indices) rather than large payloads.
    """
    jobs = resolve_n_jobs(n_jobs)
    if jobs == 1:
        initializer(*initargs)
        return tuple(function(item) for item in items)

    ctx = mp.get_context("fork")
    with ctx.Pool(
        effective_n_jobs(jobs),
        initializer=initializer,
        initargs=initargs,
    ) as pool:
        return tuple(pool.map(function, items))


class ResidentForkPool:
    """A fork pool spawned once and kept, so its setup is paid per owner rather than per call.

    :func:`parallel_map_fork` stands a pool up and tears it down around a single map. That flat
    ~0.2 s is why small batches come out slower than running them serially. An owner that maps
    repeatedly over the same large read-only state — a compiled rule set, say — wants the pool
    to outlive the call: the state is inherited once, and every later map costs only its own
    payload. Measured on a five-molecule frontier against 68k RDKit rules, that is 1.22 s per
    call against 0.70 s, and it is what lets a *one*-molecule frontier be worth parallelising
    at all, which a per-call pool can never be.

    The state still reaches workers as inherited memory rather than a pickled payload, because
    it rides in ``initargs`` under the fork start method. Note the trade against
    :func:`parallel_map_fork`: a pool that predates the data has to receive each call's items
    the ordinary way, so this wins only when the resident state dwarfs the per-call payload.

    Idle workers block on the task queue and cost nothing but memory. ``n_jobs=1`` keeps no pool
    at all and maps inline, so callers need not branch on it.

    **Workers drift upward in memory and there is deliberately no automatic recycling.**
    Copy-on-write shares only the pages nobody writes, so a worker reaching into new parts of
    the inherited state keeps dirtying them — measured against a 68k-rule RDKit set, eight
    workers grew from 46 MB to 869 MB over four thousand distinct rules, sublinearly but
    steadily. ``Pool(maxtasksperchild=...)`` is the obvious cap and it **deadlocks here**: a
    replacement worker is forked by the pool's own ``_handle_workers`` helper thread, so the
    child inherits locks held by the other threads and hangs on its first task (observed on
    Python 3.14, every process idle at 0% CPU, the child's main thread stuck inside an inherited
    thread body). This is precisely the hazard behind 3.14's fork-from-a-multi-threaded-process
    warning. The initial spawn is safe by comparison because it happens on the owner's own
    thread. A caller that needs a ceiling should therefore call :meth:`close` at a point it
    controls — between searches, say — which respawns from that same safe position.

    Not picklable, and an owner holding one must not itself be sent to a worker.
    """

    def __init__(
        self,
        *,
        initializer: Callable[..., None],
        initargs: tuple[Any, ...] = (),
        n_jobs: int | None = None,
    ) -> None:
        jobs = resolve_n_jobs(n_jobs)
        self.initializer = initializer
        self.initargs = initargs
        self.n_workers = 1 if jobs == 1 else effective_n_jobs(jobs)
        self._pool: Any | None = None
        self._finalizer: weakref.finalize | None = None

    def map[T, U](
        self, function: Callable[[T], U], items: Sequence[T]
    ) -> tuple[U, ...]:
        if self.n_workers == 1:
            # Every inline map re-initialises, and it must. An initializer's job is to publish
            # state where the worker function will look for it, which in practice means module
            # globals — and on the inline path "the worker" is this process, shared with every
            # other owner. Initialising once per owner would let two matchers in one process
            # overwrite each other's state and then read the survivor's: not a crash, but the
            # wrong chemistry, silently. The pooled path is immune because each worker has its
            # own copy of the globals, forked once.
            self.initializer(*self.initargs)
            return tuple(function(item) for item in items)
        return tuple(self._ensure_pool().map(function, items))

    def _ensure_pool(self) -> Any:
        """Spawn on first use rather than in ``__init__``.

        Same steady state — the pool is stood up once and reused — but an owner that is built
        and never used, which is most of them in a test run, forks nothing. It also keeps the
        fork on the caller's own thread; see the class docstring on why that matters.
        """
        if self._pool is None:
            self._pool = mp.get_context("fork").Pool(
                self.n_workers,
                initializer=self.initializer,
                initargs=self.initargs,
            )
            # Explicit close() is the intended path; this is the backstop for owners that are
            # simply dropped. `__del__` would not do: it is not guaranteed to run and can hang
            # at interpreter shutdown. The callback must not close over `self`, or the
            # finalizer would keep the owner alive and never fire.
            self._finalizer = weakref.finalize(self, _terminate_pool, self._pool)
        return self._pool

    def close(self) -> None:
        """Terminate the workers. Idempotent; a later ``map`` spawns a fresh pool."""
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None
        self._pool = None

    def __enter__(self) -> ResidentForkPool:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _terminate_pool(pool: Any) -> None:
    pool.terminate()
    pool.join()
