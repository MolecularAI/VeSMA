# AGENTS.md

## Architecture and programming style

- Optimize for readability and simplicity.
- Prefer elegant, algebraic, minimal-yet-expressive design.
- Prefer declarative style and functional programming inspired decomposition.
- Avoid object oriented programming verbosity/boilerplate.
- Prefer data classes if classes are needed. 
- Prefer frozen dataclasses with copy-on-write operations.
- Use collection types to indicate intent: lists, Sequence, Collection, Container for variable-size collections of the same type; tuples for fixed-size collections of different type. Use frozenset when order does not matter and hash is desirable.
- Use types and type hints in function signatures.
- Treat type hints as a contract: if caller violates the contract, the function do not have to return a correct answer, or return at all, it can simply raise, or let further underlying error escalate, no catching
- Do not do defensive type coersion, if type hint said it have to be int, it is up to the caller to ensure it was int, not up to the function to cast argument to int.
- Use static type checkers to make sure types are correct when a typed function is called.
- Fail fast, do not mask errors, let errors escalate.
- Optimizes for "minimal architecture" over "won’t break".
- Prefer no leading underscores in function and variable names. It is Python, nothing is private, we are all "consenting adults". If something have to be extra public, add it to `__all__` in the beginning of the module. Strive for a design where everything is public and externally usable. If function is not general enough to become public, consider if it is general enough to exist at all.
- Strive for a minimal set of modular reusable components/functions that allow for a rich expressive behavior to the users.
- Strive for a layered/onion architecture where outer-level/top-level functions call lower-level functions, but not the other way around. Outer layer is UI/IO/Database/State/Interaction, it consists of actions that can have side effects, i.e. actions depend on when and how many times they are called. Middle layers are calculations that are pure functions, they describe business logic. Innermost are libraries and language constructs.
- Prefer dependency-injection-style composition over string-based dispatch: express varying behavior as a typed `Callable[...]` parameter (or a Protocol if stateful multi-method behavior is needed), not `style="foo"` with if/elif, dispatch dicts, or factories. Let callers compose behavior at the call site, add named shortcuts only for common assemblies; keep string-to-implementation parsing at the outer I/O boundary (CLI/config).
- Keep an eye for algebraic structures, semirings, folds, valuations, dynamic programming. Prefer domian language to get composability benefits of algebras without introducing the steep learning curve. 

## Project-specific

### Running

- Put all tests into `tests/`, use test markers to enable or disable tests (e.g. slow, or specific GPU).
- Use pytest to run performance tests (`tests/**/perf_*.py`).
- Use pytest to run benchmarks (`tests/**/bench_*.py`); add targets to Makefile to run benchmarks.
- Paper compilation is documented in `Makefile`.
- How to run benchmarks is recorded in `Makefile`.

### Environment

- Use `uv` to create environment locally in `./.venv` in the worktree.
- If inside sandbox, keep `uv` cache in the worktree with `UV_CACHE_DIR="$PWD/.uv-cache"`.
- For repeated tests after syncing, prefer:
  - `./.venv/bin/python -m pytest`
  - `./.venv/bin/python -m ruff check .`
- Before handoff, run the full CI-compatible gate with `UV_CACHE_DIR="$PWD/.uv-cache" make check`.
- Paper compilation with Quarto needs socket access for Jupyter/Marimo.
- Accessing Metal on macOS needs sandbox elevation.
