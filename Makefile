.PHONY: \
	all test check sync-dev sync-cuda paper-render paper-si-render paper-approval-bundle \
	paper-release-check \
	bench-audit-rules-keys \
	bench-rdkit bench-rdchiral bench-precursor-audit \
	bench-cpu bench-cpu-match bench-metal bench-cuda bench-cuda-match \
	bench-metal-parity bench-cuda-parity \
	bench-cuda-replay bench-scaling \
	bench-folded-fingerprint bench-cuda-wl-key \
	bench-route-replay-rdkit bench-route-replay-cpu bench-route-replay-cuda \
	bench-route-replay-metal bench-route-replay \
	bench-route-replay-precursor-rdkit bench-route-replay-precursor-cpu \
	bench-route-replay-precursor-cuda bench-route-replay-precursor-metal \
	bench-route-replay-precursor \
	bench-route-recovery-rdkit bench-route-recovery-cpu bench-route-recovery-cuda \
	bench-route-recovery-cuda-matrix \
	bench-route-recovery-used14k-full \
	bench-route-recovery-metal bench-route-recovery _route_recovery_one \
	bench-route-recovery-uspto42k-rdkit bench-route-recovery-uspto42k-cpu \
	bench-route-recovery-uspto42k \
	bench-fanout-depth bench-fanout-depth-rdkit bench-fanout-depth-cuda \
	bench-fanout-depth-metal \
	bench-aostar bench-aostar-cuda bench-aostar-host bench-template-gaps bench-frontier-selection bench-remaining-targets bench-metal-pipeline bench-metal-precursor-streaming \
	bench-metal-assignment-capacity bench-metal-retro-phases bench-metal-match-grain \
	bench-refinement-dfs-depth-scaling bench-forward-calibration bench-silent-pairings bench-ring-membership \
	dataset-n1-d2 dataset-n1-d3 paroutes-raw-template-library \
	dataset-syntheseus-targets syntheseus-emolecules-inventory \
	uspto42k-compat-templates paroutes-n1-d2-match-uspto42k route-templates \
	raw68k-classified raw68k-curated raw68k-classified-download raw68k-classified-classify \
	bench-route-pipeline chemdots-enamine-annotated

PYTEST := .venv/bin/pytest
# Benchmarks are perf-marked and the default suite skips perf (pyproject addopts), so bench
# recipes re-enable them with `-m perf` (a later -m overrides the addopts default).
BENCH_PYTEST := $(PYTEST) -m perf
PYTHON := .venv/bin/python
VESMA_TEMPLATE_JOBS ?= -1
VESMA_N_JOBS ?= -1
BENCH_ENV := XDG_CACHE_HOME="$(PWD)/.profile-cache" UV_CACHE_DIR="$(PWD)/.uv-cache" VESMA_TEMPLATE_JOBS="$(VESMA_TEMPLATE_JOBS)" VESMA_N_JOBS="$(VESMA_N_JOBS)"
QUARTO_HOME ?= $(PWD)/.quarto-home

# --------------------------------------------------------------------------- #
# Vesma scaling benchmarks (pytest-benchmark)
#
# The Makefile selects which backends run; pytest parametrizes the molecule
# sizes. Each backend is one pytest invocation that writes one JSON file via
# --benchmark-json. Rounds/warmup are folded into pytest-benchmark.
#
#   make bench-rdkit                       # rdkit backends, default sizes
#   make bench-cpu CPU_SIZES="1 10"        # cpu backends, small sizes
#   make bench-cuda BENCH_HOST=rtx-4090    # cuda backends on a RunPod box
#   make bench-metal                       # metal backends on an Apple box
# --------------------------------------------------------------------------- #
BENCH_ROUNDS ?= 3
# Warm-up rounds fed to benchmark.pedantic (pytest-benchmark's --benchmark-warmup CLI flag is
# ignored by pedantic). Default 1; slow benches (precursor replay) override this to 0.
BENCH_WARMUP_ROUNDS ?= 1
# Combined cpu+gpu box slug (e.g. epyc-7763+rtx-4090); override with BENCH_HOST.
BENCH_HOST ?= $(shell $(PYTHON) -c "from tests.vertsy.paper_provenance import host_slug; print(host_slug())")
BENCH_DIR := docs/paper/benchmarks
# Per-family default molecule sizes (override on the command line).
RDKIT_SIZES ?= 1 10 100 1000 2881
CPU_SIZES   ?= 1 10 100
GPU_SIZES   ?= 1 10 100 1000 2881
# RDChiral is a pure-Python wrapper measured at roughly 9x rdkit-serial, and compiling
# the 42,554 templates costs about three minutes on top. One timed round, no warm-up.
RDCHIRAL_SIZES  ?= 1 10 100 1000
RDCHIRAL_ROUNDS ?= 1
RDCHIRAL_WARMUP ?= 0
ROUTE_REPLAY_RDKIT_SIZES ?= 1 10 100
ROUTE_REPLAY_CPU_SIZES   ?= 1 10
ROUTE_REPLAY_GPU_SIZES   ?= 1 10 100 1000
ROUTE_REPLAY_MODES ?= guided topology
# guided/topology build the matcher from the route's own reactions; precursor replay filters a
# full library by precursor set (rule ids need not align), so it takes a real rule set.
ROUTE_REPLAY_RULE_SETS ?= route-reactions
# Precursor replay: used14k is the aligned sanity baseline (n1n5-used subset, ~100% recovery),
# raw68k the full raw USPTO library, uspto42k the curated foreign-library generalization test.
# Precursor replay applies the whole library per node, so it is minutes-long: CUDA takes 100
# routes, rdkit/cpu only 10. One measured round, no warm-up (a warm-up would double wall time).
ROUTE_REPLAY_PRECURSOR_GPU_SIZES   ?= 100
ROUTE_REPLAY_PRECURSOR_RDKIT_SIZES ?= 10
ROUTE_REPLAY_PRECURSOR_CPU_SIZES   ?= 10
ROUTE_REPLAY_PRECURSOR_ROUNDS      ?= 1
ROUTE_REPLAY_PRECURSOR_WARMUP      ?= 0
ROUTE_REPLAY_PRECURSOR_RULE_SETS   ?= used14k raw68k uspto42k
# Route-level worker processes for rdkit/cpu replay (joblib n_jobs; -1 = all cores, 1 = serial).
ROUTE_REPLAY_JOBS ?= -1
# Route recovery searches the full library, so it is heavy: one target, one round here.
ROUTE_RECOVERY_TARGETS ?= 1
ROUTE_RECOVERY_ROUNDS  ?= 1
# used14k: recovery rate correct, timing misleading (library is a subset). uspto42k: standard
# uspto_unique_templates, so both recovery rate and end-to-end timing are meaningful.
ROUTE_RECOVERY_RULE_SETS ?= used14k
# Fanout-per-depth: full deep expansion of the smallest library (used14k) by default, probing
# depth on a single target first. Heavy — cuda-host (GPU) is the intended path; rdkit only for a
# shallow local smoke. FRONTIER_CEILING aborts a runaway expansion before it OOMs the box.
FANOUT_DEPTH_RULE_SETS ?= used14k
FANOUT_DEPTH_TARGETS   ?= 1
FANOUT_DEPTH_MAX_DEPTH ?= 3
FANOUT_DEPTH_FRONTIER_CEILING ?= 2000000
# Target set (frontier 0) to expand. Default n1-d2-f0 (depth-2 targets); use n1-d3-f0.smi.gz for
# the depth-3 targets so a depth-3 expansion recovers real routes instead of over-expanding.
FANOUT_DEPTH_TARGETS_FILE ?= n1-d2-f0.smi.gz
# Refinement-DFS depth sweep self-times per depth, so one round is plenty.
REFINEMENT_DFS_DEPTH_ROUNDS ?= 1
BENCH_RUN_ENV = $(BENCH_ENV) BENCH_ROUNDS=$(BENCH_ROUNDS) BENCH_WARMUP_ROUNDS=$(BENCH_WARMUP_ROUNDS) BENCH_HOST_LABEL="$(BENCH_HOST)"

# $(call run_match_only,<backend>,<sizes>). The timestamp is taken in the shell at
# run time (not Make recipe-expansion time) so the filename reflects the actual run.
define run_match_only
stamp=$$(date -u +%Y%m%d-%H%M%S); mkdir -p $(BENCH_DIR)/match-only-scaling/$(1) && $(BENCH_RUN_ENV) BENCH_SCALING_SIZES="$(2)" BENCH_MATCH_ONLY_BACKENDS="$(1)" $(BENCH_PYTEST) tests/vesma/bench_match_scaling.py::test_match_only_scaling -v --benchmark-json=$(BENCH_DIR)/match-only-scaling/$(1)/$(BENCH_HOST)__$$stamp.json $(EXTRA_ARGS)
endef

# Optional RDChiral diagnostic, outside the publication performance comparison.
define run_rdchiral
stamp=$$(date -u +%Y%m%d-%H%M%S); mkdir -p $(BENCH_DIR)/rdchiral/$(1) && $(BENCH_RUN_ENV) BENCH_SCALING_SIZES="$(2)" BENCH_SPLIT_BACKENDS="$(1)" $(BENCH_PYTEST) tests/vesma/bench_match_scaling.py::test_match_plus_split_scaling -v --benchmark-json=$(BENCH_DIR)/rdchiral/$(1)/$(BENCH_HOST)__$$stamp.json $(EXTRA_ARGS)
endef

# $(call run_route_replay,<backend>,<sizes>)
define run_route_replay
stamp=$$(date -u +%Y%m%d-%H%M%S); mkdir -p $(BENCH_DIR)/route-replay/$(1) && $(BENCH_RUN_ENV) BENCH_ROUTE_REPLAY_SIZES="$(2)" BENCH_ROUTE_REPLAY_BACKENDS="$(1)" BENCH_ROUTE_REPLAY_MODES="$(ROUTE_REPLAY_MODES)" BENCH_ROUTE_REPLAY_RULE_SETS="$(ROUTE_REPLAY_RULE_SETS)" BENCH_ROUTE_REPLAY_JOBS="$(ROUTE_REPLAY_JOBS)" $(BENCH_PYTEST) tests/vertsy/bench_route_replay.py::test_bench_route_replay_n1 -v --benchmark-json=$(BENCH_DIR)/route-replay/$(1)/$(BENCH_HOST)__$$stamp.json $(EXTRA_ARGS)
endef

# $(call run_route_recovery,<backend>,<targets>)
define run_route_recovery
stamp=$$(date -u +%Y%m%d-%H%M%S); mkdir -p $(BENCH_DIR)/route-recovery/$(1) && $(BENCH_RUN_ENV) BENCH_ROUNDS="$(ROUTE_RECOVERY_ROUNDS)" BENCH_ROUTE_RECOVERY_TARGETS="$(2)" BENCH_ROUTE_RECOVERY_BACKENDS="$(1)" BENCH_ROUTE_RECOVERY_RULE_SETS="$(ROUTE_RECOVERY_RULE_SETS)" $(BENCH_PYTEST) tests/vertsy/bench_route_recovery.py::test_bench_route_recovery_n1_d2 -v --benchmark-json=$(BENCH_DIR)/route-recovery/$(1)/$(BENCH_HOST)__$$stamp.json $(EXTRA_ARGS)
endef

# $(call run_fanout_depth,<backend>) — deep fanout-per-depth expansion (config via FANOUT_DEPTH_*).
define run_fanout_depth
stamp=$$(date -u +%Y%m%d-%H%M%S); mkdir -p $(BENCH_DIR)/fanout-depth/$(1) && $(BENCH_RUN_ENV) BENCH_FANOUT_DEPTH_BACKENDS="$(1)" BENCH_FANOUT_DEPTH_RULE_SETS="$(FANOUT_DEPTH_RULE_SETS)" BENCH_FANOUT_DEPTH_TARGETS="$(FANOUT_DEPTH_TARGETS)" BENCH_FANOUT_DEPTH_TARGETS_FILE="$(FANOUT_DEPTH_TARGETS_FILE)" BENCH_FANOUT_DEPTH_MAX_DEPTH="$(FANOUT_DEPTH_MAX_DEPTH)" BENCH_FANOUT_DEPTH_FRONTIER_CEILING="$(FANOUT_DEPTH_FRONTIER_CEILING)" $(BENCH_PYTEST) tests/vertsy/bench_fanout_depth.py::test_bench_fanout_depth -v --benchmark-json=$(BENCH_DIR)/fanout-depth/$(1)/$(BENCH_HOST)__$$stamp.json $(EXTRA_ARGS)
endef

# $(call run_refinement_dfs_depth_scaling) — CUDA refinement-DFS depth sweep (cuda only).
define run_refinement_dfs_depth_scaling
stamp=$$(date -u +%Y%m%d-%H%M%S); mkdir -p $(BENCH_DIR)/refinement-dfs-depth-scaling/cuda && $(BENCH_RUN_ENV) BENCH_ROUNDS="$(REFINEMENT_DFS_DEPTH_ROUNDS)" $(BENCH_PYTEST) tests/vesma/bench_refinement_dfs_depth_scaling.py::test_bench_refinement_dfs_depth_scaling -v --benchmark-json=$(BENCH_DIR)/refinement-dfs-depth-scaling/cuda/$(BENCH_HOST)__$$stamp.json $(EXTRA_ARGS)
endef

# $(call run_folded_fingerprint) — CPU folded-fingerprint bit-size sweep.
define run_folded_fingerprint
stamp=$$(date -u +%Y%m%d-%H%M%S); mkdir -p $(BENCH_DIR)/folded-fingerprint/cpu && $(BENCH_RUN_ENV) $(BENCH_PYTEST) tests/vesma/perf_folded_fingerprint.py::test_bench_folded_fingerprint -v --benchmark-json=$(BENCH_DIR)/folded-fingerprint/cpu/$(BENCH_HOST)__$$stamp.json $(EXTRA_ARGS)
endef

# $(call run_cuda_wl_key) — standalone short-ring WL kernel microbenchmark (cuda only).
define run_cuda_wl_key
stamp=$$(date -u +%Y%m%d-%H%M%S); mkdir -p $(BENCH_DIR)/cuda-short-ring-wl-key/cuda && $(BENCH_RUN_ENV) $(BENCH_PYTEST) tests/vesma/perf_cuda_wl_key.py::test_perf_cuda_wl_key -v --benchmark-json=$(BENCH_DIR)/cuda-short-ring-wl-key/cuda/$(BENCH_HOST)__$$stamp.json $(EXTRA_ARGS)
endef

define run_audit_rules_keys
stamp=$$(date -u +%Y%m%d-%H%M%S); mkdir -p $(BENCH_DIR)/paper-validation/cpu && $(BENCH_RUN_ENV) $(BENCH_PYTEST) tests/vesma/bench_audit_rules_keys.py -v --benchmark-json=$(BENCH_DIR)/paper-validation/cpu/$(BENCH_HOST)__$$stamp.json $(EXTRA_ARGS)
endef

all: test

# rdkit ships with the dev extra, so `--extra dev` already pulls it in.
sync-dev:
	UV_CACHE_DIR="$(PWD)/.uv-cache" uv sync --extra dev

sync-cuda:
	UV_CACHE_DIR="$(PWD)/.uv-cache" uv sync --python 3.14 --extra dev --extra cuda

# The full gate CI runs (ruff + both formatters + ty + pytest). Run it on the Mac for the
# Metal backend and on a GPU instance for the CUDA suite -- the substitute for GPU CI.
check:
	uv run ruff check .
	uv run ruff format --check .
	uv run black --check .
	uv run ty check
	$(BENCH_ENV) $(PYTEST) tests/ $(EXTRA_ARGS)

test:
	$(BENCH_ENV) $(PYTEST) tests/ $(EXTRA_ARGS)

bench-rdkit: bench-application-rdkit
bench-rdkit: tests/data/external/paroutes/processed/n1-d2-f0.smi.gz tests/data/external/remix/paroutes-n1-d2-match-uspto42k/processed/n1-d2-f0-uspto42k-compat-matches.json.gz
	$(call run_match_only,rdkit-filtercatalog-rules,$(RDKIT_SIZES))
	$(call run_match_only,rdkit-substructlibrary-frontier,$(RDKIT_SIZES))

bench-cpu: bench-application-cpu
bench-cpu: tests/data/external/paroutes/processed/n1-d2-f0.smi.gz tests/data/external/remix/paroutes-n1-d2-match-uspto42k/processed/n1-d2-f0-uspto42k-compat-matches.json.gz
	$(call run_match_only,cpu-serial,$(CPU_SIZES))
	$(call run_match_only,cpu-parallel,$(CPU_SIZES))

# Multicore CPU match-only scaling without also starting serial match-only or
# match-plus-split measurements.
bench-cpu-match: tests/data/external/paroutes/processed/n1-d2-f0.smi.gz tests/data/external/remix/paroutes-n1-d2-match-uspto42k/processed/n1-d2-f0-uspto42k-compat-matches.json.gz
	$(call run_match_only,cpu-parallel,$(CPU_SIZES))

# RDChiral comparison backend. Results stay OUT of the paper tree: `rdchiral-serial` is
# not in the paper loaders' supported-backend list, and any JSON under
# docs/paper/benchmarks with an unknown backend fails both the paper render and
# tests/paper/test_paper_loaders.py. Delete the BENCH_DIR override to publish.
bench-rdchiral: BENCH_DIR = output/benchmarks
bench-rdchiral: BENCH_ROUNDS = $(RDCHIRAL_ROUNDS)
bench-rdchiral: BENCH_WARMUP_ROUNDS = $(RDCHIRAL_WARMUP)
bench-rdchiral: tests/data/external/paroutes/processed/n1-d2-f0.smi.gz tests/data/external/remix/paroutes-n1-d2-match-uspto42k/processed/n1-d2-f0-uspto42k-compat-matches.json.gz
	$(call run_rdchiral,rdchiral-serial,$(RDCHIRAL_SIZES))
	$(call run_rdchiral,rdchiral-serial-nocombine,$(RDCHIRAL_SIZES))

# Application-level precursor audit: vesma vs RDKit vs RDChiral. A correctness artifact,
# not a timing one -- it runs at rounds=1 and its wall time is context only. Kept out of
# the paper tree for the same reason as bench-rdchiral.
#
# The artifact is written uncapped, so it carries every divergence and every refusal in
# full; that is the point of it, and it is why the recipe gzips afterwards rather than
# trimming. pytest-benchmark can only write plain JSON, hence the separate compression
# step and the removal of the uncompressed original.
AUDIT_SIZES ?= 100

bench-precursor-audit: BENCH_DIR = output/benchmarks
bench-precursor-audit: tests/data/external/paroutes/processed/n1-d2-f0.smi.gz tests/data/external/remix/paroutes-n1-d2-match-uspto42k/processed/n1-d2-f0-uspto42k-compat-matches.json.gz
	stamp=$$(date -u +%Y%m%d-%H%M%S); out=$(BENCH_DIR)/precursor-audit/cpu; mkdir -p $$out && $(BENCH_RUN_ENV) BENCH_AUDIT_SIZES="$(AUDIT_SIZES)" $(BENCH_PYTEST) tests/vesma/bench_precursor_audit.py -v --benchmark-json=$$out/$(BENCH_HOST)__$$stamp.json $(EXTRA_ARGS) && gzip -9 -f $$out/$(BENCH_HOST)__$$stamp.json

# CPU/GPU chemistry parity: same precursors and the same typed refusals, over the corpus.
# A correctness artifact, not a timing one.
PARITY_SIZES ?= 3 25 100
bench-metal-parity:
	$(BENCH_RUN_ENV) PARITY_SIZES="$(PARITY_SIZES)" $(BENCH_PYTEST) tests/vesma/bench_metal_parity.py -v -s $(EXTRA_ARGS)

bench-cuda-parity:
	$(BENCH_RUN_ENV) PARITY_SIZES="$(PARITY_SIZES)" $(BENCH_PYTEST) tests/vesma/bench_cuda_parity.py -v -s $(EXTRA_ARGS)

# How well the forward rule table describes chemistry that demonstrably worked: recall against
# recorded literature reactions, and the rate at which the probe calls one of them a mixture.
# No search, so it is minutes rather than days, and it says nothing about any retro library --
# the record supplies its own product, so the intended reaction is read off the data.
# Which pairings report nothing on real molecules, ranked by volume. The composition lint in
# test_forward_templates.py sees one example per row; this sees every substrate the corpus offers,
# and a silent pairing costs a route real money because silence is charged as low confidence.
SILENT_PAIRING_PRODUCTS ?= 600
bench-silent-pairings:
	$(BENCH_RUN_ENV) SILENT_PAIRING_PRODUCTS="$(SILENT_PAIRING_PRODUCTS)" $(BENCH_PYTEST) tests/vertsy/bench_silent_pairings.py -v -s $(EXTRA_ARGS)

FORWARD_CALIBRATION_RECORDS ?= 400
bench-forward-calibration:
	$(BENCH_RUN_ENV) FORWARD_CALIBRATION_RECORDS="$(FORWARD_CALIBRATION_RECORDS)" $(BENCH_PYTEST) tests/vertsy/bench_forward_calibration.py -v -s $(EXTRA_ARGS)

# Probe Metal for machine_info provenance only when actually benchmarking it.
bench-metal: bench-application-metal
bench-metal: BENCH_RUN_ENV += BENCH_PROBE_METAL=1
bench-metal: tests/data/external/paroutes/processed/n1-d2-f0.smi.gz tests/data/external/remix/paroutes-n1-d2-match-uspto42k/processed/n1-d2-f0-uspto42k-compat-matches.json.gz
	$(call run_match_only,metal,$(GPU_SIZES))
	$(call run_match_only,metal-raw,$(GPU_SIZES))

bench-cuda: bench-cuda-match bench-application-cuda bench-cuda-replay

bench-cuda-match: tests/data/external/paroutes/processed/n1-d2-f0.smi.gz tests/data/external/remix/paroutes-n1-d2-match-uspto42k/processed/n1-d2-f0-uspto42k-compat-matches.json.gz
	$(call run_match_only,cuda,$(GPU_SIZES))
	$(call run_match_only,cuda-raw,$(GPU_SIZES))

bench-cuda-replay: tests/data/external/paroutes/processed/used14k-reactions.tsv.gz
	$(call run_route_replay,cuda,$(ROUTE_REPLAY_GPU_SIZES))

# CPU/RDKit fan-out is safe on any machine; GPU families are run explicitly.
bench-scaling: bench-rdkit bench-cpu

# Folded linear-fingerprint bit-size sweep (collision rate + prune ratio).
bench-folded-fingerprint:
	$(call run_folded_fingerprint)

# CUDA refinement-DFS depth sweep (skips without CUDA).
bench-refinement-dfs-depth-scaling:
	$(call run_refinement_dfs_depth_scaling)

# Standalone short-ring WL kernel microbenchmark (skips without CUDA).
bench-cuda-wl-key:
	$(call run_cuda_wl_key)

# SMARTS R/r perception (Horton + tiered GF(2), BFS-distance r) against the short-ring check:
# CPU timings and word-op counts, then Metal kernels (skip off macOS; need direct GPU access).
bench-ring-membership: BENCH_DIR = output/benchmarks
bench-ring-membership:
	stamp=$$(date -u +%Y%m%d-%H%M%S); mkdir -p $(BENCH_DIR)/ring-membership && $(BENCH_RUN_ENV) $(BENCH_PYTEST) tests/vesma/perf_ring_membership.py -v -s --benchmark-json=$(BENCH_DIR)/ring-membership/$(BENCH_HOST)__$$stamp.json $(EXTRA_ARGS)

bench-audit-rules-keys:
	$(call run_audit_rules_keys)

dataset-n1-d2:
	$(BENCH_ENV) $(PYTHON) tests/data/external/paroutes/extract_n1_d2.py --routes tests/data/external/paroutes/processed/n1-routes.json.gz --out-targets tests/data/external/paroutes/processed/n1-d2-targets.smi.gz
	$(BENCH_ENV) $(PYTHON) tests/data/external/paroutes/generate_f0_f1_f01.py --targets tests/data/external/paroutes/processed/n1-d2-targets.smi.gz --rules tests/data/external/paroutes/processed/uspto_unique_templates.csv.gz --f0-out tests/data/external/paroutes/processed/n1-d2-f0.smi.gz --f1-out tests/data/external/paroutes/processed/n1-d2-uspto42k-f1.smi.gz --f01-out tests/data/external/paroutes/processed/n1-d2-uspto42k-f01.smi.gz --n-jobs 32

# n1-d3-f0: targets (frontier 0) whose reference route needs three steps, disjoint from n1-d2.
# Just the target SMILES — the fanout bench needs no f1/f01 variants (an n1-d3-f1 could be added
# later for deeper studies). Committed, so RunPod only pulls and runs.
dataset-n1-d3: tests/data/external/paroutes/processed/n1-d3-f0.smi.gz

tests/data/external/paroutes/processed/n1-d3-f0.smi.gz: tests/data/external/paroutes/processed/n1-routes.json.gz
	$(BENCH_ENV) $(PYTHON) tests/data/external/paroutes/extract_n1_d2.py --routes tests/data/external/paroutes/processed/n1-routes.json.gz --out-targets tests/data/external/paroutes/processed/n1-d3-f0.smi.gz --depth 3

tests/data/external/paroutes/processed/n1-d2-f0.smi.gz:
	$(MAKE) dataset-n1-d2

# Retro* Hard: the 190 targets the syntheseus benchmark searches for. Committed (3 KB), so
# this rule runs only on a checkout that is missing them.
dataset-syntheseus-targets: tests/data/external/syntheseus/targets/retrostar190.smi.gz

tests/data/external/syntheseus/targets/retrostar190.smi.gz:
	$(BENCH_ENV) $(PYTHON) tests/data/external/syntheseus/prepare_syntheseus_targets.py --only retrostar190

# The 23M-molecule eMolecules inventory those targets are searched against: 273 MB, never
# committed, fetched only when a run actually needs the stock.
syntheseus-emolecules-inventory: tests/data/external/syntheseus/raw/emolecules-inventory.csv.gz

tests/data/external/syntheseus/raw/emolecules-inventory.csv.gz:
	$(BENCH_ENV) $(PYTHON) tests/data/external/syntheseus/prepare_syntheseus_targets.py --only emolecules-inventory --skip-existing

# The ChemoDOTS Enamine catalogue keeping its vendor ids: ~41 MB downloaded, derived into a
# gitignored sibling of the committed SMILES-only file. Only a *report* needs it -- it is what
# lets a route's starting materials print the catalogue number a chemist orders by -- and a
# search against either file is the same search on the same cache. The script writes both.
chemdots-enamine-annotated: tests/data/external/chemdots-stock/processed/chemdots_enamine_bbs_annotated.smi.bz2

tests/data/external/chemdots-stock/processed/chemdots_enamine_bbs_annotated.smi.bz2:
	bash tests/data/external/chemdots-stock/prepare.sh

paroutes-raw-template-library: tests/data/external/paroutes/raw/uspto_raw_template_library.csv.gz

tests/data/external/paroutes/raw/uspto_raw_template_library.csv.gz:
	$(BENCH_ENV) $(PYTHON) tests/data/external/paroutes/prepare_paroutes.py --only uspto-raw-template-library --skip-existing

# used14k-reactions.tsv.gz and raw68k-templates.tsv.gz are committed derived artifacts, so
# their file rules carry NO source prerequisite. Declaring the raw ~820k-template library as
# a prerequisite made Make download it (and needlessly rebuild the committed output) whenever
# the raw file was absent -- i.e. on every fresh checkout that already had the committed
# artifacts -- just to run a benchmark. With no prerequisites, Make runs a rule only when its
# committed target is actually missing. To regenerate the pair from scratch, run
# `make route-templates`, which fetches the raw library first and forces both builds.
route-templates: paroutes-raw-template-library
	$(BENCH_ENV) $(PYTHON) tests/data/external/paroutes/build_used14k_reactions.py
	$(BENCH_ENV) $(PYTHON) tests/data/external/paroutes/build_raw68k_templates.py

tests/data/external/paroutes/processed/used14k-reactions.tsv.gz:
	$(BENCH_ENV) $(PYTHON) tests/data/external/paroutes/build_used14k_reactions.py

# raw68k: full raw library deduplicated to its 67,557 unique templates (committed, ~3.5 MB).
tests/data/external/paroutes/processed/raw68k-templates.tsv.gz:
	$(BENCH_ENV) $(PYTHON) tests/data/external/paroutes/build_raw68k_templates.py

uspto42k-compat-templates: tests/data/external/remix/uspto42k-compat/processed/uspto42k-compat.csv.gz

tests/data/external/remix/uspto42k-compat/processed/uspto42k-compat.csv.gz: tests/data/external/paroutes/processed/uspto_unique_templates.csv.gz
	$(BENCH_ENV) $(PYTHON) tests/data/external/remix/uspto42k-compat/filter_uspto42k_compat.py --source tests/data/external/paroutes/processed/uspto_unique_templates.csv.gz --output tests/data/external/remix/uspto42k-compat/processed/uspto42k-compat.csv.gz

# raw68k-classified: every raw68k template labelled by two reaction classifiers, with a
# verdict marking the ones they agree on. The normal path reads only committed files --
# vendored/ plus the paroutes templates -- so it needs no network and takes seconds.
#
# The two optional targets below are deliberately NOT prerequisites of anything. Making them
# prerequisites would mean a fresh checkout, or a stale timestamp on a vendored file, silently
# triggering a 317 MB download or hours of classification.
raw68k-classified: tests/data/external/remix/raw68k-classified/processed/raw68k-classified.tsv.gz

tests/data/external/remix/raw68k-classified/processed/raw68k-classified.tsv.gz: tests/data/external/paroutes/processed/raw68k-templates.tsv.gz tests/data/external/remix/raw68k-classified/vendored/reactionclassifier-labels.json.gz tests/data/external/remix/raw68k-classified/vendored/rxn-insight-labels.json.gz
	$(BENCH_ENV) $(PYTHON) tests/data/external/remix/raw68k-classified/build_raw68k_classified.py

# The curated projection: the rows whose two classifiers agree. A pure function of the
# committed file above, so it is gitignored and rebuilt here in about a second rather than
# tracked. The output is a gzipped TSV with a `retro_template` column, so it is directly
# usable as a vertsy rule set. `--verdict` selects any other verdict, e.g. to inspect what
# curation removed: `... filter_curated.py --verdict unconfirmed`.
raw68k-curated: tests/data/external/remix/raw68k-classified/processed/raw68k-agree.tsv.gz

tests/data/external/remix/raw68k-classified/processed/raw68k-agree.tsv.gz: tests/data/external/remix/raw68k-classified/processed/raw68k-classified.tsv.gz
	$(BENCH_ENV) $(PYTHON) tests/data/external/remix/raw68k-classified/filter_curated.py

# Optional. Fetches what only re-classification needs: the 106 MB PaRoutes raw library (the
# reactions the classifiers read) and the 211 MB published corpus that vendored/zenodo-labels
# was projected from. Both gitignored.
raw68k-classified-download:
	$(BENCH_ENV) $(PYTHON) tests/data/external/paroutes/prepare_paroutes.py --only uspto-raw-template-library
	curl -sL --retry 2 --create-dirs -o tests/data/external/remix/raw68k-classified/raw/reaction_db_public.parquet https://zenodo.org/records/21097758/files/reaction_db_public.parquet
	printf 'af8482038abf24adc91e33dfd009392c61b9629dca4c2f5aff96d53295390549  tests/data/external/remix/raw68k-classified/raw/reaction_db_public.parquet\n' | shasum -a 256 -c -

# Optional, and slow: ReactionClassifier ~10 min, Rxn-INSIGHT ~3.5 h (99 ms/reaction over
# ~127k reactions). Rewrites the committed vendored/*-labels.json.gz, so run it only to
# re-derive them; `make raw68k-classified` does not need it. Requires raw68k-classified-download
# first, and layers each classifier over the venv with uv rather than installing it.
raw68k-classified-classify:
	OMP_NUM_THREADS=1 UV_CACHE_DIR="$(PWD)/.uv-cache" uv run --no-project --with 'reactionclassifier==0.1.1' --python ./.venv/bin/python python tests/data/external/remix/raw68k-classified/classify_reactionclassifier.py
	OMP_NUM_THREADS=1 UV_CACHE_DIR="$(PWD)/.uv-cache" uv run --no-project --with 'rxn-insight' --python ./.venv/bin/python python tests/data/external/remix/raw68k-classified/classify_rxn_insight.py

paroutes-n1-d2-match-uspto42k: tests/data/external/remix/paroutes-n1-d2-match-uspto42k/processed/n1-d2-f0-uspto42k-compat-matches.json.gz

tests/data/external/remix/paroutes-n1-d2-match-uspto42k/processed/n1-d2-f0-uspto42k-compat-matches.json.gz: tests/data/external/paroutes/processed/n1-d2-f0.smi.gz tests/data/external/remix/uspto42k-compat/processed/uspto42k-compat.csv.gz
	$(BENCH_ENV) $(PYTHON) tests/data/external/remix/paroutes-n1-d2-match-uspto42k/generate_matches.py --targets tests/data/external/paroutes/processed/n1-d2-f0.smi.gz --patterns tests/data/external/remix/uspto42k-compat/processed/uspto42k-compat.csv.gz --output tests/data/external/remix/paroutes-n1-d2-match-uspto42k/processed/n1-d2-f0-uspto42k-compat-matches.json.gz

# --------------------------------------------------------------------------- #
# Vertsy route replay benchmarks (pytest-benchmark)
# --------------------------------------------------------------------------- #
bench-route-replay-rdkit: tests/data/external/paroutes/processed/used14k-reactions.tsv.gz
	$(call run_route_replay,rdkit,$(ROUTE_REPLAY_RDKIT_SIZES))

bench-route-replay-cpu: tests/data/external/paroutes/processed/used14k-reactions.tsv.gz
	$(call run_route_replay,cpu,$(ROUTE_REPLAY_CPU_SIZES))

bench-route-replay-cuda: bench-cuda-replay

bench-route-replay-metal: BENCH_RUN_ENV += BENCH_PROBE_METAL=1
bench-route-replay-metal: tests/data/external/paroutes/processed/used14k-reactions.tsv.gz
	$(call run_route_replay,metal,$(ROUTE_REPLAY_GPU_SIZES))

bench-route-replay: bench-route-replay-rdkit bench-route-replay-cpu

# Precursor replay: apply a full library and keep only reference precursor sets. used14k rule
# ids align (baseline), raw68k/uspto42k rule ids do not (foreign libraries). rdkit/cpu use
# route-level joblib parallelism by default; a foreign library is matched per node, so keep sizes
# modest. One measured round, no warm-up (ROUTE_REPLAY_PRECURSOR_ROUNDS/_WARMUP): a warm-up would
# double the minutes-long run.
bench-route-replay-precursor-rdkit: ROUTE_REPLAY_MODES = precursor
bench-route-replay-precursor-rdkit: ROUTE_REPLAY_RULE_SETS = $(ROUTE_REPLAY_PRECURSOR_RULE_SETS)
bench-route-replay-precursor-rdkit: BENCH_ROUNDS = $(ROUTE_REPLAY_PRECURSOR_ROUNDS)
bench-route-replay-precursor-rdkit: BENCH_WARMUP_ROUNDS = $(ROUTE_REPLAY_PRECURSOR_WARMUP)
bench-route-replay-precursor-rdkit: tests/data/external/paroutes/processed/used14k-reactions.tsv.gz tests/data/external/paroutes/processed/raw68k-templates.tsv.gz tests/data/external/paroutes/processed/uspto_unique_templates.csv.gz
	$(call run_route_replay,rdkit,$(ROUTE_REPLAY_PRECURSOR_RDKIT_SIZES))

bench-route-replay-precursor-cpu: ROUTE_REPLAY_MODES = precursor
bench-route-replay-precursor-cpu: ROUTE_REPLAY_RULE_SETS = $(ROUTE_REPLAY_PRECURSOR_RULE_SETS)
bench-route-replay-precursor-cpu: BENCH_ROUNDS = $(ROUTE_REPLAY_PRECURSOR_ROUNDS)
bench-route-replay-precursor-cpu: BENCH_WARMUP_ROUNDS = $(ROUTE_REPLAY_PRECURSOR_WARMUP)
bench-route-replay-precursor-cpu: tests/data/external/paroutes/processed/used14k-reactions.tsv.gz tests/data/external/paroutes/processed/raw68k-templates.tsv.gz tests/data/external/paroutes/processed/uspto_unique_templates.csv.gz
	$(call run_route_replay,cpu,$(ROUTE_REPLAY_PRECURSOR_CPU_SIZES))

# CUDA precursor replay: one route at a time on the GPU (RunPod path), 100 routes by default.
bench-route-replay-precursor-cuda: ROUTE_REPLAY_MODES = precursor
bench-route-replay-precursor-cuda: ROUTE_REPLAY_RULE_SETS = $(ROUTE_REPLAY_PRECURSOR_RULE_SETS)
bench-route-replay-precursor-cuda: ROUTE_REPLAY_JOBS = 1
bench-route-replay-precursor-cuda: BENCH_ROUNDS = $(ROUTE_REPLAY_PRECURSOR_ROUNDS)
bench-route-replay-precursor-cuda: BENCH_WARMUP_ROUNDS = $(ROUTE_REPLAY_PRECURSOR_WARMUP)
bench-route-replay-precursor-cuda: tests/data/external/paroutes/processed/used14k-reactions.tsv.gz tests/data/external/paroutes/processed/raw68k-templates.tsv.gz tests/data/external/paroutes/processed/uspto_unique_templates.csv.gz
	$(call run_route_replay,cuda,$(ROUTE_REPLAY_PRECURSOR_GPU_SIZES))

# Metal precursor replay: raw vs folded-fingerprint prefilter (the only replay mode the
# prefilter helps -- topology/guided pass explicit pairs and skip it). Runs both backends
# as an A/B on the same routes. Metal is single-process on-device, so JOBS stays 1.
bench-route-replay-precursor-metal: BENCH_RUN_ENV += BENCH_PROBE_METAL=1
bench-route-replay-precursor-metal: ROUTE_REPLAY_MODES = precursor
bench-route-replay-precursor-metal: ROUTE_REPLAY_RULE_SETS = $(ROUTE_REPLAY_PRECURSOR_RULE_SETS)
bench-route-replay-precursor-metal: ROUTE_REPLAY_JOBS = 1
bench-route-replay-precursor-metal: BENCH_ROUNDS = $(ROUTE_REPLAY_PRECURSOR_ROUNDS)
bench-route-replay-precursor-metal: BENCH_WARMUP_ROUNDS = $(ROUTE_REPLAY_PRECURSOR_WARMUP)
bench-route-replay-precursor-metal: tests/data/external/paroutes/processed/used14k-reactions.tsv.gz tests/data/external/paroutes/processed/raw68k-templates.tsv.gz tests/data/external/paroutes/processed/uspto_unique_templates.csv.gz
	$(call run_route_replay,metal,$(ROUTE_REPLAY_PRECURSOR_GPU_SIZES))
	$(call run_route_replay,metal-prefilter,$(ROUTE_REPLAY_PRECURSOR_GPU_SIZES))

bench-route-replay-precursor: bench-route-replay-precursor-rdkit

# Heuristic AO* search, rdkit backend. Compares the uninformed (admissible) baseline against
# similarity-to-stock on the same chemistry; the reported number is molecules expanded.
AOSTAR_TARGETS ?= 5
AOSTAR_HEURISTICS ?= zero,stock
AOSTAR_WEIGHT ?= 1.0
AOSTAR_MAX_DEPTH ?= 6
AOSTAR_MAX_EXPANSIONS ?= 500
AOSTAR_CUDA_TARGETS ?= 1
AOSTAR_CUDA_WIDTH ?= 32
AOSTAR_CUDA_PASSES ?= 8
AOSTAR_CUDA_EFFORTS ?=
AOSTAR_CUDA_RULE_SET ?= used14k

bench-aostar: tests/data/external/paroutes/processed/used14k-reactions.tsv.gz
	stamp=$$(date -u +%Y%m%d-%H%M%S); mkdir -p $(BENCH_DIR)/aostar && $(BENCH_RUN_ENV) BENCH_AOSTAR_TARGETS="$(AOSTAR_TARGETS)" BENCH_AOSTAR_HEURISTICS="$(AOSTAR_HEURISTICS)" BENCH_AOSTAR_WEIGHT="$(AOSTAR_WEIGHT)" BENCH_AOSTAR_MAX_DEPTH="$(AOSTAR_MAX_DEPTH)" BENCH_AOSTAR_MAX_EXPANSIONS="$(AOSTAR_MAX_EXPANSIONS)" $(BENCH_PYTEST) tests/vertsy/bench_aostar.py::test_bench_aostar_n1 -v --benchmark-json=$(BENCH_DIR)/aostar/$(BENCH_HOST)__$$stamp.json $(EXTRA_ARGS)

bench-aostar-cuda: tests/data/external/paroutes/processed/used14k-reactions.tsv.gz tests/data/external/paroutes/processed/raw68k-templates.tsv.gz
	stamp=$$(date -u +%Y%m%d-%H%M%S); mkdir -p $(BENCH_DIR)/aostar/cuda-hybrid-prefilter && $(BENCH_RUN_ENV) BENCH_ROUNDS=1 AOSTAR_CUDA_TARGETS="$(AOSTAR_CUDA_TARGETS)" AOSTAR_CUDA_WIDTH="$(AOSTAR_CUDA_WIDTH)" AOSTAR_CUDA_PASSES="$(AOSTAR_CUDA_PASSES)" AOSTAR_CUDA_EFFORTS="$(AOSTAR_CUDA_EFFORTS)" AOSTAR_CUDA_RULE_SET="$(AOSTAR_CUDA_RULE_SET)" $(BENCH_PYTEST) tests/vertsy/bench_cuda_aostar.py::test_bench_cuda_aostar_n1 -v -s --benchmark-json=$(BENCH_DIR)/aostar/cuda-hybrid-prefilter/$(BENCH_HOST)__$$stamp.json $(EXTRA_ARGS)

# Heuristic AO* on one host, Metal then rdkit over the same targets, stock and rules. Targets
# come from a SMILES file rather than a PaRoutes sample, so this answers "can this machine solve
# these molecules and how fast", which the corpus benchmarks above deliberately do not.
# AOSTAR_HOST_STOCK_LIMIT truncates the catalogue: the heuristic is exact against whatever it is
# given and costs O(stock) per query, so a million-row stock wants AOSTAR_HOST_SCAN=metal.
# `cpu` is vesma end to end on the host and the only arm the forward probe runs in full.
AOSTAR_HOST_BACKENDS ?= metal,rdkit
# Comma-separated, and entries may be registry names or paths: raw68k, raw68k-curated,
# used14k, hartenfeller, or any .csv/.tsv (gzipped or not) with a `retro_template` column.
# Order is precedence -- the sets are concatenated and then deduplicated first-wins, so the
# better-labelled or better-curated set goes first.
AOSTAR_HOST_RULE_SET ?= raw68k
# Comma-separated .smarts or .smarts.gz files, one retro SMARTS per line, appended to the rule set.
AOSTAR_HOST_EXTRA_TEMPLATES ?=
# Comma-separated too: registry names (enamine, n1) or paths. Unioned, order-preserving,
# and AOSTAR_HOST_STOCK_LIMIT applies to the union.
AOSTAR_HOST_STOCK ?= enamine
AOSTAR_HOST_STOCK_LIMIT ?=
AOSTAR_HOST_TARGETS_FILE ?=
AOSTAR_HOST_TARGET_LIMIT ?= 1
AOSTAR_HOST_TARGET_OFFSET ?= 0
# Fixed widths (16x8) or linear widths by iteration (4nx8 = 4,8,...,32).
AOSTAR_HOST_EFFORTS ?= 64x16
# `atoms` since 2026-09-29: the estimate scaled by molecule size, so it is a distance in atoms
# rather than in [0, 1]. Same reach as `stock` on both benchmarks, half the molecules interned,
# about 20% faster. `stock` and `zero` remain for comparison; `unexplained` is the old name.
AOSTAR_HOST_HEURISTIC ?= atoms
# Where the nearest-stock popcounts run: metal (~94x) or host. Empty picks metal where a GPU is
# reachable, and the same numbers on numpy where one is not.
AOSTAR_HOST_SCAN ?=
AOSTAR_HOST_WEIGHT ?= 1.0
# Sequential by default: the caches below are built once and the cores are better spent elsewhere.
AOSTAR_HOST_JOBS ?= 1
# One JSON line per finished target, fsynced as it goes, so an interrupted sweep keeps what it did.
# Empty puts it beside the run's --benchmark-json with the same stamp.
AOSTAR_HOST_RECORDS_FILE ?=
# 1 runs the forward selectivity probe: every disconnection's chemistry is run forwards and its
# distinct products counted, and a step making several is priced as several steps. **On by default
# since 2026-09-29** -- empty defers to the bench, which now defaults it on -- so the default run
# carries every measurement this repository can make. Set 0 for a run without it.
# Requires AOSTAR_HOST_BACKENDS=cpu or rdkit -- the probe runs over molecules, and the Metal batch
# is packed arrays. Prefer `cpu`: RDKit's RunReactants takes one molecule per reactant template
# component, so its arm runs only the 10 unimolecular forward rules and no competitor channel, while
# vesma's matcher runs all 44. Each run prints, and records, how much of its rule set is covered.
AOSTAR_HOST_FORWARD ?=
# How many steps a step is worth, per measurement channel. `steps` is n+1, the shape whose threshold
# clears a three-step protection detour at n=3; `flat:M` charges any mixture M steps; `none` measures
# and records the channel without pricing it, which separates the probe's cost from its effect.
#
# `sites` is the intended reaction happening in more than one place -- a mixture to separate.
# `competitors` is a different reaction the same starting materials could undergo instead, which may
# mean the step does not work at all. It runs only multi-component forward rules, because a rule
# needing two starting materials can fire only if both are in the pot, and they are; a unimolecular
# rule matching elsewhere needs its own reagent and would measure molecular size instead.
AOSTAR_HOST_PENALTY_SITES ?= steps
AOSTAR_HOST_PENALTY_COMPETITORS ?= none

bench-aostar-host: tests/data/external/paroutes/processed/raw68k-templates.tsv.gz
	stamp=$$(date -u +%Y%m%d-%H%M%S); records="$(AOSTAR_HOST_RECORDS_FILE)"; records=$${records:-$(BENCH_DIR)/aostar/host/$(BENCH_HOST)__$$stamp.jsonl}; mkdir -p $(BENCH_DIR)/aostar/host && $(BENCH_RUN_ENV) BENCH_ROUNDS=1 AOSTAR_HOST_RECORDS_FILE="$$records" AOSTAR_HOST_BACKENDS="$(AOSTAR_HOST_BACKENDS)" AOSTAR_HOST_RULE_SET="$(AOSTAR_HOST_RULE_SET)" AOSTAR_HOST_EXTRA_TEMPLATES="$(AOSTAR_HOST_EXTRA_TEMPLATES)" AOSTAR_HOST_STOCK="$(AOSTAR_HOST_STOCK)" AOSTAR_HOST_STOCK_LIMIT="$(AOSTAR_HOST_STOCK_LIMIT)" AOSTAR_HOST_TARGETS_FILE="$(AOSTAR_HOST_TARGETS_FILE)" AOSTAR_HOST_TARGET_LIMIT="$(AOSTAR_HOST_TARGET_LIMIT)" AOSTAR_HOST_TARGET_OFFSET="$(AOSTAR_HOST_TARGET_OFFSET)" AOSTAR_HOST_EFFORTS="$(AOSTAR_HOST_EFFORTS)" AOSTAR_HOST_HEURISTIC="$(AOSTAR_HOST_HEURISTIC)" AOSTAR_HOST_SCAN="$(AOSTAR_HOST_SCAN)" AOSTAR_HOST_WEIGHT="$(AOSTAR_HOST_WEIGHT)" AOSTAR_HOST_JOBS="$(AOSTAR_HOST_JOBS)" AOSTAR_HOST_FORWARD="$(AOSTAR_HOST_FORWARD)" AOSTAR_HOST_PENALTY_SITES="$(AOSTAR_HOST_PENALTY_SITES)" AOSTAR_HOST_PENALTY_COMPETITORS="$(AOSTAR_HOST_PENALTY_COMPETITORS)" AOSTAR_HOST_PENALTY_SILENT_SITES="$(AOSTAR_HOST_PENALTY_SILENT_SITES)" $(BENCH_PYTEST) tests/vertsy/bench_host_aostar.py -v -s --benchmark-json=$(BENCH_DIR)/aostar/host/$(BENCH_HOST)__$$stamp.json $(EXTRA_ARGS)

# Which targets a curated collection cannot reach, and at what effort. Escalates 8x8 -> 16x16
# -> 64x16, each tier searching only what the one before it left unsolved, and writes the
# unsolved tail -- the gap list, which is the output worth having.
#
# Deliberately NOT a route benchmark: no extraction, no route counting, no replay, no forward
# probe, and the search stops at the first solution. That is ~30x faster per target and is why a
# 9,999-target sweep is hours rather than days. Its timings are not comparable with bench-metal.
#
# Single-core by construction (GAPS_JOBS is not offered); the GPU does the matching, so the
# other cores stay free. Templates and stock are built once and stay resident across targets.
GAPS_BACKEND ?= metal
GAPS_RULE_SET ?= hartenfeller,medchem,protecting-groups,carbohydrate,selenium
GAPS_STOCK ?= n1,common-reagents
GAPS_TARGETS_FILE ?= tests/data/external/paroutes/processed/n1-targets.smi.gz
GAPS_EFFORTS ?= 8x8,16x16,64x16
GAPS_TARGET_LIMIT ?=
GAPS_TARGET_OFFSET ?=
GAPS_OUT_DIR ?= docs/paper2/benchmarks/template-gaps
# Names the run directory. Set it per tier when running the ladder one tier at a time,
# feeding each tier's unsolved.smi in as the next tier's GAPS_TARGETS_FILE.
GAPS_STAMP ?=
bench-template-gaps:
	$(BENCH_RUN_ENV) GAPS_BACKEND="$(GAPS_BACKEND)" GAPS_RULE_SET="$(GAPS_RULE_SET)" GAPS_STOCK="$(GAPS_STOCK)" GAPS_TARGETS_FILE="$(GAPS_TARGETS_FILE)" GAPS_EFFORTS="$(GAPS_EFFORTS)" GAPS_TARGET_LIMIT="$(GAPS_TARGET_LIMIT)" GAPS_TARGET_OFFSET="$(GAPS_TARGET_OFFSET)" GAPS_OUT_DIR="$(GAPS_OUT_DIR)" GAPS_STAMP="$(GAPS_STAMP)" $(BENCH_PYTEST) tests/vertsy/bench_template_gaps.py -q -s $(EXTRA_ARGS)

# targets -> search -> replay -> PDF, from parameters alone. Computes the run directory once
# and hands it to both stages, so the chain is one command and the directory name is never typed.
#
#   make bench-route-pipeline \
#     AOSTAR_HOST_TARGETS_FILE=~/targets.smi \
#     AOSTAR_HOST_RULE_SET=raw68k-curated,hartenfeller \
#     AOSTAR_HOST_STOCK=enamine
#
# Both stages are still separate pytest modules, so a finished run can be re-rendered without
# re-searching: point HOST_ROUTE_RUN_DIR at it and run bench_route_reports alone.
# Into output/, not the default docs/paper/benchmarks: this target's artifacts are a working
# run -- a multi-megabyte PDF, route archives, a JSONL of every record -- where that directory
# holds the small committed numbers a paper cites. /output/ is gitignored.
bench-route-pipeline: BENCH_DIR = output/benchmarks
bench-route-pipeline: tests/data/external/paroutes/processed/raw68k-templates.tsv.gz
	stamp=$$(date -u +%Y%m%d-%H%M%S); run="$(BENCH_DIR)/aostar/host/$(BENCH_HOST)__$$stamp"; mkdir -p "$$run" && $(BENCH_RUN_ENV) BENCH_ROUNDS=1 AOSTAR_HOST_RUN_DIR="$$run" AOSTAR_HOST_BACKENDS="$(AOSTAR_HOST_BACKENDS)" AOSTAR_HOST_RULE_SET="$(AOSTAR_HOST_RULE_SET)" AOSTAR_HOST_EXTRA_TEMPLATES="$(AOSTAR_HOST_EXTRA_TEMPLATES)" AOSTAR_HOST_STOCK="$(AOSTAR_HOST_STOCK)" AOSTAR_HOST_STOCK_LIMIT="$(AOSTAR_HOST_STOCK_LIMIT)" AOSTAR_HOST_TARGETS_FILE="$(AOSTAR_HOST_TARGETS_FILE)" AOSTAR_HOST_TARGET_LIMIT="$(AOSTAR_HOST_TARGET_LIMIT)" AOSTAR_HOST_TARGET_OFFSET="$(AOSTAR_HOST_TARGET_OFFSET)" AOSTAR_HOST_EFFORTS="$(AOSTAR_HOST_EFFORTS)" AOSTAR_HOST_HEURISTIC="$(AOSTAR_HOST_HEURISTIC)" AOSTAR_HOST_SCAN="$(AOSTAR_HOST_SCAN)" AOSTAR_HOST_WEIGHT="$(AOSTAR_HOST_WEIGHT)" AOSTAR_HOST_JOBS="$(AOSTAR_HOST_JOBS)" AOSTAR_HOST_PREPROCESS="$(AOSTAR_HOST_PREPROCESS)" AOSTAR_HOST_ROUTE_CAP="$(AOSTAR_HOST_ROUTE_CAP)" AOSTAR_HOST_REPLAY_CAP="$(AOSTAR_HOST_REPLAY_CAP)" $(BENCH_PYTEST) tests/vertsy/bench_host_aostar.py -q -s $(EXTRA_ARGS) && echo "Generating report PDF" && $(BENCH_RUN_ENV) HOST_ROUTE_RUN_DIR="$$run" $(BENCH_PYTEST) tests/vertsy/bench_route_reports.py::test_saved_host_route_reports -q -s && echo "report: $$run/report.pdf"

# Off by default: `vertsy.preprocess` needs the optional rdkit extra, and most targets need
# nothing done to them. Set to 1 to flag duplicates and run the free-base probe.
AOSTAR_HOST_PREPROCESS ?=
# Routes are ranked by utility, so a smaller cap keeps the best and drops the tail.
AOSTAR_HOST_ROUTE_CAP ?=
# Distinct replayed routes kept per selected route; >1 surfaces genuine application multiplicity.
AOSTAR_HOST_REPLAY_CAP ?=

# Controlled hard-target selector experiment, with data and routes outside the repo.
FRONTIER_EXPERIMENT_DIR ?=
FRONTIER_STOCK_CACHE ?=
FRONTIER_EFFORTS ?= 32x8
bench-frontier-selection:
	$(BENCH_RUN_ENV) FRONTIER_EXPERIMENT_DIR="$(FRONTIER_EXPERIMENT_DIR)" FRONTIER_STOCK_CACHE="$(FRONTIER_STOCK_CACHE)" FRONTIER_EFFORTS="$(FRONTIER_EFFORTS)" $(BENCH_PYTEST) tests/vertsy/bench_bottleneck_selection.py -v -s $(EXTRA_ARGS)

# Focused larger-budget probes for whichever targets the campaign directory holds.
bench-remaining-targets:
	$(BENCH_RUN_ENV) FRONTIER_EXPERIMENT_DIR="$(FRONTIER_EXPERIMENT_DIR)" FRONTIER_STOCK_CACHE="$(FRONTIER_STOCK_CACHE)" $(BENCH_PYTEST) tests/vertsy/bench_remaining_targets.py -v -s $(EXTRA_ARGS)

# One expansion (match, construct, key, intern, fingerprint, scan) from representative
# 1/8/64-molecule frontiers. Set STOCK_LIMIT=0 to profile the full cached catalogue.
METAL_PIPELINE_TARGETS_FILE ?= $(AOSTAR_HOST_TARGETS_FILE)
METAL_PIPELINE_STOCK_LIMIT ?= 50000
METAL_PIPELINE_ROUNDS ?= 3
METAL_PIPELINE_ASSIGNMENT_PROBE_CAP ?= 8
bench-metal-pipeline:
	$(BENCH_RUN_ENV) METAL_PIPELINE_TARGETS_FILE="$(METAL_PIPELINE_TARGETS_FILE)" METAL_PIPELINE_STOCK_LIMIT="$(METAL_PIPELINE_STOCK_LIMIT)" METAL_PIPELINE_ROUNDS="$(METAL_PIPELINE_ROUNDS)" METAL_PIPELINE_ASSIGNMENT_PROBE_CAP="$(METAL_PIPELINE_ASSIGNMENT_PROBE_CAP)" $(BENCH_PYTEST) tests/vertsy/bench_metal_pipeline.py -k 'not assignment_capacity' -v -s $(EXTRA_ARGS)

# Does Metal reaction application hold its throughput as the workload grows? Four legs over the
# same sizes. `accumulate` and `stream` isolate delivery: one joined array against pulling
# `match_batches` and dropping each piece. `keyed` and `folded` are the A/B that matters, both
# ending with a keyed batch: joining then keying it, against folding identity in per piece with
# `keyed_match_folding` so only distinct molecules are ever held. Device work is identical
# throughout, so every difference is host retention -- which is what used to bend the scaling
# curve down past ~100 molecules. All legs carry the matcher's phase diagnostics and the
# machine's page-compression counters; `folded` also records its dedup ratio. Set STREAMING_LEGS
# to a subset. Diagnostic records, so they land in output/ rather than the paper tree.
STREAMING_ROUNDS ?= 3
STREAMING_LEGS ?= accumulate,stream,keyed,folded
bench-metal-precursor-streaming: tests/data/external/paroutes/processed/n1-d2-f0.smi.gz tests/data/external/remix/paroutes-n1-d2-match-uspto42k/processed/n1-d2-f0-uspto42k-compat-matches.json.gz
	$(BENCH_RUN_ENV) STREAMING_ROUNDS="$(STREAMING_ROUNDS)" STREAMING_LEGS="$(STREAMING_LEGS)" $(BENCH_PYTEST) tests/vesma/bench_metal_precursor_streaming.py -v -s -p no:randomly $(EXTRA_ARGS)

bench-metal-assignment-capacity:
	$(BENCH_RUN_ENV) METAL_PIPELINE_TARGETS_FILE="$(METAL_PIPELINE_TARGETS_FILE)" METAL_PIPELINE_STOCK_LIMIT="$(METAL_PIPELINE_STOCK_LIMIT)" METAL_PIPELINE_ROUNDS="$(METAL_PIPELINE_ROUNDS)" $(BENCH_PYTEST) tests/vertsy/bench_metal_pipeline.py -k assignment_capacity -v -s $(EXTRA_ARGS)

METAL_RETRO_PHASE_STOCK_LIMIT ?= 1000
bench-metal-retro-phases:
	$(BENCH_RUN_ENV) METAL_RETRO_PROFILE_WL="$(METAL_RETRO_PROFILE_WL)" METAL_PIPELINE_SOURCE=targets-stock METAL_PIPELINE_TARGETS_FILE="$(METAL_PIPELINE_TARGETS_FILE)" METAL_PIPELINE_STOCK_LIMIT="$(METAL_RETRO_PHASE_STOCK_LIMIT)" METAL_PIPELINE_ROUNDS="$(METAL_PIPELINE_ROUNDS)" $(BENCH_PYTEST) tests/vertsy/bench_metal_retro_phases.py -v -s $(EXTRA_ARGS)

bench-metal-match-grain:
	$(BENCH_RUN_ENV) METAL_PIPELINE_RULE_SET=used14k METAL_PIPELINE_SOURCE=stock-only METAL_PIPELINE_TARGETS_FILE="$(METAL_PIPELINE_TARGETS_FILE)" METAL_PIPELINE_STOCK_LIMIT=1000 METAL_PIPELINE_ROUNDS="$(METAL_PIPELINE_ROUNDS)" $(BENCH_PYTEST) tests/vertsy/bench_metal_match_grain.py -v -s $(EXTRA_ARGS)

bench-route-recovery-rdkit: tests/data/external/paroutes/processed/used14k-reactions.tsv.gz
	$(call run_route_recovery,rdkit,$(ROUTE_RECOVERY_TARGETS))

bench-route-recovery-cpu: tests/data/external/paroutes/processed/used14k-reactions.tsv.gz
	$(call run_route_recovery,cpu,$(ROUTE_RECOVERY_TARGETS))

bench-route-recovery-cuda: tests/data/external/paroutes/processed/used14k-reactions.tsv.gz
	$(call run_route_recovery,cuda-host,$(ROUTE_RECOVERY_TARGETS))
	$(call run_route_recovery,cuda-host-prefilter,$(ROUTE_RECOVERY_TARGETS))
	$(call run_route_recovery,cuda-device,$(ROUTE_RECOVERY_TARGETS))
	$(call run_route_recovery,cuda-device-prefilter,$(ROUTE_RECOVERY_TARGETS))

bench-route-recovery-cuda-matrix: ROUTE_RECOVERY_TARGETS = 10
bench-route-recovery-cuda-matrix: ROUTE_RECOVERY_RULE_SETS = used14k raw68k uspto42k
bench-route-recovery-cuda-matrix: ROUTE_RECOVERY_ROUNDS = 1
bench-route-recovery-cuda-matrix: BENCH_WARMUP_ROUNDS = 0
bench-route-recovery-cuda-matrix: tests/data/external/paroutes/processed/raw68k-templates.tsv.gz tests/data/external/paroutes/processed/uspto_unique_templates.csv.gz
bench-route-recovery-cuda-matrix: bench-route-recovery-cuda

bench-route-recovery-used14k-full: ROUTE_RECOVERY_TARGETS = 2881
bench-route-recovery-used14k-full: ROUTE_RECOVERY_RULE_SETS = used14k
bench-route-recovery-used14k-full: ROUTE_RECOVERY_ROUNDS = 1
bench-route-recovery-used14k-full: BENCH_WARMUP_ROUNDS = 0
bench-route-recovery-used14k-full: tests/data/external/paroutes/processed/used14k-reactions.tsv.gz
	$(call run_route_recovery,cuda-device-prefilter,$(ROUTE_RECOVERY_TARGETS))

# Arms ALTERNATED, not one whole sweep at a time. Running each arm's sweep in turn reported the
# folded delivery 19% slower than the joined one; alternating the same two arms reversed the sign
# to 8.8% faster, twice. The same arm measured 49.74 s running second of four and 55.93-57.87 s in
# a clean alternating pair -- a 16% position effect, larger than most of what gets compared here.
# Each arm is still its own process, so peak RSS stays attributable to it.
#
# ROUTE_RECOVERY_PASSES is the number of alternating passes over the arm list, and it is what
# makes a comparison between arms sound. It is not BENCH_ROUNDS, which repeats inside one process
# and therefore cannot see across processes: the scaling sweep ran three rounds and a warm-up and
# still mismeasured a delivery by running it third of three processes. One pass records the arms;
# a claim about a difference between them wants two or more.
ROUTE_RECOVERY_PASSES ?= 1
ROUTE_RECOVERY_METAL_ARMS ?= metal metal-prefilter metal-joined metal-prefilter-joined
# The `-joined` arms are the pre-fold delivery, kept as the fold's baseline rather than as a row
# to publish, so their records go here instead of into the paper tree -- `si.qmd` groups whatever
# it finds under $(BENCH_DIR)/route-recovery, so a record landing there is a published table row.
# BENCH_DIR is the existing knob for a diagnostic record (see bench-rdchiral), set per arm below.
ROUTE_RECOVERY_ABLATION_DIR ?= output/benchmarks

bench-route-recovery-metal: tests/data/external/paroutes/processed/used14k-reactions.tsv.gz
	for pass in $$(seq 1 $(ROUTE_RECOVERY_PASSES)); do \
	  for arm in $(ROUTE_RECOVERY_METAL_ARMS); do \
	    case "$$arm" in \
	      *-joined) out="$(ROUTE_RECOVERY_ABLATION_DIR)";; \
	      *) out="$(BENCH_DIR)";; \
	    esac; \
	    echo "== route-recovery pass $$pass arm $$arm -> $$out =="; \
	    $(MAKE) --no-print-directory _route_recovery_one ARM="$$arm" BENCH_DIR="$$out" \
	      BENCH_HOST="$(BENCH_HOST)" BENCH_ROUNDS="$(BENCH_ROUNDS)" \
	      BENCH_WARMUP_ROUNDS="$(BENCH_WARMUP_ROUNDS)" \
	      ROUTE_RECOVERY_TARGETS="$(ROUTE_RECOVERY_TARGETS)" \
	      ROUTE_RECOVERY_ROUNDS="$(ROUTE_RECOVERY_ROUNDS)" \
	      ROUTE_RECOVERY_RULE_SETS="$(ROUTE_RECOVERY_RULE_SETS)" || exit 1; \
	  done; \
	done

# One arm, one process. Split out so the alternating loop above can invoke it per arm; not meant
# to be called directly, which is why ARM has no default.
#
# The loop forwards every variable it depends on explicitly, because a sub-make inherits only the
# environment and the command line: a target-specific override on `bench-route-recovery-metal` --
# the idiom its siblings use (`bench-route-recovery-cuda-matrix: ROUTE_RECOVERY_TARGETS = 10`) --
# would otherwise be dropped here and the run would quietly measure the default workload instead
# of failing. Anything added to `run_route_recovery` wants adding to that list too.
_route_recovery_one: BENCH_RUN_ENV += BENCH_PROBE_METAL=1
_route_recovery_one:
	@test -n "$(ARM)" || { echo "_route_recovery_one needs ARM=<backend>"; exit 2; }
	$(call run_route_recovery,$(ARM),$(ROUTE_RECOVERY_TARGETS))

bench-route-recovery: bench-route-recovery-rdkit

# uspto42k rule set (standard uspto_unique_templates): reaction-agnostic comparators + real
# timing. Uses the committed compatible-template subset built from uspto_unique_templates.
bench-route-recovery-uspto42k-rdkit: ROUTE_RECOVERY_RULE_SETS = uspto42k
bench-route-recovery-uspto42k-rdkit: tests/data/external/paroutes/processed/uspto_unique_templates.csv.gz
	$(call run_route_recovery,rdkit,$(ROUTE_RECOVERY_TARGETS))

bench-route-recovery-uspto42k-cpu: ROUTE_RECOVERY_RULE_SETS = uspto42k
bench-route-recovery-uspto42k-cpu: tests/data/external/paroutes/processed/uspto_unique_templates.csv.gz
	$(call run_route_recovery,cpu,$(ROUTE_RECOVERY_TARGETS))

bench-route-recovery-uspto42k: bench-route-recovery-uspto42k-rdkit

# --------------------------------------------------------------------------- #
# Fanout-per-depth: how the frontier branches at each retro depth, per rule set.
# cuda-host is the intended (RunPod) path; rdkit is a shallow local smoke only.
# --------------------------------------------------------------------------- #
bench-fanout-depth-rdkit: tests/data/external/paroutes/processed/used14k-reactions.tsv.gz
	$(call run_fanout_depth,rdkit)

bench-fanout-depth-cuda: tests/data/external/paroutes/processed/used14k-reactions.tsv.gz
	$(call run_fanout_depth,cuda-host)

bench-fanout-depth-metal: BENCH_RUN_ENV += BENCH_PROBE_METAL=1
bench-fanout-depth-metal: tests/data/external/paroutes/processed/used14k-reactions.tsv.gz
	$(call run_fanout_depth,metal)

# Default deep-fanout run targets the GPU (cuda-host).
bench-fanout-depth: bench-fanout-depth-cuda

# --------------------------------------------------------------------------- #
# Paper
# --------------------------------------------------------------------------- #
PAPER_FORMAT ?= all
paper-render:
	HOME="$(QUARTO_HOME)" QUARTO_PYTHON="$(PWD)/$(PYTHON)" /usr/local/bin/quarto render docs/paper/paper/index.qmd --to $(PAPER_FORMAT) --execute

paper-si-render:
	HOME="$(QUARTO_HOME)" QUARTO_PYTHON="$(PWD)/$(PYTHON)" /usr/local/bin/quarto render docs/paper/paper/si.qmd --to $(PAPER_FORMAT) --execute

paper-approval-bundle: paper-render paper-si-render
	$(PYTHON) docs/paper/package_approval.py

paper-release-check:
	$(PYTHON) docs/paper/check_release_sentinels.py

# Canonical paper workload: streamed, key-free host delivery.
# One backend/size per process; parent RSS includes setup, not workers.
APPLICATION_DIR ?= docs/paper/benchmarks/reaction-application
APPLICATION_SIZES ?= 1 10 100 1000 2881
APPLICATION_CPU_SERIAL_SIZES ?= 1 10 100
APPLICATION_PAIRS_PER_BATCH ?= 4096

define run_application
for size in $(2); do \
  stamp=$$(date -u +%Y%m%d-%H%M%S); \
  mkdir -p $(APPLICATION_DIR)/$(1); \
  $(BENCH_RUN_ENV) APPLICATION_BACKENDS="$(1)" APPLICATION_SIZES="$$size" APPLICATION_PAIRS_PER_BATCH=$(APPLICATION_PAIRS_PER_BATCH) $(BENCH_PYTEST) tests/vesma/bench_reaction_application.py -v --benchmark-json=$(APPLICATION_DIR)/$(1)/$(BENCH_HOST)__$$size-$$stamp.json $(EXTRA_ARGS) || exit $$?; \
done
endef

.PHONY: bench-application-cpu bench-application-rdkit bench-application-metal bench-application-cuda
bench-application-cpu bench-application-rdkit bench-application-metal bench-application-cuda: tests/data/external/paroutes/processed/n1-d2-f0.smi.gz tests/data/external/remix/paroutes-n1-d2-match-uspto42k/processed/n1-d2-f0-uspto42k-compat-matches.json.gz
bench-application-cpu:
	$(call run_application,cpu-serial,$(APPLICATION_CPU_SERIAL_SIZES))
	$(call run_application,cpu-parallel,$(APPLICATION_SIZES))

bench-application-rdkit:
	$(call run_application,rdkit-serial,$(APPLICATION_SIZES))
	$(call run_application,rdkit-parallel,$(APPLICATION_SIZES))

bench-application-metal: BENCH_RUN_ENV += BENCH_PROBE_METAL=1
bench-application-metal:
	$(call run_application,metal,$(APPLICATION_SIZES))

bench-application-cuda:
	$(call run_application,cuda-host,$(APPLICATION_SIZES))

# Keyed retained frontier diagnostic: distinct from streamed unkeyed host output.
FRONTIER_MODE ?= device
FRONTIER_SIZE ?= 2881
FRONTIER_PAIR_CAP ?= 0
FRONTIER_DIR ?= output/benchmarks/cuda-frontier
.PHONY: bench-cuda-frontier
bench-cuda-frontier:
	mkdir -p $(FRONTIER_DIR)
	$(BENCH_RUN_ENV) FRONTIER_MODE=$(FRONTIER_MODE) FRONTIER_SIZE=$(FRONTIER_SIZE) FRONTIER_PAIR_CAP=$(FRONTIER_PAIR_CAP) $(BENCH_PYTEST) tests/vesma/perf_cuda_frontier_residency.py -v --benchmark-json=$(FRONTIER_DIR)/$(BENCH_HOST)__$(FRONTIER_MODE)-$(FRONTIER_PAIR_CAP)-$$(date -u +%Y%m%d-%H%M%S).json $(EXTRA_ARGS)
