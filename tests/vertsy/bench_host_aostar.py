"""AO* retrosynthesis on one host, over Metal and RDKit, against an arbitrary target list.

The counterpart of ``tests/vertsy/bench_cuda_aostar.py`` for a machine with no CUDA. Two things
differ and both are the point:

- **Backends run one after another** over the same targets, stock and rule library, so the
  wall-clock difference is the reaction matcher and its heuristic rather than the chemistry.
  Each arm keeps its *native* molecular identity — the Metal arm keys precursors with
  ``short_ring_wl_keyer``, the RDKit arm with canonical SMILES — so the two searches build
  different graphs and the timings are a throughput comparison, not a like-for-like one. That is
  stated rather than fixed: converting between the two per batch would time the converter.
- **Metal fingerprints generated precursors on its GPU; RDKit fingerprints on the host.**
  Both arms score distance to stock with :mod:`vertsy.host_similarity` over packed bits, exact
  against the whole catalogue either way. Which scan is the difference between a catalogue-scale
  stock being affordable and not: ``metal`` is ~94x the numpy path, so a full Enamine stock costs
  minutes per target where ``host`` costs hours. Below ~100k rows it does not matter; see
  ``AOSTAR_HOST_STOCK_LIMIT``.

Targets come from a file rather than a seeded PaRoutes sample, because the question here is
"can this host solve *these* molecules", not "how does the search score on a corpus".

``AOSTAR_HOST_RECORDS_FILE`` makes that survivable: one JSON line per finished target, fsynced
before the next one starts, so a sweep killed at target 19 keeps nineteen results. The
``--benchmark-json`` report is still written at the end and is still the provenance record; the JSONL
is the one you can read while the run is going.

Configure with ``AOSTAR_HOST_*`` and run with ``make bench-aostar-host``.
``AOSTAR_HOST_EXTRA_TEMPLATES`` accepts comma-separated plain or gzipped files with
one retro reaction SMARTS per line. These rules are appended to the selected library
before the usual mass-loss lint pass.

``AOSTAR_HOST_VETOES`` is ``off``, ``measure`` or ``refuse``. ``measure`` computes the verdict
column and acts on nothing, so a hand-written row states how often it *would* fire over a real
sweep before it earns the right to remove anything; ``refuse`` enforces it and changes which routes
exist. It runs on the ``cpu`` and ``metal`` arms, which can both match a pattern into a pot; the
``rdkit`` arm has no screen and says so rather than measuring nothing.
"""

from __future__ import annotations

import bz2
import csv
import gzip
import hashlib
import json
import math
import os
import warnings
from collections import Counter
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

import numpy as np
import pytest

from tests.datadirs import external_data_root
from tests.vertsy.paper_provenance import source_artifact
from tests.vertsy.paroutes_reference import (
    load_route_template_rows,
    route_recovery_rule_index_from_rows,
)
from vertsy.aostar import (
    Select,
    anytime_step,
    aostar,
    bottleneck_fold,
    committed_application_leaves,
    global_leaves,
)
from vertsy.costs import (
    UNPROBED,
    Factor,
    Factors,
    Multiplier,
    Penalties,
    StepPrice,
    Substrate,
    banded,
    by_rule,
    compounded,
    constant_step,
    cost_of_factor,
    doubted,
    flat,
    graded,
    rule_price,
    selective,
    shared,
    steps_equivalent,
    uncurated,
)
from vertsy.expansions import Expansions
from vertsy.filters import (
    CLEAR,
    DeficitReport,
    Veto,
    VetoReport,
    host_screen,
    read_tags,
    read_vetoes,
    refusal_wins,
    refused,
    unaccounted,
    validated,
    vetoed,
    vetoes_in_scope,
)
from vertsy.heuristics import zero_heuristic
from vertsy.host_similarity import (
    HostNearestScan,
    NearestScan,
    cmol_stock,
    ecfp4_stock,
    packed_stock,
)
from vertsy.interner import DictInterner
from vertsy.metrics import (
    GraphState,
    precursor_route_counts_by_first_disconnection,
    precursor_route_depth_profile,
    prepare_graph,
)
from vertsy.mixture import ExpectedPeak, expected_mixture
from vertsy.replay import (
    ReplayMismatch,
    step_precursor_names,
    template_replay_routes,
)
from vertsy.route_alternatives import (
    render_report_markdown,
    report_from_json,
    route_report,
    stored_veto_validator,
)
from vertsy.route_selection import (
    RouteSelection,
    disconnections,
    select_solved_routes,
)
from vertsy.routes import (
    Route,
    SidRoute,
    collapsed_route_key,
    decode_route,
)
from vertsy.search import (
    concat_object_batches,
    expansion,
    measured,
    object_mol_array,
)
from vertsy.search import step as expansion_step
from vertsy.selectivity import (
    CentreAtom,
    Competes,
    ProbeReport,
    competitor_probe,
    conditions_allow,
    declared_molecularity,
    different_site,
    distinct_outcomes,
    forward_index,
    grouped_applications,
    object_combine,
    reaction_centre,
    same_site,
    severity_outcomes,
    site_probe,
    template_arities,
    unmeasured,
)
from vertsy.store.numpy_store import NumpyExpansionStore
from vertsy.template_lint import duplicates, lint
from vesma import (
    cmol_from_smiles,
    gmol_from_cmol,
    reaction_patterns_from_smarts,
)
from vesma.circularfp import (
    CircularFingerprintConfig,
    circular_fingerprint_words,
)
from vesma.keyer import cmol_rdkit_smiles_keyer, short_ring_wl_keyer
from vesma.loading import cached_function
from vesma.parallel import parallel_map
from vesma.rejection import ChemistryRejection

from .catalogue import Catalogue, Catalogues, merge
from .route_report_data import archive_route

pytestmark = [
    pytest.mark.perf,
    pytest.mark.external_data,
    pytest.mark.filterwarnings(
        "ignore:Setting the shape on a NumPy array has been deprecated.*:DeprecationWarning:joblib.numpy_pickle"
    ),
]

EXTERNAL = external_data_root() / "external"
PAROUTES = EXTERNAL / "paroutes" / "processed"
# Hand-written rule sets, as opposed to the vendored collections under `external`.
TEMPLATES = external_data_root() / "templates"
# The bench's default target set: a vendored PaRoutes subset, so a bare `make
# bench-aostar-host` searches something real without naming a campaign file that is not here.
DEFAULT_TARGETS = PAROUTES / "n1-d2-targets.smi.gz"

STOCK_FILES = {
    "enamine": EXTERNAL
    / "chemdots-stock"
    / "processed"
    / "chemdots_enamine_bbs.smi.bz2",
    # The same catalogue keeping its Enamine ids. Gitignored and re-fetchable: run
    # `tests/data/external/chemdots-stock/prepare.sh`, which writes both from one download.
    # Searching against either is identical -- `open_smiles` keeps the first column, so the two
    # read as the same tuple of SMILES and share one prepared-stock cache, tested. The only
    # difference is that a report can print the catalogue number a chemist orders by.
    "enamine-annotated": EXTERNAL
    / "chemdots-stock"
    / "processed"
    / "chemdots_enamine_bbs_annotated.smi.bz2",
    "n1": PAROUTES / "n1-stock.smi.gz",
    # Organoselenium reagents, each confirmed against a supplier catalogue. The Enamine
    # building blocks carry no selenium at all and the n1 stock carries six, so a selenium
    # route has nothing to terminate on without this.
    "selenium": TEMPLATES / "stock-selenium.smi",
    # Pairs with `templates-phosphorus.csv`, and switchable with it for the same reason:
    # `P 2` and `P 4` emit a dialkyl phosphite that no general catalogue carries. Diphenyl-
    # phosphine, which `P 3` wants, is not here because the n1 stock already has it.
    "phosphorus": TEMPLATES / "stock-phosphorus.smi",
    # The starting materials DESP's goal-constrained cases specify. Loaded as stock because a
    # molecule cannot end a route the search cannot buy, and the recovery question is whether a
    # solution subgraph reaches one. Re-fetchable, never vendored -- Pistachio-derived.
    "desp-pistachio-hard-sm": EXTERNAL
    / "desp"
    / "processed"
    / "pistachio_hard-starting-materials.smi",
    # The reagents every medicinal chemist has on the shelf -- methanol, methylamine,
    # iodomethane, ammonia, acetaldehyde, Boc2O, TsCl. The Enamine file is a screening-compound
    # catalogue and carries almost none of them (nothing below benzyl-sized), which kills every
    # methylation and reductive-amination AND-node on its trivial partner however good the other
    # precursor is. Buyable by the drum; absent only by the catalogue's scope.
    "common-reagents": TEMPLATES / "stock-common-reagents.smi",
    # Macrocyclic chelators. The 12-membered ring shared by two campaign targets is cyclen, a
    # catalogue item rather than something a route should close.
    "chelators": TEMPLATES / "stock-chelators.smi",
}
TEMPLATE_FILES = {
    "used14k": PAROUTES / "used14k-reactions.tsv.gz",
    "raw68k": PAROUTES / "raw68k-templates.tsv.gz",
    "hartenfeller": TEMPLATES / "templates-hartenfeller.csv",
    # raw68k filtered to the templates two independent reaction classifiers agree on: 42,737
    # of 67,557 SMARTS, but 74.6% of `library_occurence`, because what survives curation is
    # what fires often. A build product -- run `make raw68k-curated` first.
    "raw68k-curated": EXTERNAL
    / "remix"
    / "raw68k-classified"
    / "processed"
    / "raw68k-agree.tsv.gz",
    # Selenium bond formation, absent from every vendored collection. Written for a
    # target 15, a cyclic selenonium salt.
    "selenium": TEMPLATES / "templates-selenium.csv",
    # Protecting groups, retro-removal only: each fires where the group is present, so the
    # branching factor stays low. See the README for why the reverse direction is absent.
    "protecting-groups": TEMPLATES / "templates-protecting-groups.csv",
    # The standard medicinal-chemistry toolkit the vendored collections lack: SNAr with O
    # and S nucleophiles, nitro reduction, Sandmeyer and Balz-Schiemann, carbonyl reduction,
    # Buchwald C-O, azole N-arylation, and pyrimidine ring formation.
    "medchem": TEMPLATES / "templates-medchem.csv",
    # Anomeric carbon-carbon bond formation. Kept out of the medchem set on purpose: it is
    # reliable carbohydrate chemistry rather than general-purpose, it needs an anomeric leaving
    # group and neighbouring-group participation to be stereocontrolled, and written loosely it
    # would fire on any alpha-alkoxy centre a drug-like molecule happens to contain.
    "carbohydrate": TEMPLATES / "templates-carbohydrate.csv",
    # P-C bond formation and phosphonate ester hydrolysis. Separate for the reason the selenium
    # set is: the reagents are not in a general catalogue -- dialkyl phosphites and
    # chlorodiphenylphosphine are absent from both the Enamine blocks and the PaRoutes stock --
    # so these rules terminate only where their own stock is loaded too. Phosphonate drugs are
    # real medicinal chemistry (tenofovir, alendronate), but triarylphosphine synthesis is
    # ligand chemistry, and one file covering both belongs outside the always-on set.
    "phosphorus": TEMPLATES / "templates-phosphorus.csv",
}
# `cpu` is vesma end to end on the host: its matcher, its molecules, its keys, no device. It is the
# reference arm rather than a slow one, and the only arm the forward probe runs in full.
BACKENDS = ("cpu", "metal", "rdkit")

# The two rungs of `anytime_step`: dive to a first route, then relax and keep going. Only the
# ratio of a step to the heuristic's range decides anything, so these are settable -- a penalty
# that multiplies the step is worth measuring against both rungs.
GREEDY_STEP = float(os.environ.get("AOSTAR_HOST_GREEDY_STEP", "").strip() or 0.001)
RELAXED_STEP = float(os.environ.get("AOSTAR_HOST_RELAXED_STEP", "").strip() or 1.0)
HEURISTIC_WEIGHT = 1.0
# Fingerprint width, for every arm at once. One knob rather than two constants because the arms
# have to scan the same way or the wall-clock difference between them stops being the matcher --
# see `default_scan_kind`.
#
# **Narrow is faster and the limit is known.** The nearest-stock scan is 98% of the heuristic and
# the heuristic is the largest cost left in a search; its inner loop is one AND plus one popcount
# per 64-bit word, so the cost is linear in the width and 256 bits is 4 words against 2048's 32.
# What narrowing risks is the property the width was chosen for: `vertsy.similarity`'s docstring
# rejects 128 bits and Hamming together, because both "smuggle molecule size back in", and a
# heuristic that is really a size proxy re-derives what the search already knows. Re-running that
# test -- the correlation of nearest-stock distance with heavy-atom count, over 400 n1 queries
# against the n1 stock -- locates the cliff below 256 rather than at it:
#
#     bits   2048     1024      512      256      128
#     corr  -0.028   -0.030   -0.048   -0.084   -0.183
#
# against Hamming's +0.80. So 256 keeps the size-independence and 128 is where it goes, which is
# exactly the width that docstring refused. The cost at 256 is ~4 points of distance compression
# from bit collisions (mean 0.512 to 0.468), which inflates similarity without reordering by size.
# Changing this re-prepares the catalogue, which takes ~17 minutes for 988k rows and then caches:
# `cached_stock` passes `FINGERPRINT` into the key and `CircularFingerprintConfig` is frozen, so
# each width gets its own entry and both survive. The width also reaches the run manifest through
# `asdict(FINGERPRINT)`, so a recorded benchmark always says which one produced it.
#
# **That cache is keyed on the arguments and not on the code**, which is worth knowing before
# comparing two widths: the 2048 entry here still reports one `SmartsError` refusal that the
# current parser does not produce -- the 256 entry, built today, refuses 25 rows where 2048's
# refuses 26, with every chemistry reason identical. So an old entry can outlive the fix that
# changed what it holds. One row in 920,700 moves nothing measurable, but a comparison that has to
# be exact should clear `prepare_stock`'s cache first.
FINGERPRINT_BITS = int(
    os.environ.get("AOSTAR_HOST_FINGERPRINT_BITS", "").strip() or 256
)
FINGERPRINT = CircularFingerprintConfig(n_bits=FINGERPRINT_BITS, radius=2)
ECFP4_BITS = FINGERPRINT_BITS
STOCK_CACHE_VERSION = 2
# Route enumeration is a cartesian product over per-precursor options, so the report is
# capped rather than complete. Routes come out ranked by utility, so a smaller cap keeps the
# best ones and only drops the tail; `AOSTAR_HOST_ROUTE_CAP` lowers it for a readable report.
ROUTE_REPORT_CAP = int(os.environ.get("AOSTAR_HOST_ROUTE_CAP", "").strip() or 20)
# Guided replay fires the recorded rule, but one template can have several applications
# whose precursors each accept the next template. Those are genuinely different routes --
# possibly different intermediates reaching the same stock -- so a cap above 1 surfaces
# them instead of stopping at the first.
REPLAY_CAP = int(os.environ.get("AOSTAR_HOST_REPLAY_CAP", "").strip() or 1)
ROUTE_COUNT_CAP = 10**100
# What the report is allowed to consider beyond the shortest and cheapest route it found. These
# are work bounds, and they are also the reason a longer route can be invisible: a protection
# detour costs two extra reactions, so a slack of 2 admits one on a one-reaction target and
# excludes one everywhere else. Settable because tuning a route preference you cannot widen the
# candidate box for tunes nothing.
ROUTE_DEPTH_SLACK = int(
    os.environ.get("AOSTAR_HOST_ROUTE_DEPTH_SLACK", "").strip() or 2
)
# The forward rule table, and whether to run it. **On by default since 2026-09-29**, so the
# default run is the one carrying every measurement this repository knows how to make. It roughly
# doubles a target's wall clock -- the probe runs forward chemistry on every application -- and it
# changes what a route costs, so a run that switches it *off* is the one that has to say so now.
#
# Its effect on the search is currently small and the reason is written up in
# `docs/note-forward-selectivity.md`: `vertsy.costs.selective` is a multiplier on the step cost,
# and the greedy rung charges 0.001 a step, so even a sixfold penalty is 0.006 against an estimate
# in [0, 1]. It is on regardless because the coverage report and the recorded prices are worth
# having on every run, and because the channel is the only thing that makes protection thinkable.
FORWARD_TEMPLATES = TEMPLATES / "templates-forward.csv"
FORWARD_PROBE = bool(int(os.environ.get("AOSTAR_HOST_FORWARD", "").strip() or 1))
# Also measure the competitor channel split by *what fixes the competition*: a candidate wanting
# the same atom as the intended reaction, which no protection can help, against one wanting a
# different atom, which a three-step detour answers. Off by default because it is a second and a
# third probe pass; the halves partition the candidate rules, so together they cost one pass, and
# running them beside the unsplit channel costs two.
#
# Kept despite earning no price -- the same-atom half reads 2.7x where the whole channel reads 2.8x
# and carries 86% of its volume -- because the bit it carries is about what to *do*, not what to
# charge. An engine that inserts protect/react/deprotect into a route needs to know whether
# protection would help at all, and that is precisely the different-atom half. Measuring it now is
# what makes that feature possible later.
SPLIT_COMPETITORS = bool(
    int(os.environ.get("AOSTAR_HOST_SPLIT_COMPETITORS", "").strip() or 0)
)
# Also measure the competitor channel split by *how bad the competition is*: a candidate whose own
# row says `destructive` consumes the substrate, where the unsplit channel can only say that some
# other reaction was possible. On by default, unlike the site split, because it is nearly free: the
# site halves each run ~64 candidate rules per application, this one runs 6, and `candidates()` is
# empty for every intended rule supplying neither `halogenating` nor `oxidant` -- 39 of 309 paired
# codes -- so the group is skipped before a molecule is built.
#
# Measured and not priced. The undifferentiated `competitors` channel was priced, judged and
# retired on five independent lines, the decisive one being that every veto row it is enriched on
# is a row `B 23` to `B 34` already refuses. The question this asks is whether the destructive
# slice is the rare, enriched signal the whole channel was not.
SEVERITY = bool(int(os.environ.get("AOSTAR_HOST_SEVERITY", "").strip() or 1))
# `AOSTAR_HOST_SPAN` is gone. It counted only what a group's starting materials do *together*, by
# probing each of them alone and subtracting, because a two-component forward pattern matched both
# of its components inside one precursor whenever that precursor carried both groups. The forward
# rules now say it themselves -- every multi-component left-hand side is written `(A).(B)`, which
# the matcher enforces -- so the subtraction became provably empty: measured over 15,601
# applications on 20 targets, `AOSTAR_HOST_SPAN=1` and `=0` produced byte-identical
# cross-tabulations. `vertsy.template_lint.ungrouped_components` is what keeps it that way.
# Whether a competing reaction has to be one the intended reaction's medium could turn over.
# On by default: a Grignard addition needs the organometallic and a Friedel-Crafts needs the Lewis
# acid, neither of which is a precursor, so a rule whose reagent is implicit otherwise competes
# wherever its two functional groups appear. Off is for measuring what the condition classes buy.
CONDITIONS = bool(int(os.environ.get("AOSTAR_HOST_CONDITIONS", "").strip() or 1))
# Whether a *unimolecular* forward rule may compete. Off by default, and the two flags above and
# below are the same argument twice: "only multi-component rules" was a proxy for "its reagent is
# not in the pot", which `CONDITIONS` now states directly. With every forward row declaring its own
# conditions the proxy is redundant, and turning this on admits the reactions whose reagent the pot
# really does supply -- a reduction competing in a reducing pot. Measured at 1.0 gated mixture peaks
# per application against 0.5 for the proxy. It changes what every report says, so it is a choice.
UNIMOLECULAR = bool(int(os.environ.get("AOSTAR_HOST_UNIMOLECULAR", "").strip() or 0))
VETO_TEMPLATES = TEMPLATES / "templates-vetoes.csv"
# Which rules belong to a condition class, so a veto about a *catalyst* has one pattern and many
# rules. Also the first entry of the `requires` column the competes predicate wants: the two are
# the same per-rule attribute, and one curated list serves both.
VETO_TAGS = TEMPLATES / "template-tags.csv"
VETO_TAG_VOCABULARY = TEMPLATES / "template-tag-vocabulary.csv"
# What the veto table is allowed to do: nothing, measure, or remove.
#
# `off` is the default and the only setting that leaves every number comparable with the runs
# recorded in `docs/paper2/benchmarks/`. `measure` wires the verdict column and prices nothing,
# which is the composition order `vertsy.filters` argues for -- a hand-written row states how often
# it *would* fire, over a real sweep, before it earns the right to remove anything. `refuse`
# enforces it, and changes which routes exist.
VETOES = os.environ.get("AOSTAR_HOST_VETOES", "").strip() or "withhold"
#: Whether the run *asked* for that mode or inherited it from the default above.
#:
#: It decides what to do with an arm that cannot screen: the rdkit arm holds RDKit molecules and
#: there is no screen over those, so it can carry no veto table. Asking for vetoes on it is a
#: contradiction and still raises. Inheriting them from a default it never chose is not, and used
#: to raise anyway the moment `withhold` became the default -- on the Makefile's own
#: `metal,rdkit`. So an unasked-for table is skipped for that arm, and the run says it skipped.
VETOES_REQUESTED = bool(os.environ.get("AOSTAR_HOST_VETOES", "").strip())
# `validate` is option (C) in prototype: the search runs with no vetoes at all, and the same rows
# are evaluated during *replay* instead, where molecules exist again. A condemned application
# cannot be reproduced, so the route fails to replay and the next candidate is taken -- which is
# "extract, validate, take the next" with no route-exclusion machinery. The verdict is not yet
# cached per application, so a later extraction rediscovers it; that is the step that would turn
# this from a validator into a learned constraint.
# Two axes, not one: *where the verdict is computed* and *where it is acted on*. The enum
# conflates them because only four of the cells are reachable -- a verdict computed at replay
# cannot be acted on during a search that has already finished.
#
#     computed          acted on                       mode
#     --                --                             off
#     during search     nowhere                        measure
#     during search     during search (row dropped)    refuse
#     during search     at extraction (by lookup)      withhold
#     at replay         at extraction (by replaying)   validate
#
# `withhold` is the one a chemist wants and the only one that does both halves: every condemned
# application stays in the graph, so the report shows the road not taken *and* the row that
# condemned it, while no route the run recommends contains a condemned step. `refuse` cannot
# report them -- the application never reaches the store -- and `validate` cannot either, since
# that mode writes no veto columns during the search at all. What it withholds is the
# recommendation, not the chemistry.
VETO_MODES = ("off", "measure", "refuse", "validate", "withhold")
# Cross-tabulate the veto verdict against the competitor count, per veto row. Off by default
# because it costs a column per row and answers one question: whether a row condemns chemistry the
# competitor channel already finds, and so could be priced there instead of refused here.
VETO_OVERLAP = bool(int(os.environ.get("AOSTAR_HOST_VETO_OVERLAP", "").strip() or 0))
# Whether a retro application's precursors have to account for every atom of its product.
#
# A separate knob from AOSTAR_HOST_VETOES, and deliberately: it is a claim about the *library*
# rather than about chemistry, so a vendored table nobody intends to repair can be measured with
# it and searched without it. The two compose by both being named to `refused`, and neither
# wrapper mentions the other.
MASS = os.environ.get("AOSTAR_HOST_MASS", "").strip() or "off"
MASS_MODES = ("off", "measure", "refuse")
MASS_SLACK = int(os.environ.get("AOSTAR_HOST_MASS_SLACK", "").strip() or 0)
# How many steps a step is worth, per measurement channel. "steps" is `n + 1`, the shape whose
# threshold clears a three-step protection detour at n=3; "flat:M" charges any mixture M steps;
# "none" measures and records the channel without pricing it.
#
# `sites` is the intended reaction happening in more than one place -- a mixture to separate.
# `competitors` is a different reaction the same starting materials could undergo instead, which
# may mean the step does not work at all. Separate knobs because they are separate chemistry, and
# because whether the second carries signal or merely molecular size is a question to settle by
# measurement rather than assumption.
#
# **`sites` is priced and `competitors` is not, and both halves are measured.** Over 20 PaRoutes
# targets, 18 reporting, 54 routes under every shape, counting the steps a report *recommends* that
# a hand-written veto row objects to -- an independent criterion, since the rows are chemistry
# neither channel reads:
#
#     sites        competitors   condemned top   any route
#     none         none                      5          14
#     flat:1.2     none                      2           8
#     graded:0.3   none                      1           7
#     steps        none                      1           5
#     none         graded:0.1                5          17
#     steps        graded:0.1                2           5
#     steps        steps                     3           7
#
# `sites` carries signal and the response is monotone in its strength, so it is priced at the shape
# derived for it: a mixture costs `n + 1` steps, and the threshold of 3 is what lets a
# protect/react/deprotect detour win. A milder shape cannot express the actual workaround for an
# unselective step, which is to protect.
#
# `competitors` is not priced, and the reason is that the count barely distinguishes a step worth
# worrying about from any other. Taking the veto rows as independent ground truth -- they are
# hand-written chemistry neither channel reads -- over 401,321 applications on those 20 targets:
#
#     channel        fires >=2 on any application   on one a veto row condemns   enrichment
#     sites                                 15.1%                        46.9%         3.1x
#     competitors                           38.9%                        48.1%         1.2x
#
# A channel that fires on two of five applications and is 1.2x enriched where trouble is known to
# be is a near-uniform tax, and a uniform tax reshuffles routes without being about anything. That
# is why no magnitude helped: the gentlest shape tested, `graded:0.1`, charges 1.1 at n=2.
#
# **It is not noise everywhere, which is the part worth keeping.** Per row the two channels are
# complementary: on B 12 and B 13, where an electrophile lands in several places, `sites` is at 93%
# and 100% against `competitors` at 27% and 36%; on B 25 to B 27, the non-isolable species,
# `competitors` reaches 61% to 81% where `sites` sees 11% to 18% -- a bifunctional precursor is one
# many multi-component rules can attack, which is the same property that makes it unstable. So the
# channel has signal for one fault family and noise elsewhere, a flat price averages the two into a
# tax, and that family is already **refused** by B 23 to B 29. `competitor_probe`'s `competes` hook
# is the place a narrowed version would enter. See `docs/note-forward-selectivity.md`.
# Shapes to try on *the same graph*, as a comma-separated list --
# `AOSTAR_HOST_PENALTY_SWEEP=flat:1.2,graded:0.1/graded:0.2,steps/none`. An entry is one shape for
# both channels, or `<sites>/<competitors>` for one each, because the two channels are not reading
# the same kind of thing: `sites` counts the products the intended reagent makes, which is a
# product count and what `steps` was derived for, while `competitors` counts rival reactions, which
# is a warning. A sweep that could only move both together could not test that distinction.
#
# Off by default, and nearly free: extraction is a pure function of the finished graph, so N shapes
# cost N `select_solved_routes` calls at ~20 ms each against seconds of search -- and they differ by
# the price and by nothing else. *Selections*, not whole extractions: replay and the mixture report
# are not repeated, and they are what makes extraction as a whole ~28% of the search.
PENALTY_SWEEP = tuple(
    shape
    for shape in os.environ.get("AOSTAR_HOST_PENALTY_SWEEP", "").strip().split(",")
    if shape
)
# Which reading of a measurement the price uses. `yield` states a step as the fraction of material
# that survives it -- a number a chemist can argue with -- and `steps` is the step-multiple every
# measurement in `docs/note-forward-selectivity.md` was taken under, kept selectable so those stay
# reproducible.
PRICE = os.environ.get("AOSTAR_HOST_PRICE", "").strip() or "yield"
# The per-competitor share under `yield`, as `shared:<share>[:<floor>]` or `none`. Deliberately
# close to what `steps` already charges: `steps_equivalent` gives a selectivity factor of
# `0.9 ** n`, which is 0.81 / 0.73 / 0.66 at two, three and four products, against this curve's
# 0.85 / 0.72 / 0.61. Starting flat is the point -- the first thing a run should surface is where
# the yield data is thin, not a reshuffled route set.
YIELD_SITES = os.environ.get("AOSTAR_HOST_YIELD_SITES", "").strip() or "shared:0.85:0.5"
YIELD_COMPETITORS = (
    os.environ.get("AOSTAR_HOST_YIELD_COMPETITORS", "").strip() or "none"
)
# What a reaction yields when no forward row names it, which is most retro rules. **At or below the
# curated median of 0.80, and that is a measured trap rather than a style note**: `shape_of` takes
# the minimum over an application's templates, so a default kinder than the curated values makes
# every uncurated template systematically win -- the fault `vertsy.costs.uncurated` records as
# "which side this knob penalises is decided by which rules happen to be curated".
YIELD_DEFAULT = float(os.environ.get("AOSTAR_HOST_YIELD_DEFAULT", "").strip() or 0.65)
# Whether a destructive competitor is priced under `yield`. The *shares* are not a knob: they
# follow from the chemistry, and this only says whether to charge them.
#
# **The band is relative, and that is the point.** How badly a degradation hurts depends on what it
# is competing against, so the severity reduction computes `y / (y + e)` -- the intended reaction's
# own yield against the degradation's efficiency -- where both reactions are known, and
# `vertsy.costs.banded` decodes the band it writes. Two reactions competing for one substrate split
# it by their rates, and a standalone yield is the only proxy for a rate these tables have.
#
# The first version of this charged `1 - e`, a **mass balance**: it assumed the degradation ran
# first and took its share, which is the wrong shape for parallel competitors and gave 0.20, a
# charge of seven typical steps. The relative form gives 0.36 to 0.64 over the yield vocabulary --
# two to four and a half steps -- and lets a robust intended reaction resist a weak degradation
# better than a hard one does, which the absolute form could not express.
SEVERITY_PRICE = bool(
    int(os.environ.get("AOSTAR_HOST_YIELD_SEVERITY", "").strip() or 1)
)
# Descending survival thresholds: a band counts how many of these the worst survival falls below.
# Four of them for the reason `typical_yield` has four bands -- both inputs are class-level priors
# from that same vocabulary, so the thirteen distinct ratios they can produce would claim a
# resolution nobody measured. The edges split those thirteen into four non-empty groups.
SEVERITY_EDGES = (0.56, 0.49, 0.41)
# What each band keeps, as the representative survival of the ratios that land in it. Band 1 is
# clean and has to keep exactly 1.0, because a channel's `1` is the price's free reference.
SEVERITY_SURVIVALS = (1.0, 0.60, 0.52, 0.44, 0.37)
PENALTY_SITES = os.environ.get("AOSTAR_HOST_PENALTY_SITES", "").strip() or "steps"
PENALTY_COMPETITORS = (
    os.environ.get("AOSTAR_HOST_PENALTY_COMPETITORS", "").strip() or "none"
)
# What a *silent* sites probe costs: the forward chemistry paired with this disconnection ran and
# made nothing at all, which a correct pairing cannot do -- the forward rule is the retro rule's
# inverse on this application, so the intended product must be among its outcomes. So the step
# probably does not work. Probably, and hence a factor rather than a refusal: the same observation
# is produced by a retro rule proposing partners its reaction will not couple and by a forward rule
# this repo flattened too narrowly, and no product count separates them. 1.0 restores the older
# behaviour, where silence was free and a broken pairing therefore looked clean.
#
# Sites only. On `competitors`, silence is the *good* answer -- nothing else fires on this pot.
# What an application nobody measured costs. A claim about our curation coverage rather than
# about any molecule: `UNPROBED` means this rule has no forward partner, so nothing has examined
# what it does. Small values break ties toward examined chemistry; 1.0 is off and is the default.
PENALTY_UNPROBED = float(
    os.environ.get("AOSTAR_HOST_PENALTY_UNPROBED", "").strip() or 1.0
)
PENALTY_SILENT_SITES = float(
    os.environ.get("AOSTAR_HOST_PENALTY_SILENT_SITES", "").strip() or 2.0
)

# What the report considers beyond the cheapest route it found, in step-equivalents of the
# cheapest reaction the graph offers. 3 admits one protect/react/deprotect detour wherever the
# target sits, which the old integer reaction slack only managed on a one-reaction target.
ROUTE_COST_SLACK = float(
    os.environ.get("AOSTAR_HOST_ROUTE_COST_SLACK", "").strip() or 3.0
)
# The report's per-step utility. `-log 0.9` is 0.105, which is the step cost the search was
# calibrated to -- the same number seen from the other end.
ROUTE_STEP_FACTOR = float(
    os.environ.get("AOSTAR_HOST_ROUTE_STEP_FACTOR", "").strip() or 0.9
)
ROUTE_CANDIDATE_LIMIT = int(
    os.environ.get("AOSTAR_HOST_ROUTE_CANDIDATE_LIMIT", "").strip() or 400
)
ROUTE_STATE_LIMIT = int(
    os.environ.get("AOSTAR_HOST_ROUTE_STATE_LIMIT", "").strip() or 20_000
)
# How early two reported routes must part company. `None`, the default, is no floor: the order is
# already diversity-interleaved, so a floor only *removes* routes, and what it should be is a
# question for a chemist reading reports rather than a number to bake in.
ROUTE_DIVERSITY_CUTOFF = (
    int(text)
    if (text := os.environ.get("AOSTAR_HOST_ROUTE_DIVERSITY", "").strip())
    else None
)

R = TypeVar("R")


# --------------------------------------------------------------------------- configuration


def env_text(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def env_int(name: str, default: int) -> int:
    return int(env_text(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(env_text(name, str(default)))


def effort_widths(width: int | str, passes: int) -> tuple[int, ...]:
    """Resolve fixed width or ``4n`` into one width per one-based search iteration."""
    if passes <= 0:
        raise ValueError("AO* passes must be positive.")
    if isinstance(width, int):
        widths = (width,) * passes
    else:
        if not width.endswith("n"):
            raise ValueError("a widening width must end in n, e.g. 4n")
        scale = int(width[:-1])
        widths = tuple(scale * iteration for iteration in range(1, passes + 1))
    if any(value <= 0 for value in widths):
        raise ValueError("AO* widths must be positive.")
    return widths


def parse_efforts(configured: str) -> tuple[tuple[int | str, int], ...]:
    """Read fixed ``16x8`` or linearly widening ``4nx8`` effort shapes."""
    fields = [item.lower().split("x") for item in configured.split(",") if item.strip()]
    if not fields or any(len(pair) != 2 for pair in fields):
        raise ValueError("AOSTAR_HOST_EFFORTS must contain width x passes pairs.")
    efforts: list[tuple[int | str, int]] = []
    for raw_width, raw_passes in fields:
        text = raw_width.strip()
        width = f"{int(text[:-1])}n" if text.endswith("n") else int(text)
        passes = int(raw_passes)
        effort_widths(width, passes)
        efforts.append((width, passes))
    if len(set(efforts)) != len(efforts):
        raise ValueError("AOSTAR_HOST_EFFORTS must not contain duplicates.")
    return tuple(efforts)


def backends() -> tuple[str, ...]:
    chosen = tuple(
        name.strip() for name in env_text("AOSTAR_HOST_BACKENDS", "metal").split(",")
    )
    unknown = [name for name in chosen if name not in BACKENDS]
    if unknown:
        raise ValueError(
            f"AOSTAR_HOST_BACKENDS must be one of {BACKENDS}, got {unknown}."
        )
    return chosen


# --------------------------------------------------------------------------- inputs


def open_smiles(path: Path) -> Iterator[str]:
    """First whitespace-delimited column of a ``.smi``, ``.smi.gz`` or ``.smi.bz2`` file."""
    opener = {".bz2": bz2.open, ".gz": gzip.open}.get(path.suffix, open)
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            if line.strip():
                yield line.split()[0]


def read_stock_smiles(path: Path, limit: int | None) -> tuple[str, ...]:
    """The catalogue, optionally truncated.

    ``limit`` takes a prefix rather than a random sample: the point of truncating is to make the
    heuristic affordable while keeping the run reproducible, and a prefix needs no seed.
    """
    rows = open_smiles(path)
    return tuple(row for _, row in zip(range(limit or 10**9), rows))


def read_merged_stock(paths: Sequence[Path], limit: int | None) -> tuple[str, ...]:
    """Several catalogues as one, in order, without repeats.

    The limit applies to the union rather than to each file, so it still means "this many
    molecules in the heuristic" however the catalogue was assembled. Deduplicated on the SMILES
    text: two catalogues overlapping is the normal case, and a repeated row costs a fingerprint
    and a key for nothing.
    """
    seen: set[str] = set()
    merged: list[str] = []

    for path in paths:
        for smiles in read_stock_smiles(path, None):
            if smiles in seen:
                continue

            seen.add(smiles)
            merged.append(smiles)

            if limit is not None and len(merged) >= limit:
                return tuple(merged)

    return tuple(merged)


def stock_slices(
    paths: Sequence[Path], limit: int | None
) -> tuple[tuple[Path, tuple[str, ...]], ...]:
    """The same union as :func:`read_merged_stock`, kept split by the file each row came from.

    Preparing the catalogue is the expensive half of setup -- two keys and two fingerprints for
    every row -- and caching it against the *merged* tuple means adding fifty commodity reagents
    to a 988k-row Enamine file re-keys all 988k. Splitting first lets each file be cached on its
    own rows, so a new file costs only itself.

    Dedup still runs across files in order, so a row is prepared once no matter how many
    catalogues carry it, and the limit still applies to the union. Both of those make a file's
    slice depend on what precedes it: appending a file leaves the earlier slices untouched and
    their caches warm, while inserting one ahead of others invalidates what follows.
    """
    seen: set[str] = set()
    slices: list[tuple[Path, tuple[str, ...]]] = []
    remaining = limit

    for path in paths:
        taken: list[str] = []

        for smiles in read_stock_smiles(path, None):
            if remaining is not None and remaining <= 0:
                break
            if smiles in seen:
                continue

            seen.add(smiles)
            taken.append(smiles)

            if remaining is not None:
                remaining -= 1

        slices.append((path, tuple(taken)))

    return tuple(slices)


def annotated_catalogue(path: Path) -> bool:
    """Whether a catalogue writes anything after the SMILES, from its first row.

    Cheap on purpose -- one line of a possibly compressed million-row file. The question is about
    the file's *format*, and a catalogue does not annotate some rows and not others.
    """
    for _smiles, rest in read_stock_rows(path):
        return bool(rest)
    return False


def prepare_labelled_slices(
    slices: Sequence[tuple[Path, tuple[str, ...]]], n_jobs: int | None
) -> list[tuple[Path, Stock]]:
    """:func:`prepare_slices`, keeping the file each prepared part came from.

    Split out rather than folded in because the parts are what :func:`merge_stocks` wants and the
    paths are what a report wants, and one caller needs each. Both walk the same loop once.
    """
    parts: list[tuple[Path, Stock]] = []

    for path, rows in slices:
        if not rows:
            continue

        try:
            parts.append((path, cached_stock(rows, n_jobs)))
        except ValueError:
            print(f"stock: {path.name} contributed no usable row, skipped", flush=True)

    return parts


ANNOTATED_CATALOGUE_ROWS = 2_000_000
"""How large an *annotated* catalogue this re-keys for a report before giving up on its notes.

A cost guard, not the criterion -- :func:`annotated_catalogue` decides, by whether a file says
anything after the SMILES at all. The vendored ChemoDOTS and PaRoutes files are bare SMILES, so
for them there is nothing to recover and their prepared keys already say all there is to say.

The limit admits the full annotated ChemoDOTS catalogue at 988k rows, because that is what it
exists for. ``enamine-annotated`` costs one more keying pass, about the length of stock
preparation and paid once per report, against a render that takes minutes. Past the limit a run
prints that it is naming molecules by file only, rather than quietly spending longer keying the
catalogue than it spent searching.
"""


def read_stock_rows(path: Path) -> Iterator[tuple[str, str]]:
    """``(smiles, the rest of the line)`` for a catalogue, the rest joined on a single space.

    Whatever follows the SMILES is treated as opaque text. A curated file writes a name and a
    vendor id, a downloaded one writes nothing, and a third might write something else again; none
    of that is worth a schema when the only consumer prints it.
    """
    opener = {".bz2": bz2.open, ".gz": gzip.open}.get(path.suffix, open)
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            if line.strip():
                smiles, _, rest = line.strip().partition("\t")
                yield smiles.split()[0], " ".join(rest.split("\t")).strip()


def merge_stocks(parts: Sequence[Stock]) -> Stock:
    """Prepared catalogues as one, concatenated in order.

    No deduplication here: :func:`stock_slices` already removed repeats before any row was
    prepared, so the parts are disjoint by construction and concatenating them reproduces
    exactly what preparing the merged tuple would have produced.
    """
    usable = [part for part in parts if part.wl_keys]

    if not usable:
        raise ValueError("no stock row could be read by both toolkits.")

    refusals: Counter[str] = Counter()

    for part in parts:
        refusals.update(part.refusals)

    return Stock(
        wl_keys=tuple(key for part in usable for key in part.wl_keys),
        wl_fingerprints=np.concatenate([part.wl_fingerprints for part in usable]),
        rdkit_keys=tuple(key for part in usable for key in part.rdkit_keys),
        rdkit_fingerprints=np.concatenate([part.rdkit_fingerprints for part in usable]),
        row_count=sum(part.row_count for part in parts),
        refusals=dict(refusals),
    )


def catalogues_of(
    paths: Sequence[Path],
    key_of: Callable[[str], Any],
    n_jobs: int | None,
    limit: int | None = None,
) -> Catalogues:
    """Every catalogue a run searched, as one :class:`~tests.vertsy.catalogue.Catalogues`.

    Built for a *report*, after the search, which is why it can afford to keep every file's keys
    separately: a molecule three suppliers carry should be reported three times, and the merged
    index the search uses deduplicated before anything was prepared so it can only name the first.

    Preparation is the same cached, per-file work the search did -- about 11 s warm over a million
    rows, against a report that spends minutes rendering -- so nothing is keyed twice within a
    file. Annotated files are re-keyed through ``key_of`` on top of that, because their row text is
    the point and a few hundred rows cost nothing.
    """
    parts = []
    for path, prepared in prepare_labelled_slices(stock_slices(paths, limit), n_jobs):
        notes: dict[Any, tuple[str, ...]] = {}
        if annotated_catalogue(path):
            rows = tuple(read_stock_rows(path))
            if len(rows) > ANNOTATED_CATALOGUE_ROWS:
                print(
                    f"stock: {path.name} has {len(rows):,} annotated rows, past the "
                    f"{ANNOTATED_CATALOGUE_ROWS:,} this re-keys for a report; its molecules "
                    "will be named by file only",
                    flush=True,
                )
            else:
                for smiles, rest in rows:
                    try:
                        key = key_of(smiles)
                    except Exception:
                        continue
                    note = f"{path.name}: {rest}" if rest else path.name
                    if note not in notes.get(key, ()):
                        notes[key] = (*notes.get(key, ()), note)
        parts.append(
            Catalogue(
                name=path.name,
                keys=frozenset((*prepared.wl_keys, *prepared.rdkit_keys)),
                notes=notes,
            )
        )
    return merge(*parts)


def read_targets(path: Path, limit: int, offset: int) -> tuple[str, ...]:
    rows = tuple(open_smiles(path))
    return rows[offset : offset + limit]


def read_template_rows(path: Path) -> tuple[dict[str, str], ...]:
    """Every column of every row, delimiter taken from the extension.

    The whole row, not just the SMARTS, because a rule's index is the only thing that
    survives into a route and a rule set's own columns are the only place a human-readable
    name can come from. Dropping them here is what forced both the hand-written
    `rule_notes.py` and an externally produced `templates.json`.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    table_path = path.with_suffix("") if path.suffix == ".gz" else path
    delimiter = "," if table_path.suffix == ".csv" else "\t"

    with opener(path, "rt", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle, delimiter=delimiter))


def read_templates(path: Path) -> tuple[str, ...]:
    """Just the SMARTS, for callers that only need the rule set."""
    return tuple(row["retro_template"] for row in read_template_rows(path))


def parse_inputs(
    value: str, registry: Mapping[str, Path], variable: str
) -> tuple[tuple[str, Path], ...]:
    """A comma-separated list of registry names or paths, resolved in the order given.

    Names and paths are interchangeable so a shipped rule set and a one-off file can sit side by
    side in the same list. Order is meaningful for templates -- it decides which source wins a
    duplicate -- so this preserves it rather than returning a set.

    An entry that is neither a known name nor an existing path fails here naming the valid keys,
    because the alternative is a search that runs to completion against a silently smaller
    library.
    """
    resolved: list[tuple[str, Path]] = []

    for entry in (part.strip() for part in value.split(",")):
        if not entry:
            continue

        if entry in registry:
            resolved.append((entry, registry[entry]))
            continue

        path = Path(entry).expanduser()

        if not path.exists():
            raise ValueError(
                f"{variable} entry {entry!r} is neither one of {tuple(registry)} "
                f"nor an existing path"
            )

        resolved.append((path.name.split(".")[0], path))

    if not resolved:
        raise ValueError(f"{variable} is empty")

    return tuple(resolved)


def read_extra_templates(path: Path) -> tuple[str, ...]:
    """Read one retro SMARTS per line from a plain or gzipped supplemental file."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return tuple(
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        )


@dataclass(frozen=True)
class RuleSource:
    """Where one rule came from, and what its source file said about it.

    Carried parallel to `Library.smarts` and therefore indexed by the same rule id, which is
    what makes a name recoverable from a route. `columns` holds the source row minus
    `retro_template`, verbatim and unmerged: the columns differ per rule set -- raw68k has
    none, uspto42k's `template_code` is its row index and its `classification` a placeholder,
    Hartenfeller has both filled, the curated set has `rc_code` -- so normalising them here
    would mean this table asserting something no source said.
    """

    source: str
    row: int
    columns: Mapping[str, str]


def multiplier_of(shape: str) -> Multiplier:
    """One channel's penalty from its env spelling.

    `graded:<slope>` and `graded:<slope>:<cap>` are the tunable middle: `flat` reads a two-way and
    a seven-way ambiguity alike, `steps` charges a seven-way reading eight steps of synthesis, and
    neither is the right statement about a competitor count.
    """
    if shape == "steps":
        return steps_equivalent
    if shape.startswith("flat:"):
        return flat(float(shape.removeprefix("flat:")))
    if shape.startswith("graded:"):
        slope, _, cap = shape.removeprefix("graded:").partition(":")
        return graded(float(slope), cap=float(cap) if cap else math.inf)
    raise ValueError(
        f"a penalty must be 'steps', 'flat:<multiple>' or 'graded:<slope>[:<cap>]', "
        f"got {shape!r}; 'none' switches a channel off and is handled by the caller."
    )


def channel_coverage(reports: Sequence[ProbeReport]) -> dict[str, Any]:
    """One coverage block per channel, summed over the layers that channel was measured on.

    Per channel and never summed across them: `sites` and `competitors` count different things over
    different template sets, so one "probed" number covering both would mean nothing. The penalty
    each was priced at goes in with it, because coverage without the price it fed is half a fact.
    """
    shapes = {"sites": PENALTY_SITES, "competitors": PENALTY_COMPETITORS}
    # Reported beside the shape because `silent` is the largest single row in most coverage blocks
    # and what it cost is not recoverable from the shape name.
    silent_factor = {"sites": PENALTY_SILENT_SITES, "competitors": 1.0}
    return {
        channel: {
            "applications": sum(r.applications for r in rows),
            "probed": sum(r.probed for r in rows),
            "unpaired": sum(r.unpaired for r in rows),
            "silent": sum(r.silent for r in rows),
            "unbuildable": sum(r.unbuildable for r in rows),
            "layers": len(rows),
            "penalty": shapes.get(channel),
            "silent_factor": silent_factor.get(channel, 1.0),
        }
        for channel in sorted({report.channel for report in reports})
        if (rows := [report for report in reports if report.channel == channel])
    }


def rule_attribution(
    reports: Sequence[ProbeReport], codes: Sequence[str], channel: str
) -> dict[str, Any]:
    """Per forward rule on one channel, how often it was offered and how often it fired.

    The named counterpart of :func:`channel_coverage`'s aggregates, and what each channel needs it
    for differs. On ``competitors`` a rule's fire count is which reaction competes -- the question
    that decides whether a veto row can be retired in favour of this channel, which co-occurrence
    alone cannot answer. On ``sites`` the gap between offered and fired names the rules that cannot
    reproduce their own partner's step, which is what the aggregate ``silent`` count only totals.

    Codes rather than ids, because an id is a position in whichever table this run compiled and
    means nothing in a record read later.
    """
    offered: Counter[int] = Counter()
    fired: Counter[int] = Counter()
    for report in reports:
        if report.channel != channel:
            continue
        offered.update(report.offered_by_rule)
        fired.update(report.fired_by_rule)
    return {
        codes[rule] if rule < len(codes) else str(rule): {
            "offered": count,
            "fired": fired.get(rule, 0),
        }
        for rule, count in sorted(
            offered.items(), key=lambda row: -fired.get(row[0], 0)
        )
    }


def retro_rule_label(codes: Sequence[str]) -> Callable[[int], str]:
    """A retro rule's template code where this run compiled one, else its id.

    Codes rather than ids, for the reason :func:`rule_attribution` gives: an id is a position in
    whichever table this run compiled and means nothing in a record read a month later, while a
    code is what a reader greps the template CSV for.
    """

    def label(rule: int) -> str:
        return codes[rule] if 0 <= rule < len(codes) else str(rule)

    return label


def replayed_names(
    routes: Sequence[Route[Any]], key_of: Callable[[str], Any]
) -> dict[Any, str]:
    """Search key -> SMILES for every molecule replay reproduced, so a report can name them.

    Exact by construction rather than by walking the two trees in step: a replayed node's key is
    whatever this arm's keyer makes of its SMILES, so keying the SMILES *is* the join and no
    assumption about precursor order can put a name on the wrong molecule.
    """
    names: dict[Any, str] = {}
    pending = list(routes)
    while pending:
        node = pending.pop()
        smiles = node.product
        if isinstance(smiles, str):
            try:
                names[key_of(smiles)] = smiles
            except Exception:
                pass
        pending.extend(node.precursors)
    return names


def paired_nodes(
    route: SidRoute,
    replayed: Route[Any],
    state: GraphState[Any],
    key_of: Callable[[str], Any],
) -> Iterator[tuple[int, str]]:
    """Each node of a route beside the SMILES replay gave it, wherever the two name one molecule.

    Paired by position and *checked by key*: replay rebuilt this very route, so the trees have the
    same shape and the same precursor order, and the check is what makes that an assertion rather
    than an assumption. A pair the keys disagree on yields nothing, so a disagreement can cost a
    name and can never invent one.
    """
    smiles = replayed.product
    if not isinstance(smiles, str):
        return
    try:
        if key_of(smiles) != state.expansions.decode([route.product])[0]:
            return
    except Exception:
        return
    yield route.product, smiles
    if len(route.precursors) != len(replayed.precursors):
        return
    for child, mirror in zip(route.precursors, replayed.precursors):
        yield from paired_nodes(child, mirror, state, key_of)


def alternative_names(
    pairs: Sequence[tuple[SidRoute, Route[Any]]],
    state: GraphState[Any],
    *,
    cost: Callable[[int, tuple[int, ...]], float],
    key_of: Callable[[str], Any],
    name_precursors: Callable[[str, Sequence[int]], Mapping[Any, str]],
) -> dict[Any, str]:
    """Name the molecules every *alternative* disconnection on a reported route would make.

    The report's readability problem, and it is not a small one: replay names the route that was
    chosen, so a table of the alternatives beside it came out as rows of search keys under the
    hashing arms -- a reader could see that a disconnection was turned down but not what it would
    have produced. Re-applying those rules to the product, which replay has already made real,
    names them from the graph the selector actually searched.

    Asked of the same :func:`vertsy.route_selection.disconnections` the selector chose among and
    the report explains, so the three cannot drift into different ideas of what an alternative is.
    """
    names: dict[Any, str] = {}
    for route, replayed in pairs:
        for sid, smiles in paired_nodes(route, replayed, state, key_of):
            rules = [rule for rule, _ in disconnections(state, cost, sid)]
            names.update(name_precursors(smiles, rules))
    return names


def molecule_name(names: Mapping[Any, str]) -> Callable[[Any], str]:
    """Name a molecule by its SMILES where one is known, else by a short digest of its key.

    Only replay produces SMILES, and only along a route it reproduced, so the molecules an
    *alternative* disconnection would make are usually nameless under the Metal arm -- its search
    keys are WL hashes. A digest says so plainly: a reader can still tell two alternatives apart
    and can still join a report against a graph, without a hash being mistaken for a structure.
    """

    def named(key: Any) -> str:
        if key in names:
            return names[key]
        if isinstance(key, str):
            return key
        digest = hashlib.blake2b(repr(key).encode(), digest_size=4).hexdigest()
        return f"#{digest}"

    return named


def alternatives_summary(results: Sequence[TargetResult]) -> dict[str, Any]:
    """What the per-step reports say when read as one sweep: a curation worklist.

    `believed_clean` is the part worth acting on. Each entry is a step the report *recommended* --
    a chosen step, not a rejected alternative -- on which a veto row fired while the `sites` probe
    reported **one clean product**. Keyed by template and veto row, which is the pair a curator
    has to open.

    **What the disagreement means depends on the row, and this column cannot tell you which.** A
    row that names a *different product* -- an electrophile landing where the ring is deactivated,
    say -- disagreeing with a clean `sites` count is a forward-pattern gap: the partner cannot see
    what the reaction makes, and the forward table is what to fix. A row that names a destroyed
    substrate, a species no bottle holds, or a reagent that outruns the catalyst is **not** a gap
    at all: no count of products can see it, which is exactly why it is a veto and not a price,
    and a clean `sites` count beside it is correct rather than wrong. Of the four pairs this
    surfaced on six targets, all four were the second kind -- B 6 (an acyl chloride acylates
    before palladium turns over), B 27 (a beta-halo amine closes to the aziridinium), B 30 (a
    metal opens an epoxide rather than inserting), B 31 (a base enolises an aldehyde before it
    deprotonates the alcohol) -- and none of their retro templates could be narrowed instead,
    each needing a negated environment the SMARTS subset has no way to express.

    `sites` alone decides it, which is the correction this came back wrong on. The first version
    also required no competitor, reasoning that a clean verdict needs both channels silent -- and
    found nothing, on a run where five chosen steps fired a veto row. All five reported `sites: 1`
    and every one of them had competitors, between 1 and 7. The two channels answer different
    questions: `sites` asks whether the intended reagent attacks one place, `competitors` asks what
    *other* reaction these same materials could undergo. Only the first can contradict a veto row
    that says the intended reaction destroys the substrate, so only the first belongs in the test.

    The split by `competitors` is kept beside it rather than folded in, because it answers the
    question :func:`overlap_tally` asks of a veto row: `competed` means the other channel found
    something here too and could be priced instead, `silent` means this row is the only thing
    standing between a reader and the step. On the six-target run above, all five were `competed`.
    """
    standings: Counter[str] = Counter()
    fired: Counter[str] = Counter()
    fired_on_chosen: Counter[str] = Counter()
    believed_clean: dict[str, Counter[str]] = {}
    refusals: Counter[str] = Counter()
    targets = 0

    for result in results:
        if result.alternatives is None:
            continue
        targets += 1
        # The routes that were *not* reported, which is the other half of what a reader needs: one
        # of these two reasons is a knob they can widen and the other is chemistry.
        for dropped in result.alternatives["rejected"]:
            standings[f"route:{dropped['standing']}"] += 1
            if dropped["reason"] is not None:
                refusals[dropped["reason"]] += 1
        for route in result.alternatives["routes"]:
            standings[f"route:{route['standing']}"] += 1
            for step in route["retro_steps"]:
                for option in (step["chosen"], *step["alternatives"]):
                    standings[option["standing"]] += 1
                    chosen = option["standing"] == "chosen"
                    measured = option["measurements"]
                    for code in option["vetoes_fired"]:
                        fired[code] += 1
                        if not chosen:
                            continue
                        fired_on_chosen[code] += 1
                        if measured.get("sites") not in (None, 1):
                            continue
                        tally = believed_clean.setdefault(
                            f"{option['rule_label']} / {code}", Counter()
                        )
                        tally["steps"] += 1
                        tally[
                            "silent" if not measured.get("competitors") else "competed"
                        ] += 1

    return {
        "targets": targets,
        "standings": dict(standings.most_common()),
        "vetoes_fired": dict(fired.most_common()),
        "vetoes_fired_on_a_chosen_step": dict(fired_on_chosen.most_common()),
        "refused_at_validation": dict(refusals.most_common()),
        "believed_clean": {
            pair: dict(tally.most_common())
            for pair, tally in sorted(
                believed_clean.items(), key=lambda row: -row[1]["steps"]
            )
        },
    }


def price_sweep_summary(results: Sequence[TargetResult]) -> dict[str, Any]:
    """One row per price shape, summed over the targets that reported a route.

    The sweep's actual output. Each shape saw the same graphs, so the rows are comparable by
    construction, and `condemned_steps_on_the_top_route` is the column to read: it is the only
    number here that does not come from the mechanism being tuned. **Steps rather than codes**,
    because one step can name two rows -- a *disputed* row is a candidate and not a fact, so
    summing the code counts reads one condemned step as two. The code tally stays beside it, where
    naming every candidate is the point rather than a distortion.

    What it cannot tell you is whether a *different* route is a better one. It says a shape moved
    the answer and whether the move agreed with the veto table, which is a direction and not a
    verdict -- and on a target set whose veto rows were written while watching these same failures,
    not even a blind one. The before/after trees are in each target's record for reading by hand.
    """
    rows: dict[str, dict[str, Any]] = {}
    for result in results:
        for shape, entry in (result.price_sweep or {}).items():
            row = rows.setdefault(
                shape,
                {
                    "targets": 0,
                    "top_changed": 0,
                    "chemistry_changed": 0,
                    "condemned_steps_on_the_top_route": 0,
                    "condemned_steps_on_any_reported_route": 0,
                    "vetoes_on_the_top_route": 0,
                    "vetoes_on_any_reported_route": 0,
                    "by_row": Counter(),
                },
            )
            row["targets"] += 1
            row["top_changed"] += bool(entry["top_changed"])
            row["chemistry_changed"] += bool(entry["chemistry_changed"])
            for name in (
                "condemned_steps_on_the_top_route",
                "condemned_steps_on_any_reported_route",
            ):
                row[name] += entry.get(name, 0)
            row["vetoes_on_the_top_route"] += sum(
                entry["vetoes_on_the_top_route"].values()
            )
            row["vetoes_on_any_reported_route"] += sum(
                entry["vetoes_on_any_reported_route"].values()
            )
            row["by_row"].update(entry["vetoes_on_the_top_route"])
    return {
        shape: row | {"by_row": dict(row["by_row"].most_common())}
        for shape, row in rows.items()
    }


def deficit_tally(
    reports: Sequence[DeficitReport], codes: Sequence[str]
) -> dict[str, Any]:
    """The no-mass-loss pass summed over layers, with the offending rules named.

    `by_rule` is the output worth having: it is the list of retro templates whose precursors cannot
    account for their product, which is the same list as "templates that still leave a reagent
    implicit" -- found by running the library rather than by reading it.
    """
    by_rule: Counter[int] = Counter()
    for report in reports:
        by_rule.update(report.by_rule)
    return {
        "mode": MASS,
        "slack": MASS_SLACK,
        "applications": sum(report.applications for report in reports),
        "unaccounted": sum(report.unaccounted for report in reports),
        "atoms": sum(report.atoms for report in reports),
        "layers": len(reports),
        "by_rule": {
            (codes[rule] if rule < len(codes) and codes[rule] else str(rule)): count
            for rule, count in by_rule.most_common()
        },
    }


def veto_tally(reports: Sequence[VetoReport]) -> dict[str, Any]:
    """The veto pass summed over the layers it ran on, with a fire count per row.

    Per row, because that is the unit of the decision: a table is curated one row at a time, and a
    pass total cannot say which row is carrying it or which two are catching the same thing. The
    totals come first anyway, since a row's count only means something against how many
    disconnections were offered.

    ``mode`` is carried so a recorded tally says whether those refusals happened or were only
    predicted -- the same reason :func:`channel_coverage` carries the penalty it fed.
    """
    return {
        "mode": VETOES,
        "applications": sum(report.applications for report in reports),
        "examined": sum(report.examined for report in reports),
        "unexamined": sum(report.unexamined for report in reports),
        "condemned": sum(report.condemned for report in reports),
        "layers": len(reports),
        # Busiest row first, since the question a tally is read for is which row is carrying it.
        "fired": dict(
            sorted(
                (
                    (code, sum(r.fired.get(code, 0) for r in reports))
                    for code in {code for r in reports for code in r.fired}
                ),
                key=lambda row: (-row[1], row[0]),
            )
        ),
    }


def factor_of(shape: str) -> Factor:
    """Resolve one spelling into a yield curve. `shared:<share>[:<floor>]`, and nothing else yet."""
    if shape.startswith("shared:"):
        parts = shape.split(":")[1:]
        floor = float(parts[1]) if len(parts) > 1 else 0.0
        return shared(float(parts[0]), floor=floor)
    raise ValueError(
        f"unknown yield shape {shape!r}; expected 'shared:<share>[:<floor>]'. "
        "'none' switches a channel off and is handled by the caller."
    )


def factors_of(sites: str, competitors: str, severity: bool = False) -> Factors | None:
    """The `yield` counterpart of :func:`penalties_of`, channel by channel.

    Measuring without pricing works the same way here, and so does switching one channel off: the
    two measure different chemistry and whether either carries signal is answered one at a time.

    No `doubted` or `uncurated` wrapper yet, so a silent or unprobed measurement reads as a clean
    1.0 -- which is what the bare curves already do and what `steps` did before
    `PENALTY_SILENT_SITES` was turned on. Pricing silence under a yield means naming the fraction a
    silent probe should imply, and that is a separate decision from naming the curve.
    """
    chosen = {
        channel: factor_of(shape)
        for channel, shape in (("sites", sites), ("competitors", competitors))
        if shape != "none"
    }
    if severity:
        # The one competitor reading that is priced, and the only one shaped for it. Its column is
        # a *band* and not a count, so the curve is a lookup rather than a geometry: three mild
        # competitors are not a worse band than one, so there is nothing for a geometry to
        # compound. Band 1 is "nothing destructive fired", which puts clean on the price's free
        # reference -- a count channel cannot do that, because `drop_intended` makes its `0` clean
        # and its `1` one competitor.
        chosen["competitors:severity"] = banded(SEVERITY_SURVIVALS)
    return chosen or None


def yield_base(yields: Mapping[int, float]) -> Callable[[int], float]:
    """A per-rule base cost from the curated yield: `-log(typical_yield(rule))`.

    `YIELD_DEFAULT` for a rule no forward row names, which is 154 of the 442 rules a
    hartenfeller + medchem + protecting-groups run compiles -- so the default is not an edge case
    and its value is load-bearing. See the knob for why it must stay at or below the curated median.
    """

    def base(rule: int) -> float:
        return cost_of_factor(yields.get(rule, YIELD_DEFAULT))

    return base


def substrate_of() -> Substrate | None:
    """How this run makes its price substrate-aware, or `None` to price nothing.

    One place, so the search and the report cannot drift into two readings of one measurement --
    which is the thing the comment beside `route_price` has always promised and which a second
    hand-rolled `selective` call would quietly break.
    """
    if not FORWARD_PROBE:
        return None
    if PRICE == "yield":
        factors = factors_of(YIELD_SITES, YIELD_COMPETITORS, SEVERITY_PRICE)
        return None if factors is None else partial(compounded, factors=factors)
    penalties = penalties_of(PENALTY_SITES, PENALTY_COMPETITORS)
    return None if penalties is None else partial(selective, penalties=penalties)


def penalties_of(sites: str, competitors: str) -> Penalties | None:
    """Resolve the two channel spellings into per-channel multipliers, or `None` to price neither.

    Measuring without pricing is a real thing to want: it fills the stores and the coverage report
    while leaving every cost exactly as it was, which is how the probe's wall clock is measured
    separately from its effect. A single channel can be switched off the same way, which is how the
    two are measured apart -- competing sites and competing reactions are different chemistry, and
    whether either carries signal is a question to answer one at a time.
    """
    chosen = {
        channel: multiplier_of(shape)
        for channel, shape in (("sites", sites), ("competitors", competitors))
        if shape != "none"
    }
    # Only `sites` doubts its silence, because only there does silence mean something went wrong.
    if "sites" in chosen and PENALTY_SILENT_SITES != 1.0:
        chosen["sites"] = doubted(chosen["sites"], silent=PENALTY_SILENT_SITES)
    # Both channels can charge for *not having looked*, which is a claim about our curation and
    # not about the chemistry -- see `vertsy.costs.uncurated`. Off by default for that reason.
    if PENALTY_UNPROBED != 1.0:
        chosen = {
            channel: uncurated(multiplier, unprobed=PENALTY_UNPROBED)
            for channel, multiplier in chosen.items()
        }
    return chosen or None


@dataclass(frozen=True)
class ForwardLibrary:
    """The forward rule table, compiled, and which retro rules each row probes."""

    smarts: tuple[str, ...]
    codes: tuple[str, ...]
    requires: tuple[frozenset[str], ...]
    """Condition classes each forward rule needs, from its own `requires` column.

    Its own column and not its partners' tags. The derived version tied the conditions gate to the
    pairing -- a row's conditions were the intersection over its `probes` partners -- so a row with
    no partner required nothing, and a condition only one partner needed was dropped. 162 of 190
    rows came out empty, and a rule requiring nothing competes in every pot, which is why the gate
    kept 99.2% of the mixture it was asked to filter. The vocabulary is still the shared one in
    `template-tag-vocabulary.csv`; only the list is the forward table's own."""
    conditions: Mapping[int, frozenset[str]]
    """Retro rule id -> the condition classes that reaction's medium supplies."""
    efficiency: tuple[float, ...]
    """Per **forward** rule, how completely it consumes its substrate -- and `0.0` where that is not
    a question worth asking.

    It is the row's own `typical_yield`, read in exactly the sense every other row's is: what
    fraction of material this reaction turns into its own product. Nothing is reinterpreted for a
    degradation row; the column means one thing, and a reaction nobody wants is still a reaction
    whose conversion a chemist can state. What the *price* does with it -- turn "how completely it
    destroys" into "what is left for the intended reaction" -- is a relative question and lives in
    :func:`vertsy.selectivity.severity_outcomes`, not here.

    Zero for every row in the default severity band, because a competitor that merely makes a second
    product takes nothing in this sense: you lose a separation, not the substrate."""
    severity: tuple[str, ...]
    """How bad each forward rule is as a competitor, from its own `severity` column.

    Empty for the default band -- a competitor that makes a second product, so the step costs a
    separation -- and `destructive` for one that consumes the substrate, where no separation
    recovers it. Indexed by **forward** rule id, like :attr:`requires` and for the same reason: it
    is a property of the candidate's own row, which is what a `Competes` is handed.

    Two bands and not four. The question a curator answers is "does this competitor cost you a
    separation, or your substrate?", which is one bit and settles without a literature search;
    ranking 204 rows on a severity scale nobody has measured would claim the resolution
    `typical_yield` already declines to claim for a yield."""
    arities: tuple[int, ...]
    """Top-level pieces on each forward rule's left-hand side, which is what its own example
    precursors have to match. Not the competitor filter -- see :attr:`declared`."""
    declared: tuple[int, ...]
    """How many separate starting materials each row *states* it needs, ``0`` when it states
    nothing. This is what makes a rule a competitor: a bare dot matches inside one molecule as
    readily as across two, so a row that has not said `(A).(B)` is not asked."""
    index: Mapping[int, tuple[int, ...]]
    """Retro rule id -> forward rule ids. Empty for every rule nobody wrote a forward rule for."""
    describes: Mapping[int, tuple[int, ...]]
    """Retro rule id -> the forward rule ids that are its **inverse**: `probes` and not `scope`.

    :attr:`index` unions the two because the sites channel wants both -- it asks what else the
    intended reagent makes from this pot. The inverse relation on its own is what can stand in for
    a reaction's own centre, because a `scope` entry describes a different transformation whose
    centre is somewhere else."""
    intended_centres: Mapping[int, tuple[CentreAtom, ...]]
    """Retro rule id -> that reaction's centre atoms, as a forward rule describes them.

    What the site split compares a candidate against, and it has to be in the candidate's own frame:
    a candidate is a forward rule querying the atoms **in the pot**, so the intended reaction's
    centre must be queried the same way. Only a forward row can say that, which caps coverage at the
    305 of 459 rules one describes -- see the note in :func:`load_forward_library` for the
    broader-but-wrong alternative and the measurement that rejected it.

    A rule absent here is one the split cannot judge, and both halves abstain on it rather than
    picking a side."""

    yields: Mapping[int, float]
    """Retro rule id -> the fraction of material the reaction is expected to give.

    The **maximum** over the rule's forward partners, because if any known way of running the step
    goes at 0.9 then the step goes at 0.9; a second, worse way of doing it is not evidence against
    the first. A rule no forward row names is absent here and takes the caller's default, which is
    most of them.
    """

    @property
    def paired_rules(self) -> int:
        return len(self.index)


def load_forward_library(sources: Sequence[RuleSource]) -> ForwardLibrary:
    """Read `templates-forward.csv` and resolve its pairings against the rules this run compiled.

    ``sources`` carries a ``template_code`` per compiled rule, which is the only place codes and ids
    meet -- a rule id is a position in whichever rule sets a run loaded, so the pairing has to be
    resolved here rather than in `vertsy.selectivity`, which sees only ints.

    A pairing naming a retro rule this run did not load is dropped. That is not an error: a run may
    legitimately load a subset, and a pairing with nothing to do is a pairing with nothing to do. A
    code that exists *nowhere* is caught by `test_forward_templates`, before any run.
    """
    with FORWARD_TEMPLATES.open() as handle:
        rows = tuple(csv.DictReader(handle))
    codes = tuple(row["forward_code"] for row in rows)
    # A rule id is the *compiled* position -- the index into `sources` -- and not `source.row`,
    # which is where the rule sat in whichever table it came from.
    retro_ids = {
        source.columns["template_code"]: rule_id
        for rule_id, source in enumerate(sources)
        if "template_code" in source.columns
    }
    smarts = tuple(row["forward_template"] for row in rows)
    # The tag table says what a *retro* rule's medium supplies, which is `conditions` below. What a
    # forward rule *needs* is its own `requires` column; the vocabulary is shared, the lists are not.
    tags_of: dict[str, set[str]] = {}
    for tag, tagged in read_tags(VETO_TAGS).items():
        for code in tagged:
            tags_of.setdefault(code, set()).add(tag)
    # Read from the row's own `requires` column, not derived from its `probes` partners. The
    # derivation tied the conditions gate to the pairing: a row's conditions were the intersection
    # of its partners' tags, so a row naming no partner required nothing and competed everywhere.
    # 162 of 190 rows landed there, which is why the gate kept 99.2% of the mixture it was asked to
    # filter. Conditions are a property of the reaction, so the reaction's own row states them.
    requires = tuple(
        frozenset(tag.strip() for tag in row["requires"].split(";") if tag.strip())
        for row in rows
    )
    # What a retro rule's medium *supplies*, from two sources unioned.
    #
    # The hand-curated tags are one: `template-tags.csv` says which reactions run on palladium, in
    # strong acid, through a carbanion -- written for the veto layer, where the question is what the
    # medium destroys, which is the same per-rule attribute.
    #
    # The pairing is the other, and it carries most of the weight. A reaction's medium supplies
    # whatever its own forward realisation needs: if `FWD MC 5` reduces an aldehyde with a hydride,
    # then the pot of the retro rule it inverts has a hydride in it. So `provides` is seeded from
    # the `requires` of every forward row naming the rule. **Union and not intersection**, unlike
    # the `requires` derivation this replaced: a larger `provides` admits more candidates, and for a
    # mixture report the conservative direction is to keep a peak rather than to hide one.
    #
    # A retro rule no forward row names supplies nothing, so only condition-free candidates compete
    # in its pot. That is the honest reading -- nothing here knows what its medium is.
    supplied: dict[str, set[str]] = {}
    for row, needs in zip(rows, requires):
        for code in row["probes"].split(";"):
            if code.strip():
                supplied.setdefault(code.strip(), set()).update(needs)
    conditions = {
        rule_id: frozenset(
            set(tags_of.get(source.columns["template_code"], ()))
            | supplied.get(source.columns["template_code"], set())
        )
        for rule_id, source in enumerate(sources)
        if "template_code" in source.columns
    }
    # Per forward row, then folded onto the retro rules it probes. Taking the max rather than the
    # mean or the min: a rule's best known realisation is what it can do, and a second worse way of
    # running it is not evidence against the first.
    row_yield = tuple(float(row["typical_yield"]) for row in rows)
    # `.get` for the reason `scope` uses it: a table written before this column existed still reads,
    # and a blank is the default band rather than a missing value.
    severity = tuple((row.get("severity") or "").strip() for row in rows)
    efficiency = tuple(
        value if band == "destructive" else 0.0
        for value, band in zip(row_yield, severity, strict=True)
    )
    best: dict[int, float] = {}
    for forward_id, row in enumerate(rows):
        for code in row["probes"].split(";"):
            retro = retro_ids.get(code.strip())
            if retro is not None:
                best[retro] = max(best.get(retro, 0.0), row_yield[forward_id])
    describes = forward_index(
        {row["forward_code"]: row["probes"].split(";") for row in rows},
        retro_ids,
        {code: position for position, code in enumerate(codes)},
    )
    # The centre of the forward rows that say they are this rule's inverse, and **only** those.
    #
    # The retro template has a centre of its own, and reading it here instead would cover 452 of 459
    # rules against this 305 -- which was tried, and is wrong. A retro rule's left-hand side is the
    # *product*, so its centre describes atoms as they are **after** the reaction, while every
    # candidate is a forward rule describing atoms as they are **in the pot**. Mirroring the bond
    # roles is necessary -- a disconnection breaks what its coupling makes -- and nowhere near
    # sufficient: measured over the 271 rules with exactly one describing forward row, only 86
    # centres coincide, 22 differ in the aromaticity mask alone and **163 differ in element or
    # role**. An amine nitrogen that becomes an aromatic ring nitrogen is genuinely a different
    # query on each side, and no swap fixes that. It is the same fact that makes this table exist:
    # a retro SMARTS cannot be reversed mechanically.
    forward_patterns = reaction_patterns_from_smarts(smarts, cache=True)
    intended_centres: dict[int, tuple[CentreAtom, ...]] = {}
    for retro, forward_rules in describes.items():
        described = tuple(
            atom
            for rule in forward_rules
            if 0 <= rule < len(forward_patterns)
            for atom in reaction_centre(forward_patterns[rule])
        )
        if described:
            intended_centres[retro] = described
    return ForwardLibrary(
        smarts=smarts,
        codes=codes,
        requires=requires,
        conditions=conditions,
        severity=severity,
        efficiency=efficiency,
        arities=template_arities(smarts),
        declared=declared_molecularity(smarts),
        # `probes` and `scope` are unioned here and nowhere else. They mean different things and
        # are kept apart in the table for that reason: `probes` says "I am this retro rule's
        # inverse", which `test_a_pairing_that_can_be_composed_does_compose` round-trips, and
        # `scope` says "the reagent that does this retro rule also does me", which is not an
        # inverse relation and cannot be round-tripped -- an ester reduction is not a ketone
        # reduction's mirror image, it is the other thing the same hydride reaches. The sites
        # channel wants both: it asks what else the intended reagent makes from this pot.
        yields=best,
        index=forward_index(
            {
                row["forward_code"]: [
                    *row["probes"].split(";"),
                    *(row.get("scope") or "").split(";"),
                ]
                for row in rows
            },
            retro_ids,
            {code: position for position, code in enumerate(codes)},
        ),
        describes=describes,
        intended_centres=intended_centres,
    )


@dataclass(frozen=True)
class VetoLayer:
    """The veto table, with its scopes already resolved against the rules this run compiled.

    Built once beside the forward library and handed to whichever arms a run asked for, which is
    the whole of the dependency injection: an arm is given a table and supplies the two
    arm-specific operations -- a :data:`vertsy.filters.Screen` over its own molecules and a
    :data:`vertsy.selectivity.Combine` to build a pot -- and nothing else about vetoes reaches
    into an arm.

    ``refuse`` is carried rather than read from the environment where it is used, so one object
    says both what the vetoes are and what they are allowed to do.
    """

    table: tuple[Veto, ...]
    in_scope: tuple[tuple[int, ...], ...]
    """Per compiled rule id, which rows apply. Resolved once; the inner loop indexes a tuple."""
    refuse: bool
    required: bool = True
    """Whether the run asked for this table or inherited it from the mode default.

    Carried here for the reason `refuse` is: one object says what the vetoes are, what they may do,
    and whether they were chosen. An arm that cannot screen refuses a table it was *asked* to
    apply and skips one it merely inherited -- see `AOSTAR_HOST_VETOES`.
    """
    withhold: bool = False
    """Record every verdict during the search and refuse at extraction, by looking the column up.

    Neither `refuse` nor `validate`, which is what makes the existing gates do the right thing
    untouched: the column block runs for anything that is not `validate`, and `enforced` reads
    `refuse` alone, so this writes every column and prunes nothing. The only thing it changes is
    which validator extraction is handed.
    """
    validate: bool = False
    """Evaluate during replay rather than during search, and let a refusal fail the route.

    The two are exclusive by construction: a run either prunes the graph as it is built or
    validates candidate routes afterwards, and doing both would charge twice for one fault.
    """

    @property
    def scoped_rules(self) -> int:
        return sum(1 for indices in self.in_scope if indices)


def load_veto_layer(sources: Sequence[RuleSource], mode: str) -> VetoLayer | None:
    """Read the veto table and resolve its scopes, or ``None`` when vetoes are off.

    Scope resolution is the same codes-meet-ids problem :func:`load_forward_library` has, solved the
    same way and in the same place: a rule id is a position in whichever rule sets this run loaded,
    so a row naming a template nothing compiled applies to no rule. That is not an error -- a run
    may load a subset -- and :func:`vertsy.filters.vetoes_in_scope` drops it silently, which is why
    the caller prints how many rules ended up in scope.
    """
    if mode not in VETO_MODES:
        raise ValueError(
            f"AOSTAR_HOST_VETOES must be one of {VETO_MODES}, got {mode!r}."
        )
    if mode == "off":
        return None
    table = read_vetoes(VETO_TEMPLATES)
    codes = tuple(source.columns.get("template_code", "") for source in sources)
    return VetoLayer(
        table=table,
        in_scope=vetoes_in_scope(table, codes, read_tags(VETO_TAGS)),
        refuse=mode == "refuse",
        validate=mode == "validate",
        withhold=mode == "withhold",
        required=VETOES_REQUESTED,
    )


def veto_reactions(
    vetoes: VetoLayer,
    reports: list[VetoReport],
    *,
    make_screen: Callable[[Sequence[Veto]], Any],
    combine: Any,
    per_row: bool = False,
) -> dict[str, Any]:
    """The verdict column, under the one name :func:`vertsy.filters.refused` enforces by.

        One entry rather than one per row, because a verdict is a *count* of vetoes fired: the axis is
        the application, and which row fired is a question for the report rather than for the column.
        Shaped like :func:`channel_probes` so both measurements enter a match the same way -- as named
        reaction scores in one mapping, never as nested wrappers.

        ``make_screen`` is a *factory* and not a screen, because a screen closes over its compiled
        patterns and answers ``(pattern index, molecule row)`` pairs in its own numbering. A column over
        a subset of the table therefore needs a screen over that same subset; handing it the whole
        table's screen would silently evaluate row 0's pattern for every row.
    With ``per_row``, one further column per veto row, named ``vetoed:<code>``. Nothing enforces by
        them, and they began as cross-tabulation fodder, but they are now what the report reads to
        *name* a fired row: the verdict column carries a count, so it cannot say which row fired, and
        a question about one row needs a column about one row. They are also the only veto columns
        keyed by step rather than by chemistry, which is what makes their attribution per-application
        rather than per-(precursors, rule). Splitting adds no screened pairs -- a row only ever screens its own scope -- so
        the cost is a handful of extra dispatches per layer over a signal that fires on a quarter of a
        percent of applications.
    """
    # Two columns, because a fired count cannot mean "remove this" and "charge for this" at once.
    # `vetoed` is what `refused` enforces; `discouraged` is what a multiplier prices, through the
    # same `selective` that prices a selectivity count -- a verdict is a channel like any other.
    built: dict[str, Any] = {
        "vetoed": vetoed(
            vetoes.table,
            vetoes.in_scope,
            screen=make_screen(vetoes.table),
            cost="pattern",
            force="refuse",
            combine=combine,
            report=reports.append,
        ),
        "discouraged": vetoed(
            vetoes.table,
            vetoes.in_scope,
            screen=make_screen(vetoes.table),
            cost="pattern",
            force="price",
            combine=combine,
            report=reports.append,
        ),
    }
    if per_row:
        for index, veto in enumerate(vetoes.table):
            built[f"vetoed:{veto.code}"] = vetoed(
                (veto,),
                # Scope is resolved against the whole table, so narrowing to one row means keeping
                # the rules this row reaches and renumbering its index to the only one left.
                tuple((0,) if index in indices else () for indices in vetoes.in_scope),
                screen=make_screen((veto,)),
                cost="pattern",
                combine=combine,
            )
    return built


@dataclass(frozen=True)
class OverlapReport:
    """How the veto verdict and the competitor count judged the *same* applications, one layer.

    The question a veto table cannot answer about itself: a row is worth keeping as a refusal only
    if nothing cheaper already finds the same chemistry, and the competitor channel is the thing
    that might. Both are reaction-score columns over one set of hit rows, so the join is an index --
    which is the second thing keeping the verdict on that axis bought.
    """

    applications: int
    """Applications this layer offered, and the denominator for every cell."""
    cells: Mapping[tuple[str, str], int]
    """``(veto row, competitor verdict) -> applications``, for the rows that condemned something.

    The veto key is a code, or ``"any"`` for the whole table's column. The competitor key is
    ``competed`` (at least one other reaction makes something from this pot), ``silent`` (the channel
    ran and found none), or ``unprobed`` (it never ran -- an unpaired rule or a pot no backend could
    build), which has to stay separate because a channel that did not look is not a channel that
    looked and found nothing.
    """


# What a channel's count says about one application, as four buckets shared by both channels.
# Named by the count rather than by a reading of it, because the reading differs: on `competitors`
# anything at or above `one` is a competing reaction, while on `sites` `one` is the *clean* answer
# and only `many` is a mixture. One vocabulary, and each channel's meaning derived from it.
OVERLAP_BUCKETS = ("unprobed", "silent", "one", "many")


def overlap_buckets(column: Any) -> np.ndarray:
    counts = np.asarray(column, dtype=np.int64)
    return np.where(
        counts == UNPROBED,
        "unprobed",
        np.where(counts == 0, "silent", np.where(counts == 1, "one", "many")),
    )


def overlapped(
    match: Any,
    *,
    codes: Sequence[str],
    channels: Sequence[str] = ("competitors", "sites"),
    report: Callable[[OverlapReport], None],
) -> Any:
    """Observe how the veto verdict and each channel's count agree, and pass the layer through.

    Deliberately **between** measuring and refusing. After, because the columns have to exist;
    before, because refusing first removes exactly the applications the cross-tabulation is about
    and would report an empty join as agreement.

    **Both channels, because a veto row's chemistry may live on either.** A row about one reagent
    hitting two groups -- a hydride reducing the amide it was meant to and the ester it was not --
    is a `sites` fact: the intended reagent makes more than one product. A row about two starting
    materials reacting with each other is a `competitors` fact. Joining against one channel only
    reads the rows of the other kind as uncovered, which is exactly the mistake this parameter
    fixes: `competitor_probe` excludes rules *paired* with the intended one, so an ester reduction
    scoped to an amide reduction can never appear there, however often it fires.

    A channel whose column is absent is skipped, so a run that measures vetoes without the forward
    probe is still a valid run -- it just has nothing to cross-tabulate.
    """

    def matched(mols: Any) -> Any:
        found = match(mols)
        rows = np.asarray(found.hits, dtype=np.int64).reshape(-1, 4)
        scores = found.reaction_scores
        if not rows.shape[0] or "vetoed" not in scores:
            return found
        live = [channel for channel in channels if channel in scores]
        if not live:
            return found
        order, starts, _ = grouped_applications(rows)
        # Every column here is written per application, so one row stands for its whole group.
        representative = order[starts]
        bucketed = {
            channel: overlap_buckets(scores[channel])[representative]
            for channel in live
        }
        cells: dict[tuple[str, str, str], int] = {}
        everything = np.ones(len(starts), dtype=bool)
        # `(all)` is the base rate, and without it the other rows cannot be read at all: a row whose
        # condemnations carry a channel's signal 70% of the time has found nothing if the channel
        # says 70% about everything. The whole first reading of the competitor channel turned on
        # this, so the denominator travels with the numerators rather than being looked up later.
        for name, column in (
            ("(all)", None),
            ("any", "vetoed"),
            *((code, f"vetoed:{code}") for code in codes),
        ):
            if column is not None and column not in scores:
                continue
            fired = (
                everything
                if column is None
                else np.asarray(scores[column], dtype=np.int64)[representative] > CLEAR
            )
            for channel, buckets in bucketed.items():
                for bucket, count in zip(
                    *np.unique(buckets[fired], return_counts=True)
                ):
                    cells[name, channel, str(bucket)] = int(count)
        report(OverlapReport(applications=len(starts), cells=cells))
        return found

    return matched


def overlap_tally(reports: Sequence[OverlapReport]) -> dict[str, Any]:
    """The cross-tabulation summed over layers, as one block per veto row per channel.

    Per row and with the row's own total beside it, because the decision is per row: a row whose
    condemnations all carry a channel's signal is one that channel already sees and could price
    instead, and a row whose condemnations are `silent` there is one only the veto catches.
    """
    names = sorted({name for report in reports for name, _, _ in report.cells})
    channels = sorted({channel for report in reports for _, channel, _ in report.cells})
    return {
        "applications": sum(report.applications for report in reports),
        "rows": {
            name: {
                channel: {
                    bucket: sum(
                        report.cells.get((name, channel, bucket), 0)
                        for report in reports
                    )
                    for bucket in OVERLAP_BUCKETS
                }
                for channel in channels
            }
            for name in names
        },
    }


def replay_validator(
    vetoes: VetoLayer | None, refusals: Counter[str]
) -> Callable[[Any], Any] | None:
    """Wrap an arm's replay match so a condemned application cannot be reproduced, or ``None``.

    Host-screened on every arm, and that is a fact rather than a shortcut: replay rehydrates
    ``CMol`` molecules and re-runs the templates on the host even when the search ran on the
    device, so one screen serves all three arms here where the layer path needed one each.

    Every row runs, whatever its cost. A `geometry` row could *only* run here; the `pattern` rows
    run here too in this mode, which is what makes it a debugging lever -- the same table, moved
    from the graph to the candidate routes, so the two placements can be compared on one library.
    """
    if vetoes is None or not vetoes.validate:
        return None
    from vesma.cmol import cmol_union

    verdict = vetoed(
        vetoes.table,
        vetoes.in_scope,
        screen=host_screen(vetoes.table),
        cost="pattern",
        force="refuse",
        combine=object_combine(cmol_union),
    )

    def wrap(match: Any) -> Any:
        return validated(
            match,
            verdict=verdict,
            report=lambda counts: refusals.update(counts),
        )

    return wrap


def route_validator(
    arm: Arm, expansions: Expansions[Any], smiles: str
) -> Callable[[SidRoute], str | None] | None:
    """Ask replay whether a candidate route can be made, for extraction to pick the next if not.

    The second of the two veto layers, and the reason they are two: the search may *price* a row,
    which steers towards more promising branches while keeping every measurement in the graph, and
    replay may *refuse* one, which is where a route is actually lost. `replay_validator` has
    already wrapped this arm's replay with the refusing table, so asking replay is asking the
    table -- and asking it against real molecules, which is the only place a `geometry` row can be
    answered at all.

    What this changes is what happens next. A condemned route used to fail at the readable-route
    stage, after selection had finished: the route vanished and `routes_lost` counted it. Asked
    here instead, the same refusal lets the selector take the next *diverse* candidate, so a
    refused route is replaced rather than subtracted.

    One replayed variant is enough, because the question is whether the plan can be made at all;
    the chosen routes are replayed again afterwards for their variants and their SMILES.
    """
    if arm.replay_routes is None:
        return None

    def validate(route: SidRoute) -> str | None:
        mismatches: list[ReplayMismatch[Any]] = []
        if arm.replay_routes(decode_route(route, expansions), smiles, 1, mismatches):
            return None
        return str(mismatches[0]) if mismatches else "replay produced no route"

    return validate


def layered_match(
    lifted: Any,
    *,
    probes: Mapping[str, Any],
    vetoes: VetoLayer | None,
    veto_reports: list[VetoReport],
    overlap_reports: list[OverlapReport] | None = None,
    make_screen: Callable[[Sequence[Veto]], Any] | None = None,
    combine: Any = None,
    sizes: Any = None,
    deficit_reports: list[DeficitReport] | None = None,
) -> Any:
    """Stack every measurement a run asked for onto one match, then whatever enforcement follows.

    One function rather than three copies, because the *order* is the load-bearing part and it is
    the same on every arm: measure, then refuse. :mod:`vertsy.filters` spells out why the two do not
    commute -- measuring first costs a verdict on every application and can report what it refused,
    refusing first is cheaper and cannot -- and a run whose whole purpose is to find out how often a
    hand-written row fires has to pay for the reportable order.

    Measurements compose as **entries in one mapping**, never as nested wrappers: a selectivity count
    and a veto verdict are both :data:`vertsy.search.ReactionScore` columns, so they arrive together
    in a single :func:`vertsy.search.measured` and the arm-specific halves stay inside the scores.
    That is why adding vetoes needed no new layer in the search -- only one more key.

    ``make_screen`` and ``combine`` are the two operations an arm supplies for itself. An arm with
    no screen cannot run vetoes at all. Asked to anyway it raises; handed a table it merely
    inherited from the mode default it drops it and says so, which is the difference between a
    contradictory request and a default meeting a capability gap.
    """
    reactions = dict(probes)
    # In `validate` mode the search is built without the veto wrapper at all, which is the whole
    # point of the wrappers being injected: the same table is handed to replay instead.
    if make_screen is None and vetoes is not None and not vetoes.validate:
        if vetoes.required:
            raise ValueError(
                "this arm supplies no veto screen, so it cannot evaluate a veto table; "
                "run vetoes on the cpu or metal arm, or set AOSTAR_HOST_VETOES=off."
            )
        # Inherited rather than asked for. Dropped for this arm alone, and said out loud: a
        # silently veto-free arm beside a veto-carrying one would make the two incomparable
        # without anything in the output admitting it.
        print(
            "[vetoes] this arm has no screen over its own molecules, so it carries no veto "
            "table. The mode was inherited from the default, not asked for; set "
            "AOSTAR_HOST_VETOES=off to silence this, or run vetoes on cpu or metal.",
            flush=True,
        )
        vetoes = None
    if vetoes is not None and not vetoes.validate:
        reactions |= veto_reactions(
            vetoes,
            veto_reports,
            make_screen=make_screen,
            combine=combine,
            # Always, not only when a cross-tabulation asked for them. These are the columns
            # keyed by *step* -- `(product, rule, precursors)` -- and so the only ones that
            # attribute a veto to the application that earned it. The aggregate goes under the
            # chemistry key `(precursors, rule)`, where two products sharing precursors and rule
            # collapse into one entry and the report can only say `B 13?`. Gating correct
            # attribution on `AOSTAR_HOST_VETO_OVERLAP`, a flag about something else, meant the
            # default run had only the collapsing column. Splitting screens no extra pairs.
            per_row=True,
        )
    if MASS != "off":
        if mode_error := (MASS not in MASS_MODES):
            raise ValueError(
                f"AOSTAR_HOST_MASS must be one of {MASS_MODES}, got {MASS!r}."
            )
        del mode_error
        if sizes is None:
            raise ValueError(
                "this arm supplies no way to count a batch's heavy atoms, so it cannot check "
                "that precursors account for a product; set AOSTAR_HOST_MASS=off."
            )
        reactions["unaccounted"] = unaccounted(
            sizes,
            slack=MASS_SLACK,
            report=None if deficit_reports is None else deficit_reports.append,
        )
    # Each channel's resolution policy comes from the module that owns the channel, declared here
    # beside the column because this is where the names are minted. The store asks; it never
    # decides. `unmeasured`: a count that contradicts itself measured nothing. `refusal_wins`: a
    # verdict that contradicts itself still found a refusal.
    reduce = {
        name: (
            refusal_wins
            if name.split(":")[0] in ("vetoed", "discouraged")
            else unmeasured
        )
        for name in reactions
    }
    match = (
        measured(lifted, reactions=reactions, reduce=reduce) if reactions else lifted
    )
    # Observing sits between the two arrows: the columns have to exist, and the applications it is
    # about have to still be there.
    if vetoes is not None and not vetoes.validate and overlap_reports is not None:
        match = overlapped(
            match,
            codes=[veto.code for veto in vetoes.table],
            # Every channel this run measures, so a split competitor channel is cross-tabulated
            # half by half. The enrichment of a half against the veto rows -- which are chemistry
            # neither channel reads -- is the test of whether splitting found anything.
            channels=[
                channel
                for channel in ("competitors", "sites")
                + (
                    ("competitors:destructive", "competitors:severity")
                    if SEVERITY
                    else ()
                )
                + (
                    ("competitors:same-site", "competitors:different-site")
                    if SPLIT_COMPETITORS
                    else ()
                )
                if channel in reactions
            ],
            report=overlap_reports.append,
        )
    # Enforcement is the outer wrapper and reads the column by name, which is what makes `measure`
    # mode a faithful prediction of `refuse` mode: the same verdict is computed either way, and the
    # only difference is whether anything acts on it.
    # Named together and combined by disjunction, so neither veto knows the other exists and
    # either can be switched off without the other noticing.
    enforced = (
        *(("vetoed",) if vetoes is not None and vetoes.refuse else ()),
        *(("unaccounted",) if MASS == "refuse" else ()),
    )
    if enforced:
        match = refused(match, applications=enforced)
    return match


def load_template_library(
    rule_set: str, path: Path
) -> tuple[int, tuple[str, ...], tuple[RuleSource, ...]]:
    """The same canonical rule collections the route-recovery benchmarks use.

    `used14k` is deduplicated by an index of its own before it gets here, so its rules cannot
    be traced back to a single source row and their `columns` are empty. That is the truth
    about them rather than a gap: the dedup is what makes the rule ids line up with the
    reference routes.
    """
    if rule_set == "used14k":
        rows = load_route_template_rows(path)
        smarts = route_recovery_rule_index_from_rows(rows).smarts
        sources = tuple(
            RuleSource(source=rule_set, row=index, columns={})
            for index in range(len(smarts))
        )
        return len(rows), smarts, sources

    rows = read_template_rows(path)
    sources = tuple(
        RuleSource(
            source=rule_set,
            row=index,
            columns={
                key: value
                for key, value in row.items()
                if key != "retro_template" and value not in (None, "")
            },
        )
        for index, row in enumerate(rows)
    )

    return len(rows), tuple(row["retro_template"] for row in rows), sources


@dataclass(frozen=True)
class Library:
    """A linted rule library, with the counts a run has to report to be reproducible."""

    smarts: tuple[str, ...]
    #: Parallel to `smarts`, so `sources[rule_id]` says where that rule came from. This is the
    #: only route from a rule id back to a name; nothing else survives the compile.
    sources: tuple[RuleSource, ...]
    rule_sets: tuple[tuple[str, Path], ...]
    extra_paths: tuple[Path, ...]
    extra_counts: tuple[int, ...]
    row_count: int
    input_count: int
    rejected_indices: tuple[int, ...]
    duplicate_indices: tuple[int, ...]
    load_seconds: float

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_indices)

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicate_indices)

    @property
    def source_counts(self) -> dict[str, int]:
        """Rules each source contributed *after* linting and dedup.

        The number worth reporting: a set listed second contributes only what the first did not
        already provide, and that is the whole meaning of the input order.
        """
        counts: dict[str, int] = {}

        for source in self.sources:
            counts[source.source] = counts.get(source.source, 0) + 1

        return counts


def load_library(
    rule_set: str, path: Path, extra_paths: tuple[Path, ...] = ()
) -> Library:
    """One rule set. Several are merged by :func:`load_merged_library`."""
    return load_merged_library(((rule_set, path),), extra_paths)


def load_merged_library(
    rule_sets: Sequence[tuple[str, Path]], extra_paths: tuple[Path, ...] = ()
) -> Library:
    """Several rule sets as one library, in the order given.

    **Order is how precedence is expressed.** The sets are concatenated and then deduplicated
    first-wins, so a template present in two sources keeps the spelling -- and the source
    columns, hence the name -- of whichever was listed first. Putting the better-labelled or
    better-curated set first is the whole mechanism; there is no scoring.

    Linting runs on the merged list rather than per source, so a duplicate spanning two sets is
    caught. That is the case that matters: within one shipped set, raw68k is deduplicated at
    build time and used14k by its own index, and only the hand-curated Hartenfeller table
    repeats itself.
    """
    started = perf_counter()
    templates: tuple[str, ...] = ()
    sources: tuple[RuleSource, ...] = ()
    row_count = 0

    for rule_set, path in rule_sets:
        rows, smarts, origins = load_template_library(rule_set, path)
        row_count += rows
        templates += smarts
        sources += origins

    extra_rows = tuple(read_extra_templates(extra) for extra in extra_paths)
    templates += tuple(smarts for rows in extra_rows for smarts in rows)
    # Supplemental files are one SMARTS per line, so there are no columns to carry. Recorded
    # with their path as the source rather than omitted, so every rule id has an origin.
    sources += tuple(
        RuleSource(source=str(path), row=index, columns={})
        for path, rows in zip(extra_paths, extra_rows)
        for index in range(len(rows))
    )
    rejected = tuple(index for index, _, _ in lint(templates))
    # Repeated SMARTS are dropped too, and counted apart from the mass-losing ones: they are a
    # different defect with a different cause. A duplicate is matched, applied and expanded
    # twice for identical precursors, so it costs a search real work and buys nothing. The
    # hand-curated Hartenfeller+iSCB table is where they turn up; raw68k is deduplicated at
    # build time and used14k goes through an index that already dedups.
    duplicated = tuple(duplicates(templates))
    dropped = frozenset(rejected) | frozenset(duplicated)
    retained = tuple(
        smarts for index, smarts in enumerate(templates) if index not in dropped
    )
    return Library(
        smarts=retained,
        sources=tuple(
            source for index, source in enumerate(sources) if index not in dropped
        ),
        rule_sets=tuple(rule_sets),
        extra_paths=extra_paths,
        extra_counts=tuple(len(rows) for rows in extra_rows),
        row_count=row_count,
        input_count=len(templates),
        rejected_indices=rejected,
        duplicate_indices=duplicated,
        load_seconds=perf_counter() - started,
    )


# --------------------------------------------------------------------------- stock preparation


@dataclass(frozen=True)
class StockRow:
    """One catalogue row as both arms need it: an identity and a fingerprint each."""

    wl_key: tuple[int, int]
    wl_fingerprint: np.ndarray
    rdkit_key: str
    rdkit_fingerprint: np.ndarray


@dataclass(frozen=True)
class Stock:
    """The shared catalogue: both arms' keys and fingerprints over exactly the same rows."""

    wl_keys: tuple[tuple[int, int], ...]
    wl_fingerprints: np.ndarray
    rdkit_keys: tuple[str, ...]
    rdkit_fingerprints: np.ndarray
    row_count: int
    refusals: dict[str, int]

    @property
    def refused(self) -> int:
        return sum(self.refusals.values())


def stock_row(
    smiles: str,
    *,
    config: CircularFingerprintConfig,
    n_bits: int,
) -> StockRow | str:
    """Both toolkits' view of one catalogue row, or the name of whatever refused it.

    A supplier catalogue is not a curated corpus. Enamine's building blocks include structures
    vesma's aromaticity model rejects outright (a five-membered delocalized circuit whose
    electron count fails Hückel, most often), and RDKit and vesma do not reject the same ones. So
    the failure is *returned* rather than raised: a row only one toolkit can read would give the
    two arms different catalogues, and comparing solve rates across different stocks measures
    nothing. Every arm uses the rows both toolkits accept, and the run reports what that cost.
    """
    from rdkit import Chem

    from vertsy.host_similarity import packed_ecfp4_words
    from vesma.keyer import rdkit_mol_canonical_smiles

    try:
        molecule = cmol_from_smiles(smiles)[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "RdkitParseFailure"
        return StockRow(
            wl_key=short_ring_wl_keyer(molecule),
            wl_fingerprint=circular_fingerprint_words(gmol_from_cmol(molecule), config),
            rdkit_key=rdkit_mol_canonical_smiles(mol),
            rdkit_fingerprint=packed_ecfp4_words([mol], n_bits=n_bits)[0],
        )
    except Exception as failure:  # noqa: BLE001 - the reason is data, not control flow
        return type(failure).__name__


def prepare_stock(
    smiles: tuple[str, ...],
    config: CircularFingerprintConfig,
    n_bits: int,
    cache_version: int,
    n_jobs: int | None,
) -> Stock:
    """Parse the catalogue once, for both arms, keeping the rows both toolkits accept.

    One pass rather than one per backend: the per-row work is the union of what the two arms
    need, so doing it together is both cheaper than two passes and the only way the arms are
    guaranteed to be searching against the same catalogue.
    """
    if cache_version != STOCK_CACHE_VERSION:
        raise ValueError("Unsupported host AO* stock cache version.")
    rows = parallel_map(
        partial(stock_row, config=config, n_bits=n_bits),
        smiles,
        n_jobs=n_jobs,
    )
    kept = [row for row in rows if isinstance(row, StockRow)]
    if not kept:
        raise ValueError("no stock row could be read by both toolkits.")
    return Stock(
        wl_keys=tuple(row.wl_key for row in kept),
        wl_fingerprints=np.stack([row.wl_fingerprint for row in kept]),
        rdkit_keys=tuple(row.rdkit_key for row in kept),
        rdkit_fingerprints=np.stack([row.rdkit_fingerprint for row in kept]),
        row_count=len(smiles),
        refusals=dict(Counter(row for row in rows if not isinstance(row, StockRow))),
    )


def cached_stock(smiles: tuple[str, ...], n_jobs: int | None) -> Stock:
    cached = cached_function(prepare_stock, ignore=("n_jobs",))
    return cached(smiles, FINGERPRINT, ECFP4_BITS, STOCK_CACHE_VERSION, n_jobs)


# --------------------------------------------------------------------------- backend assembly


@dataclass(frozen=True)
class Arm:
    """Everything one backend contributes to :func:`vertsy.aostar.aostar`, already built.

    The callables are typed loosely on purpose. Each backend's molecule and key types are its
    own — a packed Metal batch keyed by WL hash against an object array of ``Mol`` keyed by
    canonical SMILES — and they only have to agree with *themselves*, which is exactly what AO*'s
    own generic parameters express and what a container holding both cannot.
    """

    backend: str
    stock: frozenset[Any]
    heuristic: Callable[[Any], np.ndarray]
    match: Callable[[Any], Any]
    stack_mols: Callable[[Sequence[Any]], Any]
    concat_mols: Callable[[Sequence[Any]], Any]
    key_of: Callable[[str], Any]
    mol_of: Callable[[str], Any]
    setup_seconds: dict[str, float]
    stock_size: int
    stock_bytes: int
    #: The arm's reaction matcher, for a bench that needs the device internals behind it.
    #: Handed over rather than recovered: `bench_metal_retro_phases` used to reach through
    #: `arm.match.__self__`, which only worked while `match` was a bound method on the matcher. It
    #: is a composed function now -- `keyed_match(all_vs_all(matcher), keyer)` -- and nothing
    #: should have to unpick a closure to find what went into it.
    matcher: Any = None
    diagnostics: Callable[[], dict[str, float]] = field(default=dict)
    """Search key -> the stock file that supplied it; empty when nobody asked.

    Only a report reads it, so it is built once at setup and carried rather than recomputed per
    target -- and it is a plain mapping because that is all a lookup needs.
    """
    probe_reports: list[ProbeReport] = field(default_factory=list)
    """One report per expansion layer the forward probe measured, appended as the search runs.

    Empty when no probe is wired, which is every arm by default. A list rather than a total because
    coverage is worth reading per layer: the first pass is the target alone and the last is thousands
    of molecules, and a probe that covers one well and the other badly is worth noticing.

    Append-only across the whole sweep, since the arm outlives any one target -- so a per-target
    reading is a slice from where that target started, never the whole list.
    """
    probe_seconds: dict[str, float] = field(default_factory=dict)
    #: Which measurement channels this run asked for, so the report can show all of them.
    #:
    #: Set by `replace` after the arm is built, for the reason the provenance notes beside it are:
    #: it is a *reporting* concern, and threading it through every arm builder would make each one
    #: carry a parameter none of them uses. Without it `route_report` fell back to its own default
    #: of `("sites", "competitors")`, so every channel added since -- the site halves, and both
    #: severity readings -- was measured into the graph and then invisible to the one artefact a
    #: chemist reads. An arm that narrows what it runs leaves the channels it skipped blank, which
    #: is what `Alternative.channels` means by an empty column reading as "not measured".
    channels: tuple[str, ...] = ()
    """Channel -> seconds the forward probe spent in it, accumulated across the whole sweep.

    Cumulative like `probe_reports`, so a per-target reading is the difference from where that
    target started. It exists because the probe runs *inside* `arm.match`: its cost was inside the
    search interval but outside all four named phases, so it landed in the `orchestration` residual
    and nothing could read it apart from everything else there -- while the probe is the single
    largest thing the search does that nobody had ever timed.

    A **component of** the `match` phase, never an addition to it.
    """
    mixture_reports: list[MixtureReport] = field(default_factory=list)
    """One report per reported step whose mixture block was derived, cumulative across the sweep.

    A per-target reading is the difference from where that target started, like `probe_reports`.
    Counts rather than seconds, so it says the same thing on a busy host as on an idle one.
    """
    veto_reports: list[VetoReport] = field(default_factory=list)
    """One report per expansion layer the veto table judged, on the same terms as `probe_reports`.

    Empty unless `AOSTAR_HOST_VETOES` asked for them, and populated in `measure` mode exactly as in
    `refuse` mode: the verdict is the same column either way, and only whether anything reads it
    differs. That is what makes the measuring run a prediction of the refusing one.
    """
    validation_refusals: Counter[str] = field(default_factory=Counter)
    """Applications refused during replay, when `AOSTAR_HOST_VETOES=validate`.

    Append-only across the sweep like the report lists, so a target's own count is the difference
    from where it started.
    """
    deficit_reports: list[DeficitReport] = field(default_factory=list)
    """One report per layer the no-mass-loss check judged, when `AOSTAR_HOST_MASS` asked for it.

    Its `by_rule` is a worklist: the retro templates that still leave a reagent implicit, named by
    the runtime rather than by reading the table.
    """
    step_mixture: (
        Callable[[str, tuple[str, ...], int], tuple[ExpectedPeak, ...]] | None
    ) = None
    """What to expect in the pot for a reported step, or ``None`` where no forward table is wired.

    On the arm rather than passed at report time because it closes over the forward library, which
    is built once per sweep at setup. ``None`` reads as "not derived" in the report, which is what
    an arm with no forward rules can honestly say.
    """
    veto_rationales: Mapping[str, str] = field(default_factory=dict)
    """Veto code -> the chemistry that row states, for the report to cite.

    Carried on the arm for the reason `retro_codes` and `rule_yields` are: it is read once at
    setup and `run_target` is handed an arm and a target, nothing else. Empty when no veto table
    was loaded, which is what a report with no veto column should say.
    """
    retro_codes: tuple[str, ...] = ()
    """Names for the *retro* rule ids, which is what `deficit_reports` is keyed by.

    Beside `forward_codes` and for the same reason: a rule id is a position in whichever rule sets
    this run compiled, and a record read later needs the name.
    """
    rule_yields: Mapping[int, float] = field(default_factory=dict)
    """Retro rule id -> its curated yield, for the report's per-rule base cost.

    On the arm for the same reason `forward_codes` is: it is in the arm's own rule numbering, and a
    rule id means nothing without the table it indexes. Empty when the forward library is off, in
    which case every rule falls to `YIELD_DEFAULT`.
    """
    forward_codes: tuple[str, ...] = ()
    """Names for the rule ids in `probe_reports`, since an id is a position in this run's table.

    On the arm rather than threaded through `run_target`, because the reports are in the arm's own
    numbering and whatever can read them has to be able to name them.
    """
    overlap_reports: list[OverlapReport] = field(default_factory=list)
    """One cross-tabulation per layer, when `AOSTAR_HOST_VETO_OVERLAP` asked for one.

    Separate from `veto_reports` because it answers a different question -- not how often a row
    fires, but whether another channel already finds what it fires on.
    """
    replay_routes: (
        Callable[
            [Route[Any], str, int, list[ReplayMismatch[Any]] | None],
            tuple[Route[str], ...],
        ]
        | None
    ) = None
    precursor_names: Callable[[str, Sequence[int]], Mapping[Any, str]] | None = None
    """Given one molecule's SMILES and the rules offered on it, name what each rule would make.

    Beside `replay_routes` and built from the same matcher, but deliberately *not* through the
    validating wrapper: this names alternatives so a reader can judge them, and a refusing veto row
    must not also erase the evidence for its own refusal.
    """


def heuristic_of(
    kind: str,
    build: Callable[[], Callable[[Any], np.ndarray]],
    sizes: Callable[[Any], np.ndarray] | None = None,
) -> Callable[[Any], np.ndarray]:
    """One of the three estimates, built for the backend whose batches `sizes` can count.

    `unexplained` is accepted as the old spelling of `atoms`. It named the estimate by what it was
    not; recorded manifests and shell history still carry it, and refusing them buys nothing.
    """
    if kind == "zero":
        return zero_heuristic
    if kind == "stock":
        return build()
    if kind in ("atoms", "unexplained"):
        if sizes is None:
            raise ValueError(
                "the atoms-from-stock estimate needs a way to count a batch's atoms, and this "
                "backend supplied none."
            )
        return AtomsFromStock(build(), sizes)
    raise ValueError(
        f"AOSTAR_HOST_HEURISTIC must be 'stock', 'atoms' or 'zero', got {kind!r}."
    )


def cmol_atom_counts(molecules: Any) -> np.ndarray:
    """Atom counts for an object array of `cmol`, which is what the cpu arm searches over."""
    return np.fromiter(
        (molecule.atom_count for molecule in molecules),
        dtype=np.float32,
        count=len(molecules),
    )


def rdkit_atom_counts(molecules: Any) -> np.ndarray:
    """Atom counts for an object array of RDKit `Mol`. Heavy atoms, matching every other arm."""
    return np.fromiter(
        (molecule.GetNumHeavyAtoms() for molecule in molecules),
        dtype=np.float32,
        count=len(molecules),
    )


def packed_atom_counts(molecules: Any) -> np.ndarray:
    """Atom counts for a packed Metal batch, which already carries them."""
    return np.asarray(molecules.atom_counts, dtype=np.float32)


# Divisor for the atoms-from-stock estimate. 1.0 leaves it unscaled, which is what measured best;
# 64.0 bounds it by one reaction's cost and so restores admissibility. See AtomsFromStock for
# what that trade costs, which is more than it sounds.
ATOM_SCALE = 1.0


@dataclass(frozen=True)
class AtomsFromStock:
    """``|m| x (1 - similarity)``: how many of a molecule's atoms the catalogue does not yet supply.

    A distance denominated in **atoms**, where the estimate it wraps is a distance in ``[0, 1]``.
    Naming it by its unit is the useful part: ``stock`` and this one answer the same question at
    different scales, and which scale an estimate lives on is exactly what the step cost has to be
    commensurable with -- see `docs/note-forward-selectivity.md`.

    Wraps an existing estimate rather than replacing it, so the scan, the fingerprint and the
    catalogue residency are all unchanged and the only new work is counting atoms.

    The motive is what the search ranks. A leaf's score is ``context + inside``, and ``context``
    sums the other open siblings' estimates, so a leaf is scored by roughly the sum of the estimate
    over its solution graph's open frontier. A coefficient that saturates near 0.55 for everything
    far from stock therefore makes that sum count open molecules rather than weigh them. Scaling by
    the molecule removes the floor: the estimate still vanishes at the stock, and it now grows with
    what is left to build.

    Measured over 114 known-good steps, ranked through the search's own selector: mean normalised
    rank 0.233 against the incumbent's 0.423, top quartile 66% against 38%.

    Unscaled it reaches about 30, so it is **not admissible**: the relaxed phase of the anytime
    schedule charges 1.0 a step, and an estimate above that can overshoot the true remaining cost.
    ``scale`` exists to fix that -- ``|m| x (1 - similarity) <= |m|``, so dividing by a bound on
    molecule size gives ``h <= 1``, and any estimate under the cheapest reaction cost is admissible
    for free, since a molecule not in the stock needs at least one reaction.

    **It is not worth doing, measured.** Dividing by a constant looks ranking-neutral and is not:
    the greedy phase is unaffected, where the estimate dominates a 0.001 step at any scale, but the
    relaxed phase charges 1.0 and a scaled estimate is then dominated *by the step*, which turns
    that phase toward breadth-first. On hard-7, incumbent against unscaled against scaled by 64:
    all three solve 7/7 with a mean of 6.00 reactions, while molecules seen run 4571, **2585** and
    3801 and seconds run 34, **26** and 35. Admissibility bought no better route and gave back the
    whole efficiency gain, so the default leaves it off and the knob stays for anyone who needs the
    guarantee rather than the speed.
    """

    inner: Any
    sizes: Callable[[Any], np.ndarray]
    """How to count one batch's atoms, which every backend spells differently.

    A packed Metal batch carries `atom_counts` already; a `cmol` object array has `atom_count` per
    molecule; RDKit has `GetNumHeavyAtoms`. Supplied by the arm rather than guessed at here,
    because guessing is how this silently became metal-only.
    """

    scale: float = ATOM_SCALE

    def __call__(self, molecules: Any) -> np.ndarray:
        estimate = self.inner(molecules)
        if not len(molecules):
            return estimate
        counts = np.asarray(self.sizes(molecules), dtype=np.float32)[: len(estimate)]
        return (estimate * counts / self.scale).astype(np.float32)

    @property
    def nbytes(self) -> int:
        return int(getattr(self.inner, "nbytes", 0))


def default_scan_kind() -> str:
    """Metal where there is a GPU to use, and the same numbers on the host where there is not.

    Chosen independently of ``AOSTAR_HOST_BACKENDS`` so that the two arms always scan the same way
    and the wall-clock difference between them stays the matcher, which is what this bench measures.
    """
    from vesma.metal_circularfp import metal_circularfp_available

    return "metal" if metal_circularfp_available() else "host"


def scan_of(kind: str, stock_fingerprints: np.ndarray) -> NearestScan:
    """Where the popcounts happen, for one catalogue.

    Built per arm rather than shared, because a Metal scan holds a token for a resident catalogue
    and the two arms fingerprint their stock differently — one token per array is the invariant.
    """
    if kind == "host":
        return HostNearestScan()
    if kind == "metal":
        from vertsy.metal_similarity import metal_scan

        return metal_scan(stock_fingerprints)
    raise ValueError(f"AOSTAR_HOST_SCAN must be 'host' or 'metal', got {kind!r}.")


def metal_arm(
    library: Library,
    stock: Stock,
    *,
    heuristic_kind: str,
    scan_kind: str,
    weight: float,
    n_jobs: int | None,
    assignment_probe_cap: int = 8,
    forward: ForwardLibrary | None = None,
    channels: Collection[str] = (),
    vetoes: VetoLayer | None = None,
) -> Arm:
    from vesma.cpu_matcher import CpuReactionMatcher
    from vesma.metal_circularfp import compute_circular_fingerprints_metal
    from vesma.metal_reaction_matcher import (
        MetalFoldedPrefilteredReactionMatcher,
        MetalReactionMatcher,
        metal_combine,
        metal_concat_batches,
        metal_screen,
        metal_short_ring_wl_key,
        metal_stack_mols,
    )

    seconds: dict[str, float] = {}
    started = perf_counter()
    patterns = reaction_patterns_from_smarts(library.smarts, cache=True, n_jobs=n_jobs)
    seconds["template_parse"] = perf_counter() - started

    started = perf_counter()
    matcher = MetalFoldedPrefilteredReactionMatcher(
        MetalReactionMatcher(patterns, assignment_probe_cap=assignment_probe_cap)
    )
    seconds["matcher_build"] = perf_counter() - started

    started = perf_counter()
    replay_matcher = CpuReactionMatcher(patterns)
    seconds["replay_matcher_build"] = perf_counter() - started

    started = perf_counter()
    keys = frozenset(stock.wl_keys)
    heuristic = heuristic_of(
        heuristic_kind,
        lambda: packed_stock(
            stock.wl_fingerprints,
            weight=weight,
            config=FINGERPRINT,
            scan=scan_of(scan_kind, stock.wl_fingerprints),
            words=compute_circular_fingerprints_metal,
        ),
        packed_atom_counts,
    )
    seconds["stock_index"] = perf_counter() - started

    def diagnostics() -> dict[str, float]:
        """The wrapped matcher's per-stage report for the call that just finished.

        Read after each ``match`` rather than at the end, because these are last-call
        attributes and the search makes one call per pass.
        """
        report = matcher.last_diagnostics
        rows = {
            "prefilter_seconds": float(matcher.last_prefilter_seconds),
            "candidate_pairs": float(matcher.last_candidate_pair_count),
        }
        if report is None:
            return rows
        return rows | {
            name: float(getattr(report, name))
            for name in (
                "match_seconds",
                "match_buffer_seconds",
                "match_dispatch_seconds",
                "match_kernel_seconds",
                "match_copyback_seconds",
                "match_other_host_seconds",
                "match_template_upload_bytes",
                "match_molecule_upload_bytes",
                "match_pair_upload_bytes",
                "match_output_copyback_bytes",
                "match_assignment_count",
                "match_positive_pairs",
                "match_reserved_slots",
                "match_retry_pairs",
                "assignment_collect_seconds",
                "retro_seconds",
                "retro_construct_kernel_seconds",
                "wl_kernel_seconds",
                "precursor_gather_seconds",
                "precursor_concat_seconds",
                "consumer_seconds",
                "total_seconds",
                "component_overflow_assignments",
                "precursor_atom_overflow_components",
                "precursor_atom_overflow_assignments",
                "edited_atom_overflow_assignments",
                "undescribed_components",
            )
        }

    def replay_match(molecules: Any, pairs: Any) -> tuple[Any, Sequence[Any], Any]:
        precursors, hits = replay_matcher.match(molecules, pairs)
        return (
            precursors,
            [short_ring_wl_keyer.encode(molecule) for molecule in precursors],
            hits,
        )

    def replay_routes(
        route: Route[Any],
        smiles: str,
        cap: int,
        mismatches: list[ReplayMismatch[Any]] | None = None,
    ) -> tuple[Route[str], ...]:
        return template_replay_routes(
            route,
            cmol_from_smiles(smiles)[0],
            match=replay_match if validator is None else validator(replay_match),
            stack_mols=object_mol_array,
            output_key=cmol_rdkit_smiles_keyer.encode,
            cap=cap,
            on_mismatch=None if mismatches is None else mismatches.append,
        )

    def precursor_names(smiles: str, rules: Sequence[int]) -> Mapping[Any, str]:
        return step_precursor_names(
            cmol_from_smiles(smiles)[0],
            rules,
            match=replay_match,
            stack_mols=object_mol_array,
            output_key=cmol_rdkit_smiles_keyer.encode,
        )

    # The same expression the cpu arm uses at its own lift, which is the point of the keyer being
    # a `Batch -> Sequence[K]` function rather than a method on the matcher: this arm has nothing
    # arm-shaped left at the seam.
    from vesma.matcher import all_vs_all, keyed_match

    lifted = expansion(keyed_match(all_vs_all(matcher), metal_short_ring_wl_key))
    probe_reports: list[ProbeReport] = []
    probe_seconds: dict[str, float] = {}
    mixture_reports: list[MixtureReport] = []
    veto_reports: list[VetoReport] = []
    overlap_reports: list[OverlapReport] | None = [] if VETO_OVERLAP else None
    deficit_reports: list[DeficitReport] = []
    validation_refusals: Counter[str] = Counter()
    validator = replay_validator(vetoes, validation_refusals)
    match = layered_match(
        lifted,
        probes=(
            metal_probes(forward, channels, probe_reports, probe_seconds)
            if forward is not None
            else {}
        ),
        vetoes=vetoes,
        veto_reports=veto_reports,
        overlap_reports=overlap_reports,
        # The device's own screen and its own pot-builder, so a veto costs one match-only dispatch
        # over the packed arrays the batch already holds and no molecule is ever unpacked.
        make_screen=lambda table: metal_screen([veto.pattern for veto in table]),
        combine=metal_combine,
        sizes=packed_atom_counts,
        deficit_reports=deficit_reports,
    )
    return Arm(
        backend="metal",
        stock=keys,
        heuristic=heuristic,
        match=match,
        matcher=matcher,
        veto_rationales=(
            {veto.code: veto.rationale for veto in vetoes.table}
            if vetoes is not None
            else {}
        ),
        probe_reports=probe_reports,
        probe_seconds=probe_seconds,
        mixture_reports=mixture_reports,
        veto_reports=veto_reports,
        overlap_reports=overlap_reports if overlap_reports is not None else [],
        deficit_reports=deficit_reports,
        validation_refusals=validation_refusals,
        step_mixture=(
            # Identity from this arm's own keyer -- two peaks are the same peak when the
            # *search* would call them one molecule -- and the reading from RDKit.
            step_mixture(
                forward,
                short_ring_wl_keyer.encode,
                cmol_rdkit_smiles_keyer.encode,
                mixture_reports.append,
            )
            if forward is not None
            else None
        ),
        retro_codes=tuple(
            source.columns.get("template_code", "") for source in library.sources
        ),
        forward_codes=() if forward is None else forward.codes,
        rule_yields={} if forward is None else forward.yields,
        stack_mols=metal_stack_mols,
        concat_mols=metal_concat_batches,
        key_of=short_ring_wl_keyer.from_smiles,
        mol_of=lambda smiles: cmol_from_smiles(smiles)[0],
        setup_seconds=seconds,
        stock_size=len(keys),
        stock_bytes=int(stock.wl_fingerprints.nbytes),
        diagnostics=diagnostics,
        replay_routes=replay_routes,
        precursor_names=precursor_names,
    )


def cpu_arm(
    library: Library,
    stock: Stock,
    *,
    heuristic_kind: str,
    scan_kind: str,
    weight: float,
    n_jobs: int | None,
    forward: ForwardLibrary | None = None,
    channels: Collection[str] = (),
    vetoes: VetoLayer | None = None,
) -> Arm:
    """Vesma end to end on the host: its matcher, its molecules, its keys, no device.

    The Metal arm with the CPU matcher in place of the kernel, which makes it the *reference*
    vesma arm rather than a slow one -- and the only arm the forward probe can run in full. RDKit's
    ``RunReactants`` takes one molecule per reactant template component, so its arm cannot receive
    the second starting material of a two-component forward rule at all; vesma's matcher matches a
    disconnected pattern into the disconnected graph :func:`vesma.cmol_union` builds, which is the
    whole of what a multi-component probe needs. Thirty-four of the forty-four forward rules, and
    the competitor channel entirely, exist only here.

    It shares the Metal arm's identity and catalogue exactly -- WL keys, packed circular
    fingerprints, the same stock rows -- so a measurement taken here is comparable with one taken
    there, and the difference between them is the matcher, which is what this bench is for.
    """
    from vesma.cpu_matcher import CpuReactionMatcher
    from vesma.matcher import all_vs_all, keyed_match

    seconds: dict[str, float] = {}
    started = perf_counter()
    patterns = reaction_patterns_from_smarts(library.smarts, cache=True, n_jobs=n_jobs)
    seconds["template_parse"] = perf_counter() - started

    started = perf_counter()
    # `n_jobs` belongs to the *match call* here rather than the constructor, unlike the RDKit
    # matcher whose workers hold the compiled rules.
    matcher = CpuReactionMatcher(patterns)
    seconds["matcher_build"] = perf_counter() - started

    started = perf_counter()
    keys = frozenset(stock.wl_keys)
    heuristic = heuristic_of(
        heuristic_kind,
        lambda: cmol_stock(
            stock.wl_fingerprints,
            weight=weight,
            config=FINGERPRINT,
            scan=scan_of(scan_kind, stock.wl_fingerprints),
        ),
        cmol_atom_counts,
    )
    seconds["stock_index"] = perf_counter() - started

    def wl_keys(precursors: Any) -> Sequence[Any]:
        return [short_ring_wl_keyer.encode(molecule) for molecule in precursors]

    from vesma.cmol import cmol_union

    lifted = expansion(keyed_match(all_vs_all(matcher), wl_keys))
    probe_reports: list[ProbeReport] = []
    probe_seconds: dict[str, float] = {}
    mixture_reports: list[MixtureReport] = []
    veto_reports: list[VetoReport] = []
    overlap_reports: list[OverlapReport] | None = [] if VETO_OVERLAP else None
    deficit_reports: list[DeficitReport] = []
    validation_refusals: Counter[str] = Counter()
    validator = replay_validator(vetoes, validation_refusals)
    match = layered_match(
        lifted,
        probes=(
            cpu_probes(forward, channels, probe_reports, probe_seconds)
            if forward is not None
            else {}
        ),
        vetoes=vetoes,
        veto_reports=veto_reports,
        overlap_reports=overlap_reports,
        make_screen=host_screen,
        combine=object_combine(cmol_union),
        sizes=cmol_atom_counts,
        deficit_reports=deficit_reports,
    )

    def selective_match(molecules: Any, pairs: Any) -> tuple[Any, Sequence[Any], Any]:
        precursors, hits = matcher.match(molecules, pairs)
        return precursors, wl_keys(precursors), hits

    def replay_routes(
        route: Route[Any],
        smiles: str,
        cap: int,
        mismatches: list[ReplayMismatch[Any]] | None = None,
    ) -> tuple[Route[str], ...]:
        return template_replay_routes(
            route,
            cmol_from_smiles(smiles)[0],
            match=(
                selective_match if validator is None else validator(selective_match)
            ),
            stack_mols=object_mol_array,
            output_key=cmol_rdkit_smiles_keyer.encode,
            cap=cap,
            on_mismatch=None if mismatches is None else mismatches.append,
        )

    def precursor_names(smiles: str, rules: Sequence[int]) -> Mapping[Any, str]:
        return step_precursor_names(
            cmol_from_smiles(smiles)[0],
            rules,
            match=selective_match,
            stack_mols=object_mol_array,
            output_key=cmol_rdkit_smiles_keyer.encode,
        )

    return Arm(
        backend="cpu",
        stock=keys,
        heuristic=heuristic,
        match=match,
        matcher=matcher,
        veto_rationales=(
            {veto.code: veto.rationale for veto in vetoes.table}
            if vetoes is not None
            else {}
        ),
        probe_reports=probe_reports,
        probe_seconds=probe_seconds,
        mixture_reports=mixture_reports,
        veto_reports=veto_reports,
        overlap_reports=overlap_reports if overlap_reports is not None else [],
        deficit_reports=deficit_reports,
        validation_refusals=validation_refusals,
        step_mixture=(
            # Identity from this arm's own keyer -- two peaks are the same peak when the
            # *search* would call them one molecule -- and the reading from RDKit.
            step_mixture(
                forward,
                short_ring_wl_keyer.encode,
                cmol_rdkit_smiles_keyer.encode,
                mixture_reports.append,
            )
            if forward is not None
            else None
        ),
        retro_codes=tuple(
            source.columns.get("template_code", "") for source in library.sources
        ),
        forward_codes=() if forward is None else forward.codes,
        rule_yields={} if forward is None else forward.yields,
        stack_mols=object_mol_array,
        concat_mols=concat_object_batches,
        key_of=short_ring_wl_keyer.from_smiles,
        mol_of=lambda smiles: cmol_from_smiles(smiles)[0],
        setup_seconds=seconds,
        stock_size=len(keys),
        stock_bytes=int(stock.wl_fingerprints.nbytes),
        replay_routes=replay_routes,
        precursor_names=precursor_names,
    )


def timed_probe(seconds: dict[str, float], channel: str, probe: Any) -> Any:
    """A :data:`vertsy.search.ReactionScore` that accumulates its own wall clock.

    Separate from :func:`timed` only in where the total lands: a probe is built once per sweep,
    inside the arm, so it cannot write into a mapping `run_target` owns. It writes into one the
    arm carries, and the per-target reading is a difference, as every other arm-side measurement
    here is.
    """
    return timed(seconds, channel, probe)


def channel_probes(
    forward: ForwardLibrary,
    channels: Collection[str],
    reports: list[ProbeReport],
    matcher: Any,
    *,
    key: Any,
    combine: Any,
    seconds: dict[str, float],
) -> dict[str, Any]:
    """One probe per requested channel, over whatever molecules an arm happens to hold.

    The arm supplies three things and nothing else: a matcher over the forward table, a way to name
    what comes out, and a way to make one molecule of several. Every arm's probe is then this
    function, which is what "one mechanism, two channels" has to mean in the wiring as well as in
    the algebra.

    Only the channels asked for are built. Measuring a channel nobody prices is a thing to want --
    that is how its wall clock is read apart from its effect -- but measuring one nobody asked for is
    just wall clock.

    `seconds` is where that wall clock finally lands. The sentence above has been in this docstring
    since the channels were written and was not true: the probe runs inside the arm's `match`, so
    its cost was inside the search interval but outside all four named phases, which left it in the
    `orchestration` residual with no way to read it apart from anything else. Each channel is now
    wrapped in its own timer, accumulating per channel across the whole sweep -- so a per-target
    reading is the difference from where that target started, exactly as `reports` is.

    These seconds are a *component of* the `match` phase, not an addition to it. Anything printing
    both has to say so or the shares read as summing past the wall clock.
    """
    built: dict[str, Any] = {}
    if "sites" in channels:
        built["sites"] = timed_probe(
            seconds,
            "sites",
            site_probe(
                matcher,
                forward.index,
                key=key,
                combine=combine,
                report=reports.append,
            ),
        )
    competitor_channels = [
        channel for channel in channels if channel.startswith("competitors")
    ]
    if competitor_channels and combine is None:
        raise ValueError(
            "the competitor channel runs multi-component forward rules and needs a way to "
            "combine an application's precursors into one molecule."
        )

    # What the relative severity reduction needs: each candidate's own degradation efficiency, and
    # the yield of whichever reaction was intended. Both come from one column read one way, and the
    # ratio between them is computed where both are known -- which is the only place it can be,
    # since a `Factor` sees one integer and cannot ask what the step was for.
    def intended_yield(rule: int) -> float:
        return forward.yields.get(rule, YIELD_DEFAULT)

    severity_reduce = severity_outcomes(
        forward.efficiency, intended_yield, SEVERITY_EDGES
    )
    for channel in competitor_channels:
        built[channel] = timed_probe(
            seconds,
            channel,
            competitor_probe(
                matcher,
                forward.declared,
                key=key,
                combine=combine,
                index=forward.index,
                competes=_competes_for(channel, forward),
                admit_unimolecular=UNIMOLECULAR,
                channel=channel,
                # Offering this channel only the destructive candidates is an optimisation that
                # cannot change its value: every other band ranks 0 and `max` over them is the same
                # 1 the empty case gives. It is the difference between 6 candidate rules and ~64.
                reduce=(
                    severity_reduce
                    if channel == "competitors:severity"
                    else distinct_outcomes
                ),
                report=reports.append,
            ),
        )
    return built


@dataclass(frozen=True)
class MixtureReport:
    """How many forward rules one reported step's mixture block was derived from.

    A count rather than a duration, and that is the point: it is the same number whatever else the
    host is doing, so it is the half of this measurement that survives a busy machine. It is also
    exactly the number of `parse_reaction_smarts` calls the derivation costs while
    :func:`vertsy.mixture.expected_mixture` is handed SMARTS -- which makes the before/after of
    handing it parsed patterns instead an exact, load-independent figure.

    Split by channel because which group dominates decides whether the cost is the pairing or the
    whole admitted table.
    """

    same_reaction: int
    same_site: int
    different_site: int
    unknown_site: int
    """Candidates the site split cannot place, because no forward row describes the intended rule.

    Its own field rather than folded into one of the halves: these used to be counted as same-site
    by an out-of-range guard, and a diagnostic that reports a guess as a measurement is worse than
    one that reports nothing."""

    @property
    def rules(self) -> int:
        return (
            self.same_reaction
            + self.same_site
            + self.different_site
            + self.unknown_site
        )


def step_mixture(
    forward: ForwardLibrary,
    key: Callable[[Any], Any],
    render: Callable[[Any], str],
    report: Callable[[MixtureReport], None],
) -> Callable[[str, tuple[str, ...], int], tuple[ExpectedPeak, ...]]:
    """What to expect in the pot for a reported step, classified the way the channels are.

    The three groups handed to :func:`vertsy.mixture.expected_mixture` are the three channels this
    run measures, resolved for one intended rule: its own paired rules, then the admitted
    multi-component rules split by whether they want the same atom. So the report cannot disagree
    with the counts beside it -- same pairings, same arity filter, same conditions gate, same
    `same_site` predicate.

    Re-derived here rather than carried through the search, which is what makes it free: a report
    covers a few dozen steps against the millions of applications a search examines, and nothing in
    the graph grows a field.

    `report` is handed one :class:`MixtureReport` per call, on the same terms the channels hand
    back a `ProbeReport`. "Free" above was asserted and never measured; this is what measures it.
    """
    patterns = reaction_patterns_from_smarts(forward.smarts, cache=True)
    same = same_site(patterns, forward.intended_centres)
    different = different_site(patterns, forward.intended_centres)
    runs_here = conditions_allow(forward.conditions, forward.requires)

    def mixture(
        product: str, precursors: tuple[str, ...], rule: int
    ) -> tuple[ExpectedPeak, ...]:
        paired = set(forward.index.get(rule, ()))
        groups: dict[str, list[tuple[str, str]]] = {
            "same_reaction": [],
            "same_site": [],
            "different_site": [],
            "unknown_site": [],
        }
        for candidate, stated in enumerate(forward.declared):
            # The parsed rule, not its text: `patterns` above is the whole table parsed once per
            # sweep, and handing `expected_mixture` the SMARTS made it re-parse every rule for
            # every reported step.
            entry = (forward.codes[candidate], patterns[candidate])
            if candidate in paired:
                groups["same_reaction"].append(entry)
                continue
            if not (UNIMOLECULAR or stated >= 2):
                continue
            if not (CONDITIONS and runs_here(rule, candidate)):
                continue
            # Three ways and not two. The halves abstain together where the intended rule has no
            # forward row describing it, so `else` is not "different site" -- it is "nobody can
            # say", and a peak that reaches the report under a label nobody derived is the defect
            # this replaces.
            if same(rule, candidate):
                bucket = "same_site"
            elif different(rule, candidate):
                bucket = "different_site"
            else:
                bucket = "unknown_site"
            groups[bucket].append(entry)
        report(
            MixtureReport(
                same_reaction=len(groups["same_reaction"]),
                same_site=len(groups["same_site"]),
                different_site=len(groups["different_site"]),
                unknown_site=len(groups["unknown_site"]),
            )
        )
        try:
            return expected_mixture(
                product,
                precursors,
                same_reaction=tuple(groups["same_reaction"]),
                same_site=tuple(groups["same_site"]),
                different_site=tuple(groups["different_site"]),
                unknown_site=tuple(groups["unknown_site"]),
                key=key,
                render=render,
            )
        except Exception as exc:
            # A displayed name this arm cannot read back as a molecule -- the Metal arm keys by WL
            # hash and only replay produces SMILES -- leaves the mixture empty rather than taking
            # the whole report down with it. An empty block reads as "not derived", which is true.
            #
            # Warned rather than silent, because "not derived" and "a bug in here" look identical
            # from the outside and one of them was true for every report ever written: `key` was
            # shadowed by a loop variable of the same name, so `expected_mixture` got the string
            # "same_site" where it wanted the keyer and raised on every step that classified a
            # candidate. Nothing said so.
            warnings.warn(
                f"no expected mixture for {product}: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return ()

    return mixture


def _competes_for(channel: str, forward: ForwardLibrary) -> Competes | None:
    """The predicate one competitor channel keeps its candidates by, from the channel's name.

    `competitors` is the crude reading, every unpaired multi-component rule. The two halves split
    it by **what fixes the competition** -- an axis that turned out to be a good diagnostic and no
    basis for a price, measured in :func:`vertsy.selectivity.same_site`: a competitor wanting a
    *different* atom is answered by protecting the other group, a bounded three-step detour the
    selector can actually buy; one wanting the **same** atom cannot be protected against, because
    protecting the site stops both reactions. Two channels rather than one filter, because the
    different-site half still costs yield -- it is cheaper to fix, not free -- and a `Competes` can
    only drop a candidate, never label it.

    `competitors:destructive` splits it on a second and independent axis -- **how bad the
    competition is** -- and that last sentence is why it is a channel too. The unsplit channel
    counts "some other rule could consume these precursors", which covers both *you will have to
    separate two products* and *your substrate is gone*; one count cannot carry both, and a
    `Competes` cannot annotate the one it kept. So the band becomes a name, and a name is already
    a channel with its own entry in the price's `Factors` -- which is what makes severity cost no
    new algebra. The axes compose rather than conflict: this one says how much the competition
    costs, the site halves say what would fix it.

    The patterns are recompiled rather than threaded through; `reaction_patterns_from_smarts`
    caches, so this costs a dictionary lookup.
    """
    runs_here = (
        conditions_allow(forward.conditions, forward.requires)
        if CONDITIONS
        else lambda intended, candidate: True
    )
    if channel == "competitors":
        return runs_here if CONDITIONS else None
    if channel in ("competitors:destructive", "competitors:severity"):
        # Indexed by `candidate`, which is the one thing a `Competes` can be sure of: `conditions_allow`
        # reads `requires[candidate]` the same way, because severity and conditions are both properties
        # of the candidate's own forward row. `intended` is a *retro* id and is only ever passed through.
        destructive = frozenset(
            rule for rule, band in enumerate(forward.severity) if band == "destructive"
        )
        return lambda intended, candidate: (
            runs_here(intended, candidate) and candidate in destructive
        )
    patterns = reaction_patterns_from_smarts(forward.smarts, cache=True)
    if channel == "competitors:same-site":
        same = same_site(patterns, forward.intended_centres)
        return lambda intended, candidate: (
            runs_here(intended, candidate) and same(intended, candidate)
        )
    if channel == "competitors:different-site":
        # `different_site` rather than `not same_site`: where no forward row describes the intended
        # reaction there is no centre to compare, and both halves abstain. Negating one would send
        # every such application here.
        different = different_site(patterns, forward.intended_centres)
        return lambda intended, candidate: (
            runs_here(intended, candidate) and different(intended, candidate)
        )
    raise ValueError(
        f"unknown competitor channel {channel!r}; expected 'competitors', "
        "'competitors:destructive', 'competitors:severity', 'competitors:same-site' or "
        "'competitors:different-site'."
    )


def metal_probes(
    forward: ForwardLibrary,
    channels: Collection[str],
    reports: list[ProbeReport],
    seconds: dict[str, float],
) -> dict[str, Any]:
    """Both channels on the device, with nothing but keys coming back.

    The Metal arm's batch is packed arrays rather than molecules, which is what used to put the probe
    out of reach -- and it turns out to be the *best* place for one. ``metal_combine`` writes a
    group's atom blocks and bond blocks straight into the packed layout the matcher consumes, so a
    multi-component pattern gets its graph without a molecule ever being built, and keying the
    result is a second dispatch over the same arrays rather than a trip through the host.
    """
    from vesma.metal_reaction_matcher import (
        MetalFoldedPrefilteredReactionMatcher,
        MetalReactionMatcher,
        metal_combine,
        metal_short_ring_wl_key,
    )

    matcher = MetalFoldedPrefilteredReactionMatcher(
        MetalReactionMatcher(reaction_patterns_from_smarts(forward.smarts, cache=True))
    )

    # The arm's own keyer, passed as it is -- exactly as `cpu_probes` passes its keyer. This was a
    # closure whose only job was to raise, because the keys it wanted were a field the device had
    # filled only for precursors; `metal_short_ring_wl_key` now computes over any batch, so the
    # frontier gather `vertsy.selectivity` keys to name the intended product is just a batch.
    return channel_probes(
        forward,
        channels,
        reports,
        matcher,
        key=metal_short_ring_wl_key,
        combine=metal_combine,
        seconds=seconds,
    )


def cpu_probes(
    forward: ForwardLibrary,
    channels: Collection[str],
    reports: list[ProbeReport],
    seconds: dict[str, float],
) -> dict[str, Any]:
    """Both channels, over vesma ``CMol`` molecules, which is what this arm's batch holds.

    Nothing is narrowed here, unlike :func:`rdkit_probes`: a multi-component forward rule matches a
    disconnected pattern into the disconnected graph `cmol_union` builds, so every row of the
    forward table can run and the competitor channel has its thirty-four rules to work with.
    """
    from vesma.cmol import cmol_union
    from vesma.cpu_matcher import CpuReactionMatcher

    matcher = CpuReactionMatcher(
        reaction_patterns_from_smarts(forward.smarts, cache=True)
    )
    combine = object_combine(cmol_union)

    def key(mols: Any) -> Sequence[Any]:
        return [short_ring_wl_keyer.encode(molecule) for molecule in mols]

    return channel_probes(
        forward, channels, reports, matcher, key=key, combine=combine, seconds=seconds
    )


def rdkit_probes(
    forward: ForwardLibrary,
    channels: Collection[str],
    reports: list[ProbeReport],
    seconds: dict[str, float],
) -> dict[str, Any]:
    """One forward probe per requested channel, over RDKit molecules -- this arm's batch.

    A probe has to speak the arm's molecule representation, because it runs the arm's own precursors
    through a matcher, so this compiles the forward table for RDKit.

    **This arm can only run unimolecular forward rules**, and the reason is RDKit's rather than
    ours. ``RunReactants`` takes one molecule per reactant *template component*, and
    :class:`vesma.matcher.ReactionMatcher` hands it one molecule per ``(molecule, rule)`` pair -- so
    a two-component forward rule has no way to receive its second starting material, combined or
    otherwise. Vesma's own matchers have no such limit: the CPU one matches a disconnected pattern
    into the graph :func:`vesma.cmol.cmol_union` builds, and the Metal one into the packed blocks
    :func:`vesma.metal_reaction_matcher.metal_combine` writes. So the multi-component half of the
    forward table, and the competitor channel entirely, live on those two.

    Multi-component rules are dropped from the index rather than attempted, and the competitor
    channel is skipped rather than refused -- a run on this arm is still a valid run, it just
    measures less. Both are printed, because coverage that quietly excluded three quarters of the
    table would be worse than a smaller number honestly stated.
    """
    from vesma.keyer import rdkit_mol_canonical_smiles
    from vesma.rdkit_matcher import RdkitReactionMatcher, rdkit_reactions_from_smarts

    wanted = tuple(
        channel for channel in channels if not channel.startswith("competitors")
    )
    unimolecular = tuple(
        rule for rule, arity in enumerate(forward.arities) if arity == 1
    )
    narrowed = {
        retro: tuple(rule for rule in found if rule in unimolecular)
        for retro, found in forward.index.items()
    }
    dropped = sum(
        len(forward.index[retro]) - len(found) for retro, found in narrowed.items()
    )
    if dropped:
        print(
            f"[forward] rdkit runs unimolecular rules only, so {dropped} pairings are unprobed "
            f"({len(unimolecular)} of {len(forward.smarts)} forward rules are unimolecular)",
            flush=True,
        )

    matcher = RdkitReactionMatcher(rdkit_reactions_from_smarts(forward.smarts))

    def key(mols):
        return [rdkit_mol_canonical_smiles(mol) for mol in mols]

    # No `combine`, so only unimolecular applications are probed -- a batch gather, which every
    # batch supports -- and the competitor channel was refused above.
    return channel_probes(
        replace(forward, index={r: f for r, f in narrowed.items() if f}),
        wanted,
        reports,
        matcher,
        key=key,
        combine=None,
        seconds=seconds,
    )


def rdkit_arm(
    library: Library,
    stock: Stock,
    *,
    heuristic_kind: str,
    scan_kind: str,
    weight: float,
    n_jobs: int | None,
    forward: ForwardLibrary | None = None,
    channels: Collection[str] = (),
    vetoes: VetoLayer | None = None,
) -> Arm:
    from rdkit import Chem, RDLogger

    # Invalid intermediate products are expected in a broad template search.
    # Emitting each RDKit diagnostic during a timed run overwhelms stderr and
    # changes the measurement without adding useful per-target information.
    RDLogger.DisableLog("rdApp.*")

    from vesma.keyer import rdkit_mol_canonical_smiles, rdkit_smiles_keyer
    from vesma.matcher import all_vs_all, keyed_match
    from vesma.rdkit_matcher import (
        RdkitFilterCatalogReactionMatcher,
        rdkit_reactions_from_smarts,
    )

    seconds: dict[str, float] = {}
    started = perf_counter()
    reactions = rdkit_reactions_from_smarts(library.smarts)
    seconds["template_parse"] = perf_counter() - started

    started = perf_counter()
    matcher = RdkitFilterCatalogReactionMatcher(
        reactions,
        [smarts.split(">>")[0] for smarts in library.smarts],
        n_jobs=n_jobs,
    )
    seconds["matcher_build"] = perf_counter() - started

    started = perf_counter()
    keys = frozenset(stock.rdkit_keys)
    heuristic = heuristic_of(
        heuristic_kind,
        lambda: ecfp4_stock(
            stock.rdkit_fingerprints,
            weight=weight,
            n_bits=ECFP4_BITS,
            scan=scan_of(scan_kind, stock.rdkit_fingerprints),
        ),
        rdkit_atom_counts,
    )
    seconds["stock_index"] = perf_counter() - started

    lifted = expansion(
        keyed_match(
            all_vs_all(matcher),
            lambda mols: [rdkit_mol_canonical_smiles(mol) for mol in mols],
        )
    )
    probe_reports: list[ProbeReport] = []
    probe_seconds: dict[str, float] = {}
    mixture_reports: list[MixtureReport] = []
    veto_reports: list[VetoReport] = []
    overlap_reports: list[OverlapReport] | None = [] if VETO_OVERLAP else None
    deficit_reports: list[DeficitReport] = []
    match = layered_match(
        lifted,
        probes=(
            rdkit_probes(forward, channels, probe_reports, probe_seconds)
            if forward is not None
            else {}
        ),
        vetoes=vetoes,
        veto_reports=veto_reports,
        # No screen: a veto pattern is query SMARTS compiled by `vesma.parsing`, and this arm holds
        # RDKit molecules, so evaluating one here would mean a second pattern compiler with its own
        # reading of the same SMARTS. `layered_match` refuses rather than measuring nothing, for the
        # same reason this arm says out loud which forward rules it cannot run.
        make_screen=None,
        combine=None,
        sizes=rdkit_atom_counts,
        deficit_reports=deficit_reports,
    )
    return Arm(
        backend="rdkit",
        stock=keys,
        heuristic=heuristic,
        match=match,
        matcher=matcher,
        veto_rationales=(
            {veto.code: veto.rationale for veto in vetoes.table}
            if vetoes is not None
            else {}
        ),
        probe_reports=probe_reports,
        probe_seconds=probe_seconds,
        mixture_reports=mixture_reports,
        veto_reports=veto_reports,
        overlap_reports=overlap_reports if overlap_reports is not None else [],
        deficit_reports=deficit_reports,
        step_mixture=(
            # This arm keys its search by canonical SMILES, so the report's idea of "the
            # same molecule" is the arm's rather than the other arms'.
            step_mixture(
                forward,
                cmol_rdkit_smiles_keyer.encode,
                cmol_rdkit_smiles_keyer.encode,
                mixture_reports.append,
            )
            if forward is not None
            else None
        ),
        retro_codes=tuple(
            source.columns.get("template_code", "") for source in library.sources
        ),
        forward_codes=() if forward is None else forward.codes,
        rule_yields={} if forward is None else forward.yields,
        stack_mols=object_mol_array,
        concat_mols=concat_object_batches,
        key_of=rdkit_smiles_keyer.from_smiles,
        mol_of=Chem.MolFromSmiles,
        setup_seconds=seconds,
        stock_size=len(keys),
        stock_bytes=int(stock.rdkit_fingerprints.nbytes),
    )


ARMS = {"cpu": cpu_arm, "metal": metal_arm, "rdkit": rdkit_arm}


# --------------------------------------------------------------------------- one search


@dataclass(frozen=True)
class TargetResult:
    backend: str
    target: str
    solved: bool
    seconds: float
    width: int | str
    pass_limit: int
    passes: int
    molecules_expanded: int
    molecules_seen: int
    routes: int
    unique_route_count: int
    unique_route_count_capped: bool
    best_reaction_count: int | None
    heuristic_queries: int
    heuristic_calls: int
    phase_seconds: dict[str, float]
    matcher_totals: dict[str, float]
    route_trees: tuple[str, ...] = ()
    readable_route_trees: tuple[str, ...] = ()
    route_records: tuple[dict[str, Any], ...] = ()
    route_selection: dict[str, Any] | None = None
    route_selection_seconds: float = 0.0
    probe_channel_seconds: dict[str, float] = field(default_factory=dict)
    """Seconds the forward probe spent, per channel, on this target alone.

    Inside the `match` phase rather than beside it, so it is never added to the other phases.
    """
    mixture_calls: int = 0
    """Reported steps whose mixture block was derived -- one `expected_mixture` call each."""
    mixture_rules: int = 0
    """Forward rules applied across those calls, and so the parse count the derivation costs.

    A count, not a duration: it reads the same on a loaded host as on an idle one, which makes it
    the trustworthy half of this measurement and an exact before/after for the parse itself.
    """
    mixture_rules_by_group: dict[str, int] = field(default_factory=dict)
    replay_failures: int = 0
    replay_variants: int = 0
    first_disconnection_counts: tuple[dict[str, Any], ...] = ()
    depth_route_profile: dict[str, Any] | None = None
    error: str | None = None
    width_schedule: tuple[int, ...] = ()
    selected_per_pass: tuple[int, ...] = ()
    # How often the selector's `fill` ran at all, and how many of its offered leaves
    # survived into a batch. Instrumentation for whether the heuristic's global
    # ranking reaches the decision or the commitment saturates the batch first.
    fill_calls: int = 0
    fill_contributed: int = 0
    route_ranking: dict[str, Any] | None = None
    """What the selectivity price changed about the report, against the same graph unpriced.

    `None` when no penalty was charged, so there was no second ranking to compare against.
    """
    forward_coverage: dict[str, Any] | None = None
    """What the forward probe covered, summed over this target's expansion layers.

    `None` when no probe ran, which is the default. A run that *did* probe has to state its own
    coverage, because an application whose rule has no forward counterpart is **unprobed** and pays
    nothing -- and a reader who cannot see how many of those there were would have no way to tell a
    selective rule set from an unexamined one.
    """
    rule_attribution: dict[str, Any] | None = None
    """Per forward rule and channel, offered and fired counts over this target's layers.

    `None` unless `AOSTAR_HOST_VETO_OVERLAP` asked for it -- it rides the same flag because it
    answers the same question from the other side: the overlap says *where* the competitor channel
    fires, and this says *which* rule fired there.
    """
    disputed_measurements: dict[str, int] | None = None
    """Per channel, disconnections whose measurement contradicted itself and was reduced.

    `None` when nothing disagreed. Reported beside the channel's own coverage because that is
    whose defect it is -- the store only asked the question. A non-empty tally means that channel
    priced nothing on those disconnections, so it is coverage this run did not have.
    """
    validation: dict[str, Any] | None = None
    """What validating candidate routes at replay refused, when that mode is on.

    `routes_lost` against `routes_offered` is the number to read: large, while routes are missing,
    says to turn validation off, get all the routes back, and audit the veto table. That is the
    whole reason it is printed rather than merely counted.
    """
    mass_balance: dict[str, Any] | None = None
    """What the no-mass-loss check found over this target's layers, and which rules were short.

    `None` when the check was off, which is the default. In `measure` mode the counts are a
    prediction, exactly as the veto tally's are.
    """
    veto_overlap: dict[str, Any] | None = None
    """How each veto row's condemnations were judged by the competitor channel, per row.

    `None` unless `AOSTAR_HOST_VETO_OVERLAP` asked for it. A row whose condemnations are all
    `competed` is one the other channel already finds and could price instead; a row whose
    condemnations are `silent` is one only the veto catches.
    """
    veto_tally: dict[str, Any] | None = None
    """What the veto table condemned over this target's layers, per row, and in which mode.

    `None` when no veto ran, which is the default. In `measure` mode the counts are a *prediction*:
    they say what a refusing run would have removed from a graph that was never changed, which is
    the only way to read a row's cost before paying it.
    """
    price_sweep: dict[str, Any] | None = None
    """Per candidate price shape, what it did to this target's report. `None` unless asked for.

    One search, several prices: extraction is a pure function of the finished graph, so each entry
    differs from the others by the price and by nothing else -- no second matcher pass, no second
    heuristic pass, and none of the confounds that make two searches incomparable.
    """
    alternatives: dict[str, Any] | None = None
    """The per-retro-step report: every other disconnection of every molecule on every route.

    Carried on the result but written to its *own* file. It is an order of magnitude larger than
    everything else here put together -- one entry per disconnection per molecule per route -- and
    `targets.jsonl` is read with `jq` for timings and counts. See :func:`append_alternatives`.
    """


def sweep_price(
    shape: str, state: GraphState[Any], yields: Mapping[int, float]
) -> StepPrice:
    """One sweep entry's price, in whichever reading it names.

    `steps`, `flat:<m>` and `graded:<s>[:<c>]` are step-multiples through
    :func:`vertsy.costs.selective`, and an entry without a `/` applies one shape to both channels --
    the spelling every recorded sweep used.

    `yield` is the run's configured yield price, and `yield:<sites>[/<competitors>]` an explicit
    curve. A yield entry carries the **per-rule base** as well as the curve, because that is what
    the price being compared actually is: swapping only the curve and leaving a flat base would
    measure half the change and attribute it to the whole.

    A bare `yield` takes the run's own channels rather than applying one curve to both, since
    `YIELD_COMPETITORS` defaults off and duplicating the sites curve onto it would silently compare
    a two-channel price against the one-channel price that ships.
    """
    if shape == "yield" or shape.startswith("yield:"):
        rest = shape.removeprefix("yield").lstrip(":")
        if rest:
            sites, _, competitors = rest.partition("/")
            competitors = competitors or "none"
        else:
            sites, competitors = YIELD_SITES, YIELD_COMPETITORS
        return compounded(
            by_rule(yield_base(yields), uniform=False),
            state.outcomes,
            # The sweep varies the two count channels; severity stays at the run's own setting, so
            # a shape comparison is not silently also a severity comparison.
            factors_of(sites, competitors, SEVERITY_PRICE) or {},
        )
    sites, _, competitors = shape.partition("/")
    return selective(
        constant_step(cost_of_factor(ROUTE_STEP_FACTOR)),
        state.outcomes,
        penalties_of(sites, competitors or sites) or {},
    )


def shape_comparison(
    shape: str,
    *,
    target: int,
    state: GraphState[Any],
    expansions: Expansions[Any],
    smarts: Sequence[str],
    unpriced: RouteSelection,
    selector: Callable[..., RouteSelection],
    retro_codes: Sequence[str],
    yields: Mapping[int, float],
) -> dict[str, Any]:
    """What one price shape does to one target's report, and whether the veto table agrees.

    `vetoes_on_the_top_route` is the part worth reading and the only criterion here that is
    *independent of the thing being tuned*. The two selectivity channels and the price are one
    mechanism: asking whether a shape lowers the price it itself assigns is circular. The veto
    rows are hand-written chemistry, written against named failures and read by neither channel,
    so a shape whose top route fires fewer of them has been measured against something it cannot
    have been fitted to.

    It is evidence and not a verdict. A row firing on a recommended step is a *prediction* that
    the step is bad, as good as the row, and the counts are small -- six targets produced five such
    steps. Read it as a direction, and read the before/after trees beside it.
    """
    price = sweep_price(shape, state, yields)
    selected = selector(price=price)
    report = route_report(
        target,
        state,
        selected,
        price=price,
        rule_label=retro_rule_label(retro_codes),
    )
    fired = Counter(
        code
        for route in report.routes
        for step in route.retro_steps
        for code in step.chosen.vetoes_fired
    )
    top = Counter(
        code
        for step in (report.routes[0].retro_steps if report.routes else ())
        for code in step.chosen.vetoes_fired
    )
    return ranking_change(selected, unpriced, expansions, smarts) | {
        "vetoes_on_the_top_route": dict(top.most_common()),
        "vetoes_on_any_reported_route": dict(fired.most_common()),
        # Steps, not codes, and the two differ: one step can name two rows. A *disputed* row is
        # reported as a candidate rather than a fact -- two products from the same materials
        # disagreed about it -- so one condemned step can name both candidates, and summing the
        # code counts then reads as two condemned steps. What is being compared is how many
        # recommended steps a hand-written row objects to, so that is what is counted.
        "condemned_steps_on_the_top_route": sum(
            1
            for step in (report.routes[0].retro_steps if report.routes else ())
            if step.chosen.vetoes_fired
        ),
        "condemned_steps_on_any_reported_route": sum(
            1
            for route in report.routes
            for step in route.retro_steps
            if step.chosen.vetoes_fired
        ),
        "routes": len(selected.routes),
    }


def ranking_change(
    priced: RouteSelection,
    unpriced: RouteSelection,
    expansions: Expansions,
    smarts: Sequence[str],
    readable: Callable[[Route[Any]], Route[Any]] = lambda route: route,
) -> dict[str, Any]:
    """What the selectivity penalty did to the *report*, on one graph, at no search cost.

    The pair is the cleanest test the mechanism admits: extraction is a pure function of the
    graph, so the two reports differ by the price and by nothing else — no second matcher pass,
    no second heuristic pass, and none of the confounds that make two searches incomparable. A
    run that changed no report has measured that the penalty is inert at this scale, which is a
    result; a run that changed one has a before/after pair a chemist can read.

    Trees are rendered only for a *changed* top route, so the payload stays proportional to what
    actually happened rather than to how many targets were tried. They go through ``readable``
    because on a device arm the route's own products are WL hashes, and a before/after pair that
    a chemist cannot read is not the deliverable.

    ``chemistry_changed`` is the one a chemist would call a change. `collapsed_route_key` carries
    the rule id, so a route whose disconnection is reached by two co-canonical templates reads as
    changed the moment the price picks the *selective* one — which is the pricing working exactly
    as designed and not a different route. Dropping the label separates the two, and both are
    reported because each answers a different question: whether the credited template moved, and
    whether the molecules did.
    """

    def keys(selection: RouteSelection, labelled: bool = True) -> tuple[Any, ...]:
        return tuple(
            route_key(decode_route(item.route, expansions), labelled=labelled)
            for item in selection.routes
        )

    def tree(selection: RouteSelection) -> str:
        return render_route(
            readable(decode_route(selection.routes[0].route, expansions)), smarts
        )

    before, after = keys(unpriced), keys(priced)
    molecules_before, molecules_after = (
        keys(unpriced, labelled=False),
        keys(priced, labelled=False),
    )
    changed = bool(before[:1] != after[:1])
    return {
        "top_changed": changed,
        "chemistry_changed": bool(molecules_before[:1] != molecules_after[:1]),
        "order_changed": before != after,
        "set_changed": set(before) != set(after),
        "reactions_before": unpriced.routes[0].score.reactions if before else None,
        "reactions_after": priced.routes[0].score.reactions if after else None,
        "utility_before": unpriced.routes[0].score.utility if before else None,
        "utility_after": priced.routes[0].score.utility if after else None,
    } | (
        {"tree_before": tree(unpriced), "tree_after": tree(priced)}
        if changed and before and after
        else {}
    )


def route_key(route: Route[Any], *, labelled: bool = True) -> Any:
    """Route identity, with or without the template credited for each reaction.

    Unlabelled, this is the chemistry: the tree of molecules, stoichiometry-free the same way
    `collapsed_route_key` is. It exists because the two differ in exactly the interesting case --
    one disconnection, two co-canonical templates, and a price that prefers one of them.
    """
    if labelled:
        return collapsed_route_key(route)
    return (
        route.product,
        frozenset(route_key(child, labelled=False) for child in route.precursors),
    )


def render_route(route: Route[Any], smarts: Sequence[str], indent: str = "") -> str:
    """One route as an indented tree of products, each under the template that makes it.

    ``vertsy.routes.pretty_string`` prints the rule *index*, which is the right thing for a
    benchmark comparing runs and the wrong thing for a chemist reading the route — so the template
    is resolved to its retro SMARTS here. Metal's native route still contains WL hashes; its
    separate ``readable_route_trees`` are produced by guided Vesma CPU template replay and use
    canonical SMILES.
    """
    lines = [f"{indent}{route.product}"]
    if route.rule is None:
        return "\n".join(lines)
    rule = route.rule
    template = smarts[rule] if 0 <= rule < len(smarts) else "<rule out of range>"
    lines.append(f"{indent}  <- rule {rule}: {template}")
    lines.extend(
        render_route(child, smarts, indent + "    ") for child in route.precursors
    )
    return "\n".join(lines)


#: Phases that run *inside* the search interval `TargetResult.seconds` measures. Their sum is
#: bounded by it, and what it does not cover is `orchestration`.
#:
#: `match` is **not** here, and that is the structure rather than an omission: it runs inside
#: `expand`, which is `vertsy.search.step`. Before `prepare_graph` and `expand` were injected and
#: timed, these were the only four names and `orchestration` carried 26-30% of the search with
#: nothing able to say what was in it.
SEARCH_PHASES = ("prepare_graph", "expand", "heuristic", "fold", "select")
#: Phases that run *after* the search, so entirely outside `TargetResult.seconds`. This is what
#: "how expensive is extraction?" asks about, and until these timers existed the answer came only
#: from subtracting the search total from pytest-benchmark's wall clock.
EXTRACTION_PHASES = (
    "selection",
    "selection_unpriced",
    "route_counting",
    "replay",
    "report",
)
#: Phase -> the phase it runs inside, nested: `forward_probe` is within `match` is within
#: `expand`. These are measured for their own sake and must never be added to either total --
#: counting them twice is the one way this breakdown can lie.
COMPONENT_PHASES = {
    "match": "expand",
    "forward_probe": "match",
    "mixture": "report",
}


def timed(
    phase_seconds: dict[str, float], name: str, call: Callable[..., R]
) -> Callable[..., R]:
    def measured(*args: Any, **kwargs: Any) -> R:
        started = perf_counter()
        result = call(*args, **kwargs)
        phase_seconds[name] = phase_seconds.get(name, 0.0) + perf_counter() - started
        return result

    return measured


@dataclass
class FillShare:
    """How much of each selected batch came from a selector's ``fill``, rather than its reserve.

    Instrumentation, kept separable from the search it measures: it wraps the selector's ``fill``
    on the way in and is shown each batch on the way out, so removing it is deleting the two lines
    that construct and consult it. Nothing in :mod:`vertsy.aostar` knows it exists.

    The question it answers is whether a heuristic reaches the decision at all.
    ``committed_application_leaves`` asks each committed branch for the whole batch and returns
    early once full, which reads as though ``fill`` -- the only place the global ranking acts --
    might rarely run. Measured, it runs on every pass and supplies all but a couple of leaves,
    because ``descent_leaves`` offers what one solution graph exposes rather than what ``limit``
    allows.
    """

    calls: int = 0
    contributed: int = 0
    _offered: list[list[int]] = field(default_factory=list)

    def wrap(self, inner: Any) -> Any:
        """The ``fill`` to hand the selector."""

        def counting(*args: Any, **kwargs: Any) -> list[int]:
            self.calls += 1
            offered = inner(*args, **kwargs)
            self._offered.append(list(offered))
            return offered

        return counting

    def note(self, selected: Sequence[int]) -> None:
        """Record what survived from this pass's offers into the batch."""
        offered = {sid for row in self._offered for sid in row}
        self.contributed += sum(1 for sid in selected if sid in offered)
        self._offered.clear()


def run_target(
    arm: Arm,
    smiles: str,
    width: int | str,
    passes: int,
    smarts: Sequence[str] = (),
    *,
    selection: Select | None = None,
) -> TargetResult:
    """One target, one effort shape, on the 0.001 -> 1.0 anytime schedule with a max fold."""
    widths = effort_widths(width, passes)
    selections: list[int] = []
    phases: dict[str, float] = {}
    totals: Counter[str] = Counter()
    expansions = Expansions(NumpyExpansionStore(), DictInterner())
    fill_share = FillShare()
    # The arm is built once for the whole sweep and its probe reports only ever grow, so a target's
    # own coverage is the tail appended while it ran. Reading the whole list made every target after
    # the first report the sweep so far, which looks like coverage improving with time.
    probed_before = len(arm.probe_reports)
    vetoed_before = len(arm.veto_reports)
    overlap_before = len(arm.overlap_reports)
    deficit_before = len(arm.deficit_reports)
    refused_before = sum(arm.validation_refusals.values())
    mixtures_before = len(arm.mixture_reports)
    probe_seconds_before = dict(arm.probe_seconds)
    price, scheduled = anytime_step(
        greedy=GREEDY_STEP,
        relaxed=RELAXED_STEP,
        select=(
            committed_application_leaves(fill=fill_share.wrap(global_leaves))
            if selection is None
            else selection
        ),
    )
    # `None` when the probe is off, or when it is on and deliberately not priced -- which fills the
    # store and the coverage report while leaving every cost exactly as it was, so the probe's wall
    # clock can be measured separately from its effect.
    substrate = substrate_of()

    select_calls = 0
    measured_select = timed(phases, "select", scheduled)

    def select(*args: Any, **kwargs: Any) -> list[int]:
        nonlocal select_calls
        select_calls += 1
        selected = measured_select(*args, **kwargs)
        selections.append(len(selected))
        fill_share.note(selected)
        return selected

    queries = 0
    heuristic_calls = 0
    measured_heuristic = timed(phases, "heuristic", arm.heuristic)

    def heuristic(molecules: Any) -> np.ndarray:
        nonlocal queries, heuristic_calls
        queries += len(molecules)
        heuristic_calls += 1
        return measured_heuristic(molecules)

    measured_match = timed(phases, "match", arm.match)

    def match(molecules: Any) -> Any:
        result = measured_match(molecules)
        totals.update(arm.diagnostics())
        return result

    started = perf_counter()
    try:
        target_key = arm.key_of(smiles)
        target_mol = arm.mol_of(smiles)
    except ChemistryRejection as exc:
        # An unsupported input is a result for this benchmark, not a reason to
        # discard every completed target in the sweep. `rejected_target` below is the same
        # answer for a refusal the *search* raises -- a backend can refuse the frontier and not
        # only the keying, which `metal_stack_mols` does for a target over its 64-atom bucket --
        # so both are returned rather than one aborting the sweep with a tidier message.
        return TargetResult(
            backend=arm.backend,
            target=smiles,
            solved=False,
            seconds=perf_counter() - started,
            width=width,
            pass_limit=passes,
            passes=0,
            fill_calls=0,
            fill_contributed=0,
            molecules_expanded=0,
            molecules_seen=0,
            routes=0,
            unique_route_count=0,
            unique_route_count_capped=False,
            best_reaction_count=None,
            heuristic_queries=0,
            heuristic_calls=0,
            phase_seconds={},
            matcher_totals={},
            error=f"{type(exc).__name__}: {exc}",
            width_schedule=widths,
        )
    # `max_routes=0` skips extraction: this call is only a solved/unsolved probe, and
    # `state.is_solved` below answers that off the graph without a cartesian product, a second
    # valuation, or a third rule-labelling convention.
    try:
        aostar(
            target_key,
            target_mol=target_mol,
            stock=arm.stock,
            expansions=expansions,
            heuristic=heuristic,
            match=match,
            stack_mols=arm.stack_mols,
            concat_mols=arm.concat_mols,
            width=widths,
            passes=passes,
            max_depth=passes,
            select=select,
            fold=timed(phases, "fold", bottleneck_fold),
            # The pass loop's own two costs, which `orchestration` used to absorb whole.
            # `prepare` runs once per pass on a graph that grows every pass; `expand` contains
            # `match`, so the taxonomy below treats `match` as a component of it.
            prepare=timed(phases, "prepare_graph", prepare_graph),
            expand=timed(phases, "expand", expansion_step),
            price=price,
            substrate=substrate,
            max_routes=0,
        )
    except ChemistryRejection as exc:
        # The search can refuse what the keying accepted: `aostar` packs the target through
        # `stack_mols`, and `metal_stack_mols` refuses a molecule over its 64-atom bucket. Caught
        # here for the same reason the keying refusal above is -- one target is a result, and a
        # sweep of a hundred and fifty should not end because the forty-first is too large.
        return TargetResult(
            backend=arm.backend,
            target=smiles,
            solved=False,
            seconds=perf_counter() - started,
            width=width,
            pass_limit=passes,
            passes=0,
            fill_calls=0,
            fill_contributed=0,
            molecules_expanded=0,
            molecules_seen=0,
            routes=0,
            unique_route_count=0,
            unique_route_count_capped=False,
            best_reaction_count=None,
            heuristic_queries=0,
            heuristic_calls=0,
            phase_seconds={},
            matcher_totals={},
            error=f"{type(exc).__name__}: {exc}",
            width_schedule=widths,
        )
    elapsed = perf_counter() - started
    # The verdict channels are named here, by the module that minted them, for the reason the
    # `reduce` policy is named beside the columns: a veto read on the product is not a property of
    # the chemistry, so it needs the step-keyed index to be attributed to one disconnection.
    state = prepare_graph(
        expansions,
        arm.stock,
        # The per-row columns only. The two aggregates stay under the chemistry key, because a
        # price reads `discouraged` there and the report's unnamed fallback reads `vetoed` there;
        # a per-row verdict is read nowhere but the report, and only ever about one product.
        step_keyed=[
            channel
            for channel in expansions.channels()
            if channel.startswith("vetoed:")
        ],
    )
    target_sid = int(expansions.lookup([target_key])[0])
    selection_started = perf_counter()
    # The report's own price, stated as a cost rather than a factor: `-log(ROUTE_STEP_FACTOR)`.
    # It is the same shape the search folds with, so a substrate penalty added to one is added to
    # both, and the two can no longer drift into separate parameterisations of one number.
    route_price = (
        by_rule(yield_base(arm.rule_yields), uniform=False)
        if PRICE == "yield"
        else constant_step(cost_of_factor(ROUTE_STEP_FACTOR))
    )
    # Two modes judge candidates, and they judge the same table by different means. `validate`
    # asks replay, which is the only way a `geometry` row can be answered and costs a replay per
    # candidate. `withhold` asks the column the search already filled, which costs a lookup and
    # cannot answer a geometry row -- every shipped row is `cost="pattern"`, so nothing needs the
    # expensive one today. In `measure` and `refuse` nothing judges here: `refuse` already pruned
    # the graph, and `measure` is the mode that acts nowhere by definition.
    validate: Callable[[SidRoute], str | None] | None = None
    if VETOES == "validate":
        validate = route_validator(arm, expansions, smiles)
    elif VETOES == "withhold":
        # The same question, asked of the column the search already filled rather than of a
        # replay: no molecules, no screen, one dict lookup a step. It refuses for the row the
        # report prints beside it, because both go through `fired_vetoes`.
        validate = stored_veto_validator(state)
    if substrate is not None:
        # The same shape over the same measurements, composed from the same place the search
        # composes it. Only the base differs, which is the one thing the search and the report were
        # always entitled to disagree about -- and under `yield` even that narrows, since the base
        # is then the reaction's own curated fraction rather than a schedule.
        route_price = substrate(route_price, state.outcomes)
    selected = (
        select_solved_routes(
            target_sid,
            state,
            limit=ROUTE_REPORT_CAP,
            depth_slack=ROUTE_DEPTH_SLACK,
            cost_slack=ROUTE_COST_SLACK,
            price=route_price,
            candidate_limit=ROUTE_CANDIDATE_LIMIT,
            state_limit=ROUTE_STATE_LIMIT,
            diversity_cutoff=ROUTE_DIVERSITY_CUTOFF,
            validate=validate,
        )
        if state.is_solved(target_sid)
        else None
    )
    selection_seconds = perf_counter() - selection_started
    chosen = selected.routes if selected is not None else ()
    routes = tuple(decode_route(item.route, expansions) for item in chosen)

    # The paired comparison, and the reason it is nearly free: extraction is a *pure function of the
    # graph*, so asking what the report would have said without the penalty needs no second search --
    # no second matcher pass, no second heuristic pass, and no confound from a different graph. One
    # more `select_solved_routes` over the same state is the whole cost, and `selection_unpriced`
    # now measures it rather than asserting it: ~20 ms against seconds of search. Not to be read as
    # the cost of extraction, which `EXTRACTION_PHASES` totals at two orders of magnitude more.
    def report_selection(*, price: StepPrice) -> RouteSelection:
        """This target's report under one price, with every other setting held fixed."""
        return select_solved_routes(
            target_sid,
            state,
            limit=ROUTE_REPORT_CAP,
            depth_slack=ROUTE_DEPTH_SLACK,
            cost_slack=ROUTE_COST_SLACK,
            price=price,
            candidate_limit=ROUTE_CANDIDATE_LIMIT,
            state_limit=ROUTE_STATE_LIMIT,
            diversity_cutoff=ROUTE_DIVERSITY_CUTOFF,
            validate=validate,
        )

    unpriced = (
        timed(phases, "selection_unpriced", select_solved_routes)(
            target_sid,
            state,
            limit=ROUTE_REPORT_CAP,
            depth_slack=ROUTE_DEPTH_SLACK,
            cost_slack=ROUTE_COST_SLACK,
            price=constant_step(cost_of_factor(ROUTE_STEP_FACTOR)),
            candidate_limit=ROUTE_CANDIDATE_LIMIT,
            state_limit=ROUTE_STATE_LIMIT,
            diversity_cutoff=ROUTE_DIVERSITY_CUTOFF,
            validate=validate,
        )
        if selected is not None and substrate is not None
        else None
    )
    # One search, several prices. The baseline is the *unpriced* report, so each shape's entry
    # says what that shape did to a report nothing was charged on.
    price_sweep = (
        {
            shape: shape_comparison(
                shape,
                target=target_sid,
                state=state,
                expansions=expansions,
                smarts=smarts,
                unpriced=unpriced,
                selector=report_selection,
                retro_codes=arm.retro_codes,
                yields=arm.rule_yields,
            )
            for shape in PENALTY_SWEEP
        }
        if PENALTY_SWEEP and unpriced is not None
        else None
    )
    first_disconnections = timed(
        phases,
        "route_counting",
        precursor_route_counts_by_first_disconnection(
            target_key, ROUTE_COUNT_CAP, passes
        ),
    )(state)
    depth_profile = timed(
        phases,
        "route_counting",
        precursor_route_depth_profile(target_key, ROUTE_COUNT_CAP),
    )(state)
    unique_route_count = (
        1
        if target_sid in state.stock_sids
        else min(ROUTE_COUNT_CAP, sum(count for _, count in first_disconnections))
    )
    readable_routes: list[tuple[Route[str], dict[str, Any]]] = []
    replay_failures = 0
    replay_variants = 0

    # Both replay callers go through this, so the phase covers the whole of replay rather than
    # whichever of the two happened to be wrapped. `None` stays `None`: an arm with no replay hook
    # must not grow a timer that reports zero as if it had run.
    replay_routes = (
        timed(phases, "replay", arm.replay_routes)
        if arm.replay_routes is not None
        else None
    )

    def first_replayed(route: Route[Any]) -> Route[Any]:
        """The route with canonical SMILES where replay can supply them, else as it came.

        Falling back rather than raising: a route that cannot be replayed is still worth showing
        as hashes plus its templates, and a report that vanished because one variant failed to
        reproduce would be the worse failure.
        """
        if replay_routes is None:
            return route
        variants = replay_routes(route, smiles, 1, None)
        return variants[0] if variants else route

    # Kept beside `readable_routes` rather than zipped with it afterwards: a route that failed to
    # replay is missing from that list, so position alone would pair a sid route with another
    # route's molecules.
    replayed_pairs: list[tuple[SidRoute, Route[Any]]] = []
    for item, route in zip(chosen, routes):
        selection = asdict(item.score) | {
            "first_distinct_decision": item.first_distinct_decision
        }
        if replay_routes is None:
            readable_routes.append((route, selection))
            continue
        mismatches: list[ReplayMismatch[Any]] = []
        replayed = replay_routes(route, smiles, REPLAY_CAP, mismatches)
        replay_variants += len({collapsed_route_key(one) for one in replayed})
        if replayed:
            readable_routes.append((replayed[0], selection))
            replayed_pairs.append((item.route, replayed[0]))
        else:
            replay_failures += 1
            # A dropped route otherwise leaves only a count. Report the first node that
            # could not be reproduced, and whether the two sides disagree on arity.
            for mismatch in mismatches[:1]:
                print(
                    f"    replay mismatch: {mismatch}"
                    f"{'  <-- ARITY' if mismatch.arity_differs else ''}",
                    flush=True,
                )
    # Built from the finished graph and the same price the report ranked with, so what it says an
    # alternative would have cost is what the selector saw, not a second opinion.
    alternatives = (
        asdict(
            timed(phases, "report", route_report)(
                target_sid,
                state,
                selected,
                price=route_price,
                # Every channel this run measured, not the two this function defaults to.
                channels=arm.channels,
                rule_label=retro_rule_label(arm.retro_codes),
                name=molecule_name(
                    alternative_names(
                        replayed_pairs,
                        state,
                        cost=partial(rule_price, route_price),
                        key_of=arm.key_of,
                        name_precursors=arm.precursor_names,
                    )
                    | replayed_names(
                        [route for route, _ in readable_routes], arm.key_of
                    )
                    if arm.precursor_names is not None
                    else replayed_names(
                        [route for route, _ in readable_routes], arm.key_of
                    )
                ),
                # Timed separately *inside* `report`, because it is the one part of building a
                # report that re-runs chemistry rather than reading the graph -- and the whole
                # question was whether that re-run is affordable.
                mixture=(
                    timed(phases, "mixture", arm.step_mixture)
                    if arm.step_mixture is not None
                    else None
                ),
                diversity_cutoff=ROUTE_DIVERSITY_CUTOFF,
                # The table's own rationale column, which nothing read until now. Only the rows a
                # report cites are kept on it, so a stored report stays about its own target.
                veto_rationale=arm.veto_rationales,
                rule_yield=arm.rule_yields,
            )
        )
        # The report names its target as replay canonicalised it; the row the sweep was
        # given is what joins back to `targets.jsonl`, so both are kept.
        | {"input_smiles": smiles}
        if selected is not None
        else None
    )
    # This target's own share of two arm-side accumulators that grow across the whole sweep.
    mixtures = arm.mixture_reports[mixtures_before:]
    probe_now = {
        channel: total - probe_seconds_before.get(channel, 0.0)
        for channel, total in arm.probe_seconds.items()
    }
    return TargetResult(
        backend=arm.backend,
        target=smiles,
        solved=state.is_solved(target_sid),
        seconds=elapsed,
        width=width,
        pass_limit=passes,
        passes=select_calls,
        fill_calls=fill_share.calls,
        fill_contributed=fill_share.contributed,
        molecules_expanded=int(expansions.store.expanded().size),
        molecules_seen=expansions.sid_count(),
        routes=len(routes),
        unique_route_count=unique_route_count,
        unique_route_count_capped=unique_route_count == ROUTE_COUNT_CAP,
        best_reaction_count=(
            min(item.score.reactions for item in chosen) if chosen else None
        ),
        heuristic_queries=queries,
        heuristic_calls=heuristic_calls,
        phase_seconds=phases
        | {
            # The one timer that predates this breakdown, folded in so a single mapping holds
            # every phase; the standalone field below stays so runs recorded before it still
            # compare.
            "selection": selection_seconds,
            # A component of `match`, not an addition to it -- the probe runs inside it. Summed
            # over channels, and the per-channel split is in `probe_channel_seconds`.
            "forward_probe": sum(probe_now.values()),
        },
        probe_channel_seconds=probe_now,
        mixture_calls=len(mixtures),
        mixture_rules=sum(one.rules for one in mixtures),
        mixture_rules_by_group={
            "same_reaction": sum(one.same_reaction for one in mixtures),
            "same_site": sum(one.same_site for one in mixtures),
            "different_site": sum(one.different_site for one in mixtures),
            "unknown_site": sum(one.unknown_site for one in mixtures),
        },
        matcher_totals=dict(totals),
        width_schedule=widths,
        selected_per_pass=tuple(selections),
        route_trees=tuple(render_route(route, smarts) for route in routes),
        readable_route_trees=tuple(
            render_route(route, smarts) for route, _ in readable_routes
        ),
        route_records=tuple(
            archive_route(route, smarts) | {"selection": selection}
            for route, selection in readable_routes
        ),
        route_selection=(
            {
                "candidates_considered": selected.candidates_considered,
                "partial_states_explored": selected.partial_states_explored,
                "truncated": selected.truncated,
                "stopped_after_limit": selected.stopped_after_limit,
                "shortest_depth": selected.shortest_depth,
                "best_shortest_reactions": selected.best_shortest_reactions,
                "depth_slack": ROUTE_DEPTH_SLACK,
                "cost_slack": ROUTE_COST_SLACK,
                "step_factor": ROUTE_STEP_FACTOR,
                "candidate_limit": ROUTE_CANDIDATE_LIMIT,
                "state_limit": ROUTE_STATE_LIMIT,
            }
            if selected is not None
            else None
        ),
        route_selection_seconds=selection_seconds,
        route_ranking=(
            ranking_change(
                selected, unpriced, expansions, smarts, readable=first_replayed
            )
            if selected is not None and unpriced is not None
            else None
        ),
        forward_coverage=(
            channel_coverage(arm.probe_reports[probed_before:])
            if len(arm.probe_reports) > probed_before
            else None
        ),
        veto_tally=(
            veto_tally(arm.veto_reports[vetoed_before:])
            if len(arm.veto_reports) > vetoed_before
            else None
        ),
        alternatives=alternatives,
        price_sweep=price_sweep,
        disputed_measurements=dict(sorted(expansions.disputed.items())) or None,
        validation=(
            {
                "mode": VETOES,
                "applications_refused": sum(arm.validation_refusals.values())
                - refused_before,
                "routes_offered": len(readable_routes) + replay_failures,
                "routes_lost": replay_failures,
                # Refused during extraction and *replaced* by the next diverse candidate, which is
                # the difference between this and `routes_lost` above.
                "routes_replaced": len(selected.censored) if selected else 0,
                "refusals": (
                    dict(
                        Counter(item.reason for item in selected.censored).most_common()
                    )
                    if selected
                    else {}
                ),
            }
            if VETOES == "validate"
            else None
        ),
        mass_balance=(
            deficit_tally(arm.deficit_reports[deficit_before:], arm.retro_codes)
            if len(arm.deficit_reports) > deficit_before
            else None
        ),
        veto_overlap=(
            overlap_tally(arm.overlap_reports[overlap_before:])
            if len(arm.overlap_reports) > overlap_before
            else None
        ),
        rule_attribution=(
            {
                channel: rule_attribution(
                    arm.probe_reports[probed_before:], arm.forward_codes, channel
                )
                for channel in ("sites", "competitors")
                + (
                    ("competitors:destructive", "competitors:severity")
                    if SEVERITY
                    else ()
                )
            }
            if VETO_OVERLAP and len(arm.probe_reports) > probed_before
            else None
        ),
        replay_failures=replay_failures,
        replay_variants=replay_variants,
        first_disconnection_counts=tuple(
            {
                "precursor_keys": [repr(key) for key in precursor_keys],
                "route_count": count,
            }
            for precursor_keys, count in first_disconnections
        ),
        depth_route_profile={
            "shortest_depth": depth_profile.shortest_depth,
            "counts": [asdict(count) for count in depth_profile.counts],
            "first_disconnections": [
                {
                    "precursor_keys": [repr(key) for key in keys],
                    "counts": [asdict(count) for count in counts],
                }
                for keys, counts in depth_profile.first_disconnections
            ],
        },
    )


def summarize(results: Sequence[TargetResult]) -> dict[str, Any]:
    solved = [result for result in results if result.solved]
    total = sum(result.seconds for result in results)
    names = sorted({name for result in results for name in result.phase_seconds})
    # Only the targets that were actually re-ranked are the denominator: a target with no solved
    # route has no report to change, and counting it as unchanged would dilute the rate with
    # targets the price was never offered.
    ranked = [r.route_ranking for r in results if r.route_ranking is not None]
    return {
        "targets": len(results),
        "solved_targets": len(solved),
        "reranked_targets": len(ranked),
        "top_route_changed": sum(bool(r["top_changed"]) for r in ranked),
        "top_route_chemistry_changed": sum(
            bool(r["chemistry_changed"]) for r in ranked
        ),
        "route_order_changed": sum(bool(r["order_changed"]) for r in ranked),
        "unsupported_targets": sum(result.error is not None for result in results),
        "solve_rate": len(solved) / len(results),
        "search_seconds": total,
        "latency_seconds_mean": float(np.mean([r.seconds for r in results])),
        "latency_seconds_median": float(np.median([r.seconds for r in results])),
        "expanded_per_target_mean": float(
            np.mean([r.molecules_expanded for r in results])
        ),
        "seen_per_target_mean": float(np.mean([r.molecules_seen for r in results])),
        "heuristic_queries_per_target_mean": float(
            np.mean([r.heuristic_queries for r in results])
        ),
        # The two totals the phase taxonomy separates, so a reader does not have to know which
        # names belong to which group. `extraction_seconds` is the number the whole breakdown was
        # added to produce: post-search work, which no timer attributed before.
        "search_phase_seconds": sum(
            r.phase_seconds.get(name, 0.0) for r in results for name in SEARCH_PHASES
        ),
        "extraction_seconds": sum(
            r.phase_seconds.get(name, 0.0)
            for r in results
            for name in EXTRACTION_PHASES
        ),
        # Counts, which say the same thing whatever else the host is doing.
        "mixture_calls": sum(r.mixture_calls for r in results),
        "mixture_rules": sum(r.mixture_rules for r in results),
        "phase_seconds": {
            name: sum(r.phase_seconds.get(name, 0.0) for r in results) for name in names
        },
    }


def append_record(path: Path, result: TargetResult, label: str) -> None:
    """One JSON object per line, flushed before the next target starts.

    ``--benchmark-json`` writes once at the end, so a run interrupted on target 19 of 22 loses all
    nineteen. A long sweep over hard targets is exactly the run that gets interrupted — by a laptop
    lid, a rebuild, or a target that takes an hour — so each result is durable the moment it exists.
    Append-only and one object per line, so a partial write costs the last line rather than the file
    and ``jq`` reads it while the run is still going.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(result)
    # The alternatives report rides on the result but belongs in its own file; see that field.
    row.pop("alternatives", None)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"label": label} | row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# How many targets per arm get their report rendered as Markdown beside the JSONL. The JSONL is
# the artefact and holds every target; this is the readable copy, for a chemist or a judge, and a
# 9,999-target sweep would otherwise write hundreds of megabytes of prose nobody opens.
ALTERNATIVES_MARKDOWN_TARGETS = 20


def append_alternatives_markdown(path: Path, result: TargetResult) -> None:
    """The same report as prose, rendered from the JSON rather than from the live objects.

    Through the file format on purpose: it is the one path that is exercised every run, so a report
    that cannot be read back cannot quietly keep working here and fail for everyone else.
    """
    if result.alternatives is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    page = render_report_markdown(report_from_json(result.alternatives))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(page + "\n\n---\n\n")
        handle.flush()


def append_alternatives(path: Path, result: TargetResult, label: str) -> None:
    """The alternatives report for one target, appended and flushed like the records beside it.

    Separate from `targets.jsonl` for size, and written per target for the same reason that one is:
    a sweep that is killed on target 19 keeps the eighteen reports it already paid for.
    """
    if result.alternatives is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"label": label, "backend": result.backend} | result.alternatives
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def archived_rule_ids(records: Sequence[Mapping[str, Any]]) -> frozenset[int]:
    """Every rule id appearing anywhere in a target's archived route trees."""
    found: set[int] = set()
    pending = list(records)

    while pending:
        node = pending.pop()

        if node.get("template_id") is not None:
            found.add(int(node["template_id"]))

        pending.extend(node.get("precursors", ()))

    return frozenset(found)


def template_rows(library: Library, rule_ids: Collection[int]) -> list[dict[str, Any]]:
    """The side table that turns a route's rule index back into something nameable.

    Restricted to the rules the routes actually used, following the projection the legacy
    archive already applied: naming needs only those, and the full library is rebuildable from
    the manifest's hashed inputs.

    `columns` is the source row verbatim. Resolving it to a display name is the report's job,
    because the answer differs per source and the rule set should not have to know how it will
    be printed.
    """
    return [
        {
            "rule_id": rule_id,
            "retro_template": library.smarts[rule_id],
            "source": library.sources[rule_id].source,
            "row": library.sources[rule_id].row,
            "columns": dict(library.sources[rule_id].columns),
        }
        for rule_id in sorted(rule_ids)
    ]


def write_run_directory(
    run_dir: Path,
    *,
    manifest: Mapping[str, Any],
    library: Library,
    results: Sequence[TargetResult],
) -> None:
    """Everything a report needs, beside the records the run already appended.

    Written after the search rather than during it: `targets.jsonl` is the durable part and is
    fsynced per target, while these two are derived and cheap to rewrite. A run interrupted
    before this point still has its molecules.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    used = (
        frozenset().union(
            *(archived_rule_ids(result.route_records) for result in results)
        )
        if results
        else frozenset()
    )

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run_dir / "alternatives-summary.json").write_text(
        json.dumps(alternatives_summary(results), indent=2) + "\n"
    )
    sweep = price_sweep_summary(results)
    if sweep:
        (run_dir / "price-sweep-summary.json").write_text(
            json.dumps(sweep, indent=2) + "\n"
        )
    (run_dir / "templates.json").write_text(
        json.dumps(template_rows(library, used), indent=2) + "\n"
    )


def report(result: TargetResult, label: str, index: int, count: int) -> None:
    measured = sum(result.phase_seconds.get(name, 0.0) for name in SEARCH_PHASES)
    extraction = sum(result.phase_seconds.get(name, 0.0) for name in EXTRACTION_PHASES)
    error = f" error={result.error!r}" if result.error else ""
    print(
        f"[{label} {index}/{count}] done {result.seconds:.2f}s "
        f"solved={result.solved} sampled_routes={result.routes} "
        f"unique_routes={'at-least-' if result.unique_route_count_capped else ''}"
        f"{result.unique_route_count} replay_failures={result.replay_failures} "
        f"replay_variants={result.replay_variants} "
        f"best_reactions={result.best_reaction_count} "
        f"expanded={result.molecules_expanded} seen={result.molecules_seen} "
        f"passes={result.passes}/{result.pass_limit} "
        f"fill={result.fill_calls}/{result.passes}calls,"
        f"{result.fill_contributed}mols "
        f"h_queries={result.heuristic_queries} "
        f"prepare={result.phase_seconds.get('prepare_graph', 0.0):.2f}s "
        f"expand={result.phase_seconds.get('expand', 0.0):.2f}s "
        f"heuristic={result.phase_seconds.get('heuristic', 0.0):.2f}s "
        f"fold={result.phase_seconds.get('fold', 0.0):.2f}s "
        f"select={result.phase_seconds.get('select', 0.0):.2f}s "
        f"orchestration={max(result.seconds - measured, 0.0):.2f}s "
        # Outside `seconds` entirely, so stated as its own total and as a share of the search
        # rather than folded in. This is the answer to "how expensive is extraction".
        f"| extraction={extraction:.2f}s"
        f"({extraction / max(result.seconds, 1e-9):.1%} of search) "
        f"selection={result.phase_seconds.get('selection', 0.0):.2f}s "
        f"unpriced={result.phase_seconds.get('selection_unpriced', 0.0):.2f}s "
        f"counting={result.phase_seconds.get('route_counting', 0.0):.2f}s "
        f"replay={result.phase_seconds.get('replay', 0.0):.2f}s "
        f"report={result.phase_seconds.get('report', 0.0):.2f}s "
        # Components, named with their parent so nobody adds them to a total.
        f"| match={result.phase_seconds.get('match', 0.0):.2f}s(in expand) "
        f"probe={result.phase_seconds.get('forward_probe', 0.0):.2f}s(in match) "
        f"mixture={result.phase_seconds.get('mixture', 0.0):.2f}s(in report) "
        f"mixture_calls={result.mixture_calls} "
        f"mixture_rules={result.mixture_rules}{error}",
        flush=True,
    )
    tally = result.veto_tally
    if tally is not None:
        offered = max(1, tally["applications"])
        rows = ", ".join(f"{code}={count}" for code, count in tally["fired"].items())
        print(
            f"    vetoes ({tally['mode']}): condemned {tally['condemned']} of "
            f"{tally['applications']} applications ({tally['condemned'] / offered:.2%}), "
            f"unexamined {tally['unexamined']} over {tally['layers']} layers; "
            f"{rows or 'no row fired'}",
            flush=True,
        )
    disputed = result.disputed_measurements
    if disputed:
        print(
            "    disputed measurements (channel contradicted itself, reduced and not priced): "
            + ", ".join(f"{channel}={count}" for channel, count in disputed.items()),
            flush=True,
        )
    validation = result.validation
    if validation is not None:
        offered = max(1, validation["routes_offered"])
        print(
            f"    validation ({validation['mode']}): refused "
            f"{validation['applications_refused']} application(s) during replay, losing "
            f"{validation['routes_lost']} of {validation['routes_offered']} candidate routes "
            f"({validation['routes_lost'] / offered:.0%})"
            + (
                " -- if routes are missing, re-run with AOSTAR_HOST_VETOES=off to get them all"
                " back and audit the table"
                if validation["routes_lost"]
                else ""
            ),
            flush=True,
        )
    mass = result.mass_balance
    if mass is not None:
        offered = max(1, mass["applications"])
        worst = ", ".join(
            f"{code}={count}" for code, count in list(mass["by_rule"].items())[:8]
        )
        print(
            f"    mass ({mass['mode']}, slack={mass['slack']}): "
            f"{mass['unaccounted']} of {mass['applications']} applications "
            f"({mass['unaccounted'] / offered:.2%}) cannot account for their product, "
            f"{mass['atoms']} atoms over {len(mass['by_rule'])} rules; "
            f"{worst or 'none'}",
            flush=True,
        )
    overlap = result.veto_overlap
    if overlap is not None and overlap["rows"]:
        for channel in sorted(next(iter(overlap["rows"].values()))):
            parts = [
                f"{name} "
                + " ".join(f"{b}={cells[channel][b]}" for b in OVERLAP_BUCKETS)
                for name, cells in overlap["rows"].items()
                if any(cells[channel].values())
            ]
            if parts:
                print(f"    veto/{channel}: " + "; ".join(parts), flush=True)
    ranking = result.route_ranking
    if ranking is not None:
        print(
            f"    ranking: top_changed={ranking['top_changed']} "
            f"chemistry_changed={ranking['chemistry_changed']} "
            f"order_changed={ranking['order_changed']} "
            f"reactions {ranking['reactions_before']} -> {ranking['reactions_after']}",
            flush=True,
        )
        if ranking.get("tree_before"):
            print("    -- reported without the selectivity price:", flush=True)
            print(indented(ranking["tree_before"]), flush=True)
            print("    -- reported with it:", flush=True)
            print(indented(ranking["tree_after"]), flush=True)


def indented(text: str, prefix: str = "      ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def skip_unless_metal() -> None:
    from vesma.metal_matcher import GpuUnavailableError, probe_metal

    try:
        probe_metal()
    except (GpuUnavailableError, RuntimeError) as exc:
        pytest.skip(f"Metal backend not available: {exc}")


def test_bench_host_aostar(benchmark: Any) -> None:
    rule_sets = parse_inputs(
        env_text("AOSTAR_HOST_RULE_SET", "raw68k"),
        TEMPLATE_FILES,
        "AOSTAR_HOST_RULE_SET",
    )
    # `common-reagents` is in the default because the rule set now depends on it. Forty-six
    # retro rows name the reagent that supplies what they leave implicit -- NCS for a
    # chlorination, borohydride for a ketone reduction -- so that reagent is a precursor the
    # search has to be able to buy. A run against the Enamine catalogue alone would silently
    # lose all forty-six: it is a screening-compound catalogue and carries nothing below
    # benzyl-sized, which is the same way it already killed every methylation on its trivial
    # partner. Naming a reagent and not stocking it is the one combination that fails quietly.
    stocks = parse_inputs(
        env_text("AOSTAR_HOST_STOCK", "enamine,common-reagents"),
        STOCK_FILES,
        "AOSTAR_HOST_STOCK",
    )
    # Kept as a synonym for trailing entries of AOSTAR_HOST_RULE_SET rather than removed: it
    # names files that are one SMARTS per line and carry no columns, which the rule-set list
    # also accepts, and every existing Makefile invocation passes it.
    extra_templates = tuple(
        Path(name.strip()).expanduser()
        for name in env_text("AOSTAR_HOST_EXTRA_TEMPLATES", "").split(",")
        if name.strip()
    )
    # Defaults to a vendored PaRoutes subset so the bench runs standalone, with
    # AOSTAR_HOST_TARGET_LIMIT (1 by default) keeping that run small. Campaign targets live
    # outside the repository and are passed in; nothing here names one.
    targets_file = Path(
        env_text("AOSTAR_HOST_TARGETS_FILE", str(DEFAULT_TARGETS))
    ).expanduser()
    missing = [
        path
        for path in (
            *(path for _, path in rule_sets),
            *(path for _, path in stocks),
            targets_file,
            *extra_templates,
        )
        if not path.exists()
    ]
    if missing:
        pytest.skip(f"missing benchmark inputs: {missing}")

    chosen = backends()
    if "metal" in chosen:
        skip_unless_metal()

    efforts = parse_efforts(env_text("AOSTAR_HOST_EFFORTS", "64x16"))
    # `atoms` by default since 2026-09-29: measured on both benchmarks it reaches the same
    # targets as `stock` while interning half as many molecules and finishing about 20% sooner.
    # `tests/vertsy/perf_heuristic_signal.py` carries the numbers and the caveat -- most of the
    # halving is a preference for smaller molecules rather than better guidance.
    heuristic_kind = env_text("AOSTAR_HOST_HEURISTIC", "atoms")
    scan_kind = env_text("AOSTAR_HOST_SCAN", default_scan_kind())
    weight = env_float("AOSTAR_HOST_WEIGHT", HEURISTIC_WEIGHT)
    n_jobs = env_int("AOSTAR_HOST_JOBS", 1)
    stock_limit = env_int("AOSTAR_HOST_STOCK_LIMIT", 0) or None
    # A run directory holds everything a report needs; the records file is one of its members.
    # Both stay configurable because the older flat-file invocation is still valid and some
    # sweeps only want the JSONL.
    configured_run_dir = env_text("AOSTAR_HOST_RUN_DIR", "")
    run_dir = Path(configured_run_dir).expanduser() if configured_run_dir else None
    configured_records = env_text("AOSTAR_HOST_RECORDS_FILE", "")

    if configured_records:
        records_file = Path(configured_records).expanduser()
    elif run_dir is not None:
        records_file = run_dir / "targets.jsonl"
    else:
        records_file = None

    # The repository ships templates and stock, never a campaign's output. A targets file from
    # outside the checkout *is* a campaign, so its records and run directory belong outside too.
    # Enforced rather than documented, because the comment above saying so was followed only most
    # of the time: campaign runs left 48 MB of hard-22 and hard-7 output under `output/` before
    # anyone noticed. `output/` is at least ignored -- the Makefile's default `BENCH_DIR` is
    # `docs/paper/benchmarks`, which is tracked, so the same slip there would have been committed.
    repo_root = Path(__file__).resolve().parents[2]
    if not targets_file.resolve().is_relative_to(repo_root):
        inside = [
            str(path)
            for path in (run_dir, records_file)
            if path is not None and path.resolve().is_relative_to(repo_root)
        ]
        if inside:
            raise ValueError(
                f"targets come from outside the repository ({targets_file}), so this run's "
                f"output belongs outside it too, but these paths are inside: {inside}. Point "
                "AOSTAR_HOST_RUN_DIR and AOSTAR_HOST_RECORDS_FILE -- and the Makefile's "
                "BENCH_DIR, which pytest-benchmark writes its own JSON under -- at the "
                "campaign's own directory."
            )

    targets = read_targets(
        targets_file,
        env_int("AOSTAR_HOST_TARGET_LIMIT", 1),
        env_int("AOSTAR_HOST_TARGET_OFFSET", 0),
    )
    if not targets:
        pytest.skip(f"no targets read from {targets_file}")

    # Off by default, and imported inside the branch: `vertsy.preprocess` needs the `rdkit`
    # extra, which is optional, and a benchmark should not acquire a dependency for a step most
    # runs do not want. Targets are searched exactly as supplied unless this is asked for.
    preprocessing = "none"
    prepared: tuple[str, ...] = targets

    if env_text("AOSTAR_HOST_PREPROCESS", "") in ("1", "true", "yes"):
        from vertsy.preprocess import prepare_targets

        prepared_targets = prepare_targets(targets)
        prepared = tuple(target.searched for target in prepared_targets)
        preprocessing = (
            "; ".join(
                f"{index}: {target.note}"
                for index, target in enumerate(prepared_targets, start=1)
                if target.note != "none"
            )
            or "no target needed preparation"
        )
        print(f"target preprocessing: {preprocessing}", flush=True)

    targets = prepared

    library = load_merged_library(rule_sets, extra_templates)
    slices = stock_slices([path for _, path in stocks], stock_limit)
    started = perf_counter()
    labelled = prepare_labelled_slices(slices, n_jobs)
    stock = merge_stocks([part for _, part in labelled])
    # Only a report reads this, and only for the handful of molecules a route ends on, so it is
    # built once here rather than per target. It is a dictionary over keys that already exist.
    stock_seconds = perf_counter() - started
    print(
        f"rule_sets={library.source_counts} rows={library.row_count} "
        f"rules={library.input_count} "
        f"extra_templates={tuple(zip(library.extra_paths, library.extra_counts))} "
        f"rejected_mass_losing={library.rejected_count} "
        f"rejected_duplicate={library.duplicate_count} compiled={len(library.smarts)} "
        f"stock={[name for name, _ in stocks]} rows={stock.row_count} "
        f"usable={len(stock.wl_keys)} "
        f"refused={stock.refused} {stock.refusals} prepared_in={stock_seconds:.1f}s "
        f"heuristic={heuristic_kind} scan={scan_kind} weight={weight} jobs={n_jobs} "
        "select=committed-application+global "
        f"backends={chosen} efforts={efforts} targets={len(targets)} "
        f"records_file={records_file}",
        flush=True,
    )

    forward = load_forward_library(library.sources) if FORWARD_PROBE else None
    # Every channel the probe knows, not only the ones a penalty was asked for: a channel priced
    # `none` is still measured, which is how its wall clock and its coverage are read without
    # letting it move a route. What a given arm can actually run is its own business -- the RDKit
    # arm narrows this to `sites` over unimolecular rules, and says so.
    channels = ("sites", "competitors") if FORWARD_PROBE else ()
    if FORWARD_PROBE and SEVERITY:
        # Both readings of one probe: the count says how often and which rows, the severity says
        # how bad, and only the second is shaped for a price. Keeping the count is what makes the
        # decision to price severity checkable against the thing it replaces.
        channels += ("competitors:destructive", "competitors:severity")
    if FORWARD_PROBE and SPLIT_COMPETITORS:
        channels += ("competitors:same-site", "competitors:different-site")
    if forward is not None:
        print(
            f"[forward] {len(forward.smarts)} rules pair with "
            f"{forward.paired_rules} of {len(library.sources)} compiled templates "
            f"({forward.paired_rules / max(1, len(library.sources)):.1%}); "
            + (
                f"price=yield sites={YIELD_SITES} competitors={YIELD_COMPETITORS}"
                if PRICE == "yield"
                else f"price=steps sites={PENALTY_SITES} competitors={PENALTY_COMPETITORS}"
            ),
            flush=True,
        )
        if PRICE == "yield":
            # How much of the price is curated and how much is the catch-all. The number that says
            # whether the yield column is done, and it belongs beside the run rather than in a
            # separate report: a route priced mostly on `YIELD_DEFAULT` is a route priced on one
            # number somebody picked, however carefully.
            uncurated_rules = len(library.sources) - len(forward.yields)
            print(
                f"[forward] {len(forward.yields)} of {len(library.sources)} rules carry a curated "
                f"yield; {uncurated_rules} take the {YIELD_DEFAULT} catch-all "
                f"({uncurated_rules / max(1, len(library.sources)):.1%})",
                flush=True,
            )
        if "rdkit" in chosen:
            print(
                "[forward] the rdkit arm runs unimolecular forward rules only, so the "
                f"competitor channel and {sum(1 for a in forward.arities if a >= 2)} of "
                f"{len(forward.arities)} rows are unavailable there; cpu and metal run all of them",
                flush=True,
            )

    vetoes = load_veto_layer(library.sources, VETOES)
    if vetoes is not None:
        print(
            f"[vetoes] {len(vetoes.table)} rows "
            f"({', '.join(veto.code for veto in vetoes.table)}) in scope for "
            f"{vetoes.scoped_rules} of {len(library.sources)} compiled templates; "
            f"mode={VETOES}"
            + ("" if vetoes.refuse else " -- measured, nothing refused"),
            flush=True,
        )

    arms = {}
    for backend in chosen:
        started = perf_counter()
        built = ARMS[backend](
            library,
            stock,
            heuristic_kind=heuristic_kind,
            scan_kind=scan_kind,
            weight=weight,
            n_jobs=n_jobs,
            **({"forward": forward, "channels": channels} if forward else {}),
            **({"vetoes": vetoes} if vetoes is not None else {}),
        )
        # `replace` rather than a factory argument: provenance is a reporting concern, and every
        # arm builder would otherwise thread a parameter none of them uses. The notes have to be
        # keyed by *this* arm's keyer, which is why they are built here and not beside the stock.
        arms[backend] = replace(built, channels=channels)
        print(
            f"[{backend}] setup {perf_counter() - started:.1f}s "
            f"stock_keys={arms[backend].stock_size} "
            f"stock_fingerprints_mb={arms[backend].stock_bytes / 1e6:.0f} "
            f"{arms[backend].setup_seconds}",
            flush=True,
        )

    def run() -> tuple[TargetResult, ...]:
        results = []
        for backend in chosen:
            for width, passes in efforts:
                label = f"{backend} {width}x{passes}"
                for index, smiles in enumerate(targets, start=1):
                    print(
                        f"[{label} {index}/{len(targets)}] start {smiles}", flush=True
                    )
                    result = run_target(
                        arms[backend], smiles, width, passes, library.smarts
                    )
                    results.append(result)
                    report(result, label, index, len(targets))
                    if records_file is not None:
                        append_record(records_file, result, label)
                    if run_dir is not None:
                        append_alternatives(
                            run_dir / "alternatives.jsonl", result, label
                        )
                        if index <= ALTERNATIVES_MARKDOWN_TARGETS:
                            append_alternatives_markdown(
                                run_dir / "alternatives.md", result
                            )
        return tuple(results)

    results = benchmark.pedantic(
        run, rounds=env_int("BENCH_ROUNDS", 1), warmup_rounds=0, iterations=1
    )
    summaries = {
        f"{backend}-{width}x{passes}": summarize(
            tuple(
                result
                for result in results
                if result.backend == backend
                and result.width == width
                and result.pass_limit == passes
            )
        )
        for backend in chosen
        for width, passes in efforts
    }
    benchmark.extra_info.update(
        {
            "experiment": "host-aostar-anytime",
            "select": "committed-application+global",
            "backends": chosen,
            "rule_sets": tuple(name for name, _ in rule_sets),
            "rule_set_contributions": library.source_counts,
            "stock": tuple(name for name, _ in stocks),
            "stock_rows": stock.row_count,
            "stock_usable_rows": len(stock.wl_keys),
            "stock_refusals": stock.refusals,
            "stock_prepare_seconds": stock_seconds,
            "stock_limit": stock_limit,
            "efforts": tuple(
                {
                    "width": width,
                    "passes": passes,
                    "max_depth": passes,
                    "width_schedule": effort_widths(width, passes),
                }
                for width, passes in efforts
            ),
            "greedy_step": GREEDY_STEP,
            "relaxed_step": RELAXED_STEP,
            "fold": "max",
            # The report's own parameters. They decide which routes are ranked at all, so a run
            # that does not state them cannot be compared with one that used different ones.
            "route_step_factor": ROUTE_STEP_FACTOR,
            "route_depth_slack": ROUTE_DEPTH_SLACK,
            "route_cost_slack": ROUTE_COST_SLACK,
            "forward_probe": FORWARD_PROBE,
            "penalty_sites": PENALTY_SITES if FORWARD_PROBE else None,
            "penalty_competitors": PENALTY_COMPETITORS if FORWARD_PROBE else None,
            "route_candidate_limit": ROUTE_CANDIDATE_LIMIT,
            "route_state_limit": ROUTE_STATE_LIMIT,
            "heuristic": heuristic_kind,
            "heuristic_scan": scan_kind,
            "heuristic_weight": weight,
            "fingerprint": asdict(FINGERPRINT),
            "template_row_count": library.row_count,
            "extra_template_paths": tuple(str(path) for path in library.extra_paths),
            "extra_template_counts": library.extra_counts,
            "template_count_input": library.input_count,
            "template_count_rejected": library.rejected_count,
            "template_count_compiled": len(library.smarts),
            "rejected_template_indices": library.rejected_indices,
            # Both drop reasons, so compiled index -> source row can be inverted. Only the
            # rejected set was recorded before, which made the mapping one-way.
            "template_count_duplicate": library.duplicate_count,
            "duplicate_template_indices": library.duplicate_indices,
            "setup_seconds": {name: arm.setup_seconds for name, arm in arms.items()},
            "stock_sizes": {name: arm.stock_size for name, arm in arms.items()},
            "targets": targets,
            "targets_file": str(targets_file),
            "records_file": str(records_file) if records_file else None,
            "summaries": summaries,
            "records": [asdict(result) for result in results],
            "inputs": {
                "stock": tuple(source_artifact(path) for _, path in stocks),
                "templates": tuple(source_artifact(path) for _, path in rule_sets),
                "extra_templates": tuple(
                    source_artifact(path) for path in extra_templates
                ),
                "targets": source_artifact(targets_file),
            },
        }
    )
    if run_dir is not None:
        write_run_directory(
            run_dir,
            # The benchmark's own extra_info, minus the per-route bulk: a manifest is for
            # reproducing and naming a run, and `records` is already in targets.jsonl.
            manifest={
                key: value
                for key, value in benchmark.extra_info.items()
                if key != "records"
            }
            | {
                "target_preprocessing": preprocessing,
                "template_library": ", ".join(name for name, _ in rule_sets),
            },
            library=library,
            results=results,
        )
        print(f"run directory: {run_dir}", flush=True)

    for name, summary in summaries.items():
        print(f"{name}: {summary}", flush=True)
