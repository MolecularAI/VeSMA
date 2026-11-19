"""Render benchmark DataFrames into Typst tables for the paper.

Extraction lives in :mod:`paperlib.loaders`; this module holds the table renderers
(`qtable`, `qtable_mi`, `compact_multiindex_rows`), the cell formatters, and backend label
data. The ``load_*`` loaders are re-exported here so `benchmarks.qmd` keeps calling them as
`pt.load_*`; the notebook does its own filtering, pivoting, and throughput math on the frames.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from math import floor, isfinite, log10
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.io.formats.style import Styler
from paperlib.loaders import (
    explode_records,
    load_benchmarks,
    load_json,
    require_columns,
)

# Public API: the re-exported loaders plus the renderers/formatters used by benchmarks.qmd.
__all__ = [
    # loaders (re-exported from paperlib.loaders)
    "load_benchmarks",
    "explode_records",
    "load_json",
    "require_columns",
    "require_agreeing_records",
    "require_one_record",
    "require_unique_records",
    # rendering / formatting
    "qtable",
    "qtable_mi",
    "compact_multiindex_rows",
    "fraction_cell",
    "precursor_audit_agreement_table",
    "precursor_audit_class_table",
    "format_duration",
    "format_throughput",
    "format_significant",
    "format_microseconds",
    "with_rates",
    "select_records",
    "load_application",
    "scaling_cell",
    "scaling_table",
    "style_table",
    "backend_label",
    "backend_order",
    "host_label",
]

TABLE_SIZES = (1, 10, 100, 1000, 2881)
# Publication aggregates from the completed full CPU precursor audit at the frozen
# chemistry boundary. Keeping the small reduction here lets the main article and SI render
# from one tracked source without turning the 4.6 MB application-level audit into a paper
# build dependency.
PRECURSOR_AUDIT_PAIRS = 768_015
PRECURSOR_AUDIT_VESMA_APPLICATIONS = 2_468_381
PRECURSOR_AUDIT_RDKIT_OUTCOMES = 2_468_494
PRECURSOR_AUDIT_AGREEMENTS = (
    ("VeSMA-RDChiral", 755_047, 283, 754_780),
    ("VeSMA-RDKit", 710_592, 11_251, 705_147),
    ("RDKit-RDChiral", 711_864, 11_266, 706_189),
)

BACKEND_LABELS = {
    "cpu": "Vesma CPU",
    "cpu-parallel": "Vesma CPU parallel",
    "cpu-serial-gpu-parity": "Vesma CPU serial (GPU parity)",
    "cpu-parallel-gpu-parity": "Vesma CPU parallel (GPU parity)",
    "cpu-serial": "Vesma CPU serial",
    "cuda": "CUDA",
    "cuda-host": "CUDA host",
    "cuda-device": "CUDA device",
    "cuda-host-prefilter": "CUDA host + prefilter",
    "cuda-device-prefilter": "CUDA device + prefilter",
    "cuda-raw": "CUDA raw",
    "metal": "Metal",
    "metal-joined": "Metal, joined precursors",
    "metal-prefilter": "Metal w/ prefilter",
    "metal-prefilter-joined": "Metal w/ prefilter, joined precursors",
    "metal-raw": "Metal raw",
    "rdkit": "RDKit",
    "rdkit-serial": "RDKit",
    "rdkit-parallel": "RDKit parallel",
    "rdkit-filtercatalog-rules": "RDKit FilterCatalog",
    "rdkit-substructlibrary-frontier": "RDKit SubstructLibrary",
}
BACKEND_ORDER = {
    "rdkit-filtercatalog-rules": 0,
    "rdkit-substructlibrary-frontier": 1,
    "rdkit": 2,
    "rdkit-serial": 2,
    "rdkit-parallel": 2,
    "cpu": 3,
    "cpu-serial": 3,
    "cpu-serial-gpu-parity": 3,
    "cpu-parallel": 4,
    "cpu-parallel-gpu-parity": 4,
    "metal": 5,
    "metal-joined": 5,
    "metal-prefilter": 5,
    "metal-prefilter-joined": 5,
    "metal-raw": 6,
    "cuda-device": 7,
    "cuda-device-prefilter": 8,
    "cuda-host": 9,
    "cuda-host-prefilter": 10,
    "cuda": 11,
    "cuda-raw": 12,
}


def backend_label(backend: str) -> str:
    return BACKEND_LABELS[backend]


def backend_order(backend: str) -> int:
    return BACKEND_ORDER[backend]


def host_label(record: pd.Series, joiner: str = "plus") -> str:
    """Build a display label from the hardware metadata in a benchmark record."""
    cpu = str(
        record.get("machine.cpu.brand_raw") or record.get("machine.host_type") or ""
    )
    for noise in (" 64-Core Processor", " 32-Core Processor", " Processor"):
        cpu = cpu.replace(noise, "")
    cpu = " ".join(cpu.split())
    device = record.get("machine.cuda.device_name")
    if not device or pd.isna(device):
        return cpu
    return f"{cpu} {joiner} {device}"


def require_one_record(frame: pd.DataFrame, *, context: str) -> pd.Series:
    """Return the sole selected record or identify every conflicting JSON file."""
    if len(frame) == 1:
        return frame.iloc[0]
    if frame.empty:
        raise ValueError(f"{context}: expected exactly one record, found none")
    if "source_path" not in frame.columns:
        raise ValueError(
            f"{context}: expected exactly one record, found {len(frame)}; "
            "the selected rows do not include source_path"
        )
    counts = frame["source_path"].value_counts().sort_index()
    conflicts = "\n".join(
        f"- {path}" + (f" ({count} matching records)" if count > 1 else "")
        for path, count in counts.items()
    )
    raise ValueError(
        f"{context}: expected exactly one record, found {len(frame)}.\n"
        f"Conflicting benchmark JSON files:\n{conflicts}\n"
        "Keep the intended publication record and delete or move every other listed "
        "JSON file; the paper deliberately does not choose one implicitly."
    )


def require_agreeing_records(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    context: str,
    ignore_nested_keys: frozenset[str] = frozenset(),
) -> pd.Series:
    """Return one representative row when all selected result fields agree."""

    value_columns = list(columns)
    require_columns(
        frame,
        ["source_path", *value_columns],
        context=context,
    )
    if frame.empty:
        raise ValueError(f"{context}: expected at least one record, found none")

    def comparable(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: comparable(item)
                for key, item in sorted(value.items())
                if key not in ignore_nested_keys
            }
        if isinstance(value, list):
            return [comparable(item) for item in value]
        if isinstance(value, tuple):
            return tuple(comparable(item) for item in value)
        if hasattr(value, "item"):
            return value.item()
        return value

    def encoded(value: object) -> str:
        return json.dumps(
            comparable(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    reference = frame.iloc[0]
    reference_path = reference["source_path"]
    disagreements = []
    for _, record in frame.iloc[1:].iterrows():
        differing_fields = [
            column
            for column in value_columns
            if encoded(reference[column]) != encoded(record[column])
        ]
        if differing_fields:
            disagreements.append(
                f"- {record['source_path']}: {', '.join(differing_fields)}"
            )
    if disagreements:
        details = "\n".join(disagreements)
        raise ValueError(
            f"{context}: retained audit records disagree with "
            f"{reference_path}.\nDiffering records and fields:\n{details}"
        )
    return reference


def require_unique_records(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    context: str,
) -> None:
    """Reject duplicate experimental coordinates and list their source files."""
    keys = list(columns)
    require_columns(frame, [*keys, "source_path"], context=context)
    duplicates = frame[frame.duplicated(keys, keep=False)]
    if duplicates.empty:
        return
    conflicts = []
    for values, group in duplicates.groupby(keys, sort=True, dropna=False):
        coordinates = values if isinstance(values, tuple) else (values,)
        selector = ", ".join(
            f"{key}={value!r}" for key, value in zip(keys, coordinates, strict=True)
        )
        paths = "\n".join(
            f"  - {path}" for path in sorted(group["source_path"].unique())
        )
        conflicts.append(f"- {selector}\n{paths}")
    raise ValueError(
        f"{context}: duplicate benchmark records were selected:\n"
        + "\n".join(conflicts)
        + "\nKeep the intended publication record for each coordinate and delete or "
        "move the other listed JSON files; the paper deliberately does not "
        "choose one implicitly."
    )


def fraction_cell(recovered: Any, total: Any) -> str:
    """Render a required recovered/total count with its percentage."""
    if pd.isna(recovered) or pd.isna(total):
        raise ValueError(
            "fraction_cell requires non-missing recovered and total values"
        )
    recovered = int(recovered)
    total = int(total)
    pct = 0.0 if total == 0 else 100.0 * recovered / total
    return f"{recovered:,}/{total:,}\\\n({pct:.1f}%)"


def precursor_audit_agreement_table() -> pd.DataFrame:
    """Pairwise exact outcome agreement in the completed full precursor audit."""
    return pd.DataFrame(
        [
            {
                "Comparison": comparison,
                "All positive pairs": fraction_cell(equal, PRECURSOR_AUDIT_PAIRS),
                "Comparable pairs": fraction_cell(
                    equal_comparable,
                    PRECURSOR_AUDIT_PAIRS - incomparable,
                ),
            }
            for comparison, equal, incomparable, equal_comparable in (
                PRECURSOR_AUDIT_AGREEMENTS
            )
        ]
    )


def precursor_audit_class_table() -> pd.DataFrame:
    """Major semantic classes retained for interpretation in the SI."""

    def frequency(count: int, total: int, decimals: int = 2) -> str:
        return f"{count:,}/{total:,}\\\n({100 * count / total:.{decimals}f}%)"

    return pd.DataFrame(
        [
            {
                "Class": "Circuit-relative aromaticity",
                "Frequency in audit": "Focused regression example",
                "Chemical or semantic interpretation": (
                    "VeSMA and RDKit make different explicit graph-model choices for a "
                    "fused-ring bond. This is a difference between aromaticity models, "
                    "not a measurement of physical aromaticity."
                ),
            },
            {
                "Class": "Delocalized bond released from its circuit",
                "Frequency in audit": frequency(
                    29_215,
                    PRECURSOR_AUDIT_VESMA_APPLICATIONS,
                ),
                "Chemical or semantic interpretation": (
                    "VeSMA stores delocalized bonds categorically, without a hidden "
                    "single or double order to reveal after editing. It therefore refuses "
                    "an under-specified edit instead of choosing a localization."
                ),
            },
            {
                "Class": "Other typed chemistry refusal",
                "Frequency in audit": frequency(
                    5_218,
                    PRECURSOR_AUDIT_VESMA_APPLICATIONS,
                ),
                "Chemical or semantic interpretation": (
                    "The edited graph asks for a circuit outside automatic discovery or "
                    "for an undeclared, incompatible, or invalid closed-shell state. "
                    "An explicit refusal preserves the model's stated domain."
                ),
            },
            {
                "Class": "Derived RHS property differs",
                "Frequency in audit": frequency(
                    3_936,
                    PRECURSOR_AUDIT_VESMA_APPLICATIONS,
                ),
                "Chemical or semantic interpretation": (
                    "The RHS is applied as a graph rewrite. Derived SMARTS properties are "
                    "recomputed from the result, so the record is neutral telemetry rather "
                    "than a rejection or silent coercion."
                ),
            },
            {
                "Class": "Component has no RHS atom",
                "Frequency in audit": frequency(
                    540,
                    PRECURSOR_AUDIT_VESMA_APPLICATIONS,
                    decimals=3,
                ),
                "Chemical or semantic interpretation": (
                    "The template does not anchor that disconnected material on the RHS. "
                    "VeSMA omits the unanchored component and reports the event instead of "
                    "inventing its intended role."
                ),
            },
            {
                "Class": "Multi-component RHS meets a ring",
                "Frequency in audit": frequency(49_750, PRECURSOR_AUDIT_PAIRS),
                "Chemical or semantic interpretation": (
                    "The match lies outside the usual intermolecular reading of the "
                    "template. VeSMA conserves atoms while opening the ring; RDKit realizes "
                    "the declared product-component topology. Both are consistent readings "
                    "of an under-specified ring match."
                ),
            },
            {
                "Class": "Hydrogen completion after a local edit",
                "Frequency in audit": frequency(
                    1_773,
                    PRECURSOR_AUDIT_VESMA_APPLICATIONS,
                    decimals=3,
                ),
                "Chemical or semantic interpretation": (
                    "VeSMA completes hydrogen from the atom's preserved closed-shell "
                    "configuration after a local edit. The event is commoner than a visible "
                    "outcome difference: only four applications differ solely for it."
                ),
            },
            {
                "Class": "RDChiral stereochemical suppression",
                "Frequency in audit": frequency(5_026, PRECURSOR_AUDIT_PAIRS),
                "Chemical or semantic interpretation": (
                    "RDChiral applies stereochemical checks that VeSMA deliberately omits. "
                    "The count reflects different supported semantics, not disagreement on "
                    "the underlying non-stereochemical graph edit."
                ),
            },
            {
                "Class": "Outcome outside the shared comparison key",
                "Frequency in audit": (
                    "RDKit "
                    f"{frequency(17_140, PRECURSOR_AUDIT_RDKIT_OUTCOMES)}; "
                    "VeSMA "
                    f"{frequency(281, PRECURSOR_AUDIT_VESMA_APPLICATIONS, decimals=3)}"
                ),
                "Chemical or semantic interpretation": (
                    "The shared key requires RDKit sanitization and SMILES serialization. "
                    "Failure at that exchange boundary makes an outcome incomparable; it is "
                    "not, by itself, a verdict on the native chemistry result."
                ),
            },
            {
                "Class": "Open-shell molecule literal",
                "Frequency in audit": "Not exercised by this audit",
                "Chemical or semantic interpretation": (
                    "A bracketed atom in a SMILES literal is read as OpenSMILES "
                    "specifies, so [N] is atomic nitrogen rather than ammonia and TEMPO "
                    "is a nitroxide radical. RDKit and RDChiral read it the same way; "
                    "the divergence is that VeSMA's closed-shell model then refuses the "
                    "radical by name rather than building it. The audit imports "
                    "molecules through RDKit and therefore does not exercise the "
                    "literal path."
                ),
            },
        ]
    )


def format_duration(seconds: float, decimals: int = 1) -> str:
    """Format a wall-clock duration, auto-scaling to s / ms / µs.

    Used for the per-route and per-target average-time columns, whose values span
    milliseconds (a GPU or parallel-CPU replay) to hundreds of seconds (a single-target
    full-library search), so a fixed unit would either lose precision or read awkwardly.
    """
    if not isfinite(seconds):
        return f"{seconds} s"
    if seconds >= 1:
        return f"{seconds:.{decimals}f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.{decimals}f} ms"
    return f"{seconds * 1e6:.{decimals}f} µs"


def style_table(df: pd.DataFrame, caption: str) -> Styler:
    return (
        df.style.hide(axis="index")
        .set_caption(caption)
        .set_properties(**{"text-align": "left"})
    )


def qtable(
    df: pd.DataFrame,
    *,
    caption: str | None = None,
    table_id: str | None = None,
    colwidths: list[int] | None = None,
    aligns: str | None = None,
) -> str:

    attrs = [".list-table"]

    if table_id:
        attrs.insert(0, f"#{table_id}")

    if colwidths:
        attrs.append(f'tbl-colwidths="[{",".join(str(w) for w in colwidths)}]"')

    if aligns:
        attrs.append(f'aligns="{aligns}"')

    out = [f"::: {{{' '.join(attrs)}}}"]

    if caption:
        out += [caption, ""]

    def emit_cell(prefix: str, value: object) -> None:
        text = "" if pd.isna(value) else str(value)
        lines = text.splitlines() or [""]
        out.append(prefix + lines[0])
        for line in lines[1:]:
            out.append("    " + line)

    # Header
    columns = [str(c) for c in df.columns]
    emit_cell("- - ", columns[0])
    for col in columns[1:]:
        emit_cell("  - ", col)
    out.append("")

    # Body
    for row in df.itertuples(index=False, name=None):
        emit_cell("- - ", row[0])
        for value in row[1:]:
            emit_cell("  - ", value)
        out.append("")

    out.append(":::")
    return "\n".join(out)


MULTIPLIERS = {
    "": 1,
    "k": 1e3,
    "M": 1e6,
}


def format_throughput(
    value: float,
    scale: str = "",
    unit: str = "",
    decimals: int = 1,
) -> str:
    """
    Format throughput using a fixed scale, e.g. 3.5M pairs/s.

    For scaled values >= 1, uses `decimals` decimal places.
    For scaled values < 1, uses enough decimals to avoid displaying 0.0M.
    """

    if scale not in MULTIPLIERS:
        raise ValueError(
            f"Unknown scale {scale!r}. Expected one of {list(MULTIPLIERS)}"
        )

    if not isfinite(value):
        return f"{value} {unit}"

    scaled = value / MULTIPLIERS[scale]

    if scaled == 0:
        text = f"{0:.{decimals}f}"
    elif abs(scaled) >= 1:
        text = f"{scaled:.{decimals}f}"
    else:
        # enough decimal places to show at least one significant digit
        extra_decimals = -floor(log10(abs(scaled)))
        n_decimals = max(decimals, extra_decimals)
        text = f"{scaled:.{n_decimals}f}"

    return f"{text}{scale} {unit}"


def format_significant(value: float, digits: int = 3) -> str:
    """Round to ``digits`` significant figures, with thousands separators: 6781 -> 6,780."""
    if value == 0 or not isfinite(value):
        return f"{value:,}"
    ndigits = digits - 1 - floor(log10(abs(value)))
    return f"{round(value, ndigits):,.{max(0, ndigits)}f}"


def format_microseconds(seconds: float, decimals: int = 1) -> str:
    """A duration as a bare number of microseconds, for a column headed with the unit."""
    return f"{seconds * 1e6:,.{decimals}f}"


def with_rates(records: pd.DataFrame) -> pd.DataFrame:
    """Add each record's positive-pair throughput and its time per raw precursor, in seconds."""
    require_columns(
        records, ["positive_pairs", "precursors", "stats.mean"], context="rates"
    )
    return records.assign(
        pairs_per_second=records["positive_pairs"] / records["stats.mean"],
        seconds_per_precursor=records["stats.mean"] / records["precursors"],
    )


def load_application(root: Path) -> pd.DataFrame:
    """Load the streamed host-output workload used by the main performance table."""
    records = load_benchmarks(root)
    require_columns(
        records,
        ["contract", "precursor_identity", "delivery", "stats.rounds"],
        context="reaction application",
    )
    for column, expected in (
        ("experiment", "reaction-application"),
        ("contract", "streamed-unkeyed-host-v1"),
        ("precursor_identity", "none"),
        ("delivery", "host-batches-released"),
        ("stats.rounds", 3),
    ):
        if not records[column].eq(expected).all():
            raise ValueError(f"Reaction application requires {column}={expected!r}")
    require_unique_records(
        records,
        ["backend", "machine.host_type", "molecules"],
        context="reaction application",
    )
    for size, same_size in records.groupby("molecules"):
        require_agreeing_records(
            same_size,
            [
                "rules",
                "positive_pairs",
                "pair_sha256",
                "smiles_sha256",
                "templates_sha256",
            ],
            context=f"reaction-application inputs at {size} targets",
        )
        vesma = same_size[~same_size["backend"].str.startswith("rdkit")]
        require_agreeing_records(
            vesma,
            ["precursors", "precursor_atom_cap", "reaction_processing.after_split"],
            context=f"reaction-application VeSMA outputs at {size} targets",
        )
    return with_rates(records)


def select_records(
    records: pd.DataFrame, keys: Iterable[tuple], columns: list[str]
) -> pd.DataFrame:
    """The one record for each key, in the order the keys are given.

    ``keys`` are value tuples for ``columns``. A key matching no record or several raises,
    so a table never silently lacks a row or picks between duplicates.
    """
    wanted = pd.DataFrame(list(keys), columns=pd.Index(columns))
    merged = wanted.merge(records, on=columns, how="left", indicator=True)
    matches = merged.groupby(columns, sort=False)["_merge"].agg(
        lambda found: int((found == "both").sum())
    )
    if (matches != 1).any():
        raise ValueError(
            f"Expected exactly one record per key; matches found:\n{matches[matches != 1]}"
        )
    return merged.drop(columns="_merge")


def scaling_cell(record: pd.Series) -> str:
    """Wall time over throughput, as one table cell."""
    time = format_duration(float(record["stats.mean"]))
    rate = format_throughput(record["pairs_per_second"], "k", "p.pairs/s")
    return f"{time}\\\n({rate})"


def scaling_table(
    records: pd.DataFrame, *, index_name: str = "Backend and machine"
) -> pd.DataFrame:
    """Backends over machines as rows and workload sizes as columns, one cell per record.

    ``records`` need the ``with_rates`` columns. A backend-machine pair lacking a size
    leaves that cell empty.
    """
    require_unique_records(
        records, ["backend", "machine.host_type", "molecules"], context="scaling table"
    )
    labelled = records.assign(
        Backend=records["backend"].map(backend_label),
        Machine=records["machine.host_type"],
        cell=records.apply(scaling_cell, axis=1),
    )
    in_order = labelled.assign(order=labelled["backend"].map(backend_order))
    labelled["Backend"] = pd.Categorical(
        labelled["Backend"],
        categories=in_order.sort_values(["order", "Backend"])["Backend"].unique(),
        ordered=True,
    )
    table = labelled.pivot(
        index=["Backend", "Machine"], columns="molecules", values="cell"
    )
    table.columns = [f"{int(size):,}" for size in table.columns]
    return compact_multiindex_rows(table, index_name=index_name)


def qtable_mi(
    df: pd.DataFrame,
    *,
    caption: str | None = None,
    table_id: str | None = None,
    colwidths: list[int] | None = None,
    aligns: str | None = None,
) -> str:
    attrs = [".list-table"]

    if table_id:
        attrs.insert(0, f"#{table_id}")

    if colwidths:
        attrs.append(f'tbl-colwidths="[{",".join(map(str, colwidths))}]"')

    if aligns:
        attrs.append(f'aligns="{aligns}"')

    out = [f"::: {{{' '.join(attrs)}}}"]

    if caption:
        out += [caption, ""]

    def is_missing(value: object) -> bool:
        if value is None:
            return True
        try:
            result = pd.isna(value)
            return bool(result) if isinstance(result, (bool, type(pd.NA))) else False
        except TypeError, ValueError:
            return False

    def text_of(value: object) -> str:
        return "" if is_missing(value) else str(value)

    def span_cell(
        value: object,
        *,
        rowspan: int = 1,
        colspan: int = 1,
    ) -> str:
        attrs = []
        if rowspan > 1:
            attrs.append(f"rowspan={rowspan}")
        if colspan > 1:
            attrs.append(f"colspan={colspan}")

        text = text_of(value)

        if attrs:
            return f"[]{{{' '.join(attrs)}}} {text}"
        else:
            return text

    def emit_cell(prefix: str, value: object) -> None:
        text = text_of(value)
        lines = text.splitlines() or [""]
        out.append(prefix + lines[0])
        for line in lines[1:]:
            out.append("    " + line)

    def emit_row(cells: list[object]) -> None:
        if not cells:
            return

        emit_cell("- - ", cells[0])
        for cell in cells[1:]:
            emit_cell("  - ", cell)
        out.append("")

    # ---------- normalize axes ----------

    row_index = df.index
    col_index = df.columns

    col_nlevels = col_index.nlevels

    row_tuples = list(row_index)
    if not isinstance(row_index, pd.MultiIndex):
        row_tuples = [(x,) for x in row_tuples]

    col_tuples = list(col_index)
    if not isinstance(col_index, pd.MultiIndex):
        col_tuples = [(x,) for x in col_tuples]

    row_names = [name if name is not None else "" for name in row_index.names]

    # ---------- header ----------

    if isinstance(col_index, pd.MultiIndex):
        for level in range(col_nlevels):
            cells = []

            # Row-index header cells. They span all column-header rows.
            if level == 0:
                for name in row_names:
                    cells.append(span_cell(name, rowspan=col_nlevels))

            c = 0
            while c < len(col_tuples):
                prefix = col_tuples[c][: level + 1]
                label = col_tuples[c][level]

                j = c + 1
                while j < len(col_tuples) and col_tuples[j][: level + 1] == prefix:
                    j += 1

                cells.append(span_cell(label, colspan=j - c))
                c = j

            emit_row(cells)

    else:
        cells = row_names + [text_of(c) for c in col_index]
        emit_row(cells)

    # ---------- body ----------

    values = df.to_numpy(dtype=object)

    for i, idx_tuple in enumerate(row_tuples):
        cells = []

        for level, label in enumerate(idx_tuple):
            prefix = idx_tuple[: level + 1]

            previous_same = i > 0 and row_tuples[i - 1][: level + 1] == prefix

            if previous_same:
                continue

            j = i + 1
            while j < len(row_tuples) and row_tuples[j][: level + 1] == prefix:
                j += 1

            cells.append(span_cell(label, rowspan=j - i))

        cells.extend(values[i])
        emit_row(cells)

    out.append(":::")
    return "\n".join(out)


def compact_multiindex_rows(
    df: pd.DataFrame,
    *,
    index_name: str = "Backend and Machine",
    indent: str = "\u00a0\u00a0\u00a0\u00a0",
) -> pd.DataFrame:
    """
    Convert a MultiIndex-row dataframe into a normal dataframe where the
    MultiIndex hierarchy is rendered as an outline in a single first column.

    Parent index rows get blank data cells.
    Leaf index rows get the actual dataframe values.
    """
    if not isinstance(df.index, pd.MultiIndex):
        out = df.reset_index()
        out = out.rename(columns={out.columns[0]: index_name})
        return out

    df = df.sort_index()

    rows = []
    nlevels = df.index.nlevels
    data_columns = list(df.columns)

    previous_key = None

    for key, values in df.iterrows():
        if not isinstance(key, tuple):
            key = (key,)

        # Emit parent levels when they change.
        for level in range(nlevels - 1):
            prefix = key[: level + 1]

            if previous_key is None or previous_key[: level + 1] != prefix:
                rows.append(
                    {
                        index_name: f"{indent * level}{key[level]}",
                        **dict.fromkeys(data_columns, ""),
                    }
                )

        # Emit leaf row with actual data.
        leaf_label = key[-1]
        rows.append(
            {
                index_name: f"{indent * (nlevels - 1)}{leaf_label}",
                **dict(zip(data_columns, values)),
            }
        )

        previous_key = key

    return pd.DataFrame(rows)
