from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PAPER = Path(__file__).parents[2] / "docs" / "paper" / "paper"
sys.path.insert(0, str(PAPER))

from paperlib.paper_tables import (  # noqa: E402
    Table,
    format_microseconds,
    format_significant,
    host_label,
    list_table,
    load_application,
    pipe_table,
    qtable,
    scaling_cell,
    scaling_table,
    select_records,
    table_from_dataframe,
    with_rates,
)

KEYS = ["backend", "machine.host_type"]


def records() -> pd.DataFrame:
    """Two backends on two machines at two workload sizes; the GPU lacks the large size."""
    rows = [
        ("rdkit-serial", "laptop", 1, 10, 40, 0.5),
        ("rdkit-serial", "laptop", 100, 1_000, 4_000, 50.0),
        ("cuda-host", "server", 1, 10, 30, 0.02),
        ("cpu-serial", "laptop", 1, 10, 30, 2.0),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "backend",
            "machine.host_type",
            "molecules",
            "positive_pairs",
            "precursors",
            "stats.mean",
        ],
    ).assign(source_path=lambda frame: frame.index.map("record-{}.json".format))


def test_with_rates_divides_by_wall_time_and_by_precursors() -> None:
    rated = with_rates(records())
    assert rated["pairs_per_second"].tolist() == [20.0, 20.0, 500.0, 5.0]
    assert rated["seconds_per_precursor"].tolist() == [
        0.5 / 40,
        50.0 / 4_000,
        0.02 / 30,
        2.0 / 30,
    ]


def test_with_rates_names_the_missing_column() -> None:
    with pytest.raises(ValueError, match="precursors"):
        with_rates(records().drop(columns="precursors"))


def test_select_records_keeps_the_requested_order() -> None:
    chosen = select_records(
        records().query("molecules == 1"),
        [("cuda-host", "server"), ("rdkit-serial", "laptop")],
        KEYS,
    )
    assert chosen["backend"].tolist() == ["cuda-host", "rdkit-serial"]
    assert chosen["stats.mean"].tolist() == [0.02, 0.5]


def test_select_records_rejects_a_missing_key() -> None:
    with pytest.raises(ValueError, match="metal-raw"):
        select_records(records(), [("metal-raw", "laptop")], KEYS)


def test_select_records_rejects_ambiguous_records() -> None:
    with pytest.raises(ValueError, match="rdkit-serial.*2"):
        select_records(records(), [("rdkit-serial", "laptop")], KEYS)


@pytest.mark.parametrize(
    ("value", "digits", "text"),
    [
        (6781.3, 3, "6,780"),
        (2939.0, 3, "2,940"),
        (50447.0, 3, "50,400"),
        (0.04567, 3, "0.0457"),
        (12.34, 2, "12"),
        (0.0, 3, "0.0"),
    ],
)
def test_format_significant(value: float, digits: int, text: str) -> None:
    assert format_significant(value, digits) == text


def test_format_microseconds_is_a_bare_number() -> None:
    assert format_microseconds(6.97e-6) == "7.0"
    assert format_microseconds(79.38e-6, decimals=2) == "79.38"


def test_scaling_cell_stacks_time_over_throughput() -> None:
    cell = scaling_cell(with_rates(records()).iloc[2])
    assert cell == "20.0 ms\\\n(0.5k p.pairs/s)"


def test_scaling_table_lays_out_backends_by_machine_and_size() -> None:
    table = scaling_table(with_rates(records()))
    assert list(table.columns) == ["Backend and machine", "1", "100"]
    labels = table["Backend and machine"].str.strip("\u00a0").tolist()
    assert labels == [
        "RDKit",
        "laptop",
        "Vesma CPU serial",
        "laptop",
        "CUDA host",
        "server",
    ]
    cells = table.set_index("Backend and machine")
    assert pd.isna(cells.loc["\u00a0" * 4 + "server", "100"])
    assert cells.iloc[1]["100"].startswith("50.0 s")


def test_scaling_table_rejects_duplicate_records() -> None:
    duplicated = pd.concat([records(), records().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate benchmark records"):
        scaling_table(with_rates(duplicated))


def test_host_label_joiner() -> None:
    record = pd.Series(
        {
            "machine.cpu.brand_raw": "AMD EPYC 9534 64-Core Processor",
            "machine.cuda.device_name": "NVIDIA GeForce RTX 5090",
        }
    )
    assert host_label(record) == "AMD EPYC 9534 plus NVIDIA GeForce RTX 5090"
    assert host_label(record, "+") == "AMD EPYC 9534 + NVIDIA GeForce RTX 5090"


def test_published_scaling_records_agree_on_precursor_processing() -> None:
    scaling = load_application(PAPER.parent / "benchmarks" / "reaction-application")
    assert not scaling.empty


def test_qtable_renders_list_table_with_widths_and_default_left_alignment() -> None:
    df = pd.DataFrame(
        {
            "Backend": ["RDKit", "VeSMA"],
            "Rate": ["4,640\\\n(1x)", None],
        }
    )
    rendered = qtable(
        df,
        caption="Benchmark summary.",
        table_id="tbl-bench",
        colwidths=[40, 60],
    )
    assert rendered == "\n".join(
        [
            '::: {#tbl-bench .list-table widths="40,60" aligns="l,l"}',
            "Benchmark summary.",
            "",
            "- - Backend",
            "  - Rate",
            "",
            "- - RDKit",
            "  - 4,640\\",
            "    (1x)",
            "",
            "- - VeSMA",
            "  -",
            "",
            ":::",
        ]
    )


def test_qtable_accepts_compact_and_sequence_alignments() -> None:
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    assert 'aligns="l,c,r"' in qtable(df, aligns="lcr")
    assert 'aligns="l,c,r"' in qtable(df, aligns="l, c, r")
    assert 'aligns="l,c,r"' in qtable(df, aligns=("l", "c", "r"))


def test_qtable_include_index_renders_rowspans_without_leading_space_and_preserves_widths() -> (
    None
):
    df = pd.DataFrame(
        [
            {"Backend": "VeSMA", "Comparator": "Complete", "Recovered": 9970},
            {"Backend": "VeSMA", "Comparator": "Stock", "Recovered": 9972},
            {"Backend": "RDKit", "Comparator": "Complete", "Recovered": 9954},
        ]
    ).set_index(["Backend", "Comparator"])

    rendered = qtable(
        df,
        caption="Replay summary.",
        table_id="tbl-replay",
        colwidths=[40, 35, 25],
        aligns="l,l,r",
        include_index=True,
    )
    assert 'widths="40,35,25" aligns="l,l,r"' in rendered
    assert "- - []{rowspan=2}VeSMA" in rendered
    assert "- - []{rowspan=2} VeSMA" not in rendered
    assert "- - RDKit" in rendered


def test_qtable_renders_multiindex_columns_with_header_rows() -> None:
    columns = pd.MultiIndex.from_tuples(
        [("Timing", "Mean"), ("Timing", "Min"), ("Throughput", "Pairs/s")]
    )
    index = pd.Index(["VeSMA", "RDKit"], name="Backend")
    df = pd.DataFrame(
        [[9.7, 9.5, 79_400], [56.8, 56.1, 13_500]], index=index, columns=columns
    )

    rendered = qtable(df, include_index=True, aligns="l,r,r,r")
    assert 'header-rows=2 aligns="l,r,r,r"' in rendered
    assert "- - []{rowspan=2}Backend" in rendered
    assert "  - []{colspan=2}Timing" in rendered
    assert "  - Throughput" in rendered
    assert "- - Mean" in rendered


def test_pipe_table_renders_portable_markdown_via_injected_renderer() -> None:
    df = pd.DataFrame(
        {
            "Pattern": ["C|N", "c1ccccc1"],
            "Count": [12, 340],
        }
    )
    rendered = qtable(
        df,
        caption="Pattern counts.",
        table_id="tbl-patterns",
        colwidths=[60, 40],
        aligns="l,r",
        render=pipe_table,
    )
    assert rendered == "\n".join(
        [
            "| Pattern | Count |",
            "|:---|---:|",
            r"| C\|N | 12 |",
            "| c1ccccc1 | 340 |",
            "",
            ': Pattern counts. {#tbl-patterns tbl-colwidths="[60,40]"}',
        ]
    )


def test_qtable_accepts_any_custom_table_renderer() -> None:
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    def summary_renderer(table: Table) -> str:
        return f"{table.table_id}:{table.column_count}x{len(table.rows)}"

    assert (
        qtable(df, table_id="tbl-custom", render=summary_renderer) == "tbl-custom:2x2"
    )
    built = table_from_dataframe(df, table_id="tbl-custom")
    assert list_table(built) == qtable(df, table_id="tbl-custom")


def test_pipe_table_rejects_features_pipe_tables_cannot_represent() -> None:
    multiline = pd.DataFrame({"A": ["line1\\\nline2"]})
    with pytest.raises(ValueError, match="multi-line"):
        qtable(multiline, render=pipe_table)

    multi_col = pd.DataFrame(
        [[1, 2]],
        columns=pd.MultiIndex.from_tuples([("G", "a"), ("G", "b")]),
    )
    with pytest.raises(ValueError, match="single header row"):
        qtable(multi_col, render=pipe_table)

    multi_row = pd.DataFrame(
        {"Val": [1, 2]},
        index=pd.MultiIndex.from_tuples(
            [("G", "a"), ("G", "b")], names=["Group", "Item"]
        ),
    )
    with pytest.raises(ValueError, match="rowspan or colspan"):
        qtable(multi_row, include_index=True, render=pipe_table)


def test_qtable_validates_arguments_fail_fast() -> None:
    df = pd.DataFrame({"A": [1], "B": [2]})

    with pytest.raises(ValueError, match="2 column widths"):
        qtable(df, colwidths=[100])
    with pytest.raises(ValueError, match="finite positive"):
        qtable(df, colwidths=[50, 0])
    with pytest.raises(ValueError, match="finite positive"):
        qtable(df, colwidths=[True, 50])
    with pytest.raises(ValueError, match="2 column alignments"):
        qtable(df, aligns="l")
    with pytest.raises(ValueError, match="Unknown alignment"):
        qtable(df, aligns="l,x")
    with pytest.raises(ValueError, match="single paragraph"):
        qtable(df, caption="Para one.\n\nPara two.")
    with pytest.raises(ValueError, match="bare identifier"):
        qtable(df, table_id="#tbl-bad")
    with pytest.raises(ValueError, match="zero columns"):
        qtable(pd.DataFrame())
