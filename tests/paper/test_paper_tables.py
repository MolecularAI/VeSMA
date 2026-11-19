from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PAPER = Path(__file__).parents[2] / "docs" / "paper" / "paper"
sys.path.insert(0, str(PAPER))

from paperlib.paper_tables import (  # noqa: E402
    format_microseconds,
    format_significant,
    host_label,
    load_application,
    scaling_cell,
    scaling_table,
    select_records,
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
