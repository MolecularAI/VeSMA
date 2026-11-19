from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

PAPER = Path(__file__).parents[2] / "docs" / "paper" / "paper"
sys.path.insert(0, str(PAPER))

from paperlib.figures import (  # noqa: E402
    SERIES,
    SMALL_SIZE,
    Margins,
    compact_count,
    decade_limits,
    growth,
    log_axis,
    profile_panel,
    qfigure,
    series_colors,
    trend_panel,
    typst_string,
)

SIZES = pd.Series([10, 2_000, 300_000], index=[0, 1, 2])
PROFILES = pd.DataFrame(
    [[1e3, 1e5, np.nan], [1e4, np.nan, 1e6]],
    index=[0, 1],
    columns=["first", "second", "third"],
)


def test_growth_is_the_ratio_of_consecutive_entries() -> None:
    assert growth(np.array([2, 8, 24])).tolist() == [4.0, 3.0]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([2_879, 4_582_701_278], (1e3, 1e10)),
        ([1e3, 9e9], (1e3, 1e10)),
        ([5, 50], (1.0, 100.0)),
    ],
)
def test_decade_limits_round_outward_to_powers_of_ten(
    values: list[float], expected: tuple[float, float]
) -> None:
    assert decade_limits(np.array(values)) == expected


@pytest.mark.parametrize(
    ("value", "label"),
    [
        (0, "0"),
        (999, "999"),
        (2_879, "2.9k"),
        (327_851, "328k"),
        (1_381_750, "1.4M"),
        (20_502_803, "21M"),
        (4_582_701_278, "4.6B"),
    ],
)
def test_compact_count(value: int, label: str) -> None:
    assert compact_count(value) == label


def test_series_colors_are_stable_and_repeat_past_the_palette() -> None:
    assert series_colors(2) == list(SERIES[:2])
    assert series_colors(len(SERIES) + 1) == [*SERIES, SERIES[0]]


@pytest.mark.parametrize(
    ("limits", "labelled_exponents"),
    [
        ((1e3, 1e10), [3, 6, 9]),
        ((1e0, 1e2), [0, 1, 2]),
        ((1e0, 1e12), [0, 5, 10]),
        ((1e2, 1e3), [2, 3]),
    ],
)
def test_log_axis_ticks_like_a_log_scale_but_labels_at_most_three_decades(
    limits: tuple[float, float], labelled_exponents: list[int]
) -> None:
    low, high = (round(np.log10(limit)) for limit in limits)
    fig, ax = plt.subplots()
    try:
        log_axis(ax, limits)
        # a tick at every decade, and log-spaced minor ticks at 2 to 9 times each decade
        assert np.log10(ax.yaxis.get_majorticklocs()).round().astype(
            int
        ).tolist() == list(range(low, high + 1))
        expected_minor = [m * 10.0**e for e in range(low, high) for m in range(2, 10)]
        assert np.allclose(np.sort(ax.yaxis.get_minorticklocs()), expected_minor)
        # ...but only a few of the decade ticks carry a label
        shown = [text.get_text() for text in ax.get_yticklabels() if text.get_text()]
        assert len(shown) == len(labelled_exponents) <= 3
        assert all(
            f"10^{{{e}}}" in label for e, label in zip(labelled_exponents, shown)
        )
        assert ax.get_ylim() == limits
    finally:
        plt.close(fig)


def test_trend_panel_values_every_point_and_every_step() -> None:
    fig = trend_panel(SIZES, limits=(1.0, 1e6), xlabel="step", ylabel="count")
    try:
        (ax,) = fig.axes
        assert sorted(text.get_text() for text in ax.texts) == sorted(
            ["10", "2.0k", "300k", "×200", "×150"]
        )
        assert [label.get_text() for label in ax.get_xticklabels()] == ["0", "1", "2"]
        assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == (
            "step",
            "count",
            "",
        )
    finally:
        plt.close(fig)


def test_profile_panel_leaves_out_missing_values_and_draws_the_guide() -> None:
    fig = profile_panel(
        PROFILES,
        limits=(1e3, 1e6),
        ylabel="count",
        label=lambda name: f"Layer {name}",
        guide=(1e4, "reference"),
    )
    try:
        (ax,) = fig.axes
        assert sorted(text.get_text() for text in ax.texts) == sorted(
            ["1.0k", "100k", "10k", "1.0M", "reference"]
        )
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "Layer 0",
            "Layer 1",
        ]
        assert [label.get_text() for label in ax.get_xticklabels()] == list(
            PROFILES.columns
        )
    finally:
        plt.close(fig)


def test_profile_panel_omits_the_values_a_mask_switches_off() -> None:
    labelled = pd.DataFrame(True, index=PROFILES.index, columns=PROFILES.columns)
    labelled.iloc[1, 0] = False
    fig = profile_panel(PROFILES, limits=(1e3, 1e6), ylabel="count", labelled=labelled)
    try:
        (ax,) = fig.axes
        assert sorted(text.get_text() for text in ax.texts) == sorted(
            ["1.0k", "100k", "1.0M"]
        )
        first = next(text for text in ax.texts if text.get_text() == "1.0k")
        assert (first.get_horizontalalignment(), first.get_verticalalignment()) == (
            "left",
            "center",
        )
    finally:
        plt.close(fig)


def test_profile_panel_rotates_stage_names_on_request() -> None:
    fig = profile_panel(PROFILES, limits=(1e3, 1e6), ylabel="count", xtick_rotation=40)
    try:
        assert {label.get_rotation() for label in fig.axes[0].get_xticklabels()} == {40}
    finally:
        plt.close(fig)


def test_margins_are_inches_so_panels_of_one_size_share_a_plot_area() -> None:
    fig = plt.figure(figsize=(4, 3))
    try:
        assert np.allclose(
            Margins(left=1, bottom=0.5, right=0.2, top=0.3).rect(fig),
            (0.25, 0.5 / 3, 0.7, 1 - 0.8 / 3),
        )
    finally:
        plt.close(fig)
    figs = [
        trend_panel(SIZES, limits=(1.0, 1e6), xlabel="x", ylabel="y"),
        profile_panel(PROFILES, limits=(1.0, 1e6), ylabel="y"),
    ]
    try:
        first, second = (fig.axes[0].get_position().bounds for fig in figs)
        assert np.allclose(first, second)
    finally:
        for fig in figs:
            plt.close(fig)


def test_panels_use_the_shared_text_size_and_one_y_range() -> None:
    limits = (1.0, 1e6)
    figs = [
        trend_panel(SIZES, limits=limits, xlabel="x", ylabel="y"),
        profile_panel(PROFILES, limits=(1.0, 1e6), ylabel="y"),
    ]
    try:
        for fig in figs:
            (ax,) = fig.axes
            assert ax.get_ylim() == limits
            assert {text.get_fontsize() for text in ax.texts} == {SMALL_SIZE}
    finally:
        for fig in figs:
            plt.close(fig)


def test_panels_leave_global_matplotlib_style_untouched() -> None:
    before = dict(plt.rcParams)
    for fig in (
        trend_panel(SIZES, limits=(1.0, 1e6), xlabel="x", ylabel="y"),
        profile_panel(PROFILES, limits=(1.0, 1e6), ylabel="y"),
    ):
        plt.close(fig)
    assert dict(plt.rcParams) == before


def test_panels_take_their_size_from_the_surrounding_frame() -> None:
    with plt.rc_context({"figure.figsize": (2.85, 3.0)}):
        fig = trend_panel(SIZES, limits=(1.0, 1e6), xlabel="x", ylabel="y")
    try:
        assert np.allclose(fig.get_size_inches(), (2.85, 3.0))
    finally:
        plt.close(fig)


def test_panels_carry_their_own_size_and_corner_tag() -> None:
    fig = profile_panel(
        PROFILES, limits=(1e3, 1e6), ylabel="count", size=(3.35, 3.0), tag="(b)"
    )
    try:
        assert np.allclose(fig.get_size_inches(), (3.35, 3.0))
        assert [text.get_text() for text in fig.texts] == ["(b)"]
        assert fig.texts[0].get_fontweight() == "bold"
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("text", "literal"),
    [
        ("plain", '"plain"'),
        ('say "hi"', r'"say \"hi\""'),
        ("a\\b", r'"a\\b"'),
        ("two\nlines", r'"two\nlines"'),
    ],
)
def test_typst_string_escapes_quotes_backslashes_and_newlines(
    text: str, literal: str
) -> None:
    assert typst_string(text) == literal


def panels_for_markup() -> list:
    return [
        trend_panel(
            SIZES, limits=(1.0, 1e6), xlabel="x", ylabel="y", size=(2.3, 3.0), tag="(a)"
        ),
        profile_panel(
            PROFILES, limits=(1.0, 1e6), ylabel="y", size=(3.35, 3.0), tag="(b)"
        ),
    ]


def test_qfigure_is_one_numbered_figure_with_inline_boxes_and_one_caption() -> None:
    markup = qfigure(
        panels_for_markup(),
        alts=["panel a", "panel b"],
        caption="Title. (a) First. (b) Second.",
        figure_id="fig-x",
    )
    lines = markup.splitlines()
    assert lines[0] == "::: {#fig-x}" and lines[-1] == ":::"
    assert lines[1] == "```{=typst}" and lines[3] == "```"
    assert (
        lines[2].count("#box(image(bytes(") == 2
    )  # one line: the renderer wraps the boxes
    assert lines[2].count('alt: "panel a"') == lines[2].count('alt: "panel b"') == 1
    assert lines[-2] == "Title. (a) First. (b) Second."


def test_qfigure_closes_the_panels_and_renders_identically_each_time() -> None:
    first = qfigure(
        panels_for_markup(), alts=["a", "b"], caption="c", figure_id="fig-x"
    )
    assert plt.get_fignums() == []
    second = qfigure(
        panels_for_markup(), alts=["a", "b"], caption="c", figure_id="fig-x"
    )
    assert first == second
