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
    compact_count,
    decade_limits,
    growth,
    log_axis,
    multipanel,
    plot_profiles,
    plot_trend,
    series_colors,
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
        assert np.log10(ax.yaxis.get_majorticklocs()).round().astype(
            int
        ).tolist() == list(range(low, high + 1))
        expected_minor = [m * 10.0**e for e in range(low, high) for m in range(2, 10)]
        assert np.allclose(np.sort(ax.yaxis.get_minorticklocs()), expected_minor)
        shown = [text.get_text() for text in ax.get_yticklabels() if text.get_text()]
        assert len(shown) == len(labelled_exponents) <= 3
        assert all(
            f"10^{{{e}}}" in label for e, label in zip(labelled_exponents, shown)
        )
        assert ax.get_ylim() == limits
    finally:
        plt.close(fig)


def test_plot_trend_values_every_point_and_every_step() -> None:
    fig = multipanel(
        [
            lambda ax: plot_trend(
                ax, SIZES, limits=(1.0, 1e6), xlabel="step", ylabel="count"
            )
        ]
    )
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


def test_plot_profiles_leaves_out_missing_values_and_draws_the_guide() -> None:
    fig = multipanel(
        [
            lambda ax: plot_profiles(
                ax,
                PROFILES,
                limits=(1e3, 1e6),
                ylabel="count",
                label=lambda name: f"Layer {name}",
                guide=(1e4, "reference"),
            )
        ]
    )
    try:
        (ax,) = fig.axes
        assert sorted(text.get_text() for text in ax.texts) == sorted(
            ["1.0k", "100k", "10k", "1.0M", "reference"]
        )
        legend = ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == [
            "Layer 0",
            "Layer 1",
        ]
        assert [label.get_text() for label in ax.get_xticklabels()] == list(
            PROFILES.columns
        )
    finally:
        plt.close(fig)


def test_plot_profiles_omits_the_values_a_mask_switches_off() -> None:
    labelled = pd.DataFrame(True, index=PROFILES.index, columns=PROFILES.columns)
    labelled.iloc[1, 0] = False
    fig = multipanel(
        [
            lambda ax: plot_profiles(
                ax, PROFILES, limits=(1e3, 1e6), ylabel="count", labelled=labelled
            )
        ]
    )
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


def test_plot_profiles_rotates_stage_names_on_request() -> None:
    fig = multipanel(
        [
            lambda ax: plot_profiles(
                ax, PROFILES, limits=(1e3, 1e6), ylabel="count", xtick_rotation=40
            )
        ]
    )
    try:
        assert {label.get_rotation() for label in fig.axes[0].get_xticklabels()} == {40}
    finally:
        plt.close(fig)


def test_multipanel_aligns_vertical_axes_and_applies_tags_and_style() -> None:
    limits = (1.0, 1e6)
    fig = multipanel(
        [
            lambda ax: plot_trend(ax, SIZES, limits=limits, xlabel="x", ylabel="y1"),
            lambda ax: plot_profiles(
                ax, PROFILES, limits=limits, ylabel="y2", xtick_rotation=40
            ),
        ],
        size=(6.0, 3.2),
        width_ratios=(1.5, 2.5),
        tags=("(a)", "(b)"),
    )
    try:
        fig.canvas.draw()
        ax_a, ax_b = fig.axes
        assert np.allclose(fig.get_size_inches(), (6.0, 3.2))
        assert ax_a.get_ylim() == ax_b.get_ylim() == limits
        # Constrained layout aligns the bottom and height of both axes identically.
        assert np.isclose(ax_a.get_position().y0, ax_b.get_position().y0)
        assert np.isclose(ax_a.get_position().height, ax_b.get_position().height)
        assert ax_b.get_position().width > ax_a.get_position().width
        assert ax_a.get_title(loc="left") == "(a)"
        assert ax_b.get_title(loc="left") == "(b)"
        for ax in (ax_a, ax_b):
            assert {text.get_fontsize() for text in ax.texts} == {SMALL_SIZE}
    finally:
        plt.close(fig)


def test_multipanel_leaves_global_matplotlib_style_untouched() -> None:
    before = dict(plt.rcParams)
    fig = multipanel(
        [
            lambda ax: plot_trend(ax, SIZES, limits=(1.0, 1e6), xlabel="x", ylabel="y"),
            lambda ax: plot_profiles(ax, PROFILES, limits=(1.0, 1e6), ylabel="y"),
        ]
    )
    plt.close(fig)
    assert dict(plt.rcParams) == before


def test_multipanel_validates_arguments_fail_fast() -> None:
    with pytest.raises(ValueError, match="at least one"):
        multipanel([])
    with pytest.raises(ValueError, match="1 width ratios"):
        multipanel([lambda ax: None], width_ratios=(1, 2))
    with pytest.raises(ValueError, match="1 panel tags"):
        multipanel([lambda ax: None], tags=("(a)", "(b)"))
