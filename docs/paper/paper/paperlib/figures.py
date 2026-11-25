"""Plotting helpers for benchmark papers: counts spanning orders of magnitude on a log axis.

Nothing here knows about one particular figure or output backend. Panel drawers take a
Matplotlib ``Axes`` and draw onto it; ``multipanel`` composes any sequence of drawers
into one aligned Matplotlib ``Figure`` that Quarto renders natively to SVG (Typst),
PDF (LaTeX), or PNG (DOCX).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from itertools import cycle, islice
from math import ceil, floor, log10
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter
from matplotlib.typing import RcKeyType

__all__ = [
    "STYLE",
    "SERIES",
    "LABEL_SIZE",
    "SMALL_SIZE",
    "PanelDrawer",
    "series_colors",
    "growth",
    "decade_limits",
    "compact_count",
    "log_axis",
    "annotate_all",
    "plot_trend",
    "plot_profiles",
    "multipanel",
]

type PanelDrawer = Callable[[Axes], None]

# Where a point's value goes, in points from the point. The first point sits at the left edge,
# where the line leaves steeply, so its label goes beside it; later points are labelled above and
# to the right, clear of the line arriving from the left.
FIRST_LABEL: dict[str, Any] = {"xytext": (8, 0), "ha": "left", "va": "center"}
NEXT_LABEL: dict[str, Any] = {"xytext": (3, 7), "ha": "left"}

# Points, set against 11 pt body text: axis labels and panel tags just below it; ticks, value labels
# and legend one step down.
LABEL_SIZE, SMALL_SIZE = 10, 9

GREY, FOCAL = "#888888", "#1b4965"
SERIES = ("#5c8a3a", "#c46210", "#0072b2", "#cc79a7", "#56b4e9", "#b0a02b")

STYLE: dict[RcKeyType, Any] = {
    "font.family": "sans-serif",
    "font.size": SMALL_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": SMALL_SIZE,
    "ytick.labelsize": SMALL_SIZE,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "ytick.minor.size": 2,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "ytick.minor.width": 0.5,
    "legend.fontsize": SMALL_SIZE,
    "legend.frameon": False,
}


def series_colors(count: int) -> list[str]:
    """One colour per series, in a fixed order, repeating when there are more series than colours."""
    return list(islice(cycle(SERIES), count))


def growth(values: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    """Multiplier between consecutive entries."""
    return values[1:] / values[:-1]


def decade_limits(values: npt.NDArray[np.number]) -> tuple[float, float]:
    """Smallest range of whole powers of ten containing every value."""
    return 10.0 ** floor(log10(values.min())), 10.0 ** ceil(log10(values.max()))


def compact_count(value: float) -> str:
    """Short label for a count: 2879, 328k, 20M, 4.6B."""
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if value >= scale:
            scaled = value / scale
            return f"{scaled:.1f}{suffix}" if scaled < 10 else f"{scaled:.0f}{suffix}"
    return f"{value:.0f}"


def log_axis(ax: Axes, limits: tuple[float, float], max_labels: int = 3) -> None:
    """Logarithmic y axis with the usual log-spaced ticks and no more than ``max_labels`` labels.

    ``limits`` are whole powers of ten (see ``decade_limits``). There is a tick at every decade
    and minor ticks at 2 to 9 times each decade; only ``max_labels`` evenly spaced decades are
    labelled so labels stay readable at body-text size.
    """
    low, high = (round(log10(limit)) for limit in limits)
    decades = range(low, high + 1)
    labelled = decades[:: ceil(len(decades) / max_labels)]
    ax.set(yscale="log", ylim=limits)
    ax.set_yticks(
        [10.0**e for e in decades],
        [rf"$\mathdefault{{10^{{{e}}}}}$" if e in labelled else "" for e in decades],
    )
    ax.set_yticks(
        [m * 10.0**e for e in range(low, high) for m in range(2, 10)], minor=True
    )
    ax.yaxis.set_minor_formatter(NullFormatter())


def annotate_all(
    ax: Axes,
    xs: Iterable[float],
    ys: Iterable[float],
    texts: Iterable[str],
    **style: Any,
) -> None:
    """Place one text per data point, offset from it in points."""
    for x, y, text in zip(xs, ys, texts):
        ax.annotate(text, (x, y), textcoords="offset points", **style)


def plot_trend(
    ax: Axes,
    sizes: pd.Series,
    *,
    limits: tuple[float, float],
    xlabel: str,
    ylabel: str,
) -> None:
    """Plot one count per step on a log axis, valuing each point and marking each step's multiplier.

    ``sizes`` is indexed by the step's name; any number of steps works.
    """
    values = sizes.to_numpy()
    steps = np.arange(len(values))
    log_midpoints = np.sqrt(values[:-1] * values[1:])

    ax.plot(steps, values, "-o", color=FOCAL, mec="white", ms=8, lw=2, zorder=3)
    log_axis(ax, limits)
    ax.set(
        xticks=steps,
        xticklabels=[str(name) for name in sizes.index],
        xlabel=xlabel,
        ylabel=ylabel,
    )
    ax.set_xlim(steps[0] - 0.4, steps[-1] + 0.15)

    texts = list(map(compact_count, values))
    annotate_all(
        ax,
        steps[:1],
        values[:1],
        texts[:1],
        color=FOCAL,
        fontweight="bold",
        **FIRST_LABEL,
    )
    annotate_all(
        ax,
        steps[1:],
        values[1:],
        texts[1:],
        xytext=(-8, 6),
        ha="right",
        color=FOCAL,
        fontweight="bold",
    )
    annotate_all(
        ax,
        steps[:-1] + 0.5,
        log_midpoints,
        (f"×{g:.0f}" for g in growth(values)),
        xytext=(-6, 6),
        ha="right",
        color=GREY,
        fontweight="bold",
    )


def plot_profiles(
    ax: Axes,
    profiles: pd.DataFrame,
    *,
    limits: tuple[float, float],
    ylabel: str,
    label: Callable[[Hashable], str] = str,
    guide: tuple[float, str] | None = None,
    labelled: pd.DataFrame | None = None,
    xtick_rotation: float = 0.0,
) -> None:
    """Plot one line per row across the ordered columns on a log axis, with valued points.

    Rows are series (labelled in the legend by ``label(row index)``), columns are the stages
    they pass through, and missing values (NaN) are omitted. Every point is valued unless the
    boolean frame ``labelled`` (same shape as ``profiles``) says otherwise. ``guide`` draws a
    dashed horizontal reference line at a level, with its caption.
    """
    stages = np.arange(profiles.shape[1])

    if guide is not None:
        level, caption = guide
        ax.hlines(
            level,
            stages[0],
            stages[-1],
            ls=(0, (4, 3)),
            color=GREY,
            lw=1.0,
            zorder=1,
        )
        ax.annotate(
            caption,
            (stages[-1], level),
            textcoords="offset points",
            xytext=(0, -6),
            ha="right",
            va="top",
            color=GREY,
            style="italic",
        )

    for (name, row), color in zip(profiles.iterrows(), series_colors(len(profiles))):
        counts = row.to_numpy(dtype=float)
        ax.plot(
            stages,
            counts,
            "-o",
            color=color,
            mec="white",
            ms=7,
            lw=1.6,
            zorder=2,
            label=label(name),
        )
        present = ~np.isnan(counts)
        shown = (
            np.ones(len(counts), bool)
            if labelled is None
            else labelled.loc[name].to_numpy()
        )
        for index, (x, y, show) in enumerate(
            zip(stages[present], counts[present], shown[present])
        ):
            if show:
                ax.annotate(
                    compact_count(y),
                    (x, y),
                    textcoords="offset points",
                    color=color,
                    **(FIRST_LABEL if index == 0 else NEXT_LABEL),
                )

    log_axis(ax, limits)
    ax.set(
        xticks=stages,
        xlim=(stages[0] - 0.5, stages[-1] + 0.85),
        ylabel=ylabel,
    )
    ax.set_xticklabels(
        profiles.columns,
        rotation=xtick_rotation,
        ha="right" if xtick_rotation else "center",
        rotation_mode="anchor",
    )
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1, 1.05),
        ncols=min(len(profiles), 4),
    )


def multipanel(
    drawers: Sequence[PanelDrawer],
    *,
    size: tuple[float, float] | None = None,
    width_ratios: Sequence[float] | None = None,
    tags: Sequence[str] | None = None,
    sharey: bool = False,
) -> Figure:
    """Compose panel drawers side by side into one aligned Matplotlib ``Figure``.

    Uses constrained layout so all panels share exact vertical alignment without clipping
    rotated axis labels, and leaves global Matplotlib ``rcParams`` untouched.
    """
    if not drawers:
        raise ValueError("multipanel requires at least one panel drawer")
    if width_ratios is not None and len(width_ratios) != len(drawers):
        raise ValueError(
            f"Expected {len(drawers)} width ratios, got {len(width_ratios)}"
        )
    if tags is not None and len(tags) != len(drawers):
        raise ValueError(f"Expected {len(drawers)} panel tags, got {len(tags)}")

    gridspec_kw = {"width_ratios": list(width_ratios)} if width_ratios else None
    with plt.rc_context(STYLE):
        fig, axes_raw = plt.subplots(
            1,
            len(drawers),
            figsize=size,
            sharey=sharey,
            gridspec_kw=gridspec_kw,
            layout="constrained",
        )
        axes = [axes_raw] if isinstance(axes_raw, Axes) else list(axes_raw)
        for ax, draw in zip(axes, drawers):
            draw(ax)
        if tags is not None:
            for ax, tag in zip(axes, tags):
                ax.set_title(
                    tag,
                    loc="left",
                    fontweight="bold",
                    fontsize=LABEL_SIZE,
                    pad=12,
                )
    return fig
