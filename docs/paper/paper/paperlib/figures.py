"""Plotting helpers for benchmark papers: counts that span orders of magnitude on a log axis.

Nothing here knows about one particular figure. Data arrive as pandas objects, and a
figure-specific chunk in the paper composes these helpers.

Text is sized to sit beside the paper's own text, so a panel is drawn at the size it is printed:
each panel carries its own size, and ``qfigure`` places the panels in the document as inline
boxes inside one captioned figure, which the renderer wraps onto separate lines when the page is
too narrow to hold them side by side.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
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
    "series_colors",
    "growth",
    "decade_limits",
    "compact_count",
    "log_axis",
    "annotate_all",
    "Margins",
    "MARGINS",
    "trend_panel",
    "profile_panel",
    "qfigure",
]

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


@dataclass(frozen=True)
class Margins:
    """Room around the axes, in inches. Text is sized in points, so the room it needs is absolute.

    Panels drawn with the same margins at the same figure size have identical plot areas, so a
    shared y range lines up across panels.
    """

    left: float = 0.8  # tick labels and a two-line axis label
    bottom: float = 1.0  # rotated stage names
    right: float = 0.1
    top: float = 0.4  # legend row, clear of the labels above the highest point

    def rect(self, fig: Figure) -> tuple[float, float, float, float]:
        """The axes rectangle in figure fractions: left, bottom, width, height."""
        width, height = fig.get_size_inches()
        return (
            self.left / width,
            self.bottom / height,
            1 - (self.left + self.right) / width,
            1 - (self.bottom + self.top) / height,
        )


MARGINS = Margins()


def new_panel(
    margins: Margins, size: tuple[float, float] | None, tag: str | None
) -> tuple[Figure, Axes]:
    """A figure of ``size`` inches (default: the current frame) with its axes and corner tag.

    Call inside ``plt.rc_context(STYLE)``.
    """
    fig = plt.figure(figsize=size)
    ax = fig.add_axes(margins.rect(fig))
    if tag is not None:
        width, height = fig.get_size_inches()
        fig.text(
            0.02 / width,
            1 - 0.03 / height,
            tag,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=LABEL_SIZE,
        )
    return fig, ax


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
    and minor ticks at 2 to 9 times each decade, which bunch towards the next decade as on any
    logarithmic scale; only ``max_labels`` evenly spaced decades are labelled, so the labels
    stay readable at body-text size.
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


def trend_panel(
    sizes: pd.Series,
    *,
    limits: tuple[float, float],
    xlabel: str,
    ylabel: str,
    margins: Margins = MARGINS,
    size: tuple[float, float] | None = None,
    tag: str | None = None,
) -> Figure:
    """One count per step on a log axis, each point valued and each step marked with its multiplier.

    ``sizes`` is indexed by the step's name; any number of steps works.
    """
    values = sizes.to_numpy()
    steps = np.arange(len(values))
    # the visual midpoint of a step on a log axis
    log_midpoints = np.sqrt(values[:-1] * values[1:])
    with plt.rc_context(STYLE):
        fig, ax = new_panel(margins, size, tag)
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
        annotate_all(  # the first point: beside it, as in a profile panel
            ax,
            steps[:1],
            values[:1],
            texts[:1],
            color=FOCAL,
            fontweight="bold",
            **FIRST_LABEL,
        )
        annotate_all(  # later points: above and left, which is above the rising line
            ax,
            steps[1:],
            values[1:],
            texts[1:],
            xytext=(-8, 6),
            ha="right",
            color=FOCAL,
            fontweight="bold",
        )
        annotate_all(  # above the line, left of its midpoint
            ax,
            steps[:-1] + 0.5,
            log_midpoints,
            (f"×{g:.0f}" for g in growth(values)),
            xytext=(-6, 6),
            ha="right",
            color=GREY,
            fontweight="bold",
        )
    return fig


def profile_panel(
    profiles: pd.DataFrame,
    *,
    limits: tuple[float, float],
    ylabel: str,
    label: Callable[[Hashable], str] = str,
    guide: tuple[float, str] | None = None,
    labelled: pd.DataFrame | None = None,
    xtick_rotation: float = 0.0,
    margins: Margins = MARGINS,
    size: tuple[float, float] | None = None,
    tag: str | None = None,
) -> Figure:
    """One line per row across the ordered columns, on a log axis, points valued.

    Rows are series (labelled in the legend by ``label(row index)``), columns are the stages
    they pass through, and missing values (NaN) are left out. Every point is valued unless the
    boolean frame ``labelled`` (same shape as ``profiles``) says otherwise. ``guide`` draws a
    dashed horizontal reference line at a level, with its caption. The legend is one row above
    the axes. Stage names that do not fit side by side can be rotated by ``xtick_rotation``
    degrees.
    """
    stages = np.arange(profiles.shape[1])
    with plt.rc_context(STYLE):
        fig, ax = new_panel(margins, size, tag)
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
        for (name, row), color in zip(
            profiles.iterrows(), series_colors(len(profiles))
        ):
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
            bbox_to_anchor=(1, 1),
            ncols=min(len(profiles), 4),
            borderaxespad=1.6,
        )
    return fig


def typst_string(text: str) -> str:
    """``text`` as a Typst string literal."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def typst_box(fig: Figure, alt: str) -> str:
    """Typst source for a figure as an inline box, its SVG carried in the source itself."""
    svg = io.StringIO()
    with plt.rc_context({"svg.hashsalt": "paper"}):  # same bytes on every render
        fig.savefig(svg, format="svg", metadata={"Date": None})
    return f'#box(image(bytes({typst_string(svg.getvalue())}), format: "svg", alt: {typst_string(alt)}))'


def qfigure(
    panels: Sequence[Figure],
    *,
    alts: Sequence[str],
    caption: str,
    figure_id: str,
) -> str:
    """Quarto markup for one captioned figure holding ``panels`` side by side, for ``output: asis``.

    The panels are inline boxes in one Typst paragraph, so the renderer puts them on one line when
    they fit and wraps them under each other when they do not. The figure is a Quarto figure
    div, so it is numbered and cross-referenced as ``@figure_id`` and its caption is Quarto
    markdown (cross-references included). The figures are closed: the document holds them as
    markup, not as outputs. Typst output only.
    """
    boxes = " ".join(typst_box(fig, alt) for fig, alt in zip(panels, alts))
    for fig in panels:
        plt.close(fig)
    return f"::: {{#{figure_id}}}\n```{{=typst}}\n{boxes}\n```\n\n{caption}\n:::"
