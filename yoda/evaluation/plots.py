"""Figures for a run, and overlays across runs, written next to the artifacts.

Four forms, each picked for the job the data does:

* cumulative return - change over time, one line per run;
* drawdown - change over time, filled to the zero baseline;
* rolling Sharpe - change over time, one panel per window;
* weight allocation - composition over time, stacked area.

Series colors come from the validated categorical palette in fixed slot order -
never cycled, never generated. The weight plot has 36 candidate assets and a
categorical palette has 8 slots, so it shows the largest average holdings and
folds the rest into a single muted "Other" band rather than inventing hues.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from yoda.common.logger import logger  # noqa: E402
from yoda.evaluation.metrics import drawdown, rolling_sharpe  # noqa: E402

# Validated categorical palette (light surface), assigned in fixed slot order.
SERIES: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
OTHER = "#b8b7b0"
TOP_ASSETS = 7  # + "Other": one categorical slot each, none generated


def _figure(title: str, ylabel: str, height: float = 3.6):
    figure, axes = plt.subplots(figsize=(9, height), dpi=160)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    axes.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    axes.set_ylabel(ylabel, color=MUTED, fontsize=9)
    axes.grid(True, color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=MUTED, labelsize=8)
    return figure, axes


def _finish(figure, axes, path: Path, labelled: bool) -> Path:
    if not labelled:
        legend = axes.legend(frameon=False, fontsize=8, loc="upper left")
        for text in legend.get_texts():
            text.set_color(MUTED)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
    return path


def _label_end(axes, series: pd.Series, label: str, color: str) -> None:
    """Direct label at the last point - used only when there are few series."""
    axes.annotate(
        label,
        xy=(series.index[-1], series.iloc[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        color=color,
        fontsize=8,
        va="center",
    )


def cumulative_return(curves: dict[str, pd.Series], path: Path) -> Path:
    figure, axes = _figure("Cumulative return", "growth of 1")
    direct = len(curves) <= 4
    for slot, (label, returns) in enumerate(curves.items()):
        equity = (1.0 + returns).cumprod()
        color = SERIES[slot % len(SERIES)]
        axes.plot(
            equity.index, equity.to_numpy(), linewidth=2, color=color, label=label
        )
        if direct:
            _label_end(axes, equity, label, color)
    axes.axhline(1.0, color=GRID, linewidth=1)
    return _finish(figure, axes, path, direct)


def drawdown_curve(curves: dict[str, pd.Series], path: Path) -> Path:
    figure, axes = _figure("Drawdown", "peak-to-trough")
    single = len(curves) == 1
    for slot, (label, returns) in enumerate(curves.items()):
        series = drawdown(returns)
        color = SERIES[slot % len(SERIES)]
        axes.plot(
            series.index, series.to_numpy(), linewidth=2, color=color, label=label
        )
        if single:
            axes.fill_between(
                series.index, series.to_numpy(), 0, color=color, alpha=0.15
            )
    return _finish(figure, axes, path, single)


def rolling_sharpe_plot(
    curves: dict[str, pd.Series], windows: tuple[int, ...], path: Path
) -> Path:
    figure, panels = plt.subplots(
        len(windows), 1, figsize=(9, 2.8 * len(windows)), dpi=160, sharex=True
    )
    panels = [panels] if len(windows) == 1 else list(panels)
    figure.patch.set_facecolor(SURFACE)
    for axes, window in zip(panels, windows, strict=True):
        axes.set_facecolor(SURFACE)
        axes.set_title(f"Rolling Sharpe, {window}d", color=INK, fontsize=11, loc="left")
        axes.grid(True, color=GRID, linewidth=0.8)
        axes.set_axisbelow(True)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        axes.tick_params(colors=MUTED, labelsize=8)
        axes.axhline(0.0, color=GRID, linewidth=1)
        for slot, (label, returns) in enumerate(curves.items()):
            series = rolling_sharpe(returns, window).dropna()
            axes.plot(
                series.index,
                series.to_numpy(),
                linewidth=2,
                color=SERIES[slot % len(SERIES)],
                label=label,
            )
    legend = panels[0].legend(frameon=False, fontsize=8, loc="upper left")
    for text in legend.get_texts():
        text.set_color(MUTED)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
    return path


def weight_area(weights: pd.DataFrame, path: Path, label: str = "") -> Path:
    """Stacked-area allocation over time from the wide weight matrix."""
    ranked = weights.mean().sort_values(ascending=False)
    top = list(ranked.index[:TOP_ASSETS])
    shown = weights[top].copy()
    remainder = weights.drop(columns=top)
    if not remainder.empty:
        shown["Other"] = remainder.sum(axis=1)

    figure, axes = _figure(
        f"Weight allocation over time{f' - {label}' if label else ''}", "weight", 4.2
    )
    colors = [SERIES[index] for index in range(len(top))]
    if "Other" in shown:
        colors.append(OTHER)
    axes.stackplot(
        shown.index,
        shown.to_numpy().T,
        labels=list(shown.columns),
        colors=colors,
        linewidth=0.5,
        edgecolor=SURFACE,  # 2px-equivalent surface gap between segments
    )
    axes.set_ylim(0, 1)
    axes.margins(x=0)
    legend = axes.legend(
        frameon=False,
        fontsize=8,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    for text in legend.get_texts():
        text.set_color(MUTED)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
    return path


def html_summary(
    path: Path, title: str, tables: dict[str, pd.DataFrame], figures: list[Path]
) -> Path:
    """One self-contained page: the metric tables plus the figures beside them."""
    blocks = [
        f"<h2>{name}</h2>{frame.to_html(float_format=lambda v: f'{v:,.4f}', border=0)}"
        for name, frame in tables.items()
    ]
    images = "".join(
        f'<figure><img src="{figure.name}" alt="{figure.stem}"></figure>'
        for figure in figures
    )
    path.write_text(
        f"""<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>
 body{{background:{SURFACE};color:{INK};max-width:1100px;margin:2rem auto;
       font:14px/1.5 system-ui,sans-serif}}
 h1{{font-size:1.4rem}} h2{{font-size:1.05rem;color:{MUTED};margin-top:2rem}}
 table{{border-collapse:collapse;width:100%;font-size:12px}}
 th,td{{padding:.35rem .5rem;text-align:right;border-bottom:1px solid {GRID}}}
 th:first-child,td:first-child{{text-align:left}}
 figure{{margin:1.5rem 0}} img{{width:100%;border:1px solid {GRID};border-radius:4px}}
</style>
<h1>{title}</h1>{"".join(blocks)}<h2>Figures</h2>{images}"""
    )
    logger.info("report_written path=%s figures=%d", path, len(figures))
    return path
