"""``evaluate`` - score saved runs over any set of intervals, offline.

Takes one or more ``run_id``s and produces, for every interval, one metric row
per run plus the figures. Nothing here re-runs a backtest; it reads
``weights.parquet`` and ``ledger.parquet``, which is what makes side-by-side
baseline comparison on *identical* intervals cheap and exact.

The only thing that needs data beyond the artifacts is ENB, which requires an
asset covariance. Pass the panel (or let ``evaluate`` build it) and it is
computed per interval; without it ENB comes back NaN and everything else still
scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from yoda.backtest.artifacts import Run, load_run
from yoda.common.alignment import AlignedPanel, build_panel
from yoda.common.logger import logger
from yoda.config.settings import Config
from yoda.evaluation import plots
from yoda.evaluation.intervals import (
    Interval,
    resolve,
    slice_ledger,
    slice_weights,
    weight_matrix,
)
from yoda.evaluation.metrics import METRIC_ORDER, performance


@dataclass
class Evaluation:
    tables: dict[str, pd.DataFrame]  # interval name -> run x metric
    figures: list[Path] = field(default_factory=list)
    report: Path | None = None

    def overall(self) -> pd.DataFrame:
        return self.tables.get("full", next(iter(self.tables.values())))


def _returns(run: Run) -> pd.Series:
    ledger = run.ledger.copy()
    ledger["date"] = pd.to_datetime(ledger["date"])
    return ledger.set_index("date")["port_return"].sort_index()


def _covariance(
    panel: AlignedPanel | None, interval: Interval, assets: list[str]
) -> np.ndarray | None:
    if panel is None:
        return None
    rows = panel.positions(interval.start, interval.end)
    if len(rows) < len(assets):
        return None
    columns = [panel.assets.index(asset) for asset in assets if asset in panel.assets]
    if len(columns) != len(assets):
        return None
    return np.cov(panel.returns[rows][:, columns], rowvar=False)


def evaluate(
    config: Config,
    run_ids: str | list[str],
    *,
    panel: AlignedPanel | None = None,
    make_plots: bool | None = None,
    title: str = "",
) -> Evaluation:
    """Score every run over every resolved interval and write the report."""
    ids = [run_ids] if isinstance(run_ids, str) else list(run_ids)
    root = config.path(config.research.backtest.runs)
    runs = [load_run(root, run_id) for run_id in ids]
    settings = config.research.evaluation
    alpha = config.research.optimizer.alpha
    draw = settings.plots if make_plots is None else make_plots
    if panel is None and draw:
        panel = build_panel(config)

    intervals = resolve(runs[0].ledger, config.research.split, settings)
    tables: dict[str, pd.DataFrame] = {}
    for interval in intervals:
        rows = {}
        for run in runs:
            ledger = slice_ledger(run.ledger, interval)
            books = weight_matrix(slice_weights(run.weights, interval))
            average = books.mean().to_numpy() if not books.empty else None
            rows[run.label] = performance(
                _returns(run).loc[interval.start : interval.end],
                alpha=alpha,
                periods_per_year=settings.periods_per_year,
                turnover=ledger["turnover"] if "turnover" in ledger else None,
                weights=average,
                covariance=_covariance(panel, interval, list(books.columns))
                if average is not None
                else None,
            )
        tables[interval.name] = pd.DataFrame(rows).T[list(METRIC_ORDER)]

    evaluation = Evaluation(tables=tables)
    if not draw:
        return evaluation

    target = runs[0].path if len(runs) == 1 else root / "_compare"
    target.mkdir(parents=True, exist_ok=True)
    curves = {run.label: _returns(run) for run in runs}
    evaluation.figures = [
        plots.cumulative_return(curves, target / "cumulative_return.png"),
        plots.drawdown_curve(curves, target / "drawdown.png"),
        plots.rolling_sharpe_plot(
            curves, tuple(settings.rolling_windows), target / "rolling_sharpe.png"
        ),
    ]
    for run in runs:
        books = weight_matrix(run.weights)
        evaluation.figures.append(
            plots.weight_area(books, target / f"weights_{run.run_id}.png", run.label)
        )
    evaluation.report = plots.html_summary(
        target / "report.html",
        title or " vs ".join(run.label for run in runs),
        tables,
        evaluation.figures,
    )
    logger.info(
        "evaluated runs=%s intervals=%s report=%s",
        ",".join(ids),
        ",".join(tables),
        evaluation.report,
    )
    return evaluation
