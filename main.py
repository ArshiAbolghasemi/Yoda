"""TailRiskFlow entry points.

    uv run main.py data                # rebuild financial_dataset.parquet
    uv run main.py tail-voli-risk      # no-RL stack, static risk policy
    uv run main.py tail-voli-risk-rl   # same stack, RL risk controller
    uv run main.py experiments         # every baseline and ablation, one table

Everything else - splits, intervals, the news backend, credentials - comes from
``.env`` via Dynaconf. See ``.env.example``.
"""

from __future__ import annotations

import argparse

from yoda.common.alignment import build_panel
from yoda.config import load_config
from yoda.data.pipeline import DataPipeline
from yoda.evaluation.prediction import evaluate_specialists
from yoda.pipeline import run_tail_voli_risk, run_tail_voli_risk_rl
from yoda.pipeline.experiments import run_experiments
from yoda.specialists import load_jev_table


def main() -> None:
    parser = argparse.ArgumentParser(prog="yoda", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("data", help="run the dataset pipeline")
    for name in ("tail-voli-risk", "tail-voli-risk-rl"):
        run = sub.add_parser(name, help=f"run {name}")
        run.add_argument(
            "--gate",
            default="tailvoi",
            help="tailvoi | accuracy | attention | equal_weight | cio "
            "(cio takes over the gate, the policy and the allocator)",
        )
        run.add_argument("--run-id", default=name.replace("-", "_"))
    specialists = sub.add_parser(
        "specialists", help="score specialist prediction quality (before portfolios)"
    )
    specialists.add_argument(
        "--channels", nargs="+", default=["technical", "volatility", "news"]
    )
    experiments = sub.add_parser("experiments", help="run the baselines and ablations")
    experiments.add_argument(
        "--families",
        nargs="+",
        default=["gate", "optimizer", "policy", "openjev", "sources", "horizon"],
        help="which experiment families to run",
    )
    arguments = parser.parse_args()

    if arguments.command == "data":
        DataPipeline(load_config().data).run()
        return

    config = load_config()
    if arguments.command == "specialists":
        panel = build_panel(config)
        rows = panel.positions(
            config.research.split.test_start, config.research.split.test_end
        )
        tables = {}
        for channel in arguments.channels:
            try:
                tables[channel] = load_jev_table(config, channel)
            except FileNotFoundError:
                print(f"no cached OpenJev answers for {channel!r}; skipping")
        if not tables:
            raise SystemExit("nothing to score - build the OpenJev features first")
        print(evaluate_specialists(tables, panel, rows).round(4).to_string())
        return

    if arguments.command == "experiments":
        evaluation = run_experiments(config, families=tuple(arguments.families))
    else:
        static = arguments.command == "tail-voli-risk"
        evaluation = (
            run_tail_voli_risk(config, run_id=arguments.run_id, gate=arguments.gate)
            if static
            else run_tail_voli_risk_rl(
                config, run_id=arguments.run_id, gate=arguments.gate
            )
        )

    for name, table in evaluation.tables.items():
        print(f"\n== {name} ==")
        print(table.round(4).to_string())
    if evaluation.report:
        print(f"\nreport: {evaluation.report}")


if __name__ == "__main__":
    main()
