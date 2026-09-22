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

from yoda.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="yoda", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("data", help="run the dataset pipeline")
    for name in ("tail-voli-risk", "tail-voli-risk-rl"):
        run = sub.add_parser(name, help=f"run {name}")
        run.add_argument("--gate", default="tailvoi")
        run.add_argument("--run-id", default=name.replace("-", "_"))
    experiments = sub.add_parser("experiments", help="run the baselines and ablations")
    experiments.add_argument(
        "--families",
        nargs="+",
        default=["gate", "system", "policy", "news"],
        help="which experiment families to run",
    )
    arguments = parser.parse_args()

    if arguments.command == "data":
        from yoda.data.pipeline import DataPipeline

        DataPipeline(load_config().data).run()
        return

    config = load_config()
    if arguments.command == "experiments":
        from yoda.pipeline.experiments import run_experiments

        evaluation = run_experiments(config, families=tuple(arguments.families))
    else:
        from yoda.pipeline import run_tail_voli_risk, run_tail_voli_risk_rl

        static = arguments.command == "tail-voli-risk"
        run = run_tail_voli_risk if static else run_tail_voli_risk_rl
        evaluation = run(config, run_id=arguments.run_id, gate=arguments.gate)

    for name, table in evaluation.tables.items():
        print(f"\n== {name} ==")
        print(table.round(4).to_string())
    if evaluation.report:
        print(f"\nreport: {evaluation.report}")


if __name__ == "__main__":
    main()
