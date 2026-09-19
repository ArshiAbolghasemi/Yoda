"""End-to-end dataset orchestration."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from yoda.common.logger import logger
from yoda.config import load_config
from yoda.data.download import download_market, download_news
from yoda.data.features import add_indicators
from yoda.data.merge import build_dataset
from yoda.data.news import align_and_aggregate_news, process_news
from yoda.data.preprocess import preprocess_market
from yoda.data.settings import DataConfig
from yoda.data.validate import validate_dataset


@dataclass
class DataPipeline:
    config: DataConfig

    def run(self) -> Path:
        storage = self.config.storage
        root = storage.root
        raw_market_dir, raw_news_dir = (
            root / storage.raw_market,
            root / storage.raw_news,
        )
        processed_dir = root / storage.processed
        processed_dir.mkdir(parents=True, exist_ok=True)

        raw_market = download_market(
            self.config.market, raw_market_dir, self.config.download.workers
        )
        market = preprocess_market(raw_market)
        market.to_parquet(processed_dir / "market.parquet", index=False)
        features = add_indicators(
            market, fill_infinite=self.config.indicators.fill_infinite
        )

        news_symbols = [
            asset for asset, kind in self.config.market.assets.items() if kind != "fx"
        ]
        raw_news = download_news(
            self.config.news,
            news_symbols,
            self.config.market.start,
            self.config.market.end,
            raw_news_dir,
            self.config.download.workers,
        )
        news = process_news(raw_news, self.config.market.assets, self.config.news)
        news.to_parquet(processed_dir / "news.parquet", index=False)
        daily_news = align_and_aggregate_news(news, market, self.config.news.separator)

        dataset = build_dataset(features, daily_news, self.config.targets.horizons)
        validate_dataset(dataset, self.config.targets.horizons)
        if self.config.dataset.drop_incomplete_targets:
            dataset = dataset.dropna(
                subset=[
                    f"future_return_{horizon}d"
                    for horizon in self.config.targets.horizons
                ]
            )
        output = processed_dir / storage.final_filename
        dataset.to_parquet(output, index=False)
        for raw_dir in (raw_market_dir, raw_news_dir):
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
        logger.info("raw_data_cleaned paths=%s,%s", raw_market_dir, raw_news_dir)
        logger.info(
            "pipeline_done rows=%d assets=%d start=%s end=%s output=%s",
            len(dataset),
            dataset["asset"].nunique(),
            dataset["date"].min().date(),
            dataset["date"].max().date(),
            output,
        )
        return output


def main() -> None:
    DataPipeline(load_config().data).run()


if __name__ == "__main__":
    main()
