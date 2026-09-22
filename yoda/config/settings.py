"""Shared Dynaconf configuration entry point."""

from dataclasses import dataclass
from pathlib import Path

from dynaconf import Dynaconf

from yoda.common.logger import logger
from yoda.config.research import ResearchConfig, build_research_config, resolve
from yoda.data.settings import DataConfig, build_data_config


@dataclass(frozen=True)
class Config:
    data: DataConfig
    research: ResearchConfig

    @property
    def data_root(self) -> Path:
        return self.data.storage.root

    def path(self, value: str) -> Path:
        """Resolve a storage-relative research setting against the data root."""
        return resolve(self.data_root, value)

    @property
    def dataset_path(self) -> Path:
        """Location of ``financial_dataset.parquet``, wherever DVC put it."""
        if self.research.panel.dataset:
            return self.path(self.research.panel.dataset)
        storage = self.data.storage
        processed = storage.root / storage.processed / storage.final_filename
        return (
            processed if processed.exists() else storage.root / storage.final_filename
        )


def load_config() -> Config:
    settings = Dynaconf(
        envvar_prefix=False,
        environments=False,
        load_dotenv=True,
    )
    config = Config(
        data=build_data_config(settings),
        research=build_research_config(settings),
    )
    logger.info(
        "config_loaded dataset=%s splits=%s",
        config.dataset_path,
        config.research.split.ranges,
    )
    return config
