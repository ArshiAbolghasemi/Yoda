"""Shared Dynaconf configuration entry point.

Everything is read from the repo-root ``.env``, and both that file and the
default storage root are anchored to the repository rather than to the current
working directory. Otherwise running from a subdirectory - ``llm-serve/``, a
notebook, a scratch folder - silently resolves ``data/`` somewhere else and the
run writes its artifacts into a directory nobody will look in.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from dynaconf import Dynaconf

from yoda.common.logger import logger
from yoda.config.research import ResearchConfig, build_research_config, resolve
from yoda.data.settings import DataConfig, build_data_config


def project_root() -> Path:
    """The directory holding ``pyproject.toml``; cwd if the package is zipped."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


PROJECT_ROOT = project_root()
DOTENV = PROJECT_ROOT / ".env"


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
    # Explicit path, not discovery: python-dotenv walks up from the cwd, which
    # finds the right file from inside the repo and the wrong one (or none)
    # from anywhere else. Real environment variables still win over the file.
    settings = Dynaconf(
        envvar_prefix=False,
        environments=False,
        load_dotenv=True,
        dotenv_path=DOTENV,
    )
    config = Config(
        data=_anchor(build_data_config(settings)),
        research=build_research_config(settings),
    )
    logger.info(
        "config_loaded root=%s dotenv=%s dataset=%s splits=%s",
        config.data_root,
        DOTENV if DOTENV.is_file() else "absent",
        config.dataset_path,
        config.research.split.ranges,
    )
    return config


def _anchor(data: DataConfig) -> DataConfig:
    """Resolve a relative ``STORAGE__ROOT`` against the repo, not the cwd."""
    storage = data.storage
    if storage.root.is_absolute():
        return data
    return dataclasses.replace(
        data, storage=dataclasses.replace(storage, root=PROJECT_ROOT / storage.root)
    )
