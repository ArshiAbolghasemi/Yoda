"""Shared Dynaconf configuration entry point."""

from dataclasses import dataclass

from dynaconf import Dynaconf

from yoda.data.settings import DataConfig, build_data_config


@dataclass(frozen=True)
class Config:
    data: DataConfig


def load_config() -> Config:
    settings = Dynaconf(
        envvar_prefix=False,
        environments=False,
        load_dotenv=True,
    )
    return Config(data=build_data_config(settings))
