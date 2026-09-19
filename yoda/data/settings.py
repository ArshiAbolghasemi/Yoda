"""Configuration domains for the dataset pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import getenv
from pathlib import Path
from typing import Any

from dynaconf import Dynaconf

DOW_30 = (
    "AAPL",
    "AMGN",
    "AXP",
    "BA",
    "CAT",
    "CRM",
    "CSCO",
    "CVX",
    "DIS",
    "DOW",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "JPM",
    "KO",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "V",
    "VZ",
    "WMT",
)
DEFAULT_FX = ("EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X")
DEFAULT_FX_KEYWORDS = {
    "EURUSD=X": ("euro", "eur", "ecb", "european central bank", "federal reserve"),
    "GBPUSD=X": ("pound", "sterling", "gbp", "bank of england", "federal reserve"),
    "USDJPY=X": ("yen", "jpy", "bank of japan", "boj", "federal reserve"),
    "AUDUSD=X": ("australian dollar", "aud", "reserve bank of australia", "rba"),
    "USDCAD=X": ("canadian dollar", "cad", "bank of canada", "boc"),
    "USDCHF=X": ("swiss franc", "chf", "swiss national bank", "snb"),
}


@dataclass(frozen=True)
class MarketDataConfig:
    start: str = "2017-01-01"
    end: str = "2026-09-18"
    bitcoin: tuple[str, ...] = ("BTC-USD",)
    fx: tuple[str, ...] = DEFAULT_FX
    equities: tuple[str, ...] = DOW_30

    @property
    def assets(self) -> dict[str, str]:
        return {
            **dict.fromkeys(self.bitcoin, "bitcoin"),
            **dict.fromkeys(self.fx, "fx"),
            **dict.fromkeys(self.equities, "equity"),
        }


@dataclass(frozen=True)
class NewsConfig:
    base_url: str = "https://data.alpaca.markets/v1beta1/news"
    api_key: str = ""
    api_secret: str = ""
    page_size: int = 50
    symbol_batch_size: int = 20
    market_timezone: str = "America/New_York"
    market_close_hour: int = 16
    separator: str = " || "
    fx_keywords: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: DEFAULT_FX_KEYWORDS.copy()
    )


@dataclass(frozen=True)
class IndicatorConfig:
    fill_infinite: bool = True


@dataclass(frozen=True)
class DatasetConfig:
    drop_incomplete_targets: bool = False


@dataclass(frozen=True)
class DownloadConfig:
    workers: int = 4

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("Download workers must be positive")


@dataclass(frozen=True)
class StorageConfig:
    root: Path = Path("data")
    raw_market: str = "raw/market"
    raw_news: str = "raw/news"
    processed: str = "processed"
    final_filename: str = "financial_dataset.parquet"


@dataclass(frozen=True)
class TargetConfig:
    horizons: tuple[int, ...] = (1, 5, 10, 20)


@dataclass(frozen=True)
class DataConfig:
    market: MarketDataConfig
    news: NewsConfig
    indicators: IndicatorConfig
    dataset: DatasetConfig
    download: DownloadConfig
    storage: StorageConfig
    targets: TargetConfig


def _values(settings: Dynaconf, section: str) -> dict[str, Any]:
    value = settings.get(section, {})
    return (
        {str(key).lower(): item for key, item in dict(value).items()} if value else {}
    )


def build_data_config(settings: Dynaconf) -> DataConfig:
    """Build the data domain from the shared project settings."""
    data = settings.get("data", {})

    def section(name: str) -> dict[str, Any]:
        value = data.get(name, {}) if data else {}
        if not value:
            return _values(settings, name)
        return {str(key).lower(): item for key, item in dict(value).items()}

    market = section("market")
    news = section("news")
    news.setdefault("api_key", getenv("APCA_API_KEY_ID", ""))
    news.setdefault("api_secret", getenv("APCA_API_SECRET_KEY", ""))
    storage = section("storage")
    if "root" in storage:
        storage["root"] = Path(storage["root"])
    if "fx_keywords" in news:
        news["fx_keywords"] = {k: tuple(v) for k, v in news["fx_keywords"].items()}
    for key in ("bitcoin", "fx", "equities"):
        if key in market:
            market[key] = tuple(market[key])
    for key in ("start", "end"):
        if key in market and hasattr(market[key], "isoformat"):
            market[key] = market[key].isoformat()
    targets = section("targets")
    if "horizons" in targets:
        targets["horizons"] = tuple(int(value) for value in targets["horizons"])
    return DataConfig(
        market=MarketDataConfig(**market),
        news=NewsConfig(**news),
        indicators=IndicatorConfig(**section("indicators")),
        dataset=DatasetConfig(**section("dataset")),
        download=DownloadConfig(**section("download")),
        storage=StorageConfig(**storage),
        targets=TargetConfig(**targets),
    )
