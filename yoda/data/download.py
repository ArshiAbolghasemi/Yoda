"""Raw Yahoo Finance and Alpaca News downloads."""

from __future__ import annotations

import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.auto import tqdm

from yoda.common.logger import logger
from yoda.data.settings import MarketDataConfig, NewsConfig

RETRY = {
    "stop": stop_after_attempt(3),
    "wait": wait_exponential(multiplier=1, min=1, max=10),
    "reraise": True,
}


@retry(
    retry=retry_if_exception_type(
        (KeyError, TypeError, ValueError, requests.RequestException)
    ),
    **RETRY,
)
def _download_market_asset(symbol: str, start: str, end: str) -> pd.DataFrame:
    return yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )


@retry(retry=retry_if_exception_type(requests.RequestException), **RETRY)
def _download_news_page(url: str, headers: dict, params: dict) -> dict:
    response = requests.get(url, headers=headers, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def download_market(
    config: MarketDataConfig, raw_dir: Path, workers: int = 4
) -> pd.DataFrame:
    """Download and preserve one raw Yahoo CSV per asset."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    end_exclusive = (date.fromisoformat(config.end) + timedelta(days=1)).isoformat()
    logger.info(
        "market_download_start assets=%d start=%s end=%s",
        len(config.assets),
        config.start,
        config.end,
    )
    def load(symbol: str, asset_type: str) -> pd.DataFrame | None:
        path = raw_dir / f"{symbol.replace('=', '_')}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            logger.info("market_download_reused asset=%s input=%s", symbol, path)
        else:
            frame = _download_market_asset(symbol, config.start, end_exclusive)
        if frame.empty:
            logger.warning("market_download_empty asset=%s", symbol)
            return None
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        if not path.exists():
            frame.to_csv(path)
        return frame.reset_index().assign(asset=symbol, asset_type=asset_type)

    with (
        ThreadPoolExecutor(max_workers=workers) as executor,
        tqdm(total=len(config.assets), desc="Market assets", unit="asset") as progress,
    ):
        futures = {
            executor.submit(load, symbol, asset_type): symbol
            for symbol, asset_type in config.assets.items()
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                frame = future.result()
                if frame is not None:
                    frames.append(frame)
            except (KeyError, TypeError, ValueError, requests.RequestException):
                logger.exception("market_download_failed asset=%s", symbol)
            progress.update()
    if not frames:
        raise RuntimeError("Yahoo Finance returned no market data")
    result = pd.concat(frames, ignore_index=True)
    logger.info(
        "market_download_done rows=%d assets=%d output=%s",
        len(result),
        result["asset"].nunique(),
        raw_dir,
    )
    return result


def _batches(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def download_news(
    config: NewsConfig,
    symbols: list[str],
    start: str,
    end: str,
    raw_dir: Path,
    workers: int = 4,
) -> list[dict]:
    """Download paginated Alpaca headlines and preserve the exact JSON responses."""
    if not config.api_key or not config.api_secret:
        raise ValueError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
    raw_dir.mkdir(parents=True, exist_ok=True)
    headers = {
        "APCA-API-KEY-ID": config.api_key,
        "APCA-API-SECRET-KEY": config.api_secret,
    }
    all_items: dict[int | str, dict] = {}
    batches = list(_batches(symbols, config.symbol_batch_size)) + [[]]
    logger.info(
        "news_download_start symbol_batches=%d start=%s end=%s",
        len(batches),
        start,
        end,
    )
    progress = tqdm(desc="News pages", unit="page")

    def load_batch(batch_index: int, batch: list[str]) -> list[dict]:
        items: list[dict] = []
        token = None
        page = 0
        reused = 0
        while True:
            params = {
                "start": f"{start}T00:00:00Z",
                "end": f"{end}T23:59:59Z",
                "limit": config.page_size,
                "sort": "asc",
                "include_content": "false",
            }
            if batch:
                params["symbols"] = ",".join(batch)
            if token:
                params["page_token"] = token
            path = raw_dir / f"batch_{batch_index:03d}_page_{page:05d}.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                reused += 1
            else:
                payload = _download_news_page(config.base_url, headers, params)
                path.write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
            progress.update()
            items.extend(payload.get("news", []))
            token = payload.get("next_page_token")
            page += 1
            if not token:
                break
        if reused:
            logger.info("news_download_reused batch=%d pages=%d", batch_index, reused)
        return items

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(
                lambda indexed: load_batch(*indexed), enumerate(batches)
            )
            for batch_index, items in enumerate(results):
                for item_index, item in enumerate(items):
                    all_items[item.get("id", f"{batch_index}:{item_index}")] = item
    finally:
        progress.close()
    result = list(all_items.values())
    logger.info("news_download_done items=%d output=%s", len(result), raw_dir)
    return result
