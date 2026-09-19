# Financial dataset

The `yoda.data` pipeline builds a daily multimodal dataset for Bitcoin, foreign-exchange
pairs, and the Dow Jones component stocks. Its default collection period is January 1,
2017 through September 18, 2026.

## Setup

Copy `.env.example` to `.env` and supply an Alpaca key ID and secret. The `.env` file is
ignored by Git. Configuration is loaded exclusively from environment variables through
`yoda.config.load_config()`. Data-specific configuration classes and defaults live in
`yoda/data/settings.py`; there is no project-specific environment prefix. Alpaca
credentials use their standard names:

```dotenv
APCA_API_BASE_URL=https://paper-api.alpaca.markets/v2
APCA_API_KEY_ID=your_alpaca_key_id
APCA_API_SECRET_KEY=your_alpaca_secret_key
```

The paper-trading endpoint is available to other Yoda packages through the environment.
Historical news requests use `https://data.alpaca.markets/v1beta1/news`, configured as
`data.news.base_url` in Dynaconf.

Adjust dates, assets, target horizons, storage paths, or news attribution with the
`DATA__...` variables listed in `.env.example`. Dynaconf parses lists, numbers, and
booleans from those values. Install the locked runtime and development dependencies into
the active environment, then run with the defaults:

```sh
./scripts/install-dependencies.sh
```

```sh
./scripts/download-data.sh
```

## Collection and processing

Yahoo Finance supplies unadjusted open, high, low, close, volume, and adjusted close.
The end date is converted to an exclusive Yahoo query boundary so September 18, 2026 is
included when it is a trading day. Each downloaded asset is temporarily written as a CSV
under `data/raw/market/` before normalization.

Alpaca supplies historical news. The downloader requests metadata and headlines with
article content disabled, follows pagination, and temporarily writes each response page
under `data/raw/news/`. Equity and Bitcoin news use Alpaca symbol tags. FX news uses the
configurable `data.news.fx_keywords` mapping for currencies, macroeconomic terms, and
central banks.

Each Yahoo asset request and Alpaca page request is retried up to three times with
exponential backoff. Four worker threads download market assets and news batches in
parallel by default; set `DATA__DOWNLOAD__WORKERS` to change the count. Pages within a
news batch remain sequential because each page supplies the next token. After the final
dataset is written successfully, both raw directories are deleted. Failed runs retain
their raw files; the next run reuses existing asset CSVs and news pages, then continues
from the first missing page rather than starting over.

The command displays progress bars for completed market assets and processed news pages.
Alpaca does not report a total page count, so the news bar shows a running page count.

Headlines are whitespace-normalized and deduplicated. News published at or after the
configured New York market close is assigned to the next available trading session.
Weekend and holiday news is also assigned to the first later session. Multiple headlines
for one asset and date are joined with the configured separator, and `news_count` records
the number of unique headlines.

Indicators are calculated independently for each asset in chronological order using only
the current and earlier observations. The feature module emits exactly the configured
schema of 60 indicators: moving averages, momentum, volatility and trend strength,
Bollinger Bands, MACD, volume indicators, Parabolic SAR, Keltner Channels, and Donchian
Channels.

Yahoo generally reports missing or zero FX volume. Normalization converts zero FX volume
to missing values, and all eight volume-dependent indicators remain missing for FX rows.
This prevents synthetic zero volume from being interpreted as a market signal.

Future-return targets use adjusted close at later observations of the same asset:

```text
future_return_Nd = adjusted_close[t + N] / adjusted_close[t] - 1
```

The default horizons are 1, 5, 10, and 20 trading observations. Targets are labels and are
never inputs to indicator calculation. Rows near the dataset end retain missing targets by
default; set `data.dataset.drop_incomplete_targets = true` to omit them from final output.

## Outputs

The pipeline writes normalized market data to `data/processed/market.parquet`, attributed
headline data to `data/processed/news.parquet`, and the unified dataset to
`data/processed/financial_dataset.parquet`.

Each final row contains:

- date, asset identifier, and asset type;
- open, high, low, close, adjusted close, and volume;
- exactly 60 technical-indicator columns;
- aggregated processed headlines and news count; and
- one `future_return_Nd` column per configured horizon.

## Assets

The default dataset covers 37 instruments:

- Bitcoin: `BTC-USD`;
- foreign exchange: `EURUSD=X`, `GBPUSD=X`, `USDJPY=X`, `AUDUSD=X`, `USDCAD=X`,
  and `USDCHF=X`; and
- equities: the 30 Dow Jones component symbols configured in `yoda/data/settings.py`.

The constituents are configuration defaults, not a historical membership record. Change
them with `DATA__MARKET__BITCOIN`, `DATA__MARKET__FX`, or
`DATA__MARKET__EQUITIES`.

## Column reference

### Identity, market, news, and targets

| Column | Contents |
| --- | --- |
| `date` | Normalized UTC-naive trading-session date. |
| `asset` | Yahoo Finance symbol identifying the instrument. |
| `asset_type` | Instrument class: `bitcoin`, `fx`, or `equity`. |
| `open` | Unadjusted session opening price. |
| `high` | Unadjusted session high price. |
| `low` | Unadjusted session low price. |
| `close` | Unadjusted session closing price used by the indicators. |
| `adjusted_close` | Close adjusted by Yahoo Finance for corporate actions; used for targets. |
| `volume` | Reported session trading volume; normally missing for FX. |
| `headlines` | Unique, whitespace-normalized Alpaca headlines attributed to the asset and session, joined by the configured separator. An empty string means no matching news. |
| `news_count` | Number of unique headlines represented by `headlines`; zero means none. |
| `future_return_1d` | Adjusted-close return one later trading observation ahead. |
| `future_return_5d` | Adjusted-close return five later trading observations ahead. |
| `future_return_10d` | Adjusted-close return ten later trading observations ahead. |
| `future_return_20d` | Adjusted-close return twenty later trading observations ahead. |

Target columns follow the configured horizons, so overrides can add, remove, or rename
the default four columns.

### Technical features

Window suffixes are counts of trading observations. Initial rows are missing when an
indicator does not yet have enough history. Price-based indicators use unadjusted OHLC
values.

| Feature | Meaning |
| --- | --- |
| `sma_5` | 5-observation simple moving average of close. |
| `sma_10` | 10-observation simple moving average of close. |
| `sma_20` | 20-observation simple moving average of close. |
| `sma_50` | 50-observation simple moving average of close. |
| `sma_100` | 100-observation simple moving average of close. |
| `sma_200` | 200-observation simple moving average of close. |
| `ema_5` | 5-observation exponential moving average of close. |
| `ema_10` | 10-observation exponential moving average of close. |
| `ema_20` | 20-observation exponential moving average of close. |
| `ema_50` | 50-observation exponential moving average of close. |
| `ema_100` | 100-observation exponential moving average of close. |
| `ema_200` | 200-observation exponential moving average of close. |
| `wma_10` | 10-observation linearly weighted moving average of close, favoring recent values. |
| `wma_20` | 20-observation linearly weighted moving average of close, favoring recent values. |
| `wma_50` | 50-observation linearly weighted moving average of close, favoring recent values. |
| `roc_5` | Percentage close change from 5 observations earlier. |
| `roc_10` | Percentage close change from 10 observations earlier. |
| `roc_20` | Percentage close change from 20 observations earlier. |
| `rsi_7` | 7-observation relative strength index of gains versus losses, scaled 0–100. |
| `rsi_14` | 14-observation relative strength index of gains versus losses, scaled 0–100. |
| `rsi_21` | 21-observation relative strength index of gains versus losses, scaled 0–100. |
| `stoch_k_14` | Close position within the 14-observation high-low range, scaled 0–100. |
| `stoch_d_14` | 3-observation simple moving average of `stoch_k_14`. |
| `williams_r_14` | Close position within the 14-observation high-low range, scaled -100–0. |
| `williams_r_28` | Close position within the 28-observation high-low range, scaled -100–0. |
| `cci_14` | 14-observation commodity channel index of typical price versus its mean deviation. |
| `cci_20` | 20-observation commodity channel index of typical price versus its mean deviation. |
| `atr_7` | Wilder-smoothed 7-observation average true range, measuring price-range volatility. |
| `atr_14` | Wilder-smoothed 14-observation average true range. |
| `atr_21` | Wilder-smoothed 21-observation average true range. |
| `adx_14` | Wilder-smoothed 14-observation average directional index, measuring trend strength. |
| `adx_pos_14` | 14-observation positive directional indicator. |
| `adx_neg_14` | 14-observation negative directional indicator. |
| `bb_middle_20` | 20-observation simple moving average forming the Bollinger center line. |
| `bb_upper_20` | Bollinger upper band: center plus two population standard deviations. |
| `bb_lower_20` | Bollinger lower band: center minus two population standard deviations. |
| `bb_width_20` | Bollinger band width divided by the center line. |
| `bb_percent_20` | Close position between the lower and upper Bollinger bands. |
| `macd_12_26` | Difference between the 12- and 26-observation exponential moving averages. |
| `macd_signal_9` | 9-observation exponential moving average of MACD. |
| `macd_histogram` | MACD minus its signal line. |
| `obv` | Cumulative signed volume based on the direction of each close change. |
| `cmf_20` | 20-observation Chaikin money flow: close location value weighted by volume. |
| `mfi_14` | 14-observation money flow index comparing positive and negative typical-price volume. |
| `force_index_13` | 13-observation exponential average of close change multiplied by volume. |
| `ease_of_movement_14` | 14-observation average of midpoint movement scaled by range and volume. |
| `vwap_14` | 14-observation volume-weighted average typical price. |
| `volume_price_trend` | Cumulative volume multiplied by percentage close change. |
| `negative_volume_index` | Index starting at 1,000 and updated by close returns only when volume falls. |
| `psar_up` | Parabolic SAR value while the detected trend is rising; otherwise missing. |
| `psar_down` | Parabolic SAR value while the detected trend is falling; otherwise missing. |
| `psar_up_indicator` | `1` when `psar_up` is present, otherwise `0`. |
| `psar_down_indicator` | `1` when `psar_down` is present, otherwise `0`. |
| `keltner_middle` | 20-observation exponential moving average forming the Keltner center line. |
| `keltner_upper` | Keltner upper channel: center plus twice `atr_14`. |
| `keltner_lower` | Keltner lower channel: center minus twice `atr_14`. |
| `keltner_width` | Keltner channel width divided by the center line. |
| `keltner_percent` | Close position between the lower and upper Keltner channels. |
| `donchian_upper_20` | Highest high over 20 observations. |
| `donchian_lower_20` | Lowest low over 20 observations. |

The eight volume-derived features (`obv`, `cmf_20`, `mfi_14`,
`force_index_13`, `ease_of_movement_14`, `vwap_14`, `volume_price_trend`, and
`negative_volume_index`) are missing for FX rather than being calculated from absent
volume.

Before writing the final parquet file, validation checks the required schema, indicator
count, unique asset-date keys, OHLC completeness, FX volume-indicator policy, and target
calculations. Lifecycle logging is limited to downloads, failures, row and asset counts,
date ranges, and output paths. All packages share `yoda/common/logger.py`.
