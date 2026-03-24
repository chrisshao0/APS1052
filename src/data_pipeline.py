from __future__ import annotations

import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from src.config import Settings


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/csv,text/plain,*/*",
        }
    )
    return session


def _request_with_retry(url: str, params: dict | None = None, timeout: int = 30, max_attempts: int = 3):
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = _session().get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            print(f"Request failed on attempt {attempt}/{max_attempts}: {url}")
            print(error)

            if attempt < max_attempts:
                time.sleep(2 * attempt)

    raise last_error


def _read_or_download_csv(url: str, file_path: Path, timeout: int = 30, max_attempts: int = 3) -> pd.DataFrame:
    if file_path.exists():
        return pd.read_csv(file_path)

    response = _request_with_retry(url=url, timeout=timeout, max_attempts=max_attempts)
    file_path.write_bytes(response.content)
    return pd.read_csv(StringIO(response.text))


def _save_frame(frame: pd.DataFrame, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(file_path, index=False)


def _normalize_date_column(frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
    frame = frame.copy()
    frame[column_name] = pd.to_datetime(frame[column_name]).dt.tz_localize(None).dt.normalize()
    return frame


def _flatten_yfinance_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [col[0] if isinstance(col, tuple) else col for col in frame.columns]

    frame = frame.reset_index()

    if "Date" not in frame.columns and "index" in frame.columns:
        frame = frame.rename(columns={"index": "Date"})

    return frame


def download_coinmetrics_btc(settings: Settings) -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
    file_path = settings.data_raw_dir / "coinmetrics_btc.csv"

    frame = _read_or_download_csv(url, file_path)
    frame = _normalize_date_column(frame, "time")
    frame = frame.rename(columns={"time": "date", "PriceUSD": "btc_close"})
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    frame = frame[frame["date"] >= pd.Timestamp(settings.start_date)].copy()
    return frame


def download_fear_greed(settings: Settings) -> pd.DataFrame:
    file_path = settings.data_raw_dir / "fear_greed_daily.csv"

    if file_path.exists():
        frame = pd.read_csv(file_path)
        frame = _normalize_date_column(frame, "date")
        return frame

    url = "https://api.alternative.me/fng/?limit=0&format=json"
    response = _request_with_retry(url=url, timeout=30, max_attempts=3)

    payload = response.json()["data"]
    frame = pd.DataFrame(payload)

    frame["date"] = pd.to_datetime(frame["timestamp"].astype(int), unit="s").dt.normalize()
    frame["fear_greed_index"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[["date", "fear_greed_index"]].sort_values("date").drop_duplicates("date", keep="last")
    frame = frame[frame["date"] >= pd.Timestamp(settings.start_date)].reset_index(drop=True)

    _save_frame(frame, file_path)
    return frame


def download_binance_funding(settings: Settings) -> pd.DataFrame:
    file_path = settings.data_raw_dir / "binance_btcusdt_funding_daily.csv"

    if file_path.exists():
        frame = pd.read_csv(file_path)
        frame = _normalize_date_column(frame, "date")
        return frame

    endpoint = "https://fapi.binance.com/fapi/v1/fundingRate"
    symbol = "BTCUSDT"

    start_timestamp_ms = int(pd.Timestamp(settings.start_date).timestamp() * 1000)
    end_timestamp_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)

    all_rows = []
    current_start = start_timestamp_ms

    while current_start < end_timestamp_ms:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": end_timestamp_ms,
            "limit": 1000,
        }

        response = _request_with_retry(url=endpoint, params=params, timeout=30, max_attempts=3)
        chunk = response.json()

        if not chunk:
            break

        all_rows.extend(chunk)

        last_timestamp = int(chunk[-1]["fundingTime"])
        current_start = last_timestamp + 1
        time.sleep(0.20)

    if not all_rows:
        raise RuntimeError("No funding data was returned by Binance.")

    frame = pd.DataFrame(all_rows)
    frame["date"] = pd.to_datetime(frame["fundingTime"].astype("int64"), unit="ms").dt.normalize()
    frame["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")

    frame = (
        frame.groupby("date", as_index=False)["funding_rate"]
        .mean()
        .sort_values("date")
        .reset_index(drop=True)
    )

    _save_frame(frame, file_path)
    return frame


def download_yahoo_close(settings: Settings, ticker: str, file_name: str, column_name: str) -> pd.DataFrame:
    file_path = settings.data_raw_dir / file_name

    if file_path.exists():
        frame = pd.read_csv(file_path)
        frame = _normalize_date_column(frame, "date")
        return frame

    frame = yf.download(
        ticker,
        start=settings.start_date,
        progress=False,
        auto_adjust=True,
        actions=False,
        threads=False,
    )

    if frame.empty:
        raise RuntimeError(f"No Yahoo Finance data returned for {ticker}.")

    frame = _flatten_yfinance_columns(frame)
    close_candidates = [col for col in frame.columns if str(col).lower() == "close"]

    if not close_candidates:
        raise RuntimeError(f"Close column not found for {ticker}.")

    frame = frame.rename(columns={"Date": "date", close_candidates[0]: column_name})
    frame = frame[["date", column_name]].copy()
    frame = _normalize_date_column(frame, "date")
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    _save_frame(frame, file_path)
    return frame


def download_us_dollar_index(settings: Settings) -> pd.DataFrame:
    """
    Try Yahoo's DXY symbol first.
    If that fails, fall back to UUP as a dollar proxy.
    """
    candidates = [
        ("DX-Y.NYB", "us_dollar_index_close.csv", "us_dollar_index_close"),
        ("UUP", "uup_close.csv", "us_dollar_index_close"),
    ]

    last_error = None

    for ticker, file_name, column_name in candidates:
        try:
            frame = download_yahoo_close(settings, ticker, file_name, column_name)
            frame["us_dollar_index_source"] = ticker
            print(f"Using dollar series: {ticker}")
            return frame
        except Exception as error:
            last_error = error
            print(f"Warning: failed to download {ticker}")
            print(error)

    print("Warning: no dollar index proxy was available. Skipping this feature.")
    return pd.DataFrame(columns=["date", "us_dollar_index_close", "us_dollar_index_source"])


def build_master_dataset(settings: Settings) -> pd.DataFrame:
    settings.make_directories()

    coinmetrics = download_coinmetrics_btc(settings)
    fear_greed = download_fear_greed(settings)
    funding = download_binance_funding(settings)
    spx = download_yahoo_close(settings, "^GSPC", "spx_close.csv", "spx_close")
    vix = download_yahoo_close(settings, "^VIX", "vix_close.csv", "vix_close")
    gold = download_yahoo_close(settings, "GLD", "gld_close.csv", "gld_close")
    dollar_index = download_us_dollar_index(settings)

    frame = coinmetrics.copy()

    for other in [fear_greed, funding, spx, vix, gold, dollar_index]:
        if not other.empty:
            frame = frame.merge(other, on="date", how="left")

    frame = frame.sort_values("date").reset_index(drop=True)

    fill_columns = [
        "fear_greed_index",
        "funding_rate",
        "spx_close",
        "vix_close",
        "gld_close",
        "us_dollar_index_close",
    ]
    for column in fill_columns:
        if column in frame.columns:
            frame[column] = frame[column].ffill()

    frame = frame[frame["date"] >= pd.Timestamp(settings.start_date)].reset_index(drop=True)

    _save_frame(frame, settings.data_processed_dir / "master_dataset.csv")
    return frame