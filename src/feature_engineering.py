from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Settings


def safe_divide(left: pd.Series, right: pd.Series) -> pd.Series:
    result = left / right
    return result.replace([np.inf, -np.inf], np.nan)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return safe_divide(series - rolling_mean, rolling_std)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    diff = close.diff()
    gains = diff.clip(lower=0.0)
    losses = -diff.clip(upper=0.0)

    average_gain = gains.rolling(window).mean()
    average_loss = losses.rolling(window).mean()

    relative_strength = safe_divide(average_gain, average_loss)
    return 100 - (100 / (1 + relative_strength))


def first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def build_feature_dataset(master_data: pd.DataFrame, settings: Settings):
    """Create supervised learning features and next-period target labels."""
    frame = master_data.copy()
    frame = frame.sort_values("date").reset_index(drop=True)

    close = frame["btc_close"]
    one_day_return = close.pct_change(fill_method=None)
    target_horizon = settings.target_horizon_days
    multi_day_return = close.pct_change(target_horizon, fill_method=None)
    
    frame["target_return"] = multi_day_return.shift(-target_horizon)
    frame["target"] = (frame["target_return"] > 0).astype(int)

    feature_names: list[str] = []
    feature_rows: list[dict] = []

    def add_feature(name: str, values: pd.Series, group: str, source: str) -> None:
        frame[name] = values
        feature_names.append(name)
        feature_rows.append(
            {
                "feature": name,
                "group": group,
                "source": source,
            }
        )

    add_feature("btc_return_1d", one_day_return, "price", "btc")
    add_feature("btc_return_5d", close.pct_change(5, fill_method=None), "price", "btc")
    add_feature("btc_return_21d", close.pct_change(21, fill_method=None), "price", "btc")
    add_feature("btc_volatility_7d", one_day_return.rolling(7).std(), "price", "btc")
    add_feature("btc_volatility_21d", one_day_return.rolling(21).std(), "price", "btc")
    add_feature("btc_ma_gap_7_21", safe_divide(close.rolling(7).mean(), close.rolling(21).mean()) - 1, "price", "btc")
    add_feature("btc_price_zscore_21d", rolling_zscore(close, 21), "price", "btc")
    add_feature("btc_drawdown_30d", safe_divide(close, close.rolling(30).max()) - 1, "price", "btc")
    add_feature("btc_rsi_14", compute_rsi(close, 14), "price", "btc")
    add_feature("btc_momentum_10d", close.pct_change(10, fill_method=None), "price", "btc")

    if "fear_greed_index" in frame.columns:
        add_feature("fear_greed_index", frame["fear_greed_index"], "external", "alternative_me")
        add_feature("fear_greed_change_1d", frame["fear_greed_index"].diff(), "external", "alternative_me")
        add_feature("fear_greed_zscore_21d", rolling_zscore(frame["fear_greed_index"], 21), "external", "alternative_me")

    if "funding_rate" in frame.columns:
        add_feature("funding_rate_daily", frame["funding_rate"], "external", "binance")
        add_feature("funding_rate_mean_7d", frame["funding_rate"].rolling(7).mean(), "external", "binance")
        add_feature("funding_rate_abs_mean_7d", frame["funding_rate"].abs().rolling(7).mean(), "external", "binance")

    if "spx_close" in frame.columns:
        add_feature("spx_return_1d", frame["spx_close"].pct_change(fill_method=None), "external", "spx")
        add_feature("spx_return_5d", frame["spx_close"].pct_change(5, fill_method=None), "external", "spx")

    if "vix_close" in frame.columns:
        add_feature("vix_level", frame["vix_close"], "external", "vix")
        add_feature("vix_change_1d", frame["vix_close"].pct_change(fill_method=None), "external", "vix")
        add_feature("vix_zscore_21d", rolling_zscore(frame["vix_close"], 21), "external", "vix")

    if "gld_close" in frame.columns:
        add_feature("gold_return_5d", frame["gld_close"].pct_change(5, fill_method=None), "external", "gold")
        add_feature("gold_btc_ratio", safe_divide(frame["gld_close"], frame["btc_close"]), "external", "gold")

    if "us_dollar_index_close" in frame.columns:
        add_feature("usd_index_return_5d", frame["us_dollar_index_close"].pct_change(5, fill_method=None), "external", "usd_index")
        add_feature("usd_index_zscore_21d", rolling_zscore(frame["us_dollar_index_close"], 21), "external", "usd_index")

    active_addresses_column = first_existing_column(frame, ["AdrActCnt", "AdrAct"])
    transactions_column = first_existing_column(frame, ["TxCnt", "TxTfrCnt"])
    hash_rate_column = first_existing_column(frame, ["HashRate"])
    mvrv_column = first_existing_column(frame, ["CapMVRVCur", "CapMVRVFF"])
    nvt_column = first_existing_column(frame, ["NVTAdj90", "NVTAdjFF90", "NVT90"])
    fee_native_column = first_existing_column(frame, ["FeeTotNtv"])
    fee_usd_column = first_existing_column(frame, ["FeeTotUSD"])
    market_cap_column = first_existing_column(frame, ["CapMrktCurUSD"])
    realized_cap_column = first_existing_column(frame, ["CapRealUSD"])
    adjusted_transfer_value_column = first_existing_column(frame, ["TxTfrValAdjUSD", "TxTfrValMeanUSD"])
    roi_30d_column = first_existing_column(frame, ["ROI30d"])
    volatility_30d_column = first_existing_column(frame, ["VtyDayRet30d"])

    if active_addresses_column:
        add_feature("active_addresses_change_7d", frame[active_addresses_column].pct_change(7, fill_method=None), "on_chain", active_addresses_column)
        add_feature("active_addresses_zscore_30d", rolling_zscore(frame[active_addresses_column], 30), "on_chain", active_addresses_column)

    if transactions_column:
        add_feature("transaction_count_change_7d", frame[transactions_column].pct_change(7, fill_method=None), "on_chain", transactions_column)

    if hash_rate_column:
        add_feature("hash_rate_change_7d", frame[hash_rate_column].pct_change(7, fill_method=None), "on_chain", hash_rate_column)
        add_feature("hash_rate_zscore_30d", rolling_zscore(frame[hash_rate_column], 30), "on_chain", hash_rate_column)

    if mvrv_column:
        add_feature("mvrv_level", frame[mvrv_column], "on_chain", mvrv_column)

    if nvt_column:
        add_feature("nvt_ratio", frame[nvt_column], "on_chain", nvt_column)

    if roi_30d_column:
        add_feature("onchain_roi_30d", frame[roi_30d_column], "on_chain", roi_30d_column)

    if volatility_30d_column:
        add_feature("onchain_volatility_30d", frame[volatility_30d_column], "on_chain", volatility_30d_column)

    if fee_usd_column and transactions_column:
        add_feature("fee_per_transaction_usd", safe_divide(frame[fee_usd_column], frame[transactions_column]), "on_chain", fee_usd_column)
    elif fee_native_column and transactions_column:
        add_feature("fee_per_transaction_usd", safe_divide(frame[fee_native_column] * frame["btc_close"], frame[transactions_column]), "on_chain", fee_native_column)

    if market_cap_column and adjusted_transfer_value_column:
        add_feature("market_cap_to_transfer_value", safe_divide(frame[market_cap_column], frame[adjusted_transfer_value_column]), "on_chain", f"{market_cap_column}|{adjusted_transfer_value_column}")

    if market_cap_column and realized_cap_column:
        add_feature("market_cap_to_realized_cap", safe_divide(frame[market_cap_column], frame[realized_cap_column]), "on_chain", f"{market_cap_column}|{realized_cap_column}")

    keep_columns = ["date", "btc_close", "target", "target_return"] + feature_names
    dataset = frame[keep_columns].copy()

    feature_group_map = {row["feature"]: row["group"] for row in feature_rows}
    lag_by_group = {
        "price": settings.price_feature_lag_days,
        "external": settings.external_feature_lag_days,
        "on_chain": settings.onchain_feature_lag_days,
    }

    for col in feature_names:
        group = feature_group_map.get(col, "external")
        lag_days = lag_by_group.get(group, settings.onchain_feature_lag_days)
        dataset[col] = dataset[col].shift(lag_days)
        
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna().reset_index(drop=True)
    feature_catalog = pd.DataFrame(feature_rows).drop_duplicates("feature").reset_index(drop=True)
    dataset = dataset.drop(columns=[col for col in dataset.columns if dataset[col].std() == 0], errors="ignore")
    # Keep feature_names aligned with dropped constant columns.
    feature_names = [col for col in feature_names if col in dataset.columns]
    return dataset, feature_names, feature_catalog
