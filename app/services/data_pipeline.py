"""
NeuroTrade — Data Pipeline
Handles fetching, cleaning, feature engineering, and sequence creation.

Flow:
    fetch_ohlcv()
        → add_technical_indicators()
        → clean_and_validate()
        → build_sequences()
        → scale_features()
"""

from __future__ import annotations

import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from cachetools import TTLCache
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

from app.core.config import get_settings

settings = get_settings()

# ── In-memory TTL cache (symbol → DataFrame) ─────────────────────────────────
_data_cache: TTLCache = TTLCache(maxsize=50, ttl=settings.cache_ttl_seconds)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ohlcv(
    symbol: str,
    period: str = "5y",
    interval: str = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance with caching.

    Returns a DataFrame with columns:
        Open, High, Low, Close, Volume, Dividends, Stock Splits
    Index: DatetimeIndex (UTC normalized)
    """
    cache_key = f"{symbol}_{period}_{interval}"
    if not force_refresh and cache_key in _data_cache:
        logger.debug(f"Cache hit for {symbol}")
        return _data_cache[cache_key].copy()

    logger.info(f"Downloading {symbol} | period={period} interval={interval}")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'. Check ticker validity.")

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)
    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)

    _data_cache[cache_key] = df.copy()
    logger.info(f"Fetched {len(df)} rows for {symbol}")
    return df


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators and return an enriched DataFrame.

    Features added (all normalized later via MinMaxScaler):
        Price:       Close, Open, High, Low
        Volume:      Volume, OBV
        Trend:       SMA_20, SMA_50, EMA_20, MACD, MACD_Signal, MACD_Diff
        Momentum:    RSI_14, Stoch_K, Stoch_D
        Volatility:  BB_High, BB_Low, BB_Width, BB_Pct, ATR_14
        Custom:      Daily_Return, Log_Return, HL_Ratio, OC_Ratio
    """
    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    close = out["Close"]
    high  = out["High"]
    low   = out["Low"]
    vol   = out["Volume"]

    # ── Trend ──────────────────────────────────────────────────────────────
    out["SMA_20"]      = SMAIndicator(close, window=20).sma_indicator()
    out["SMA_50"]      = SMAIndicator(close, window=50).sma_indicator()
    out["EMA_20"]      = EMAIndicator(close, window=20).ema_indicator()

    _macd              = MACD(close)
    out["MACD"]        = _macd.macd()
    out["MACD_Signal"] = _macd.macd_signal()
    out["MACD_Diff"]   = _macd.macd_diff()

    # ── Momentum ───────────────────────────────────────────────────────────
    out["RSI_14"]      = RSIIndicator(close, window=14).rsi()
    _stoch             = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    out["Stoch_K"]     = _stoch.stoch()
    out["Stoch_D"]     = _stoch.stoch_signal()

    # ── Volatility ─────────────────────────────────────────────────────────
    _bb                = BollingerBands(close, window=20, window_dev=2)
    out["BB_High"]     = _bb.bollinger_hband()
    out["BB_Low"]      = _bb.bollinger_lband()
    out["BB_Width"]    = _bb.bollinger_wband()
    out["BB_Pct"]      = _bb.bollinger_pband()
    out["ATR_14"]      = AverageTrueRange(high, low, close, window=14).average_true_range()

    # ── Volume ─────────────────────────────────────────────────────────────
    out["OBV"]         = OnBalanceVolumeIndicator(close, vol).on_balance_volume()

    # ── Custom Derived ─────────────────────────────────────────────────────
    out["Daily_Return"]= close.pct_change()
    out["Log_Return"]  = np.log(close / close.shift(1))
    out["HL_Ratio"]    = (high - low) / close          # intraday range normalized
    out["OC_Ratio"]    = (close - out["Open"]) / out["Open"]  # open-close momentum

    # ── Target: next-day close ─────────────────────────────────────────────
    out["Target"]      = close.shift(-1)

    # Drop rows with NaN (from indicator warm-up + last row has no target)
    out.dropna(inplace=True)
    return out


def scale_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler_path: Optional[Path] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    MinMax-scale the selected feature columns to [0, 1].
    Optionally save/load scaler from disk for inference consistency.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))

    if not fit and scaler_path and scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info(f"Loaded scaler from {scaler_path}")
        scaled = scaler.transform(df[feature_cols])
    else:
        scaled = scaler.fit_transform(df[feature_cols])
        if scaler_path:
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            logger.info(f"Saved scaler to {scaler_path}")

    scaled_df = pd.DataFrame(scaled, columns=feature_cols, index=df.index)
    scaled_df["Target"] = df["Target"].values
    return scaled_df, scaler


def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Target",
    seq_len: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a flat DataFrame into (X, y) pairs for LSTM training.

    X shape: (n_samples, seq_len, n_features)
    y shape: (n_samples,)
    """
    data    = df[feature_cols].values
    targets = df[target_col].values
    X, y    = [], []

    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len : i])   # window of past seq_len rows
        y.append(targets[i])              # next-day close (raw, not scaled)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_test_split_temporal(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Time-aware train/test split — NO shuffling to prevent look-ahead bias.
    """
    split = int(len(X) * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]


def get_feature_cols() -> List[str]:
    """Return the ordered list of feature column names used by the model."""
    return [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_20", "SMA_50", "EMA_20",
        "MACD", "MACD_Signal", "MACD_Diff",
        "RSI_14", "Stoch_K", "Stoch_D",
        "BB_High", "BB_Low", "BB_Width", "BB_Pct",
        "ATR_14", "OBV",
        "Daily_Return", "Log_Return", "HL_Ratio", "OC_Ratio",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: Full pipeline in one call
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data_for_training(
    symbol: str,
    period: str = "5y",
    seq_len: int = 60,
    test_ratio: float = 0.2,
    scaler_save_dir: Optional[Path] = None,
) -> Dict:
    """
    End-to-end pipeline: fetch → features → scale → sequences → split.

    Returns a dict with keys:
        X_train, X_test, y_train, y_test,
        scaler, feature_cols, raw_df, scaled_df,
        train_dates, test_dates
    """
    logger.info(f"Running full data pipeline for {symbol}")

    raw_df       = fetch_ohlcv(symbol, period=period)
    feat_df      = build_feature_dataframe(raw_df)
    feature_cols = get_feature_cols()

    scaler_path  = (
        scaler_save_dir / f"{symbol}_scaler.pkl"
        if scaler_save_dir else None
    )
    scaled_df, scaler = scale_features(feat_df, feature_cols, scaler_path=scaler_path, fit=True)

    X, y = build_sequences(scaled_df, feature_cols, seq_len=seq_len)
    X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_ratio)

    # Align dates to sequences
    valid_dates = feat_df.index[seq_len:]
    split       = int(len(valid_dates) * (1 - test_ratio))

    logger.info(
        f"{symbol}: {len(X_train)} train samples, {len(X_test)} test samples, "
        f"{len(feature_cols)} features, seq_len={seq_len}"
    )

    return {
        "X_train":     X_train,
        "X_test":      X_test,
        "y_train":     y_train,
        "y_test":      y_test,
        "scaler":      scaler,
        "feature_cols":feature_cols,
        "raw_df":      raw_df,
        "feat_df":     feat_df,
        "scaled_df":   scaled_df,
        "train_dates": valid_dates[:split],
        "test_dates":  valid_dates[split:],
    }


def prepare_inference_sequence(
    symbol: str,
    seq_len: int = 60,
    scaler_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Fetch the most recent `seq_len` days and return a ready-to-predict tensor.

    Returns:
        X_infer: shape (1, seq_len, n_features)
        recent_df: the last seq_len rows of feature data (for context)
    """
    raw_df       = fetch_ohlcv(symbol, period="1y")
    feat_df      = build_feature_dataframe(raw_df)
    feature_cols = get_feature_cols()

    scaler_path  = (
        scaler_dir / f"{symbol}_scaler.pkl"
        if scaler_dir else None
    )
    _, scaler    = scale_features(feat_df, feature_cols, scaler_path=scaler_path, fit=False)
    scaled_arr   = scaler.transform(feat_df[feature_cols])

    if len(scaled_arr) < seq_len:
        raise ValueError(
            f"Not enough data for inference: need {seq_len} rows, got {len(scaled_arr)}"
        )

    window  = scaled_arr[-seq_len:]
    X_infer = window[np.newaxis, :, :]   # → (1, seq_len, n_features)

    return X_infer.astype(np.float32), feat_df.iloc[-seq_len:]
