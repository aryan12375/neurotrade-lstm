"""
NeuroTrade — Prediction Service
Orchestrates model loading, inference, confidence scoring,
multi-day forecasting, and SHAP explanation retrieval.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from cachetools import TTLCache
from loguru import logger

from app.core.config import get_settings
from app.services.data_pipeline import (
    fetch_ohlcv,
    build_feature_dataframe,
    get_feature_cols,
    prepare_inference_sequence,
)

settings = get_settings()

# ── Prediction cache: 5 minutes ──────────────────────────────────────────────
_pred_cache: TTLCache = TTLCache(maxsize=30, ttl=300)

# ── In-process model registry ────────────────────────────────────────────────
_model_registry:  Dict[str, object]  = {}   # symbol → keras.Model
_scaler_registry: Dict[str, object]  = {}   # symbol → MinMaxScaler


# ─────────────────────────────────────────────────────────────────────────────
# Registry helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model_for_symbol(symbol: str):
    """Load (and cache in-process) the Keras model for a given symbol."""
    if symbol in _model_registry:
        return _model_registry[symbol]

    from tensorflow import keras
    model_path = settings.model_dir / f"{symbol}_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found for {symbol} at {model_path}. "
            f"Run: python scripts/train.py --symbol {symbol}"
        )
    model = keras.models.load_model(str(model_path))
    _model_registry[symbol] = model
    logger.info(f"Loaded model for {symbol}")
    return model


def load_scaler_for_symbol(symbol: str):
    """Load (and cache in-process) the MinMaxScaler for a given symbol."""
    if symbol in _scaler_registry:
        return _scaler_registry[symbol]

    scaler_path = settings.model_dir / f"{symbol}_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"No scaler found for {symbol} at {scaler_path}. "
            f"Run training first."
        )
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    _scaler_registry[symbol] = scaler
    logger.info(f"Loaded scaler for {symbol}")
    return scaler


def is_model_trained(symbol: str) -> bool:
    model_path  = settings.model_dir / f"{symbol}_model.keras"
    scaler_path = settings.model_dir / f"{symbol}_scaler.pkl"
    return model_path.exists() and scaler_path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Core prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_next_day(symbol: str) -> Dict:
    """
    Predict the next trading day's closing price for `symbol`.

    Returns
    -------
    dict with keys:
        symbol, current_price, predicted_price, predicted_change,
        predicted_change_pct, confidence, signal, timestamp
    """
    cache_key = f"predict_{symbol}"
    if cache_key in _pred_cache:
        return _pred_cache[cache_key]

    model  = load_model_for_symbol(symbol)
    scaler = load_scaler_for_symbol(symbol)

    X_infer, recent_df = prepare_inference_sequence(
        symbol,
        seq_len=settings.sequence_length,
        scaler_dir=settings.model_dir,
    )

    raw_pred = float(model.predict(X_infer, verbose=0)[0, 0])

    # The scaler was fit on all features; to inverse-transform just the
    # Close price we reconstruct a dummy full-feature row.
    feature_cols  = get_feature_cols()
    close_idx     = feature_cols.index("Close")
    dummy         = np.zeros((1, len(feature_cols)))
    dummy[0, close_idx] = raw_pred
    full_inv      = scaler.inverse_transform(dummy)
    predicted_price = float(full_inv[0, close_idx])

    current_price = float(recent_df["Close"].iloc[-1])
    delta         = predicted_price - current_price
    delta_pct     = delta / current_price * 100

    # Confidence: inverse of normalized prediction uncertainty
    # (proxy — real confidence would require Monte Carlo Dropout)
    confidence = _estimate_confidence(model, X_infer, n_passes=20)

    signal = _derive_signal(delta_pct, confidence)

    result = {
        "symbol":               symbol,
        "current_price":        round(current_price, 2),
        "predicted_price":      round(predicted_price, 2),
        "predicted_change":     round(delta, 2),
        "predicted_change_pct": round(delta_pct, 3),
        "confidence":           round(confidence, 1),
        "signal":               signal,
        "timestamp":            datetime.utcnow().isoformat(),
    }
    _pred_cache[cache_key] = result
    return result


def forecast_n_days(symbol: str, n: int = 10) -> List[Dict]:
    """
    Iterative multi-step forecast: feed each prediction back as input
    to generate an n-day price trajectory.

    Returns a list of {date, predicted_price, lower_bound, upper_bound}.
    """
    cache_key = f"forecast_{symbol}_{n}"
    if cache_key in _pred_cache:
        return _pred_cache[cache_key]

    model  = load_model_for_symbol(symbol)
    scaler = load_scaler_for_symbol(symbol)

    X_infer, recent_df = prepare_inference_sequence(
        symbol,
        seq_len=settings.sequence_length,
        scaler_dir=settings.model_dir,
    )

    feature_cols = get_feature_cols()
    close_idx    = feature_cols.index("Close")
    seq          = X_infer[0].copy()   # (seq_len, n_features)

    forecasts     = []
    current_price = float(recent_df["Close"].iloc[-1])
    today         = datetime.utcnow().date()

    for step in range(1, n + 1):
        inp   = seq[np.newaxis, :, :]   # (1, seq_len, n_features)
        raw   = float(model.predict(inp, verbose=0)[0, 0])

        # Inverse transform
        dummy               = np.zeros((1, len(feature_cols)))
        dummy[0, close_idx] = raw
        pred_price          = float(scaler.inverse_transform(dummy)[0, close_idx])

        # Uncertainty widens with horizon (simple linear expansion)
        uncertainty = current_price * 0.008 * step

        forecasts.append({
            "step":          step,
            "date":          str(today + timedelta(days=step)),
            "predicted_price": round(pred_price, 2),
            "lower_bound":   round(pred_price - uncertainty, 2),
            "upper_bound":   round(pred_price + uncertainty, 2),
        })

        # Roll the window: append new Close to sequence
        new_row            = seq[-1].copy()
        new_row[close_idx] = raw
        seq                = np.vstack([seq[1:], new_row])

    _pred_cache[cache_key] = forecasts
    return forecasts


def get_watchlist_summary(symbols: List[str]) -> List[Dict]:
    """
    Run predict_next_day for every symbol in the watchlist and
    return a combined summary list — for the watchlist table endpoint.
    """
    results = []
    for sym in symbols:
        try:
            pred = predict_next_day(sym)
            results.append(pred)
        except FileNotFoundError:
            # Model not trained yet — return live price only
            try:
                df     = fetch_ohlcv(sym, period="5d")
                price  = float(df["Close"].iloc[-1])
                prev   = float(df["Close"].iloc[-2]) if len(df) > 1 else price
                results.append({
                    "symbol":               sym,
                    "current_price":        round(price, 2),
                    "predicted_price":      None,
                    "predicted_change":     None,
                    "predicted_change_pct": round((price - prev) / prev * 100, 3),
                    "confidence":           None,
                    "signal":               "UNTRAINED",
                    "timestamp":            datetime.utcnow().isoformat(),
                })
            except Exception as exc:
                logger.warning(f"Could not fetch {sym}: {exc}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_confidence(model, X: np.ndarray, n_passes: int = 20) -> float:
    """
    Monte Carlo Dropout: run n stochastic forward passes (Dropout active)
    and measure predictive variance. Lower variance → higher confidence.
    """
    try:
        import tensorflow as tf
        preds = np.array([
            model(X, training=True).numpy()[0, 0]
            for _ in range(n_passes)
        ])
        std  = float(preds.std())
        mean = float(abs(preds.mean())) + 1e-8
        cv   = std / mean   # coefficient of variation
        conf = max(0.0, min(100.0, (1 - cv) * 100))
        return round(conf, 1)
    except Exception:
        return 75.0   # fallback


def _derive_signal(delta_pct: float, confidence: float) -> str:
    """Map predicted change and confidence to a trading signal."""
    if confidence < 55:
        return "UNCERTAIN"
    if delta_pct > 3.0:
        return "STRONG BUY"
    if delta_pct > 0.8:
        return "BUY"
    if delta_pct < -3.0:
        return "STRONG SELL"
    if delta_pct < -0.8:
        return "SELL"
    return "HOLD"
