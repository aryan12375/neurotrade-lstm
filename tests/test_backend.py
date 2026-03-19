"""
NeuroTrade — Test Suite
Covers data pipeline, model architecture, and prediction service.

Run:
    pytest tests/ -v
    pytest tests/ -v --tb=short
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with 300 daily bars."""
    n     = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 150.0 + np.cumsum(np.random.randn(n) * 1.5)
    open_ = close - np.random.uniform(0, 2, n)
    high  = close + np.random.uniform(0, 3, n)
    low   = close - np.random.uniform(0, 3, n)
    vol   = np.random.randint(30_000_000, 80_000_000, n).astype(float)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )
    df.index.name = "Date"
    return df


@pytest.fixture
def feature_df(sample_ohlcv):
    from app.services.data_pipeline import build_feature_dataframe
    return build_feature_dataframe(sample_ohlcv)


@pytest.fixture
def seq_data(feature_df):
    from app.services.data_pipeline import (
        build_sequences, get_feature_cols, scale_features
    )
    feature_cols  = get_feature_cols()
    scaled_df, sc = scale_features(feature_df, feature_cols)
    X, y = build_sequences(scaled_df, feature_cols, seq_len=30)
    return X, y, sc


# ─────────────────────────────────────────────────────────────────────────────
# Data Pipeline Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildFeatureDataFrame:
    def test_returns_dataframe(self, sample_ohlcv):
        from app.services.data_pipeline import build_feature_dataframe
        df = build_feature_dataframe(sample_ohlcv)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self, feature_df):
        required = [
            "Close", "RSI_14", "MACD", "SMA_20", "SMA_50",
            "BB_High", "BB_Low", "ATR_14", "Daily_Return", "Target",
        ]
        for col in required:
            assert col in feature_df.columns, f"Missing column: {col}"

    def test_no_nan_in_output(self, feature_df):
        assert not feature_df.isnull().any().any(), "Feature DataFrame contains NaN values"

    def test_target_is_next_day_close(self, feature_df):
        # Target[i] should equal Close[i+1] (approximately, after dropna)
        diff = (feature_df["Target"] - feature_df["Close"].shift(-1)).dropna()
        assert (diff.abs() < 1e-6).all()

    def test_rows_reduced_from_warmup(self, sample_ohlcv, feature_df):
        # Indicators with 50-day window will trim ≥50 rows
        assert len(feature_df) < len(sample_ohlcv)


class TestScaleFeatures:
    def test_scaled_range(self, feature_df):
        from app.services.data_pipeline import get_feature_cols, scale_features
        feature_cols   = get_feature_cols()
        scaled_df, sc  = scale_features(feature_df, feature_cols)
        data = scaled_df[feature_cols].values
        assert data.min() >= -0.01   # allow tiny float error
        assert data.max() <= 1.01

    def test_scaler_invertible(self, feature_df):
        from app.services.data_pipeline import get_feature_cols, scale_features
        feature_cols  = get_feature_cols()
        scaled_df, sc = scale_features(feature_df, feature_cols)
        inv = sc.inverse_transform(scaled_df[feature_cols].values)
        orig = feature_df[feature_cols].values
        np.testing.assert_allclose(inv, orig, rtol=1e-4)


class TestBuildSequences:
    def test_shape(self, seq_data):
        X, y, _ = seq_data
        assert X.ndim == 3
        assert y.ndim == 1
        assert X.shape[1] == 30   # seq_len
        assert X.shape[0] == y.shape[0]

    def test_dtype(self, seq_data):
        X, y, _ = seq_data
        assert X.dtype == np.float32
        assert y.dtype == np.float32

    def test_no_nan(self, seq_data):
        X, y, _ = seq_data
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()


class TestTemporalSplit:
    def test_sizes(self, seq_data):
        from app.services.data_pipeline import train_test_split_temporal
        X, y, _ = seq_data
        Xtr, Xte, ytr, yte = train_test_split_temporal(X, y, test_ratio=0.2)
        total = len(X)
        assert len(Xtr) + len(Xte) == total
        assert abs(len(Xte) / total - 0.2) < 0.02   # within 2% of target

    def test_no_overlap(self, seq_data):
        from app.services.data_pipeline import train_test_split_temporal
        X, y, _ = seq_data
        Xtr, Xte, _, _ = train_test_split_temporal(X, y)
        split = len(Xtr)
        # First test row != last train row
        assert not np.array_equal(Xtr[-1], Xte[0])


# ─────────────────────────────────────────────────────────────────────────────
# Model Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLSTMModel:
    @pytest.fixture
    def small_cfg(self):
        from app.models.lstm_model import TrainingConfig
        return TrainingConfig(
            seq_len=30, n_features=24,
            lstm_units_1=16, lstm_units_2=8,
            dense_units=8, dropout_rate=0.1,
            epochs=2, batch_size=16,
        )

    def test_build(self, small_cfg):
        from app.models.lstm_model import build_lstm_model
        model = build_lstm_model(small_cfg)
        assert model is not None
        assert model.input_shape == (None, 30, 24)
        assert model.output_shape == (None, 1)

    def test_forward_pass(self, small_cfg):
        from app.models.lstm_model import build_lstm_model
        model = build_lstm_model(small_cfg)
        X = np.random.rand(4, 30, 24).astype(np.float32)
        out = model.predict(X, verbose=0)
        assert out.shape == (4, 1)
        assert not np.isnan(out).any()

    def test_train_reduces_loss(self, small_cfg, seq_data):
        from app.models.lstm_model import train_model
        X, y, _ = seq_data
        Xtr, Xte = X[:80], X[80:100]
        ytr, yte = y[:80], y[80:100]
        model, history = train_model(Xtr, ytr, Xte, yte, small_cfg)
        first_loss = history[0].train_loss
        last_loss  = history[-1].train_loss
        assert last_loss < first_loss, "Loss did not decrease during training"

    def test_evaluate_returns_metrics(self, small_cfg, seq_data):
        from app.models.lstm_model import build_lstm_model, evaluate_model
        X, y, _ = seq_data
        model = build_lstm_model(small_cfg)
        result = evaluate_model(model, X[:50], y[:50], symbol="TEST")
        assert result.rmse >= 0
        assert result.mae >= 0
        assert 0 <= result.directional_acc <= 100
        assert result.r2 <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Signal derivation
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalDerivation:
    def test_strong_buy(self):
        from app.services.prediction_service import _derive_signal
        assert _derive_signal(4.0, 85) == "STRONG BUY"

    def test_buy(self):
        from app.services.prediction_service import _derive_signal
        assert _derive_signal(1.5, 80) == "BUY"

    def test_hold(self):
        from app.services.prediction_service import _derive_signal
        assert _derive_signal(0.2, 70) == "HOLD"

    def test_sell(self):
        from app.services.prediction_service import _derive_signal
        assert _derive_signal(-1.5, 80) == "SELL"

    def test_strong_sell(self):
        from app.services.prediction_service import _derive_signal
        assert _derive_signal(-4.0, 85) == "STRONG SELL"

    def test_uncertain_low_confidence(self):
        from app.services.prediction_service import _derive_signal
        assert _derive_signal(5.0, 40) == "UNCERTAIN"


# ─────────────────────────────────────────────────────────────────────────────
# Reddit Service (simulated path)
# ─────────────────────────────────────────────────────────────────────────────

class TestRedditService:
    def test_simulated_returns_all_symbols(self):
        from app.services.reddit_service import RedditHypeService
        svc    = RedditHypeService()   # no credentials → simulated
        result = svc.get_mentions(["AAPL", "NVDA", "TSLA"])
        syms   = {r["symbol"] for r in result}
        assert syms == {"AAPL", "NVDA", "TSLA"}

    def test_mentions_positive(self):
        from app.services.reddit_service import RedditHypeService
        svc    = RedditHypeService()
        result = svc.get_mentions(["MSFT"])
        assert result[0]["mentions"] >= 0

    def test_hype_flag_values(self):
        from app.services.reddit_service import RedditHypeService
        svc    = RedditHypeService()
        result = svc.get_mentions(["AAPL","NVDA","TSLA","MSFT","AMZN"])
        for r in result:
            assert r["hype_flag"] in {"low","medium","high"}
