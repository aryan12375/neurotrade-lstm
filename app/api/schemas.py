"""
NeuroTrade — Pydantic Schemas
Request and response models for all FastAPI endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Shared / base
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str = "1.0.0"
    ts:      str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ─────────────────────────────────────────────────────────────────────────────
# Historical data
# ─────────────────────────────────────────────────────────────────────────────

class OHLCVBar(BaseModel):
    date:   str
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float


class HistoricalDataResponse(BaseModel):
    symbol:     str
    period:     str
    interval:   str
    bars:       List[OHLCVBar]
    total_rows: int


# ─────────────────────────────────────────────────────────────────────────────
# Technical indicators
# ─────────────────────────────────────────────────────────────────────────────

class IndicatorPoint(BaseModel):
    date:  str
    value: Optional[float]


class TechnicalIndicatorsResponse(BaseModel):
    symbol:    str
    sma_20:    List[IndicatorPoint]
    sma_50:    List[IndicatorPoint]
    ema_20:    List[IndicatorPoint]
    rsi_14:    List[IndicatorPoint]
    macd:      List[IndicatorPoint]
    macd_signal: List[IndicatorPoint]
    bb_high:   List[IndicatorPoint]
    bb_low:    List[IndicatorPoint]
    bb_pct:    List[IndicatorPoint]
    atr_14:    List[IndicatorPoint]


# ─────────────────────────────────────────────────────────────────────────────
# Predictions
# ─────────────────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    symbol:               str
    current_price:        float
    predicted_price:      float
    predicted_change:     float
    predicted_change_pct: float
    confidence:           float = Field(..., ge=0, le=100)
    signal:               str
    timestamp:            str


class ForecastPoint(BaseModel):
    step:            int
    date:            str
    predicted_price: float
    lower_bound:     float
    upper_bound:     float


class ForecastResponse(BaseModel):
    symbol:    str
    forecast:  List[ForecastPoint]
    generated: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class WatchlistItem(BaseModel):
    symbol:               str
    current_price:        Optional[float]
    predicted_price:      Optional[float]
    predicted_change:     Optional[float]
    predicted_change_pct: Optional[float]
    confidence:           Optional[float]
    signal:               str
    timestamp:            str


class WatchlistResponse(BaseModel):
    watchlist: List[WatchlistItem]
    count:     int


# ─────────────────────────────────────────────────────────────────────────────
# Model evaluation
# ─────────────────────────────────────────────────────────────────────────────

class EpochMetricItem(BaseModel):
    epoch:      int
    train_loss: float
    val_loss:   float
    train_mae:  float
    val_mae:    float


class WalkForwardFold(BaseModel):
    fold:            int
    train_size:      int
    test_size:       int
    rmse:            float
    mae:             float
    directional_acc: float


class EvaluationResponse(BaseModel):
    symbol:             str
    rmse:               float
    mae:                float
    mape:               float
    r2:                 float
    directional_acc:    float
    sharpe_ratio:       float
    epoch_history:      List[EpochMetricItem]
    walk_forward_folds: List[WalkForwardFold]


# ─────────────────────────────────────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────────────────────────────────────

class SHAPFeature(BaseModel):
    feature:    str
    shap_value: float
    direction:  str
    rank:       int


class SHAPResponse(BaseModel):
    symbol:   str
    features: List[SHAPFeature]
    note:     str = "SHAP values represent mean attribution across last 60-day sequence"


class CorrelationItem(BaseModel):
    feature:     str
    correlation: float


class CorrelationResponse(BaseModel):
    symbol:       str
    correlations: List[CorrelationItem]


# ─────────────────────────────────────────────────────────────────────────────
# Reddit / Hype
# ─────────────────────────────────────────────────────────────────────────────

class HypeMention(BaseModel):
    symbol:          str
    mentions:        int
    sentiment_score: float
    sentiment_label: str
    top_posts:       List[str]
    hype_flag:       str
    timestamp:       str


class HypeResponse(BaseModel):
    data:         List[HypeMention]
    hours_window: int


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    symbol:        str
    period:        str = "5y"
    epochs:        int = Field(default=50, ge=5, le=300)
    batch_size:    int = Field(default=32, ge=8, le=256)
    seq_len:       int = Field(default=60, ge=20, le=120)
    walk_forward:  bool = True

    @field_validator("symbol")
    @classmethod
    def upper_symbol(cls, v: str) -> str:
        return v.strip().upper()


class TrainResponse(BaseModel):
    symbol:    str
    status:    str
    message:   str
    eval:      Optional[EvaluationResponse] = None
    duration_s: Optional[float] = None
