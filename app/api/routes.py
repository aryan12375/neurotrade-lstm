"""
NeuroTrade — API Routes
All endpoints grouped by domain router.

Base path: /api/v1

Routes:
  GET  /health
  GET  /stocks/{symbol}/history
  GET  /stocks/{symbol}/indicators
  GET  /stocks/{symbol}/predict
  GET  /stocks/{symbol}/forecast
  GET  /stocks/{symbol}/shap
  GET  /stocks/{symbol}/correlations
  GET  /stocks/{symbol}/eval
  GET  /watchlist
  GET  /hype
  POST /train
  GET  /models/status
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from loguru import logger

from app.api.schemas import (
    CorrelationResponse,
    EvaluationResponse,
    ForecastResponse,
    HealthResponse,
    HistoricalDataResponse,
    HypeResponse,
    OHLCVBar,
    PredictionResponse,
    SHAPResponse,
    TechnicalIndicatorsResponse,
    TrainRequest,
    TrainResponse,
    WatchlistResponse,
)
from app.core.config import get_settings
from app.services.data_pipeline import (
    build_feature_dataframe,
    fetch_ohlcv,
    get_feature_cols,
    prepare_data_for_training,
    prepare_inference_sequence,
)
from app.services.prediction_service import (
    forecast_n_days,
    get_watchlist_summary,
    is_model_trained,
    predict_next_day,
)
from app.services.reddit_service import RedditHypeService

settings    = get_settings()
router      = APIRouter(prefix="/api/v1", tags=["NeuroTrade"])
_hype_svc   = RedditHypeService()


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


# ─────────────────────────────────────────────────────────────────────────────
# Historical OHLCV
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/stocks/{symbol}/history", response_model=HistoricalDataResponse)
async def get_history(
    symbol:   str,
    period:   str = Query("1y",  description="yfinance period string, e.g. 1y, 2y, 5y"),
    interval: str = Query("1d",  description="yfinance interval string, e.g. 1d, 1h"),
) -> HistoricalDataResponse:
    """
    Return raw OHLCV bars for a symbol.
    Powers the main price chart and volume chart on the dashboard.
    """
    symbol = symbol.upper()
    try:
        df = fetch_ohlcv(symbol, period=period, interval=interval)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    bars = [
        OHLCVBar(
            date=str(idx.date()),
            open=round(float(row["Open"]), 2),
            high=round(float(row["High"]), 2),
            low=round(float(row["Low"]), 2),
            close=round(float(row["Close"]), 2),
            volume=round(float(row["Volume"]), 0),
        )
        for idx, row in df.iterrows()
    ]
    return HistoricalDataResponse(
        symbol=symbol,
        period=period,
        interval=interval,
        bars=bars,
        total_rows=len(bars),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Technical Indicators
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/stocks/{symbol}/indicators", response_model=TechnicalIndicatorsResponse)
async def get_indicators(
    symbol: str,
    period: str = Query("1y"),
) -> TechnicalIndicatorsResponse:
    """
    Compute and return technical indicators for the Technical tab on the dashboard.
    """
    symbol = symbol.upper()
    try:
        raw_df  = fetch_ohlcv(symbol, period=period)
        feat_df = build_feature_dataframe(raw_df)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    def _series(col: str):
        return [
            {"date": str(idx.date()), "value": round(float(v), 4) if v == v else None}
            for idx, v in feat_df[col].items()
        ]

    return TechnicalIndicatorsResponse(
        symbol=symbol,
        sma_20=_series("SMA_20"),
        sma_50=_series("SMA_50"),
        ema_20=_series("EMA_20"),
        rsi_14=_series("RSI_14"),
        macd=_series("MACD"),
        macd_signal=_series("MACD_Signal"),
        bb_high=_series("BB_High"),
        bb_low=_series("BB_Low"),
        bb_pct=_series("BB_Pct"),
        atr_14=_series("ATR_14"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/stocks/{symbol}/predict", response_model=PredictionResponse)
async def get_prediction(symbol: str) -> PredictionResponse:
    """
    Return the LSTM next-day price prediction for a symbol.
    Requires the model to have been trained via POST /train first.
    """
    symbol = symbol.upper()
    if not is_model_trained(symbol):
        raise HTTPException(
            status_code=425,
            detail=f"Model for {symbol} not trained yet. POST /api/v1/train with {{\"symbol\":\"{symbol}\"}}",
        )
    try:
        result = predict_next_day(symbol)
    except Exception as exc:
        logger.exception(f"Prediction error for {symbol}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return PredictionResponse(**result)


# ─────────────────────────────────────────────────────────────────────────────
# Forecast
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/stocks/{symbol}/forecast", response_model=ForecastResponse)
async def get_forecast(
    symbol: str,
    days:   int = Query(10, ge=1, le=30),
) -> ForecastResponse:
    """
    Multi-step iterative forecast for the next `days` trading days.
    Each prediction is fed back as input for the next step.
    """
    symbol = symbol.upper()
    if not is_model_trained(symbol):
        raise HTTPException(
            status_code=425,
            detail=f"Model for {symbol} not trained yet.",
        )
    try:
        points = forecast_n_days(symbol, n=days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ForecastResponse(symbol=symbol, forecast=points)


# ─────────────────────────────────────────────────────────────────────────────
# Watchlist
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/watchlist", response_model=WatchlistResponse)
async def get_watchlist(
    symbols: Optional[str] = Query(
        None,
        description="Comma-separated list of symbols. Defaults to env WATCHLIST.",
    )
) -> WatchlistResponse:
    sym_list = (
        [s.strip().upper() for s in symbols.split(",")]
        if symbols else settings.watchlist
    )
    items = get_watchlist_summary(sym_list)
    return WatchlistResponse(watchlist=items, count=len(items))


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Explainability
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/stocks/{symbol}/shap", response_model=SHAPResponse)
async def get_shap(symbol: str) -> SHAPResponse:
    """
    Return SHAP feature attributions for the most recent prediction.
    Requires TensorFlow + shap installed.
    """
    symbol = symbol.upper()
    if not is_model_trained(symbol):
        raise HTTPException(status_code=425, detail=f"Model for {symbol} not trained yet.")

    try:
        from app.services.shap_service import SHAPExplainer
        from app.services.prediction_service import (
            load_model_for_symbol,
            load_scaler_for_symbol,
        )

        model    = load_model_for_symbol(symbol)
        pipeline = prepare_data_for_training(
            symbol, scaler_save_dir=settings.model_dir
        )
        X_train  = pipeline["X_train"]
        X_test   = pipeline["X_test"]

        explainer  = SHAPExplainer(model, X_train)
        shap_vals  = explainer.explain(X_test[-10:])   # last 10 test windows
        features   = explainer.feature_report(shap_vals, get_feature_cols())
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=f"SHAP not available: {exc}")
    except Exception as exc:
        logger.exception(f"SHAP error for {symbol}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return SHAPResponse(symbol=symbol, features=features)


@router.get("/stocks/{symbol}/correlations", response_model=CorrelationResponse)
async def get_correlations(symbol: str) -> CorrelationResponse:
    """
    Return feature-to-target Pearson correlations for the heatmap.
    """
    symbol = symbol.upper()
    try:
        raw_df      = fetch_ohlcv(symbol, period="2y")
        feat_df     = build_feature_dataframe(raw_df)
        feature_cols = get_feature_cols()

        corr_list = []
        for col in feature_cols:
            corr = feat_df[col].corr(feat_df["Target"])
            corr_list.append({
                "feature":     col,
                "correlation": round(float(corr), 3) if corr == corr else 0.0,
            })
        corr_list.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return CorrelationResponse(symbol=symbol, correlations=corr_list)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/stocks/{symbol}/eval", response_model=EvaluationResponse)
async def get_evaluation(symbol: str) -> EvaluationResponse:
    """
    Return saved evaluation metrics for a trained model.
    """
    symbol    = symbol.upper()
    eval_path = settings.model_dir / f"{symbol}_eval.json"
    if not eval_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation found for {symbol}. Train the model first.",
        )
    with open(eval_path) as f:
        data = json.load(f)
    return EvaluationResponse(**data)


# ─────────────────────────────────────────────────────────────────────────────
# Model Status
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/models/status")
async def get_model_status():
    """List which symbols have trained models available."""
    return {
        sym: {
            "trained":    is_model_trained(sym),
            "model_path": str(settings.model_dir / f"{sym}_model.keras"),
            "eval_ready": (settings.model_dir / f"{sym}_eval.json").exists(),
        }
        for sym in settings.watchlist
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reddit Hype
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/hype", response_model=HypeResponse)
async def get_hype(
    symbols: Optional[str] = Query(None),
    hours:   int = Query(24, ge=1, le=72),
) -> HypeResponse:
    """
    Return Reddit mention counts + sentiment for the Hype vs Reality gauge.
    """
    sym_list = (
        [s.strip().upper() for s in symbols.split(",")]
        if symbols else settings.watchlist
    )
    data = _hype_svc.get_mentions(sym_list, hours=hours)
    return HypeResponse(data=data, hours_window=hours)


# ─────────────────────────────────────────────────────────────────────────────
# Training  (runs synchronously — wrap in BackgroundTasks for production)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/train", response_model=TrainResponse)
async def train_model_endpoint(req: TrainRequest) -> TrainResponse:
    """
    Train or retrain the LSTM model for a given symbol.

    This endpoint runs synchronously and may take several minutes.
    For production, move the training call to a Celery/ARQ background task.
    """
    t0     = time.time()
    symbol = req.symbol

    logger.info(f"Training requested: {symbol} | epochs={req.epochs}")

    try:
        from app.models.lstm_model import (
            TrainingConfig,
            train_model,
            evaluate_model,
            walk_forward_validation,
            save_model,
            save_eval_result,
            EvaluationResult,
        )

        pipeline = prepare_data_for_training(
            symbol,
            period=req.period,
            seq_len=req.seq_len,
            scaler_save_dir=settings.model_dir,
        )
        cfg = TrainingConfig(
            seq_len=req.seq_len,
            n_features=len(pipeline["feature_cols"]),
            epochs=req.epochs,
            batch_size=req.batch_size,
            lstm_units_1=settings.lstm_units_1,
            lstm_units_2=settings.lstm_units_2,
            dropout_rate=settings.dropout_rate,
            learning_rate=settings.learning_rate,
        )

        # Train
        model, epoch_history = train_model(
            pipeline["X_train"], pipeline["y_train"],
            pipeline["X_test"],  pipeline["y_test"],
            cfg,
            save_path=settings.model_dir / f"{symbol}_model.keras",
        )

        # Evaluate
        eval_result = evaluate_model(
            model,
            pipeline["X_test"],
            pipeline["y_test"],
            symbol=symbol,
            epoch_history=epoch_history,
        )

        # Optional walk-forward validation
        if req.walk_forward:
            X_all = pipeline["X_train"]
            y_all = pipeline["y_train"]
            folds = walk_forward_validation(X_all, y_all, cfg, n_folds=8)
            eval_result.walk_forward_folds = folds

        # Persist
        save_model(model, settings.model_dir / f"{symbol}_model.keras")
        save_eval_result(eval_result, settings.model_dir / f"{symbol}_eval.json")

        # Evict prediction cache
        from app.services.prediction_service import _pred_cache, _model_registry
        _pred_cache.clear()
        _model_registry.pop(symbol, None)

        elapsed = round(time.time() - t0, 1)
        logger.info(f"Training complete for {symbol} in {elapsed}s")

        eval_dict = eval_result.to_dict()
        return TrainResponse(
            symbol=symbol,
            status="success",
            message=f"Model trained in {elapsed}s. RMSE={eval_result.rmse}, R²={eval_result.r2}",
            eval=EvaluationResponse(**eval_dict),
            duration_s=elapsed,
        )

    except Exception as exc:
        logger.exception(f"Training failed for {symbol}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
