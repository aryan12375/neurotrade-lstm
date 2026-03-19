#!/usr/bin/env python3
"""
NeuroTrade — Standalone Training Script
========================================
Trains an LSTM model for one or more stock symbols and saves:
  - saved_models/{SYMBOL}_model.keras
  - saved_models/{SYMBOL}_scaler.pkl
  - saved_models/{SYMBOL}_eval.json

Usage examples:
  # Train single symbol with defaults
  python scripts/train.py --symbol AAPL

  # Train multiple symbols
  python scripts/train.py --symbols AAPL NVDA TSLA MSFT AMZN

  # Custom hyperparameters
  python scripts/train.py --symbol NVDA --epochs 75 --seq-len 60 --period 5y

  # Train full watchlist (from .env)
  python scripts/train.py --all
"""

import argparse
import sys
import time
from pathlib import Path

# ── Make sure project root is on sys.path ──────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.models.lstm_model import (
    EvaluationResult,
    TrainingConfig,
    evaluate_model,
    save_eval_result,
    save_model,
    train_model,
    walk_forward_validation,
)
from app.services.data_pipeline import prepare_data_for_training

settings = get_settings()


def train_symbol(
    symbol: str,
    period: str = "5y",
    epochs: int = 50,
    batch_size: int = 32,
    seq_len: int = 60,
    walk_forward: bool = True,
    n_folds: int = 8,
) -> EvaluationResult:
    """Train, evaluate, and persist a model for a single symbol."""
    t0 = time.time()
    logger.info(f"{'='*50}")
    logger.info(f"  Training: {symbol}")
    logger.info(f"  period={period} | epochs={epochs} | seq_len={seq_len}")
    logger.info(f"{'='*50}")

    # 1. Data pipeline
    pipeline = prepare_data_for_training(
        symbol,
        period=period,
        seq_len=seq_len,
        scaler_save_dir=settings.model_dir,
    )

    # 2. Build training config
    cfg = TrainingConfig(
        seq_len=seq_len,
        n_features=len(pipeline["feature_cols"]),
        lstm_units_1=settings.lstm_units_1,
        lstm_units_2=settings.lstm_units_2,
        dropout_rate=settings.dropout_rate,
        learning_rate=settings.learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=7,
    )

    # 3. Train
    model_save_path = settings.model_dir / f"{symbol}_model.keras"
    model, epoch_history = train_model(
        pipeline["X_train"], pipeline["y_train"],
        pipeline["X_test"],  pipeline["y_test"],
        cfg,
        save_path=model_save_path,
    )

    # 4. Evaluate
    wf_folds = []
    if walk_forward:
        logger.info(f"Running walk-forward validation ({n_folds} folds) …")
        wf_folds = walk_forward_validation(
            pipeline["X_train"], pipeline["y_train"],
            cfg, n_folds=n_folds,
        )

    eval_result = evaluate_model(
        model,
        pipeline["X_test"], pipeline["y_test"],
        symbol=symbol,
        epoch_history=epoch_history,
        wf_folds=wf_folds,
    )

    # 5. Save
    save_model(model, model_save_path)
    save_eval_result(eval_result, settings.model_dir / f"{symbol}_eval.json")

    elapsed = time.time() - t0
    logger.success(
        f"✓ {symbol} | RMSE={eval_result.rmse} | R²={eval_result.r2} "
        f"| DirAcc={eval_result.directional_acc}% | {elapsed:.1f}s"
    )
    return eval_result


def main() -> None:
    setup_logging("INFO")

    parser = argparse.ArgumentParser(
        description="Train NeuroTrade LSTM model(s)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol",       type=str,            help="Single symbol to train")
    parser.add_argument("--symbols",      type=str, nargs="+", help="Multiple symbols to train")
    parser.add_argument("--all",          action="store_true", help="Train full watchlist from .env")
    parser.add_argument("--period",       type=str, default="5y")
    parser.add_argument("--epochs",       type=int, default=50)
    parser.add_argument("--batch-size",   type=int, default=32)
    parser.add_argument("--seq-len",      type=int, default=60)
    parser.add_argument("--no-wf",        action="store_true", help="Skip walk-forward validation")
    parser.add_argument("--folds",        type=int, default=8)
    args = parser.parse_args()

    if args.all:
        symbols = settings.watchlist
    elif args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        parser.error("Specify --symbol AAPL, --symbols AAPL NVDA, or --all")
        return

    logger.info(f"Training {len(symbols)} symbol(s): {symbols}")
    results = {}

    for sym in symbols:
        try:
            result = train_symbol(
                sym,
                period=args.period,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                walk_forward=not args.no_wf,
                n_folds=args.folds,
            )
            results[sym] = {"status": "ok", "rmse": result.rmse, "r2": result.r2}
        except Exception as exc:
            logger.error(f"✗ {sym} failed: {exc}")
            results[sym] = {"status": "error", "detail": str(exc)}

    # Summary table
    logger.info("\n" + "="*55)
    logger.info("  TRAINING SUMMARY")
    logger.info("="*55)
    for sym, res in results.items():
        if res["status"] == "ok":
            logger.success(f"  ✓ {sym:6s}  RMSE={res['rmse']:<8}  R²={res['r2']}")
        else:
            logger.error(f"  ✗ {sym:6s}  {res['detail']}")
    logger.info("="*55)


if __name__ == "__main__":
    main()
