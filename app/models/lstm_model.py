"""
NeuroTrade — LSTM Model
Architecture, training loop, walk-forward validation, and evaluation metrics.

Architecture:
    Input (seq_len × n_features)
        → LSTM(128, return_sequences=True)
        → Dropout(0.2)
        → LSTM(64, return_sequences=False)
        → Dropout(0.2)
        → Dense(32, activation='relu')
        → Dense(1)   ← predicted next-day close price
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# TensorFlow / Keras imports — guarded so the module can be imported
# in environments without GPU for linting purposes.
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, optimizers, losses, metrics

    tf.get_logger().setLevel("ERROR")  # suppress TF noise
    _TF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TF_AVAILABLE = False
    logger.warning("TensorFlow not installed — model training unavailable")

from app.core.config import get_settings

settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    seq_len:       int   = 60
    n_features:    int   = 24
    lstm_units_1:  int   = 128
    lstm_units_2:  int   = 64
    dense_units:   int   = 32
    dropout_rate:  float = 0.2
    learning_rate: float = 0.001
    batch_size:    int   = 32
    epochs:        int   = 50
    patience:      int   = 7   # EarlyStopping patience


@dataclass
class EpochMetrics:
    epoch:     int
    train_loss: float
    val_loss:   float
    train_mae:  float
    val_mae:    float


@dataclass
class EvaluationResult:
    symbol:            str
    rmse:              float
    mae:               float
    mape:              float
    r2:                float
    directional_acc:   float
    sharpe_ratio:      float
    epoch_history:     List[EpochMetrics] = field(default_factory=list)
    walk_forward_folds: List[Dict]        = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Round floats for clean JSON
        for k in ("rmse","mae","mape","r2","directional_acc","sharpe_ratio"):
            d[k] = round(d[k], 4)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_lstm_model(cfg: TrainingConfig) -> "keras.Model":
    """
    Build and compile the LSTM model.

    Input shape: (batch, seq_len, n_features)
    Output:      scalar — predicted next-day closing price
    """
    if not _TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed.")

    inp = keras.Input(shape=(cfg.seq_len, cfg.n_features), name="sequence_input")

    # ── LSTM Stack ────────────────────────────────────────────────────────
    x = layers.LSTM(
        cfg.lstm_units_1,
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="lstm_1",
    )(inp)
    x = layers.Dropout(cfg.dropout_rate, name="dropout_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)

    x = layers.LSTM(
        cfg.lstm_units_2,
        return_sequences=False,
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="lstm_2",
    )(x)
    x = layers.Dropout(cfg.dropout_rate, name="dropout_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)

    # ── Dense Head ────────────────────────────────────────────────────────
    x   = layers.Dense(cfg.dense_units, activation="relu", name="dense_1")(x)
    out = layers.Dense(1, name="price_output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="NeuroTrade_LSTM")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=losses.MeanSquaredError(),
        metrics=[metrics.MeanAbsoluteError(name="mae")],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    cfg:     TrainingConfig,
    save_path: Optional[Path] = None,
) -> Tuple["keras.Model", List[EpochMetrics]]:
    """
    Train the LSTM model with EarlyStopping, ReduceLROnPlateau,
    and optional model checkpoint.

    Returns the trained model and per-epoch metrics.
    """
    if not _TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed.")

    model = build_lstm_model(cfg)
    logger.info(f"Model summary:\n{model.summary()}")
    logger.info(
        f"Training: {len(X_train)} samples | "
        f"Validation: {len(X_val)} samples | "
        f"Epochs: {cfg.epochs} | Batch: {cfg.batch_size}"
    )

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cbs.append(
            callbacks.ModelCheckpoint(
                filepath=str(save_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            )
        )

    t0     = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=cbs,
        verbose=1,
        shuffle=False,  # preserve temporal order within batch
    )
    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.1f}s — best val_loss: {min(history.history['val_loss']):.6f}")

    epoch_metrics = [
        EpochMetrics(
            epoch=i + 1,
            train_loss=float(history.history["loss"][i]),
            val_loss=float(history.history["val_loss"][i]),
            train_mae=float(history.history["mae"][i]),
            val_mae=float(history.history["val_mae"][i]),
        )
        for i in range(len(history.history["loss"]))
    ]
    return model, epoch_metrics


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_validation(
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainingConfig,
    n_folds: int = 8,
    min_train_ratio: float = 0.5,
) -> List[Dict]:
    """
    Walk-forward (time-series) cross-validation.

    Each fold:
        - Training set = all data before the test window
        - Test set     = the next contiguous chunk
    No data leakage. Each fold sees strictly future data only in test.

    Returns a list of fold result dicts.
    """
    if not _TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed.")

    n         = len(X)
    fold_size = int(n * (1 - min_train_ratio) / n_folds)
    results   = []

    logger.info(f"Walk-forward validation: {n_folds} folds, fold_size={fold_size}")

    for fold in range(n_folds):
        test_end   = n - fold * fold_size
        test_start = test_end - fold_size
        if test_start < int(n * min_train_ratio):
            break

        X_tr, y_tr = X[:test_start],              y[:test_start]
        X_te, y_te = X[test_start : test_end],    y[test_start : test_end]

        model = build_lstm_model(cfg)
        model.fit(
            X_tr, y_tr,
            validation_split=0.1,
            epochs=30,                # shorter for CV
            batch_size=cfg.batch_size,
            callbacks=[callbacks.EarlyStopping(patience=4, restore_best_weights=True)],
            verbose=0,
            shuffle=False,
        )

        preds = model.predict(X_te, verbose=0).flatten()
        rmse  = float(np.sqrt(np.mean((preds - y_te) ** 2)))
        mae   = float(np.mean(np.abs(preds - y_te)))
        dir_acc = float(
            np.mean(np.sign(np.diff(preds)) == np.sign(np.diff(y_te))) * 100
        )

        fold_result = {
            "fold":            fold + 1,
            "train_size":      len(X_tr),
            "test_size":       len(X_te),
            "rmse":            round(rmse, 4),
            "mae":             round(mae, 4),
            "directional_acc": round(dir_acc, 2),
        }
        results.append(fold_result)
        logger.info(f"Fold {fold+1}: RMSE={rmse:.4f} MAE={mae:.4f} DirAcc={dir_acc:.1f}%")

        # Free memory
        del model
        keras.backend.clear_session()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model:   "keras.Model",
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    symbol:  str,
    epoch_history: Optional[List[EpochMetrics]] = None,
    wf_folds: Optional[List[Dict]] = None,
) -> EvaluationResult:
    """
    Compute RMSE, MAE, MAPE, R², Directional Accuracy, and Sharpe Ratio.
    """
    preds = model.predict(X_test, verbose=0).flatten()

    # Core regression metrics
    rmse  = float(np.sqrt(np.mean((preds - y_test) ** 2)))
    mae   = float(np.mean(np.abs(preds - y_test)))
    mape  = float(np.mean(np.abs((preds - y_test) / np.maximum(y_test, 1e-8))) * 100)

    # R²
    ss_res = np.sum((y_test - preds) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2     = float(1 - ss_res / (ss_tot + 1e-8))

    # Directional accuracy (are we right about up/down?)
    dir_acc = float(
        np.mean(np.sign(np.diff(preds)) == np.sign(np.diff(y_test))) * 100
    )

    # Sharpe-like ratio on predicted returns
    pred_returns = np.diff(preds) / (preds[:-1] + 1e-8)
    sharpe = float(
        pred_returns.mean() / (pred_returns.std() + 1e-8) * np.sqrt(252)
    )

    result = EvaluationResult(
        symbol=symbol,
        rmse=round(rmse, 4),
        mae=round(mae, 4),
        mape=round(mape, 4),
        r2=round(r2, 4),
        directional_acc=round(dir_acc, 2),
        sharpe_ratio=round(sharpe, 4),
        epoch_history=epoch_history or [],
        walk_forward_folds=wf_folds or [],
    )
    logger.info(
        f"Evaluation [{symbol}]: RMSE={rmse:.4f} MAE={mae:.4f} "
        f"MAPE={mape:.2f}% R²={r2:.4f} DirAcc={dir_acc:.1f}% Sharpe={sharpe:.2f}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model: "keras.Model", path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    logger.info(f"Saved model → {path}")


def load_model(path: Path) -> "keras.Model":
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    model = keras.models.load_model(str(path))
    logger.info(f"Loaded model ← {path}")
    return model


def save_eval_result(result: EvaluationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    logger.info(f"Saved eval result → {path}")
