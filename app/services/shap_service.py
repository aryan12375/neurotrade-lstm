"""
NeuroTrade — SHAP Explainability Service
Wraps the trained LSTM model in a SHAP DeepExplainer to generate
per-feature attribution values (SHAP values).

Usage:
    explainer = SHAPExplainer(model, background_data)
    shap_vals = explainer.explain(X_instance)
    report    = explainer.feature_report(shap_vals, feature_cols)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning("shap not installed — XAI features disabled")

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


class SHAPExplainer:
    """
    Wraps a Keras LSTM model for SHAP-based feature attribution.

    Parameters
    ----------
    model : keras.Model
        The trained LSTM model.
    background : np.ndarray
        Background dataset for DeepExplainer, shape (n, seq_len, n_features).
        Typically a random sample of 100–200 training sequences.
    """

    def __init__(self, model, background: np.ndarray) -> None:
        if not _SHAP_AVAILABLE:
            raise RuntimeError("Install shap: pip install shap")
        if not _TF_AVAILABLE:
            raise RuntimeError("TensorFlow required for SHAP DeepExplainer")

        # Sample background down to ≤100 rows for performance
        if len(background) > 100:
            idx        = np.random.choice(len(background), 100, replace=False)
            background = background[idx]

        logger.info("Initialising SHAP DeepExplainer …")
        self._model      = model
        self._background = background
        self._explainer  = shap.DeepExplainer(model, background)
        logger.info("SHAP DeepExplainer ready")

    # ------------------------------------------------------------------
    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for X.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_samples, seq_len, n_features)

        Returns
        -------
        shap_values : np.ndarray
            Shape (n_samples, seq_len, n_features)
            Positive = feature pushed prediction UP
            Negative = feature pushed prediction DOWN
        """
        raw = self._explainer.shap_values(X)
        # DeepExplainer may return a list [output_0] or plain array
        if isinstance(raw, list):
            raw = raw[0]
        return np.array(raw)

    # ------------------------------------------------------------------
    def feature_report(
        self,
        shap_values: np.ndarray,
        feature_cols: List[str],
        top_n: int = 10,
    ) -> List[Dict]:
        """
        Aggregate SHAP values across the time dimension and return a
        ranked list of feature attributions.

        Parameters
        ----------
        shap_values : np.ndarray
            Shape (n_samples, seq_len, n_features) from `explain()`.
        feature_cols : List[str]
            Feature names aligned to the last axis.
        top_n : int
            Return only the top-N most influential features.

        Returns
        -------
        List of dicts: {feature, shap_value, direction, rank}
        """
        # Average across samples and time steps → (n_features,)
        mean_shap = shap_values.mean(axis=(0, 1))

        ranked = sorted(
            zip(feature_cols, mean_shap.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]

        return [
            {
                "feature":    feat,
                "shap_value": round(val, 4),
                "direction":  "positive" if val >= 0 else "negative",
                "rank":       i + 1,
            }
            for i, (feat, val) in enumerate(ranked)
        ]

    # ------------------------------------------------------------------
    def correlation_with_target(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_cols: List[str],
    ) -> List[Dict]:
        """
        Compute Pearson correlation of each feature's last time-step value
        against the target. Useful for the heatmap endpoint.
        """
        last_step = X[:, -1, :]        # (n_samples, n_features)
        results   = []
        for i, feat in enumerate(feature_cols):
            corr = float(np.corrcoef(last_step[:, i], y)[0, 1])
            results.append({
                "feature":     feat,
                "correlation": round(corr, 3) if not np.isnan(corr) else 0.0,
            })
        results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return results
