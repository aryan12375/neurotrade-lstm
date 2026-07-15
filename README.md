# NeuroTrade — LSTM Stock Intelligence Platform

> **⚠️ STRICT WARNING: NOT FINANCIAL ADVICE**
> This project is built entirely for **educational and portfolio demonstration purposes**.
> AI/LSTM models can hallucinate, lag behind real-time events, or misinterpret market data.
> Never make real investment decisions based on this tool.

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Training Models](#training-models)
6. [API Reference](#api-reference)
7. [Frontend Integration](#frontend-integration)
8. [Configuration](#configuration)
9. [Tech Stack](#tech-stack)

---

## Overview

NeuroTrade is a full-stack AI stock analysis platform featuring:

| Feature | Details |
|---|---|
| **LSTM Model** | 2-layer LSTM + Dropout + BatchNorm + Dense |
| **24 Features** | OHLCV + RSI, MACD, Bollinger Bands, ATR, OBV, MA crossovers, custom derived |
| **XAI (SHAP)** | DeepExplainer attributions — shows *why* the model predicted a price |
| **Walk-Forward CV** | 8-fold time-aware validation, zero look-ahead bias |
| **Forecast** | 10-day iterative multi-step price trajectory with confidence bounds |
| **Reddit Hype** | r/stocks mention tracking vs LSTM signal ("Hype vs Reality" gauge) |
| **REST API** | FastAPI with auto-generated Swagger docs at `/docs` |
| **Dashboard** | Single-file HTML frontend with 10+ Chart.js charts, dark/light mode |

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │        NeuroTrade Frontend        │
                    │   (HTML + Chart.js Dashboard)     │
                    └──────────────┬──────────────────-─┘
                                   │ HTTP / REST
                    ┌──────────────▼──────────────────-─┐
                    │         FastAPI Backend            │
                    │   /api/v1/...  (uvicorn server)    │
                    └──┬───────┬──────────┬────────────-─┘
                       │       │          │
           ┌───────────▼─┐  ┌──▼──────┐  ┌▼──────────────┐
           │ Data        │  │  LSTM   │  │ SHAP          │
           │ Pipeline    │  │  Model  │  │ Explainer     │
           │ (yfinance   │  │ (Keras/ │  │ (DeepExplain) │
           │ + ta lib)   │  │  TF)    │  └───────────────┘
           └─────────────┘  └─────────┘
                       │
           ┌───────────▼─────────────┐
           │   saved_models/         │
           │   ├── AAPL_model.keras  │
           │   ├── AAPL_scaler.pkl   │
           │   └── AAPL_eval.json    │
           └─────────────────────────┘
```

### LSTM Architecture

```
Input  (batch, 60, 24)
  │
  ├─→ LSTM(128, return_sequences=True)   ← captures long-term memory
  ├─→ Dropout(0.2)
  ├─→ BatchNormalization
  │
  ├─→ LSTM(64, return_sequences=False)   ← compresses to fixed vector
  ├─→ Dropout(0.2)
  ├─→ BatchNormalization
  │
  ├─→ Dense(32, relu)                    ← non-linear mapping
  └─→ Dense(1)                           ← predicted next-day close price

Loss:      MSE  |  Optimizer: Adam(lr=0.001)
Callbacks: EarlyStopping(patience=7) + ReduceLROnPlateau + ModelCheckpoint
```

### Data Pipeline

```
yfinance.download(symbol, period='5y')
    │
    ├─ 24 Technical Features computed via `ta` library:
    │     Price:      Open, High, Low, Close, Volume
    │     Trend:      SMA_20, SMA_50, EMA_20, MACD, MACD_Signal, MACD_Diff
    │     Momentum:   RSI_14, Stoch_K, Stoch_D
    │     Volatility: BB_High, BB_Low, BB_Width, BB_Pct, ATR_14
    │     Volume:     OBV
    │     Custom:     Daily_Return, Log_Return, HL_Ratio, OC_Ratio
    │
    ├─ MinMaxScaler → scale all features to [0, 1]
    ├─ Build (X, y) sequences: window=60 days → predict day 61
    └─ Temporal train/test split: 80% train / 20% test  (NO shuffling)
```

---

## Project Structure

```
neurotrade/
├── app/
│   ├── main.py                  # FastAPI app factory + lifespan
│   ├── api/
│   │   ├── routes.py            # All endpoint handlers
│   │   └── schemas.py           # Pydantic request/response models
│   ├── core/
│   │   ├── config.py            # Settings (pydantic-settings + .env)
│   │   └── logging.py           # Loguru setup
│   ├── models/
│   │   └── lstm_model.py        # Model build, train, eval, walk-forward CV
│   └── services/
│       ├── data_pipeline.py     # yfinance fetch + feature engineering
│       ├── prediction_service.py # Inference + confidence + forecast
│       ├── shap_service.py      # SHAP DeepExplainer wrapper
│       └── reddit_service.py    # Reddit mention tracker (PRAW)
├── scripts/
│   └── train.py                 # CLI training script
├── tests/
│   └── test_backend.py          # Pytest test suite (25 tests)
├── models/                      # MODEL_DIR used by the provided .env.example
├── data/                        # Auto-created on first run
├── logs/                        # Auto-created on first run
├── lstm_stock_predictor_fixed.html # Frontend dashboard (drop-in)
├── run.py                       # Server entry point
├── requirements.txt
└── .env.example                 # Copy to .env and configure
```

---

## Quick Start

### 1. Clone & Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env; replace optional credential placeholders or clear them to use
# the built-in local fallbacks.
```

### 3. Train Your First Model

```bash
# Train AAPL (takes ~3-5 minutes on CPU, ~1 min on GPU)
python scripts/train.py --symbol AAPL

# Train every symbol configured in DEFAULT_WATCHLIST
python scripts/train.py --all

# Custom options
python scripts/train.py --symbol NVDA --epochs 75 --period 5y --seq-len 60
```

You'll see output like:
```
══════════════════════════════════════════════════
  Training: AAPL
  period=5y | epochs=50 | seq_len=60
══════════════════════════════════════════════════
Epoch 1/50  loss: 0.0842  val_loss: 0.0931
Epoch 2/50  loss: 0.0631  val_loss: 0.0714
...
✓ AAPL  RMSE=1.847   R²=0.9312   DirAcc=78.4%   47.3s
```

### 4. Start the API Server

```bash
python run.py
# Server starts at http://localhost:8000
# Interactive API docs at http://localhost:8000/docs
```

### 5. Open the Dashboard

Simply open `lstm_stock_predictor_fixed.html` in your browser.
The dashboard auto-connects to `http://localhost:8000/api/v1`.

---

## Training Models

### CLI Reference

```bash
python scripts/train.py [OPTIONS]

Options:
  --symbol AAPL          Train a single symbol
  --symbols AAPL NVDA    Train multiple symbols
  --all                  Train full watchlist from .env
  --period 5y            Historical data period (1y/2y/5y/10y)
  --epochs 50            Max training epochs (EarlyStopping applies)
  --batch-size 32        Mini-batch size
  --seq-len 60           Lookback window (days)
  --no-wf                Skip walk-forward validation (faster)
  --folds 8              Number of walk-forward folds
```

### What Gets Saved

After training `AAPL`, you'll have:
```
models/
├── AAPL_model.keras     # Full Keras model (architecture + weights)
├── AAPL_scaler.pkl      # MinMaxScaler (must match training features)
└── AAPL_eval.json       # Evaluation metrics + epoch history + WF folds
```

---

## API Reference

Full interactive docs: **http://localhost:8000/docs**

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/stocks/{symbol}/history` | Raw OHLCV bars |
| `GET` | `/api/v1/stocks/{symbol}/indicators` | All technical indicators |
| `GET` | `/api/v1/stocks/{symbol}/predict` | Next-day LSTM prediction |
| `GET` | `/api/v1/stocks/{symbol}/forecast` | Configurable multi-day price forecast |
| `GET` | `/api/v1/stocks/{symbol}/shap` | SHAP feature attributions |
| `GET` | `/api/v1/stocks/{symbol}/correlations` | Feature-target correlations |
| `GET` | `/api/v1/stocks/{symbol}/eval` | Model evaluation metrics |
| `GET` | `/api/v1/watchlist` | All watchlist predictions |
| `GET` | `/api/v1/hype` | Reddit mention counts |
| `GET` | `/api/v1/models/status` | Which models are trained |
| `POST` | `/api/v1/train` | Train/retrain a model |
| `POST` | `/api/v1/ai-advice` | Optional Gemini-generated educational commentary |

### Example Requests

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Get prediction for AAPL
curl http://localhost:8000/api/v1/stocks/AAPL/predict

# Get 10-day forecast
curl "http://localhost:8000/api/v1/stocks/AAPL/forecast?days=10"

# Get SHAP explainability
curl http://localhost:8000/api/v1/stocks/AAPL/shap

# Train NVDA via API
curl -X POST http://localhost:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{"symbol":"NVDA","epochs":50,"walk_forward":true}'

# Reddit hype (last 24 hours)
curl "http://localhost:8000/api/v1/hype?hours=24"

# Watchlist summary
curl http://localhost:8000/api/v1/watchlist
```

### Example Response: `/stocks/AAPL/predict`

```json
{
  "symbol": "AAPL",
  "current_price": 189.84,
  "predicted_price": 192.31,
  "predicted_change": 2.47,
  "predicted_change_pct": 1.301,
  "confidence": 84.2,
  "signal": "BUY",
  "timestamp": "2026-03-19T10:22:01.483Z"
}
```

---

## Frontend Integration

The dashboard (`lstm_stock_predictor_fixed.html`) connects to the backend via the `BACKEND_URL` constant at the top of its `<script>` section. By default:

```javascript
const BACKEND_URL = 'http://localhost:8000/api/v1';
```

When a trained model exists, the dashboard will:
- Replace simulated chart data with **real LSTM predictions**
- Show **live SHAP values** in the explainability panel
- Display **real confidence scores** (Monte Carlo Dropout)
- Pull **actual Reddit mention counts** from the hype gauge

When no model is trained yet, the dashboard gracefully falls back to simulated data so it always looks complete for demos.

---

## Configuration

All settings are in `.env` (copy from `.env.example`):

```env
# Server
HOST=0.0.0.0
PORT=8000

# CORS (add your frontend URL)
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:5500

# Model hyperparameters
MODEL_DIR=models
SEQUENCE_LENGTH=60       # days of history fed into LSTM
FORECAST_DAYS=7
EPOCHS=50
LEARNING_RATE=0.001
DROPOUT_RATE=0.2
LSTM_UNITS_1=128
LSTM_UNITS_2=64

# Data
DEFAULT_PERIOD=2y        # yfinance history period
CACHE_TTL_SECONDS=900    # API response cache lifetime

# Watchlist
DEFAULT_WATCHLIST=AAPL,MSFT,NVDA,TSLA

# Reddit (optional)
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
GEMINI_API_KEY=              # optional; required only for /api/v1/ai-advice
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5 + Chart.js 4.4 + Vanilla JS |
| **API** | FastAPI 0.111 + Uvicorn + Pydantic v2 |
| **ML Framework** | TensorFlow 2.16 / Keras 3.3 |
| **Data** | yfinance + pandas + NumPy |
| **Indicators** | `ta` library (RSI, MACD, Bollinger, ATR, OBV, ...) |
| **Explainability** | SHAP (DeepExplainer for Keras) |
| **Reddit** | PRAW (Python Reddit API Wrapper) |
| **Caching** | cachetools TTLCache (in-memory) |
| **Logging** | Loguru |
| **Testing** | Pytest (25 tests) |

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v

# Expected output:
# tests/test_backend.py::TestBuildFeatureDataFrame::test_returns_dataframe PASSED
# tests/test_backend.py::TestBuildFeatureDataFrame::test_has_expected_columns PASSED
# ...
# 25 passed in X.XXs
```

---

*NeuroTrade — Built for educational portfolio demonstration. Not financial advice.*
