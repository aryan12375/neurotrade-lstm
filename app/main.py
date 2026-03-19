"""
NeuroTrade — FastAPI Application Factory
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import setup_logging

settings = get_settings()

from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

class AIRequest(BaseModel):
    prompt: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle handler."""
    setup_logging(settings.log_level)
    logger.info("=" * 60)
    logger.info("  NeuroTrade LSTM Backend — Starting Up")
    logger.info(f"  Model dir : {settings.model_dir}")
    logger.info(f"  Watchlist : {settings.watchlist}")
    logger.info(f"  Reddit    : {'enabled' if settings.reddit_enabled else 'simulated'}")
    logger.info("=" * 60)

    # Pre-create directories
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    yield   # ── application running ──────────────────────────────────────

    logger.info("NeuroTrade Backend — Shutting Down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="NeuroTrade — LSTM Stock Intelligence API",
        description=(
            "REST API powering the NeuroTrade dashboard. "
            "Provides LSTM predictions, SHAP explainability, "
            "technical indicators, and Reddit hype data.\n\n"
            "**⚠️ NOT FINANCIAL ADVICE — FOR EDUCATIONAL PURPOSES ONLY**"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list + ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── GZip compression ────────────────────────────────────────────────
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Routes ──────────────────────────────────────────────────────────
    app.include_router(router)

    # ── Serve frontend static files if present ──────────────────────────
    frontend_dir = Path("frontend")
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
        logger.info(f"Serving frontend from {frontend_dir.resolve()}")

    # ── Global exception handler ────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)},
        )

    return app


app = create_app()

@app.post("/api/v1/ai-advice")
async def get_ai_advice(req: AIRequest):
    try:
        response = gemini_model.generate_content(req.prompt)
        return {"advice": response.text}
    except Exception as e:
        return {"error": str(e)}