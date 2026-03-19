"""
NeuroTrade — Application Configuration
Loads from .env file using pydantic-settings.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    log_level: str = "info"

    # ── Security ──────────────────────────────────────────
    cors_origins: str = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5500"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    # ── Paths ──────────────────────────────────────────────
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    model_dir: Path = Path("./saved_models")
    data_dir: Path = Path("./data")

    def model_post_init(self, __context) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ── Model Hyperparameters ─────────────────────────────
    sequence_length: int = 60
    forecast_days: int = 10
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    lstm_units_1: int = 128
    lstm_units_2: int = 64

    # ── Data ──────────────────────────────────────────────
    default_period: str = "5y"
    default_interval: str = "1d"
    cache_ttl_seconds: int = 300

    # ── Watchlist ─────────────────────────────────────────
    default_watchlist: str = "AAPL,NVDA,TSLA,MSFT,AMZN"

    @property
    def watchlist(self) -> List[str]:
        return [s.strip().upper() for s in self.default_watchlist.split(",")]

    # ── Reddit ────────────────────────────────────────────
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "NeuroTrade/1.0"

    @property
    def reddit_enabled(self) -> bool:
        return bool(self.reddit_client_id and self.reddit_client_secret)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
