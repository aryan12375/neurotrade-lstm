#!/usr/bin/env python3
"""
NeuroTrade — Server Entry Point
================================
Start the FastAPI server with uvicorn.

Usage:
    python run.py                   # development (hot reload)
    python run.py --env production  # production (no reload, 4 workers)
    python run.py --port 9000       # custom port
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import uvicorn
from app.core.config import get_settings

settings = get_settings()


def main():
    parser = argparse.ArgumentParser(description="Start NeuroTrade API server")
    parser.add_argument("--host",    default=settings.host,  help="Bind host")
    parser.add_argument("--port",    default=settings.port,  type=int, help="Bind port")
    parser.add_argument("--env",     default="development",  choices=["development","production"])
    parser.add_argument("--workers", default=1,              type=int)
    args = parser.parse_args()

    is_dev = args.env == "development"

    print(f"""
╔══════════════════════════════════════════════════╗
║          NeuroTrade LSTM Backend v1.0.0          ║
╠══════════════════════════════════════════════════╣
║  URL  : http://{args.host}:{args.port}
║  Docs : http://{args.host}:{args.port}/docs
║  Mode : {args.env}
╚══════════════════════════════════════════════════╝
""")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=is_dev,
        workers=1 if is_dev else args.workers,
        log_level=settings.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
