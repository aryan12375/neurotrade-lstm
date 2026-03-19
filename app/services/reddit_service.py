"""
NeuroTrade — Reddit Hype Service
Tracks mention volume for watchlist stocks on financial subreddits
to power the "Hype vs Reality" gauge.

Subreddits monitored:
    r/stocks, r/investing, r/wallstreetbets, r/StockMarket, r/options
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from cachetools import TTLCache
from loguru import logger

from app.core.config import get_settings

settings = get_settings()

# TTL cache: refresh mentions every 10 minutes
_mention_cache: TTLCache = TTLCache(maxsize=20, ttl=600)

SUBREDDITS = ["stocks", "investing", "wallstreetbets", "StockMarket", "options"]

# Sentiment keyword sets (very lightweight — not a full NLP model)
_POSITIVE_WORDS = {
    "bull", "bullish", "buy", "moon", "rally", "breakout",
    "upgrade", "beat", "strong", "growth", "hold", "long",
}
_NEGATIVE_WORDS = {
    "bear", "bearish", "sell", "crash", "dump", "downgrade",
    "miss", "weak", "short", "puts", "decline", "fall",
}


class RedditHypeService:
    """
    Fetches Reddit post/comment mentions of stock symbols.

    If PRAW credentials are not configured, falls back to realistic
    simulated data so the dashboard still functions.
    """

    def __init__(self) -> None:
        self._reddit = None
        if settings.reddit_enabled:
            try:
                import praw
                self._reddit = praw.Reddit(
                    client_id=settings.reddit_client_id,
                    client_secret=settings.reddit_client_secret,
                    user_agent=settings.reddit_user_agent,
                    read_only=True,
                )
                logger.info("Reddit PRAW client initialised (read-only)")
            except Exception as exc:
                logger.warning(f"Reddit init failed: {exc} — using simulated data")

    # ------------------------------------------------------------------
    def get_mentions(
        self,
        symbols: List[str],
        hours: int = 24,
        limit: int = 500,
    ) -> List[Dict]:
        """
        Return mention counts + basic sentiment for each symbol.

        Returns
        -------
        List of dicts:
            {symbol, mentions, sentiment_score, sentiment_label,
             top_posts, hype_flag, timestamp}
        """
        cache_key = f"{'_'.join(sorted(symbols))}_{hours}"
        if cache_key in _mention_cache:
            return _mention_cache[cache_key]

        if self._reddit:
            result = self._fetch_live(symbols, hours, limit)
        else:
            result = self._simulate(symbols)

        _mention_cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    def _fetch_live(
        self,
        symbols: List[str],
        hours: int,
        limit: int,
    ) -> List[Dict]:
        cutoff     = datetime.now(timezone.utc) - timedelta(hours=hours)
        sym_upper  = [s.upper() for s in symbols]
        counts: Dict[str, int]         = defaultdict(int)
        sentiment: Dict[str, List[float]] = defaultdict(list)
        top_posts: Dict[str, List[str]]   = defaultdict(list)

        for sub_name in SUBREDDITS:
            try:
                subreddit = self._reddit.subreddit(sub_name)
                for post in subreddit.new(limit=limit):
                    if datetime.fromtimestamp(post.created_utc, tz=timezone.utc) < cutoff:
                        continue
                    text = f"{post.title} {post.selftext}".upper()
                    for sym in sym_upper:
                        if re.search(rf"\b{sym}\b", text):
                            counts[sym] += 1
                            score = _sentiment_score(text)
                            sentiment[sym].append(score)
                            if len(top_posts[sym]) < 3:
                                top_posts[sym].append(post.title[:100])
            except Exception as exc:
                logger.warning(f"r/{sub_name} fetch error: {exc}")

        return self._format_results(symbols, counts, sentiment, top_posts)

    # ------------------------------------------------------------------
    def _simulate(self, symbols: List[str]) -> List[Dict]:
        """Plausible simulated data when Reddit credentials are absent."""
        import random, hashlib
        seed = int(datetime.now().strftime("%Y%m%d%H")) % 1000
        random.seed(seed)

        base_mentions = {"AAPL":1247,"NVDA":3891,"TSLA":5234,"MSFT":892,"AMZN":1103}
        counts    = {}
        sentiment = {}
        for sym in symbols:
            base = base_mentions.get(sym, 800)
            counts[sym] = base + random.randint(-200, 400)
            sentiment[sym] = [random.uniform(-0.3, 0.8) for _ in range(50)]

        return self._format_results(symbols, counts, sentiment, {})

    # ------------------------------------------------------------------
    @staticmethod
    def _format_results(
        symbols:   List[str],
        counts:    Dict,
        sentiment: Dict,
        top_posts: Dict,
    ) -> List[Dict]:
        results = []
        max_mentions = max(counts.values(), default=1)

        for sym in symbols:
            n    = counts.get(sym, 0)
            vals = sentiment.get(sym, [0.0])
            avg_sent = float(sum(vals) / max(len(vals), 1))

            if avg_sent > 0.2:
                sent_label = "positive"
            elif avg_sent < -0.2:
                sent_label = "negative"
            else:
                sent_label = "neutral"

            # Hype flag: based on mention volume relative to peers
            ratio = n / max(max_mentions, 1)
            if ratio > 0.7:
                hype_flag = "high"
            elif ratio > 0.35:
                hype_flag = "medium"
            else:
                hype_flag = "low"

            results.append({
                "symbol":          sym,
                "mentions":        n,
                "sentiment_score": round(avg_sent, 3),
                "sentiment_label": sent_label,
                "top_posts":       top_posts.get(sym, []),
                "hype_flag":       hype_flag,
                "timestamp":       datetime.utcnow().isoformat(),
            })

        results.sort(key=lambda x: x["mentions"], reverse=True)
        return results


def _sentiment_score(text: str) -> float:
    words  = set(text.lower().split())
    pos    = len(words & _POSITIVE_WORDS)
    neg    = len(words & _NEGATIVE_WORDS)
    total  = pos + neg
    return (pos - neg) / total if total > 0 else 0.0
