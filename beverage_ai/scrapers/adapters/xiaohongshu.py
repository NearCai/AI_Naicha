"""Xiaohongshu (小红书) note scraper SKELETON.

⚠️  Same caveats as DianpingScraper. Xiaohongshu's anti-bot is even more
aggressive (device fingerprinting, behavioural detection). DO NOT run against
production without explicit ToS compliance.

For development, use MockScraper / LocalFileScraper.

This file documents the intended flow:
    1. Login (cookies via persistent storage_state)
    2. Search by keyword
    3. Iterate note cards
    4. Click each → extract title + body + comments
    5. Hash user_id
    6. Yield ReviewRecord per note (or per top-level comment)
"""
from __future__ import annotations

from collections.abc import Iterable

from ..base import ReviewRecord


class XiaohongshuScraper:
    source_name = "xiaohongshu"

    def __init__(self, *, headless: bool = True, rate_limit_sec: float = 5.0):
        try:
            import playwright.sync_api  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "playwright required. Install with: pip install -e '.[scrape]'"
            ) from e
        self.headless = headless
        self.rate_limit_sec = rate_limit_sec

    def scrape(
        self,
        *,
        keywords: list[str] | None = None,
        brand: str | None = None,
        max_records: int | None = None,
    ) -> Iterable[ReviewRecord]:
        raise NotImplementedError(
            "XiaohongshuScraper is a SKELETON. See module docstring for "
            "intended flow, and SCRAPING_NOTICE.md for ToS considerations."
        )
