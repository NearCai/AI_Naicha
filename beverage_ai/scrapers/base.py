"""ReviewRecord schema + BaseScraper protocol."""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from typing import Protocol

from pydantic import BaseModel, Field


class ReviewRecord(BaseModel):
    """A single raw review captured from any source.

    `review_id` is deterministic — derived from source + brand + text — so that
    repeated runs of a scraper don't create duplicates.
    """

    review_id: str = Field(min_length=1)
    source: str = Field(min_length=1)             # dianping / xiaohongshu / mock / local_file / ...
    brand: str | None = None
    sku: str | None = None
    text: str = Field(min_length=1)
    customization_raw: str | None = None          # raw text like "三分糖去冰加芋圆"
    rating: float | None = None                   # 1-5 if available
    user_id_hashed: str | None = None             # never store real user ids
    source_url: str | None = None
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)


def make_review_id(source: str, brand: str | None, text: str) -> str:
    """Deterministic id so the same review isn't re-ingested twice."""
    h = hashlib.sha256(f"{source}|{brand or ''}|{text}".encode()).hexdigest()
    return f"rv_{h[:16]}"


class BaseScraper(Protocol):
    """Every source adapter implements this minimal interface."""

    source_name: str

    def scrape(
        self,
        *,
        keywords: list[str] | None = None,
        brand: str | None = None,
        max_records: int | None = None,
    ) -> Iterable[ReviewRecord]:
        """Yield ReviewRecord instances. Should be lazy/streaming."""
        ...


def normalize_text(text: str) -> str:
    """Light normalization: strip, collapse whitespace, drop control chars."""
    import re
    text = text.replace("​", "").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def hash_user_id(raw: str | None) -> str | None:
    """One-way hash for user identifiers; we never persist plaintext."""
    if not raw:
        return None
    return "u_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def yield_with_cap(it: Iterable[ReviewRecord], cap: int | None) -> Iterator[ReviewRecord]:
    if cap is None:
        yield from it
        return
    for i, rec in enumerate(it):
        if i >= cap:
            return
        yield rec
