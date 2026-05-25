"""Weibo (微博) Open API adapter — Path A source #2.

⚠️  Reality check (read before using):
  - Weibo's official open API public-search endpoints have been progressively
    restricted. Free tier currently allows very limited queries.
  - You need: registered app at open.weibo.com + OAuth2 access token.
  - Free tier quota: ~150 requests/hour, ~50 statuses per query.
  - Realistic yield: a few hundred to a few thousand statuses per day.
    Not enough alone for the 50K target — combine with other sources.

Auth setup:
  1. Register at https://open.weibo.com
  2. Create an app, get App Key + Secret
  3. Complete OAuth2 flow (web or device flow) to get an access token
  4. export WEIBO_ACCESS_TOKEN=...

Usage:
    scraper = WeiboAPIScraper()    # token from env
    records = list(scraper.scrape(
        keywords=["奶茶", "茶饮新品"],
        max_records=500,
    ))

If your token is invalid or quota exceeded, the scraper raises
`WeiboAPIError` with the upstream message intact.
"""
from __future__ import annotations

import os
import time
from typing import Any, Iterable

from ..base import ReviewRecord, hash_user_id, make_review_id, normalize_text


WEIBO_SEARCH_URL = "https://api.weibo.com/2/search/statuses.json"
WEIBO_DEFAULT_COUNT_PER_PAGE = 50
WEIBO_DEFAULT_RATE_PER_HOUR = 150


class WeiboAPIError(RuntimeError):
    """Raised when the Weibo upstream returns an error."""


class WeiboAPIScraper:
    source_name = "weibo_api"

    def __init__(
        self,
        *,
        access_token: str | None = None,
        max_per_hour: int = WEIBO_DEFAULT_RATE_PER_HOUR,
        count_per_page: int = WEIBO_DEFAULT_COUNT_PER_PAGE,
        api_url: str = WEIBO_SEARCH_URL,
    ):
        token = access_token or os.environ.get("WEIBO_ACCESS_TOKEN")
        if not token:
            raise ValueError(
                "Need access_token (env var WEIBO_ACCESS_TOKEN). "
                "Get one via https://open.weibo.com OAuth2 flow."
            )
        self.token = token
        self.max_per_hour = max_per_hour
        self.count_per_page = count_per_page
        self.api_url = api_url
        self._call_count = 0
        self._window_start = time.time()

    def scrape(
        self,
        *,
        keywords: list[str] | None = None,
        brand: str | None = None,
        max_records: int | None = None,
    ) -> Iterable[ReviewRecord]:
        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx required. Install with: pip install -e '.[scrape]'"
            ) from e

        keywords = keywords or ["奶茶"]
        emitted = 0
        with httpx.Client(timeout=30) as client:
            for keyword in keywords:
                page = 1
                while True:
                    if max_records is not None and emitted >= max_records:
                        return
                    self._rate_limit()
                    params = {
                        "q": keyword,
                        "count": self.count_per_page,
                        "page": page,
                        "access_token": self.token,
                    }
                    try:
                        resp = client.get(self.api_url, params=params)
                    except httpx.HTTPError as e:
                        raise WeiboAPIError(f"HTTP error: {e}") from e

                    if resp.status_code != 200:
                        raise WeiboAPIError(
                            f"HTTP {resp.status_code}: {resp.text[:200]}"
                        )

                    payload = resp.json()
                    if "error" in payload:
                        raise WeiboAPIError(
                            f"Weibo API: {payload.get('error')} "
                            f"(code={payload.get('error_code')})"
                        )

                    statuses = payload.get("statuses") or []
                    if not statuses:
                        break

                    for st in statuses:
                        if max_records is not None and emitted >= max_records:
                            return
                        rec = _status_to_record(st, brand)
                        if rec is None:
                            continue
                        yield rec
                        emitted += 1

                    page += 1
                    # Weibo caps pagination depth too
                    if page > 50:
                        break

    def _rate_limit(self) -> None:
        """Sleep just enough to stay under max_per_hour."""
        self._call_count += 1
        elapsed = time.time() - self._window_start
        if elapsed > 3600:
            self._call_count = 1
            self._window_start = time.time()
            return
        if self._call_count > self.max_per_hour:
            sleep_for = max(3600 - elapsed, 1.0)
            time.sleep(sleep_for)
            self._call_count = 1
            self._window_start = time.time()
        else:
            # Pace evenly: 1 call every (3600 / max_per_hour) seconds
            min_interval = 3600.0 / self.max_per_hour
            time.sleep(max(0.0, min_interval - 0.01))


def _status_to_record(status: dict[str, Any], brand: str | None) -> ReviewRecord | None:
    """Convert a Weibo status JSON into a ReviewRecord."""
    text = status.get("text") or status.get("text_raw") or ""
    text = normalize_text(text)
    if not text or len(text) < 10:
        return None
    user = status.get("user") or {}
    user_id = str(user.get("id")) if user else None
    return ReviewRecord(
        review_id=make_review_id("weibo_api", brand, text),
        source="weibo_api",
        brand=brand,
        text=text,
        user_id_hashed=hash_user_id(user_id),
        source_url=f"https://weibo.com/{user.get('id')}/{status.get('id')}"
                   if user_id and status.get("id") else None,
        metadata={
            "weibo_status_id": status.get("id"),
            "reposts_count": status.get("reposts_count"),
            "comments_count": status.get("comments_count"),
            "attitudes_count": status.get("attitudes_count"),
        },
    )
