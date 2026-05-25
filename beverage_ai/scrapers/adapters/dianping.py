"""Dianping (大众点评) scraper SKELETON.

⚠️  IMPORTANT: 大众点评's ToS prohibits automated scraping. Anti-bot defences
(rotating selectors, CAPTCHA, IP blocking) make a robust scraper a moving
target. For academic / research use, treat this file as a *reference outline*
for the steps a real adapter would take — DO NOT run it against production
without (a) legal clearance, (b) rate limiting / proxy strategy, (c) selector
maintenance pipeline.

For local development, use `MockScraper` or `LocalFileScraper` instead.

Implementation notes:
  1. CSS selectors below are placeholders (`TODO_SELECTOR_*`). They WILL break
     across site redesigns. Maintain in a separate `selectors.yaml` so they
     can be tuned without redeploying code.
  2. Need to handle the lazy-load (scroll-triggered) review list.
  3. Sentiment from `data-score` or star widget if rendered.
  4. User ids must be hashed (`base.hash_user_id`) before persistence.
"""
from __future__ import annotations

from collections.abc import Iterable

from ..base import ReviewRecord, hash_user_id, make_review_id, normalize_text

# Placeholder selectors — keep them in code only as documentation; in
# production, load from data/scrapers/dianping_selectors.yaml
SELECTORS_PLACEHOLDER = {
    "shop_search_input":  "TODO_SELECTOR_input.search",
    "shop_card":          "TODO_SELECTOR_div.shop-card",
    "review_card":        "TODO_SELECTOR_div.review-card",
    "review_text":        "TODO_SELECTOR_div.review-content",
    "review_rating":      "TODO_SELECTOR_span.score",
    "review_user":        "TODO_SELECTOR_a.user-name",
    "next_page":          "TODO_SELECTOR_a.next",
}


class DianpingScraper:
    """Skeleton scraper. Real selectors and anti-bot handling left as exercise."""

    source_name = "dianping"

    def __init__(
        self,
        *,
        headless: bool = True,
        rate_limit_sec: float = 3.0,
        max_pages: int = 5,
        selectors: dict[str, str] | None = None,
    ):
        try:
            import playwright.sync_api  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "playwright required. Install with: pip install -e '.[scrape]' && "
                "playwright install chromium"
            ) from e

        self.headless = headless
        self.rate_limit_sec = rate_limit_sec
        self.max_pages = max_pages
        self.selectors = selectors or SELECTORS_PLACEHOLDER

    def scrape(
        self,
        *,
        keywords: list[str] | None = None,
        brand: str | None = None,
        max_records: int | None = None,
    ) -> Iterable[ReviewRecord]:
        raise NotImplementedError(
            "DianpingScraper is a SKELETON. See docstring for what to implement, "
            "and SCRAPING_NOTICE.md for ToS considerations. Prefer MockScraper / "
            "LocalFileScraper for development."
        )

    # ----- intended building blocks (for when you implement the real version) -----

    def _parse_review_card(self, card_html: str, brand: str | None) -> ReviewRecord | None:
        """Given an HTML chunk for a single review card, build a ReviewRecord.

        Suggested implementation:
            1. parse with selectolax or BeautifulSoup
            2. text = card.css_first(self.selectors["review_text"]).text(strip=True)
            3. rating = float(card.css_first(self.selectors["review_rating"])["data-score"])
            4. user_raw = card.css_first(self.selectors["review_user"]).text()
            5. return ReviewRecord(
                   review_id=make_review_id(self.source_name, brand, text),
                   source=self.source_name,
                   brand=brand,
                   text=normalize_text(text),
                   rating=rating,
                   user_id_hashed=hash_user_id(user_raw),
               )
        """
        _ = (card_html, brand, make_review_id, normalize_text, hash_user_id)
        raise NotImplementedError("Fill in based on actual page HTML structure")
