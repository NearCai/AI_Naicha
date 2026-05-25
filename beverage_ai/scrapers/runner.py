"""Multi-source scrape orchestrator."""
from __future__ import annotations

from dataclasses import dataclass

from ..utils.logging import get_logger
from .base import BaseScraper, yield_with_cap
from .store import RawReviewStore

logger = get_logger("scrapers.runner")


@dataclass
class ScrapeStats:
    source: str
    requested: int
    written: int


class ScrapeRunner:
    """Drive one or more BaseScraper instances and persist to a RawReviewStore."""

    def __init__(self, store: RawReviewStore):
        self.store = store

    def run(
        self,
        scraper: BaseScraper,
        *,
        shard: str,
        keywords: list[str] | None = None,
        brand: str | None = None,
        max_records: int | None = None,
    ) -> ScrapeStats:
        logger.info(
            f"starting scrape source={scraper.source_name} "
            f"keywords={keywords} brand={brand} cap={max_records}"
        )
        records = list(
            yield_with_cap(
                scraper.scrape(keywords=keywords, brand=brand, max_records=max_records),
                max_records,
            )
        )
        n_written = self.store.append(shard, records)
        logger.info(f"finished: requested={len(records)} written_new={n_written}")
        return ScrapeStats(
            source=scraper.source_name,
            requested=len(records),
            written=n_written,
        )
