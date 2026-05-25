"""Batch aspect-extraction pipeline.

Reads raw reviews from a RawReviewStore, extracts aspects via an extractor,
caches results in DuckDB, and reports stats (count, cost, cache hit rate).

Self-consistency support: if `self_consistency > 1`, extract N times and
take the median for each numerical dim (per 技术方案书 §3.3.1 R7).
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import pandas as pd

from ..scrapers.base import ReviewRecord
from ..scrapers.store import RawReviewStore
from ..utils.logging import get_logger
from .cache import AspectCache
from .extractor import AspectExtractor
from .schema import ALL_DIMS, Customization, ExtractedAspects

logger = get_logger("aspects.pipeline")


@dataclass
class ExtractionStats:
    total_reviews: int = 0
    cache_hits: int = 0
    extracted: int = 0
    errors: int = 0
    cost_usd: float = 0.0
    elapsed_sec: float = 0.0
    error_messages: list[str] = field(default_factory=list)

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.total_reviews if self.total_reviews else 0.0

    def to_dict(self) -> dict:
        return {
            "total_reviews": self.total_reviews,
            "cache_hits": self.cache_hits,
            "extracted": self.extracted,
            "errors": self.errors,
            "cost_usd": round(self.cost_usd, 4),
            "elapsed_sec": round(self.elapsed_sec, 2),
            "cache_hit_rate": round(self.cache_hit_rate, 3),
        }


class AspectExtractionPipeline:
    """Orchestrates aspect extraction over raw reviews."""

    def __init__(
        self,
        extractor: AspectExtractor,
        cache: AspectCache,
        self_consistency: int = 1,
    ):
        if self_consistency < 1:
            raise ValueError("self_consistency must be >= 1")
        self.extractor = extractor
        self.cache = cache
        self.self_consistency = self_consistency

    # ----- main entry points -----

    def run_on_records(
        self,
        records: list[ReviewRecord],
        *,
        limit: int | None = None,
        cost_ceiling_usd: float | None = None,
    ) -> ExtractionStats:
        stats = ExtractionStats()
        t0 = time.time()

        for i, rec in enumerate(records):
            if limit and i >= limit:
                break
            stats.total_reviews += 1

            cached = self.cache.get(rec.review_id, self.extractor.version)
            if cached is not None:
                stats.cache_hits += 1
                continue

            if cost_ceiling_usd is not None and stats.cost_usd >= cost_ceiling_usd:
                logger.warning(
                    f"cost ceiling ${cost_ceiling_usd:.2f} reached after "
                    f"{stats.extracted} extractions; stopping"
                )
                break

            try:
                result = self._extract_one(rec)
            except Exception as e:
                stats.errors += 1
                stats.error_messages.append(f"{rec.review_id}: {type(e).__name__}: {e}")
                logger.error(f"extract failed for {rec.review_id}: {e}")
                continue

            self.cache.put(result)
            stats.extracted += 1
            stats.cost_usd += result.cost_estimate_usd

            if stats.extracted % 50 == 0:
                logger.info(
                    f"extracted={stats.extracted} cache_hit={stats.cache_hits} "
                    f"cost=${stats.cost_usd:.2f}"
                )

        stats.elapsed_sec = time.time() - t0
        logger.info(f"pipeline done: {stats.to_dict()}")
        return stats

    def run_on_store(
        self,
        store: RawReviewStore,
        *,
        shard: str | None = None,
        limit: int | None = None,
        cost_ceiling_usd: float | None = None,
    ) -> ExtractionStats:
        df = store.read(shard=shard)
        if df.empty:
            logger.warning("no raw reviews to process")
            return ExtractionStats()
        # Reconstruct minimal ReviewRecord just for extraction
        records = [
            ReviewRecord(
                review_id=row["review_id"], source=row["source"], brand=row.get("brand"),
                sku=row.get("sku"), text=row["text"],
                customization_raw=row.get("customization_raw"),
                rating=row.get("rating"), source_url=row.get("source_url"),
            )
            for row in df.to_dict(orient="records")
        ]
        return self.run_on_records(records, limit=limit, cost_ceiling_usd=cost_ceiling_usd)

    # ----- internals -----

    def _extract_one(self, rec: ReviewRecord) -> ExtractedAspects:
        if self.self_consistency <= 1:
            return self.extractor.extract(rec)
        runs = [self.extractor.extract(rec) for _ in range(self.self_consistency)]
        return _aggregate_consistency(runs, rec.review_id)


def _aggregate_consistency(runs: list[ExtractedAspects], review_id: str) -> ExtractedAspects:
    """Median aggregation for numerical aspects; first non-null for customization."""
    if not runs:
        raise ValueError("empty runs")
    version = runs[0].extractor_version + "|sc"
    raw_response = " ||| ".join(r.raw_response or "" for r in runs)

    aspects: dict[str, float | None] = {}
    for dim in ALL_DIMS:
        vals = [r.aspects.get(dim) for r in runs if r.aspects.get(dim) is not None]
        aspects[dim] = round(float(statistics.median(vals)), 3) if vals else None

    # First non-null customization slot wins
    custom = Customization()
    for r in runs:
        if not custom.sugar_level and r.customization.sugar_level:
            custom.sugar_level = r.customization.sugar_level
        if not custom.ice_level and r.customization.ice_level:
            custom.ice_level = r.customization.ice_level
        if not custom.size and r.customization.size:
            custom.size = r.customization.size
        if r.customization.toppings and not custom.toppings:
            custom.toppings = list(r.customization.toppings)

    return ExtractedAspects(
        review_id=review_id,
        extractor_version=version,
        aspects=aspects,
        customization=custom,
        confidence=round(statistics.median([r.confidence for r in runs]), 3),
        raw_response=raw_response,
        cost_estimate_usd=sum(r.cost_estimate_usd for r in runs),
    )


def aspects_to_dataframe(
    cache: AspectCache,
    extractor_version: str | None = None,
) -> pd.DataFrame:
    """Materialize all cached aspects into a DataFrame for analysis / training."""
    if extractor_version:
        rows = cache.con.execute(
            "SELECT review_id, extractor_version, aspects_json, customization_json, "
            "       confidence, cost_estimate_usd "
            "FROM aspects_cache WHERE extractor_version = ?",
            [extractor_version],
        ).fetchall()
    else:
        rows = cache.con.execute(
            "SELECT review_id, extractor_version, aspects_json, customization_json, "
            "       confidence, cost_estimate_usd FROM aspects_cache"
        ).fetchall()

    import json as _json

    records = []
    for review_id, version, aspects_json, custom_json, conf, cost in rows:
        aspects = _json.loads(aspects_json) if aspects_json else {}
        custom = _json.loads(custom_json) if custom_json else {}
        flat = {"review_id": review_id, "extractor_version": version,
                "confidence": conf, "cost_usd": cost}
        for dim in ALL_DIMS:
            flat[f"aspect_{dim}"] = aspects.get(dim)
        flat["sugar_level"] = custom.get("sugar_level")
        flat["ice_level"] = custom.get("ice_level")
        flat["toppings"] = ",".join(custom.get("toppings") or [])
        flat["size"] = custom.get("size")
        records.append(flat)
    return pd.DataFrame(records)
