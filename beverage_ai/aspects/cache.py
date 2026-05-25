"""DuckDB-backed cache for extracted aspects.

Keyed by (review_id, extractor_version) so that:
  - re-running the pipeline only re-extracts NEW reviews
  - switching extractor (mock → claude, or prompt revision) automatically invalidates

We persist the full ExtractedAspects JSON, so audit / inspection is trivial.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb

from .schema import ExtractedAspects

_SCHEMA = """
CREATE TABLE IF NOT EXISTS aspects_cache (
    review_id           VARCHAR NOT NULL,
    extractor_version   VARCHAR NOT NULL,
    aspects_json        JSON,
    customization_json  JSON,
    confidence          DOUBLE,
    cost_estimate_usd   DOUBLE,
    raw_response        TEXT,
    extracted_at        TIMESTAMP,
    PRIMARY KEY (review_id, extractor_version)
);
"""


class AspectCache:
    """Persistent cache mapping (review_id, version) → ExtractedAspects."""

    def __init__(self, db_path: str | Path = "data/reviews/aspects_cache.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.db_path))
        self.con.execute(_SCHEMA)

    def get(self, review_id: str, version: str) -> ExtractedAspects | None:
        row = self.con.execute(
            "SELECT aspects_json, customization_json, confidence, "
            "       cost_estimate_usd, raw_response, extracted_at "
            "FROM aspects_cache WHERE review_id = ? AND extractor_version = ?",
            [review_id, version],
        ).fetchone()
        if row is None:
            return None
        aspects = json.loads(row[0]) if row[0] else {}
        customization = json.loads(row[1]) if row[1] else {}
        return ExtractedAspects(
            review_id=review_id,
            extractor_version=version,
            aspects=aspects,
            customization=customization,
            confidence=row[2] or 0.0,
            cost_estimate_usd=row[3] or 0.0,
            raw_response=row[4],
            extracted_at=row[5] if isinstance(row[5], datetime) else datetime.now(),
        )

    def put(self, result: ExtractedAspects) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO aspects_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                result.review_id,
                result.extractor_version,
                json.dumps(result.aspects, ensure_ascii=False),
                json.dumps(result.customization.model_dump(), ensure_ascii=False),
                result.confidence,
                result.cost_estimate_usd,
                result.raw_response,
                result.extracted_at,
            ],
        )

    def has(self, review_id: str, version: str) -> bool:
        row = self.con.execute(
            "SELECT 1 FROM aspects_cache WHERE review_id = ? AND extractor_version = ?",
            [review_id, version],
        ).fetchone()
        return row is not None

    def count(self, version: str | None = None) -> int:
        if version is None:
            n = self.con.execute("SELECT COUNT(*) FROM aspects_cache").fetchone()[0]
        else:
            n = self.con.execute(
                "SELECT COUNT(*) FROM aspects_cache WHERE extractor_version = ?",
                [version],
            ).fetchone()[0]
        return int(n)

    def list_versions(self) -> list[tuple[str, int]]:
        rows = self.con.execute(
            "SELECT extractor_version, COUNT(*) FROM aspects_cache "
            "GROUP BY extractor_version ORDER BY 2 DESC"
        ).fetchall()
        return [(r[0], int(r[1])) for r in rows]

    def total_cost_usd(self, version: str | None = None) -> float:
        if version is None:
            x = self.con.execute("SELECT COALESCE(SUM(cost_estimate_usd), 0) FROM aspects_cache").fetchone()[0]
        else:
            x = self.con.execute(
                "SELECT COALESCE(SUM(cost_estimate_usd), 0) FROM aspects_cache "
                "WHERE extractor_version = ?",
                [version],
            ).fetchone()[0]
        return float(x)

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> AspectCache:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
