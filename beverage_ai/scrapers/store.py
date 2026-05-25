"""Storage layer for raw reviews — Parquet + DuckDB.

Schema is kept simple so we can evolve via Parquet column-add.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from ..utils.logging import get_logger
from .base import ReviewRecord

logger = get_logger("scrapers.store")

_PARQUET_NAME = "raw_reviews.parquet"


class RawReviewStore:
    """Append-only store for raw reviews.

    Backed by a single parquet file per shard. Use one shard per source
    or per scrape session, depending on how you organize data/.
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _shard_path(self, shard: str) -> Path:
        return self.base_dir / shard / _PARQUET_NAME

    def append(self, shard: str, records: Iterable[ReviewRecord]) -> int:
        """Append records to a shard (creates the shard if needed).

        Returns the number of NEW records written (after dedup).
        """
        records = list(records)
        if not records:
            return 0

        new_df = pd.DataFrame([_record_to_row(r) for r in records])

        target = self._shard_path(shard)
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists():
            existing = pd.read_parquet(target)
            combined = pd.concat([existing, new_df], ignore_index=True)
            before = len(combined)
            combined = combined.drop_duplicates(subset=["review_id"], keep="first")
            n_new = len(combined) - len(existing)
            combined.to_parquet(target, index=False)
            logger.info(
                f"shard={shard}: +{n_new} new ({before - len(combined)} dups removed)"
            )
            return n_new
        else:
            new_df = new_df.drop_duplicates(subset=["review_id"], keep="first")
            new_df.to_parquet(target, index=False)
            logger.info(f"shard={shard}: created with {len(new_df)} records")
            return len(new_df)

    def list_shards(self) -> list[str]:
        return sorted(p.name for p in self.base_dir.iterdir() if p.is_dir())

    def read(self, shard: str | None = None) -> pd.DataFrame:
        """Load one shard or all shards into a DataFrame."""
        if shard is not None:
            p = self._shard_path(shard)
            if not p.exists():
                return pd.DataFrame()
            return pd.read_parquet(p)
        dfs = []
        for sh in self.list_shards():
            p = self._shard_path(sh)
            if p.exists():
                dfs.append(pd.read_parquet(p))
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def count(self) -> int:
        return sum(len(self.read(sh)) for sh in self.list_shards())

    def query(self, sql: str) -> pd.DataFrame:
        """Run an ad-hoc SQL query against all shards via DuckDB.

        Inside the query, reference the table as `raw`.
        """
        df = self.read()
        if df.empty:
            return pd.DataFrame()
        con = duckdb.connect(":memory:")
        con.register("raw", df)
        result = con.execute(sql).df()
        con.close()
        return result


def _record_to_row(r: ReviewRecord) -> dict:
    d = r.model_dump()
    d["metadata"] = json.dumps(d.get("metadata") or {}, ensure_ascii=False)
    return d
