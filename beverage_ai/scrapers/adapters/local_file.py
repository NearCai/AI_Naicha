"""Local file scraper — read reviews from CSV/JSON/JSONL.

The expected format is one review per row, with at minimum a `text` column.
Optional columns: source, brand, sku, customization_raw, rating, source_url.

Useful for:
  - Importing manually curated data
  - Reusing public datasets
  - Re-running aspect extraction over an existing corpus
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..base import ReviewRecord, make_review_id, normalize_text


class LocalFileScraper:
    source_name = "local_file"

    def __init__(self, path: str | Path, default_source: str = "local_file"):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.default_source = default_source

    def scrape(
        self,
        *,
        keywords: list[str] | None = None,
        brand: str | None = None,
        max_records: int | None = None,
    ) -> Iterable[ReviewRecord]:
        for i, row in enumerate(self._iter_rows()):
            if max_records is not None and i >= max_records:
                return
            raw_text = row.get("text")
            if raw_text is None or (isinstance(raw_text, float) and pd.isna(raw_text)):
                continue
            text = normalize_text(str(raw_text))
            if not text or text.lower() == "nan":
                continue
            if keywords and not any(k.lower() in text.lower() for k in keywords):
                continue
            row_brand = row.get("brand") or brand
            if brand and row_brand != brand:
                continue
            source = row.get("source") or self.default_source
            yield ReviewRecord(
                review_id=make_review_id(source, row_brand, text),
                source=source,
                brand=row_brand,
                sku=row.get("sku"),
                text=text,
                customization_raw=row.get("customization_raw"),
                rating=_safe_float(row.get("rating")),
                source_url=row.get("source_url"),
            )

    def _iter_rows(self) -> Iterable[dict]:
        suffix = self.path.suffix.lower()
        if suffix in (".csv", ".tsv"):
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(self.path, sep=sep)
            yield from df.to_dict(orient="records")
        elif suffix == ".jsonl":
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        elif suffix == ".json":
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                yield from data
            else:
                yield data
        else:
            raise ValueError(f"Unsupported file extension: {suffix}")


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
        return v if not (v != v) else None  # NaN check
    except (TypeError, ValueError):
        return None
