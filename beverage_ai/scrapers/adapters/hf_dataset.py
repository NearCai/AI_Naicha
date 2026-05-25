"""HuggingFace dataset adapter — recommended Path A source #1.

Free, fully ToS-compliant, immediately usable. Trade-off:
no dataset is tea-drink-specific, so we filter by keyword (奶茶/茶饮/拿铁/...)
to extract on-topic rows. Even off-topic Chinese-review datasets are useful
for GNN Stage 1 pretraining because the *language* of sensory description
transfers across food domains.

Recommended datasets (verify availability on huggingface.co/datasets):
    seamew/ChnSentiCorp                 ~12K general Chinese sentiment
    XiangPan/waimai_10k                 ~12K 外卖 reviews (food delivery, very relevant)
    OneFly/Chinese-Online-Shopping-Reviews  ~60K shopping reviews
    BUAA-NLP/MultiCNRC                  Chinese review collection (multi-domain)
    seamew/ChnDianpingCorp              Dianping reviews (academic release)

Usage:
    scraper = HFDatasetScraper(
        dataset_name="XiangPan/waimai_10k",
        text_column="review",       # auto-detected if omitted
        rating_column="label",      # optional
    )
    records = list(scraper.scrape(keywords=["奶茶", "茶饮", "拿铁"], max_records=5000))
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..base import ReviewRecord, hash_user_id, make_review_id, normalize_text

# Common text-column names across public Chinese review datasets
_TEXT_COLUMN_CANDIDATES = (
    "text", "review", "content", "comment", "review_text",
    "评论", "内容", "正文", "review_content",
)
_LABEL_COLUMN_CANDIDATES = (
    "label", "rating", "score", "stars", "sentiment", "评分", "星级",
)
_USER_COLUMN_CANDIDATES = (
    "user_id", "user", "uid", "author", "用户id",
)


class HFDatasetScraper:
    source_name = "hf_dataset"

    def __init__(
        self,
        dataset_name: str,
        *,
        split: str = "train",
        text_column: str | None = None,
        rating_column: str | None = None,
        rating_scale: tuple[float, float] = (0.0, 1.0),
        cache_dir: str | None = None,
        streaming: bool = True,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.text_column = text_column
        self.rating_column = rating_column
        self.rating_scale = rating_scale
        self.cache_dir = cache_dir
        self.streaming = streaming
        # Tag source with dataset name so downstream can audit provenance
        self.source_name = f"hf:{dataset_name}"

    def scrape(
        self,
        *,
        keywords: list[str] | None = None,
        brand: str | None = None,           # used as tag only; HF data has no brand
        max_records: int | None = None,
    ) -> Iterable[ReviewRecord]:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "datasets package required. Install with: pip install -e '.[hf]'"
            ) from e

        ds = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=self.streaming,
            cache_dir=self.cache_dir,
        )

        # First-row detection
        first = next(iter(ds))
        text_col = self.text_column or _autodetect_column(first, _TEXT_COLUMN_CANDIDATES)
        rating_col = self.rating_column or _autodetect_column(first, _LABEL_COLUMN_CANDIDATES)
        user_col = _autodetect_column(first, _USER_COLUMN_CANDIDATES)

        if text_col is None:
            raise ValueError(
                f"Could not auto-detect text column in {self.dataset_name}; "
                f"available keys: {list(first.keys())}. "
                f"Pass text_column= explicitly."
            )

        # We already consumed `first`; chain it back in
        from itertools import chain
        full_stream = chain([first], ds)

        emitted = 0
        for row in full_stream:
            if max_records is not None and emitted >= max_records:
                return

            raw_text = row.get(text_col)
            if not isinstance(raw_text, str):
                continue
            text = normalize_text(raw_text)
            if not text:
                continue

            if keywords and not any(k in text for k in keywords):
                continue

            rating = _normalize_rating(row.get(rating_col) if rating_col else None,
                                        self.rating_scale)
            user_raw = str(row.get(user_col)) if user_col and row.get(user_col) else None

            yield ReviewRecord(
                review_id=make_review_id(self.source_name, brand, text),
                source=self.source_name,
                brand=brand,
                text=text,
                rating=rating,
                user_id_hashed=hash_user_id(user_raw),
                metadata={
                    "hf_dataset": self.dataset_name,
                    "hf_split": self.split,
                    "hf_row_keys": list(row.keys()),
                },
            )
            emitted += 1


def _autodetect_column(row: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    lower_keys = {k.lower(): k for k in row.keys()}
    for cand in candidates:
        if cand.lower() in lower_keys:
            return lower_keys[cand.lower()]
    return None


def _normalize_rating(value: Any, scale: tuple[float, float]) -> float | None:
    """Coerce a raw rating value into a 1-5 Likert.

    For binary sentiment (0/1), maps to 1.5 / 4.5.
    For 0-1 floats (sentiment probs), linear-scales.
    For 1-5 already, passes through.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:                # NaN
        return None
    lo, hi = scale
    if hi <= lo:
        return None
    # If looks like binary {0, 1}
    if v in (0, 0.0):
        return 1.5
    if v in (1, 1.0) and hi <= 1.0:
        return 4.5
    # If looks like 1-5 already, pass through
    if 1 <= v <= 5:
        return round(v, 1)
    # General scaling to 1-5
    scaled = 1 + 4 * (v - lo) / (hi - lo)
    return round(max(1.0, min(5.0, scaled)), 1)
