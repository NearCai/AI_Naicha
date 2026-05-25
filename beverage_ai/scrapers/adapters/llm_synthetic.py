"""LLM-generated synthetic reviews — Path A source #6 (alternative to scraping).

Generates high-quality, diverse Chinese tea-drink reviews via Claude API
(Haiku for cost). Each API call produces a batch of N reviews in one
structured-JSON response.

Provenance is unambiguously labeled (`source=llm_synthetic:<model>`) so the
data is never confused with real consumer reviews. Useful for:
  - Padding Path A toward 50K target when real-data routes are short
  - GNN Stage 1 pretraining (the LANGUAGE of sensory descriptions transfers)
  - Validating the aspect extractor before spending money on real data

Cost reality (Claude Haiku, ~2025-Q2 pricing):
  - ~$0.002 per generation call (input + output)
  - Each call returns 5-10 reviews
  - 5,000 reviews → ~500 calls → ~$1-3

NOT a substitute for real data when validating real consumer preferences.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable

from ..base import ReviewRecord, make_review_id, normalize_text

# 8 mainstream brands × ~6 SKU each = 48 product variations
_BRAND_SKUS = [
    ("喜茶", ["多肉葡萄", "芝士绿妍", "鸭屎香拿铁", "桂花乌龙轻乳茶", "酪酪桃桃", "椰椰芒芒"]),
    ("奈雪", ["霸气芝士葡萄", "宝藏茶", "金色山脉烤奶", "鸭屎香奶茶", "霸气橙柚子", "鹰嘴豆拿铁"]),
    ("茶颜悦色", ["幽兰拿铁", "声声乌龙", "桂花弄", "蔓越阑珊", "栀晓"]),
    ("蜜雪冰城", ["珍珠奶茶", "棒打鲜橙", "柠檬水", "草莓圣代", "摩奇奇"]),
    ("书亦烧仙草", ["招牌烧仙草", "芋圆奶茶", "波波鲜奶", "桂花酒酿"]),
    ("古茗", ["超A芝士葡萄", "茉莉鲜奶茶", "杨枝甘露", "云岭茉莉"]),
    ("霸王茶姬", ["伯牙绝弦", "桂馥兰香", "万里木兰", "幻紫青莲"]),
    ("茶百道", ["桃豆冰沙", "鸭屎香奶茶", "茉莉奶绿", "杨梅西番莲"]),
]

_DEFAULT_BATCH_SIZE = 8
_HAIKU_COST_PER_CALL = 0.002       # rough estimate


_SYSTEM_PROMPT = """你是一个茶饮评论生成器。给定一组(品牌, SKU, 期望情感)输入,生成对应数量的中文奶茶/茶饮短评。

要求:
1. 每条评论用第一人称, 长度 30-120 字
2. 评论里必须出现至少一个明确的感官描述 (甜度/苦度/茶香/奶香/果香/顺滑/涩/咸/油腻/层次/清新/浓郁等)
3. 多数评论提到糖度档位或冰量等定制 (无糖/三分/五分/七分/全糖, 去冰/少冰/正常冰/多冰)
4. 多样化用词, 不要每条都说"好喝"或"难喝", 用具体描述
5. 情感强度匹配输入: positive_strong / positive / neutral / negative / negative_strong
6. 风格类似真实消费者短评 (小红书/大众点评风格), 偶尔出现表情或标点重复

严格按下面 JSON Schema 输出, 不要任何额外说明:
{
  "reviews": [
    {"brand": "...", "sku": "...", "sentiment": "positive_strong|positive|neutral|negative|negative_strong",
     "text": "...", "customization": "...(可选)", "rating": 1.0-5.0}
  ]
}"""


class LLMSyntheticScraper:
    source_name = "llm_synthetic"

    def __init__(
        self,
        *,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_tokens: int = 2000,
        cost_ceiling_usd: float | None = None,
        seed: int = 42,
    ):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package required. Install with: pip install -e '.[llm]'"
            ) from e
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.cost_ceiling_usd = cost_ceiling_usd
        self.source_name = f"llm_synthetic:{model}"
        self._rng = _StableRNG(seed)
        self.total_cost_usd = 0.0

    def scrape(
        self,
        *,
        keywords: list[str] | None = None,
        brand: str | None = None,
        max_records: int | None = None,
    ) -> Iterable[ReviewRecord]:
        target = max_records or 100
        emitted = 0
        consecutive_empty_batches = 0
        MAX_EMPTY_BATCHES = 5      # bail out if API keeps returning unusable rows

        while emitted < target:
            if (self.cost_ceiling_usd is not None
                    and self.total_cost_usd >= self.cost_ceiling_usd):
                return
            if consecutive_empty_batches >= MAX_EMPTY_BATCHES:
                return

            batch_specs = self._build_batch(brand)
            emitted_before = emitted
            try:
                batch = self._generate_batch(batch_specs)
            except Exception:
                # Skip this batch on failure; loop will try again
                continue

            for item in batch:
                if emitted >= target:
                    return
                rec = self._item_to_record(item)
                if rec is None:
                    continue
                yield rec
                emitted += 1

            # Bail-out guard against infinite loops when API keeps returning
            # batches that are all rejected (e.g. always-short text).
            if emitted == emitted_before:
                consecutive_empty_batches += 1
            else:
                consecutive_empty_batches = 0

    # --------------------- internals ---------------------

    def _build_batch(self, brand_filter: str | None) -> list[dict]:
        """Pick a random batch of (brand, sku, sentiment) specs to generate."""
        sentiments = (
            "positive_strong", "positive_strong", "positive",
            "neutral",
            "negative", "negative_strong",
        )
        specs = []
        pool = _BRAND_SKUS if not brand_filter else [
            (b, skus) for b, skus in _BRAND_SKUS if b == brand_filter
        ]
        if not pool:
            pool = _BRAND_SKUS
        for _ in range(self.batch_size):
            b, skus = self._rng.choice(pool)
            sku = self._rng.choice(skus)
            sent = self._rng.choice(sentiments)
            specs.append({"brand": b, "sku": sku, "sentiment": sent})
        return specs

    def _generate_batch(self, specs: list[dict]) -> list[dict]:
        user_msg = (
            "请生成以下评论:\n"
            + json.dumps({"requests": specs}, ensure_ascii=False)
        )
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        self.total_cost_usd += _HAIKU_COST_PER_CALL
        text = resp.content[0].text
        text = _strip_fences(text)
        payload = json.loads(text)
        return payload.get("reviews", [])

    def _item_to_record(self, item: dict) -> ReviewRecord | None:
        text = normalize_text(item.get("text") or "")
        if not text or len(text) < 15:
            return None
        brand = item.get("brand")
        sku = item.get("sku")
        rating = item.get("rating")
        try:
            rating = round(max(1.0, min(5.0, float(rating))), 1) if rating is not None else None
        except (TypeError, ValueError):
            rating = None
        return ReviewRecord(
            review_id=make_review_id(self.source_name, brand, text),
            source=self.source_name,
            brand=brand,
            sku=sku,
            text=text,
            customization_raw=item.get("customization"),
            rating=rating,
            metadata={
                "synthetic": True,
                "llm_model": self.model,
                "sentiment_target": item.get("sentiment"),
            },
        )


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text


class _StableRNG:
    """Deterministic-by-seed pseudo-RNG using hashlib (no numpy dep needed)."""

    def __init__(self, seed: int):
        self._counter = 0
        self._seed = seed

    def _next(self) -> int:
        self._counter += 1
        h = hashlib.sha256(f"{self._seed}|{self._counter}".encode()).digest()
        return int.from_bytes(h[:8], "big")

    def choice(self, seq):
        return seq[self._next() % len(seq)]
