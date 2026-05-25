"""AspectExtractor — extracts sensory + customization from a single review.

Two implementations:
  - MockAspectExtractor    keyword-based, runs anywhere, used in tests
  - ClaudeAspectExtractor  real Anthropic Claude (Haiku by default)

Pick via `get_default_extractor()` (env vars decide).
"""
from __future__ import annotations

import json
import os
import re
from typing import Protocol

from ..scrapers.base import ReviewRecord
from .customization import parse_customization_regex
from .schema import ALL_DIMS, CORE_DIMS, Customization, ExtractedAspects


class AspectExtractor(Protocol):
    version: str

    def extract(self, review: ReviewRecord) -> ExtractedAspects: ...


# -------------------------------------------------------------------- prompts


def build_system_prompt(include_extended: bool = True) -> str:
    dims_to_extract = CORE_DIMS if not include_extended else ALL_DIMS
    dim_list = "、".join(dims_to_extract)
    return (
        "你是新式茶饮评论的感官分析专家。给定一条用户评论, 抽取以下感官维度的"
        f"评分(0.0–1.0, null 表示评论中未提及): {dim_list}。\n\n"
        "评分规则:\n"
        "- 0.0 = 完全没有 / 极轻; 1.0 = 极强烈; 0.5 = 中等\n"
        "- 喜爱度: 0.0 = 极差, 1.0 = 极喜欢, 0.5 = 中性\n"
        "- 评论中没有明确提到的维度, 必须输出 null, 不要凭空猜\n\n"
        "同时抽取用户的定制选择:\n"
        "- sugar_level: 无糖 / 三分 / 五分 / 七分 / 全糖, 或 null\n"
        "- ice_level: 无冰 / 少冰 / 正常 / 多冰, 或 null\n"
        "- toppings: 评论中提到的加料的列表 (中文短语)\n"
        "- size: 小 / 中 / 大, 或 null\n\n"
        "并给出 confidence (0.0–1.0): 你对本次抽取的信心。\n\n"
        "严格输出 JSON, 不要任何额外说明或 markdown 包裹。"
    )


def build_user_prompt(review: ReviewRecord) -> str:
    ctx = []
    if review.brand:
        ctx.append(f"品牌: {review.brand}")
    if review.sku:
        ctx.append(f"SKU: {review.sku}")
    header = " / ".join(ctx)
    return (f"{header}\n评论: {review.text}" if header else f"评论: {review.text}")


# -------------------------------------------------------------------- helpers


def _safe_score(x) -> float | None:
    """Coerce x into a [0,1] float, or None."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return max(0.0, min(1.0, v))


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text


def parse_llm_payload(raw: str) -> dict:
    """Best-effort JSON extraction from an LLM response.

    Tolerates accidental markdown fences and stray prose around the JSON.
    """
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Find the first {...} block as fallback
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def payload_to_aspects(
    payload: dict,
    review_id: str,
    version: str,
    raw_response: str | None,
    cost_usd: float,
) -> ExtractedAspects:
    aspects_in = (payload.get("aspects") or payload) or {}
    aspects: dict[str, float | None] = {}
    for dim in ALL_DIMS:
        aspects[dim] = _safe_score(aspects_in.get(dim))

    custom_in = payload.get("customization") or {}
    custom = Customization(
        sugar_level=custom_in.get("sugar_level"),
        ice_level=custom_in.get("ice_level"),
        toppings=list(custom_in.get("toppings") or []),
        size=custom_in.get("size"),
    )
    confidence = _safe_score(payload.get("confidence")) or 0.5

    return ExtractedAspects(
        review_id=review_id,
        extractor_version=version,
        aspects=aspects,
        customization=custom,
        confidence=confidence,
        raw_response=raw_response,
        cost_estimate_usd=cost_usd,
    )


# -------------------------------------------------------------------- mock impl


class MockAspectExtractor:
    """Keyword-based extractor. Realistic enough to test the pipeline.

    Maps Chinese sensory phrases to 0-1 scores. Unmentioned dims → None.
    Customization comes from `parse_customization_regex`.
    """

    version = "mock|kw_v1"

    # Cues for each dim — (pattern, score)
    _CUES: dict[str, list[tuple[str, float]]] = {
        "甜度": [
            (r"特别甜|齁甜|过甜|甜得发齁", 0.95),
            (r"很甜|偏甜|甜腻", 0.85),
            (r"甜度刚好|微甜|甜得正好", 0.55),
            (r"少甜|不太甜|甜得淡", 0.35),
            (r"无糖|不甜|没有糖", 0.05),
        ],
        "苦度": [
            (r"特别苦|苦得发涩", 0.9),
            (r"苦|发苦|后味发苦", 0.7),
            (r"略苦|微苦", 0.4),
            (r"不苦", 0.1),
        ],
        "茶香": [
            (r"茶味很浓|茶香扑鼻|茶味厚重|压得住", 0.9),
            (r"茶味浓|茶香足|茶味突出", 0.75),
            (r"茶味淡|茶味寡|茶味不足|喝不出茶", 0.25),
        ],
        "奶香": [
            (r"奶香足|奶味浓|厚乳很惊艳", 0.85),
            (r"奶感顺滑|奶味顺", 0.7),
            (r"奶味假|奶味淡|植脂末", 0.2),
        ],
        "喜爱度": [
            (r"无脑回购|强烈推荐|必喝|爱了|超好喝", 0.95),
            (r"好喝|推荐|不错|可以回购|惊艳", 0.8),
            (r"一般|普通|见仁见智|尚可", 0.5),
            (r"踩雷|不推荐|不会回购|难喝|失望", 0.15),
        ],
        "涩": [
            (r"特别涩|涩得难受", 0.9),
            (r"涩|有涩感", 0.6),
            (r"不涩", 0.1),
        ],
        "酸": [
            (r"特别酸|酸得很", 0.9),
            (r"酸|微酸|带酸味", 0.55),
            (r"不酸", 0.1),
        ],
        "回甘": [
            (r"回甘明显|回甘很好|回味甘", 0.85),
            (r"有回甘|微微回甘", 0.6),
        ],
        "顺滑": [
            (r"特别顺滑|顺滑不腻", 0.9),
            (r"顺滑|顺", 0.7),
            (r"颗粒感|不顺", 0.25),
        ],
        "果香": [
            (r"果香浓|果味足|新鲜水果", 0.85),
            (r"果香|果味", 0.6),
        ],
        "咸": [
            (r"咸|海盐|偏咸", 0.7),
        ],
        "油腻": [
            (r"油腻|喝了腻|发腻", 0.75),
            (r"不腻|清爽不腻", 0.15),
        ],
        "清新": [
            (r"清新|清爽|轻盈", 0.8),
        ],
        "浓郁": [
            (r"浓郁|浓厚|厚实", 0.85),
        ],
        "层次": [
            (r"层次丰富|有层次", 0.8),
            (r"层次单一|没层次", 0.2),
        ],
    }

    def extract(self, review: ReviewRecord) -> ExtractedAspects:
        text = review.text
        aspects: dict[str, float | None] = {dim: None for dim in ALL_DIMS}
        n_matches = 0
        for dim, cues in self._CUES.items():
            for pattern, score in cues:
                if re.search(pattern, text):
                    aspects[dim] = score
                    n_matches += 1
                    break

        customization = parse_customization_regex(text)
        # Confidence grows with explicit signal density (cap at 0.9)
        confidence = min(0.4 + 0.05 * n_matches, 0.9)
        return ExtractedAspects(
            review_id=review.review_id,
            extractor_version=self.version,
            aspects=aspects,
            customization=customization,
            confidence=confidence,
            raw_response=None,
            cost_estimate_usd=0.0,
        )


# -------------------------------------------------------------------- claude impl


# Rough Anthropic Haiku pricing (input + output, per call, ~500 token system + ~150 token user + ~250 token output)
# Update from current price list when running for real.
_HAIKU_USD_PER_CALL = 0.002


class ClaudeAspectExtractor:
    """Real Anthropic Claude extractor.

    Requires `anthropic` package + ANTHROPIC_API_KEY. By default uses Haiku
    for cost. Will raise ImportError early if anthropic isn't installed.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        include_extended_dims: bool = True,
        max_tokens: int = 800,
    ):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic required. Install with: pip install -e '.[llm]'"
            ) from e
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.include_extended_dims = include_extended_dims
        self.max_tokens = max_tokens
        self._system_prompt = build_system_prompt(include_extended_dims)
        self.version = f"{model}|p_v1{'_ext' if include_extended_dims else ''}"

    def extract(self, review: ReviewRecord) -> ExtractedAspects:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._system_prompt,
            messages=[{"role": "user", "content": build_user_prompt(review)}],
        )
        raw = msg.content[0].text
        try:
            payload = parse_llm_payload(raw)
        except json.JSONDecodeError:
            # Fall back to empty/default aspect record so the pipeline survives
            return ExtractedAspects(
                review_id=review.review_id,
                extractor_version=self.version,
                aspects={dim: None for dim in ALL_DIMS},
                customization=parse_customization_regex(review.text),
                confidence=0.0,
                raw_response=raw,
                cost_estimate_usd=_HAIKU_USD_PER_CALL,
            )
        # Also merge in regex customization as a safety net
        out = payload_to_aspects(
            payload=payload,
            review_id=review.review_id,
            version=self.version,
            raw_response=raw,
            cost_usd=_HAIKU_USD_PER_CALL,
        )
        if not out.customization.toppings and not out.customization.sugar_level:
            out.customization = parse_customization_regex(review.text)
        return out


def get_default_extractor() -> AspectExtractor:
    """Use Claude if API key + flag set; else Mock."""
    use_real = (
        os.environ.get("ANTHROPIC_API_KEY")
        and os.environ.get("BEVERAGE_AI_USE_REAL_LLM", "0") == "1"
    )
    if use_real:
        try:
            return ClaudeAspectExtractor()
        except ImportError:
            pass
    return MockAspectExtractor()


# Re-export so external modules don't need to know about pydantic types
__all__ = [
    "AspectExtractor",
    "MockAspectExtractor",
    "ClaudeAspectExtractor",
    "ExtractedAspects",
    "get_default_extractor",
    "build_system_prompt",
    "build_user_prompt",
    "parse_llm_payload",
    "payload_to_aspects",
]

# Re-export schema for convenience
ExtractedAspects = ExtractedAspects
