"""Extract customization choices from review text.

Per 技术方案书 §3.3.1 R8 fix:
Reviews often mention what the user actually selected (糖度, 冰量, 加料),
which is critical for tying the review back to the actual recipe.

Two-tier strategy:
  1. Regex parser — fast, deterministic, catches explicit mentions
  2. LLM fallback (in extractor) — for implicit / paraphrased cases
"""
from __future__ import annotations

import re

from .schema import Customization, CupSizeLabel, IceLevel, SugarLevel

# Sugar mentions: "三分糖", "三分", "去糖", "无糖", "100%糖"
_SUGAR_PATTERNS: list[tuple[str, SugarLevel]] = [
    (r"无糖|不加糖|去糖|0%糖|0糖", "无糖"),
    (r"三分糖?|30%糖|少甜|微糖", "三分"),
    (r"半糖|五分糖?|50%糖", "五分"),
    (r"七分糖?|70%糖", "七分"),
    (r"全糖|正常糖|100%糖|标准糖|标准甜度", "全糖"),
]

_ICE_PATTERNS: list[tuple[str, IceLevel]] = [
    (r"去冰|不加冰|无冰|常温|温的", "无冰"),
    (r"少冰|微冰", "少冰"),
    (r"正常冰|标准冰|标冰", "正常"),
    (r"多冰|加冰|去冰\*0|双倍冰", "多冰"),
]

_SIZE_PATTERNS: list[tuple[str, CupSizeLabel]] = [
    (r"小杯|small", "小"),
    (r"中杯|standard|standard cup", "中"),
    (r"大杯|large", "大"),
]

# Common toppings — keep this list aligned with vocab `topping_*`
_TOPPING_KEYWORDS = {
    "黑糖珍珠": ["黑糖珍珠", "黑珍珠"],
    "白珍珠": ["白珍珠"],
    "珍珠": ["珍珠", "波霸"],
    "芋圆": ["芋圆", "紫薯圆", "地瓜圆", "五彩芋圆", "三色芋圆"],
    "芋泥": ["芋泥", "紫薯泥"],
    "椰果": ["椰果"],
    "仙草": ["仙草", "烧仙草"],
    "红豆": ["红豆", "蜜豆", "蜜红豆"],
    "布丁": ["布丁", "鸡蛋布丁", "焦糖布丁", "抹茶布丁"],
    "爆爆珠": ["爆爆珠", "啵啵"],
    "西米": ["西米", "西米露"],
    "燕麦": ["燕麦"],
}


def parse_customization_regex(text: str) -> Customization:
    """Regex-only customization extraction.

    Catches explicit mentions; returns Customization with `None` fields
    where nothing was matched (so an LLM can backfill).
    """
    sugar = _first_match(text, _SUGAR_PATTERNS)
    ice = _first_match(text, _ICE_PATTERNS)
    size = _first_match(text, _SIZE_PATTERNS)

    toppings: list[str] = []
    for canonical, variants in _TOPPING_KEYWORDS.items():
        if any(v in text for v in variants):
            toppings.append(canonical)
    # Dedup but preserve order
    seen = set()
    toppings = [t for t in toppings if not (t in seen or seen.add(t))]

    return Customization(
        sugar_level=sugar, ice_level=ice, toppings=toppings, size=size
    )


def _first_match(text: str, patterns):
    for pattern, label in patterns:
        if re.search(pattern, text):
            return label
    return None


class CustomizationParser:
    """Stateful wrapper around the regex parser, with an optional LLM fallback.

    For v1 we use regex only. The LLM fallback hook is a clean place to plug in
    a second pass when regex returns mostly None fields.
    """

    def __init__(self, llm_fallback: bool = False):
        self.llm_fallback = llm_fallback

    def parse(self, text: str) -> Customization:
        return parse_customization_regex(text)
