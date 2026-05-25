"""LLM Planner — parse natural-language requests into structured spec.

Per 技术方案书 §3.1 and v1 实现方案 §6.10.

Two implementations:
  * LLMPlanner       — real Anthropic Claude (requires anthropic + API key)
  * MockLLMPlanner   — keyword-driven, used in tests and when no API key set
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

PLANNER_SCHEMA = {
    "type": "object",
    "required": ["style_hint", "cup_volume_ml", "sugar_level", "health"],
    "properties": {
        "style_hint": {
            "enum": ["纯茶", "奶茶", "果茶", "咖啡奶茶", "冰沙", "特调"],
        },
        "cup_volume_ml": {"enum": [380, 500, 700]},
        "sugar_level": {
            "enum": ["无糖", "三分", "五分", "七分", "全糖"],
        },
        "health": {
            "type": "object",
            "properties": {
                "sugar_limit_g": {"type": "number"},
                "calorie_limit_kcal": {"type": "number"},
                "caffeine_limit_mg": {"type": "number"},
                "trans_fat_zero": {"type": "boolean"},
                "excluded_allergens": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "context": {
            "type": "object",
            "properties": {
                "season": {
                    "enum": ["spring", "summer", "autumn", "winter"],
                },
                "target_age": {"enum": ["youth", "mature"]},
                "health_strict": {"type": "boolean"},
            },
        },
        "flavor_keywords": {"type": "array", "items": {"type": "string"}},
        "price_range_cny": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
        },
    },
}

_SYSTEM_PROMPT = (
    "你是新式茶饮的产品研发助手。把用户的自然语言需求转成结构化的 JSON 目标。"
    "严格按下面的 JSON Schema 输出, 不要任何额外文字、解释、markdown 包裹。\n"
    f"Schema:\n{json.dumps(PLANNER_SCHEMA, ensure_ascii=False)}"
)


class PlannerInterface(Protocol):
    def plan(self, user_request: str) -> dict[str, Any]: ...


# ---------- Mock implementation (keyword-based) ----------


class MockLLMPlanner:
    """Keyword-based planner — runs without any API key.

    Recognizes common Chinese keywords (季节/糖度/健康约束/风格) and
    populates the JSON spec. Good enough for tests and local demo;
    swap in LLMPlanner for production.
    """

    SEASON_KW = {
        "夏": "summer", "春": "spring", "秋": "autumn", "冬": "winter",
    }
    SUGAR_KW = {
        "无糖": "无糖", "去糖": "无糖",
        "三分": "三分", "少甜": "三分",
        "半甜": "五分", "五分": "五分",
        "七分": "七分",
        "正常甜": "全糖", "全糖": "全糖",
    }
    STYLE_KW = {
        "纯茶": "纯茶",
        "果茶": "果茶", "水果茶": "果茶",
        "奶茶": "奶茶",
        "咖啡": "咖啡奶茶",
        "冰沙": "冰沙",
        "特调": "特调",
    }
    YOUTH_KW = ["年轻", "学生", "女生", "Z世代"]
    MATURE_KW = ["成熟", "上班族", "中年"]
    HEALTH_KW = ["健康", "轻负担", "低卡", "低糖", "控糖", "无负担"]

    def plan(self, user_request: str) -> dict[str, Any]:
        text = user_request
        text_lower = text.lower()

        # Style detection
        style = "奶茶"
        for kw, st in self.STYLE_KW.items():
            if kw in text:
                style = st
                break

        # Cup volume
        cup = 500
        m = re.search(r"(380|500|700)\s*ml", text_lower)
        if m:
            cup = int(m.group(1))

        # Sugar
        sugar_level = "五分"
        for kw, sl in self.SUGAR_KW.items():
            if kw in text:
                sugar_level = sl
                break

        # Season
        season = None
        for kw, s in self.SEASON_KW.items():
            if kw in text:
                season = s
                break

        # Target age
        target_age = None
        if any(k in text for k in self.YOUTH_KW):
            target_age = "youth"
        elif any(k in text for k in self.MATURE_KW):
            target_age = "mature"

        # Health
        is_strict = any(k in text for k in self.HEALTH_KW)
        sugar_limit = 15.0 if is_strict else 30.0
        calorie_limit = 200.0 if is_strict else 400.0

        # Flavor keywords (heuristic noun extraction)
        flavor_keywords: list[str] = []
        for tag in ["花香", "果香", "厚乳", "茶香", "奶香", "清爽",
                    "微甜", "鲜果", "桂花", "茉莉", "葡萄", "芒果",
                    "草莓", "蓝莓", "百香果", "柠檬", "海盐", "焦糖"]:
            if tag in text:
                flavor_keywords.append(tag)

        # Price range
        price_range = None
        m = re.search(r"(\d+)\s*[-—]\s*(\d+)\s*元", text)
        if m:
            price_range = [float(m.group(1)), float(m.group(2))]

        spec: dict[str, Any] = {
            "style_hint": style,
            "cup_volume_ml": cup,
            "sugar_level": sugar_level,
            "health": {
                "sugar_limit_g": sugar_limit,
                "calorie_limit_kcal": calorie_limit,
                "caffeine_limit_mg": 200.0,
                "trans_fat_zero": is_strict,
                "excluded_allergens": [],
            },
            "context": {},
            "flavor_keywords": flavor_keywords,
        }
        if season:
            spec["context"]["season"] = season
        if target_age:
            spec["context"]["target_age"] = target_age
        if is_strict:
            spec["context"]["health_strict"] = True
        if price_range:
            spec["price_range_cny"] = price_range
        return spec


# ---------- Real implementation ----------


class LLMPlanner:
    """Real Claude-based planner."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package required. Install with: pip install -e .[llm]"
            ) from e
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def plan(self, user_request: str) -> dict[str, Any]:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_request}],
        )
        text = msg.content[0].text.strip()
        # Strip markdown fences if model returns them
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        return json.loads(text)


def get_default_planner() -> PlannerInterface:
    """Return real planner if API key + env flag set, otherwise mock."""
    use_real = (
        os.environ.get("ANTHROPIC_API_KEY")
        and os.environ.get("BEVERAGE_AI_USE_REAL_LLM", "0") == "1"
    )
    if use_real:
        try:
            return LLMPlanner()
        except ImportError:
            pass
    return MockLLMPlanner()
