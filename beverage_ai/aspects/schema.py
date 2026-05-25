"""Aspect dimensions and ExtractedAspects schema.

Aligned with 技术方案书 §3.3.1 双层输出头:
  - CORE_DIMS (5): trained on both path A and path C (shared)
  - EXT_DIMS (10): trained only on path A (frozen during stage 2)
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

CORE_DIMS = ("甜度", "苦度", "茶香", "奶香", "喜爱度")

EXT_DIMS = (
    "涩", "酸", "回甘", "顺滑", "果香",
    "咸", "油腻", "清新", "浓郁", "层次",
)

ALL_DIMS = (*CORE_DIMS, *EXT_DIMS)

SugarLevel = Literal["无糖", "三分", "五分", "七分", "全糖"]
IceLevel = Literal["无冰", "少冰", "正常", "多冰"]
CupSizeLabel = Literal["小", "中", "大"]


class Customization(BaseModel):
    sugar_level: SugarLevel | None = None
    ice_level: IceLevel | None = None
    toppings: list[str] = Field(default_factory=list)
    size: CupSizeLabel | None = None


class ExtractedAspects(BaseModel):
    """Result of extracting aspects from a single review."""

    review_id: str
    extractor_version: str           # e.g. "claude-haiku-4-5|p_v1" or "mock|kw_v1"
    aspects: dict[str, float | None] # dim → score in [0,1] or None if not mentioned
    customization: Customization = Field(default_factory=Customization)
    confidence: float = 0.5          # extractor's self-assessed confidence
    raw_response: str | None = None  # for auditing
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    cost_estimate_usd: float = 0.0
