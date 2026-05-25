"""LLM aspect extraction — turn raw review text into sensory scores.

Per 技术方案书 §3.3.1 路径 A: 用 LLM 抽取细粒度感官标签,作为感官 GNN Stage 1
预训练标签。这套模块负责:

  1. 解析单条评论 → (aspect_scores, customization) 的结构化结果
  2. self-consistency 投票降噪 (3 次抽取取中位数)
  3. DuckDB 缓存避免重复 API 调用
  4. 批处理 + 成本追踪
"""

from .cache import AspectCache
from .customization import CustomizationParser, parse_customization_regex
from .extractor import (
    AspectExtractor,
    ClaudeAspectExtractor,
    ExtractedAspects,
    MockAspectExtractor,
    get_default_extractor,
)
from .pipeline import AspectExtractionPipeline, ExtractionStats
from .schema import CORE_DIMS, EXT_DIMS

__all__ = [
    "AspectExtractor",
    "ClaudeAspectExtractor",
    "MockAspectExtractor",
    "get_default_extractor",
    "ExtractedAspects",
    "CustomizationParser",
    "parse_customization_regex",
    "AspectCache",
    "AspectExtractionPipeline",
    "ExtractionStats",
    "CORE_DIMS",
    "EXT_DIMS",
]
