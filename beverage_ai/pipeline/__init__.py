"""End-to-end orchestration. Corresponds to v1 实现方案 §6.12."""

from .end_to_end import PipelineResult, run_pipeline

__all__ = ["run_pipeline", "PipelineResult"]
