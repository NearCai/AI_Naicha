"""Review scraping infrastructure — Path A data ingestion.

Per 技术方案书 §3.3.1 路径 A: 大规模噪声标注 (评论数据).
Target: 5–10 万条 reviews → LLM aspect extraction → GNN Stage 1 pretraining.

⚠️  See `docs/SCRAPING_NOTICE.md` for ToS / legal considerations.
For local development and tests, prefer `MockScraper` or `LocalFileScraper`.
"""

from .base import BaseScraper, ReviewRecord
from .runner import ScrapeRunner
from .store import RawReviewStore

__all__ = ["BaseScraper", "ReviewRecord", "ScrapeRunner", "RawReviewStore"]
