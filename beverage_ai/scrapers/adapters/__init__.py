"""Source-specific scraper adapters.

Built-in adapters (always available):
  - MockScraper       — synthetic reviews for tests and demos
  - LocalFileScraper  — read from CSV/JSON files

Real-source stubs (require [scrape] extras, manual selector maintenance):
  - DianpingScraper      — see SCRAPING_NOTICE.md
  - XiaohongshuScraper   — see SCRAPING_NOTICE.md
"""

from .local_file import LocalFileScraper
from .mock import MockScraper

__all__ = ["MockScraper", "LocalFileScraper"]
