"""Feedback persistence — DuckDB.

Per 技术方案书 §3.7 + v1 实现方案 §6.11.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from ..recipes.schema import Recipe

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    session_id   VARCHAR,
    recipe_id    VARCHAR,
    recipe_json  JSON,
    pred_json    JSON,
    actual_json  JSON,
    context_json JSON,
    ts           TIMESTAMP
);

CREATE TABLE IF NOT EXISTS panel_score (
    session_id  VARCHAR,
    recipe_id   VARCHAR,
    panelist_id VARCHAR,
    dimension   VARCHAR,
    score       SMALLINT,
    cup_order   SMALLINT,
    block       SMALLINT,
    session_dt  TIMESTAMP
);
"""


class FeedbackRecorder:
    def __init__(self, db_path: str | Path = "data/feedback.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.db_path))
        self.con.execute(_SCHEMA)

    def record_recipe(
        self,
        session_id: str,
        recipe: Recipe,
        predicted: dict[str, Any],
        context: dict | None = None,
    ) -> None:
        self.con.execute(
            "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                session_id,
                recipe.recipe_id,
                recipe.model_dump_json(),
                json.dumps(predicted, ensure_ascii=False, default=_json_default),
                None,
                json.dumps(context or {}, ensure_ascii=False, default=_json_default),
                datetime.now(UTC),
            ],
        )

    def record_actual(
        self,
        session_id: str,
        recipe_id: str,
        actual: dict[str, Any],
    ) -> None:
        self.con.execute(
            "UPDATE feedback SET actual_json = ? WHERE session_id = ? AND recipe_id = ?",
            [json.dumps(actual, ensure_ascii=False, default=_json_default), session_id, recipe_id],
        )

    def record_panel(
        self,
        session_id: str,
        recipe_id: str,
        panelist_id: str,
        dimension: str,
        score: int,
        cup_order: int = 0,
        block: int = 0,
    ) -> None:
        self.con.execute(
            "INSERT INTO panel_score VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                session_id, recipe_id, panelist_id, dimension,
                int(score), int(cup_order), int(block),
                datetime.now(UTC),
            ],
        )

    def list_sessions(self) -> list[str]:
        rows = self.con.execute(
            "SELECT DISTINCT session_id FROM feedback ORDER BY session_id"
        ).fetchall()
        return [r[0] for r in rows]

    def get_recipes(self, session_id: str) -> list[tuple[str, dict]]:
        rows = self.con.execute(
            "SELECT recipe_id, recipe_json FROM feedback WHERE session_id = ?",
            [session_id],
        ).fetchall()
        return [(r[0], json.loads(r[1])) for r in rows]

    def get_panel_scores(self, session_id: str):
        return self.con.execute(
            "SELECT * FROM panel_score WHERE session_id = ?", [session_id]
        ).fetchall()

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> FeedbackRecorder:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _json_default(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)
