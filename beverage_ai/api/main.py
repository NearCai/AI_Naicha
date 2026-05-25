"""FastAPI service exposing `run_pipeline()` to the Next.js frontend.

Per 茶饮研发闭环AI系统_v2前后端融合方案.md §6 + §9.1.

Run locally:
    uvicorn beverage_ai.api.main:app --reload --port 8000

Endpoints:
    GET  /api/v2/health             — liveness probe
    POST /api/v2/pipeline           — wrap run_pipeline(), return Pareto-front candidates
    POST /api/v2/feedback           — store frontend rating into DuckDB (dual-write)
    POST /api/v2/update             — trigger scripts/update_from_feedback.py, returns job_id
    GET  /api/v2/update/{job_id}    — poll job status + read audit JSON
    GET  /api/v2/update             — list recent update jobs
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as e:  # pragma: no cover - clear hint at import time
    raise ImportError(
        "FastAPI not installed. Install with: pip install -e .[api]"
    ) from e

from ..feedback.recorder import FeedbackRecorder
from ..ingredients.aliases import AliasResolver, load_default_aliases
from ..ingredients.vocab import Vocab, load_default_vocab
from ..pipeline.end_to_end import PipelineResult, run_pipeline
from ..recipes.schema import Recipe
from .jobs import build_update_cmd
from .jobs import registry as job_registry
from .translator import (
    display_to_vocab_ids,
    frontend_constraints_to_targets,
    merge_targets_into_spec,
    recipe_to_display,
)

logger = logging.getLogger("beverage_ai.api")

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------


class StoreIngredientIn(BaseModel):
    """Mirrors the frontend `StoreIngredient`. All fields optional —
    accepted so the frontend can forward its store profile unchanged."""

    id: str | None = None
    name: str | None = None
    category: str | None = None
    costPerUnit: str | None = None
    quantity: str | None = None
    flavorTags: list[str] | None = None
    allergens: list[str] | None = None
    availability: str | None = None
    equipment: list[str] | None = None


class GenerationConstraintsIn(BaseModel):
    season: str | None = None
    targetAudience: str | None = None
    priceBand: str | None = None
    maxIngredientCost: str | None = None
    maxMakeTime: str | None = None
    sweetness: str | None = None
    temperature: str | None = None


class PipelineRequest(BaseModel):
    prompt: str = Field(min_length=1)
    constraints: GenerationConstraintsIn | None = None
    availableIngredients: list[StoreIngredientIn] | None = None
    n_candidates: int = Field(default=30, ge=5, le=100)
    pool_size: int = Field(default=200, ge=20, le=1000)
    kappa: float = 1.0
    seed: int = 42


class NutritionOut(BaseModel):
    energy_kcal: float = 0.0
    sugar_g: float = 0.0
    fat_g: float = 0.0
    trans_fat_g: float = 0.0
    caffeine_mg: float = 0.0
    sodium_mg: float = 0.0


class IngredientDisplayOut(BaseModel):
    name: str
    amount: str


class CandidateOut(BaseModel):
    candidate_id: str
    style: str
    cup_volume_ml: int
    sugar_level: str
    ingredients: dict[str, float]
    ingredients_display: list[IngredientDisplayOut]
    predicted_preference: float
    preference_sigma: float
    sales_proxy_lcb: float
    sales_proxy_sigma: float
    cost_cny: float
    repurchase_score: float
    nutrition: NutritionOut
    feasibility: str
    constraint_notes: list[str] = Field(default_factory=list)


class PipelineStats(BaseModel):
    n_generated: int
    n_feasible: int
    n_pareto: int
    elapsed_sec: float


class PipelineResponse(BaseModel):
    session_id: str
    spec: dict
    stats: PipelineStats
    candidates: list[CandidateOut]


# ----- feedback -----


class FeedbackIngredientIn(BaseModel):
    name: str
    amount: str


class FeedbackRecipeIn(BaseModel):
    """Loose mirror of the frontend `DrinkRecipe`. All Chinese strings —
    we resolve `ingredients[].name` to vocab ids best-effort."""

    name: str
    description: str | None = None
    ingredients: list[FeedbackIngredientIn]
    steps: list[str] | None = None
    style: str | None = None
    cup_volume_ml: int | None = None
    sugar_level: str | None = None


class FeedbackRatingIn(BaseModel):
    score: float = Field(ge=0, le=10)
    panelist_id: str | None = None
    comment: str | None = None
    dimension: str = "喜爱度"


class FeedbackRequest(BaseModel):
    session_id: str | None = None
    recipe_id: str | None = None
    recipe: FeedbackRecipeIn
    feedbacks: list[FeedbackRatingIn] = Field(min_length=1)
    constraints: GenerationConstraintsIn | None = None
    library: str | None = None  # "recipe_skill_library" | "bad_recipe_skill_library"


class FeedbackResponse(BaseModel):
    ok: bool
    session_id: str
    recipe_id: str
    n_panel_rows: int
    resolved_ingredients: int
    unresolved_ingredients: list[str]
    recipe_persisted: bool


# ----- update -----


class UpdateRequest(BaseModel):
    session_id: str = Field(min_length=1)
    feedback_db: str | None = None
    stage2_epochs: int = Field(default=30, ge=1, le=300)
    dirichlet_lr: float = Field(default=0.3, gt=0, le=1.0)
    serving_top_quantile: float = Field(default=0.6, ge=0.1, le=0.95)
    skip_stage2: bool = True
    audit_dir: str | None = None
    note: str | None = None


class UpdateResponse(BaseModel):
    job_id: str
    status: str
    audit_path: str | None = None


class UpdateJobOut(BaseModel):
    job_id: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    returncode: int | None = None
    stdout_tail: list[str] = Field(default_factory=list)
    stderr_tail: list[str] = Field(default_factory=list)
    audit_path: str | None = None
    audit: dict | None = None  # parsed audit JSON, when available
    error: str | None = None
    note: str | None = None
    cmd: list[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------


def _parse_cors_origins(raw: str | None) -> list[str]:
    """Parse `BEVERAGE_AI_CORS_ORIGINS` (comma-separated) into a clean list.

    Empty / unset → empty list (caller falls back to the dev defaults).
    Use the literal value `"*"` to allow any origin (credentials forced off).
    """
    if not raw:
        return []
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]


def _build_app() -> FastAPI:
    app = FastAPI(
        title="beverage_ai v2 API",
        version="2.0.0",
        description="HTTP wrapper around run_pipeline() for the Next.js frontend.",
    )

    # CORS — configurable via env so the same image runs in dev and prod:
    #   BEVERAGE_AI_CORS_ORIGINS=https://app.example.com,https://admin.example.com
    #   BEVERAGE_AI_CORS_ORIGIN_REGEX=^https://([a-z0-9-]+\.)*example\.com$
    #   BEVERAGE_AI_CORS_ALLOW_CREDENTIALS=true|false (default: false)
    #
    # Defaults to localhost:3000 / 127.0.0.1:3000 only — never use the
    # default list in production behind a real domain.
    origins = _parse_cors_origins(os.environ.get("BEVERAGE_AI_CORS_ORIGINS"))
    origin_regex = os.environ.get("BEVERAGE_AI_CORS_ORIGIN_REGEX") or None
    allow_credentials = os.environ.get(
        "BEVERAGE_AI_CORS_ALLOW_CREDENTIALS", "false"
    ).lower() in {"1", "true", "yes"}

    if not origins and not origin_regex:
        # Dev fallback. Logged at startup so prod operators notice if they
        # forget to set the env var.
        origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
        logger.warning(
            "CORS using dev defaults (%s). Set BEVERAGE_AI_CORS_ORIGINS or "
            "BEVERAGE_AI_CORS_ORIGIN_REGEX for production.",
            origins,
        )

    wildcard = origins == ["*"]
    if wildcard and allow_credentials:
        # Browsers reject `Access-Control-Allow-Origin: *` together with
        # credentials. Force credentials off rather than ship a broken config.
        logger.warning(
            "CORS allow_origins=='*' is incompatible with allow_credentials=true; "
            "forcing credentials off."
        )
        allow_credentials = False

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=origin_regex,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    return app


app = _build_app()
_vocab_singleton: Vocab | None = None
_aliases_singleton: AliasResolver | None = None
_FEEDBACK_DB = os.environ.get("BEVERAGE_AI_FEEDBACK_DB", "data/feedback.duckdb")
_AUDIT_DIR = os.environ.get("BEVERAGE_AI_AUDIT_DIR", "data/feedback")


def _get_vocab() -> Vocab:
    global _vocab_singleton
    if _vocab_singleton is None:
        _vocab_singleton = load_default_vocab()
    return _vocab_singleton


def _get_aliases() -> AliasResolver:
    global _aliases_singleton
    if _aliases_singleton is None:
        _aliases_singleton = load_default_aliases(_get_vocab())
    return _aliases_singleton


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@app.get("/api/v2/health")
def health() -> dict[str, Any]:
    vocab = _get_vocab()
    return {
        "status": "ok",
        "version": "2.0.0",
        "vocab_size": len(vocab),
    }


@app.post("/api/v2/pipeline", response_model=PipelineResponse)
def pipeline(req: PipelineRequest) -> PipelineResponse:
    vocab = _get_vocab()
    targets = frontend_constraints_to_targets(
        req.constraints.model_dump() if req.constraints else None
    )

    try:
        result: PipelineResult = run_pipeline(
            user_request=req.prompt,
            top_k=req.n_candidates,
            n_candidates=req.pool_size,
            kappa=req.kappa,
            seed=req.seed,
            vocab=vocab,
        )
    except Exception as e:
        logger.exception("pipeline failed")
        raise HTTPException(status_code=500, detail=f"pipeline_error: {e}") from e

    merged_spec = merge_targets_into_spec(result.spec, targets)

    candidates: list[CandidateOut] = []
    for c in result.top_recipes:
        recipe = c["recipe"]
        means = c["means"]
        sigmas = c["sigmas"]
        nutrition = c["nutrition"]
        candidates.append(
            CandidateOut(
                candidate_id=recipe["recipe_id"],
                style=recipe["style"],
                cup_volume_ml=int(recipe["cup_volume_ml"]),
                sugar_level=recipe["sugar_level"],
                ingredients={k: float(v) for k, v in recipe["ingredients"].items()},
                ingredients_display=[
                    IngredientDisplayOut(**d) for d in recipe_to_display(recipe, vocab)
                ],
                predicted_preference=float(means.get("preference", 0.0)),
                preference_sigma=float(sigmas.get("preference", 0.0)),
                sales_proxy_lcb=float(means.get("sales_proxy", 0.0)),
                sales_proxy_sigma=float(sigmas.get("sales_proxy", 0.0)),
                cost_cny=float(means.get("cost_cny", 0.0)),
                repurchase_score=float(means.get("repurchase", 0.0)),
                nutrition=NutritionOut(
                    energy_kcal=float(nutrition.get("energy_kcal", 0.0)),
                    sugar_g=float(nutrition.get("sugar_g", 0.0)),
                    fat_g=float(nutrition.get("fat_g", 0.0)),
                    trans_fat_g=float(nutrition.get("trans_fat_g", 0.0)),
                    caffeine_mg=float(nutrition.get("caffeine_mg", 0.0)),
                    sodium_mg=float(nutrition.get("sodium_mg", 0.0)),
                ),
                feasibility="OK" if c["feasible"] else "VIOLATION",
                constraint_notes=[],
            )
        )

    return PipelineResponse(
        session_id=result.session_id,
        spec=merged_spec,
        stats=PipelineStats(
            n_generated=result.n_generated,
            n_feasible=result.n_feasible,
            n_pareto=result.n_pareto,
            elapsed_sec=round(result.elapsed_sec, 3),
        ),
        candidates=candidates,
    )


# -----------------------------------------------------------------------------
# Feedback (P1 #6) — frontend rate-drink dual-write target
# -----------------------------------------------------------------------------


def _score_010_to_likert_15(score_010: float) -> int:
    """0..10 (frontend) → 1..5 (Likert, what panel_score expects)."""
    likert = 1.0 + (max(0.0, min(10.0, score_010)) / 10.0) * 4.0
    return int(round(likert))


def _build_recipe_from_feedback(
    fb: FeedbackRecipeIn,
    vocab: Vocab,
    aliases: AliasResolver,
    recipe_id: str,
) -> tuple[Recipe | None, list[str]]:
    """Best-effort: resolve Chinese ingredient names → vocab ids and
    construct a schema-compliant Recipe.

    Returns (recipe_or_none, unresolved_names). The Recipe is None when
    too few ingredients resolve to make a usable record (< 1).
    """
    # Resolve each ingredient individually so we can record which input
    # names failed to map. (display_to_vocab_ids returns a dict keyed by
    # the *output* vocab_id, which loses the input-name → outcome link.)
    resolved: dict[str, float] = {}
    unresolved: list[str] = []
    for ing in fb.ingredients:
        single = display_to_vocab_ids(
            [{"name": ing.name, "amount": ing.amount}], vocab, aliases
        )
        if single:
            resolved.update(single)
        else:
            unresolved.append(ing.name)
    if not resolved:
        return None, unresolved
    valid_styles = {"纯茶", "奶茶", "果茶", "咖啡奶茶", "冰沙", "特调"}
    style = fb.style if fb.style in valid_styles else "奶茶"
    cup = fb.cup_volume_ml if fb.cup_volume_ml in (380, 500, 700) else 500
    valid_sweet = {"无糖", "三分", "五分", "七分", "全糖"}
    sweet = fb.sugar_level if fb.sugar_level in valid_sweet else "五分"
    recipe = Recipe(
        recipe_id=recipe_id,
        style=style,
        cup_volume_ml=cup,
        sugar_level=sweet,
        ingredients=resolved,
        metadata={
            "source": "frontend_feedback",
            "display_name": fb.name,
            "description": fb.description or "",
            "steps": fb.steps or [],
        },
    )
    return recipe, unresolved


@app.post("/api/v2/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest) -> FeedbackResponse:
    vocab = _get_vocab()
    aliases = _get_aliases()
    now = datetime.now(timezone.utc)
    session_id = req.session_id or f"user_{now.strftime('%Y%m%d')}"
    recipe_id = req.recipe_id or f"user_recipe_{int(now.timestamp())}"

    recipe, unresolved = _build_recipe_from_feedback(
        req.recipe, vocab, aliases, recipe_id
    )

    recorder = FeedbackRecorder(db_path=_FEEDBACK_DB)
    try:
        recipe_persisted = False
        if recipe is not None:
            recorder.record_recipe(
                session_id=session_id,
                recipe=recipe,
                predicted={
                    "library": req.library,
                    "user_avg_010": sum(f.score for f in req.feedbacks) / len(req.feedbacks),
                    "n_raters": len(req.feedbacks),
                },
                context={
                    "constraints": req.constraints.model_dump() if req.constraints else None,
                    "frontend_recipe_name": req.recipe.name,
                    "frontend_description": req.recipe.description,
                    "frontend_steps": req.recipe.steps,
                    "unresolved_ingredients": unresolved,
                },
            )
            recipe_persisted = True

        for idx, fb in enumerate(req.feedbacks):
            panelist = fb.panelist_id or f"anon_{idx}"
            likert = _score_010_to_likert_15(fb.score)
            recorder.record_panel(
                session_id=session_id,
                recipe_id=recipe_id,
                panelist_id=panelist,
                dimension=fb.dimension,
                score=likert,
                cup_order=idx,
                block=0,
            )
    except Exception as e:
        logger.exception("feedback persistence failed")
        recorder.close()
        raise HTTPException(status_code=500, detail=f"feedback_error: {e}") from e
    finally:
        try:
            recorder.close()
        except Exception:
            pass

    return FeedbackResponse(
        ok=True,
        session_id=session_id,
        recipe_id=recipe_id,
        n_panel_rows=len(req.feedbacks),
        resolved_ingredients=(len(recipe.ingredients) if recipe else 0),
        unresolved_ingredients=unresolved,
        recipe_persisted=recipe_persisted,
    )


# -----------------------------------------------------------------------------
# Update (P1 #7) — trigger scripts/update_from_feedback.py
# -----------------------------------------------------------------------------


def _project_root() -> Path:
    """Project root = directory that contains `scripts/` + `data/`."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "scripts").is_dir() and (p / "data").is_dir():
            return p
    return Path.cwd()


def _expected_audit_path(audit_dir: str, session_id: str) -> Path:
    return _project_root() / audit_dir / f"update_session_{session_id}.json"


@app.post("/api/v2/update", response_model=UpdateResponse)
def update(req: UpdateRequest) -> UpdateResponse:
    audit_dir = req.audit_dir or _AUDIT_DIR
    feedback_db = req.feedback_db or _FEEDBACK_DB
    cmd = build_update_cmd(
        session_id=req.session_id,
        feedback_db=feedback_db,
        stage2_epochs=req.stage2_epochs,
        dirichlet_lr=req.dirichlet_lr,
        serving_top_quantile=req.serving_top_quantile,
        skip_stage2=req.skip_stage2,
        audit_dir=audit_dir,
    )
    expected_audit = str(_expected_audit_path(audit_dir, req.session_id))
    job = job_registry.submit(
        cmd,
        cwd=_project_root(),
        audit_path=expected_audit,
        note=req.note or f"update session {req.session_id}",
    )
    return UpdateResponse(job_id=job.job_id, status=job.status, audit_path=expected_audit)


def _read_audit_json(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


@app.get("/api/v2/update/{job_id}", response_model=UpdateJobOut)
def update_status(job_id: str) -> UpdateJobOut:
    job = job_registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job_not_found: {job_id}")
    d = job.to_dict()
    audit = None
    if job.status == "completed":
        audit = _read_audit_json(job.audit_path)
    return UpdateJobOut(**{**d, "audit": audit})


@app.get("/api/v2/update", response_model=list[UpdateJobOut])
def update_list() -> list[UpdateJobOut]:
    out: list[UpdateJobOut] = []
    for job in sorted(
        job_registry.list(),
        key=lambda j: (j.started_at or datetime.min.replace(tzinfo=timezone.utc)),
        reverse=True,
    ):
        d = job.to_dict()
        audit = _read_audit_json(job.audit_path) if job.status == "completed" else None
        out.append(UpdateJobOut(**{**d, "audit": audit}))
    return out


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
