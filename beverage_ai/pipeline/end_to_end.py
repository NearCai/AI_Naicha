"""End-to-end pipeline: user request → Top-K candidate recipes.

Per 技术方案书 §3 system architecture + v1 实现方案 §6.12.

Flow (single function, six stages):
    1. LLM Planner parses natural-language request
    2. Generator produces N seed candidates (warm start)
    3. Simulators score each candidate
    4. Pareto front extracted (non-dominated sort)
    5. MMR Top-K selection from Pareto front
    6. Persist to feedback DuckDB
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..feedback.recorder import FeedbackRecorder
from ..ingredients.vocab import Vocab, load_default_vocab
from ..optimizer.mmr import mmr_select
from ..optimizer.nsga2 import ScoredCandidate, pareto_front
from ..optimizer.problem import score_candidates
from ..planner.llm_planner import PlannerInterface, get_default_planner
from ..priors.engine import PriorEngine, load_default_engine
from ..recipes.generator import RecipeGenerator
from ..simulators.repurchase.v1_weighted import RepurchasePredictorV1
from ..simulators.sales.predict import MockSalesPredictor, SalesPredictor
from ..simulators.sensory.predict import MockSensoryPredictor, SensoryPredictor
from ..utils.logging import get_logger

logger = get_logger("pipeline")


@dataclass
class PipelineResult:
    session_id: str
    spec: dict[str, Any]
    top_recipes: list[dict[str, Any]]
    n_generated: int
    n_feasible: int
    n_pareto: int
    elapsed_sec: float

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "spec": self.spec,
            "top_recipes": self.top_recipes,
            "stats": {
                "n_generated": self.n_generated,
                "n_feasible": self.n_feasible,
                "n_pareto": self.n_pareto,
                "elapsed_sec": round(self.elapsed_sec, 2),
            },
        }


def run_pipeline(
    user_request: str,
    *,
    top_k: int = 5,
    n_candidates: int = 200,
    kappa: float = 1.0,
    mmr_lambda: float = 0.6,
    seed: int = 42,
    # Injectable for testing / production swap
    vocab: Vocab | None = None,
    prior: PriorEngine | None = None,
    planner: PlannerInterface | None = None,
    sensory: SensoryPredictor | None = None,
    sales: SalesPredictor | None = None,
    repurchase: RepurchasePredictorV1 | None = None,
    recorder: FeedbackRecorder | None = None,
    record: bool = False,
) -> PipelineResult:
    """Run end-to-end pipeline and return PipelineResult.

    All major components are injectable so tests can swap mocks
    and production can swap real GNN/LightGBM trained models.
    """
    t0 = time.time()

    vocab = vocab or load_default_vocab()
    prior = prior or load_default_engine()
    planner = planner or get_default_planner()
    sensory = sensory or MockSensoryPredictor(vocab, seed=seed)
    sales = sales or MockSalesPredictor(vocab, seed=seed)
    repurchase = repurchase or RepurchasePredictorV1()

    # ----- Stage 1: parse request -----
    spec = planner.plan(user_request)
    logger.info(f"planner returned: style={spec.get('style_hint')} "
                f"sugar={spec.get('sugar_level')}")

    # ----- Stage 2: generate seed population -----
    generator = RecipeGenerator(vocab, prior, seed=seed)
    seeds = generator.generate(spec, n_candidates=n_candidates)
    logger.info(f"generator produced {len(seeds)} unique candidates")

    # ----- Stage 3: score with simulators -----
    targets = spec.get("health", {})
    scored: list[ScoredCandidate] = score_candidates(
        recipes=seeds,
        vocab=vocab,
        sensory=sensory,
        sales=sales,
        repurchase=repurchase,
        targets=targets,
        kappa=kappa,
    )
    n_feasible = sum(1 for c in scored if c.feasible)
    logger.info(f"{n_feasible}/{len(scored)} candidates feasible")

    # ----- Stage 4: Pareto front -----
    pf = pareto_front(scored)
    logger.info(f"Pareto front size = {len(pf)}")

    if not pf:
        # No feasible candidates at all; fall back to highest-preference infeasible
        # (and surface this in result)
        logger.warning("No feasible candidates! Falling back to best-effort ranking")
        ranked = sorted(scored, key=lambda c: c.objectives[0])
        pf = ranked[: max(top_k * 2, 5)]

    # ----- Stage 5: MMR Top-K -----
    pf_scores = np.array([-c.objectives[0] for c in pf])    # higher preference = better
    pf_embs = np.stack([c.embedding for c in pf])
    top_idx = mmr_select(pf_scores, pf_embs, k=top_k, lam=mmr_lambda)
    top_candidates = [pf[i] for i in top_idx]

    # ----- Stage 6: persist (optional) -----
    session_id = f"s_{int(time.time())}"
    if record:
        rec = recorder or FeedbackRecorder()
        for c in top_candidates:
            rec.record_recipe(
                session_id=session_id,
                recipe=c.recipe,
                predicted={
                    "means": c.means,
                    "sigmas": c.sigmas,
                    "nutrition": c.nutrition,
                    "feasible": c.feasible,
                },
                context=spec.get("context"),
            )
        if recorder is None:
            rec.close()

    top_recipes_payload = [_candidate_to_payload(c) for c in top_candidates]

    return PipelineResult(
        session_id=session_id,
        spec=spec,
        top_recipes=top_recipes_payload,
        n_generated=len(seeds),
        n_feasible=n_feasible,
        n_pareto=len(pf),
        elapsed_sec=time.time() - t0,
    )


def _candidate_to_payload(c: ScoredCandidate) -> dict:
    return {
        "recipe": c.recipe.model_dump(),
        "means": c.means,
        "sigmas": c.sigmas,
        "nutrition": c.nutrition,
        "feasible": c.feasible,
    }
