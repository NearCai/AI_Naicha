"""Repurchase v1 — weighted formula.

Per 技术方案书 §3.3.3 v1: this is an honest "alias of liking score" until
real POS data is available. Output is in [0, 1].

    score = α * normalized_preference
          + β * review_intent_proxy
          + γ * social_decay_proxy

Default weights are calibrated against a small set of known products.
For v1, β and γ are stubbed (no real review/social data); only α active.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..sensory.predict import SensoryPrediction


@dataclass
class RepurchasePrediction:
    score: float        # in [0, 1]
    components: dict    # for transparency

    def to_dict(self) -> dict:
        return {"score": self.score, "components": dict(self.components)}


class RepurchasePredictorV1:
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.2,
        gamma: float = 0.1,
        review_intent_proxy: float = 0.5,
        social_decay_proxy: float = 0.5,
    ):
        if not (0.99 < alpha + beta + gamma < 1.01):
            raise ValueError("alpha + beta + gamma must sum to 1.0")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.review_intent = review_intent_proxy
        self.social_decay = social_decay_proxy

    def predict(self, sensory: SensoryPrediction) -> RepurchasePrediction:
        # Map 1-5 喜爱度 to [0, 1]
        liking = sensory.means.get("喜爱度", 3.0)
        liking_norm = (liking - 1.0) / 4.0
        liking_norm = max(0.0, min(1.0, liking_norm))

        score = (
            self.alpha * liking_norm
            + self.beta * self.review_intent
            + self.gamma * self.social_decay
        )
        return RepurchasePrediction(
            score=round(score, 3),
            components={
                "liking_norm": round(liking_norm, 3),
                "review_intent": self.review_intent,
                "social_decay": self.social_decay,
                "weights": {"α": self.alpha, "β": self.beta, "γ": self.gamma},
            },
        )
