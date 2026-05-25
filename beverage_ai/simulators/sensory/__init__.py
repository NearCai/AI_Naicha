"""Sensory predictor — §3.3.1.

Exposes `SensoryPredictor` (Protocol), `MockSensoryPredictor`
(used when no trained model is available), and the real
`SensoryGAT` model class (requires torch + torch_geometric).
"""

from .predict import (
    CORE_DIMS,
    EXT_DIMS,
    MockSensoryPredictor,
    SensoryPrediction,
    SensoryPredictor,
)

__all__ = [
    "SensoryPredictor",
    "MockSensoryPredictor",
    "SensoryPrediction",
    "CORE_DIMS",
    "EXT_DIMS",
]
