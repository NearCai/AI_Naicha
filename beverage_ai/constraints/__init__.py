"""Hard + soft constraint checker. Corresponds to 技术方案书 §3.4 + §E.3."""

from .checker import (
    ConstraintViolation,
    Severity,
    check_constraints,
    is_feasible,
)

__all__ = ["ConstraintViolation", "Severity", "check_constraints", "is_feasible"]
