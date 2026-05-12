from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Result:
    """
    Output from any engine run.

    All engines return this same structure so the Solver and any
    downstream code (LLM layer, tests, examples) are engine-agnostic.

    Fields:
        best_state      The lowest-cost State seen during the run.
        best_cost       C(best_state). math.inf means no feasible state found.
        initial_cost    C(state) at iteration 0. Useful for measuring improvement.
        cost_trace      C(current_state) sampled every `trace_every` iterations.
        temp_trace      Temperature at each trace point (coldest chain for PT).
        iterations      Total iterations completed.
        engine          Engine identifier string — "SA", "PT", etc.
        meta            Engine-specific extras (e.g. PT swap acceptance rate).
    """

    best_state: Any                              # State — typed as Any to avoid circular import
    best_cost: float
    initial_cost: float
    cost_trace: List[float]       = field(default_factory=list)
    temp_trace: List[float]       = field(default_factory=list)
    iterations: int               = 0
    engine: str                   = ""
    meta: Dict[str, Any]          = field(default_factory=dict)

    # ── convenience ───────────────────────────────────────────────────────────

    @property
    def improvement(self) -> float:
        """Absolute cost reduction from initial to best."""
        return self.initial_cost - self.best_cost

    @property
    def improvement_pct(self) -> float:
        """Percentage improvement. Returns 0.0 if initial cost is inf or zero."""
        if self.initial_cost in (0.0, math.inf):
            return 0.0
        return 100.0 * self.improvement / self.initial_cost

    @property
    def converged(self) -> bool:
        """
        True if the cost trace flattened out in the final 20% of the run.

        A rough signal — useful for diagnosing whether to run longer or
        tune the cooling schedule. Not a formal convergence guarantee.
        """
        if len(self.cost_trace) < 10:
            return False
        tail = self.cost_trace[int(len(self.cost_trace) * 0.8):]
        return max(tail) - min(tail) < 1e-6

    @property
    def feasible(self) -> bool:
        """True if the best state satisfies all hard constraints."""
        return self.best_cost < math.inf

    def breakdown(self, scorer) -> Dict[str, float]:
        """Per-constraint cost breakdown for best_state."""
        return scorer.breakdown(self.best_state)

    def __repr__(self) -> str:
        feasible_str = "feasible" if self.feasible else "INFEASIBLE"
        return (
            f"Result({self.engine}, {feasible_str}, "
            f"cost={self.best_cost:.3f}, "
            f"improvement={self.improvement_pct:.1f}%, "
            f"iters={self.iterations:,})"
        )
