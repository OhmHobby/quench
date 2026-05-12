from __future__ import annotations

import math
from typing import Dict, List, Optional

from .constraints import Constraint


class Scorer:
    """
    Aggregates constraints into a single scalar cost C(S).

        C(S) = inf              if any hard constraint fires
               Σ wᵢ · fᵢ(S)   otherwise

    Hard constraints short-circuit: the first violation returns inf
    immediately and soft costs are never computed. This makes hard
    constraint checking O(violations) rather than O(all constraints)
    in the common case where the state is feasible.

    Usage:
        scorer = Scorer()
        scorer.add(hard(fn_capacity, name="capacity"))
        scorer.add(soft(fn_walking, weight=1.0, name="walking"))
        scorer.add(soft(fn_repeat,  weight=2.0, name="repeat_pairs"))

        cost = scorer.score(state)          # C(S)
        ok   = scorer.is_feasible(state)    # all hard constraints pass
        info = scorer.breakdown(state)      # per-constraint costs

    Constraints can also be passed at construction:
        scorer = Scorer([c1, c2, c3])
    """

    def __init__(self, constraints: Optional[List[Constraint]] = None) -> None:
        self._hard: List[Constraint] = []
        self._soft: List[Constraint] = []
        for c in (constraints or []):
            self.add(c)

    # ── registration ─────────────────────────────────────────────────────────

    def add(self, constraint: Constraint) -> "Scorer":
        """Register a constraint. Returns self so calls can be chained."""
        if constraint.is_hard:
            self._hard.append(constraint)
        else:
            self._soft.append(constraint)
        return self

    @property
    def constraints(self) -> List[Constraint]:
        """All registered constraints, hard first."""
        return self._hard + self._soft

    # ── core ─────────────────────────────────────────────────────────────────

    def score(self, state) -> float:
        """
        Compute C(S).

        Returns math.inf if any hard constraint is violated.
        Returns the weighted sum of soft penalties otherwise.
        """
        for c in self._hard:
            if c(state) == math.inf:
                return math.inf
        return sum(c(state) for c in self._soft)

    def is_feasible(self, state) -> bool:
        """True iff all hard constraints are satisfied."""
        return all(c(state) < math.inf for c in self._hard)

    # ── delta ─────────────────────────────────────────────────────────────────

    def delta(self, old_state, new_state) -> float:
        """
        ΔC = C(new_state) − C(old_state).

        Used by engines on every accepted/rejected step. The default
        implementation is a full re-score of both states — correct but O(n).

        For large problems, subclass Scorer and override delta() with an
        incremental implementation that only recomputes the terms affected
        by the perturbation. The engine interface is identical either way.
        """
        return self.score(new_state) - self.score(old_state)

    # ── diagnostics ───────────────────────────────────────────────────────────

    def breakdown(self, state) -> Dict[str, float]:
        """
        Per-constraint cost breakdown for a given state.

        Returns an ordered dict: hard constraints first, then soft.
        Useful for debugging which constraints are dominating cost,
        and for the LLM layer to explain why an arrangement was chosen.

        Example output:
            {
                "capacity":     0.0,      # hard: satisfied
                "walking":      47.3,     # soft: distance penalty
                "repeat_pairs": 120.0,    # soft: history penalty
            }
        """
        result: Dict[str, float] = {}
        for c in self._hard:
            key = c.name or repr(c)
            result[key] = c(state)
        for c in self._soft:
            key = c.name or repr(c)
            result[key] = c(state)
        return result

    def violated_hard(self, state) -> List[Constraint]:
        """All hard constraints currently violated by this state."""
        return [c for c in self._hard if c(state) == math.inf]

    def soft_total(self, state) -> float:
        """
        Sum of soft costs only, ignoring hard constraints.

        Useful for comparing two feasible states directly without
        repeating the hard constraint checks you already know pass.
        """
        return sum(c(state) for c in self._soft)

    def soft_breakdown(self, state) -> Dict[str, float]:
        """Soft costs only, as a name → cost dict."""
        return {
            (c.name or repr(c)): c(state)
            for c in self._soft
        }

    # ── display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Scorer("
            f"{len(self._hard)} hard, "
            f"{len(self._soft)} soft)"
        )
