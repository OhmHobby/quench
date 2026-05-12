from __future__ import annotations

import json
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from .constraints import SoftConstraint
    from .state import Entity, State


class History:
    """
    Tracks how often each pair of entities has shared the same slot
    across previous arrangement sessions.

    The co-occurrence matrix is symmetric:
        history[a][b] == history[b][a]
                      == number of sessions where a and b were in the same slot

    Usage pattern:
        history = History()

        # --- session 1 ---
        result = solver.solve()
        history.update(result.best_state)   # record pairings

        # --- session 2 ---
        scorer.add(history.as_soft_constraint(weight=2.0))
        result = solver.solve()             # engine now penalises repeat pairs
        history.update(result.best_state)

    The History object is a *data source*, not a constraint itself. It
    exposes `as_soft_constraint()` to produce a constraint that closes over
    it — the engine and scorer remain unaware of history as a concept.
    """

    def __init__(self) -> None:
        # _m[a_id][b_id] = co-occurrence count
        self._m: Dict[object, Dict[object, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._sessions: int = 0

    # ── recording ─────────────────────────────────────────────────────────────

    def update(self, state: "State") -> None:
        """
        Record all pairings from a completed arrangement.

        Call this after each session with the accepted best_state.
        Pairs within the same slot are recorded; entities in different
        slots are not (they did not share a slot this session).
        """
        for entities in state.groups().values():
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1:]:
                    self._m[e1.id][e2.id] += 1
                    self._m[e2.id][e1.id] += 1
        self._sessions += 1

    # ── querying ──────────────────────────────────────────────────────────────

    def co_occurrence(self, e1: "Entity", e2: "Entity") -> int:
        """How many past sessions did e1 and e2 share a slot?"""
        return self._m[e1.id][e2.id]

    def penalty(self, state: "State") -> float:
        """
        Total co-occurrence cost for the given state.

        For every pair of entities currently sharing a slot, adds their
        historical co-occurrence count. Higher = more repeated pairings.

        This is the raw penalty before weighting. The weight is applied
        by the SoftConstraint returned from as_soft_constraint().
        """
        total = 0
        for entities in state.groups().values():
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1:]:
                    total += self._m[e1.id][e2.id]
        return float(total)

    def most_paired(self, entity: "Entity", top_n: int = 5) -> list:
        """
        Return the top_n entities most frequently paired with the given entity.
        Useful for diagnostics and the future LLM explanation layer.
        """
        row = self._m.get(entity.id, {})
        return sorted(row.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    @property
    def sessions(self) -> int:
        """Number of times update() has been called."""
        return self._sessions

    # ── constraint factory ────────────────────────────────────────────────────

    def as_soft_constraint(self, weight: float = 1.0) -> "SoftConstraint":
        """
        Return a SoftConstraint that penalises repeat pairings.

        The constraint closes over this History object, so it reflects
        the state of the matrix at the time of evaluation — not at the
        time of constraint creation. Update history between sessions and
        the same constraint object automatically becomes stricter.

        Example:
            history = History()
            repeat_penalty = history.as_soft_constraint(weight=2.0)
            scorer.add(repeat_penalty)
            # ... run solver, call history.update(result.best_state)
            # ... next solver run automatically uses updated matrix
        """
        from .constraints import soft
        return soft(self.penalty, weight=weight, name="history_repeat")

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Serialise the co-occurrence matrix to a JSON file.

        Keys are coerced to strings (JSON limitation). If your entity IDs
        are non-string types, ensure they have a consistent __str__.
        """
        payload = {
            "sessions": self._sessions,
            "matrix": {
                str(k): {str(k2): v2 for k2, v2 in v.items()}
                for k, v in self._m.items()
            },
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "History":
        """Load a previously saved History from a JSON file."""
        h = cls()
        with open(path, "r") as f:
            payload = json.load(f)
        h._sessions = payload.get("sessions", 0)
        for k, row in payload["matrix"].items():
            for k2, count in row.items():
                h._m[k][k2] = count
        return h

    # ── utilities ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all recorded history. Irreversible unless saved first."""
        self._m.clear()
        self._sessions = 0

    def __repr__(self) -> str:
        n_pairs = sum(len(v) for v in self._m.values()) // 2
        return f"History(sessions={self._sessions}, tracked_pairs={n_pairs})"
