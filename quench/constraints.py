from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Callable


class Constraint(ABC):
    """
    Base class for all constraints.

    A constraint is a callable that maps a State to a float cost:
        hard → 0.0 if satisfied, math.inf if violated
        soft → non-negative real penalty (already weighted)

    Subclass this directly only if you need stateful constraints
    (e.g. one that holds a reference to a history tracker).
    For simple lambda-based constraints, use hard() and soft().
    """

    def __init__(self, name: str = "") -> None:
        self.name = name

    @abstractmethod
    def __call__(self, state) -> float: ...

    @property
    @abstractmethod
    def is_hard(self) -> bool: ...

    def __repr__(self) -> str:
        kind = "hard" if self.is_hard else "soft"
        label = self.name or "unnamed"
        return f"<{kind}:{label}>"


class HardConstraint(Constraint):
    """
    A constraint that makes a state inadmissible when violated.

    fn(state) -> bool
        True  = constraint is VIOLATED  → cost = inf
        False = constraint is satisfied → cost = 0.0

    The engine will never accept a state where any hard constraint fires.
    Use this for physical impossibilities: room over capacity, a person
    assigned to two places at once, etc.
    """

    def __init__(self, fn: Callable, name: str = "") -> None:
        super().__init__(name)
        self._fn = fn

    @property
    def is_hard(self) -> bool:
        return True

    def __call__(self, state) -> float:
        return math.inf if self._fn(state) else 0.0


class SoftConstraint(Constraint):
    """
    A constraint that is penalised but not forbidden.

    fn(state) -> float
        Returns a raw non-negative penalty.
        The final contribution to C(S) is weight * fn(state).

    Weights let you express relative importance across constraints.
    A weight of 0.0 registers the constraint without effect (useful
    for logging/debugging without influencing the optimiser).
    """

    def __init__(
        self,
        fn: Callable,
        weight: float = 1.0,
        name: str = "",
    ) -> None:
        super().__init__(name)
        if weight < 0:
            raise ValueError(f"Soft constraint weight must be >= 0, got {weight}.")
        self._fn = fn
        self.weight = weight

    @property
    def is_hard(self) -> bool:
        return False

    def __call__(self, state) -> float:
        return self.weight * self._fn(state)


# ── Convenience constructors ──────────────────────────────────────────────────

def hard(fn: Callable, name: str = "") -> HardConstraint:
    """
    Declare a hard constraint from a boolean function.

    fn(state) -> bool   True = VIOLATED.

    Example — enforce slot capacity:
        hard(
            lambda s: any(
                len(s.group(slot)) > slot.capacity
                for slot in slots
            ),
            name="capacity"
        )
    """
    return HardConstraint(fn, name)


def soft(
    fn: Callable,
    weight: float = 1.0,
    name: str = "",
) -> SoftConstraint:
    """
    Declare a soft constraint from a penalty function.

    fn(state) -> float   non-negative penalty.

    Example — penalise total walking distance:
        soft(
            lambda s: sum(
                distance(s[e], prev_slot[e]) for e in entities
            ),
            weight=1.0,
            name="walking"
        )

    Example — penalise repeat group pairings via history:
        soft(
            lambda s: sum(
                history[e1.id][e2.id]
                for slot in slots
                for e1, e2 in combinations(s.group(slot), 2)
            ),
            weight=2.0,
            name="repeat_pairs"
        )
    """
    return SoftConstraint(fn, weight, name)
