from __future__ import annotations

import random
from typing import List, Literal, Optional

from .engine_pt import PTEngine
from .engine_sa import SAEngine
from .neighbor import NeighborFn
from .result import Result
from .scorer import Scorer
from .state import Entity, Slot, State

# Entity count above which PT is selected over SA in auto mode.
# SA is fast and sufficient for small state spaces (kⁿ where n < threshold).
# PT's multi-chain overhead pays off once the landscape becomes multimodal.
_PT_THRESHOLD = 40

EngineChoice = Literal["auto", "sa", "pt"]
InitStrategy = Literal["random", "balanced"]


class Solver:
    """
    Public API for the constraint-forge engine.

    Accepts the problem definition (entities, slots, scorer), selects
    and configures an engine, and exposes two methods:

        .solve()    → Result  (single best arrangement)
        .sample(n)  → List[Result]  (n independent runs, sorted by cost)

    Engine selection:
        "auto"  selects SA for n_entities < 40, PT otherwise.
        "sa"    forces Simulated Annealing.
        "pt"    forces Parallel Tempering.

    Example — group rotation with history:
        entities = [Entity(name) for name in roster]
        slots    = [Slot(f"Group{i}", capacity=5) for i in range(4)]

        history  = History()
        scorer   = Scorer([
            hard(capacity_ok, name="capacity"),
            soft(walk_cost,   weight=1.0, name="walking"),
            history.as_soft_constraint(weight=2.0),
        ])

        solver = Solver(entities, slots, scorer)
        result = solver.solve(seed=42)

        print(result)
        print(result.breakdown(scorer))
        history.update(result.best_state)
    """

    def __init__(
        self,
        entities: List[Entity],
        slots: List[Slot],
        scorer: Scorer,
        engine: EngineChoice = "auto",
        init: InitStrategy = "random",
        neighbor_fn: Optional[NeighborFn] = None,
        rng: Optional[random.Random] = None,
        **engine_kwargs,
    ) -> None:
        if not entities:
            raise ValueError("entities list must not be empty.")
        if not slots:
            raise ValueError("slots list must not be empty.")

        self.entities = entities
        self.slots = slots
        self.scorer = scorer
        self.engine_choice = engine
        self.init = init
        self.neighbor_fn = neighbor_fn
        self.rng = rng or random.Random()
        self.engine_kwargs = engine_kwargs

    def _init_state(self) -> State:
        if self.init == "balanced":
            return State.balanced(self.entities, self.slots, self.rng)
        return State.random(self.entities, self.slots, self.rng)

    # ── engine factory ────────────────────────────────────────────────────────

    def _build_engine(self) -> SAEngine | PTEngine:
        choice = self.engine_choice
        if choice == "auto":
            choice = "sa" if len(self.entities) < _PT_THRESHOLD else "pt"

        shared = dict(
            neighbor_fn=self.neighbor_fn,
            rng=self.rng,
            **self.engine_kwargs,
        )

        if choice == "sa":
            # Strip PT-only kwargs to avoid TypeError
            sa_kwargs = {
                k: v for k, v in shared.items()
                if k in ("neighbor_fn", "rng", "T0", "alpha",
                         "iterations", "trace_every")
            }
            return SAEngine(**sa_kwargs)

        if choice == "pt":
            pt_kwargs = {
                k: v for k, v in shared.items()
                if k in ("neighbor_fn", "rng", "T_min", "T_max",
                         "n_chains", "iterations", "swap_interval", "trace_every")
            }
            return PTEngine(**pt_kwargs)

        raise ValueError(
            f"Unknown engine {choice!r}. Choose from: 'auto', 'sa', 'pt'."
        )

    # ── public API ────────────────────────────────────────────────────────────

    def solve(self, seed: Optional[int] = None) -> Result:
        """
        Run one search and return the best arrangement found.

        If `seed` is provided, the RNG is re-seeded before the run —
        useful for reproducible benchmarks and tests.
        """
        if seed is not None:
            self.rng = random.Random(seed)

        initial = self._init_state()
        engine = self._build_engine()
        return engine.run(initial, self.scorer, self.slots)

    def sample(self, n: int = 5, seed: Optional[int] = None) -> List[Result]:
        """
        Run `n` independent searches and return all results sorted by cost.

        The lowest-cost result is at index 0. Use this when you want to:
          - Present multiple good arrangements (e.g. for an LLM to explain)
          - Estimate the variance of the engine on your problem
          - Run in parallel (embarrassingly parallel — each call is independent)

        Example:
            results = solver.sample(n=10, seed=0)
            best    = results[0]
            top3    = results[:3]
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}.")
        if seed is not None:
            self.rng = random.Random(seed)

        results = []
        for _ in range(n):
            initial = self._init_state()
            engine = self._build_engine()
            results.append(engine.run(initial, self.scorer, self.slots))

        return sorted(results, key=lambda r: r.best_cost)

    # ── diagnostics ───────────────────────────────────────────────────────────

    @property
    def selected_engine(self) -> str:
        """Which engine would be selected for the current configuration."""
        if self.engine_choice != "auto":
            return self.engine_choice
        return "sa" if len(self.entities) < _PT_THRESHOLD else "pt"

    def __repr__(self) -> str:
        return (
            f"Solver("
            f"{len(self.entities)} entities, "
            f"{len(self.slots)} slots, "
            f"engine={self.selected_engine!r})"
        )
