from __future__ import annotations

import math
import random
from typing import List, Optional

from .neighbor import NeighborFn, default_neighbor
from .result import Result
from .scorer import Scorer
from .state import Slot, State


class SAEngine:
    """
    Simulated Annealing engine.

    A time-inhomogeneous Markov chain over the state space.
    At each step, a neighbor state S' is proposed. It is accepted if
    it improves cost (ΔC < 0), or with Boltzmann probability exp(−ΔC/T)
    otherwise. Temperature T decays geometrically each iteration.

    Parameters:
        T0              Initial temperature. Rule of thumb: set so that
                        exp(−ΔC_typical / T0) ≈ 0.8, i.e. ~80% of bad
                        moves are accepted at the start.
        alpha           Geometric cooling factor ∈ (0, 1). Closer to 1 =
                        slower cooling = more exploration = longer runtime.
                        Typical range: 0.99–0.9999.
        iterations      Total number of steps.
        neighbor_fn     Perturbation function. Defaults to swap().
        rng             Seeded Random instance for reproducibility.
        trace_every     Record cost/temperature every N iterations.
    """

    def __init__(
        self,
        T0: float = 100.0,
        alpha: float = 0.995,
        iterations: int = 50_000,
        neighbor_fn: Optional[NeighborFn] = None,
        rng: Optional[random.Random] = None,
        trace_every: int = 500,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if T0 <= 0:
            raise ValueError(f"T0 must be positive, got {T0}.")

        self.T0 = T0
        self.alpha = alpha
        self.iterations = iterations
        self.neighbor_fn = neighbor_fn or default_neighbor
        self.rng = rng or random.Random()
        self.trace_every = trace_every

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self, initial_state: State, scorer: Scorer, slots: List[Slot]) -> Result:
        """
        Run the SA chain from initial_state and return the best state found.

        The chain runs for self.iterations steps regardless of apparent
        convergence. To stop early, subclass and override run().
        """
        state = initial_state.copy()
        T = self.T0

        current_cost = scorer.score(state)
        best_state = state.copy()
        best_cost = current_cost
        initial_cost = current_cost

        cost_trace: List[float] = []
        temp_trace: List[float] = []
        accepts = 0
        rejects = 0

        for k in range(self.iterations):

            # ── propose ───────────────────────────────────────────────────────
            candidate = self.neighbor_fn(state, slots, self.rng)
            candidate_cost = scorer.score(candidate)
            delta = candidate_cost - current_cost

            # ── accept / reject ───────────────────────────────────────────────
            # When current state is infeasible (cost=inf), standard Boltzmann
            # comparison breaks: inf - inf = nan, and the engine freezes.
            # Fall back to soft-cost comparison so the chain can navigate
            # toward feasibility rather than locking in place.
            if current_cost == math.inf:
                if candidate_cost < math.inf:
                    accept = True                    # always take a feasible state
                else:
                    d_soft = scorer.soft_total(candidate) - scorer.soft_total(state)
                    accept = d_soft < 0 or (
                        T > 1e-10 and self.rng.random() < math.exp(-d_soft / T)
                    )
            else:
                accept = delta < 0 or (
                    T > 1e-10 and self.rng.random() < math.exp(-delta / T)
                )

            if accept:
                state = candidate
                current_cost = candidate_cost
                accepts += 1
                if current_cost < best_cost:
                    best_state = state.copy()
                    best_cost = current_cost
            else:
                rejects += 1

            # ── cool ──────────────────────────────────────────────────────────
            T *= self.alpha

            # ── trace ─────────────────────────────────────────────────────────
            if k % self.trace_every == 0:
                cost_trace.append(current_cost)
                temp_trace.append(T)

        return Result(
            best_state=best_state,
            best_cost=best_cost,
            initial_cost=initial_cost,
            cost_trace=cost_trace,
            temp_trace=temp_trace,
            iterations=self.iterations,
            engine="SA",
            meta={
                "T_final": T,
                "accept_rate": accepts / self.iterations,
                "T0": self.T0,
                "alpha": self.alpha,
            },
        )

    # ── diagnostics ───────────────────────────────────────────────────────────

    def T_final(self) -> float:
        """Temperature at the last iteration: T0 * alpha^iterations."""
        return self.T0 * (self.alpha ** self.iterations)

    def __repr__(self) -> str:
        return (
            f"SAEngine(T0={self.T0}, alpha={self.alpha}, "
            f"iters={self.iterations:,})"
        )
