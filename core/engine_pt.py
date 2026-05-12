from __future__ import annotations

import math
import random
from typing import List, Optional

from .neighbor import NeighborFn, default_neighbor
from .result import Result
from .scorer import Scorer
from .state import Slot, State


class PTEngine:
    """
    Parallel Tempering (Replica Exchange MCMC) engine.

    Runs `n_chains` independent SA chains simultaneously on a geometric
    temperature ladder T_min ... T_max. Every `swap_interval` steps,
    adjacent chains propose to exchange their states via a
    Metropolis-Hastings acceptance criterion:

        P(swap i ↔ j) = min(1, exp((1/Tᵢ − 1/Tⱼ) · (C(Sᵢ) − C(Sⱼ))))

    The cold chains (low T) exploit fine structure; the hot chains (high T)
    explore freely and escape local minima. Good solutions propagate down
    the ladder via swaps. The best state is always tracked on the coldest
    chain (index 0).

    Why PT beats single-chain SA on hard problems:
        SA's mixing time at low T is exponential in the depth of cost barriers.
        PT avoids this: the hot chain crosses barriers freely, and swaps carry
        those states to the cold chain without requiring the cold chain to
        cross the barrier directly.

    Parameters:
        T_min           Temperature of the coldest (most exploitative) chain.
        T_max           Temperature of the hottest (most exploratory) chain.
        n_chains        Number of chains. More chains → better coverage,
                        higher compute cost. 4–8 is typical.
        iterations      Steps per chain per run. Total cost = iterations × n_chains.
        swap_interval   Propose inter-chain swaps every N steps.
                        Too frequent = chains don't mix between swaps.
                        Too infrequent = slow propagation of good solutions.
                        Rule of thumb: iterations / 100.
        neighbor_fn     Perturbation function. Defaults to swap().
        rng             Seeded Random instance for reproducibility.
        trace_every     Record cold-chain cost/temperature every N iterations.
    """

    def __init__(
        self,
        T_min: float = 0.5,
        T_max: float = 100.0,
        n_chains: int = 6,
        iterations: int = 50_000,
        swap_interval: int = 500,
        neighbor_fn: Optional[NeighborFn] = None,
        rng: Optional[random.Random] = None,
        trace_every: int = 500,
    ) -> None:
        if T_min <= 0 or T_max <= 0:
            raise ValueError("Temperatures must be positive.")
        if T_min >= T_max:
            raise ValueError("T_min must be strictly less than T_max.")
        if n_chains < 2:
            raise ValueError("PT requires at least 2 chains. Use SAEngine for n_chains=1.")

        self.T_min = T_min
        self.T_max = T_max
        self.n_chains = n_chains
        self.iterations = iterations
        self.swap_interval = swap_interval
        self.neighbor_fn = neighbor_fn or default_neighbor
        self.rng = rng or random.Random()
        self.trace_every = trace_every

    # ── temperature ladder ────────────────────────────────────────────────────

    def _ladder(self) -> List[float]:
        """
        Geometric temperature ladder from T_min (index 0) to T_max (index K-1).

        Geometric spacing is standard — it gives approximately equal swap
        acceptance rates between adjacent chains, which maximises the
        information flow down the ladder.
        """
        ratio = (self.T_max / self.T_min) ** (1.0 / (self.n_chains - 1))
        return [self.T_min * (ratio ** i) for i in range(self.n_chains)]

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self, initial_state: State, scorer: Scorer, slots: List[Slot]) -> Result:
        """
        Run all chains from independent random initialisations and return
        the best state found on the coldest chain.

        Each chain starts from a fresh copy of initial_state — they diverge
        immediately via their independent neighbor proposals. Using different
        random initialisations per chain (State.random() × n_chains) would
        also work and gives more initial diversity, but complicates the
        Solver interface. The chains diverge quickly enough from a shared start.
        """
        temps = self._ladder()

        # Each chain gets its own copy of the initial state
        chains: List[State] = [initial_state.copy() for _ in range(self.n_chains)]
        costs: List[float] = [scorer.score(s) for s in chains]

        best_state = chains[0].copy()
        best_cost = costs[0]
        initial_cost = costs[0]

        cost_trace: List[float] = []
        temp_trace: List[float] = []

        swap_proposals = 0
        swap_accepts = 0
        chain_accepts = [0] * self.n_chains

        for k in range(self.iterations):

            # ── SA step on every chain independently ──────────────────────────
            for i in range(self.n_chains):
                T = temps[i]
                candidate = self.neighbor_fn(chains[i], slots, self.rng)
                c_cost = scorer.score(candidate)
                delta = c_cost - costs[i]

                if costs[i] == math.inf:
                    if c_cost < math.inf:
                        accept = True
                    else:
                        d_soft = scorer.soft_total(candidate) - scorer.soft_total(chains[i])
                        accept = d_soft < 0 or (
                            T > 1e-10 and self.rng.random() < math.exp(-d_soft / T)
                        )
                else:
                    accept = delta < 0 or (
                        T > 1e-10 and self.rng.random() < math.exp(-delta / T)
                    )

                if accept:
                    chains[i] = candidate
                    costs[i] = c_cost
                    chain_accepts[i] += 1

            # ── track best on coldest chain ───────────────────────────────────
            if costs[0] < best_cost:
                best_state = chains[0].copy()
                best_cost = costs[0]

            # ── replica swap between adjacent chains ──────────────────────────
            if k % self.swap_interval == 0:
                # Sweep upward: propose swaps (0,1), (1,2), ..., (K-2, K-1)
                # Alternating sweep direction reduces correlation between swaps
                order = range(self.n_chains - 1)
                if (k // self.swap_interval) % 2 == 1:
                    order = reversed(order)         # type: ignore[assignment]

                for i in order:
                    j = i + 1
                    Ti, Tj = temps[i], temps[j]
                    Ci, Cj = costs[i], costs[j]

                    # Metropolis-Hastings swap acceptance
                    # Derivation: detailed balance with both chains' Boltzmann factors
                    log_accept = (1.0 / Ti - 1.0 / Tj) * (Ci - Cj)
                    swap_proposals += 1

                    if log_accept >= 0 or self.rng.random() < math.exp(log_accept):
                        chains[i], chains[j] = chains[j], chains[i]
                        costs[i], costs[j] = costs[j], costs[i]
                        swap_accepts += 1

            # ── trace coldest chain ───────────────────────────────────────────
            if k % self.trace_every == 0:
                cost_trace.append(costs[0])
                temp_trace.append(temps[0])

        swap_rate = swap_accepts / swap_proposals if swap_proposals > 0 else 0.0

        return Result(
            best_state=best_state,
            best_cost=best_cost,
            initial_cost=initial_cost,
            cost_trace=cost_trace,
            temp_trace=temp_trace,
            iterations=self.iterations,
            engine="PT",
            meta={
                "n_chains": self.n_chains,
                "T_ladder": temps,
                "swap_accept_rate": round(swap_rate, 4),
                "chain_accept_rates": [
                    round(a / self.iterations, 4) for a in chain_accepts
                ],
                "T_min": self.T_min,
                "T_max": self.T_max,
            },
        )

    def __repr__(self) -> str:
        return (
            f"PTEngine(chains={self.n_chains}, "
            f"T=[{self.T_min}, {self.T_max}], "
            f"iters={self.iterations:,})"
        )
