from __future__ import annotations

import random
from typing import Callable, List

from .state import Slot, State

# ── Type alias ────────────────────────────────────────────────────────────────

# A neighbor function takes the current state, the full slot list, and an RNG;
# returns a new State with a small perturbation applied.
# The returned state is always a fresh copy — the input state is not mutated.
NeighborFn = Callable[[State, List[Slot], random.Random], State]


# ── Built-in neighbor functions ───────────────────────────────────────────────

def swap(state: State, slots: List[Slot], rng: random.Random) -> State:
    """
    Exchange the slot assignments of two randomly chosen entities.

    This is the primary neighbor function for problems where the total
    entity-per-slot distribution should stay roughly constant (e.g. group
    rotation where every group must be the same size). A swap never changes
    group sizes — it only changes who is in which group.

    Ergodic: any assignment is reachable from any other via a sequence of swaps,
    provided all entities are already assigned to some slot (which State.random
    guarantees).
    """
    entities = state.entities()
    if len(entities) < 2:
        return state.copy()
    e1, e2 = rng.sample(entities, 2)
    neighbor = state.copy()
    neighbor.swap(e1, e2)
    return neighbor


def move(state: State, slots: List[Slot], rng: random.Random) -> State:
    """
    Reassign one randomly chosen entity to a randomly chosen slot.

    Use this when group sizes are allowed to vary — e.g. timetabling where
    some rooms may hold more students than others. Unlike swap, move *does*
    change the slot population distribution.

    Note: if all slots have equal capacity and you want to enforce equal
    group sizes, prefer swap — move will require a hard constraint to pull
    sizes back, which wastes acceptance probability on infeasible states.
    """
    entities = state.entities()
    entity = rng.choice(entities)
    new_slot = rng.choice(slots)
    neighbor = state.copy()
    neighbor.assign(entity, new_slot)
    return neighbor


def mixed(state: State, slots: List[Slot], rng: random.Random) -> State:
    """
    Randomly apply either swap (70%) or move (30%).

    Good default for problems where group sizes have soft rather than
    hard capacity constraints — mostly preserves distribution, but
    occasionally explores distribution changes.
    """
    if rng.random() < 0.7:
        return swap(state, slots, rng)
    return move(state, slots, rng)


# ── Default ───────────────────────────────────────────────────────────────────

#: The default neighbor function used by engines when none is specified.
#: swap is the conservative default: it preserves slot population sizes,
#: which is the right prior for most arrangement problems.
default_neighbor: NeighborFn = swap


# ── Factory for custom step sizes ─────────────────────────────────────────────

def make_swap_k(k: int = 2) -> NeighborFn:
    """
    Return a neighbor function that swaps k pairs of entities simultaneously.

    Larger k = bigger jumps through state space. Useful early in the search
    when the temperature is high and you want to cover more ground per step.
    At low temperature, stick with k=1 (the default swap) for fine-grained
    local refinement.

    Example — 2-pair swap for faster early exploration:
        engine = SAEngine(neighbor_fn=make_swap_k(2))
    """
    def _swap_k(state: State, slots: List[Slot], rng: random.Random) -> State:
        entities = state.entities()
        n_swaps = min(k, len(entities) // 2)
        if n_swaps < 1:
            return state.copy()
        chosen = rng.sample(entities, n_swaps * 2)
        neighbor = state.copy()
        for i in range(n_swaps):
            neighbor.swap(chosen[2 * i], chosen[2 * i + 1])
        return neighbor

    _swap_k.__name__ = f"swap_{k}"
    return _swap_k
