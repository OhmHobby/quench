from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class Entity:
    """
    Something being assigned.
    Examples: a person, a class, a job.

    Identity is determined solely by `id` — two Entity objects with the
    same id are the same entity regardless of their metadata.
    """
    id: Any
    meta: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Entity) and self.id == other.id

    def __repr__(self) -> str:
        return f"Entity({self.id!r})"


@dataclass(frozen=True)
class Slot:
    """
    Something being assigned to.
    Examples: a group, a (room, timeslot) pair, a job role.

    `capacity` is the maximum number of entities that can occupy this slot.
    Enforcing capacity is the job of a HardConstraint — Slot just declares it.
    """
    id: Any
    capacity: int = field(default=1, compare=False, hash=False)
    meta: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Slot) and self.id == other.id

    def __repr__(self) -> str:
        return f"Slot({self.id!r})"


class State:
    """
    A complete assignment: every Entity mapped to exactly one Slot.

    Internally mutable for engine performance. Always call .copy() before
    perturbing if you need to preserve the original — engines do this
    automatically before proposing a neighbor.

    Direct dict-style access:
        slot = state[entity]

    Mutation:
        state.assign(entity, new_slot)   # reassign one entity
        state.swap(entity_a, entity_b)   # exchange two entities' slots

    Query:
        state.group(slot)   # all entities currently in a slot
    """

    def __init__(self, assignment: Dict[Entity, Slot]) -> None:
        self._a: Dict[Entity, Slot] = dict(assignment)

    # ── read ─────────────────────────────────────────────────────────────────

    def __getitem__(self, entity: Entity) -> Slot:
        return self._a[entity]

    def __contains__(self, entity: Entity) -> bool:
        return entity in self._a

    def __len__(self) -> int:
        return len(self._a)

    def __iter__(self) -> Iterator[Tuple[Entity, Slot]]:
        return iter(self._a.items())

    def entities(self) -> List[Entity]:
        return list(self._a.keys())

    def slots_assigned(self) -> List[Slot]:
        """All slots currently assigned (with duplicates)."""
        return list(self._a.values())

    def items(self) -> Iterable[Tuple[Entity, Slot]]:
        return self._a.items()

    def group(self, slot: Slot) -> List[Entity]:
        """All entities currently assigned to the given slot."""
        return [e for e, s in self._a.items() if s == slot]

    def groups(self) -> Dict[Slot, List[Entity]]:
        """Full slot → [entity] mapping."""
        result: Dict[Slot, List[Entity]] = {}
        for e, s in self._a.items():
            result.setdefault(s, []).append(e)
        return result

    # ── write (in-place) ─────────────────────────────────────────────────────

    def assign(self, entity: Entity, slot: Slot) -> None:
        """Reassign one entity to a new slot."""
        self._a[entity] = slot

    def swap(self, e1: Entity, e2: Entity) -> None:
        """Swap the slot assignments of two entities."""
        self._a[e1], self._a[e2] = self._a[e2], self._a[e1]

    # ── copy ─────────────────────────────────────────────────────────────────

    def copy(self) -> State:
        """Shallow copy — new assignment dict, same Entity/Slot objects."""
        return State(dict(self._a))

    # ── factory ──────────────────────────────────────────────────────────────

    @classmethod
    def random(
        cls,
        entities: List[Entity],
        slots: List[Slot],
        rng: Optional[random.Random] = None,
    ) -> State:
        """
        Uniformly random initial assignment.

        Each entity is assigned to a slot chosen uniformly at random.
        Slot capacity is NOT enforced here — that is a constraint's job.
        Pass a seeded `rng` for reproducible initialisation.

        Warning: random init produces a Poisson-distributed group size
        which may be far from balanced. If your problem has equal-capacity
        slots and uses `swap` as the neighbor function, use `State.balanced()`
        instead — swap cannot change group sizes, so an unbalanced random
        init will never reach a balanced feasible state.
        """
        r = rng or random
        return cls({e: r.choice(slots) for e in entities})

    @classmethod
    def balanced(
        cls,
        entities: List[Entity],
        slots: List[Slot],
        rng: Optional[random.Random] = None,
    ) -> State:
        """
        Distribute entities as evenly as possible across slots.

        Entities are shuffled then assigned round-robin: slot[i % k].
        With n entities and k slots every slot gets floor(n/k) or
        ceil(n/k) entities — the most balanced distribution possible.

        Use this instead of random() when:
          - Slots have equal capacity
          - The neighbor function is swap() (which preserves group sizes)
          - The problem requires all groups to be approximately equal size

        Balanced init + swap neighbor keeps the engine in the feasible
        region throughout the run, so the hard capacity constraint never
        fires and all iterations contribute to optimising soft costs.
        """
        r = rng or random
        shuffled = list(entities)
        r.shuffle(shuffled)
        return cls({e: slots[i % len(slots)] for i, e in enumerate(shuffled)})

    # ── display ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        pairs = ", ".join(f"{e.id}→{s.id}" for e, s in self._a.items())
        return f"State({pairs})"
