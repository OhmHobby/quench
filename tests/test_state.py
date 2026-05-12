import math
import random

import pytest

from quench import Entity, Slot, State


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def people():
    return [Entity(name) for name in ["Alice", "Bob", "Carol", "Dan"]]

@pytest.fixture
def groups():
    return [Slot(f"G{i}", capacity=2) for i in range(2)]

@pytest.fixture
def balanced_state(people, groups):
    return State.balanced(people, groups, rng=random.Random(0))


# ── Entity ────────────────────────────────────────────────────────────────────

class TestEntity:
    def test_equality_by_id(self):
        assert Entity("Alice") == Entity("Alice")

    def test_inequality_different_id(self):
        assert Entity("Alice") != Entity("Bob")

    def test_meta_does_not_affect_equality(self):
        a = Entity("Alice", meta={"age": 30})
        b = Entity("Alice", meta={"age": 99})
        assert a == b

    def test_hashable_usable_as_dict_key(self):
        d = {Entity("Alice"): "value"}
        assert d[Entity("Alice")] == "value"

    def test_hashable_usable_in_set(self):
        s = {Entity("Alice"), Entity("Alice"), Entity("Bob")}
        assert len(s) == 2

    def test_repr(self):
        assert "Alice" in repr(Entity("Alice"))


# ── Slot ──────────────────────────────────────────────────────────────────────

class TestSlot:
    def test_equality_by_id(self):
        assert Slot("G1") == Slot("G1")

    def test_inequality_different_id(self):
        assert Slot("G1") != Slot("G2")

    def test_capacity_does_not_affect_equality(self):
        assert Slot("G1", capacity=2) == Slot("G1", capacity=99)

    def test_default_capacity(self):
        assert Slot("G1").capacity == 1

    def test_hashable(self):
        d = {Slot("G1"): "a", Slot("G2"): "b"}
        assert len(d) == 2


# ── State ─────────────────────────────────────────────────────────────────────

class TestStateRandom:
    def test_all_entities_assigned(self, people, groups):
        s = State.random(people, groups, rng=random.Random(0))
        assert len(s) == len(people)

    def test_all_slots_are_valid(self, people, groups):
        s = State.random(people, groups, rng=random.Random(0))
        for _, slot in s:
            assert slot in groups

    def test_seeded_reproducible(self, people, groups):
        s1 = State.random(people, groups, rng=random.Random(42))
        s2 = State.random(people, groups, rng=random.Random(42))
        for e in people:
            assert s1[e] == s2[e]

    def test_different_seeds_differ(self, people, groups):
        results = set()
        for seed in range(20):
            s = State.random(people, groups, rng=random.Random(seed))
            results.add(tuple(s[e].id for e in people))
        # With 4 people and 2 groups, there are 2^4=16 assignments
        # — we should see more than 1 distinct assignment across 20 seeds
        assert len(results) > 1


class TestStateBalanced:
    def test_sizes_differ_by_at_most_one(self, people, groups):
        s = State.balanced(people, groups, rng=random.Random(0))
        sizes = [len(s.group(g)) for g in groups]
        assert max(sizes) - min(sizes) <= 1

    def test_all_entities_assigned(self, people, groups):
        s = State.balanced(people, groups, rng=random.Random(0))
        assert len(s) == len(people)

    def test_total_entities_conserved(self, people, groups):
        s = State.balanced(people, groups, rng=random.Random(0))
        assert sum(len(s.group(g)) for g in groups) == len(people)

    def test_odd_split(self):
        entities = [Entity(i) for i in range(5)]
        slots = [Slot("A"), Slot("B")]
        s = State.balanced(entities, slots, rng=random.Random(0))
        sizes = sorted(len(s.group(sl)) for sl in slots)
        assert sizes == [2, 3]

    def test_seeded_reproducible(self, people, groups):
        s1 = State.balanced(people, groups, rng=random.Random(7))
        s2 = State.balanced(people, groups, rng=random.Random(7))
        for e in people:
            assert s1[e] == s2[e]


class TestStateMutation:
    def test_swap_exchanges_slots(self, balanced_state, people):
        s = balanced_state.copy()
        a, b = people[0], people[1]
        slot_a_before = s[a]
        slot_b_before = s[b]
        s.swap(a, b)
        assert s[a] == slot_b_before
        assert s[b] == slot_a_before

    def test_swap_preserves_group_sizes(self, balanced_state, groups):
        sizes_before = [len(balanced_state.group(g)) for g in groups]
        s = balanced_state.copy()
        s.swap(s.entities()[0], s.entities()[-1])
        sizes_after = [len(s.group(g)) for g in groups]
        assert sizes_before == sizes_after

    def test_assign_changes_slot(self, balanced_state, people, groups):
        s = balanced_state.copy()
        target_slot = groups[0]
        s.assign(people[0], target_slot)
        assert s[people[0]] == target_slot

    def test_copy_is_independent(self, balanced_state, people):
        original = balanced_state
        slots_before = {e: original[e] for e in people}   # snapshot before mutation
        copy = original.copy()
        original.swap(people[0], people[1])
        # copy must still reflect the pre-swap state
        for e in people:
            assert copy[e] == slots_before[e]


class TestStateQuery:
    def test_group_returns_correct_members(self, balanced_state, groups):
        for g in groups:
            members = balanced_state.group(g)
            for e in members:
                assert balanced_state[e] == g

    def test_groups_covers_all_entities(self, balanced_state, people, groups):
        all_in_groups = [e for g in groups for e in balanced_state.group(g)]
        assert set(all_in_groups) == set(people)

    def test_groups_dict_matches_group(self, balanced_state, groups):
        g_dict = balanced_state.groups()
        for g in groups:
            assert set(g_dict.get(g, [])) == set(balanced_state.group(g))

    def test_contains(self, balanced_state, people):
        for e in people:
            assert e in balanced_state

    def test_len(self, balanced_state, people):
        assert len(balanced_state) == len(people)

    def test_iter_yields_entity_slot_pairs(self, balanced_state, people, groups):
        for entity, slot in balanced_state:
            assert entity in people
            assert slot in groups
