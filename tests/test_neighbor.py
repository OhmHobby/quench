import math
import random
from collections import Counter

import pytest

from core import Entity, Slot, State
from core.neighbor import swap, move, mixed, make_swap_k


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def people():
    return [Entity(i) for i in range(6)]

@pytest.fixture
def groups():
    return [Slot(f"G{i}", capacity=3) for i in range(2)]

@pytest.fixture
def state(people, groups):
    return State.balanced(people, groups, rng=random.Random(0))

@pytest.fixture
def rng():
    return random.Random(42)


# ── swap ──────────────────────────────────────────────────────────────────────

class TestSwap:
    def test_returns_new_object(self, state, groups, rng):
        neighbor = swap(state, groups, rng)
        assert neighbor is not state

    def test_preserves_group_sizes(self, state, groups, rng):
        sizes_before = Counter(state[e].id for e in state.entities())
        neighbor = swap(state, groups, rng)
        sizes_after = Counter(neighbor[e].id for e in neighbor.entities())
        assert sizes_before == sizes_after

    def test_preserves_entity_set(self, state, groups, rng, people):
        neighbor = swap(state, groups, rng)
        assert set(e for e, _ in neighbor) == set(people)

    def test_original_unmodified(self, state, groups, rng, people):
        original_slots = {e: state[e] for e in people}
        swap(state, groups, rng)
        for e in people:
            assert state[e] == original_slots[e]

    def test_produces_different_assignment(self, state, groups):
        # Over many seeds, swap should occasionally change the assignment
        changed = False
        for seed in range(50):
            neighbor = swap(state, groups, random.Random(seed))
            if any(neighbor[e] != state[e] for e in state.entities()):
                changed = True
                break
        assert changed, "swap never produced a different assignment"

    def test_exactly_two_entities_change(self, state, groups, rng):
        # A single swap changes exactly 0 or 2 entity assignments
        # (0 if same group chosen, 2 if different groups)
        neighbor = swap(state, groups, rng)
        changed = [e for e in state.entities() if neighbor[e] != state[e]]
        assert len(changed) in (0, 2)

    def test_single_entity_state(self, groups, rng):
        one = [Entity("solo")]
        s = State({one[0]: groups[0]})
        neighbor = swap(s, groups, rng)
        assert neighbor[one[0]] == s[one[0]]  # can't swap with itself


# ── move ──────────────────────────────────────────────────────────────────────

class TestMove:
    def test_returns_new_object(self, state, groups, rng):
        neighbor = move(state, groups, rng)
        assert neighbor is not state

    def test_preserves_entity_set(self, state, groups, rng, people):
        neighbor = move(state, groups, rng)
        assert set(e for e, _ in neighbor) == set(people)

    def test_exactly_one_entity_changes(self, state, groups, rng):
        neighbor = move(state, groups, rng)
        changed = [e for e in state.entities() if neighbor[e] != state[e]]
        # 0 if randomly reassigned to the same slot, otherwise 1
        assert len(changed) in (0, 1)

    def test_can_change_group_sizes(self, state, groups):
        # move should sometimes change group size distribution
        # (unlike swap which never does)
        size_changes_seen = False
        for seed in range(50):
            neighbor = move(state, groups, random.Random(seed))
            before = Counter(state[e].id for e in state.entities())
            after = Counter(neighbor[e].id for e in neighbor.entities())
            if before != after:
                size_changes_seen = True
                break
        assert size_changes_seen

    def test_original_unmodified(self, state, groups, rng, people):
        original = {e: state[e] for e in people}
        move(state, groups, rng)
        for e in people:
            assert state[e] == original[e]


# ── mixed ─────────────────────────────────────────────────────────────────────

class TestMixed:
    def test_returns_valid_state(self, state, groups, rng):
        neighbor = mixed(state, groups, rng)
        assert set(e for e, _ in neighbor) == set(state.entities())

    def test_calls_both_swap_and_move_over_many_seeds(self, state, groups):
        # Over many calls, mixed should sometimes produce size changes (move)
        # and sometimes not (swap). Verify it's not always one or the other.
        size_preserved = 0
        size_changed = 0
        before = Counter(state[e].id for e in state.entities())
        for seed in range(100):
            n = mixed(state, groups, random.Random(seed))
            after = Counter(n[e].id for e in n.entities())
            if before == after:
                size_preserved += 1
            else:
                size_changed += 1
        assert size_preserved > 0, "mixed never called swap"
        assert size_changed > 0, "mixed never called move"


# ── make_swap_k ───────────────────────────────────────────────────────────────

class TestMakeSwapK:
    def test_k1_behaves_like_swap(self, state, groups, rng):
        fn = make_swap_k(1)
        neighbor = fn(state, groups, rng)
        changed = [e for e in state.entities() if neighbor[e] != state[e]]
        assert len(changed) in (0, 2)

    def test_k2_changes_up_to_4_entities(self, state, groups):
        fn = make_swap_k(2)
        for seed in range(20):
            neighbor = fn(state, groups, random.Random(seed))
            changed = [e for e in state.entities() if neighbor[e] != state[e]]
            assert len(changed) in (0, 2, 4)

    def test_preserves_group_sizes(self, state, groups):
        fn = make_swap_k(3)
        before = Counter(state[e].id for e in state.entities())
        neighbor = fn(state, groups, random.Random(0))
        after = Counter(neighbor[e].id for e in neighbor.entities())
        assert before == after

    def test_name_includes_k(self):
        fn = make_swap_k(5)
        assert "5" in fn.__name__
