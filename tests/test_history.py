import json
import math
import random
import tempfile
from pathlib import Path

import pytest

from quench import Entity, Slot, State, Scorer, hard, History


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def people():
    return [Entity(n) for n in ["Alice", "Bob", "Carol", "Dan"]]

@pytest.fixture
def groups():
    return [Slot(f"G{i}", capacity=2) for i in range(2)]

@pytest.fixture
def fixed_state(people, groups):
    """Alice+Bob in G0, Carol+Dan in G1."""
    return State({
        people[0]: groups[0],
        people[1]: groups[0],
        people[2]: groups[1],
        people[3]: groups[1],
    })

@pytest.fixture
def alt_state(people, groups):
    """Alice+Carol in G0, Bob+Dan in G1."""
    return State({
        people[0]: groups[0],
        people[2]: groups[0],
        people[1]: groups[1],
        people[3]: groups[1],
    })


# ── update and co_occurrence ──────────────────────────────────────────────────

class TestHistoryUpdate:
    def test_starts_empty(self, people):
        h = History()
        alice, bob = people[0], people[1]
        assert h.co_occurrence(alice, bob) == 0

    def test_update_increments_paired_entities(self, fixed_state, people):
        h = History()
        alice, bob = people[0], people[1]
        h.update(fixed_state)
        assert h.co_occurrence(alice, bob) == 1

    def test_update_symmetric(self, fixed_state, people):
        h = History()
        alice, bob = people[0], people[1]
        h.update(fixed_state)
        assert h.co_occurrence(alice, bob) == h.co_occurrence(bob, alice)

    def test_update_does_not_count_cross_slot_pairs(self, fixed_state, people):
        h = History()
        alice, carol = people[0], people[2]  # in different groups
        h.update(fixed_state)
        assert h.co_occurrence(alice, carol) == 0

    def test_multiple_updates_accumulate(self, fixed_state, people):
        h = History()
        alice, bob = people[0], people[1]
        h.update(fixed_state)
        h.update(fixed_state)
        assert h.co_occurrence(alice, bob) == 2

    def test_sessions_counter_increments(self, fixed_state):
        h = History()
        assert h.sessions == 0
        h.update(fixed_state)
        assert h.sessions == 1
        h.update(fixed_state)
        assert h.sessions == 2


# ── penalty ───────────────────────────────────────────────────────────────────

class TestHistoryPenalty:
    def test_penalty_zero_on_fresh_history(self, fixed_state):
        h = History()
        assert h.penalty(fixed_state) == 0.0

    def test_penalty_counts_co_occurrences(self, fixed_state, alt_state, people):
        h = History()
        h.update(fixed_state)        # Alice+Bob paired, Carol+Dan paired

        # alt_state pairs Alice+Carol (not previously paired) and Bob+Dan (not)
        assert h.penalty(alt_state) == 0.0

        # fixed_state again repeats Alice+Bob and Carol+Dan
        assert h.penalty(fixed_state) == 2.0  # 2 pairs × 1 prior co-occurrence each

    def test_penalty_additive_across_sessions(self, fixed_state, people):
        h = History()
        h.update(fixed_state)
        h.update(fixed_state)
        # Each pair has been together twice; penalty = 2 pairs × 2 = 4
        assert h.penalty(fixed_state) == 4.0


# ── as_soft_constraint ────────────────────────────────────────────────────────

class TestAsSoftConstraint:
    def test_returns_soft_constraint(self, fixed_state):
        h = History()
        h.update(fixed_state)
        c = h.as_soft_constraint(weight=2.0)
        assert not c.is_hard

    def test_constraint_reflects_live_history(self, fixed_state, people, groups):
        h = History()
        c = h.as_soft_constraint(weight=1.0)

        # Before any update, penalty is 0
        assert c(fixed_state) == pytest.approx(0.0)

        # After update, penalty is non-zero for the same arrangement
        h.update(fixed_state)
        assert c(fixed_state) > 0.0

    def test_weight_applied(self, fixed_state, people):
        h = History()
        h.update(fixed_state)
        c1 = h.as_soft_constraint(weight=1.0)
        c2 = h.as_soft_constraint(weight=5.0)
        assert c2(fixed_state) == pytest.approx(5.0 * c1(fixed_state))

    def test_integrates_with_scorer(self, fixed_state, groups):
        h = History()
        h.update(fixed_state)

        def capacity_violated(s):
            return any(len(s.group(g)) > g.capacity for g in groups)

        scorer = Scorer([
            hard(capacity_violated, name="capacity"),
            h.as_soft_constraint(weight=3.0),
        ])

        cost = scorer.score(fixed_state)
        assert cost < math.inf
        assert cost > 0.0


# ── most_paired ───────────────────────────────────────────────────────────────

class TestMostPaired:
    def test_returns_top_n(self, fixed_state, people):
        h = History()
        h.update(fixed_state)
        h.update(fixed_state)
        alice = people[0]
        top = h.most_paired(alice, top_n=1)
        assert len(top) == 1
        partner_id, count = top[0]
        assert partner_id == "Bob"
        assert count == 2

    def test_empty_for_unpaired_entity(self, people):
        h = History()
        assert h.most_paired(people[0]) == []


# ── reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_matrix(self, fixed_state, people):
        h = History()
        h.update(fixed_state)
        h.reset()
        assert h.co_occurrence(people[0], people[1]) == 0

    def test_reset_clears_sessions(self, fixed_state):
        h = History()
        h.update(fixed_state)
        h.reset()
        assert h.sessions == 0


# ── persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load_roundtrip(self, fixed_state, people):
        h = History()
        h.update(fixed_state)
        h.update(fixed_state)
        alice, bob = people[0], people[1]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        h.save(path)
        h2 = History.load(path)

        assert h2.co_occurrence(alice, bob) == h.co_occurrence(alice, bob)
        assert h2.sessions == h.sessions

    def test_saved_file_is_valid_json(self, fixed_state):
        h = History()
        h.update(fixed_state)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        h.save(path)
        with open(path) as f:
            data = json.load(f)
        assert "sessions" in data
        assert "matrix" in data

    def test_load_then_update_accumulates(self, fixed_state, people):
        h = History()
        h.update(fixed_state)
        alice, bob = people[0], people[1]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        h.save(path)
        h2 = History.load(path)
        h2.update(fixed_state)

        assert h2.co_occurrence(alice, bob) == 2
