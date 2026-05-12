import math
import random

import pytest

from quench import Entity, Slot, State, Scorer, hard, soft, HardConstraint, SoftConstraint


# ── shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def people():
    return [Entity(n) for n in ["Alice", "Bob", "Carol", "Dan"]]

@pytest.fixture
def groups():
    return [Slot(f"G{i}", capacity=2) for i in range(2)]

@pytest.fixture
def balanced(people, groups):
    return State.balanced(people, groups, rng=random.Random(0))

@pytest.fixture
def overloaded(people, groups):
    """State where G0 has 3 people (over capacity=2)."""
    return State({
        people[0]: groups[0],
        people[1]: groups[0],
        people[2]: groups[0],
        people[3]: groups[1],
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Constraints
# ═══════════════════════════════════════════════════════════════════════════════

class TestHardConstraint:
    def test_returns_zero_when_satisfied(self, balanced, groups):
        c = hard(
            lambda s: any(len(s.group(g)) > g.capacity for g in groups),
            name="capacity",
        )
        assert c(balanced) == 0.0

    def test_returns_inf_when_violated(self, overloaded, groups):
        c = hard(
            lambda s: any(len(s.group(g)) > g.capacity for g in groups),
            name="capacity",
        )
        assert c(overloaded) == math.inf

    def test_is_hard_property(self):
        c = hard(lambda s: False)
        assert c.is_hard is True

    def test_name_stored(self):
        c = hard(lambda s: False, name="my_constraint")
        assert c.name == "my_constraint"

    def test_repr_contains_hard(self):
        c = hard(lambda s: False, name="cap")
        assert "hard" in repr(c)
        assert "cap" in repr(c)


class TestSoftConstraint:
    def test_returns_weighted_penalty(self, balanced, groups):
        c = soft(lambda s: float(len(s.group(groups[0]))), weight=3.0)
        expected = 3.0 * len(balanced.group(groups[0]))
        assert c(balanced) == pytest.approx(expected)

    def test_weight_zero_returns_zero(self, balanced, groups):
        c = soft(lambda s: 999.0, weight=0.0)
        assert c(balanced) == 0.0

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            soft(lambda s: 1.0, weight=-1.0)

    def test_is_hard_property(self):
        c = soft(lambda s: 0.0)
        assert c.is_hard is False

    def test_name_stored(self):
        c = soft(lambda s: 0.0, name="walking")
        assert c.name == "walking"

    def test_default_weight_is_one(self, balanced):
        raw_penalty = 5.0
        c = soft(lambda s: raw_penalty)
        assert c(balanced) == pytest.approx(raw_penalty)


# ═══════════════════════════════════════════════════════════════════════════════
# Scorer
# ═══════════════════════════════════════════════════════════════════════════════

class TestScorer:

    def _capacity_hard(self, groups):
        return hard(
            lambda s: any(len(s.group(g)) > g.capacity for g in groups),
            name="capacity",
        )

    def _balance_soft(self, groups, weight=1.0):
        return soft(
            lambda s: sum(abs(len(s.group(g)) - 2) for g in groups),
            weight=weight,
            name="balance",
        )

    # ── basic scoring ─────────────────────────────────────────────────────────

    def test_feasible_state_returns_soft_sum(self, balanced, groups):
        scorer = Scorer([self._capacity_hard(groups), self._balance_soft(groups)])
        cost = scorer.score(balanced)
        assert cost < math.inf
        assert cost >= 0.0

    def test_infeasible_state_returns_inf(self, overloaded, groups):
        scorer = Scorer([self._capacity_hard(groups)])
        assert scorer.score(overloaded) == math.inf

    def test_no_constraints_returns_zero(self, balanced):
        scorer = Scorer()
        assert scorer.score(balanced) == 0.0

    def test_only_soft_constraints(self, balanced, groups):
        scorer = Scorer([self._balance_soft(groups, weight=2.0)])
        cost = scorer.score(balanced)
        # balanced state has equal groups — balance penalty should be 0
        assert cost == pytest.approx(0.0)

    # ── hard constraint short-circuit ─────────────────────────────────────────

    def test_hard_violation_skips_soft_evaluation(self, overloaded, groups):
        evaluated = []
        def spy_soft(s):
            evaluated.append(True)
            return 1.0

        scorer = Scorer([
            self._capacity_hard(groups),
            soft(spy_soft, name="spy"),
        ])
        scorer.score(overloaded)
        assert evaluated == [], "Soft constraint should not run when hard fires"

    def test_multiple_hard_short_circuits_on_first(self, overloaded, groups):
        calls = []
        def counting_hard(s):
            calls.append(1)
            return any(len(s.group(g)) > g.capacity for g in groups)

        scorer = Scorer([
            hard(counting_hard, name="h1"),
            hard(counting_hard, name="h2"),
        ])
        scorer.score(overloaded)
        assert len(calls) == 1, "Second hard constraint should not run after first fires"

    # ── add / chaining ────────────────────────────────────────────────────────

    def test_add_returns_self_for_chaining(self, groups):
        scorer = Scorer()
        result = scorer.add(self._capacity_hard(groups))
        assert result is scorer

    def test_add_registers_hard_and_soft_separately(self, groups):
        scorer = Scorer()
        scorer.add(self._capacity_hard(groups))
        scorer.add(self._balance_soft(groups))
        assert len(scorer._hard) == 1
        assert len(scorer._soft) == 1

    # ── is_feasible ───────────────────────────────────────────────────────────

    def test_is_feasible_true_for_valid_state(self, balanced, groups):
        scorer = Scorer([self._capacity_hard(groups)])
        assert scorer.is_feasible(balanced) is True

    def test_is_feasible_false_for_invalid_state(self, overloaded, groups):
        scorer = Scorer([self._capacity_hard(groups)])
        assert scorer.is_feasible(overloaded) is False

    def test_is_feasible_true_with_no_hard_constraints(self, overloaded):
        scorer = Scorer()
        assert scorer.is_feasible(overloaded) is True

    # ── delta ─────────────────────────────────────────────────────────────────

    def test_delta_feasible_to_feasible(self, balanced, people, groups):
        scorer = Scorer([self._balance_soft(groups)])
        s2 = balanced.copy()
        s2.swap(people[0], people[2])
        delta = scorer.delta(balanced, s2)
        expected = scorer.score(s2) - scorer.score(balanced)
        assert delta == pytest.approx(expected)

    def test_delta_zero_for_identical_states(self, balanced, groups):
        scorer = Scorer([self._balance_soft(groups)])
        assert scorer.delta(balanced, balanced.copy()) == pytest.approx(0.0)

    # ── breakdown ─────────────────────────────────────────────────────────────

    def test_breakdown_contains_all_constraints(self, balanced, groups):
        scorer = Scorer([
            self._capacity_hard(groups),
            self._balance_soft(groups),
        ])
        bd = scorer.breakdown(balanced)
        assert "capacity" in bd
        assert "balance" in bd

    def test_breakdown_values_match_score(self, balanced, groups):
        h = self._capacity_hard(groups)
        s = self._balance_soft(groups, weight=2.0)
        scorer = Scorer([h, s])
        bd = scorer.breakdown(balanced)
        assert bd["capacity"] == h(balanced)
        assert bd["balance"] == s(balanced)

    # ── violated_hard ─────────────────────────────────────────────────────────

    def test_violated_hard_empty_when_feasible(self, balanced, groups):
        scorer = Scorer([self._capacity_hard(groups)])
        assert scorer.violated_hard(balanced) == []

    def test_violated_hard_returns_violating_constraints(self, overloaded, groups):
        c = self._capacity_hard(groups)
        scorer = Scorer([c])
        violated = scorer.violated_hard(overloaded)
        assert c in violated

    # ── soft_total ────────────────────────────────────────────────────────────

    def test_soft_total_ignores_hard(self, overloaded, groups):
        scorer = Scorer([
            self._capacity_hard(groups),
            self._balance_soft(groups, weight=1.0),
        ])
        # Even though hard fires (score = inf), soft_total should still return a number
        soft_cost = scorer.soft_total(overloaded)
        assert soft_cost < math.inf
        assert soft_cost >= 0.0

    def test_soft_total_matches_score_when_feasible(self, balanced, groups):
        scorer = Scorer([
            self._capacity_hard(groups),
            self._balance_soft(groups),
        ])
        assert scorer.soft_total(balanced) == pytest.approx(scorer.score(balanced))
