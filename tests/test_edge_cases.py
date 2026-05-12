"""
Edge case and regression tests.
Each test here maps to a confirmed bug or a previously untested property.
"""
import math
import random
import warnings

import pytest

from core import (
    Entity, Slot, State, Scorer, hard, soft,
    History, SAEngine, PTEngine, Solver, Result,
)
from core.neighbor import swap


# ── shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def people():
    return [Entity(n) for n in ["A", "B", "C", "D"]]

@pytest.fixture
def groups():
    return [Slot("G0", capacity=2), Slot("G1", capacity=2)]

@pytest.fixture
def state(people, groups):
    return State.balanced(people, groups, rng=random.Random(0))

@pytest.fixture
def empty_scorer():
    return Scorer()


# ═══════════════════════════════════════════════════════════════════════════════
# BUG 1, 2 — SAEngine: iterations=0 and trace_every=0 raise cleanly
# ═══════════════════════════════════════════════════════════════════════════════

class TestSAEngineValidation:
    def test_iterations_zero_raises(self):
        with pytest.raises(ValueError, match="iterations"):
            SAEngine(iterations=0)

    def test_iterations_negative_raises(self):
        with pytest.raises(ValueError, match="iterations"):
            SAEngine(iterations=-1)

    def test_trace_every_zero_raises(self):
        with pytest.raises(ValueError, match="trace_every"):
            SAEngine(trace_every=0)

    def test_trace_every_negative_raises(self):
        with pytest.raises(ValueError, match="trace_every"):
            SAEngine(trace_every=-5)


# ═══════════════════════════════════════════════════════════════════════════════
# BUG 3 — PTEngine: swap_interval=0 and other invalid args raise cleanly
# ═══════════════════════════════════════════════════════════════════════════════

class TestPTEngineValidation:
    def test_swap_interval_zero_raises(self):
        with pytest.raises(ValueError, match="swap_interval"):
            PTEngine(n_chains=2, swap_interval=0)

    def test_swap_interval_negative_raises(self):
        with pytest.raises(ValueError, match="swap_interval"):
            PTEngine(n_chains=2, swap_interval=-1)

    def test_iterations_zero_raises(self):
        with pytest.raises(ValueError, match="iterations"):
            PTEngine(n_chains=2, iterations=0)

    def test_trace_every_zero_raises(self):
        with pytest.raises(ValueError, match="trace_every"):
            PTEngine(n_chains=2, trace_every=0)


# ═══════════════════════════════════════════════════════════════════════════════
# BUG 4 — Solver: unknown init strategy raises ValueError
# ═══════════════════════════════════════════════════════════════════════════════

class TestSolverInitValidation:
    def test_unknown_init_raises_on_solve(self, people, groups, empty_scorer):
        solver = Solver(people, groups, empty_scorer, init="typo")
        with pytest.raises(ValueError, match="init strategy"):
            solver.solve()

    def test_known_init_random_works(self, people, groups, empty_scorer):
        solver = Solver(people, groups, empty_scorer, init="random", iterations=10)
        result = solver.solve()
        assert result is not None

    def test_known_init_balanced_works(self, people, groups, empty_scorer):
        solver = Solver(people, groups, empty_scorer, init="balanced", iterations=10)
        result = solver.solve()
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════════
# BUG 5 — Scorer.breakdown(): duplicate names get numeric suffix, not overwritten
# ═══════════════════════════════════════════════════════════════════════════════

class TestScorerDuplicateNames:
    def test_duplicate_soft_names_both_appear(self, state):
        scorer = Scorer([
            soft(lambda s: 1.0, name="cost"),
            soft(lambda s: 2.0, name="cost"),
        ])
        bd = scorer.breakdown(state)
        assert len(bd) == 2
        assert "cost" in bd
        assert "cost_1" in bd

    def test_duplicate_hard_names_both_appear(self, state):
        scorer = Scorer([
            hard(lambda s: False, name="cap"),
            hard(lambda s: False, name="cap"),
        ])
        bd = scorer.breakdown(state)
        assert len(bd) == 2
        assert "cap" in bd
        assert "cap_1" in bd

    def test_three_duplicates_get_incremental_suffixes(self, state):
        scorer = Scorer([
            soft(lambda s: 1.0, name="x"),
            soft(lambda s: 2.0, name="x"),
            soft(lambda s: 3.0, name="x"),
        ])
        bd = scorer.breakdown(state)
        assert set(bd.keys()) == {"x", "x_1", "x_2"}

    def test_unique_names_unchanged(self, state):
        scorer = Scorer([
            soft(lambda s: 1.0, name="alpha"),
            soft(lambda s: 2.0, name="beta"),
        ])
        bd = scorer.breakdown(state)
        assert "alpha" in bd
        assert "beta" in bd
        assert len(bd) == 2

    def test_soft_breakdown_also_deduplicates(self, state):
        scorer = Scorer([
            soft(lambda s: 1.0, name="dup"),
            soft(lambda s: 2.0, name="dup"),
        ])
        bd = scorer.soft_breakdown(state)
        assert len(bd) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# BUG 6 — SoftConstraint: negative penalty propagates; documented not trapped
# ═══════════════════════════════════════════════════════════════════════════════

class TestNegativePenalty:
    def test_negative_soft_penalty_propagates_to_score(self, state):
        # Library does not trap negative penalties — caller's responsibility.
        # This test documents the current behaviour so any future change is explicit.
        scorer = Scorer([soft(lambda s: -5.0, name="negative")])
        assert scorer.score(state) == pytest.approx(-5.0)


# ═══════════════════════════════════════════════════════════════════════════════
# BUG 7, 8 — State.random() and State.balanced() raise on empty slots
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmptySlots:
    def test_random_empty_slots_raises(self, people):
        with pytest.raises(ValueError, match="empty"):
            State.random(people, [])

    def test_balanced_empty_slots_raises(self, people):
        with pytest.raises(ValueError, match="empty"):
            State.balanced(people, [])

    def test_random_empty_entities_returns_empty_state(self, groups):
        s = State.random([], groups)
        assert len(s) == 0

    def test_balanced_empty_entities_returns_empty_state(self, groups):
        s = State.balanced([], groups)
        assert len(s) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# BUG 9 — PTEngine: best state tracked after swap step too
# ═══════════════════════════════════════════════════════════════════════════════

class TestPTBestTracking:
    def test_best_cost_never_exceeds_initial_regardless_of_swap_timing(
        self, people, groups
    ):
        scorer = Scorer([
            soft(lambda s: float(len(s.group(groups[0]))), name="g0_size")
        ])
        state = State.balanced(people, groups, rng=random.Random(0))
        engine = PTEngine(
            T_min=0.5, T_max=50.0, n_chains=3,
            iterations=500, swap_interval=1,    # swap every step: stress test
            rng=random.Random(1),
        )
        result = engine.run(state, scorer, groups)
        # best_cost must always be <= initial_cost regardless of swap timing
        assert result.best_cost <= result.initial_cost


# ═══════════════════════════════════════════════════════════════════════════════
# BUG 10 — History: non-string IDs do not survive save/load
# ═══════════════════════════════════════════════════════════════════════════════

class TestHistoryNonStringIDs:
    def test_string_ids_survive_roundtrip(self, tmp_path):
        people = [Entity("alice"), Entity("bob"), Entity("carol"), Entity("dan")]
        groups = [Slot("G0", capacity=2), Slot("G1", capacity=2)]
        state = State({
            people[0]: groups[0], people[1]: groups[0],
            people[2]: groups[1], people[3]: groups[1],
        })
        h = History()
        h.update(state)
        path = str(tmp_path / "history.json")
        h.save(path)
        h2 = History.load(path)
        assert h2.co_occurrence(people[0], people[1]) == h.co_occurrence(people[0], people[1])

    def test_int_ids_lose_type_on_load(self, tmp_path):
        # This is EXPECTED and documented behaviour.
        # Confirmed test: int IDs become string after load.
        people = [Entity(0), Entity(1), Entity(2), Entity(3)]
        groups = [Slot("G0", capacity=2), Slot("G1", capacity=2)]
        state = State({
            people[0]: groups[0], people[1]: groups[0],
            people[2]: groups[1], people[3]: groups[1],
        })
        h = History()
        h.update(state)
        assert h.co_occurrence(people[0], people[1]) == 1

        path = str(tmp_path / "int_history.json")
        h.save(path)
        h2 = History.load(path)
        # After load, looking up by int id returns 0 (key became string "0")
        assert h2.co_occurrence(people[0], people[1]) == 0  # known limitation
        # But string-keyed lookup works
        assert h2._m["0"]["1"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Result properties — previously untested
# ═══════════════════════════════════════════════════════════════════════════════

class TestResultProperties:
    def _make_result(self, best_cost, initial_cost, trace=None):
        dummy_state = object()
        return Result(
            best_state=dummy_state,
            best_cost=best_cost,
            initial_cost=initial_cost,
            cost_trace=trace or [],
        )

    def test_feasible_true_when_finite_cost(self):
        r = self._make_result(5.0, 10.0)
        assert r.feasible is True

    def test_feasible_false_when_inf_cost(self):
        r = self._make_result(math.inf, math.inf)
        assert r.feasible is False

    def test_improvement_absolute(self):
        r = self._make_result(best_cost=3.0, initial_cost=10.0)
        assert r.improvement == pytest.approx(7.0)

    def test_improvement_zero_when_no_change(self):
        r = self._make_result(best_cost=5.0, initial_cost=5.0)
        assert r.improvement == pytest.approx(0.0)

    def test_improvement_pct_correct(self):
        r = self._make_result(best_cost=5.0, initial_cost=10.0)
        assert r.improvement_pct == pytest.approx(50.0)

    def test_improvement_pct_zero_when_initial_is_zero(self):
        r = self._make_result(best_cost=0.0, initial_cost=0.0)
        assert r.improvement_pct == 0.0

    def test_improvement_pct_zero_when_initial_is_inf(self):
        r = self._make_result(best_cost=math.inf, initial_cost=math.inf)
        assert r.improvement_pct == 0.0

    def test_converged_true_when_flat_tail(self):
        trace = [10.0] * 5 + [2.0] * 95   # flat final 80%
        r = self._make_result(best_cost=2.0, initial_cost=10.0, trace=trace)
        assert r.converged is True

    def test_converged_false_when_still_dropping(self):
        trace = list(range(100, 0, -1))   # monotonically decreasing
        r = self._make_result(best_cost=1.0, initial_cost=100.0, trace=trace)
        assert r.converged is False

    def test_converged_false_when_trace_too_short(self):
        r = self._make_result(best_cost=1.0, initial_cost=10.0, trace=[5.0] * 5)
        assert r.converged is False

    def test_repr_feasible(self):
        r = self._make_result(best_cost=3.14, initial_cost=10.0)
        r.engine = "SA"
        r.iterations = 1000
        text = repr(r)
        assert "feasible" in text
        assert "SA" in text

    def test_repr_infeasible(self):
        r = self._make_result(best_cost=math.inf, initial_cost=math.inf)
        r.engine = "PT"
        r.iterations = 500
        assert "INFEASIBLE" in repr(r)


# ═══════════════════════════════════════════════════════════════════════════════
# Solver: unknown kwargs emit a warning (not silently dropped)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSolverKwargWarning:
    def test_sa_with_pt_only_kwarg_warns(self, people, groups, empty_scorer):
        solver = Solver(
            people, groups, empty_scorer,
            engine="sa",
            init="balanced",
            iterations=10,
            T_min=1.0,   # PT-only kwarg
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver.solve()
        assert any("T_min" in str(warning.message) for warning in w)

    def test_pt_with_sa_only_kwarg_warns(self, people, groups, empty_scorer):
        solver = Solver(
            people, groups, empty_scorer,
            engine="pt",
            init="balanced",
            iterations=10,
            n_chains=2,
            T0=50.0,     # SA-only kwarg
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver.solve()
        assert any("T0" in str(warning.message) for warning in w)
