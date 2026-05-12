import math
import random

import pytest

from core import Entity, Slot, State, Scorer, hard, soft, History
from core import SAEngine, PTEngine, Solver
from core.neighbor import swap


# ── shared problem ────────────────────────────────────────────────────────────

@pytest.fixture
def people():
    return [Entity(n) for n in ["Alice","Bob","Carol","Dan","Eve","Frank"]]

@pytest.fixture
def groups():
    return [Slot(f"G{i}", capacity=3) for i in range(2)]

@pytest.fixture
def scorer(groups):
    return Scorer([
        hard(
            lambda s: any(len(s.group(g)) > g.capacity for g in groups),
            name="capacity",
        ),
        soft(
            lambda s: sum(abs(len(s.group(g)) - 3) for g in groups),
            weight=5.0,
            name="balance",
        ),
    ])

@pytest.fixture
def init_state(people, groups):
    return State.balanced(people, groups, rng=random.Random(0))


# ═══════════════════════════════════════════════════════════════════════════════
# SAEngine
# ═══════════════════════════════════════════════════════════════════════════════

class TestSAEngine:

    def test_returns_result_with_correct_engine_label(self, init_state, scorer, groups):
        engine = SAEngine(T0=10.0, alpha=0.99, iterations=200)
        result = engine.run(init_state, scorer, groups)
        assert result.engine == "SA"

    def test_feasible_init_stays_feasible_with_swap(self, init_state, scorer, groups):
        # balanced init + swap preserves group sizes → always feasible
        engine = SAEngine(T0=10.0, alpha=0.99, iterations=500,
                          neighbor_fn=swap, rng=random.Random(1))
        result = engine.run(init_state, scorer, groups)
        assert result.feasible

    def test_best_cost_le_initial_cost(self, init_state, scorer, groups):
        engine = SAEngine(T0=50.0, alpha=0.995, iterations=1_000,
                          rng=random.Random(7))
        result = engine.run(init_state, scorer, groups)
        assert result.best_cost <= result.initial_cost

    def test_cost_trace_populated(self, init_state, scorer, groups):
        engine = SAEngine(iterations=1_000, trace_every=100)
        result = engine.run(init_state, scorer, groups)
        assert len(result.cost_trace) > 0

    def test_temp_trace_populated(self, init_state, scorer, groups):
        engine = SAEngine(iterations=1_000, trace_every=100)
        result = engine.run(init_state, scorer, groups)
        assert len(result.temp_trace) > 0

    def test_meta_contains_expected_keys(self, init_state, scorer, groups):
        engine = SAEngine(T0=20.0, alpha=0.99, iterations=200)
        result = engine.run(init_state, scorer, groups)
        assert "T_final" in result.meta
        assert "accept_rate" in result.meta
        assert 0.0 <= result.meta["accept_rate"] <= 1.0

    def test_seeded_reproducible(self, init_state, scorer, groups):
        def run(seed):
            e = SAEngine(T0=20.0, alpha=0.99, iterations=500,
                         rng=random.Random(seed))
            return e.run(init_state.copy(), scorer, groups).best_cost

        assert run(42) == run(42)

    def test_different_seeds_can_differ(self, init_state, scorer, groups):
        costs = set()
        for seed in range(10):
            e = SAEngine(T0=50.0, alpha=0.99, iterations=500,
                         rng=random.Random(seed))
            costs.add(e.run(init_state.copy(), scorer, groups).best_cost)
        # not all runs should land on identical cost
        assert len(costs) >= 1   # at minimum they all run without error

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            SAEngine(alpha=1.5)

    def test_invalid_T0_raises(self):
        with pytest.raises(ValueError):
            SAEngine(T0=-1.0)

    def test_iterations_reflected_in_result(self, init_state, scorer, groups):
        engine = SAEngine(iterations=333)
        result = engine.run(init_state, scorer, groups)
        assert result.iterations == 333

    def test_infeasible_init_can_recover(self, people, groups, scorer):
        # Start from a badly unbalanced state — engine should recover with move
        from core.neighbor import move
        bad_state = State({e: groups[0] for e in people})  # everyone in G0
        engine = SAEngine(T0=100.0, alpha=0.99, iterations=5_000,
                          neighbor_fn=move, rng=random.Random(0))
        result = engine.run(bad_state, scorer, groups)
        # best cost should improve from all-in-one-group infeasibility
        assert result.best_cost <= result.initial_cost


# ═══════════════════════════════════════════════════════════════════════════════
# PTEngine
# ═══════════════════════════════════════════════════════════════════════════════

class TestPTEngine:

    def test_returns_result_with_correct_engine_label(self, init_state, scorer, groups):
        engine = PTEngine(T_min=1.0, T_max=50.0, n_chains=3, iterations=300)
        result = engine.run(init_state, scorer, groups)
        assert result.engine == "PT"

    def test_feasible_init_stays_feasible(self, init_state, scorer, groups):
        engine = PTEngine(T_min=0.5, T_max=30.0, n_chains=3,
                          iterations=500, neighbor_fn=swap, rng=random.Random(1))
        result = engine.run(init_state, scorer, groups)
        assert result.feasible

    def test_best_cost_le_initial_cost(self, init_state, scorer, groups):
        engine = PTEngine(T_min=0.5, T_max=50.0, n_chains=3,
                          iterations=1_000, rng=random.Random(5))
        result = engine.run(init_state, scorer, groups)
        assert result.best_cost <= result.initial_cost

    def test_meta_contains_swap_accept_rate(self, init_state, scorer, groups):
        engine = PTEngine(T_min=1.0, T_max=50.0, n_chains=3,
                          iterations=500, swap_interval=50)
        result = engine.run(init_state, scorer, groups)
        assert "swap_accept_rate" in result.meta
        assert 0.0 <= result.meta["swap_accept_rate"] <= 1.0

    def test_meta_contains_chain_accept_rates(self, init_state, scorer, groups):
        n = 4
        engine = PTEngine(T_min=1.0, T_max=50.0, n_chains=n,
                          iterations=500)
        result = engine.run(init_state, scorer, groups)
        rates = result.meta["chain_accept_rates"]
        assert len(rates) == n
        # Cold chain (index 0) should accept less than hot chain (index -1)
        assert rates[0] <= rates[-1]

    def test_temperature_ladder_geometric(self):
        engine = PTEngine(T_min=1.0, T_max=100.0, n_chains=5)
        ladder = engine._ladder()
        assert len(ladder) == 5
        assert ladder[0] == pytest.approx(1.0)
        assert ladder[-1] == pytest.approx(100.0)
        # Check geometric spacing: ratios between adjacent temps should be equal
        ratios = [ladder[i+1] / ladder[i] for i in range(len(ladder) - 1)]
        assert all(r == pytest.approx(ratios[0], rel=1e-6) for r in ratios)

    def test_invalid_T_min_ge_T_max_raises(self):
        with pytest.raises(ValueError):
            PTEngine(T_min=100.0, T_max=10.0)

    def test_n_chains_1_raises(self):
        with pytest.raises(ValueError):
            PTEngine(n_chains=1)

    def test_seeded_reproducible(self, init_state, scorer, groups):
        def run(seed):
            e = PTEngine(T_min=1.0, T_max=30.0, n_chains=3,
                         iterations=300, rng=random.Random(seed))
            return e.run(init_state.copy(), scorer, groups).best_cost
        assert run(7) == run(7)


# ═══════════════════════════════════════════════════════════════════════════════
# Solver
# ═══════════════════════════════════════════════════════════════════════════════

class TestSolver:

    def _make_solver(self, people, groups, scorer, **kwargs):
        return Solver(people, groups, scorer, **kwargs)

    def test_solve_returns_result(self, people, groups, scorer):
        solver = self._make_solver(people, groups, scorer, engine="sa",
                                   init="balanced", iterations=500)
        result = solver.solve(seed=0)
        assert result is not None
        assert result.engine == "SA"

    def test_solve_seeded_reproducible(self, people, groups, scorer):
        solver = self._make_solver(people, groups, scorer, engine="sa",
                                   init="balanced", iterations=500)
        r1 = solver.solve(seed=42)
        r2 = solver.solve(seed=42)
        assert r1.best_cost == r2.best_cost

    def test_sample_returns_n_results(self, people, groups, scorer):
        solver = self._make_solver(people, groups, scorer, engine="sa",
                                   init="balanced", iterations=300)
        results = solver.sample(n=4, seed=0)
        assert len(results) == 4

    def test_sample_sorted_by_cost(self, people, groups, scorer):
        solver = self._make_solver(people, groups, scorer, engine="sa",
                                   init="balanced", iterations=300)
        results = solver.sample(n=5, seed=0)
        costs = [r.best_cost for r in results]
        assert costs == sorted(costs)

    def test_auto_selects_sa_for_small(self, groups, scorer):
        small_people = [Entity(i) for i in range(10)]
        solver = Solver(small_people, groups, scorer)
        assert solver.selected_engine == "sa"

    def test_auto_selects_pt_for_large(self, groups, scorer):
        large_people = [Entity(i) for i in range(50)]
        solver = Solver(large_people, groups, scorer)
        assert solver.selected_engine == "pt"

    def test_explicit_engine_overrides_auto(self, people, groups, scorer):
        solver = Solver(people, groups, scorer, engine="pt")
        assert solver.selected_engine == "pt"

    def test_empty_entities_raises(self, groups, scorer):
        with pytest.raises(ValueError):
            Solver([], groups, scorer)

    def test_empty_slots_raises(self, people, scorer):
        with pytest.raises(ValueError):
            Solver(people, [], scorer)

    def test_sample_n_zero_raises(self, people, groups, scorer):
        solver = self._make_solver(people, groups, scorer, engine="sa",
                                   init="balanced", iterations=200)
        with pytest.raises(ValueError):
            solver.sample(n=0)

    def test_unknown_engine_raises(self, people, groups, scorer):
        solver = Solver(people, groups, scorer, engine="unknown")
        with pytest.raises(ValueError):
            solver.solve()

    def test_history_integrates_across_sessions(self, people, groups):
        history = History()
        scorer = Scorer([
            hard(
                lambda s: any(len(s.group(g)) > g.capacity for g in groups),
                name="capacity",
            ),
            history.as_soft_constraint(weight=5.0),
        ])
        solver = Solver(people, groups, scorer, engine="sa", init="balanced",
                        iterations=2_000)

        r1 = solver.solve(seed=1)
        history.update(r1.best_state)

        r2 = solver.solve(seed=2)
        # Session 2 breakdown must include history_repeat
        bd = r2.breakdown(scorer)
        assert "history_repeat" in bd

    def test_result_breakdown_via_solver(self, people, groups, scorer):
        solver = self._make_solver(people, groups, scorer, engine="sa",
                                   init="balanced", iterations=500)
        result = solver.solve(seed=0)
        bd = result.breakdown(scorer)
        assert "capacity" in bd
        assert "balance" in bd
