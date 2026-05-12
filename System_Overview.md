# System Overview

> This document is for contributors and maintainers — people working *inside*
> the codebase. For usage, installation, and examples see `README.md`.
> For planned features and open questions see `ROADMAP.md`.

---

## 1. What This System Is

A general-purpose **constraint-based arrangement engine**.

Given a set of entities (people, classes, jobs), a set of slots (groups,
rooms, timeslots, roles), and a user-defined constraint set, the engine finds
an assignment of entities to slots that minimises a cost function composed
of those constraints.

The system is intentionally domain-agnostic. It does not know what a
"timetable" or a "group rotation" is. It only knows:

- a **State** (a complete assignment)
- a **cost function** C(S) (a weighted sum of constraint penalties)
- a **search strategy** (how to walk the state space)

Domain logic lives entirely in the constraint functions the user provides.

---

## 2. Mathematical Foundation

### 2.1 The optimisation problem

Let:
- **E** = set of entities, |E| = n
- **K** = set of slots, |K| = k
- **S** = an assignment S: E → K (the state space has size kⁿ)

We want:

```
S* = argmin C(S)
         S

where C(S) = ∞                    if any hard constraint fires
             Σᵢ wᵢ · fᵢ(S)       otherwise
```

Each fᵢ: State → ℝ≥0 is a constraint function. Hard constraints are
modelled as soft constraints with weight ∞, but are evaluated separately
for short-circuit efficiency.

### 2.2 Search strategy: Simulated Annealing (baseline engine)

SA treats the problem as sampling from a Boltzmann distribution:

```
π_T(S) ∝ exp(−C(S) / T)
```

As T → 0, this distribution concentrates on states that minimise C(S).

The algorithm is a time-inhomogeneous Markov chain:

```
S₀ ← init()
for k = 0, 1, ..., N:
    S' ← neighbor(Sₖ)
    ΔC = C(S') − C(Sₖ)
    if ΔC < 0  or  Uniform(0,1) < exp(−ΔC / Tₖ):
        Sₖ₊₁ ← S'
    else:
        Sₖ₊₁ ← Sₖ
    Tₖ₊₁ ← Tₖ · α          (geometric cooling schedule)
```

The **acceptance probability** `exp(−ΔC / T)` guarantees ergodicity:
any state is reachable from any other with positive probability, so the
chain does not get permanently trapped in local minima.

**Cooling schedule.** Geometric: T(k) = T₀ · αᵏ.
The logarithmic schedule T(k) = C / log(1+k) gives a theoretical
convergence guarantee but requires impractically many iterations.

**Rule of thumb for T₀:** set it so that `exp(−ΔC_typical / T₀) ≈ 0.8`,
i.e. accept ~80% of bad moves at the start.

**Infeasible state handling.** When C(S) = ∞ (hard constraint violated),
the standard Boltzmann comparison produces `∞ − ∞ = nan`, which causes
the engine to freeze. The implementation falls back to soft-cost comparison
when the current state is infeasible, letting the chain navigate toward
feasibility before normal Boltzmann acceptance resumes:

```
if current_cost == inf:
    if candidate_cost < inf:   always accept (take any feasible state)
    else:                      compare soft costs, apply Boltzmann
else:
    standard ΔC Boltzmann acceptance
```

### 2.3 Search strategy: Parallel Tempering (production engine)

Run K chains simultaneously at a temperature ladder T₁ < T₂ < ... < T_K.
Every `swap_interval` steps, propose swaps between adjacent chains i and i+1:

```
P(swap) = min(1, exp((1/Tᵢ − 1/Tᵢ₊₁) · (C(Sᵢ) − C(Sᵢ₊₁))))
```

The hot chains explore freely; the cold chains exploit. Good solutions
propagate down the ladder. This directly addresses the spectral gap
problem that limits single-chain SA on multimodal cost landscapes.

**Feasibility-aware swap.** When chain costs involve `∞`, the standard
log-accept formula produces `nan`. The implementation handles this explicitly:

| Cold chain | Hot chain | Action |
|---|---|---|
| infeasible (∞) | infeasible (∞) | skip (no benefit from swapping two bad states) |
| infeasible (∞) | feasible | always swap (move good state down the ladder) |
| feasible | infeasible (∞) | never swap (protect the cold chain) |
| feasible | feasible | standard MH criterion |

**Best tracking.** The best state is recorded both after the SA step on
each chain *and* after each swap round. A swap that delivers a better state
to the cold chain is captured immediately, not deferred to the next iteration.

### 2.4 History: the co-occurrence penalty

For arrangement problems with temporal memory (group rotation, job scheduling
across sessions), the system maintains a co-occurrence matrix:

```
history[i][j] += 1   each time entities i and j appear in the same slot
```

The history penalty for a state S is:

```
f_history(S) = Σ  history[eᵢ.id][eⱼ.id]
               for all slots g
               for all pairs (eᵢ, eⱼ) in S.group(g)
```

This is a standard soft constraint that closes over a `History` object —
the engine has no special awareness of history. History is updated after
each accepted final arrangement, not during the search.

---

## 3. Component Architecture

```
quench/
│
│  ┌─ core/ ──────────────────────────────────────────────────────┐
│  │                                                               │
│  │   __init__.py       Public import surface — re-exports       │
│  │                     Entity, Slot, State, Scorer, hard,       │
│  │                     soft, History, Solver, swap, move,       │
│  │                     mixed, make_swap_k, NeighborFn, Result   │
│  │   state.py          Entity, Slot, State                      │
│  │   constraints.py    Constraint, HardConstraint,              │
│  │                     SoftConstraint, hard(), soft()           │
│  │   scorer.py         Scorer                                   │
│  │   neighbor.py       swap(), move(), mixed(),                 │
│  │                     make_swap_k(), NeighborFn                │
│  │   history.py        History (co-occurrence matrix)           │
│  │   result.py         Result (shared engine output type)       │
│  │   engine_sa.py      SAEngine                                 │
│  │   engine_pt.py      PTEngine                                 │
│  │   solver.py         Solver (public API surface)              │
│  │                                                               │
│  └───────────────────────────────────────────────────────────────┘
│
├── examples/
│   ├── timetable.py              spatial cost (walking distance)
│   ├── group_rotation.py         history-aware job/group diversity
│   └── pharmacy/
│       ├── pharmacy_schedule.py  real-world: 15-staff inpatient pharmacy rotation
│       ├── old_schedule.csv      customer's original schedule (reference)
│       └── prompt.txt            original customer request + requirement→constraint map
│
├── llm/
│   └── README.md           stub for natural language constraint layer
│
└── tests/
    ├── test_state.py
    ├── test_constraints_scorer.py
    ├── test_neighbor.py
    ├── test_history.py
    ├── test_engines_solver.py
    └── test_edge_cases.py      regression tests for all patched bugs
```

### 3.1 Component responsibilities

| Component | Owns | Does not own |
|---|---|---|
| `__init__.py` | public re-export surface — everything a user needs is importable from `core` directly | implementation |
| `state.py` | assignment data structure, mutation primitives, `random()` / `balanced()` factories | cost, constraints |
| `constraints.py` | constraint type hierarchy, `hard()` / `soft()` constructors | evaluation order, aggregation |
| `scorer.py` | cost aggregation, hard gating, diagnostics, name-safe breakdown | search, state mutation |
| `neighbor.py` | perturbation functions | acceptance decision, temperature |
| `history.py` | co-occurrence matrix, session tracking, JSON persistence | constraint definition |
| `result.py` | engine-agnostic output struct, convergence diagnostics | engine internals |
| `engine_sa.py` | SA walk, Boltzmann acceptance, soft fallback for infeasible states, cooling | constraint definition, init |
| `engine_pt.py` | parallel chains, feasibility-aware MH swaps, best tracking after swaps | constraint definition, init |
| `solver.py` | engine selection, init strategy, kwarg routing with warnings, `.sample()` / `.solve()` | engine internals |

### 3.2 Data flow

```
User defines:
  entities, slots, scorer (constraints)

         │
         ▼
    Solver.solve(seed?)
         │
         ├── _init_state()           State.random() or State.balanced()
         ├── _build_engine()         SAEngine or PTEngine (with kwarg warning)
         │
         └── engine.run(state, scorer, slots)
               │
               ├── neighbor(state, slots, rng)   propose S'
               ├── scorer.score(S')              compute C(S')
               ├── accept / reject               Boltzmann or soft fallback
               ├── track best_state              after SA step + after swaps (PT)
               └── T *= alpha  /  swap chains
                        │
                        ▼
               Result(
                 best_state, best_cost, initial_cost,
                 cost_trace, temp_trace,
                 iterations, engine, meta
               )
```

---

## 4. Key Design Decisions

### 4.1 State is mutable with copy-on-propose

State mutation (`swap`, `assign`) is in-place. The engine calls `.copy()`
before perturbing, then keeps the copy on accept or discards it on reject.
This avoids O(n) allocations on every rejection while keeping accept/reject
logic clean.

**Implication for subclassers:** if you hold external references to a
State object during a run, call `.copy()` explicitly. The engine does
not guarantee state stability between steps.

### 4.2 Two initialisation strategies — and their coupling to the neighbor function

`State.random()` assigns each entity to a uniformly random slot. This
produces a Poisson-distributed group size which may be far from balanced.

`State.balanced()` shuffles entities then assigns round-robin (slot[i % k]),
producing the most even possible distribution (⌊n/k⌋ or ⌈n/k⌉ per slot).

**The critical coupling:** `swap` is a size-invariant operation — it
exchanges entities between slots but never changes how many are in each
slot. If `State.random()` produces a 7-1-4 distribution across three
groups of capacity 4, no sequence of swaps can reach a balanced 4-4-4
state. The engine would be permanently infeasible.

Rule:
- `init="balanced"` + `swap` for equal-capacity group problems
- `init="random"` + `move` for variable-capacity or structurally complex problems

### 4.3 Hard constraints short-circuit; infeasible states use a soft fallback

Hard constraints are evaluated first in a short-circuiting loop: the first
violation returns `inf` immediately without computing any soft costs.

When the current state is infeasible (`C = ∞`), standard Boltzmann
acceptance breaks because `∞ − ∞ = nan`. Both engines handle this with a
soft-cost fallback: compare soft costs only, apply Boltzmann to that
difference. This lets the chain navigate toward feasibility rather than
freezing permanently at its starting infeasible state.

### 4.4 Hard constraints are not infinite-weight soft constraints

`Scorer._hard` and `Scorer._soft` are separate lists evaluated
sequentially, not merged into one sorted list. This is both a performance
decision (short-circuit avoids unnecessary soft evaluation) and a semantic
one: a soft constraint with high weight can still be probabilistically
violated; a hard constraint cannot.

### 4.5 History is a data source, not a special engine feature

The co-occurrence penalty is an ordinary soft constraint that closes over
a `History` object. The engine has no awareness of history. This means:

- History works identically with SA, PT, or any future engine
- The LLM layer can explain the history penalty via `scorer.breakdown()`
  without any special-casing
- History can be replaced with any temporal data source by passing a
  different closure to `soft()`
- The constraint reflects the live state of the matrix — updating history
  between sessions automatically makes the same constraint object stricter

### 4.6 Scorer.breakdown() deduplicates constraint names

If two constraints share the same name, a plain dict would silently
overwrite the first entry, losing data. `breakdown()` appends a numeric
suffix to duplicates: `"cost"`, `"cost_1"`, `"cost_2"`. The same
deduplication applies to `soft_breakdown()`.

### 4.7 Sampler vs optimiser: both are first-class

`Solver.sample(n)` returns n Results sorted by cost — the distribution
of good solutions found. `Solver.solve()` returns a single best Result.
Both call the same engine; `solve()` is effectively `sample(n=1)[0]`.

This matters for the planned LLM layer: presenting top-3 arrangements
with explanations requires the distribution, not just the optimum.

### 4.8 The neighbor function is injected, not hardcoded

Engines accept a `neighbor_fn` parameter (type `NeighborFn`). This is
the primary extension point for domain-specific tuning without touching
engine internals. Built-in options:

| Function | Behaviour | Use when |
|---|---|---|
| `swap` (default) | Exchange two entities' slots | Equal group sizes required |
| `move` | Reassign one entity randomly | Variable group sizes |
| `mixed` | 70% swap, 30% move | Soft capacity constraints |
| `make_swap_k(k)` | Swap k pairs simultaneously | Large k for hot exploration |

### 4.9 Solver warns on mismatched kwargs

Passing a PT-only kwarg (`T_min`, `T_max`, `n_chains`) to an SA solver,
or an SA-only kwarg (`T0`, `alpha`) to a PT solver, is a likely mistake.
Rather than silently dropping or hard-erroring, the solver emits a
`UserWarning` naming the ignored parameters and suggesting the correct
engine. This surfaces configuration mistakes without breaking code.

### 4.10 Result is engine-agnostic

All engines return the same `Result` dataclass. The Solver, examples,
tests, and the future LLM layer are all engine-agnostic as a consequence.
Switching an engine does not require changing any downstream code.

---

## 5. Known Limitations

### History save/load requires string entity IDs

`History.save()` serialises to JSON. JSON requires string keys, so all
entity IDs are coerced via `str()`. On `History.load()`, keys are always
strings. If your entity IDs are non-string (integers, tuples, etc.),
lookups after a roundtrip will miss:

```python
# entity.id = 1 (int) → saved as "1" → loaded key is "1"
# h2.co_occurrence(Entity(1), Entity(2)) looks up _m[1][2] → miss → returns 0
```

**Fix:** use string IDs for entities that will be persisted:
```python
Entity(str(user_id))   # survives save/load
Entity(user_id)        # int IDs silently break after load
```

This is a JSON constraint, not a fixable bug. It is covered by a
documented test in `test_edge_cases.py::TestHistoryNonStringIDs`.

### Soft constraint functions are trusted to return non-negative values

`SoftConstraint` does not validate its `fn`'s return value. A function
returning a negative penalty is accepted silently, which makes the global
minimum unbounded. This is a caller contract, not a runtime check — the
cost would be prohibitive on the hot path.

### swap is size-invariant (see §4.2)

Using `init="random"` with `swap` as the neighbor function on a
capacity-constrained equal-size group problem will almost certainly produce
permanent infeasibility if the random init is unbalanced. This is
documented in `State.random()` and tested in the edge case suite.

### core/__init__.py was missing public exports (now fixed)

`core/__init__.py` originally re-exported only the constraint/state types:
`Entity`, `Slot`, `State`, `Scorer`, `hard`, `soft`. `History`, `Solver`,
and all neighbor functions (`swap`, `move`, `mixed`, `make_swap_k`,
`NeighborFn`, `Result`) were not exported.

This caused both bundled examples to fail immediately with:

```
ImportError: cannot import name 'History' from 'core'
ImportError: cannot import name 'Solver' from 'core'
```

**Fix:** all public symbols are now re-exported from `__init__.py`.
The omission was not caught by tests because no test used the
`from core import ...` pattern — all tests import directly from
sub-modules (`from core.history import History`). If you add new public
types in future, update both `__init__.py` and this list.

### Windows console encoding for Unicode output (platform issue)

On Windows with a non-UTF-8 console encoding (e.g. cp874, cp1252), printing
box-drawing characters (`═`, `─`, `█`) raises `UnicodeEncodeError` at
runtime. This is a Python/Windows terminal issue, not a library bug, but
it silently breaks any script that prints formatted output.

**Fix applied to examples:** both `group_rotation.py` and `timetable.py`
now call `sys.stdout.reconfigure(encoding='utf-8')` immediately after the
`sys.path` insertion. This forces UTF-8 output regardless of the console
code page.

**If you write new scripts** that print Unicode: add the same line, or set
`PYTHONIOENCODING=utf-8` in your shell environment. The core library itself
prints nothing and is not affected.

---

## 6. What Is Intentionally Not Here

### CP-SAT / exact solvers

For small problems (n < ~100 entities), Google OR-Tools CP-SAT would
find the true optimum faster than any metaheuristic. Excluded from v1
because it introduces a heavy external dependency, is a fundamentally
different paradigm (exact, not probabilistic), and does not produce a
distribution. Natural candidate for `engine_cpsat.py` in a future version.

### Gradient / derivative information

SA and PT are zero-order methods — they only evaluate C(S), never ∇C(S).
The state space is discrete, so gradients are not defined in the standard
sense. Deterministic Annealing via mean-field approximation is theoretically
interesting but outside this library's scope.

### Built-in constraint library

The library ships with no predefined constraints beyond the base classes.
Predefined constraints would imply a domain model the library deliberately
does not have. See `examples/` for concrete domain-specific constraint sets.

---

## 7. Extension Points

### Adding a new engine

1. Create `core/engine_<name>.py`
2. Implement `run(initial_state, scorer, slots) -> Result`
3. Add to `solver.py` `_build_engine()` with an allowlisted kwarg set
4. Add to `core/__init__.py` exports
5. Add tests to `tests/test_engines_solver.py`

Engine contract: given a `State`, a `Scorer`, and the slot list, return a
`Result` with `best_state`, `best_cost`, `cost_trace`, and `engine` set
to a short identifier string. Everything else is internal.

### Adding a new constraint type

Subclass `Constraint` directly when you need a stateful constraint:

```python
class HistoryConstraint(Constraint):
    def __init__(self, history: History, weight: float):
        super().__init__(name="history")
        self._h = history
        self.weight = weight

    @property
    def is_hard(self) -> bool:
        return False

    def __call__(self, state) -> float:
        return self.weight * self._h.penalty(state)
```

### Incremental scoring

The default `scorer.delta()` re-scores both states in full — correct but
O(n). For large n, subclass `Scorer` and override `delta()`:

```python
class IncrementalScorer(Scorer):
    def delta(self, old_state, new_state) -> float:
        changed = [e for e in new_state.entities()
                   if new_state[e] != old_state[e]]
        # recompute only terms touching changed entities
        ...
```

This reduces per-step cost from O(n) to O(1) for swap-based perturbations,
and matters at n > ~500.

---

## 8. Assumptions

- **One slot per entity.** Every entity is assigned to exactly one slot.
  Multi-slot assignments require decomposing into sub-entities beforehand.

- **Slots are static.** The set of available slots does not change during
  a run. Dynamic slot creation is not supported.

- **Costs are non-negative.** Soft constraint functions must return ≥ 0.
  Negative penalties make the global minimum unbounded. Not runtime-checked.

- **The cost function is cheap to evaluate.** SA and PT call `scorer.score()`
  on every proposed neighbor. Expensive constraint functions will bottleneck
  the engine. Use incremental scoring or pre-compute inputs outside lambdas.

- **Entity IDs are strings when persistence is needed.** `History.save()`
  coerces all IDs to strings. Non-string IDs will not survive a save/load
  roundtrip (see §5).

---

## 9. Dependency Policy

`core/` has zero external dependencies. Pure Python stdlib only.

`examples/` may use numpy for distance calculations.

`llm/` will depend on the Anthropic SDK when implemented.

`tests/` uses pytest only.

This policy ensures the engine is portable and auditable without installing
anything beyond pytest.

---

## 10. Test Suite

170 tests across 6 files. All pass in < 1 second.

```
pytest tests/
```

| File | What it covers |
|---|---|
| `test_state.py` | Entity/Slot equality semantics, `random()` and `balanced()` factories, mutation primitives, copy independence |
| `test_constraints_scorer.py` | Hard gating, short-circuit, soft weighting, `breakdown()`, `delta()`, `violated_hard()`, `soft_total()` |
| `test_neighbor.py` | `swap` size-invariance, `move` size-variance, `mixed` distribution, `make_swap_k` correctness |
| `test_history.py` | Co-occurrence accumulation, symmetry, cross-slot exclusion, save/load roundtrip, live-closure `as_soft_constraint` |
| `test_engines_solver.py` | SA/PT feasibility, cost monotonicity, meta keys, reproducibility, auto engine selection, `sample()` sorting |
| `test_edge_cases.py` | All 10 patched bugs: `iterations=0`, `trace_every=0`, `swap_interval=0`, unknown init strategy, duplicate constraint names, empty slots, PT swap timing, History int-ID roundtrip, negative soft penalties, Result properties |

### Examples as real-world validation

The three examples in `examples/` double as integration tests for the full pipeline:

| Example | Domain | Key techniques |
|---|---|---|
| `group_rotation.py` | Activity group cycling (20 kids, 4 groups, 3 sessions) | `History.as_soft_constraint`, role-repeat penalty, multi-session loop |
| `timetable.py` | School timetabling (10 courses, 36 slots) | Spatial cost via `Slot.meta`, `move` neighbor, `solver.sample()` |
| `pharmacy/pharmacy_schedule.py` | Hospital inpatient pharmacy (15 staff, 15 positions, 4 weeks) | Fixed-position pattern, custom task-group history, three independent soft constraints, CSV export |

The pharmacy example (`pharmacy_schedule.py`) is the first production use case built from a real customer request. It demonstrates a pattern not covered by the other examples: **separating fixed staff from the rotatable pool** — the solver only manages the 13 rotatable (entity, slot) pairs, then the fixed assignments are merged at output time. This avoids wasting engine iterations trying to move staff whose positions are non-negotiable.