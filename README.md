# quench

Arrange people into groups — or schedule things into time slots — while following rules you define.

Built for problems like:

- **Group rotation** — cycle 20 kids through 4 activity groups over several sessions, making sure the same kids don't keep ending up together
- **Timetabling** — assign 10 classes across rooms and periods so no student is double-booked and total walking distance is minimised
- **Any arrangement problem** — staff shifts with "do not pair" rules, sports teams with balance constraints, meeting rooms with attendee requirements

The engine finds the best arrangement it can given your rules. You define what "best" means.

---

## Requirements

Python 3.7+. The core engine has zero external dependencies — pure standard library.

```bash
git clone <repo>
cd quench
```

To run the tests: `pip install pytest`

---

## Try it immediately

Two working examples are included. Run both from the project root:

```bash
python examples/group_rotation.py
python examples/timetable.py
```

### Group rotation

20 kids, 4 activity groups of 5, run across 3 sessions. The engine minimises repeat pairings and role repetition between sessions.

```
Session 1
  Group-Leader       Bob, Grace, Hank, Jack, Tina
  Group-Navigator    Frank, Iris, Leo, Mia, Noah
  Group-Timekeeper   Alice, Carol, Olivia, Paul, Rose
  Group-Scribe       Dan, Eve, Kate, Quinn, Sam

Session 2  (cost reduced 53.8%)
  Group-Leader       Alice, Eve, Grace, Noah, Sam
  ...

Summary: 0 repeat group pairs across 3 sessions
         8 role repeats out of 60 possible
```

### Timetable

10 courses across a 3x3 room grid and 4 periods. Hard constraints prevent double-booking and student conflicts. The soft constraint minimises total student walking distance.

```
              A1     A2     A3     B1       B2      B3    C1   C2   C3
  P1           —      —      —      —       PE  Science    —    —    —
  P2           —      —      —    Art    Music     Math    —    —    —
  P3           —      —      —  Drama  English  History    —    —    —
  P4           —      —      —      —       CS      Geo    —    —    —

Per-student walking distances:
  Grace  ██ 2
  Alice  █ 1
  Bob     0  ...
```

---

## Build your own

### Step 1 — Define entities and slots

**Entities** are the things being assigned (people, courses, jobs).
**Slots** are the places they go (groups, rooms, time periods, roles).

```python
from core import Entity, Slot

people = [Entity("Alice"), Entity("Bob"), Entity("Carol"), Entity("Dan")]
groups = [Slot("Morning", capacity=2), Slot("Afternoon", capacity=2)]
```

Both support optional metadata to carry domain data into constraints:

```python
Slot("Room-B2", capacity=30, meta={"floor": 2, "projector": True})
Entity("Alice", meta={"department": "Engineering", "seniority": 3})
```

### Step 2 — Write your constraints

**Hard constraints** are rules that must never be broken. The engine rejects any arrangement that violates one — full stop.

```python
from core import hard

def no_overflow(state):
    # return True if the constraint is VIOLATED
    return any(len(state.group(g)) > g.capacity for g in groups)

capacity_rule = hard(no_overflow, name="capacity")
```

**Soft constraints** are goals to minimise — the engine reduces them as much as possible.

```python
from core import soft

def prefer_separate(state):
    alice = Entity("Alice")
    bob   = Entity("Bob")
    return 1.0 if state[alice] == state[bob] else 0.0

separation_rule = soft(prefer_separate, weight=5.0, name="alice_bob_apart")
```

`weight` controls relative importance. A weight of 5.0 means this constraint matters five times more than one with weight 1.0.

### Step 3 — Solve

```python
from core import Scorer, Solver

scorer = Scorer([capacity_rule, separation_rule])
solver = Solver(people, groups, scorer, init="balanced")
result = solver.solve(seed=42)

print(result)
# Result(SA, feasible, cost=0.000, improvement=100.0%, iters=50,000)

print(result.breakdown(scorer))
# {'capacity': 0.0, 'alice_bob_apart': 0.0}

for g in groups:
    members = [e.id for e in result.best_state.group(g)]
    print(f"{g.id}: {', '.join(members)}")
# Morning:   Alice, Carol
# Afternoon: Bob, Dan
```

---

## Multiple sessions with history

For group rotation and repeated scheduling problems, the engine can remember who has been paired before and avoid repeating those pairings.

```python
from core import History

history = History()

for session in range(1, 4):
    scorer = Scorer([
        hard(no_overflow, name="capacity"),
        history.as_soft_constraint(weight=3.0),  # penalise repeat pairings
    ])
    solver = Solver(people, groups, scorer, init="balanced")
    result = solver.solve(seed=session)

    print(f"\nSession {session}")
    for g in groups:
        print(f"  {g.id}: {[e.id for e in result.best_state.group(g)]}")

    history.update(result.best_state)  # record who was grouped this session
```

The cost will naturally rise across sessions — that is expected and correct. As more pairings get recorded, fewer fresh combinations remain, so the minimum achievable cost goes up. What matters is that the engine keeps finding the best available arrangement.

History persists between runs:

```python
history.save("history.json")           # save after a session
history = History.load("history.json") # restore in the next run
```

> Use string entity IDs (`Entity("1")` not `Entity(1)`) if you plan to save history. JSON coerces integer keys to strings on load, which breaks lookups.

---

## Reading the result

```python
result.best_state        # the final arrangement
result.best_cost         # total penalty score — lower is better, 0.0 is perfect
result.feasible          # True if no hard constraint was violated
result.improvement_pct   # % cost reduction from the starting arrangement
result.converged         # rough signal: did cost flatten in the final 20% of the run?

result.breakdown(scorer) # per-constraint cost breakdown as a dict
```

---

## Solver options

### Engine

```python
Solver(..., engine="auto")  # default: SA for <40 entities, PT for >=40
Solver(..., engine="sa")    # Simulated Annealing — fast, good for smaller problems
Solver(..., engine="pt")    # Parallel Tempering — better for large or complex problems
```

### Starting arrangement

```python
Solver(..., init="balanced")  # round-robin distribution — recommended for equal-size groups
Solver(..., init="random")    # random distribution — use with the move neighbor function
```

**Important:** if your groups all have equal capacity, always use `init="balanced"`. The default `swap` operation preserves group sizes — an unbalanced starting arrangement will stay unbalanced forever because no sequence of swaps can fix it.

### Tuning SA

```python
Solver(..., engine="sa", T0=80.0, alpha=0.997, iterations=40_000)
```

| Parameter | Effect |
|---|---|
| `T0` | Starting temperature. Higher = more exploration early on. |
| `alpha` | Cooling rate (0–1). Closer to 1 = slower cooling = longer run, better result. |
| `iterations` | Total steps. Increase if the result is still improving at the end. |

### Tuning PT

```python
Solver(..., engine="pt", T_min=0.1, T_max=100.0, n_chains=6, iterations=50_000)
```

### Neighbor function

Controls how the engine proposes a new arrangement at each step:

| Function | Behaviour | Use when |
|---|---|---|
| `swap` (default) | Exchange two entities' slots | Equal-size groups required |
| `move` | Reassign one entity to a random slot | Variable group sizes |
| `mixed` | 70% swap, 30% move | Soft capacity constraints |
| `make_swap_k(k)` | Swap k pairs simultaneously | Large problems, hot exploration |

```python
from core import move, make_swap_k

Solver(..., neighbor_fn=move)
Solver(..., neighbor_fn=make_swap_k(3))  # swap 3 pairs per step
```

### Multiple independent solutions

```python
results = solver.sample(n=5, seed=0)  # 5 independent runs, sorted best-first
best = results[0]
top3 = results[:3]
```

---

## What the engine cannot guarantee

- **It may not find the global optimum.** SA and PT are heuristics — they search well but do not prove optimality. For problems with fewer than ~100 entities, an exact solver (like Google OR-Tools) would find the true optimum. This library trades proof for flexibility.
- **Hard constraints must be satisfiable.** If your hard constraints make every arrangement infeasible (for example, requiring 12 people in 10 groups of 1), the engine will return `result.feasible == False` and `best_cost == inf`.
- **More iterations generally means better results.** If you're not satisfied with the output, try increasing `iterations` or lowering `alpha` (SA) before anything else.

---

## Running the tests

```bash
pytest tests/
```

170 tests covering state manipulation, constraints, scoring, engines, history, and edge cases. All pass in under a second.

---

## Project layout

```
core/        Engine and data structures — no external dependencies
examples/    group_rotation.py and timetable.py — working end-to-end examples
tests/       170 unit and regression tests
```
