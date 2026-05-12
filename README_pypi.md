# quench

Arrange people into groups — or schedule things into time slots — while following rules you define.

Built for problems like:

- **Group rotation** — cycle 20 kids through 4 activity groups over several sessions, making sure the same kids don't keep ending up together
- **Timetabling** — assign 10 classes across rooms and periods so no student is double-booked and total walking distance is minimised
- **Any arrangement problem** — staff shifts with "do not pair" rules, sports teams with balance constraints, meeting rooms with attendee requirements

The engine finds the best arrangement it can given your rules. You define what "best" means.

Pure standard library. Zero external dependencies. Python 3.7+.

---

## Install

```bash
pip install quench
```

---

## Quick start

```python
from quench import Entity, Slot, Scorer, Solver, hard, soft

people = [Entity("Alice"), Entity("Bob"), Entity("Carol"), Entity("Dan")]
groups = [Slot("Morning", capacity=2), Slot("Afternoon", capacity=2)]

def no_overflow(state):
    return any(len(state.group(g)) > g.capacity for g in groups)

def prefer_separate(state):
    return 1.0 if state[Entity("Alice")] == state[Entity("Bob")] else 0.0

scorer = Scorer([
    hard(no_overflow, name="capacity"),
    soft(prefer_separate, weight=5.0, name="alice_bob_apart"),
])

solver = Solver(people, groups, scorer, init="balanced")
result = solver.solve(seed=42)

print(result)
# Result(SA, feasible, cost=0.000, improvement=100.0%, iters=50,000)

for g in groups:
    members = [e.id for e in result.best_state.group(g)]
    print(f"{g.id}: {', '.join(members)}")
# Morning:   Alice, Carol
# Afternoon: Bob, Dan
```

---

## Constraints

**Hard constraints** must never be broken. The engine rejects any arrangement that violates one.

```python
from quench import hard

def no_overflow(state):
    # return True if the constraint is VIOLATED
    return any(len(state.group(g)) > g.capacity for g in groups)

capacity_rule = hard(no_overflow, name="capacity")
```

**Soft constraints** are goals to minimise.

```python
from quench import soft

def prefer_separate(state):
    return 1.0 if state[Entity("Alice")] == state[Entity("Bob")] else 0.0

separation_rule = soft(prefer_separate, weight=5.0, name="alice_bob_apart")
```

`weight` controls relative importance. A weight of 5.0 means this constraint matters five times more than one with weight 1.0.

---

## Multiple sessions with history

```python
from quench import History

history = History()

for session in range(1, 4):
    scorer = Scorer([
        hard(no_overflow, name="capacity"),
        history.as_soft_constraint(weight=3.0),
    ])
    solver = Solver(people, groups, scorer, init="balanced")
    result = solver.solve(seed=session)

    history.update(result.best_state)

history.save("history.json")           # save after a session
history = History.load("history.json") # restore in the next run
```

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

### Tuning SA

```python
Solver(..., engine="sa", T0=80.0, alpha=0.997, iterations=40_000)
```

### Tuning PT

```python
Solver(..., engine="pt", T_min=0.1, T_max=100.0, n_chains=6, iterations=50_000)
```

### Neighbor function

| Function | Behaviour | Use when |
|---|---|---|
| `swap` (default) | Exchange two entities' slots | Equal-size groups required |
| `move` | Reassign one entity to a random slot | Variable group sizes |
| `mixed` | 70% swap, 30% move | Soft capacity constraints |
| `make_swap_k(k)` | Swap k pairs simultaneously | Large problems, hot exploration |

```python
from quench import move, make_swap_k

Solver(..., neighbor_fn=move)
Solver(..., neighbor_fn=make_swap_k(3))
```

### Multiple independent solutions

```python
results = solver.sample(n=5, seed=0)  # 5 independent runs, sorted best-first
```

---

## What the engine cannot guarantee

- **It may not find the global optimum.** SA and PT are heuristics — they search well but do not prove optimality.
- **Hard constraints must be satisfiable.** If your rules make every arrangement infeasible, the engine returns `result.feasible == False`.
- **More iterations generally means better results.** If you're not satisfied with the output, try increasing `iterations` or lowering `alpha` (SA) before anything else.

---

## Source and examples

Full source, working examples, and documentation: [github.com/OhmHobby/quench](https://github.com/OhmHobby/quench)
