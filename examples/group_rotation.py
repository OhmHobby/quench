"""
Example: Group Rotation with History
=====================================
Scenario: 20 camp kids, 4 activity groups of 5, run across 3 sessions.

Goals:
  - Hard: no group exceeds capacity 5
  - Soft: penalise repeat pairings (kids who've been grouped together before)
  - Soft: penalise kids being assigned the same role as last session

This example shows the core multi-session loop: solve → inspect → update history → repeat.

Run:
    python examples/group_rotation.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

# Allow running from the project root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from core import Entity, Slot, State, Scorer, hard, soft, History, Solver


# ── Problem definition ────────────────────────────────────────────────────────

KIDS = [
    "Alice", "Bob", "Carol", "Dan", "Eve",
    "Frank", "Grace", "Hank", "Iris", "Jack",
    "Kate", "Leo", "Mia", "Noah", "Olivia",
    "Paul", "Quinn", "Rose", "Sam", "Tina",
]

ROLES = ["Leader", "Navigator", "Timekeeper", "Scribe"]

# Each group slot carries a role in its metadata
# Four groups × five kids each = 20 kids total
groups = [
    Slot(f"Group-{role}", capacity=5, meta={"role": role})
    for role in ROLES
]

kids = [Entity(name) for name in KIDS]

# ── Constraints ───────────────────────────────────────────────────────────────

def capacity_violated(state: State) -> bool:
    return any(len(state.group(g)) > g.capacity for g in groups)

def role_repeat_penalty(state: State, prev_roles: dict[str, str]) -> float:
    """
    Penalise any kid assigned the same role they had last session.
    prev_roles: {kid_name: role_name} from the previous arrangement.
    """
    penalty = 0.0
    for kid in kids:
        current_role = state[kid].meta["role"]
        if prev_roles.get(kid.id) == current_role:
            penalty += 1.0
    return penalty

# ── Session runner ────────────────────────────────────────────────────────────

def run_session(
    session_num: int,
    history: History,
    prev_roles: dict[str, str],
    seed: int,
) -> tuple[State, dict[str, str]]:

    scorer = Scorer([
        hard(capacity_violated, name="capacity"),
        history.as_soft_constraint(weight=4.0),
        soft(
            lambda s: role_repeat_penalty(s, prev_roles),
            weight=2.0,
            name="role_repeat",
        ),
    ])

    solver = Solver(
        kids, groups, scorer,
        engine="sa",
        init="balanced",
        T0=80.0,
        alpha=0.997,
        iterations=40_000,
    )

    result = solver.solve(seed=seed)

    print(f"\n{'═' * 52}")
    print(f"  Session {session_num}")
    print(f"{'═' * 52}")
    print(f"  {result}")
    print(f"  Cost breakdown: {result.breakdown(scorer)}\n")

    new_roles: dict[str, str] = {}
    for g in groups:
        members = sorted(e.id for e in result.best_state.group(g))
        print(f"  {g.id:20s}  {', '.join(members)}")
        for name in members:
            new_roles[name] = g.meta["role"]

    return result.best_state, new_roles


# ── Evaluation helpers ────────────────────────────────────────────────────────

def count_repeat_pairs(states: list[State]) -> int:
    """Count total same-group pair repetitions across consecutive sessions."""
    total = 0
    for s1, s2 in zip(states, states[1:]):
        for g in groups:
            m1 = set(e.id for e in s1.group(g))
            m2 = set(e.id for e in s2.group(g))
            overlap = m1 & m2
            total += len(list(combinations(overlap, 2)))
    return total

def count_role_repeats(role_logs: list[dict[str, str]]) -> int:
    """Count how often any kid had the same role in consecutive sessions."""
    total = 0
    for r1, r2 in zip(role_logs, role_logs[1:]):
        total += sum(1 for k in r1 if r1[k] == r2.get(k))
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    history = History()
    prev_roles: dict[str, str] = {}
    states: list[State] = []
    role_logs: list[dict[str, str]] = []

    for session, seed in enumerate([42, 99, 17], start=1):
        state, roles = run_session(session, history, prev_roles, seed)
        history.update(state)
        states.append(state)
        role_logs.append(roles)
        prev_roles = roles

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 52}")
    print("  Summary across 3 sessions")
    print(f"{'═' * 52}")
    print(f"  Repeat group pairs (consecutive sessions): {count_repeat_pairs(states)}")
    print(f"  Role repeats (consecutive sessions):       {count_role_repeats(role_logs)}")

    # Show who was most often paired with Alice
    alice = kids[0]
    print(f"\n  Alice's most frequent group-mates:")
    for partner_id, count in history.most_paired(alice, top_n=5):
        print(f"    {partner_id:10s}  {count} session(s)")

    print()


if __name__ == "__main__":
    main()
