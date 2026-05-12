"""
Example: Inpatient Pharmacy Monthly Schedule
=============================================
Real-world use case: generate a 4-week rotation schedule for 15 inpatient
pharmacists across 5 task sections (S, M, L, C, V) and 15 sub-positions.

Staff and rules supplied by customer:
  - วรลักษณ์ is replaced by ชิดชนก in the roster.
  - รุ่งโรจน์ stays fixed at M4 every week — never rotated.
  - อรุณี stays fixed at L3 every week — never rotated.
  - HARD: พัชรสิรินทร์ and ศศิลักษณ์ must never be in the same task section.
  - SOFT: minimise repeat pairings within the same task section across weeks.
  - SOFT: minimise assigning the same person to the same sub-position again.
  - SOFT: ensure everyone cycles through different task sections over time.

Run:
    python examples/pharmacy_schedule.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from core import Entity, Slot, State, Scorer, hard, soft, Solver


# ── Staff ─────────────────────────────────────────────────────────────────────

# Fixed staff — pinned to one position for every week
FIXED: dict[str, str] = {
    "รุ่งโรจน์": "M4",
    "อรุณี":    "L3",
}

# Rotatable staff (วรลักษณ์ replaced by ชิดชนก)
ROTATABLE_NAMES: list[str] = [
    "สรวีย์",        "ชิดชนก",        "ศิลปนันท์",
    "พัชรสิรินทร์", "ศศิลักษณ์",     "พัสวีเลิศ",
    "ประภวิษณุ",    "สิริบุญ",        "มัทวัน",
    "กัลยรัตน์",    "พวงผกา",         "ดวงกมล",
    "สิริกร",
]


# ── Positions ─────────────────────────────────────────────────────────────────

POSITION_TASK: dict[str, str] = {
    "S1": "S", "S2": "S", "S3": "S", "S4": "S",
    "M1": "M", "M2": "M", "M3": "M", "M4": "M",
    "L1": "L", "L2": "L", "L3": "L",
    "C1": "C", "C2": "C",
    "V1": "V", "V2": "V",
}

ALL_POSITIONS_ORDERED: list[str] = [
    "S1", "S2", "S3", "S4",
    "M1", "M2", "M3", "M4",
    "L1", "L2", "L3",
    "C1", "C2",
    "V1", "V2",
]

FIXED_SLOTS     = set(FIXED.values())                                          # {"M4", "L3"}
ROTATABLE_SLOTS = [p for p in ALL_POSITIONS_ORDERED if p not in FIXED_SLOTS]  # 13 positions


# ── Engine objects ────────────────────────────────────────────────────────────
# Solver only touches the 13 rotatable staff × 13 rotatable positions.
# Fixed assignments are merged back in at display/export time.

entities  = [Entity(name) for name in ROTATABLE_NAMES]
slots     = [Slot(pos, meta={"task": POSITION_TASK[pos]}) for pos in ROTATABLE_SLOTS]
entity_by = {e.id: e for e in entities}

patcharasirin = entity_by["พัชรสิรินทร์"]
sasilak       = entity_by["ศศิลักษณ์"]


# ── History trackers ──────────────────────────────────────────────────────────
# Module-level so every week's constraint closures always see the latest data.

# How many times each pair has shared the same task section
task_pair_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

# How many times each person has been in each exact sub-position
position_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

# How many times each person has done each task section
task_type_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))


# ── Constraint functions ──────────────────────────────────────────────────────

def _separation_violated(state: State) -> bool:
    """True when พัชรสิรินทร์ and ศศิลักษณ์ land in the same task section."""
    return state[patcharasirin].meta["task"] == state[sasilak].meta["task"]


def _task_pair_penalty(state: State) -> float:
    """Sum of past co-occurrences for every pair sharing a task section."""
    by_task: dict[str, list[Entity]] = defaultdict(list)
    for e, s in state:
        by_task[s.meta["task"]].append(e)
    return float(sum(
        task_pair_counts[e1.id][e2.id]
        for members in by_task.values()
        for e1, e2 in combinations(members, 2)
    ))


def _position_repeat_penalty(state: State) -> float:
    """Penalty for assigning someone to a sub-position they have held before."""
    return float(sum(position_counts[e.id][s.id] for e, s in state))


def _task_fairness_penalty(state: State) -> float:
    """Penalty for giving someone a task section they have already done a lot."""
    return float(sum(task_type_counts[e.id][s.meta["task"]] for e, s in state))


def _update_history(state: State) -> None:
    """Record this week's result into all three trackers before next solve."""
    by_task: dict[str, list[Entity]] = defaultdict(list)
    for e, s in state:
        by_task[s.meta["task"]].append(e)

    for members in by_task.values():
        for e1, e2 in combinations(members, 2):
            task_pair_counts[e1.id][e2.id] += 1
            task_pair_counts[e2.id][e1.id] += 1

    for e, s in state:
        position_counts[e.id][s.id]               += 1
        task_type_counts[e.id][s.meta["task"]]    += 1


# ── Solver ────────────────────────────────────────────────────────────────────

def _solve_week(seed: int) -> tuple[State, object, Scorer]:
    scorer = Scorer([
        hard(_separation_violated,    name="separation"),
        soft(_task_pair_penalty,      weight=4.0, name="task_group_repeat"),
        soft(_position_repeat_penalty, weight=3.0, name="position_repeat"),
        soft(_task_fairness_penalty,   weight=2.0, name="task_fairness"),
    ])
    solver = Solver(
        entities, slots, scorer,
        engine="sa",
        init="balanced",
        T0=60.0,
        alpha=0.997,
        iterations=50_000,
    )
    result = solver.solve(seed=seed)
    return result.best_state, result, scorer


def _full_schedule(state: State) -> dict[str, str]:
    """Merge solver output with fixed assignments → position → name."""
    schedule: dict[str, str] = {pos: name for name, pos in FIXED.items()}
    for e, s in state:
        schedule[s.id] = e.id
    return schedule


# ── Display ───────────────────────────────────────────────────────────────────

TASK_SECTION_LABELS: dict[str, str] = {
    "S": "S  (Satellite)",
    "M": "M  (Main)",
    "L": "L  (Long-stay)",
    "C": "C  (Counselling)",
    "V": "V  (Verify)",
}


def _display_week(label: str, schedule: dict[str, str]) -> None:
    print(f"\n  {'═' * 40}")
    print(f"  {label}")
    print(f"  {'═' * 40}")
    prev_task = None
    for pos in ALL_POSITIONS_ORDERED:
        task = POSITION_TASK[pos]
        if task != prev_task:
            print(f"\n    {TASK_SECTION_LABELS[task]}")
            prev_task = task
        name   = schedule.get(pos, "—")
        marker = "  [fixed]" if pos in FIXED_SLOTS else ""
        print(f"      {pos:<5} {name}{marker}")


def _display_summary(all_schedules: list[dict[str, str]], n_weeks: int) -> None:
    task_types = ["S", "M", "L", "C", "V"]
    all_staff  = sorted(ROTATABLE_NAMES) + sorted(FIXED.keys())

    counts: dict[str, dict[str, int]] = {
        n: {t: 0 for t in task_types} for n in all_staff
    }
    for sched in all_schedules:
        for pos, name in sched.items():
            counts[name][POSITION_TASK[pos]] += 1

    col = 7
    print(f"\n  {'═' * 56}")
    print(f"  Task-section rotation summary  ({n_weeks} weeks)")
    print(f"  {'═' * 56}")
    print(f"  {'Staff':<22}" + "".join(f"{t:>{col}}" for t in task_types))
    print("  " + "─" * 54)
    for name in all_staff:
        row    = f"  {name:<22}" + "".join(f"{counts[name][t]:>{col}}" for t in task_types)
        marker = "  [fixed]" if name in FIXED else ""
        print(row + marker)


def _display_separation_check(all_states: list[State]) -> None:
    print(f"\n  {'═' * 56}")
    print("  Separation check  (พัชรสิรินทร์ ≠ ศศิลักษณ์ task section)")
    print(f"  {'═' * 56}")
    all_ok = True
    for i, state in enumerate(all_states):
        p_task = state[patcharasirin].meta["task"]
        s_task = state[sasilak].meta["task"]
        ok     = p_task != s_task
        status = "✓" if ok else "✗  VIOLATION"
        print(f"    Week {i + 1}:  พัชรสิรินทร์={p_task}   ศศิลักษณ์={s_task}   {status}")
        if not ok:
            all_ok = False
    print(f"\n  Result: {'All weeks passed ✓' if all_ok else 'VIOLATIONS FOUND ✗'}")


# ── CSV export ────────────────────────────────────────────────────────────────

def _export_csv(
    all_schedules: list[dict[str, str]],
    week_labels: list[str],
) -> str:
    out_path = Path(__file__).parent.parent.parent / "schedule_output.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Position"] + week_labels)
        for pos in ALL_POSITIONS_ORDERED:
            writer.writerow([pos] + [s.get(pos, "—") for s in all_schedules])
    return str(out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n_weeks: int = 4) -> None:
    # Change WEEK_LABELS to match your actual calendar dates
    WEEK_LABELS = [f"Week {i + 1}" for i in range(n_weeks)]
    SEEDS = [42, 99, 17, 31, 55, 77, 13, 88]

    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║   Inpatient Pharmacy Schedule Generator          ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print(f"  ║   Staff     : {len(ROTATABLE_NAMES) + len(FIXED)} total"
          f"  ({len(FIXED)} fixed, {len(ROTATABLE_NAMES)} rotatable){' ' * 10}║")
    print(f"  ║   Positions : {len(ALL_POSITIONS_ORDERED)} total"
          f"  ({len(FIXED_SLOTS)} fixed, {len(ROTATABLE_SLOTS)} rotatable){' ' * 10}║")
    print(f"  ║   Weeks     : {n_weeks}{' ' * 36}║")
    print("  ╚══════════════════════════════════════════════════╝")

    all_schedules: list[dict[str, str]] = []
    all_states:    list[State]          = []

    for i, label in enumerate(WEEK_LABELS):
        print(f"\n  Solving {label}…", end=" ", flush=True)
        state, result, scorer = _solve_week(SEEDS[i % len(SEEDS)])
        print(f"{result}")

        schedule = _full_schedule(state)
        all_schedules.append(schedule)
        all_states.append(state)

        _display_week(label, schedule)
        print(f"\n    Cost breakdown : {result.breakdown(scorer)}")

        _update_history(state)

    # ── Post-run reports ──────────────────────────────────────────────────────
    _display_separation_check(all_states)
    _display_summary(all_schedules, n_weeks)

    csv_path = _export_csv(all_schedules, WEEK_LABELS)
    print(f"\n  Exported → {csv_path}")
    print()


if __name__ == "__main__":
    main()
