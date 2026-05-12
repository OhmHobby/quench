"""
Example: Timetable with Minimal Walking Distance
=================================================
Scenario: 10 courses must be assigned to (room, period) slots across
a school day. Students are enrolled in multiple courses; we want to
minimise the total distance they walk between consecutive periods.

Room layout (grid coordinates):
    [A1] [A2] [A3]
    [B1] [B2] [B3]
    [C1] [C2] [C3]

Walking distance = Manhattan distance between room coordinates.

Goals:
  - Hard: each (room, period) slot holds at most 1 course
  - Hard: courses with overlapping students cannot share a period
  - Soft: minimise total student walking between consecutive periods

This example demonstrates:
  - Using Slot.meta to store spatial data (room coordinates)
  - Using Entity.meta to store relational data (student enrolment)
  - Building a cost function that reads both

Run:
    python examples/timetable.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from quench import Entity, Slot, State, Scorer, hard, soft, Solver


# ── Rooms and periods ─────────────────────────────────────────────────────────

# 3×3 grid of rooms; each room available in 3 periods = 9 slots
ROOM_COORDS: dict[str, tuple[int, int]] = {
    "A1": (0, 2), "A2": (1, 2), "A3": (2, 2),
    "B1": (0, 1), "B2": (1, 1), "B3": (2, 1),
    "C1": (0, 0), "C2": (1, 0), "C3": (2, 0),
}
PERIODS = [1, 2, 3, 4]

slots: list[Slot] = [
    Slot(
        id=f"{room}-P{period}",
        capacity=1,
        meta={"room": room, "period": period, "coords": ROOM_COORDS[room]},
    )
    for room in ROOM_COORDS
    for period in PERIODS
]

# ── Courses and enrolments ────────────────────────────────────────────────────
# Each course knows which students attend it (stored in meta).
# Students attending consecutive courses should have short walks.

ENROLMENTS: dict[str, set[str]] = {
    "Math":    {"Alice", "Bob", "Carol", "Dan", "Eve"},
    "Science": {"Alice", "Bob", "Frank", "Grace"},
    "History": {"Carol", "Dan", "Hank", "Iris"},
    "English": {"Eve", "Frank", "Jack", "Kate"},
    "Art":     {"Grace", "Hank", "Leo", "Mia"},
    "Music":   {"Iris", "Jack", "Noah", "Olivia"},
    "PE":      {"Kate", "Leo", "Paul", "Quinn"},
    "Drama":   {"Mia", "Noah", "Rose", "Sam"},
    "CS":      {"Olivia", "Paul", "Alice", "Frank"},
    "Geo":     {"Quinn", "Rose", "Bob", "Carol"},
}

courses: list[Entity] = [
    Entity(name, meta={"students": students})
    for name, students in ENROLMENTS.items()
]

# ── Constraints ───────────────────────────────────────────────────────────────

def no_double_booking(state: State) -> bool:
    """Two courses cannot occupy the same (room, period) slot."""
    for slot in slots:
        if len(state.group(slot)) > 1:
            return True
    return False

def no_period_conflict(state: State) -> bool:
    """
    Two courses with overlapping students cannot be in the same period.
    A student cannot be in two places at once.
    """
    # Group courses by period
    period_courses: dict[int, list[Entity]] = {}
    for course, slot in state:
        p = slot.meta["period"]
        period_courses.setdefault(p, []).append(course)

    for p, period_group in period_courses.items():
        for i, c1 in enumerate(period_group):
            for c2 in period_group[i + 1:]:
                shared = c1.meta["students"] & c2.meta["students"]
                if shared:
                    return True     # conflict found
    return False

def manhattan(coords1: tuple[int, int], coords2: tuple[int, int]) -> int:
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

def total_walking_distance(state: State) -> float:
    """
    For each student, sum the walking distance between consecutive periods.

    Groups courses by period, then for each consecutive period pair, finds
    which room a student's courses are in and accumulates the distance.
    """
    # Build: student → {period: room_coords}
    student_schedule: dict[str, dict[int, tuple[int, int]]] = {}
    for course, slot in state:
        coords = slot.meta["coords"]
        period = slot.meta["period"]
        for student in course.meta["students"]:
            student_schedule.setdefault(student, {})[period] = coords

    total = 0.0
    for student, schedule in student_schedule.items():
        sorted_periods = sorted(schedule)
        for p1, p2 in zip(sorted_periods, sorted_periods[1:]):
            total += manhattan(schedule[p1], schedule[p2])

    return total

# ── Build and solve ───────────────────────────────────────────────────────────

scorer = Scorer([
    hard(no_double_booking,  name="no_double_booking"),
    hard(no_period_conflict, name="no_period_conflict"),
    soft(total_walking_distance, weight=1.0, name="walking"),
])

# Timetable state space is more complex than group rotation:
# We use `move` (not swap) because we need to change which (room, period)
# a course sits in — swapping two courses' slots may not fix conflicts.
# `init="random"` is fine here since hard constraints guide the engine.
from quench.neighbor import move

solver = Solver(
    courses, slots, scorer,
    engine="sa",
    init="random",
    neighbor_fn=move,
    T0=200.0,
    alpha=0.9990,
    iterations=100_000,
)


def display_timetable(state: State) -> None:
    schedule: dict[int, dict[str, str]] = {p: {} for p in PERIODS}
    for course, slot in state:
        schedule[slot.meta["period"]][slot.meta["room"]] = course.id

    rooms = sorted(ROOM_COORDS)
    col = 12
    header = f"{'':8s}" + "".join(f"{r:>{col}}" for r in rooms)
    print(header)
    print("  " + "─" * (len(header) - 2))

    for period in PERIODS:
        row = f"  P{period}    "
        for room in rooms:
            row += f"{schedule[period].get(room, '—'):>{col}}"
        print(row)


def main() -> None:
    print("Solving timetable — minimising student walking distance…")
    print(f"  Courses: {len(courses)}   Slots: {len(slots)}   Engine: SA\n")

    # Run 3 independent searches, take the best
    results = solver.sample(n=5, seed=0)
    result  = results[0]

    print(f"Result:  {result}")
    print(f"Breakdown: {result.breakdown(scorer)}\n")

    if result.feasible:
        print("Timetable (period × room):")
        display_timetable(result.best_state)

        # Per-student walking summary
        print("\nPer-student walking distances:")
        student_distances: dict[str, int] = {}
        state = result.best_state

        schedule: dict[str, dict[int, tuple[int, int]]] = {}
        for course, slot in state:
            for student in course.meta["students"]:
                schedule.setdefault(student, {})[slot.meta["period"]] = slot.meta["coords"]

        for student, sched in sorted(schedule.items()):
            periods = sorted(sched)
            dist = sum(
                manhattan(sched[p1], sched[p2])
                for p1, p2 in zip(periods, periods[1:])
            )
            student_distances[student] = dist

        for student, dist in sorted(student_distances.items(), key=lambda x: -x[1]):
            bar = "█" * dist
            print(f"  {student:8s}  {bar} {dist}")
    else:
        print("No feasible timetable found — try more iterations or a higher T0.")

    print()


if __name__ == "__main__":
    main()
