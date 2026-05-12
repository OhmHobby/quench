# Roadmap

## Current state

quench is a code-first library. Using it today requires writing Python —
defining entities, slots, and constraint functions by hand. That works well
for developers but puts it out of reach for non-technical users who just want
to describe a scheduling problem and get a schedule back.

---

## Planned

### Natural language interface (LLM integration)

Let users describe their problem in plain language and have a language model
translate it into constraint definitions automatically.

Example flow:
```
"I have 20 staff and 4 groups. Alice and Bob must never be in the same group.
 Carol is always in Group A. Everyone should rotate roles each week."

→ hard constraint: alice ≠ bob in same group
→ hard constraint: carol fixed to Group A
→ soft constraint: role repeat penalty (history-aware)
→ Solver runs, returns schedule
→ LLM explains the result in plain English
```

The `core/` engine is already designed for this — `scorer.breakdown()` gives
per-constraint costs that an LLM can read back to the user as explanations.
The integration layer (`llm/`) is the remaining piece.

**Stack:** Anthropic SDK (Claude), prompt caching for repeated solver calls,
`solver.sample(n)` to give the LLM multiple arrangements to choose from.

---

### UI / web interface

A simple form-based interface so non-developers can:
- Enter their staff list and groups
- Tick constraint checkboxes ("never pair these two", "keep this person fixed")
- Run the solver with one click
- Download the result as a CSV

This would make quench usable by hospital coordinators, camp organizers,
school administrators — anyone with a rotation problem but no coding background.

---

## Not planned (by design)

- Exact solvers (OR-Tools / CP-SAT) — different paradigm, heavy dependency
- Built-in constraint library — the engine is intentionally domain-agnostic

See `System_Overview.md` § 6 for the full reasoning.
