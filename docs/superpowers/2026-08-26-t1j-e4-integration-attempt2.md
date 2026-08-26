# E4 integration qualification attempt 2 — corrective. **RAN, PASSED.**

**Date:** 2026-08-26 · **Status:** **RAN and PASSED.** Exit **0**, **stderr 0 bytes**, complete
stdout captured from process start. **Supersedes
[attempt 1](2026-08-25-t1j-e4-integration.md)**, preserved unchanged as failed.
**No canonical seed, no completed game, no screen verdict.**
· **The 32-game E4 screen remains UNAUTHORIZED.**

Basis: `main` @ `1063c07`. Full suite: **3070 passed, 4 skipped, 0 failed**.
Seeds consumed: **`[90002000, 90002004)`**, now recorded as spent.

---

## The three bindings review required

### 1 — The position the search JVM reconstructed

The replay binder proves *a* JVM can rebuild the history; it says nothing about the JVM that
actually searched. Those are different processes. `T1jAgent` now re-binds `A.query`'s dump — the
position reconstructed **inside the searching JVM** — against our state **before accepting the
move**, using the *same* `compare_state` the per-ply binder uses, so the two cannot drift.

**8 searched-position re-bindings, one per T1j search, all clean.** The negative control builds an
agent whose search JVM returns a *different* position at the same ply and asserts the abort — the
replay binder would never have seen it.

### 2 — JDK identity, not JDK presence

Attempt 1 checked that `bin/java` existed. All four pinned components are now compared **before
compilation and before any seed is touched**:

```
af8b1229…  bin/java     6f515930…  bin/javac
28745573…  lib/modules  cb6064fe…  release
```

Controls: a wrong file at the right path is refused; a missing component is refused.

### 3 — The reflection count, enforced

`PostCond.clean` proves the field *names* were authorized; it ignores `refl_n`, so a repeated or
missing authorized access passed. The counts were **measured, not assumed** — replay reports
**exactly 1** (`freshMatch`'s single `nextPlayer` write), a query **exactly 3** (that write plus the
two `FindMove` reads) — and each caller now asserts its own.

## The two defects from attempt 1

**Totals now come from non-resetting per-task records.** `IntegrationContext.stats` is keyed by
task and never cleared; `11_per_task_stats.json` carries all four:

| task | binds | T1j searches | searched re-binds | plies | reason | points |
|---|---:|---:|---:|---|---|---|
| `integ2-weak3-t1j_red` | 5 | 2 | 2 | 7,8,9,10 | `qualification_budget` | `None` |
| `integ2-weak3-t1j_black` | 5 | 2 | 2 | 7,8,9,10 | `qualification_budget` | `None` |
| `integ2-strong6-t1j_red` | 5 | 2 | 2 | 7,8,9,10 | `qualification_budget` | `None` |
| `integ2-strong6-t1j_black` | 5 | 2 | 2 | 7,8,9,10 | `qualification_budget` | `None` |

**Totals: 20 binder comparisons, 8 T1j searches, 8 searched-position re-bindings.**

**Everything is captured.** The driver records the command, redirects stdout and stderr from process
start, and writes the exit status to its own file: **4,361 bytes of stdout, 0 bytes of stderr,
exit 0.**

## What the run did

Four short tasks — both colour arms × depths 3 and 6 — each: the six-ply opening bound **before
either agent was built**, four real plies alternating red/black/red/black with `ply_cap=280`
explicit, one agent per colour, cleanup once, stopping on the **qualification budget** at ply 10.
Sixteen real plies. A durable **receipt**; **no screen verdict**.

Every ply the binder compared pegs, bridges, side to move, independently derived ply, the full
legal-move set, terminal state with winner attribution, T1j's **ordered history** via its own
accessors, and the POSTCOND surface — headless, zero `Window`/`Frame`, host preferences unchanged,
authorized reflection **at the exact expected count**.

---

## What this establishes, and what it does not

**Established:** the real T1j engine, the real reference agent and the real E3b binder run inside
the production harness across both colour arms and both depths with **zero divergences on every
bound observable** — including, now, the position the searching JVM itself reconstructed — under a
JDK bound component by component and an enforced reflection count.

**Not established, and not claimed:** anything about **strength** — no game was finished and no
score exists; behaviour beyond 10 plies from a single opening; that the reference plays well;
screen runtime; and absolute placement. **T1j remains uncalibrated.**
