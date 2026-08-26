# E4 integration qualification — RAN. **Gate FAILED (exit 3).**

**Date:** 2026-08-25 · **Status:** the run **completed with no engine divergence**, and the gate
**exited 3 on two of my own counter assertions.** Two defects in this package are mine; **none is
the integration's.** · **The 32-game screen remains UNAUTHORIZED.**

Basis: `main` @ `2b8b3aa`. Full suite: **3047 passed, 4 skipped, 0 failed**.
Seeds consumed: **`[90001000, 90001004)`** — four synthetic seeds, now recorded as spent.

---

## The result, stated plainly

Both engines met for the first time. Four short tasks — **both colour arms × depths 3 and 6** — ran
to a 10-ply qualification budget with the real T1j engine, the real reference agent, and the real
E3b per-ply binder inside the production harness.

**No engine ever disagreed.** The binder aborts on the first divergence and there is **no abort
record** in the log. But the gate exited 3, so this is a **FAILED qualification**, and the reason is
worth stating precisely rather than explaining away.

## Defect 1 (mine) — the two failing assertions were arithmetic

`ctx.binds` and `ctx.t1j_queries` live on a context the **state factory resets at the start of every
task**. My assertions compared them against **cross-task totals**, so they read the last task's
numbers — 5 and 2 — and failed against the expected 20 and 8.

`11_record_analysis.txt` measures the durable JSONL instead, per task:

| task | opening bound | plies | binder comparisons | T1j searches | reason | points |
|---|---|---|---|---|---|---|
| `integ-weak3-t1j_red` | ply 6 | 7,8,9,10 | 5 | 2 | `qualification_budget` | `None` |
| `integ-weak3-t1j_black` | ply 6 | 7,8,9,10 | 5 | 2 | `qualification_budget` | `None` |
| `integ-strong6-t1j_red` | ply 6 | 7,8,9,10 | 5 | 2 | `qualification_budget` | `None` |
| `integ-strong6-t1j_black` | ply 6 | 7,8,9,10 | 5 | 2 | `qualification_budget` | `None` |

**20 binder comparisons and 8 real T1j searches** across the run; 0 aborts; 0 verdicts.

## Defect 2 (mine) — the console output was never captured

I ran the qualification **without redirecting stdout to a file** and viewed it through `tail`.
Sections 1–3 — pinned artifact identity, the compile through the adapter, the reference load —
**scrolled off and are unrecoverable.**

**I did not re-run to repair this.** The run consumed four synthetic seeds by drawing from both
generators; re-running would either reuse spent seeds — the contamination the accounting exists to
prevent — or be a *different* run presented as this one. `00_console_excerpt.txt` is labelled an
excerpt, states where it begins, and is not reconstructed. The complete durable record is the
harness's own fsynced JSONL.

## What the run does show

Verified from the record, per task: the **six-ply opening bound before either agent was built**;
every applied move bound with `ply_cap=280` explicit; alternation red/black/red/black; **one agent
per colour per task**; cleanup once per task; a durable **receipt**; and **no screen verdict** —
withheld because the run executed 4 tasks, not the canonical 32.

The binder compared, every ply: pegs, bridges, side to move, independently derived ply, the full
legal-move set, terminal state with winner attribution, T1j's **ordered history** read back through
its own accessors, and the helper's POSTCOND surface — headless, zero `Window`/`Frame`, host
preferences unchanged, only authorized reflection.

## Seeds

`[90001000, 90001004)` are spent: each task built a real `SeededReferenceAgent` and drew from both
generators. They are registered in `EXPOSED_SEED_INTERVALS` and a test asserts they can never be
scheduled again. **`[202612128, 202612160)` is untouched.**

---

## What this establishes, and what it does not

**Established:** the real T1j engine, the real reference agent and the real E3b binder run inside
the production harness for 16 plies across both colour arms and both depths, with **zero
divergences** on every bound observable — and the harness still refuses a verdict.

**Not established, and not claimed:** that the qualification **passed** — it did not, the gate
exited 3; anything about strength or a complete game — **no game was finished**; behaviour beyond
10 plies from one opening; and absolute placement.

**This is a failed qualification, and the failure is the result.** Whether the corrected assertions
warrant a fresh run on new seeds is your call, not something I have acted on.
