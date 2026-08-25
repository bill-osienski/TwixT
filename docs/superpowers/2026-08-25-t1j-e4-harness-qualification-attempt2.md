# E4 execution-harness qualification attempt 2 — RAN, PASSED

**Date:** 2026-08-25 · **Status:** **RAN and PASSED.** Driver gate exit **0**.
**Supersedes [attempt 1](2026-08-25-t1j-e4-harness-qualification.md)**, preserved unchanged.
One model load, **exactly one** real agent call on synthetic seed `90000001`.
**No game, no T1j, no scheduled seed.** · **The 32-game E4 screen remains UNAUTHORIZED.**

Basis: `main` @ `c5730cf`. Evidence: 8 files plus a self-excluding manifest.
Full suite: **2998 passed, 4 skipped, 0 failed**. 85 harness tests.

---

## Why there is an attempt 2

Attempt 1 qualified a loop *shape*. The injected stub returned points, terminal reason and plies
directly, so **no committed code advanced state, alternated agents, validated moves, called the
per-ply binder, applied the cap or stopped early** — and the public path executed zero tasks. Three
more defects came with it. All four are closed.

## 1 — [P1] The play loop is now production code

`play_task` owns the loop. Per ply it picks the agent whose colour is to move, asks for a move,
**validates it before applying**, applies it, calls the **per-ply binder**, and records the ply. The
external cap applies to our own ply counter; a cap termination scores as a draw.

```
PASS   loop reached a real terminal win        PASS   plies and anchor points correct
PASS   ONE agent per colour per task, not one per ply
PASS   the loop alternated colours             PASS   the per-ply binder was called after every applied move
PASS   the external ply cap applied            PASS   a cap termination scores as a draw
PASS   NEGATIVE CONTROL: an illegal move is refused BEFORE it is applied
PASS   NEGATIVE CONTROL: the default binder refuses — the screen must bind every ply
```

Driven on a **6×6 board that reaches a real red win in 7 plies**, so the loop is exercised to a
genuine terminal state with fake agents and no inference.

> **The test caught a real defect in the loop.** My first version called `agent_for` **every ply**.
> `SeededReferenceAgent` is stateful by contract — both RNG streams advance across the game — so
> rebuilding it per ply would have silently reset the seeding and made the per-task seed
> meaningless. Each side is now built **once per task**, and `agents_built == 2` is asserted.

The default binder **refuses**: wiring it to T1j is not part of this authorization, so the harness
fails closed rather than playing unbound.

## 2 — [P1] A rebuilt evaluator now fails the gate

Identity is enforced **immediately after each construction** against the one expected evaluator. A
mismatch raises in the `agent_construction` phase, records a durable abort, and **emits no verdict**.
Attempt 1 merely counted distinct evaluators and let the run finish.

## 3 — [P1] The exit status is the gate

`python -m scripts.GPU.alphazero.e4_screen_runner` is a real command with `--help`, and every path
below was qualified **in a fresh subprocess**:

| invocation | exit |
|---|---|
| `--help` | **0** — and documents that the screen is unauthorized |
| clean public run | **0** |
| `--mode screen` | **2** — precondition refused, nothing opened |
| existing results file | **2** — and the file is left untouched |
| tampered plan | **2** |
| missing plan | **2** |

Codes: 0 ok, 2 precondition refused, 3 aborted mid-run, 4 unexpected.

## 4 — [P1] Recording is exclusive and durable

The results file is opened **`x`** — an existing path is refused, because appending would silently
merge two runs. Every record is **flushed and fsynced** before the run proceeds, so a kill leaves a
truthful prefix rather than a buffer. Failures are classified by phase — `agent_construction`,
`move`, `per_ply_binding`, `cleanup`, `classification` — a terminal abort record is written when it
can be, and `emit_terminal` **never masks** the error that caused it.

## Unchanged from attempt 1, re-run

The one real call: position `p06_e3a_center`, ply 6, red to move, seed `90000001` → **(14, 13)**,
legal, `validate_ply` passed, both RNGs advanced, compiled graph built by that inference, one
evaluator throughout. Scheduled seeds `[202612128, 202612160)` still unspent. Schedule lock intact:
paths-only public signature, pinned plan sha256 and task digest, eleven reshaping controls.

---

## What this establishes, and what it does not

**Established:** the harness owns its play loop and that loop is exercised to real terminal states,
including cap and illegal-move paths; a rebuilt evaluator aborts; the command entry point's exit
codes are qualified in fresh subprocesses; recording is exclusive, fsynced and phase-classified; and
the canonical schedule still cannot be injected or reshaped.

**Not established, and not claimed:** that the agent plays well, or at all beyond **one move, one
position**; anything about T1j — **the per-ply binder is a refusing stub here**, so nothing binds
two engines yet; game length or screen runtime; and absolute placement.

## Post-run provenance (added after `ed48f7a`)

C-series evidence added separately; no runtime artifact changed and `08_MANIFEST` still verifies
7/7. `C1_post_run_provenance.txt` binds `06_committed_sources.sha256.txt` to the **commit's**
objects, not the index:

- **A** — all **8 tracked rows** match `git rev-parse ed48f7a:<path>`; the checkpoint has no object
  in `ed48f7a` and is reported *UNTRACKED (gitignored), disk-only*.
- **B** — every file under all prior evidence directories and cards, compared against the commit
  that added **that file**: **zero changed**.
- **B2 — declared amendments.** The E4 preflight attempt-4 card was deliberately amended by
  `fee7a3b` (section 4 plus two visible correction blocks). Rather than let it read as a violation
  *or* be quietly excused, it is declared: checked against the **amending** commit, and **required
  to actually differ from its origin** — a declaration that changed nothing would be a way to exempt
  a file from checking altogether.

Nine controls, all invoking the real checkers, all passing.
