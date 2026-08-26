# E4 execution-command attempt 4 — corrective. RAN, PASSED. Substitutes only; no JVM.

**Date:** 2026-08-26 · **Status:** **RAN and PASSED.** Unit tests 0, qualification 0, stderr 0 bytes,
**102 checks**. **No model, no agent, no RNG, NO JVM, no game, no scheduled seed.**
**Supersedes [attempt 3](2026-08-26-t1j-e4-execution-command-attempt3.md)**, which is preserved and
**relabelled "technical checks passed; authorization compliance failed"**.
· **`SCREEN_AUTHORIZED = False`.**

Basis: `main` @ `48f83d0`. Full suite: **3130 passed, 4 skipped, 0 failed**. 57 command tests.

---

## Attempt 3's status, corrected

Attempt 3's card said "**RAN and PASSED**". It shouldn't have: two JVM executions happened outside
the authorization, and an unqualified "PASSED" cannot absorb that. Its status line now reads
**technical checks passed; authorization compliance failed**, with a visible correction block naming
the previous wording. The scope failure and the technical result are recorded separately.

## 1 — [P1] The checked checkpoint is now the loaded checkpoint

`check_checkpoint` verified a supplied path, while `load_reference_evaluator` resolves its own
repository-relative path — so a **byte-identical copy elsewhere** would pass the precondition while
different bytes were opened. The check now requires the supplied path to **be** the path the loader
resolves.

```
PASS   the decoy copy is BYTE-IDENTICAL
PASS   NEGATIVE CONTROL: a byte-identical copy ELSEWHERE is refused
```

The control is deliberately built from a *byte-identical* copy, so it fails only on identity of
location — a copy with different bytes would have been caught by the hash anyway and would prove
nothing.

## 2 — [P1] Verified identities are durable, and setup runs under the harness

`records` — repo HEAD and plan blob, four JDK hashes, the JAR hash, both checkpoint hashes — now
reach the harness as `_identity` and are **fsynced in the run header before any setup runs**.
Compilation, model loading and collaborator construction moved **into** a `setup` callable the
harness invokes after that header, under a new `PHASE_SETUP` classification.

```
PASS   the header is the FIRST durable record
PASS   every verified identity is in the header: [checkpoint, jar, jdk, output_path, plan, repository]
PASS   the identity header is fsynced BEFORE setup runs
PASS   a SETUP failure is classified as PHASE_SETUP
PASS   ...and the identity header survives it, with a durable abort record
```

Previously a setup failure left no run identity and no abort record at all.

## 3 — [P2] The class directory is run-unique and exclusive

`<results-parent>/t1j_classes` was shared and created with `exist_ok=True`, so a second or concurrent
run could merge with or overwrite an executable class directory. It is now
`<results-path>.t1j_classes` — tied to the exclusively-created results path — made with a bare
`os.makedirs`, and **refused if it already exists**. `_default_compile` records **compiled-source and
produced-class identities** into the run.

## The proxy error, a third time

My check for `exist_ok` was a raw-text search, and it matched **my own comment explaining why
`exist_ok` is wrong**. That is the third raw-text check in this workstream to match prose rather than
code — after `os.environ` in a comment and `canonical_tasks_executed` counting names. It walks the
AST now, looking for an actual `makedirs(exist_ok=True)` call.

## Unchanged, re-run

Six preconditions in fixed order, each immediately fatal, all before the gate; no effectful seam
reached; no results file; `random.Random` patched shows **0** generators; the real harness accepts
the canonical 32 in screen mode and aborts in a refusing state factory — **before any agent, and with
the binder substituted, before any JVM**; `qualify` still refuses canonical seeds; the harness's
public entry point and the runner CLI still refuse screen mode.

**No JVM was launched in this attempt.** The host Java preferences plist remains `6cb3a052…`,
byte-identical to E2 attempt 4's recorded value.

## Post-run closeout (added after `a68de55`)

C-series evidence added separately. No runtime artifact changed and `09_MANIFEST` still verifies
8/8. Full suite after the test correction: **3130 passed, 4 skipped, 0 failed**.

- **A** — all **7 tracked rows** match `git rev-parse a68de55:<path>`.
- **B** — the checkpoint has **no object in the commit**: gitignored, disk-only.
- **C** — all six prior evidence packages, per file, against the commit that added *that file*:
  **zero changed**.
- **D** — two **declared amendments**: the attempt-3 card, relabelled by `a68de55`; and
  `tests/test_e4_screen_command.py`, corrected by this closeout.

Four controls, all invoking the real checker, all passing.

### The test correction

One assertion checked the **retired** `<parent>/t1j_classes` path, which is never created any more,
so it passed vacuously. The compile seam is faked in that test, so setup *does* run and *does* create
the directory — what must not happen is a `javac` or `java` subprocess. The test now asserts the fake
seam ran **exactly once**, the run-specific `<results>.t1j_classes` **was** created, the retired
location was not, and — by watching `subprocess.run` — that **no javac or JVM subprocess was
spawned**. Watched, not asserted.

---

## What this establishes, and what it does not

**Established:** the command verifies the file it will actually open; every verified identity is
durable before anything effectful runs, and setup failures are classified and recorded; the class
directory cannot be shared or silently reused; and the gate is still shut.

**Not established, and not claimed:** that the screen has run, or that flipping the constant suffices
in practice — **no game, no move, no model, and the play loop past the state factory has never run
against the canonical schedule**. Strength and placement are untouched. `[202612128, 202612160)` is
unspent.
