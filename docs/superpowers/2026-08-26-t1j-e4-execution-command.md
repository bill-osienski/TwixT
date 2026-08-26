# E4 execution-command qualification — RAN, PASSED. The screen is built and refuses.

**Date:** 2026-08-26 · **Status:** **RAN and PASSED.** Unit tests **0**, qualification **0**,
stderr **0 bytes**. **No model, no agent, no RNG, no JVM, no game, no scheduled seed.**
· **`SCREEN_AUTHORIZED = False`. The 32-game screen remains UNAUTHORIZED.**

Basis: `main` @ `48f83d0`. Full suite: **3101 passed, 4 skipped, 0 failed**. 31 command tests.

---

## The gate

`scripts/GPU/alphazero/e4_screen_command.py` line 37:

```python
SCREEN_AUTHORIZED = False
```

Read **exactly once**, at line 232, as a bare module global. Proven by AST, not by grep:

| property | evidence |
|---|---|
| one assignment, one read | AST: 1 `Store`, 1 `Load` |
| no environment read | AST finds no `environ` / `environb` / `getenv` |
| no configuration reader | no `configparser` / `dotenv` / `yaml` / `toml` import |
| no enabling option | the CLI exposes only `--plan --results --repo --jdk --jar --checkpoint` |
| no injectable callables | `run_screen` takes six keyword-only **paths** |

Five environment variables (`SCREEN_AUTHORIZED=1`, `E4_SCREEN_AUTHORIZED=true`, `AUTHORIZED=yes`,
`FORCE=1`, `E4_SCREEN_ENABLE=1`) and five flags (`--authorized`, `--force`, `--yes`,
`--enable-screen`, `--screen-authorized`) were each tried in a fresh subprocess. The env vars
change nothing; the flags are rejected by the parser.

> One of my checks was wrong first: a raw-text grep for `os.environ` matched the **comment** saying
> the module never reads it. Prose is not code. It now walks the AST.

## Ordering, measured

```
all six preconditions ran, in order: ['plan','repository','jdk','jar','checkpoint','output_path']
NO effectful seam was reached: []
NO RNG was constructed ANYWHERE (random.Random patched): 0
NO results file was created while unauthorized
```

The seams record *attempted* evaluator loading, agent construction and RNG creation and raise if
touched — so "nothing ran" is measured. The RNG check patches `random.Random` itself, so it would
catch a generator built anywhere, not only through the seam.

## Each precondition, immediately fatal

| control | fatal at | after |
|---|---|---|
| a malformed plan | `plan` | — |
| a **reshaped** plan (32 names, reversed) | `plan` | — |
| a **dirty** worktree | `repository` | plan |
| a wrong JDK | `jdk` | plan, repository |
| a wrong JAR | `jar` | plan, repository, jdk |
| a wrong checkpoint | `checkpoint` | plan, repository, jdk, jar |
| an existing output path | `output_path` | the first five |
| a missing output directory | `output_path` | the first five |

No effect was reached in any of them, and the existing output file was left byte-identical. This is
the shape the integration script got wrong — its JDK gate stopped, its JAR and checkpoint gates only
accumulated. Here every one is fatal, and all six precede the authorization gate.

## The CLI

`--help` exits 0 and states the ban. A **fully valid** invocation — real plan, clean committed repo,
pinned JDK, JAR and checkpoint, fresh output path — exits **5**, names every completed precondition
on stderr, and **creates no file**. A precondition failure exits **2**.

Exit codes: 0 ok, 2 precondition refused, 3 aborted, 4 unexpected, **5 screen not authorized**.

---

## What this establishes, and what it does not

**Established:** the screen command exists, binds the canonical ordered 32-task schedule, makes every
identity check immediately fatal before anything effectful, and refuses on an authorization constant
that nothing outside the file can change — with the ordering measured rather than asserted.

**Not established, and not claimed:** that the screen *runs* — **the play wiring past the gate is
deliberately unreachable in this build**; anything about strength; and absolute placement. Enabling
the screen is a reviewed change to line 37 plus a separate authorization.
