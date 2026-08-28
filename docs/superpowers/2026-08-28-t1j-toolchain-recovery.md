# T1j Toolchain Recovery and Requalification

**Date:** 2026-08-28 · **Status:** COMPLETE. Suite **3,447 passed / 0 failed /
4 skipped**, restored from 41 identity failures.

**Authorized scope:** recovery and requalification only. No model was loaded, no
T1j move query issued, no seed registered or drawn, no game played, nothing
trained, nothing pushed, and D1 was not run. All three execution gates remain
`False`; the suite log shows no JVM start.

---

## 1. What had broken

The JDK and `t1j.jar` were pinned at a path inside a **session scratchpad** under
`/private/tmp`. That directory was cleaned and the artifacts went with it, turning
41 tests red at once. The tests were behaving correctly — failing closed on a
missing pinned component — but the qualification suite had a silent dependency on
ephemeral storage.

## 2. Reacquired from the recorded sources

Both URLs come from what E1/E2 recorded, not from anything chosen here.

| Artifact | Source | Verified against |
|---|---|---|
| `t1j.jar` | `https://github.com/johannesSchwagereit/T1j/releases/download/current/t1j.jar` | E1: 83,990 bytes, sha256 `53ec95e421db2531…`, sha1 `064370a89b8361cd…` — **all three match** |
| Temurin JDK 17.0.20.1+1 | Adoptium `temurin17-binaries` | `PINNED_JDK`: **4/4 components match** |

Placed at `~/Library/Application Support/TwixT_Game/toolchains/t1j-e1/` — outside
the repository and outside any temp directory.

## 3. Committed: the lock, not the artifacts

`scripts/GPU/alphazero/t1j_toolchain_lock.json` records source URLs, expected
hashes, version and relative layout. The artifacts stay out of the repo
(83,990-byte jar and a ~323 MB JDK tree).

The lock's JDK hashes are **read from `PINNED_JDK`, never retyped**, and a test
asserts the two cannot drift apart — a second hand-copied hash table is exactly
how a pin stops meaning anything.

## 4. Path resolution, repaired

`scripts/GPU/alphazero/t1j_toolchain.py`, 16 tests:

- **Explicit setting, with its source reported.** Argument → `TWIXT_T1J_TOOLCHAIN_ROOT`
  → recorded default, and `resolve_root` returns *which*. A default that names
  itself is not a silent fallback; substituting one without saying so is.
- **`/tmp` refused outright**, from any source, compared after `realpath` so
  `/tmp/x` cannot arrive disguised as `/private/tmp/x`. Not overridable — this is
  the failure the module exists to prevent.
- **Nothing returned unverified.** `verified_paths` hashes the jar and every
  pinned component *before* handing back a path. Tests cover tampered jar,
  tampered component, missing component, and wrong size.
- **Absent root fails loudly and actionably**, naming the lock file rather than
  guessing.

**Zero references remain in active runtime or test resolution** — no `.py` or
`.json` under `scripts/` or `tests/` mentions the old path.

> ⚠ **Not "zero repository-wide", which an earlier version of this card and of
> `c91bec0`'s commit message both claimed.** The path still appears **923
> times across 70 files** under `docs/` — historical evidence, transcripts and
> command provenance from runs that genuinely used it. That evidence is
> **immutable and is deliberately not scrubbed**: rewriting it would falsify the
> record of what those runs actually did. The original claim came from a grep
> scoped to `--include='*.py' tests/ scripts/` and was then reported as though it
> covered the whole repository.

## 5. Requalification

**3,447 passed, 0 failed, 4 skipped.** The count reconciles exactly:
3,419 at `47493e7` (before the loss) + 12 D1 tests added by the guard fixes + 16
toolchain tests = 3,447.

Verification costs 0.06 s for all 5 components, so the two repaired test
modules pay ~0.1 s at import. No caching was added.

## 6. Evidence

| File | Contents |
|---|---|
| `01_acquisition.txt` | sources, durable root, every hash compared to its pin |
| `02_lock.json.txt` | the committed lock record |
| `03_full_suite.txt` | full suite, 3,447 passed / 0 failed |
