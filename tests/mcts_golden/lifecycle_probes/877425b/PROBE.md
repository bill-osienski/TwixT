# Worker lifecycle probe — PASSED

**Date:** 2026-08-16 · **Outcome: clean exit, one complete artifact, no `.tmp`.**

One real single-case probe of the capture worker, run to test whether the lifecycle remedy
(`877425b` and its two predecessors) removes the native abort recorded in
`tests/mcts_golden/capture_failures/0a76252/`. **This is not a golden trace, not a corpus, not an
equivalence result, and not an authorization for the 92-case capture.**

## What was run

```
node tests/mcts_golden/worker.mjs G_P01_baseline_s1 \
     runs/probe_lifecycle_877425b \
     --expect-commit 877425ba29f92c141766cd4fa9d5cdb04be74d6d --capture
```

| | |
|---|---|
| commit | `877425ba29f92c141766cd4fa9d5cdb04be74d6d` |
| pinned surface commit | `74dca6e1535ee1e36d640dae3ba644c6c2ed2e5e` |
| execution-surface sha256 | `228f57b55448f44136ffd41d6f092c9da904ca469a1e7bc4055656ffd8ef77bd` |
| worktree at launch | clean |
| output directory | `runs/probe_lifecycle_877425b` (absent beforehand) |
| configuration | default Node heap, product ORT configuration, no options |
| started (UTC) | `2026-08-16T22:06:56Z` |
| ended (UTC) | `2026-08-16T22:06:57Z` |
| node | `v26.7.0` |
| onnxruntime-node | `1.23.2` |
| worker pid | `89074` |

## Result

| observation | value |
|---|---|
| **exit status** | **`0`** |
| **signal** | **`null`** — the status was returned, not the result of a signal |
| stdout | **204 bytes** — `stdout.txt`, sha256 `61102a85b6822851734e4bf242598d7398ebc1bca9e1611a61a83318750cca7d` |
| stderr | **0 bytes**, sha256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| output directory entries | **1** |
| `.tmp` files | **0** |
| artifact | **23,974 bytes** — `G_P01_baseline_s1.json`, sha256 `1837831953bd8fb85995716bd2fceb007f7ffc6a30025c513b491529158e20e2` |

**Status and signal were captured directly**, with no pipeline that could substitute another
command's status.

**On the empty stderr.** It is recorded here as a fact — 0 bytes, with the sha256 of empty input —
rather than committed as a file, since an empty file is a poor carrier for that claim. Empty
stderr is a *positive* result: it means no `SESSION_RELEASE_FAILED`, no
`SESSION_RELEASE_UNAVAILABLE`, no `SECONDARY` line, and no native abort message was produced.

**The artifact is complete:** `status: "captured"`, 524 visit-count entries matching
`fixture.n_legal`, summing to `1` for `n_simulations: 1`, `root_value −0.09874988347291946`,
`selected_move "6,17"`, and one progress entry.

## Comparison with the pre-abort artifact — scope stated precisely

The failed capture at `0a76252` published one artifact for this same case before aborting. Every
comparison-controlled field matches:

```
DIFFERENCES (failed capture 0a76252  ->  probe 877425b)
  capture_commit               failed=0a76252…   probe=877425ba…
  pid                          failed=63321      probe=89074
  trace.progress_elapsed_ms.0  failed=16         probe=24
```

`visit_counts` (all 524 entries, in order), `root_value`, `selected_move` and `progress`
(`done`/`total`/`valueEstimate`) are identical, as is the entire fixture descriptor.

**What this establishes:** for **this single one-simulation case**, every comparison-controlled
trace field reproduced exactly across the two runs.

**What it does NOT establish:** general search determinism, or that nothing else could differ
elsewhere in the 92-case matrix. Fifteen other positions, four other simulation counts, a second
model and two abort cases are untouched by this observation. The result is **consistent with the
remedy affecting lifecycle only**; it does not demonstrate that.

**A side effect worth recording:** the only differing trace field is `progress_elapsed_ms`, which
§4.3 of the design deliberately excludes from equality because it derives from `Date.now()`. Had
the original rule requiring `elapsed` to match survived review, this comparison would have failed
a correct implementation.

## What the probe does not prove

**n = 1.** The abort did not recur, which is consistent with the lifecycle remedy working. It does
**not** prove the remedy caused the non-recurrence: the original abort may have been intermittent,
and no controlled comparison against the unfixed harness was run — none is authorized, and it
would require reverting the fix.

**The post-run log line belongs to this probe.** `stdout.txt` ends with
`captured G_P01_baseline_s1 pid=89074`, showing that this worker reached the line after
`runCase` returned. **It says nothing about how far the failed worker at `0a76252` progressed.**
That worker's stdout was discarded by the old orchestrator failure branch and no longer exists;
this probe cannot recover it retroactively.

## Status

The lifecycle remedy is **supported by one clean run**. The 92-case golden capture and any
`server/mcts.js` change remain **unauthorized**.
