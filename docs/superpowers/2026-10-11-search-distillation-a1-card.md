# Search distillation — A1 execution card (PROPOSED, revision 2, 2026-10-11)

**Status: proposed for review. Nothing here is authorized yet.** Every numbered step in §5 needs its own
explicit approval: each commit, the survey, each proof, the registration, model loading and the games,
and the closing. No gate is added or changed. Nothing is pushed.

**Approved 2026-10-11:** P0, P1, S1 and S2, with commits C1–C3 only. Stop before S3 registration.

**S1 attempt 1 failed (exit 1, 216 s), preserved.** `02_seed_survey_run.txt`: the block was CLEAN but 44/45
controls passed. The failing control, "post: a plan bound to another survey record", built its wrong plan
with the same all-zero placeholder hash that survey mode uses for the absent record, so the wrong plan
equalled the right one. A control-fixture defect; seed selection is unchanged. **Recovery (approved):**
the control now flips the first hex digit of the hash, so it differs in every mode; the failed record is
kept unchanged, committed, and exempt by its exact path (appended last to `OWN_RECORDS`); the rerun writes
`02b_seed_survey_run.txt`, which the proofs and registration read. The rerun must exit 0 with 45/45 controls
and the same two-seed block, or the sequence stops.

**Purpose.** A1 is a planning measurement and real-model wiring check for Stage A of
`docs/superpowers/2026-10-11-search-distillation-proposal.md` (103a9222), on the A0 harness (43ab8d9b).
It measures loading time and per-game runtime at 1,600 vs 400 simulations, so that Stage A's runtime cap
can be frozen from a measurement. **Its two scores give no strength verdict** and are never pooled into
Stage A.

## 1. What runs

| Item | Value |
|---|---|
| Model | `calib020_0001` = `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors` in the **main checkout**, by absolute path; SHA-1 `209cf2d4fd24a48553d259dd71b4954867b9473e` verified before loading. Not copied. |
| Environment | The main checkout's `.venv/bin/python`, by absolute path. Nothing installed. |
| Tree | A clean linked worktree (§5 P1). The capture checkout is left as it is. |
| Agents | `sims1600` and `sims400`: the same checkpoint, `AgentSpec(..., mcts_sims=1600)` and `AgentSpec(..., mcts_sims=400)`. Both budgets are explicit, so the binding check verifies each one. |
| Games | 2, serial, colours reversed: game 0 has `sims1600` red, game 1 has `sims1600` black (`build_agent_pairing_tasks`). |
| Protocol | `external_learning_stage3.PROTOCOL`, imported unchanged: 24×24, 280-ply cap, `mcts_eval_batch_size` 14, `mcts_stall_flush_sims` 48, opening-temperature readout (20 plies at 1.0, then 0.1). `search_with_root`: a fresh root each ply, no noise. Only the per-agent budget differs. |
| Path | `eval_runner.run_game_tasks([task], 1, ...)` once per game: in-process, the per-ply budget check, the per-game binding check, replay sidecars. No worker, no new runner. |
| Seeds | Slots 0 and 1 of the A1 plan (`base`, `base + 1`), registered first (§2). No seed literal appears in any script. |
| Cap | **1,800 s in total, including loading and cleanup.** `timeout -k 10 1790`: TERM to the whole process group at 1,790 s, KILL 10 s later. No extension, no replacement games, no retry. |

**Planning figures, not bounds.**
* Typical: about 155 s per game (2.5 × the accepted 61.9 s for 400 vs 400), so two games take about
  5–6 minutes plus loading.
* **Full-cap extrapolation:** two 280-ply games at about 2.46 ms per simulation come to about 23 minutes
  (each game 140 × 1,600 + 140 × 400 simulations). This extrapolates a per-simulation cost measured at
  400; it is not a runtime bound. If the cap fires, the result is "no verdict" (exit 124 or 137).

## 2. Seed accounting — the established method

The method is the stage 1–3 method, imported by sha256 from the tactical line's survey and collision proof:
the union scanner, the prior terms (four registries, CONSUMED_SEEDS, named spent blocks, H3's generation
ranges, the four XOR stream masks), the check, its controls and the fail-closed enumeration. Survey →
pre-registration proof → registration → post-registration proof → (run) → usage derivation → closing edit.
Script: `docs/superpowers/evidence/search-distillation-a1-seed-survey/01_seed_survey.py`, modes
`survey`, `pre`, `post`, each writing one create-only record (02, 03, 04).

**Three adaptations, for review:**
1. **The rule** is the replacement rule's first use: `seed_selection.candidate(reserved, 2)` (ca9cb1eb),
   BASE = the least multiple of 100 leaving at least 2 unused seeds above the highest registered or
   named-spent seed. On today's registries that is the first multiple of 100 above stage 3's block.
2. **The differential controls** (planted literals) plant at offsets up to +9 from BASE, written for a
   32-seed block. They run unchanged on a 32-seed **probe** block at the same BASE, which must itself be
   CLEAN. The real 2-seed block gets the negative, exemption, enumeration and state controls directly.
3. **The scope covers two trees.** The pinned scope is every regular file under `scripts/`, `tests/`
   and `docs/superpowers/`, whatever its git state. A fresh worktree lacks the main checkout's untracked
   and ignored files, including about 41,800 ignored game logs under `scripts/GPU/logs/games`. So the
   survey and both proofs also enumerate (fail-closed) and scan the main checkout, which is found as the
   parent of the shared git directory. Nothing is narrowed. Only the worktree must be clean.

**The plan** (`docs/superpowers/evidence/search-distillation-a1/plan.json`, created at registration):
`{"kind", "base", "n_slots": 2, "survey_record_sha256"}`, never the exclusive end.

**Closing rule, fixed now** (`docs/superpowers/evidence/search-distillation-a1-seed-closing/01_derive_usage.py`):
* **RETIRE both seeds**, whatever happened.
* **EXPOSE only the started-game prefix**: slots `0..k-1`, where k is the number of `game_start` events
  in `progress.jsonl`. Each is logged *before* its game is played, so an interrupted game counts as
  started.
* A run that stops before either game (REFUSED, a loading error, a timeout during loading) **exposes
  neither seed**.
* The derivation reads only event names and slots, the outer status, `record.json`'s verdict and the
  registry; never a score or a move.
* It reconciles only if termination is confirmed (no survivors), the slots form a prefix, exit 0 means
  k = 2 and COMPLETED, exit 4 means k = 0, and the block is ACCOUNTED once and not yet EXPOSED or RETIRED.
* Its record carries 12 synthetic controls covering these cases.

## 3. Recorded

Everything goes to `docs/superpowers/evidence/search-distillation-a1/`, create-only.

* **`progress.jsonl`**: `start` (HEAD); `loaded` (seconds); per game `game_start` (game, slot, red
  agent) and `game_end` (seconds, plies, reason, both recorded budgets); `end` (verdict).
* **`record.json`**, written at the end unless the process is killed:
  * environment (Python, platform, MLX, numpy, interpreter path) and HEAD;
  * **the checkpoint path and its verified SHA-1**;
  * the load time;
  * **the effective protocol**: `PROTOCOL` as imported, the resulting `EvalConfig`, and both
    `AgentSpec`s with their readouts and budgets;
  * per game: `asdict(result)` (colours, `n_moves`, reason, `red_mcts_sims`/`black_mcts_sims`, the
    outcome fields uninterpreted), slot, seconds, and the set of `root_total_visits` per colour from the
    replay;
  * the repository modules loaded, with sha256;
  * verdict, reason and total seconds. `strength_verdict` is null.
* **`replays/`**: the runner's sidecars.
* **`outer.json`**, from the wrapper: **the outer exit status**, wall seconds, **survivors** (any process
  still matching the run script after the timeout returns), the tree, interpreter and checkpoint paths.
* **`run.stdout.txt`, `run.stderr.txt`**: the run's output.

## 4. Stops and exit statuses (outer status, from `outer.json`)

| Status | Meaning |
|---|---|
| 0 `COMPLETED` | Both games done. A measurement only. |
| 1 | One-shot refusal: an output already exists (wrapper) or `progress.jsonl` exists (script). Nothing is loaded. |
| 2 `STOP_INTEGRITY` | Any `IntegrityError` (budget, binding, telemetry). The record is written; no further game. |
| 3 `STOP_ERROR` | Any other exception. The record is written. |
| 4 `REFUSED` | Before any game: a bad `--checkpoint`, wrong SHA-1, a plan that is not 2 slots, a seed not ACCOUNTED or already unavailable, or (after loading, before game 0) a loaded repository module that is uncommitted, modified, or loaded from outside the worktree. |
| 124 | TERM at 1,790 s: **no verdict**. `record.json` is absent; `progress.jsonl` shows how far it got. |
| 137 | KILL after the 10 s grace: **no verdict**, as 124. |
| other | `timeout` itself failed: no verdict, review. |

There are no retries. The module check covers what is loaded by the time loading ends (the run script, the
registry, the runner, the loader, the network). It does not see modules imported later inside the games.

## 5. The exact sequence

`MAIN` = `/Users/bill/projects/TwixT_Game` (the capture checkout); `W` = `/Users/bill/projects/TwixT_Game-a1`.
Every Python command runs as `env PYTHONPATH=. $MAIN/.venv/bin/python -B ...` from `W`. Long steps run in the
background (the tool's foreground limit is 10 minutes). Any non-zero exit, any red test, or a timeout stops
the sequence for review.

| Step | Where | Action | Commit |
|---|---|---|---|
| P0 | MAIN | Commit **only** this card and the four scripts: `.../search-distillation-a1-seed-survey/01_seed_survey.py`, `.../search-distillation-a1/01_a1_run.py`, `.../search-distillation-a1/02_a1_wrapper.sh`, `.../search-distillation-a1-seed-closing/01_derive_usage.py`. The capture work stays out. | C1 on `main` |
| P1 | MAIN | `git worktree add -b search-distillation-a1 $W C1`; check that `git -C $W status --porcelain` is empty. | — |
| S1 | W | `timeout 1200 ... 01_seed_survey.py --mode survey` → record 02b (attempt 2; attempt 1 = record 02, failed); must exit 0 with `THE A1 BLOCK IS CLEAN`, every control passed. | C2: record 02b |
| S2 | W | `timeout 1200 ... 01_seed_survey.py --mode pre` → record 03; must exit 0. | C3: record 03 |
| S3 | W | **Registration.** (a) Append the block read from record 02b, `(BASE, BASE + 2)`, **last** to `ACCOUNTED_SEED_INTERVALS` in `e4_screen_reference.py`, with a comment holding no band literal. (b) Create `plan.json` from the record with the script's own `build_plan` (snippet below), create-only. (c) Run the data-only check below once; every test must pass. | C4: registry + plan |
| S4 | W | `timeout 1200 ... 01_seed_survey.py --mode post` → record 04; must exit 0 (`STATE HOLDS`, CLEAN). | C5: record 04 |
| R | W | On the clean tree at C5: `sh docs/superpowers/evidence/search-distillation-a1/02_a1_wrapper.sh` (background). When it returns, check `outer.json`: `survivors` must be empty. Also check that `pgrep -f search-distillation-a1/01_a1_run.py` prints nothing. | C6: all run outputs, as written |
| K1 | W | `... 01_derive_usage.py` → record 02 in the closing directory; must exit 0 (`RECONCILES`). | C7: record 02 |
| K2 | W | **Closing edit.** Append `(BASE, BASE + k)` last to `EXPOSED_SEED_INTERVALS` only if k > 0. Append `(BASE, BASE + 2)` last to `RETIRED_SEED_INTERVALS`. ACCOUNTED is unchanged. Run the data-only check below once; every test must pass. | C8: registry |
| F | MAIN | `git merge --ff-only search-distillation-a1`. The branch touches no capture file. Then `git worktree remove $W` and `git branch -d search-distillation-a1`. | none (fast-forward) |

The data-only check, for S3 and K2 (no other tests are run under this card; the collision proofs verify the
exact registry edit separately):

```sh
timeout 60 env PYTHONPATH=. "$MAIN/.venv/bin/python" -B -m pytest \
  tests/test_seed_selection.py -q -p no:cacheprovider
```

Commit order matters for the run's module check. The run script (C1) and the registry with the block
(C4) must both be committed before R, or R refuses with exit 4.

Snippet for S3(b), run from `W`:

```python
import hashlib, importlib.util, json, sys; sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("a1s", "docs/superpowers/evidence/search-distillation-a1-seed-survey/01_seed_survey.py")
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
S, P, SEL = A.load_method()
raw = open(A.OUT["survey"], "rb").read()
rec = A.read_record(raw.decode(), S, SEL)
with open(A.PLAN, "x") as fh:
    fh.write(json.dumps(A.build_plan(rec["base"], hashlib.sha256(raw).hexdigest()), indent=1, sort_keys=True) + "\n")
print(rec["block"])
```

## 6. What A1 does and does not establish

* **It does:** measure the real loading time, per-game time and plies at 1,600 vs 400 under this
  protocol. It also shows that per-agent budgets hold with the real model: the per-ply checks pass, and the
  recorded budgets and `root_total_visits` are 1,600 and 400.
* **It does not:** give any strength reading. Two games carry no statistic, and their scores are not
  Stage A data.
* **After A1:** a Stage A runtime cap with headroom is proposed from the measured times. Reaching the cap
  gives no verdict and no measurement. That proposal, Stage A's runner (with per-agent budgets in its
  summary) and Stage A's seeds are separate work.
