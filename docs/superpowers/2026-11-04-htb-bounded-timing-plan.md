# H-TB bounded timing plan (2026-11-04) — PLAN ONLY

**Status:** a paper plan for card §10 step 3 (card sha256 `96269e17…`). Nothing here is built or authorized. It
needs its own build (gate, entry, tests) and approval, and then a run approval.

**Purpose:** measure the time per unit of the label and fit runs on this machine, then set their ceilings at 1.5× the
projection. **Train-arm rows only. Time only:** no teacher output, label, prediction or loss value is written,
compared or inspected.

## 1. Preconditions for any timing run

- Environments:
  - The teacher environment from `docs/superpowers/evidence/htb-teacher-setup/`. Python 3.13.15 (standalone, archive
    sha256 `003d459a…`), the baseline freeze exactly (TensorFlow 2.21.0), and the checkout re-verified at launch by
    `twixtbot_g3_runner.verify_anchor_runtime`.
  - The project `.venv` (Python 3.14.10, MLX 0.30.3, NumPy 2.4.1) for the fit, with live versions read before launch.
- Machine state: an idle gate that the launch depends on, a BOINC check before and after, Unreal Editor closed, and a
  panic/reboot check before launch.
- Gate: a new closed gate, `HTB_TIMING_AUTHORIZED` (the 64th). It is opened for one run in a clean worktree and
  restored after. Fresh, create-only destinations: `docs/superpowers/evidence/htb-timing/{label,fit}/` and
  `~/twixt-cache/htb-timing/`. The run refuses if any exists.
- **Arm guard:** the timing entry receives only C0's train arm. It refuses any game whose `split_of` is val or test.
  It never calls `select` (which reads val keys), so no val or test position is replayed or keyed.

## 2. T-label — teacher environment, CPU TensorFlow (ceiling: TERM 1,200 s, KILL +10 s)

| Step | What is timed | Size (fixed in advance) |
|---|---|---|
| L1 | Corpus authentication, `eligible_games(arms=("train",))`. It reads, hashes and parses every file, as in the real run | the whole corpus, once |
| L2 | Replay and `state_key` throughput | every position of the first 200 train-arm games in sorted-id order (about 10.8 K positions) |
| L3 | `label_game`, both orientations, lockstep in both engines: per-call `eval_one` time and per-game replay time | plies with ply % 5 == 0 of the first 40 train-arm games in sorted-id order (about 430 rows, about 860 calls); the first 20 calls are reported but excluded from the rate |
| L4 | Writing shards | one 8,192-row shard of **zeros** (synthetic, not teacher output) to `~/twixt-cache/htb-timing/` |

- **What the record keeps:** counts, wall and CPU seconds, versions, and the idle and BOINC checks. Teacher outputs
  are discarded in memory; a test asserts the record has no target or value fields.
- **Isolation check:** the process asserts at exit that `mlx` was never imported.
- **Projection.**
  - The terms:
    - L1;
    - replay and keying of all train and val positions (C0 counts: 1,676,453 + 207,336) at L2's rate;
    - 528,000 calls at L3's steady per-call time;
    - lockstep replays for the games the selection will touch, two orientations each, at L3's per-game time (bounded
      above by C0's 34,849 train and val games);
    - 33 shards at L4's rate.
  - **Label ceiling** = 1.5 × the sum, rounded up to the next 10 minutes.

## 3. T-fit — project `.venv`, MLX on the CPU, `MLX_ENABLE_TF32=0` (ceiling: TERM 1,800 s, KILL +10 s)

| Step | What is timed | Size (fixed in advance) |
|---|---|---|
| F1 | `build_examples` throughput | 640 rows: plies with ply % 5 == 1 of train-arm games from game 41 on, in sorted-id order, until 640. The targets are **synthetic uniform** distributions over the legal moves and the values are 0, because the timing does not depend on them and no teacher output exists |
| F2 | Loading `calib020_0001` twice, after its SHA-1 check | 2 loads |
| F3 | Updates of the frozen shape: 32 rows × 2 orientations, Adam 1e-4, clip 1.0, BatchNorm in evaluation mode | 60 updates; the first 10 are excluded as warm-up. The student is discarded, not saved |
| F4 | Mirror-averaged scoring | the 640 rows, two networks |

- **What the record keeps:** counts and seconds only. No loss, KL or other metric value is recorded.
- **Projection.**
  - The terms:
    - examples for 264,000 rows at F1's rate;
    - two loads;
    - 15,000 updates at F3's steady s/update;
    - scoring of 2,048 + 2 × 24,000 rows at F4's rate;
    - one save.
  - **Fit ceiling** = 1.5 × the sum, rounded up to the next 10 minutes.

## 4. Outputs, interpretation and cost

- Two create-only records (`record.json`, each with `started.json` and a terminal status), plus a short projection
  note proposing both ceilings for review. The ceilings are not adopted until reviewed.
- The timing rows are train-arm positions and some may later be selected for training. That is harmless, because
  nothing but time is recorded. Timings depend on idleness, so an idle-gate deviation is disclosed, not discounted.
- **Cost:** at most 50 minutes of wall time including both ceilings.
- **Build needed first,** for review:
  - a `time label|time fit` entry in `htb_pilot.py` that reuses the run code paths;
  - the gate;
  - two wrappers;
  - focused fake-entry tests: arm guard, no teacher outputs in the record, no MLX in the label process, refusals
    before any load, terminal records.
