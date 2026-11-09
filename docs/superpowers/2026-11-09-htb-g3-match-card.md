# H-TB integration G3 — smoke and confirmation match card (2026-11-09)

**Status:** BUILT for one launch review; **repaired 2026-11-09** after the user's launch-clearance review (§7: the
wrapper's outer record is now required at both stages and for match admission; the runtime binding is extended). No
game has been played, no model has been loaded for G3, and no gate has been opened.

**Authorized (the user's G2 review, 2026-11-09):** building and committing this plan and its driver. The review fixed the
timing rule (§5) and kept the comparison agreed in D8 (§3, §6).

**Not authorized:** any game (smoke or match) before the launch review; deployment; training; pushing.

**Sources:** the integration proposal, revision 1 (`docs/superpowers/2026-11-06-htb-integration-proposal.md`), §7 G3
(V8a, V8b) and decisions D4 and D8; and ledger entries 54–61.

## 1. The question

**Question:** does the product's new `hard` beat today's `hard` when both are served by the product's own code?
- **New `hard`:** the pinned student, mirror averaging, 200 simulations, root halving from ply 20, and sampling at
  T 1.0 for plies 0–19.
- **Today's `hard`:** calib020_0001, 800 plain simulations, argmax.

**Why this test is needed:** entry 54's PASS (217–2–1) was against the original proven combination on the research
stack. It reaches the product only by argument, so this match is the only direct evidence about the change users would
see (proposal §7 V8b).

**What it compares:** two product settings, **not matched compute.** New `hard` runs 200 simulations with two network
calls per leaf; today's `hard` runs 800 with one. Both sides' costs and latencies are recorded (§10); neither decides
the verdict.

## 2. Players, bound to the product's code

Both players run in **one Node process** with the product's modules. No research code is involved.

| | new `hard` | today's `hard` |
|---|---|---|
| model | `78c16a31c3dba37f` (the student; `resolveHardModel`, `HARD_MODEL_ID`) | `c34b7ff3297c785a` (`resolveModel`, `DEFAULT_MODEL_ID`) |
| session | `AlphaZeroInference` + `assertSessionContract`, wrapped in `MirrorAveragingInference` | `AlphaZeroInference` + `assertSessionContract` |
| policy | `resolvePolicy({difficulty: 'hard', hardMode: 'protocol'})` (200 sims, T 1.0, `protocol`) | `resolvePolicy({difficulty: 'hard', hardMode: 'legacy'})` (800 sims, T 0, `plain`) |
| move | `protocolMove` (`server/protocol_move.js`), the function both transports call | `new MCTS(...)`, `search`, then `selectMoveForRequest`: `computeBestMove`'s plain branch |

- **Model loading:** both models load exactly as `server/index.js` loads them at startup. An empty environment is
  passed to the resolvers, so no override can reach them.
  - The driver refuses to run if `MODEL_MANIFEST`, `HARD_MODEL_MANIFEST` or `HARD_MODE` is set. The wrapper also unsets
    them.
- **Requests:** each move's state goes through `toDict`/`fromDict`, as a request's state does.
- **The one deliberate departure from a served request:** `protocolMove` receives `seeds`, its test-only argument
  (§4), instead of drawing fresh random ones. The product's per-request structure is kept: two fresh streams per move,
  with nothing carried between moves.
- **Model-free checks of this wiring** (§9): the new-`hard` adapter returns `protocolMove`'s move for the same seeds.
  The today's-`hard` adapter returns the move that the plain branch's MCTS search followed by `selectMoveForRequest`
  gives.

## 3. Games, pairs, scoring

- **Board and rules:** the empty 24×24 board, red first, no swap.
- **Game end:** the product's rule, played to `state.isTerminal()` with **no ply-280 cap**. On 24×24 the product's
  ply-600 limit is unreachable, so in practice a game ends with a win or with no legal moves (a draw); proposal §9
  item 6.
- **Pairs:** pair *p*, game 0 has new `hard` as red; game 1 has new `hard` as black.
  - Today's `hard` is deterministic, so a pair's two games differ only through new `hard`'s sampling and ties.
- **Smoke:** **1** pair (2 games). It is excluded from the match.
- **Match:** **110 fresh pairs** (220 games). No smoke game is reused, and no seed overlaps (§4).
- **Score:** new `hard` scores 1 for a win, 0.5 for a draw and 0 for a loss. The pair score is the mean of its two
  games, so the mean pair score equals the mean over all 220 games.

## 4. Seeds

- **The seed formula:**
  `sha256("htb-g3/20261109/<stage>/pair <p>/game <g>/ply <ply>")`. Hex digits 0–11 are the search seed and hex
  digits 12–23 are the readout seed.
  - These are 48-bit integers, the range `protocolMove` draws its own fresh seeds from.
  - `<stage>` is `smoke` or `match`.
- **Who uses them:** only new `hard`'s moves. Today's `hard` draws no randomness.
- **The seed namespace:** `MATCH_SEED = 20261109` is fixed here, before any game. No earlier run used the `htb-g3`
  namespace.
- **Checked (§9):**
  - The seeds are deterministic and below 2^48.
  - All seeds that the smoke and match can reach (every pair, both games, plies 0–576) are **distinct**.
  - The analyser derives them independently and gets the same values.

## 5. The timing rule (the user's G2 review), fixed before the smoke

### 5.1 The smoke's own ceiling

- **Ceiling:** TERM at **2,390 s**, KILL at **2,400 s**; cleanup is observed through 2,410 s.
- **What it covers:** two games plus startup.
- **The planning estimate it rests on** (not a bound):
  - **About 1.37 s per move for new `hard`:** G2's live check measured 1.36–1.40 s.
  - **About 3.0 s per move for today's `hard`:** derived from G2's rollback reference capture. Its 142 positions ×
    (800 + 100 + 400) simulations took 689.6 s, or about 3.74 ms per simulation.
  - **Two 64-ply games** then take about 280 s.
  - **Within the ceiling:** about 1,100 plies, at the mean of those two per-move times.
- **If the smoke is stopped:** VOID, with no retry; I return it.

### 5.2 The projection and the hard ceiling (g3_analyse.mjs `timing`, from the validated smoke only)

- **L,** the projected game length, is the greater of **64 plies** and the smoke pair's mean length. It is a planning
  assumption, not a bound.
- **s_new and s_legacy** are each player's measured seconds per move over both smoke games. Each is that player's
  total time in its move calls divided by its number of moves.
- **overhead,** the game-recording overhead in seconds per game, is the mean over the two smoke games of the game's
  time segment minus its total move time.
  - A game's segment runs from the end of the previous game (or of startup) to the moment its progress line is written.
  - It therefore includes the state round-trips, the atomic sidecar write and the background-compute check after the
    game.
- **startup** is the time from process start to both models loaded and contract-checked, `performance.now()` in the
  smoke process. It includes module loading and the binding check.

```
projection = startup + 220 × ( (L / 2) × (s_new + s_legacy) + overhead )
kill       = ⌈ 1.5 × projection / 60 ⌉ × 60        (the hard ceiling, in whole minutes)
term       = kill − 10 s
cleanup    = observed through kill + 10 s          (stated separately; not part of the 24-hour test)
```

- **The 24-hour limit applies to `kill`.** If `kill` ≤ 86,400 s the verdict is `WITHIN_24H` and the match may launch
  under that ceiling.
  - Otherwise it is `EXCEEDS_24H`: **I stop and return the estimate before the match starts**, and do not resize the
    match.
- **Excluded from the ceiling:** idle waiting (the idle gate) and the cleanup observation.
- **Planning estimate** (not a decision): at 64 plies, 1.37 s and 3.0 s per move, about 140 s per game. The
  projection is then about 8.5 h and the ceiling about 12.8 h.
  - That ceiling would reach 24 h at a projected length of about 120 plies.
- **The match wrapper:**
  - It re-derives `timing.json` from the committed smoke evidence and requires the result byte-equal to the committed
    file.
  - It then uses that file's `kill` as its ceiling.
  - The driver refuses unless `--kill-s` equals the committed value.

## 6. The verdict rule (D8, unchanged) and the prediction

- **PASS** if the mean pair score − √(ln 20 / 220) > 0.5, that is, if the mean is above 0.61669 (one-sided
  Hoeffding, 95%, on the 110 pairs).
  - Achievable totals are multiples of 0.5 out of 220. So **136/220 is the lowest passing total**, and 135.5/220 is
    the nearest failing one (checked in §9).
  - Otherwise the verdict is **FAIL**.
- **The verdict is only for a valid run:**
  - The match must be COMPLETED with all 220 games.
  - The analyser must validate it (§7).
  - Anything else is VOID or INVALID. It receives **no score**, and no partial score is computed.
- **A FAIL, VOID or INVALID run** is returned for assessment, with no retry.
- **Prediction, stated before any G3 game (mine):** **PASS**, with a mean pair score of about 0.93 (I expect it
  between 0.80 and 0.98).
  - **Basis:**
    - The student under this protocol scored 0.989 against the original proven combination (entry 54).
    - That combination had itself scored 0.679 over 509 pairs against the incumbent's plain search at 400
      simulations (entry 39's strength comparison, attempt 2).
    - Today's `hard` is plain calib020_0001 at 800 simulations with argmax.
  - **Reasons it could be lower:**
    - The product rule has no 280 cap, so longer games are untested.
    - The ORT runtime differs from MLX, within the V2-R/V3 tolerances.
    - The opponent is deterministic, so a weakness that new `hard`'s sampling finds could repeat across pairs.
- **What a PASS establishes:** stronger play than today's product `hard` on this machine and stack, under the product
  game-end rule. It is not evidence against humans or twixtbot, and **it does not authorize deployment**, which
  remains a separate decision (D4).

## 7. What is bound, and what the analyser checks

**`tests/htb_integration/g3_binding.json`** is written by `g3_match.mjs bind --write` and committed after this build
and before any launch. It records sha256 digests of:
- **the driver's whole relative-import closure,** computed from its import statements, minus the gate file:
  - `g3_match.mjs`;
  - `server/{gameLogic, inference, mcts, mirror_inference, model_manifest, protocol_mcts, protocol_move, py_random,
    readout_policy, root_halving}.js`;
- **the analyser's closure:** `g3_analyse.mjs` and `server/gameLogic.js`;
- **the model identities:** both manifests with their model ids. `resolveModel`/`resolveHardModel` re-hash the graph
  and data files at load;
- **the runtime:**
  - the Node version (v24.22.0) and **the Node executable's bytes** (`process.execPath`);
  - **every installed file of `onnxruntime-node` and `onnxruntime-common`** (1.23.2): their JavaScript, `package.json`
    files and native binaries for every platform. These are the only installed packages the run loads; importing
    `onnxruntime-node` puts nothing else in the module cache. Its other declared dependencies serve its install
    script only;
- **the files run alongside them:** `package.json`, `package-lock.json`, the background-compute check and the idle
  gate (`scripts/GPU/alphazero/{background_compute, idle_gate}.py`, with their two package `__init__.py` files), this
  card and both wrappers;
- **the match seed and the pair counts.**

**Not bound:** the operating system and its shared libraries, the hardware, and the Python interpreter that runs the
idle gate and the background-compute check. Python's path is recorded in each record. The binding covers the listed
bytes, not every byte the machine runs.

**The driver refuses before loading a model if any of these holds:**
- its stage's gate is closed;
- an override variable is set;
- a tracked file differs from HEAD;
- the binding file differs from HEAD;
- a fresh binding differs from the committed one;
- the stage's evidence directory exists;
- for the match only, **the smoke's evidence is not committed, complete and successful** (`smokeAdmission`):
  - anything under `htb-g3-smoke/` or `htb-g3-smoke-run/` is uncommitted or changed;
  - any of `record.json`, `timing.json`, `outer.json` or `pgid.txt` is missing or differs from HEAD;
  - the smoke record is not this binding's smoke;
  - the smoke's outer record fails the wrapper checks below;
  - the committed smoke timing is not `WITHIN_24H`;
  - `--kill-s` differs from that timing's `kill`.

**Wrapper checks** (the rule of earlier stages, restored). A stage's script-written COMPLETED record counts only if
its wrapper's `outer.json` shows:
- `inner_exit` and `outer_exit` both the integer 0;
- `outcome` `COMPLETED`;
- `gate_restored` true;
- `survivors` empty (so no unconfirmed cleanup);
- `record_by` `script`;
- the stage's ceiling: `term_s` 2,390 for the smoke, or the committed timing's `term` for the match, with `grace_s` 10;
- an integer `wall_seconds`.

**The outer record is tied to its inner run by two checks:**
- its `head` equals the record's;
- `pgid.txt` holds the record's `pid`. The supervisor's child is the run's process group leader, and it `exec`s `env`
  and then `node` in place, so the driver's pid is the group id the wrapper supervised.

The driver applies these checks to the smoke when admitting the match. The analyser applies its own, independent
copy to each stage it judges. The match wrapper re-derives the timing through the analyser, so a smoke whose outer
record fails cannot yield a ceiling.

**The analyser, which imports nothing from the driver:**
- first checks that its own files are the bound bytes;
- then requires, for every game:
  - the expected file set;
  - schema, stage, pair, game and colours;
  - both model ids, the binding digest, the HEAD and the Node version;
  - each move legal when replayed from the empty board with `TwixtState`;
  - the last position terminal, and no earlier position terminal;
  - result, termination, ply count and score equal to the replay;
  - per move:
    - the mover's label by colour;
    - 200 / `protocol` or 800 / `plain`;
    - `valueKind` `root_mean` below ply 20 and `played_move_mean` from ply 20 (new `hard`), and none for today's `hard`;
    - the seeds re-derived for new `hard` and none for today's;
    - a finite time;
  - `move_ms_total` equal to the sum of the move times;
  - a progress line in order, with its time increasing and agreeing with the sidecar;
- and, for the run as a whole:
  - the record COMPLETED with the full game count and the binding digest;
  - for the match, the record's `kill_s` equal to the committed smoke timing's;
  - the wrapper checks above, on `<stage_dir>-run/outer.json` and `pgid.txt`;
- and, for `verdict`, also:
  - the smoke's evidence committed with nothing pending;
  - the smoke validating in full, wrapper checks included;
  - its timing re-derived byte-equal to the committed `timing.json`.

## 8. Launch sequence (one launch review covers all of it)

**Where it runs:**
- in a **clean linked worktree** at the launch commit, with `node_modules` an APFS clone of the main checkout's;
- its bytes are checked by the binding;
- the main checkout's two preserved untracked files are not touched.

**Each wrapper:** derived from the H-TB confirmation test's wrapper. The changes are listed in its header.
- **Refuses** unless:
  - the tree is clean;
  - HEAD is a one-line commit opening only its gate (`export const G3_<STAGE>_AUTHORIZED = false;` → `true;` in
    `tests/htb_integration/g3_authorization.mjs`);
  - its output paths do not exist;
  - the idle gate (`idle_gate wait`) exits 0;
  - the background-compute check exits 0.
- **Supervises** the run's process group: TERM, then KILL 10 s later, with a bounded cleanup observation.
- **Records** `outer.json` and the survivors.
- **Restores the gate** on every exit.

The driver handles TERM by aborting the current search and writing a VOID record. It repeats the background-compute
check after every game, and a detection or failed check makes the run VOID.

**Steps, in order; each stop returns the result to the user with no retry:**

1. **The V8a browser check (interactive; no ceiling).**
   - Start `npm start` in the launch worktree, in the default mode. Through Claude in Chrome, I play the human side of
     one `hard` game in the browser client to its end, with recording off.
   - **What I check:**
     - the server banner names `Hard mode: protocol` and both model ids;
     - the AI's moves arrive;
     - the MCTS bar updates after each AI move, including across ply 20;
     - the console shows no client error;
     - the game ends normally.
   - **Records:** observations and screenshots in `docs/superpowers/evidence/htb-g3-browser/README.md`.
   - **If any check fails:** stop.
   - **If Claude in Chrome is not available:** I stop and ask, rather than skip this step.
   - **V8a's server items are not rerun.** The banner, model ids, `search`, `valueKind`, both transports and per-move
     time were covered by G2's live check, L1–L9 (entry 61). Do-not-repeat #190 bars rerunning it without a change
     that calls for it. The smoke pair adds per-move times split at ply 20 for both players.
2. **The smoke** (`htb-g3-smoke-run/01_wrapper.sh` after the gate commit).
   - **Stops** on anything but a wrapper `outcome` COMPLETED with exit 0, and on an INVALID analyser result. The
     analyser's wrapper checks cover TIMEOUT, failed gate restoration and survivors.
   - **Then:**
     - `g3_analyse.mjs timing docs/superpowers/evidence/htb-g3-smoke --write`;
     - commit the evidence and the gate restoration;
     - add a ledger entry for the smoke.
3. **The 24-hour test:**
   - `EXCEEDS_24H`: **stop** and return the estimate.
   - `WITHIN_24H`: continue.
4. **The match** (`htb-g3-match-run/01_wrapper.sh` after the gate commit, under the committed ceiling).
   - **Stops** on anything but a wrapper `outcome` COMPLETED with exit 0. The driver admits the match only on
     committed, successful smoke evidence, as above.
5. **The verdict:**
   - `g3_analyse.mjs verdict docs/superpowers/evidence/htb-g3-match --write`;
   - commit the evidence and the gate restoration;
   - add a ledger entry (long form, banner and do-not-repeat);
   - **return for review.**
   - A PASS does not lead to deployment.

## 9. Tests (model-free; `tests/htb_integration/test_g3.mjs`)

**Covered:**
- Both gates are closed in the committed file.
- `preflight` refuses a closed gate in the real state.
- The driver and analyser transcribe the same constants.
- Seed determinism, range and distinctness, with an independent re-derivation.
- **The player adapters,** with the G1 fake evaluator:
  - new `hard` equals `protocolMove` with the same seeds at plies 0 and 23, reproducibly;
  - today's `hard` equals the plain branch and is deterministic.
- **The game loop** with scripted games (a red chain that wins at ply 25 and a black chain that wins at ply 26):
  - colours, seeds only on the new side, termination and scores;
  - the abort signal;
  - a failed background-compute check giving VOID.
- **The timing formula** on a fake clock, against a hand computation:
  - startup 1 s; 0.1 s per move; overhead 2.65 s per game; L = 64;
  - projection 1,992 s; kill 3,000 s; term 2,990 s; cleanup through 3,010 s;
  - `EXCEEDS_24H` when over the limit; a measured mean above 64 plies used as is.
- **Ten kinds of tampering,** each caught and each with an untampered control, plus a VOID record, an extra file and a
  missing game.
- **The verdict boundary:** 135.5/220 FAIL and 136/220 PASS.
- **A synthetic match** validated and scored, refused under a different ceiling, and refused without a timing.
- **The wrapper's outer record** (added by the repair). Each case is checked against a clean control:
  - missing `outer.json`;
  - TIMEOUT over an inner COMPLETED;
  - the gate not restored;
  - survivors;
  - unconfirmed cleanup;
  - a wrapper-written record;
  - a string exit;
  - another HEAD;
  - another ceiling;
  - another process group (`pgid.txt`).

  The driver's and the analyser's copies of these checks agree on every case.
- **Match admission:**
  - each failure is refused against a clean control: missing `outer.json` or `pgid.txt`, every outer-record failure
    above, uncommitted smoke files, another binding, `EXCEEDS_24H`, and a different `--kill-s`;
  - in the real state, with no smoke committed, it refuses.
- **Mutation checks,** run once and not committed. Each guard was disabled in turn and the tests caught it: the
  analyser's wrapper check (2 tests failed), admission's outer check (2) and the driver's gate-restoration check (1).
- **The binding:**
  - it covers both import closures, minus the gate;
  - the analyser imports only `gameLogic`;
  - it includes the Node executable and every file of both runtime packages;
  - the committed binding equals a fresh one.

**Not covered:** a real product game. The smoke is the first, and the analyser validates it before any number is
used.

## 10. Records, costs and latency

**Evidence directories,** each create-only:
- `docs/superpowers/evidence/htb-g3-browser/`;
- `htb-g3-smoke{,-run}/`;
- `htb-g3-match{,-run}/`.

**`verdict.json` records:**
- the verdict;
- the total and mean;
- the bound and the lower limit;
- pair-score counts;
- wins by side and colour, draws and terminations;
- plies (mean, min, max);
- the number of distinct games;
- **for each player:**
  - seconds per move (mean, p50, p95, max), overall and split at ply 20;
  - simulations per move;
  - total search seconds;
- the ratio of the two players' seconds per move;
- the run's wall time against the projection and the ceiling.

The costs describe the two product settings and do not enter the verdict.

All historical failures and controls stay as they are: V2's FAIL, V7's MISMATCH, live-check run 1, and the earlier
evidence directories.

## 11. Not authorized by this card

- Any launch before the user's launch review.
- A second smoke or match attempt.
- Resizing the match.
- Changing the rule.
- Deployment, training and pushing, including after a PASS.
