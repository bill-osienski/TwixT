# Pilot card — twixtbot as an external strength anchor

**Date:** 2026-08-22 · **Status: DRAFT, nothing authorized to run.** · **Scope: anchor calibration only.**

This card **cannot** promote a checkpoint, produce a strength claim, or authorize training. Its only
output is a yes/no on whether twixtbot is usable as an external anchor, and if yes, at which setting.

## Hypothesis and prediction

**Hypothesis.** twixtbot (independently trained, unrelated lineage) can be configured to play our
exact rules, driven deterministically, and tuned to a search setting that is *unsaturated* against
both `0379` and `calib020_0001` — giving this project its first strength reference that is not
sibling-vs-sibling.

**Directional prediction, recorded before any run.** twixtbot at `trials=0` (raw policy, its
default) is beaten by both of ours; at high `trials` it is competitive with or beats both. Some
middle setting sits inside the band. Confidence is **low** — its network was trained under
different rules (below), and it may be saturated at every setting.

**Why it is not another closed line.** Every prior experiment measured our checkpoints against each
other. This measures nothing of ours against anything of ours. It is not on do-not-repeat `#1–#51`.

## Identity — everything hash-pinned

| item | identity |
|---|---|
| anchor repo | `github.com/stevens68/twixtbot-ui` @ `83749f230a0bae1766b46a05bfde0ed87f0a9a0a` (2026-02-21), MIT |
| upstream engine | `github.com/BonyJordan/twixtbot` (Jordan Lampe), MIT, ships the original model directory — **provenance only.** The **executed bytes come from the pinned UI commit's bundled `backend/` and `model/`**, which are what the hashes below cover. |
| anchor weights | one blob, three packagings — `variables.data-00000-of-00001` sha256 `1958f8476e9d56cbb87fa570db88f9cc9d389b30e20571fd66e8e779a4cefbab` (identical under `model/pb/`, `model/pbtxt/`, and as `model/meta/six-917000.data-00000-of-00001`) |
| | `saved_model.pb` sha256 `e0c2b882bc97c4661ac92af47ae5ef78443e57b42d0cb7c9c3a8cb7e8d01993b` · `variables.index` sha256 `1d4073c4da30e515a984faef2c3cc9c8b8ba402353a01933c8bb80e1f279f8cd` |
| our reference A | `calib020_0001` = `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`, SHA-1 `209cf2d4fd24a48553d259dd71b4954867b9473e` |
| our reference B | `0379` = `checkpoints/alphazero-v2-staged/model_iter_0379.safetensors`, SHA-1 `8ad62ac432c35c6ea9b0630b8a2b8c572a0b03a1` |
| our search | the **frozen research evaluation configuration used for the `0379` benchmarks** — 400 sims, `eval_batch_size 14`, `stall_flush_sims 48`, `selection_mode opening_temperature` (`opening_temp_plies 20`, `temp_high 1.0`, `temp_low 0.1`), `max_moves 280`, `draw_score_policy state_cap_and_board_full_score_0.5` — but **`--workers 1`**, not the benchmarks' 4. NOT the shipped hard configuration, which is the Node product stack at 800 sims. |
| fresh seeds | **`[202611000, 202611400)`** — reserve on authorization; disjoint from every consumed range through `202610188` |

## Rules — ours are authoritative, without exception

`allow_scl=False` · `allow_swap=False` · automatic links · no link removal · board 24×24
(`backend/twixt.py:49`, hardcoded — the UI's `board size` is **pixels**, not the game dimension).

**Our `TwixtState` is the authoritative game state — and both engines' states are bound at every ply.**
Move legality alone is insufficient: a legal move does not prove twixtbot evaluated the same
position, and given the crossing-rules difference the bridge graph is exactly where the two can
silently diverge. After opening replay and after **every** ply, assert normalized equality of:

- pegs (both colours), and **bridges** (as an order-independent set of endpoint pairs);
- side to move;
- the full set of legal moves;
- terminal status, and the winner where terminal.

Any divergence — or an illegal or unparseable move — **aborts that game loudly and is recorded**.
Never skipped, never resampled, never repaired mid-game. A divergence is a finding about the
adapter or the rules mapping, and the pilot stops until it is understood.

**The `allow_scl=True` arm is deliberately NOT run.** That setting changes the bridge graph, and our
engine cannot represent own-link crossings, so the two sides could hold different states after the
same moves. It would mix two rule systems rather than measure a handicap.

**The crossing-rules mismatch is a LIMITATION of the anchor, not a variable.** twixtbot's network was
trained with own-link crossings allowed; the repo ships the author's own counter-example
(`games/scl-issue.T1`: *"black should play …44: N12 but plays N10 — probably mislead because network
was trained with self crossing links allowed"*). We do not measure, correct, or compensate for it.
If it causes saturation under our rules, **we discard the anchor.**

## Three gates, in order. Any failure stops the pilot.

**G1 — install / import / model load.** Isolated venv; the project's own environment is untouched.
Import `backend.nneval`, load the SavedModel, run one `eval_one` on one fixed position.
**Pass:** finite `pwin`, `movelogits` of the expected shape. **Known risk:** `nneval.py` is TF1-compat
(`tf.disable_v2_behavior()`, `tf.saved_model.loader.load`) against `tensorflow>=2.20`.
**Fail → stop.** Report the error; do not patch the engine.

**G2 — determinism, measured not assumed.** One fixed position, settings `temperature=0`,
`add_noise=0`, `rotation=ROT_OFF`, **`trials=100`** — cheap, and it exercises the MCTS path so that
visit counts exist to compare. Query **20 times in one process** and **5 times in
5 fresh processes** — a within-process loop alone would miss framework-level nondeterminism.
**Pass:** all 25 return an identical move *and* identical top-3 visit counts.
**Fail → stop** and report the observed variation. Do not average it away.
(The only `random.*` call in the engine is under `ROT_RAND`, `nnmplayer.py:65`; `temperature` accepts
only 0/0.5/1.0; first move and swap bypass NN+MCTS entirely, so openings are supplied by us.)

**G3 — paired-opening saturation calibration.** 8 preregistered openings × both colours = **16 games**
per (setting × opponent), at `trials ∈ {0, 100, 400, 1000}` against **both** `0379` and
`calib020_0001`. Budget ceiling **128 games**.

**The single pass condition is non-saturation:** twixtbot's score rate lies within `[0.15, 0.85]`
against *both* references. **Selection rule: take the lowest passing `trials`** — cheapest to run at
benchmark scale, and it leaves headroom to raise the setting later if our checkpoints improve.

**Ordering is recorded descriptively and is NOT a gate.** An earlier draft of this card required
twixtbot to score better against `0379` than against `calib020_0001`. That was **circular**: it
would have licensed only an anchor that already agrees with our internal result, so the "external"
reference could never contradict us — which is the entire reason to want one. It is also
statistically unsupportable, since 16 games per reference cannot resolve an ~80 Elo difference and a
reversed ordering at this size is ordinary sampling noise.

So: report the two score rates and their direction, quote **no interval** from them, and treat a
reversed or flat ordering as **neither a pass nor a fail** — it is a question for the later
benchmark. **Discrimination is tested separately**, on *fresh* openings with an adequately sized
game count, in the benchmark card that a pass makes possible. If that benchmark later finds
twixtbot ranking our checkpoints against our internal ordering, that is a **result to investigate,
not a defect to design out.**

## What must be built, and nothing more

`eval_checkpoint_match.py` takes `--checkpoint-a/--checkpoint-b`, both ours; there is **no external
opponent seam**. The pilot needs one adapter: our state → twixtbot `Game`, twixtbot move → validated
in our state, plus the per-ply normalized state comparison above. Import twixtbot as a library —
**the engine is not modified**. No framework, no abstraction over "anchors", no new harness.

**One worker.** Concurrency is considered only after a one-process timing smoke, and never before we
know one process works — four workers would add TensorFlow and resource interactions to an unknown.

## Stopping rules

- Any gate fails → **stop**. No training, no second configuration hunt, no engine patch.
- No setting is unsaturated against both references → **twixtbot is rejected as an anchor**; fall
  back to the capacity-headroom probe on the existing replay corpus. A reversed or flat *ordering*
  is **not** a rejection ground — it is not a gate.
- Any per-ply state divergence between the two engines → **stop** until it is understood. Do not
  work around it, and do not exclude the affected games to continue.
- A setting passes → the *next* step is designing the shared-opponent benchmark. That is a separate
  card. **This pilot authorizes no training run.**
- Archives (Boardspace, Little Golem, T1j) are **out of scope** for this workstream entirely.

## What a pass would and would not mean

**Would:** one configuration exists in which an independent engine plays our exact rules, holds a
provably identical game state ply by ply, behaves deterministically, and is competitive-but-not-
saturated against both references — i.e. a usable opponent for a benchmark.

**Would not:** any absolute strength claim, any Elo for `calib020_0001` on an external scale, any
statement about `medium`, any conclusion about how the anchor *ranks* our checkpoints, or any
evidence that a training successor is achievable. Those need the benchmark this pilot only makes
possible.
