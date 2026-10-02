# H5 INCUMBENT DEFENCE MECHANISM STUDY — EXPLORATORY CARD (design only)

**Written 2026-10-02. DESIGN ONLY.** No instrumentation code, model load, query, agent change,
gate or push. Implementation and each run need their own authorization.

🔴 **EXPLORATORY. Diagnostic fixtures, not a strength evaluation.** The five positions here
are fixed diagnostic fixtures. Better behaviour on them would **not** establish stronger
play, and nothing here rescores any H5 game; H5's confirmatory result (`8ca7278`) stands.
Every causal statement in this card is a **hypothesis**, labelled `H-…`.

---

## 0. The question

The cap-completion diagnostic (its card, §3 and §8–§10; Stage 2 at `1c2f2760`) established
two things about the unchanged incumbent.

* **The misses reproduce.** At each of the three frozen episode positions (the incumbent
  positions of E1, E2, E3), it leaves a defendable immediate win open. It did so in play,
  and in 2/2 fresh re-queries it chose its recorded move again.
* **The controls work.** At CD1 and CD2 it blocks the threat, 2/2.

This card designs the observation that separates three ways a **saving move** *s* can fail
to be played:

| category | meaning |
|---|---|
| **A — excluded** | *s* is not in the legal or candidate set of the root |
| **B — retained, never visited** | *s* is a candidate, but the search never selected it |
| **C — visited, not selected** | the search visited *s*, and the readout chose another move |

It also adds one complementary observation (§2.2): whether the search ever saw the
**refutation** of the move it chose.

## 1. The frozen search path — as read from the code (2026-10-02, at `1c2f2760`)

| step | what the code does | where |
|---|---|---|
| construction | `h4_runner.production_incumbent_build(config)` → `twixtbot_g3_reference.build_reference_agent` → `SeededReferenceAgent`. Stage 2 built a **fresh agent per decision** (seed −1, by colour). In H5 play there was **one agent per game**, and its streams advanced across that game's earlier decisions. | `h4_runner.py:477`; `twixtbot_g3_reference.py:112,235`; `h5_cap_completion._incumbent_choose` |
| configuration | `frozen_argmax_config()` = `G3.eval_config()` with `selection_mode="argmax"`, then `eval_runner.cfg_from` → `MCTSConfig(n_simulations=400, eval_batch_size=14, stall_flush_sims=48, temp_high=0, temp_low=0)`. **Everything else is at its default:** `c_puct` 1.5; `fpu_value` 0.0; `fpu_policy_mass_reduction` None; edge-band and near-corner prior penalties 0 (off); closeout td=1 forcing off; closeout tie-break off; `pending_virtual_visits` 8. | `h4_runner.py:425`; `twixtbot_g3_reference.py:38`; `eval_runner.py:111`; `mcts.py:146–201` |
| search | `MCTS.search_with_root(state, add_noise=False)`:<br>• a **fresh root** each decision (no tree reuse);<br>• the root is expanded once, then **exactly 400 synchronous** `_run_single_simulation` calls.<br>The batched `search_from_root` path is **not** used, so `eval_batch_size`, `stall_flush_sims` and `pending_virtual_visits` are inert. The flush observer never fires, and the td=1 forcing hook (which lives only in `search_from_root`) is never reached. | `mcts.py:534–581, 807–871` |
| candidate set | Each expansion takes priors for **exactly** `state.legal_moves()`. The policy is a masked softmax renormalised over the legal moves. The (move, prior) pairs are shuffled **once per expansion** by `self.rng`. There is no pruning, top-k or prior penalty on this path: every legal move is a candidate at every expanded node. | `mcts.py:929–1037` (shuffle `:1021`); `local_evaluator.py` (masked softmax) |
| selection | PUCT, `q + c·P·√(N+1)/(1+n)`:<br>• `q = −Q(child)` for a visited child, `fpu_value` 0.0 for an unvisited one;<br>• scores within 1e-8 are ties, broken by `self.rng.choice`.<br>**Every** selection calls `rng.choice`, and that consumes the stream even with a single best move (checked 2026-10-02). | `mcts.py:1076–1156` |
| terminal / backup | A terminal node is never expanded; its value is ±1 from the mover's perspective. Backup alternates sign. After the search, `root.visit_count` = 400. | `mcts.py:1158–1183, 1356–1374` |
| readout | `eval_readout.select` in `MODE_ARGMAX` → `_argmax(counts)`: the most visits wins, ties broken by (row, col). The **readout stream is never consumed**. `counts` lists every legal move, with 0 for unvisited ones. | `eval_readout.py:119–182`; `mcts.py:565–570` |
| guard | `eval_integrity.validate_ply`: root visits == 400, a finite root value, and a finite Q for any visited top-2 child. | `eval_integrity.py:30`; `twixtbot_g3_reference.py:168` |
| random streams | **Search stream**: `Random(seed ^ 0xA5A5A5)` as red, `^ 0x5A5A5A` as black. It is consumed by the prior shuffles and the selection tie-breaks.<br>**Readout stream**: `^ 0xC3C3C3` / `^ 0x3C3C3C`. It is unused under argmax. | `twixtbot_g3_reference.py:120–152` |

**Consequence for interpretation.** A fresh-agent decision (seed −1) runs a different tree
from the in-game one: other shuffle orders, other tie-break draws. Stage 2 showed only that
the **move** is the same. Everything observed under this card describes the **fresh-agent
search**, not the search that ran in the game (§6).

## 2. What is observed, per position — operational definitions

The saving set at each episode position is the frozen one (`00_frozen_set.json`, sha256
`28c03f9f…`):

| position | saving moves |
|---|---|
| E1 | (16,9), (17,9) |
| E2 | (12,5) |
| E3 | (16,6), (17,7) |

At CD1 and CD2 the frozen threat is (17,9) and (13,5) respectively; the recorded blocks are
(16,9) and (12,5).

### 2.1 Each saving move *s*, read off the **final root** of the decision

| category | condition on the final root |
|---|---|
| **A** | `encode(s) ∉ root.priors` — *s* is not a legal candidate. Stage 1 made *s* legal in both engines, so A is expected to be empty; it is recorded anyway. |
| **B** | `encode(s) ∈ root.priors` and the child is absent or has `visit_count == 0`. Sub-flag **B0**: prior exactly 0.0 (softmax underflow). |
| **C** | `visit_count ≥ 1` and *s* ≠ the returned move |
| **D — selected** | *s* is the returned move. Expected only at CD1/CD2. |

Recorded with each category:

* `P(s)` and its rank among the root's legal moves;
* the visits and mean value of *s*, with the mean in the **mover's** perspective, defined only
  when visited;
* the same for the returned move (the visit leader);
* `root.nn_value`, the root value and `top2`.

### 2.2 The refutation, for the returned move *m* at E1–E3

Here *W* is the opponent's frozen immediate win. Read off the subtree of *m*:

| category | condition |
|---|---|
| **R0** | *m*'s child was never expanded |
| **R1** | *m* expanded, but *W* never selected under it |
| **R2** | *W* visited under *m* (a terminal −1 backed into *m*) |

Recorded with each: the prior of *W* at *m*'s node, *W*'s visits, and *m*'s visits and Q.

### 2.3 What each pattern would suggest — hypotheses only

| pattern | hypothesis (not a finding) |
|---|---|
| B, low `P(s)`, leader's Q > 0 | **H-PRIOR**: the policy gives *s* too little prior for PUCT to reach it in 400 simulations while an FPU of 0.0 and a well-valued leader hold the visits |
| C, *s*'s Q ≥ leader's Q, fewer visits | **H-READOUT**: visit-count selection and value disagree |
| R0/R1 | **H-REFUTATION-UNSEEN**: the search never learned that *m* allows *W*, so *m* kept its value |
| R2 and *m* still returned | **H-VALUE**: a backed-up loss was outweighed. Unexpected; it would need a closer look before anything else |

None of these establishes a cause. Each is a lead for a separately carded study.

## 3. What existing telemetry establishes — no new mechanism

* **The capture seam already exists.** `SeededReferenceAgent(capture=True)`, reachable through
  `build_reference_agent(..., capture=True)`, exposes, as `last_capture`, exactly the objects
  the decision used:
  * `counts`;
  * `root_value`;
  * the **final root node**, i.e. the whole tree: `priors`, `priors_raw`, `nn_value`, and
    `children` with `visit_count` and `value_sum`, recursively;
  * `top2` and `overrode_leader`.

  **Every quantity in §2 is derivable from that final tree.** (`twixtbot_g3_reference.py:123–180`)
* **Existing preservation evidence is real but out of scope.** `tests/test_d1_incumbent.py:72–127`
  shows four things:
  1. capture is off by default;
  2. the captured objects are **the** objects the readout used;
  3. moves and **both** stream states are equal with capture off and on;
  4. a negative control (a capture that draws) is caught.

  Its scope is a **stub evaluator, 8 simulations, a non-frozen batch configuration and a
  short opening**. It does **not** establish identity at the frozen 400-simulation
  configuration, with the real network, at these positions.
* `validate_ply` already enforces the 400-simulation budget on every decision.
* **Not available from play.** The H5 game records carry moves, actors and outcomes only.
  `h4_runner` and `h5_runner` never pass `capture`, so no search telemetry exists for any
  in-game decision.

## 4. What would need new instrumentation

| item | why | status under this card |
|---|---|---|
| a study builder | `production_incumbent_build` does not pass `capture`. A new, study-only function in a **new** module would call `G3.build_reference_agent(..., capture=True)` with **the** frozen config object, the bound identity and the Stage-2 seed. No frozen file is edited and the agent class is unchanged. | to implement, after authorization |
| a final-tree reader | It walks `last_capture["root"]` to produce §2's categories and statistics, writing JSON. It reads only, **after** the move is returned. | to implement, after authorization |
| a per-simulation trace (*optional*) | `MCTSObserver` and `MCTSSelectionObserver` exist (`mcts.py:269–302`), but `SeededReferenceAgent` builds `MCTS` without them. Attaching one means wiring onto `agent.mcts` after construction, and observer-**on** identity at the frozen configuration is unproven. It would show **when** *s* or *W* was first touched, which the final tree cannot. | **not** in the first study; only if the final-tree categories are ambiguous |
| probes off the decision path (*optional*) | Network value or policy at positions after an unvisited *s*, or after *m* then *W*. These are extra model calls outside the decision. | **not** proposed; separate authorization |

## 5. Preservation evidence — required before any observation counts

For each fixture, the same decision is run **twice from fresh agents with the same seed**:
once with capture off and once with capture on, with the reader applied afterwards. A
recorded observation counts only if every check below holds; otherwise it is a STOP.

1. **Configuration.**
   * The agent's config passes `_check_config` field by field, and is **the** object from
     `frozen_argmax_config()`.
   * `agent.mcts.config` equals `cfg_from(config)` on every field.
   * The incumbent identity equals the H5 manifest's bound identity, and the loader verifies
     the checkpoint's sha1.
2. **Randomness consumption.** After the decision, `agent.mcts.rng.getstate()` and
   `agent.readout_rng.getstate()` are equal between capture off and capture on.
3. **Search budget.** `root.visit_count == 400` in both runs (`validate_ply`), and the two
   `counts` dicts, covering every legal move, are equal.
4. **Tie-breaking.** Check 2 makes the tie-break draws identical. In addition, the readout's
   tie set (every move sharing the maximum visits) is recorded and equal across both runs.
5. **Move selection.**
   * The returned move is identical in both runs.
   * It equals the Stage-2 recorded move for that position: the incumbent's recorded move at
     E1–E3, and the recorded block at CD1/CD2.
   * A mismatch means the instrumented decision is not the one under study: STOP.
6. **Read-only reader.** Synthetic negative controls must be caught by checks 2–3 before any
   real run: a reader that draws one number, and one that changes a visit count.
   * The synthetic qualification runs at the **frozen** 400-simulation configuration with a
     stub evaluator. The existing 8-simulation test is not enough.

## 6. Fixtures and budget (proposed for a later, separate authorization)

**Positions:** the E1, E2 and E3 incumbent positions, plus CD1 and CD2, from the committed frozen
set. **Seed:** −1, the Stage-2 convention.

**Per position:** one capture-off and one capture-on decision, so **10 decisions** and **one
model load**. No T1j query, game, or engine change is involved.

**The CD1 contrast, read from the committed record of `h5s0-21-B`** (rules engine only,
2026-10-02):

| ply index | incumbent's move | facing (17,9)? | blocked? |
|---|---|---|---|
| `moves[:37]` (E1) | (1,2) | yes | no |
| `moves[:39]` | (1,3) | yes | no |
| `moves[:41]` (CD1) | (16,9) | yes | yes |

The same threat was missed twice and then blocked four plies later, in the same game. CD1 is
therefore close to a matched control for E1. The `moves[:39]` position is **not** in the frozen
set; including it would need a frozen-set amendment, which is listed here but not proposed.

**Not proposed, listed for completeness.** Each would need its own authorization and seed
decision:

* an RNG-robustness arm — further synthetic seeds, to see whether a category depends on the
  search stream;
* an in-game reconstruction — replaying each source game's earlier incumbent decisions under
  its scheduled seed to restore the exact in-game stream. That touches H5 research seeds, whose
  blocks are EXPOSED and RETIRED.

## 7. What it may conclude

**May conclude:**

* for each saving move at the five fixtures, under the fresh-agent seed −1 search, which of
  A/B/C/D it fell into, with its prior, rank, visits and value;
* whether the search saw the refutation of its chosen move (R0/R1/R2);
* that the observation was made with configuration, randomness, budget, tie-breaking and
  move all shown unchanged.

**May not conclude:**

* that the in-game searches looked the same (different streams);
* anything about other positions;
* any causal mechanism — §2.3 holds hypotheses only;
* anything about strength, or that a fix to these five positions improves play.

## 8. Evidence, gate, scope

* **Evidence.** A create-only directory
  `docs/superpowers/evidence/<date>-t1j-h5-incumbent-defence-EXPLORATORY/`. It would follow the
  cap-completion recording contract: claim first; a record on every exit; streams kept; a
  parent watchdog, if the run is a separate process.
* **Gate.** A new single-declaration gate (the 22nd), created closed with its inventory test.
  It would be opened for one run only, under its own authorization.
* **Next authorizable steps, in order:**
  1. implementation with synthetic controls (§5.6) and the gate closed;
  2. one run of §6's 10 decisions.
* **Not authorized by this card:** instrumentation code, model load, query, agent or
  configuration change, gate opening, push.
