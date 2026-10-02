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

---

## 9. Amendment A1 — 2026-10-02, corrections before implementation (user-decided)

Review of `c019cae2` kept the design and the central fresh-agent limitation. It corrected
three technical points and clarified two points of interpretation. The sections above stand
except where A1 amends them.

**A1.1 — Capture-off access (amends §5).** A capture-off agent keeps neither the root nor the
counts, so the off run cannot be compared on them as §5 assumed. Both runs therefore install
a **study-local pass-through** on the agent's own MCTS instance, `agent.mcts.search_with_root`.
It:

* calls the original bound method with the **same arguments**;
* returns **the** tuple it returned (the same object);
* lets any exception propagate **unchanged** (it catches nothing);
* keeps a reference to the returned `(counts, root_value, root)`.

No frozen file and no class is edited. Synthetic tests check:

* object identity: the retained `counts` **is** the object `eval_readout.select` received;
* arguments forwarded verbatim;
* the same exception object re-raised;
* moves, both stream states and tree fingerprints (A1.3) identical with and without the
  pass-through, at the frozen 400-simulation configuration.

The capture-on run reads `last_capture`. The capture-off run reads the pass-through. In both,
the pass-through's retained objects must be **the** objects capture reports when capture is
on.

**A1.2 — Value perspective (amends §2.2 and §2.3).** At R2, *W* under *m* is a terminal node
where T1j has just won, so `_terminal_value` returns −1 to the player to move there (the
incumbent). The backup alternates sign: each such visit adds **+1 to the value sum of the
child node *m*** (T1j's perspective, T1j being to move at *m*) and **−1 at the root** (the
incumbent's perspective). §2.2's "a terminal −1 backed into *m*" is **wrong** and is replaced
by this.

Every reported value names its perspective:

* `q_node` — the node's stored mean, from the perspective of the player to move **at that
  node**;
* `q_root_mover` — the same quantity from the root mover's (the incumbent's) perspective. It
  is `−q_node` for a root child and `+q_node` for a grandchild, by depth parity.

A mean is reported only for a visited node; otherwise it is `null`, with a note.

**A1.3 — Reader preservation (amends §5.6).** Visits and stream states alone cannot catch a
reader that changes a prior or a value. A **fingerprint** of the final tree is therefore taken
before and after reading, and must be equal. It is a sha256 over a canonical walk of every
node, children in move-id order, covering:

* the move id, `visit_count` and `value_sum` (float hex);
* `nn_value`;
* every prior and every raw prior (float hex);
* whether `priors` is `priors_raw`;
* the child keys;

plus the `counts` table and the root value.

The fingerprint must also be equal **between** the capture-off and capture-on runs. The
synthetic negative controls a qualification run must catch are readers that:

* change a prior;
* change a value sum;
* change `nn_value`;
* change a visit count;
* draw from the search stream;
* draw from the readout stream.

**A1.4 — Interpretation (amends §1 and §6).**

* The fresh-agent and in-game decisions ran on different stream histories, so their trees
  **may** differ. How they actually differ is **unobserved**: nothing in this study observes an
  in-game tree.
* CD1 is a **within-game, same-threat contrast** with E1, at a different position four plies
  later. It is **not** a within-position control.

**A1.5 — Classifier controls (amends §5).** The reader's classification is itself tested on
synthetic trees whose answers are known:

* A (absent from the candidates);
* B (a candidate never visited), and B0 (prior exactly 0.0);
* C (visited, not returned);
* D (returned);
* R0, R1 and R2 under the returned move;
* a visit tie, recording the tie set and its (row, col) resolution.

Each of these, and the perspective labels of A1.2, has a deliberately wrong classifier that the
tests must catch.

**A1.6 — The pinned saving sets.** The episode sets come from the frozen set: E1 (16,9) and
(17,9); E2 (12,5); E3 (16,6) and (17,7). For the controls, the saving set is derived by the
same rule the frozen set uses (the mover's immediate wins, plus every move that leaves no
immediate threat) and **pinned**: CD1 (16,9) and (17,9); CD2 (12,4), (12,5) and (13,5). A
re-derivation that differs refuses.

**A1.7 — Implementation scope.**

* **Positions:** the five frozen positions only.
* **Observations:** final-tree only; no per-simulation observer.
* **Qualification:** synthetic, at the frozen 400-simulation configuration, with a stub
  evaluator.
* **Gate:** the 22nd gate, `h5_incumbent_defence_authorization.H5_INCUMBENT_DEFENCE_AUTHORIZED`,
  created **False**.
* **The eventual run** follows the established contract (cap-completion card §7–§10):
  * pure refusals first;
  * a parent claim with a single-use nonce;
  * the child in its own session under a hard parent watchdog;
  * the child's streams captured to files;
  * an fsynced journal;
  * a record on every exit (COMPLETE / STOP / INTERRUPTED / INCOMPLETE);
  * `watchdog.json` on every exit.

  Any failed preservation check is a STOP, naming the position and the check.
* **Not authorized:** a real model load, any decision, a per-simulation observer, an extra
  position, an agent or configuration change, gate opening, push. The ten-decision run needs
  separate approval.
