# Search Distillation Stage B — Final Wiring Plan: corpus, selection, labels, Q, training, evaluation, closing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the accepted Stage B pieces into the pilot the card describes: the 200-game corpus (G), position selection (P), held-out labels (L1), gate Q, training labels (L2), both training arms with the fitted temperature and the held-out verdict (T), and seed closing (K).

**Architecture:**
- **Testable logic goes into the existing module** `scripts/GPU/alphazero/search_distillation_stage_b.py`, with tests appended to `tests/test_search_distillation_stage_b.py`. That logic is: a stored label becoming a training row; both arms on one schedule; the Q stop; and one game's labels.
- **One phase script and one wrapper** in the evidence directory, copied from Stage A's runner and wrapper. The script runs one phase per invocation.
- **One seed-survey copy** of Stage A's, for the 441-slot block.
- **Reused unchanged:**
  - search: `label`, `label_mcts`;
  - selection: `select_game`, `leakage_filter`;
  - labels and measures: `full_counts`, `q_game`, `gate_q`, `fit_beta_soft`, `game_measures`, `learning_verdict`;
  - closing: `closing_intervals`;
  - #53 stage 2: `batch_of`, `train_head` (with `chosen_ce` and `head_only` inside it), `forward_rows`, `save_candidate`, `surface_check`, `reload_check`;
  - games: `eval_runner`'s `build_agent_pairing_tasks` and `run_game_tasks`, and `eval_integrity.validate_result_set`.

**Tech Stack:** Python stdlib, numpy, MLX, pytest. MLX enters the tests through `make_padded_batch`, as arrays only: no network and no checkpoint.

**Spec:** `docs/superpowers/2026-10-12-search-distillation-stage-b-card.md` (committed `a9e16fc9`). It builds on checkpoint 1 (`fd435150`) and checkpoint 2A (`a9ac8437`).

## Global Constraints

- **Approved:** writing this plan only. Implementation and execution need review first.
- **Not approved:**
  - real-model work
  - seed accounting
  - games
  - labelling
  - training
- **Tests run no new search.** The file's simulation ceiling stays at checkpoint 2A's **2,800 per invocation**, all in the existing `runs` fixture. The new tests reuse its outputs or stub `label`.
- **Bounded runs.** One red run and one green run, each under `timeout 120`, with both exit statuses captured. An unexpected failure or a timeout stops for review, with no extra runs.
- **Test command:**
  `timeout 120 env PYTHONPATH=. .venv/bin/python -B -m pytest tests/test_search_distillation_stage_b.py -q -p no:cacheprovider`
- **The phase script and wrapper are not executed by this plan.** They are checked by an AST parse (`ast.parse`, which writes no `.pyc`) and `sh -n` only.
- **No seed-band literal** in any new or modified file. Afterwards, scan every one of them with the pinned scanner.
- **Card values, used unchanged:**
  - corpus: 200 games at 400 v 400; held out iff `g % 5 == 4`;
  - selection: 4 positions per game;
  - labels: 1,600 and 400 from the same seed L_g; T′ from R_h;
  - D: repeat the first position's 400 call;
  - training: 160 × 32 from `PCG64(T)`, the same schedule for both arms, lr 1e-3, clip 1.0;
  - caps: G 12 h, P 5 min, L1 85 min, L2 3.5 h, T 30 min.
- **Repository hygiene.** Keep the capture work untouched. Commit only when approved, and never push.

## Review Focus

- **The phase glue has no end-to-end test.** The script's reading of committed inputs, its replay of states and the alignment of rows with logits are checked only by review and by the assertions inside the script. The first real exercise is the run itself, and any fault there is a STOP with NO VERDICT. The tested pieces are the three the review named, one game's labelling, and two data-only input checks: label rows against the selection (`check_label_rows`) and phase termination (`require_terminated`).
- **Inputs are checked before they are consumed** (review correction). Q and T accept a label file only if it holds exactly the split's selected `(game, ply)` entries: none missing, extra, duplicated or from the other split. Each row's digest must equal both the selection's and the rebuilt state's. T checks this before any network loads or the training seed is used. A phase reads an earlier phase's output only if that phase's committed outer record shows exit 0 and no survivor. K reads a phase's events only if its committed outer record shows no survivor; there it accepts any exit status, because a stopped or timed-out phase still exposed its seeds.
- **Training input encoding.** Training rows use `state.to_tensor()` on the same `replay_state` states the searches saw. #53's `position_record` uses the same encoding, and #53 stage 2 trained this incumbent on it, so the network is the 30-channel one `LocalGPUEvaluator.build_input_tensor` also feeds `to_tensor()`.
- **Labels are written when a phase ends, not row by row.** A cap or a STOP loses that phase's labels. That outcome is NO VERDICT in any case, and nothing is retried.
- **Q and K get a 60-second operational cap.** The card prices no cap for them because neither loads a model or searches; the cap matches Stage A's analysis step.
- **The capture work's uncommitted `self_play.py` change is imported by the tests in the main checkout**, through `PositionRecord`. The diff leaves `PositionRecord` unchanged. Runs happen in a clean linked worktree, where the script refuses any loaded repository module that is uncommitted or differs from HEAD.

---

## Task 1: Module wiring and its tests

**Files:**
- Modify `scripts/GPU/alphazero/search_distillation_stage_b.py`: append the code below.
- Modify `tests/test_search_distillation_stage_b.py`: add `import types` and the `batch_of` import at the top, and append the tests below.

**Interfaces:**
- **Consumes:**
  - from checkpoint 1: `replay_state`, `state_digest`, `legal_digest`, `full_counts`, `is_heldout`, `repeat_slot`, `gate_q`, `LABEL`, `N_GAMES`, `PROTOCOL`;
  - from checkpoint 2A: `label`.
- **Produces:**
  - `Stop` and `Refused` (exceptions);
  - `SIMS_HI, SIMS_LO = 1600, 400`;
  - `TRAIN_STEPS, TRAIN_BATCH = 160, 32`;
  - `require_qualified(q)`;
  - `require_terminated(outer, need_exit_zero)`;
  - `check_label_rows(rows, selected, games, heldout)`;
  - `label_game(evaluator, g, moves, selected, label_seed, repeat_seed, log) -> rows`;
  - `label_phase(evaluator, games, selected, base, heldout, log, q=None) -> rows`;
  - `training_record(state, sparse, sims, legal_sha) -> PositionRecord`;
  - `train_schedule(seed, n_train) -> int array (160, 32)`;
  - `train_arms(q, load_network, student, control, seed, log, train) -> (schedule, {arm: (network, steps)})`.

- [ ] **Step 1: Append the tests.** Add `import types` beside the other stdlib imports. Add `from scripts.GPU.alphazero.external_learning_stage2 import batch_of` beside the other repository imports; importing it loads no MLX. Then append:

```python
# ─── wiring: a stored label into the training batch, one schedule for both arms, the Q stop, one game's labels ───

QUALIFIED = {"verdict": "QUALIFIED", "failed": []}
NOT_QUALIFIED = [B.gate_q([None] * 40), {}, {"verdict": "QUALIFIED", "failed": ["Q2"]}]


def _never(*a, **k):
    pytest.fail("called after a refusal")


def _pairs(n):
    """Fake aligned arms: what train_arms compares (ply, side, legal order, board) is equal; the targets differ."""
    pos = lambda i, c: types.SimpleNamespace(ply=i, to_move="red", legal_moves=[(i, 0), (i, 1)],
                                             board_tensor=np.full(2, i), visit_counts=c)
    return [pos(i, [3, 1]) for i in range(n)], [pos(i, [1, 3]) for i in range(n)]


def test_a_stored_label_reaches_the_training_batch_over_the_full_legal_support(runs):
    legal = [tuple(m) for m in STATE.legal_moves()]
    sha = B.legal_digest(legal)
    student = B.training_record(STATE, B.sparse_label(runs["v1600"], 1600), 1600, sha)   # as a label row stores it
    control = B.training_record(STATE, runs["label"], 400, sha)
    boards, mr, mc, mask, target, _ = (np.array(x) for x in batch_of([student, control]))
    n = len(legal)
    assert mask.shape == (2, n) and mask.sum() == 2 * n                  # every legal move, nothing truncated
    assert all(list(zip(mr[i].tolist(), mc[i].tolist())) == legal for i in (0, 1))
    assert np.array_equal(boards[0], boards[1])                          # one position, two targets
    for row, visits, sims in ((target[0], runs["v1600"], 1600), (target[1], runs["again"], 400)):
        assert np.array_equal(row, np.array([visits[m] for m in legal], dtype=np.float32) / np.float32(sims))
    assert (target[1] == 0).any()                                        # zero-visit moves stay in the support


def test_both_arms_draw_one_schedule_after_train_start():
    events, seen = [], []
    student, control = _pairs(5)

    def train(net, records, schedule, log):
        assert events == [{"event": "train_start"}]                     # the seed's use is logged first
        seen.append((net, records, schedule.copy()))
        return [{}] * B.TRAIN_STEPS

    schedule, nets = B.train_arms(QUALIFIED, object, student, control, 7, events.append, train)
    want = np.random.Generator(np.random.PCG64(7)).integers(0, 5, size=(160, 32))
    assert np.array_equal(schedule, want) and set(nets) == {"student", "control"}
    assert seen[0][1] is student and seen[1][1] is control and seen[0][0] is not seen[1][0]   # fresh copy per arm
    assert all(np.array_equal(s[2], want) for s in seen)


def test_the_arms_must_hold_the_same_positions_in_the_same_order():
    student, control = _pairs(3)
    for bad in (control[:2], control[::-1]):
        events = []
        with pytest.raises(ValueError):
            B.train_arms(QUALIFIED, _never, student, bad, 7, events.append, _never)
        assert events == []


@pytest.mark.parametrize("q", NOT_QUALIFIED)
def test_training_stops_before_any_load_or_event_when_q_did_not_qualify(q):
    events = []
    with pytest.raises(B.Refused):
        B.train_arms(q, _never, *_pairs(3), 7, events.append, _never)
    assert events == []


@pytest.mark.parametrize("q", NOT_QUALIFIED)
def test_training_labels_stop_before_any_event_or_search_when_q_did_not_qualify(q):
    events = []
    with pytest.raises(B.Refused):
        B.label_phase(None, {}, {}, BASE, False, events.append, q=q)
    assert events == []


def test_label_game_logs_each_seed_before_its_first_use_and_pairs_the_budgets(monkeypatch):
    events, calls = [], []

    def fake_label(evaluator, state, sims, seed):
        calls.append((len(events), state.ply, sims, seed))
        return [[0, 0, sims]]

    monkeypatch.setattr(B, "label", fake_label)
    sel = B.select_game(MOVES, 11)
    rows = B.label_game("ev", 9, MOVES, sel, 11, 22, events.append)    # game 9 is held out
    assert events == [{"event": "label_start", "game": 9}, {"event": "repeat_start", "game": 9}]
    first, *rest = [p["ply"] for p in sel]
    assert calls == [(1, first, 1600, 11), (1, first, 400, 11), (1, first, 400, 11), (2, first, 1600, 22)] + \
        [(2, p, s, seed) for p in rest for s, seed in ((1600, 11), (400, 11), (1600, 22))]
    assert [r["ply"] for r in rows] == [p["ply"] for p in sel]
    assert all(set(r) == {"game", "ply", "digest", "legal_sha", "t1600", "t400", "repeat"} for r in rows)


def test_label_game_stops_on_a_determinism_mismatch(monkeypatch):
    n = iter(range(1, 100))
    monkeypatch.setattr(B, "label", lambda ev, s, sims, seed: [[0, 0, next(n)]])
    with pytest.raises(B.Stop, match="determinism"):
        B.label_game("ev", 3, MOVES, B.select_game(MOVES, 11), 11, None, [].append)


def test_label_game_stops_before_any_event_when_a_position_is_not_the_selected_one(monkeypatch):
    monkeypatch.setattr(B, "label", _never)
    sel = [{**p, "digest": "0" * 64} for p in B.select_game(MOVES, 11)]
    events = []
    with pytest.raises(B.Stop, match="digest"):
        B.label_game("ev", 3, MOVES, sel, 11, None, events.append)
    assert events == []


# ─── data-only input checks: label rows against the selection; phase termination ───

ROW_GAMES = {3: MOVES, 4: MOVES, 8: MOVES}                              # 3 and 8 train; 4 is held out
ROW_SEL = {g: B.select_game(MOVES, 11 + g) for g in ROW_GAMES}
TRAIN_ROWS = [{"game": g, "ply": p["ply"], "digest": p["digest"]} for g in (3, 8) for p in ROW_SEL[g]]


def test_label_rows_that_are_exactly_the_split_selection_are_accepted():
    B.check_label_rows(TRAIN_ROWS, ROW_SEL, ROW_GAMES, heldout=False)


@pytest.mark.parametrize("rows, games", [
    (TRAIN_ROWS[:-1], ROW_GAMES),                                                         # one missing
    (TRAIN_ROWS + TRAIN_ROWS[:1], ROW_GAMES),                                             # a duplicate
    (TRAIN_ROWS + [{"game": 4, "ply": p["ply"], "digest": p["digest"]} for p in ROW_SEL[4][:1]], ROW_GAMES),  # held out
    ([{**TRAIN_ROWS[0], "digest": "0" * 64}] + TRAIN_ROWS[1:], ROW_GAMES),                # not the selection's digest
    (TRAIN_ROWS, {**ROW_GAMES, 3: MOVES[:1] + [(20, 20)] + MOVES[2:]}),                   # not the rebuilt state's
])
def test_label_rows_that_are_not_exactly_the_split_selection_stop(rows, games):
    with pytest.raises(B.Stop):
        B.check_label_rows(rows, ROW_SEL, games, heldout=False)


@pytest.mark.parametrize("outer", [{"outer_exit": 124, "survivors": ""}, {"outer_exit": 0, "survivors": "4242 "}, {}])
def test_a_phase_input_needs_exit_zero_and_no_survivor(outer):
    B.require_terminated({"outer_exit": 0, "survivors": ""}, True)
    with pytest.raises(B.Refused):
        B.require_terminated(outer, True)


def test_closing_accepts_a_stopped_or_timed_out_phase_but_not_a_survivor():
    for status in (0, 2, 124):
        B.require_terminated({"outer_exit": status, "survivors": ""}, False)
    for bad in ({"outer_exit": 124, "survivors": "4242 "}, {"survivors": ""}, None):
        with pytest.raises(B.Refused):
            B.require_terminated(bad, False)
```

- [ ] **Step 2: Red run.** Run the test command once and capture the exit status.

  Expected: exit status **1**.
  - The 50 existing tests pass.
  - The 22 new tests fail with `AttributeError` on `training_record`, `train_arms`, `Refused`, `label_phase`, `label_game`, `Stop`, `check_label_rows` or `require_terminated`.

  Any other outcome stops for review.

- [ ] **Step 3: Implement.** Append to the module:

```python
SIMS_HI, SIMS_LO = 1600, 400
TRAIN_STEPS, TRAIN_BATCH = 160, 32


class Stop(Exception):
    """Card §8: a STOP with NO VERDICT -- a determinism mismatch, a moved position, an integrity failure."""


class Refused(Exception):
    """Card §11: a phase its committed inputs do not allow."""


def require_qualified(q: Any) -> None:
    """Card §6/§11: no training label and no training unless the committed gate-Q record QUALIFIED."""
    if not isinstance(q, Mapping) or q.get("verdict") != "QUALIFIED" or q.get("failed") != []:
        raise Refused(f"gate Q did not qualify ({q.get('verdict') if isinstance(q, Mapping) else q!r}): "
                      "Stage B stops before the training labels")


def label_game(evaluator: Any, g: int, moves: Sequence[Sequence[int]], selected: Sequence[Mapping[str, Any]],
               label_seed: int, repeat_seed: Optional[int], log: Any) -> List[Dict[str, Any]]:
    """Card §5/§6: one game's labels, in ply order. Every position is rebuilt and checked against its selected
    digest FIRST; then `label_start` (and, before the first T', `repeat_start`) is logged before that seed's
    first use. Per position: T1600 and T400 from the label seed (common random numbers), and T' (1600) from the
    repeat seed when held out. Check D: the first position's 400 call is repeated and must match exactly."""
    states = []
    for p in selected:
        s = replay_state(moves[:p["ply"]])
        if state_digest(s) != p["digest"]:
            raise Stop(f"game {g} ply {p['ply']}: the rebuilt position's digest is not the selected one")
        states.append((p, s))
    if not states:
        return []
    log({"event": "label_start", "game": g})
    rows = []
    for i, (p, s) in enumerate(states):
        row = {"game": g, "ply": p["ply"], "digest": p["digest"], "legal_sha": legal_digest(s.legal_moves()),
               "t1600": label(evaluator, s, SIMS_HI, label_seed), "t400": label(evaluator, s, SIMS_LO, label_seed)}
        if i == 0 and label(evaluator, s, SIMS_LO, label_seed) != row["t400"]:
            raise Stop(f"game {g} ply {p['ply']}: determinism check D failed (a repeated 400 label differs)")
        if repeat_seed is not None:
            if i == 0:
                log({"event": "repeat_start", "game": g})
            row["repeat"] = label(evaluator, s, SIMS_HI, repeat_seed)
        rows.append(row)
    return rows


def label_phase(evaluator: Any, games: Any, selected: Any, base: int, heldout: bool, log: Any,
                q: Any = None) -> List[Dict[str, Any]]:
    """Card §11: L1 (heldout: T1600, T400, D and T') or L2 (training: T1600, T400 and D), in slot order. L2
    needs the committed gate-Q record QUALIFIED, checked before any event or search."""
    if not heldout:
        require_qualified(q)
    rows: List[Dict[str, Any]] = []
    for g in range(N_GAMES):
        if is_heldout(g) == heldout:
            rows += label_game(evaluator, g, games[g], selected[g], base + LABEL + g,
                               base + repeat_slot(g) if heldout else None, log)
    return rows


def training_record(state: TwixtState, sparse: Sequence[Sequence[int]], sims: int, legal_sha: str):
    """Card §5/§7: a stored label as a training row -- a PositionRecord over the FULL legal order holding the
    restored counts; `make_padded_batch` divides them by their sum (= sims). The board is built as #53's
    `position_record` builds it; the outcome is a required zero placeholder (no value loss)."""
    import numpy as np

    from .self_play import PositionRecord

    legal, counts = full_counts(state, sparse, sims, legal_sha)
    return PositionRecord(board_tensor=np.transpose(state.to_tensor(), (1, 2, 0)), to_move=state.to_move,
                          legal_moves=legal, visit_counts=counts, outcome=0.0,
                          active_size=PROTOCOL["board_size"], ply=state.ply)


def train_schedule(seed: int, n_train: int):
    """Card §7: TRAIN_STEPS draws of TRAIN_BATCH indices, uniform with replacement, from PCG64(T)."""
    import numpy as np

    return np.random.Generator(np.random.PCG64(seed)).integers(0, n_train, size=(TRAIN_STEPS, TRAIN_BATCH))


def _position(r: Any) -> Tuple[Any, ...]:
    return r.ply, r.to_move, tuple(r.legal_moves), r.board_tensor.tobytes()


def train_arms(q: Any, load_network: Any, student: Sequence[Any], control: Sequence[Any], seed: int, log: Any,
               train: Any) -> Tuple[Any, Dict[str, Tuple[Any, List[Dict[str, float]]]]]:
    """Card §7: the two arms, identical except for their targets -- each a fresh incumbent copy, the SAME
    positions in the same order, the SAME index schedule. Gate Q is checked before anything loads or is
    logged; `train_start` is logged before the schedule seed is used."""
    require_qualified(q)
    if len(student) != len(control) or any(_position(a) != _position(b) for a, b in zip(student, control)):
        raise ValueError("the arms do not hold the same positions in the same order")
    log({"event": "train_start"})
    schedule = train_schedule(seed, len(student))
    nets = {}
    for arm, records in (("student", student), ("control", control)):
        net = load_network()
        steps = train(net, records, schedule, log=lambda e, arm=arm: log({**e, "arm": arm}))
        if len(steps) != TRAIN_STEPS:
            raise Stop(f"{arm}: {len(steps)} of {TRAIN_STEPS} steps")
        nets[arm] = (net, steps)
    return schedule, nets


def require_terminated(outer: Any, need_exit_zero: bool) -> None:
    """Card §10/§11: a phase's committed outer record (the wrapper's) must show no survivor. An input to a later
    phase also needs exit 0; closing accepts any exit, since a stopped or timed-out phase still exposed seeds."""
    if not isinstance(outer, Mapping) or type(outer.get("outer_exit")) is not int or outer.get("survivors") != "":
        raise Refused(f"the phase's outer record shows no clean termination: {outer!r}")
    if need_exit_zero and outer["outer_exit"] != 0:
        raise Refused(f"the phase exited {outer['outer_exit']}, not 0: its outputs are not inputs")


def check_label_rows(rows: Sequence[Mapping[str, Any]], selected: Mapping[int, Sequence[Mapping[str, Any]]],
                     games: Any, heldout: bool) -> None:
    """Card §4/§5: label rows must be exactly the split's selected positions -- none missing, extra, duplicated
    or from the other split -- and each row's digest must be the selection's AND the rebuilt state's."""
    keys = [(r["game"], r["ply"]) for r in rows]
    want = {(g, p["ply"]): p["digest"] for g in selected if is_heldout(g) == heldout for p in selected[g]}
    if len(set(keys)) != len(keys):
        raise Stop("duplicate label rows")
    if set(keys) != set(want):
        raise Stop(f"the label rows are not exactly the selected {'held-out' if heldout else 'training'} positions: "
                   f"{len(set(keys) - set(want))} extra, {len(set(want) - set(keys))} missing")
    for r in rows:
        g, p = r["game"], r["ply"]
        if r["digest"] != want[(g, p)] or state_digest(replay_state(games[g][:p])) != r["digest"]:
            raise Stop(f"game {g} ply {p}: the row's digest is not the selected, rebuilt position's")
```

- [ ] **Step 4: Green run.** Run the test command once and capture the exit status.

  Expected: exit status **0**, **72 passed** (50 + 22), and the 2A fixture's one `DESCRIPTIVE` line. An unexpected failure or a timeout stops for review, with no extra runs.

## Task 2: The phase script and its wrapper

**Files:** Create two files in `docs/superpowers/evidence/search-distillation-stage-b/`:
- `01_stage_b_run.py`, a copy of Stage A's `01_stage_a_run.py`. Its diff is limited to:
  - one phase per invocation;
  - the phase bodies;
  - the 441-slot plan;
  - reading committed earlier-phase outputs;
  - the STOP class.
- `02_stage_b_wrapper.sh`, a copy of Stage A's wrapper. Its diff is limited to:
  - the phase argument;
  - the cap table;
  - the phase directory;
  - the candidates directory;
  - the header.

- [ ] **Step 1: Write `01_stage_b_run.py`:**

```python
"""SEARCH DISTILLATION STAGE B -- the pilot's phases, ONE per invocation: G corpus, P selection, L1 held-out
labels, Q the gate, L2 training labels, T training + temperature + held-out verdict, K seed closing.
Card: docs/superpowers/2026-10-12-search-distillation-stage-b-card.md. A copy of Stage A's run script
(evidence/search-distillation-stage-a/01_stage_a_run.py); its diff is limited to: the phase argument and bodies,
the 441-slot plan, reading only COMMITTED outputs of earlier phases that COMPLETED, and the STOP class.

Each phase writes create-only into <PHASE>/ beside this file. Run ONLY through 02_stage_b_wrapper.sh <PHASE>
(the phase's cap and the outer status live there). The testable logic lives in search_distillation_stage_b.
Exit 0 COMPLETED / 1 a repeat run / 2 STOP (NO VERDICT) / 3 STOP_ERROR / 4 REFUSED.

    python -B <this file> --phase G|P|L1|Q|L2|T|K --checkpoint <absolute path> --candidates <absolute dir>
"""
import dataclasses
import hashlib
import json
import pathlib
import platform
import statistics
import subprocess
import sys
import time

from scripts.GPU.alphazero import search_distillation_stage_b as B

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REL = "docs/superpowers/evidence/search-distillation-stage-b"
CARD = "docs/superpowers/2026-10-12-search-distillation-stage-b-card.md"
SHA1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"          # calib020_0001
EXIT = {"COMPLETED": 0, "STOP": 2, "STOP_ERROR": 3, "REFUSED": 4}
PHASES = {"G": (), "P": ("G",), "L1": ("G", "P"), "Q": ("G", "P", "L1"), "L2": ("G", "P", "Q"),
          "T": ("G", "P", "L1", "Q", "L2"), "K": ()}      # each phase's committed, COMPLETED inputs
MODEL = ("G", "L1", "L2", "T")
SIMS = {"t1600": B.SIMS_HI, "t400": B.SIMS_LO, "repeat": B.SIMS_HI}
PHASE = sys.argv[sys.argv.index("--phase") + 1] if "--phase" in sys.argv else None
if PHASE not in PHASES:
    sys.exit(EXIT["REFUSED"])
OUT = HERE / PHASE
T0 = time.monotonic()
log_fh = (OUT / "progress.jsonl").open("x")               # a second run stops here, exit 1


def log(e):
    log_fh.write(json.dumps({**e, "t": round(time.monotonic() - T0, 3)}) + "\n")
    log_fh.flush()


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


class Refused(Exception):
    pass


def write(name, text):
    with (OUT / name).open("x") as fh:
        fh.write(text)


def committed(rel):
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True)
    if tracked.returncode or git("status", "--porcelain", "--", rel):
        raise Refused(f"{rel} is not committed and unmodified")
    return (ROOT / rel).read_text()


def phase_input(phase, name):
    """An earlier phase's output, read only if that phase terminated cleanly with exit 0, COMPLETED, and is committed."""
    d = f"{REL}/{phase}"
    B.require_terminated(json.loads(committed(f"{d}/outer.json")), True)      # exit 0, no survivor
    if json.loads(committed(f"{d}/record.json")).get("verdict") != "COMPLETED":
        raise Refused(f"phase {phase} did not complete")
    return committed(f"{d}/{name}")


def jsonl(text):
    return [json.loads(line) for line in text.splitlines()]


def seed_base():
    from scripts.GPU.alphazero import e4_screen_reference as REG

    plan = json.loads(committed(f"{REL}/plan.json"))
    if plan.get("n_slots") != B.N_SLOTS or type(plan.get("base")) is not int:
        raise Refused(f"the plan is not exactly {B.N_SLOTS} slots from an int base")
    for slot in range(B.N_SLOTS):
        s = plan["base"] + slot
        if not REG.seed_is_accounted(s) or REG.seed_is_unavailable(s):
            raise Refused(f"seed slot {slot} is not ACCOUNTED and available")
    return plan["base"]


def modules_check(rec):
    """Repository modules loaded so far (this script included) must be committed, unmodified, from THIS tree."""
    ours = {n: m.__file__ for n, m in list(sys.modules.items())
            if (n == "__main__" or n.split(".")[0] in ("scripts", "tests")) and getattr(m, "__file__", None)}
    outside = sorted(n for n, f in ours.items() if not pathlib.Path(f).resolve().is_relative_to(ROOT))
    if outside:
        raise Refused(f"repository modules loaded from outside this tree: {outside}")
    rel = sorted(str(pathlib.Path(f).resolve().relative_to(ROOT)) for f in ours.values())
    rec["modules"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in rel}
    if git("status", "--porcelain", "--untracked-files=all", "--", *rel):
        raise Refused("a loaded repository module is uncommitted or differs from HEAD")


def load_games():
    games = json.loads(phase_input("G", "games.json"))
    if [x["game"] for x in games] != list(range(B.N_GAMES)):
        raise B.Stop("games.json is not the 200 scheduled games in slot order")
    return [x["moves"] for x in games]


def evaluator(rec, ckpt):
    from scripts.GPU.alphazero import eval_runner as ER

    t = time.monotonic()
    ev = ER._default_evaluator_factory(str(ckpt))
    rec["load_seconds"] = round(time.monotonic() - t, 3)
    log({"event": "loaded", "seconds": rec["load_seconds"]})
    return ev


def full_row(games, row, keys):
    """A stored label row (already checked by check_label_rows) with its full legal support restored."""
    s = B.replay_state(games[row["game"]][:row["ply"]])
    return {k: B.full_counts(s, row[k], SIMS[k], row["legal_sha"])[1] for k in keys}, s


# ─── the phases ───

def phase_G(rec, base, ckpt):
    """Card §3: 200 games, the incumbent at 400 v itself at 400, the frozen protocol; outcome-blind progress."""
    from scripts.GPU.alphazero import eval_runner as ER
    from scripts.GPU.alphazero.eval_integrity import validate_result_set

    ev = evaluator(rec, ckpt)

    def factory(path):
        assert path == str(ckpt), path
        return ev

    cfg = ER.EvalConfig(**B.PROTOCOL)
    rd = ER.readout_from_eval_config(cfg)
    a, b = (ER.AgentSpec(i, str(ckpt), rd, mcts_sims=B.SIMS_LO) for i in ("a400", "b400"))
    rec["protocol"] = {"PROTOCOL": dict(B.PROTOCOL), "eval_config": dataclasses.asdict(cfg),
                       "agents": [dataclasses.asdict(a), dataclasses.asdict(b)]}
    tasks = ER.build_agent_pairing_tasks("stage-b", a, b, B.N_GAMES, base_seed=base + B.GAME)
    rec["games"], results, games = [], [], []
    for task in tasks:
        slot = task.seed - base
        log({"event": "game_start", "game": task.game_idx, "slot": slot})
        t = time.monotonic()
        [r] = ER.run_game_tasks([task], 1, cfg, evaluator_factory=factory, replay_dir=str(OUT / "replays"))
        secs = round(time.monotonic() - t, 3)
        results.append(r)
        plies = json.loads(pathlib.Path(r.replay_path).read_text())["moves"]
        if [m["ply"] for m in plies] != list(range(r.n_moves)):
            raise B.Stop(f"game {task.game_idx}: the replay is not the game's {r.n_moves} plies in order")
        moves = [[m["row"], m["col"]] for m in plies]
        games.append({"game": task.game_idx, "slot": slot, "moves": moves, "transcript_sha256": B._sha(moves)})
        rec["games"].append({**dataclasses.asdict(r), "slot": slot, "seconds": secs})
        log({"event": "game_end", "game": task.game_idx, "slot": slot, "seconds": secs,      # outcome-blind
             "n_moves": r.n_moves})
    validate_result_set(results, tasks, "a400", "b400")
    rec["result_set_check"] = "validate_result_set passed"
    write("games.json", json.dumps(games, separators=(",", ":")) + "\n")      # move lists only: no outcome


def phase_P(rec, base):
    """Card §4: select_start logged BEFORE each game's draw; then the leakage rule over all 200 games."""
    games = load_games()
    selected = []
    for g in range(B.N_GAMES):
        log({"event": "select_start", "game": g})
        selected.append(B.select_game(games[g], base + B.LABEL + g))
    kept, dropped = B.leakage_filter(dict(enumerate(selected)))
    out = {"selected": selected, "kept": [kept[g] for g in sorted(kept)], "dropped": dropped}
    rec["positions"] = {"selected": sum(map(len, selected)), "heldout_kept": sum(map(len, out["kept"])),
                        "dropped": dropped, "empty_heldout_games": sum(1 for k in out["kept"] if not k)}
    write("selection.json", json.dumps(out, indent=1) + "\n")


def phase_labels(rec, base, ckpt, heldout):
    """Card §5/§11: L1 (held out) or L2 (training; refused unless gate Q QUALIFIED, before the model loads)."""
    q = None if heldout else json.loads(phase_input("Q", "q.json"))
    if not heldout:
        B.require_qualified(q)
    games, sel = load_games(), json.loads(phase_input("P", "selection.json"))["selected"]
    rows = B.label_phase(evaluator(rec, ckpt), games, sel, base, heldout, log, q=q)
    rec["labels"] = len(rows)
    write("labels.jsonl", "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))


def phase_Q(rec):
    """Card §6, once, on the committed L1 labels: Q1, Q2, and the descriptive medians and repeat agreement."""
    games, sel = load_games(), json.loads(phase_input("P", "selection.json"))
    rows = jsonl(phase_input("L1", "labels.jsonl"))
    B.check_label_rows(rows, dict(enumerate(sel["selected"])), games, heldout=True)
    by = {(r["game"], r["ply"]): r for r in rows}
    per_game, d, r_, agree = [], [], [], []
    for h, kept in enumerate(sel["kept"]):
        g = 5 * h + 4
        pos = [full_row(games, by[(g, p["ply"])], ("t1600", "t400", "repeat"))[0] for p in kept]
        for x in pos:
            d.append(B.tv(B.probs(x["t1600"]), B.probs(x["t400"])))
            r_.append(B.tv(B.probs(x["t1600"]), B.probs(x["repeat"])))
            agree.append(bool(B.argmax_set(x["t1600"]) & B.argmax_set(x["repeat"])))
        per_game.append(B.q_game(pos))
    q = {**B.gate_q(per_game), "per_game": per_game, "positions": len(d),
         "median_d": statistics.median(d) if d else None, "median_r": statistics.median(r_) if r_ else None,
         "repeat_agreement": sum(agree) / len(agree) if agree else None}
    rec["q_verdict"] = q["verdict"]
    write("q.json", json.dumps(q, indent=1) + "\n")


def phase_T(rec, base, ckpt, cand_dir):
    """Card §7/§8: both arms (gate Q checked first), surface checks, the soft temperature fit on the TRAINING
    positions' T1600 targets, then the held-out measures and the verdict, once."""
    from scripts.GPU.alphazero import external_learning_stage2 as EL2
    from scripts.GPU.alphazero.probe_eval import load_network_for_scoring

    q = json.loads(phase_input("Q", "q.json"))
    B.require_qualified(q)
    if cand_dir is None or not cand_dir.is_absolute():
        raise Refused("--candidates must name a directory by absolute path")
    games, sel = load_games(), json.loads(phase_input("P", "selection.json"))
    train_rows, held_rows = jsonl(phase_input("L2", "labels.jsonl")), jsonl(phase_input("L1", "labels.jsonl"))
    B.check_label_rows(train_rows, dict(enumerate(sel["selected"])), games, heldout=False)  # before any network
    B.check_label_rows(held_rows, dict(enumerate(sel["selected"])), games, heldout=True)    # or the training seed
    held = {(r["game"], r["ply"]): r for r in held_rows}
    student, control = [], []
    for r in train_rows:
        s = B.replay_state(games[r["game"]][:r["ply"]])
        student.append(B.training_record(s, r["t1600"], B.SIMS_HI, r["legal_sha"]))
        control.append(B.training_record(s, r["t400"], B.SIMS_LO, r["legal_sha"]))

    def load_network():
        if hashlib.sha1(ckpt.read_bytes()).hexdigest() != SHA1:
            raise B.Stop("the incumbent file is no longer the pinned checkpoint")
        net, *_ = load_network_for_scoring(str(ckpt), verbose=False)
        net.eval()
        return net

    schedule, nets = B.train_arms(q, load_network, student, control, base + B.TRAIN, log, EL2.train_head)
    rec["training"] = {"n_train": len(student), "schedule_sha256": B._sha(schedule.tolist()),
                       **{arm: {"first_loss": st[0]["loss"], "last_loss": st[-1]["loss"],
                                "max_grad_norm": max(x["grad_norm"] for x in st)} for arm, (_, st) in nets.items()}}
    cand_dir.mkdir(parents=True, exist_ok=True)
    arms, rec["candidates"] = {}, {}
    for arm, (net, _) in nets.items():
        path = cand_dir / f"stage-b-{arm}.safetensors"
        rec["candidates"][arm] = {"path": str(path), "sha256": EL2.save_candidate(net, path),
                                  "surface": EL2.surface_check(ckpt, path)}
        arms[arm] = EL2.reload_check(net, path)
    arms["incumbent"] = load_network()
    try:
        fit = B.fit_beta_soft(EL2.forward_rows(arms["incumbent"], student)[0],
                              [B.probs(r.visit_counts) for r in student])
    except ValueError as e:
        raise B.Stop(f"temperature fit: {e}")
    per_game = []
    for h, kept in enumerate(sel["kept"]):
        g = 5 * h + 4
        recs = []
        for p in kept:
            r = held[(g, p["ply"])]
            recs.append(B.training_record(B.replay_state(games[g][:p["ply"]]), r["t1600"], B.SIMS_HI, r["legal_sha"]))
        if not recs:
            per_game.append(None)                                        # an empty game (card §8)
            continue
        z = {arm: EL2.forward_rows(net, recs)[0] for arm, net in arms.items()}
        z["temperature"] = [[fit["beta"] * x for x in row] for row in z["incumbent"]]   # legal moves only: re-masked
        per_game.append(B.game_measures([{"target": x.visit_counts, "logits": {a: z[a][i] for a in B.ARMS}}
                                         for i, x in enumerate(recs)]))
    verdict = B.learning_verdict(per_game)
    present = [m["kl"] for m in per_game if m is not None]
    mean_kl = {a: sum(k[a] for k in present) / len(present) if present else None for a in B.ARMS}
    gain = (mean_kl["incumbent"] - mean_kl["student"]) / mean_kl["incumbent"] if present and mean_kl["incumbent"] else None
    rec["verdict_stage_b"] = verdict["verdict"]
    write("verdict.json", json.dumps({"verdict": verdict["verdict"], "learning": verdict, "q": q["verdict"],
                                      "beta": fit, "mean_kl": mean_kl, "relative_kl_gain_vs_incumbent": gain,
                                      "per_game": per_game, "strength_verdict": None}, indent=1) + "\n")


def phase_K(rec, base):
    """Card §9: EXPOSED = the union of slots any phase that ran logged as used (each phase's committed outer
    record must show no survivor; a stopped or timed-out phase is accepted); RETIRED = the whole block."""
    events, ran = [], []
    for ph in ("G", "P", "L1", "Q", "L2", "T"):
        if (HERE / ph / "progress.jsonl").exists():
            B.require_terminated(json.loads(committed(f"{REL}/{ph}/outer.json")), False)   # any exit, no survivor
            events += jsonl(committed(f"{REL}/{ph}/progress.jsonl"))
            ran.append(ph)
    exposed, retired = B.closing_intervals(base, events)
    rec["closing"] = {"phases": ran, "exposed": exposed, "retired": retired}
    write("closing.json", json.dumps(rec["closing"], indent=1) + "\n")


def main(rec, argv):
    arg = lambda k: pathlib.Path(argv[argv.index(k) + 1]) if k in argv else None
    for need in PHASES[PHASE]:
        phase_input(need, "record.json")
    base = None if PHASE == "Q" else seed_base()
    ckpt = None
    if PHASE in MODEL:
        ckpt = arg("--checkpoint")
        if ckpt is None or not ckpt.is_absolute() or not ckpt.is_file():
            raise Refused("--checkpoint must name an existing file by absolute path")
        sha1 = hashlib.sha1(ckpt.read_bytes()).hexdigest()
        if sha1 != SHA1:
            raise Refused(f"checkpoint SHA-1 {sha1} is not the pinned calib020_0001")
        rec["checkpoint"] = {"path": str(ckpt), "sha1_verified": sha1}
        import mlx.core
        import numpy

        rec["environment"] = {"python": sys.version, "platform": platform.platform(), "mlx": mlx.core.__version__,
                              "numpy": numpy.__version__, "executable": sys.executable}
        # every repository module a model phase runs, loaded BEFORE the check
        from scripts.GPU.alphazero import (eval_integrity, eval_runner, external_learning_stage2,  # noqa: F401
                                           local_evaluator, probe_eval, self_play, trainer)
    modules_check(rec)
    if PHASE == "G":
        phase_G(rec, base, ckpt)
    elif PHASE == "P":
        phase_P(rec, base)
    elif PHASE in ("L1", "L2"):
        phase_labels(rec, base, ckpt, PHASE == "L1")
    elif PHASE == "Q":
        phase_Q(rec)
    elif PHASE == "T":
        phase_T(rec, base, ckpt, arg("--candidates"))
    else:
        phase_K(rec, base)
    modules_check(rec)                                    # again: anything loaded lazily
    return "COMPLETED"


def _is_stop(e):
    stops = [B.Stop] + [getattr(sys.modules.get(m), c) for m, c in
                        (("scripts.GPU.alphazero.external_learning", "Stage1Stop"),
                         ("scripts.GPU.alphazero.eval_integrity", "IntegrityError")) if m in sys.modules]
    return isinstance(e, tuple(stops))


if __name__ == "__main__":
    rec = {"kind": "search-distillation-stage-b-record/1", "phase": PHASE, "card": CARD,
           "head": git("rev-parse", "HEAD"), "strength_verdict": None}
    log({"event": "start", "phase": PHASE, "head": rec["head"]})
    try:
        verdict = main(rec, sys.argv)
    except (Refused, B.Refused) as e:
        verdict, rec["reason"] = "REFUSED", str(e)
    except Exception as e:
        verdict, rec["reason"] = ("STOP" if _is_stop(e) else "STOP_ERROR"), f"{type(e).__name__}: {e}"
    rec["verdict"], rec["total_seconds"] = verdict, round(time.monotonic() - T0, 3)
    log({"event": "end", "verdict": verdict})
    write("record.json", json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    sys.exit(EXIT[verdict])
```

- [ ] **Step 2: Write `02_stage_b_wrapper.sh`:**

```sh
#!/bin/sh
# STAGE B OUTER WRAPPER -- ONE phase under its card cap: TERM at the cap less 10 s, KILL 10 s later (GNU timeout
# signals the whole process group, caffeinate included). Card: docs/superpowers/2026-10-12-search-distillation-
# stage-b-card.md §10. A copy of evidence/search-distillation-stage-a/02_stage_a_wrapper.sh; the diff: the phase
# argument, the cap table, the phase directory, the candidates directory, and this header.
#
# THE DETACHED LAUNCH (G, L1, L2; run once each, from the worktree root, in a POSIX shell). mkdir refuses an
# existing phase directory (set -e), so a second launch fails before anything starts:
#
#   sh -c 'set -euC; P=<PHASE>; D=docs/superpowers/evidence/search-distillation-stage-b/$P; mkdir "$D";
#          exec 3>"$D/launch.out.txt";
#          exec 4>"$D/launch.pid.txt";
#          nohup sh docs/superpowers/evidence/search-distillation-stage-b/02_stage_b_wrapper.sh "$P" >&3 2>&1 \
#            < /dev/null 3>&- 4>&- &
#          printf "%s\n" "$!" >&4'
#
# P, Q, T and K run in the foreground: mkdir "$D" && sh <this file> <PHASE>; echo "EXIT=$?".
# The environment, the checkpoint and the candidates directory come from the main checkout by absolute path.
# Writes, create-only: <PHASE>/run.stdout.txt, run.stderr.txt, outer.json. Exits with the outer status.
set -u
PHASE=${1:-}
case "$PHASE" in
  G) CAP=43190 ;; P) CAP=290 ;; L1) CAP=5090 ;; L2) CAP=12590 ;; T) CAP=1790 ;;   # card caps less the KILL grace
  Q|K) CAP=50 ;;                                    # operational: no model, no search
  *) echo "REFUSED: the phase must be one of G P L1 Q L2 T K" >&2; exit 4 ;;
esac
OUT_REL=docs/superpowers/evidence/search-distillation-stage-b
D="$OUT_REL/$PHASE"
W=$(cd "$(dirname "$0")/../../../.." && pwd -P) || exit 4
cd "$W" || exit 4
MAIN=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd -P) || exit 4
PY="$MAIN/.venv/bin/python"
CKPT="$MAIN/checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"
CAND="$MAIN/runs/search-distillation-stage-b"
[ -d "$D" ] || { echo "REFUSED: $D must be created by the launch" >&2; exit 4; }
for f in progress.jsonl record.json run.stdout.txt run.stderr.txt outer.json; do
  [ -e "$D/$f" ] && { echo "REFUSED: $D/$f exists (one-shot)" >&2; exit 1; }
done
set -C                                   # create-only redirections from here on
start=$(date +%s)
timeout -k 10 "$CAP" caffeinate -i env PYTHONPATH=. "$PY" -B "$OUT_REL/01_stage_b_run.py" --phase "$PHASE" \
  --checkpoint "$CKPT" --candidates "$CAND" > "$D/run.stdout.txt" 2> "$D/run.stderr.txt"
status=$?
end=$(date +%s)
survivors=$(pgrep -f "$OUT_REL/01_stage_b_run.py --phase $PHASE " | tr '\n' ' ')
printf '{"outer_exit": %d, "wall_seconds": %d, "cap_seconds": %d, "survivors": "%s", "tree": "%s", "python": "%s", "checkpoint": "%s"}\n' \
  "$status" "$((end - start))" "$CAP" "$survivors" "$W" "$PY" "$CKPT" > "$D/outer.json"
exit "$status"
```

- [ ] **Step 3: Static checks only.** Do not execute either file.
  - `.venv/bin/python -B -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" docs/superpowers/evidence/search-distillation-stage-b/01_stage_b_run.py` must exit 0.
  - `sh -n docs/superpowers/evidence/search-distillation-stage-b/02_stage_b_wrapper.sh` must exit 0.

## Task 3: The seed-survey copy (written, not run)

**Files:** Create `docs/superpowers/evidence/search-distillation-stage-b-seed-survey/01_seed_survey.py` as a byte copy of `docs/superpowers/evidence/search-distillation-stage-a-seed-survey/01_seed_survey.py`, then make exactly these edits. The pinned method files, the rule, the scope, the controls and the three modes are unchanged.

- [ ] **Step 1: Make the edits.**
  1. **Docstring.**
     - Title: "STAGE B -- THE 441-SLOT SEED SURVEY AND ITS COLLISION PROOFS".
     - Card path: the Stage B card, §9.
     - Copy note: "a COPY of the Stage A survey; its diff is limited to: the block size (441), the paths, the slot lines for the card's four roles, the plan's kind, the titles, and the most recent EXPOSED block's label".
     - The rule line: `candidate(reserved, 441)`.
     - Slot roles: G_g slots 0–199, L_g 200–399, R_h 400–439, T 440.
  2. **Paths.**
     - `HERE = "docs/superpowers/evidence/search-distillation-stage-b-seed-survey"`
     - `PLAN = "docs/superpowers/evidence/search-distillation-stage-b/plan.json"`
  3. **Slot count.** Replace `N_SLOTS = 160` with `from scripts.GPU.alphazero.search_distillation_stage_b import GAME, LABEL, REPEAT, TRAIN, N_SLOTS  # 0, 200, 400, 440, 441`, placed after the stdlib imports.
  4. **Module names.** In `load_method`, `f"_search_distillation_stage_a_{key}"` becomes `f"_search_distillation_stage_b_{key}"`.
  5. **Plan kind.** In `build_plan`, the kind becomes `"search-distillation-stage-b-plan/1"`.
  6. **`slot_lines`.** It becomes:
     ```python
     def slot_lines(block):
         """Card §9: G_g corpus game g, L_g the labels of game g, R_h the repeats of held-out game 5h + 4, T the schedule."""
         role = lambda s: (f"G game {s - GAME}" if s < LABEL else f"L game {s - LABEL}" if s < REPEAT else
                           f"R game {5 * (s - REPEAT) + 4}" if s < TRAIN else "T schedule")
         return [f"    SLOT {s} {role(s)} seed {block[0] + s}" for s in range(N_SLOTS)]
     ```
  7. **Messages.**
     - The two "the Stage A plan …" fault messages say "Stage B".
     - The title line says `SEARCH DISTILLATION STAGE B -- SEED …`.
     - `VERDICT_OK = "THE STAGE-B BLOCK IS CLEAN"`.
     - `say("SLOTS (card §9: G 0-199, L 200-399, R 400-439, T 440; slot s has seed BASE + s):")`.
     - The negative control `"the most recently EXPOSED block (stage 3's games)"` is relabelled `"(Stage A's games)"`, because Stage A's block is now `EXPOSED[-1]`.

- [ ] **Step 2: Static check.** The `ast.parse` check from Task 2, Step 3, on this file. Then `diff` it against the Stage A copy. Every hunk must be one of the seven edits above.

## Task 4: Scan, report, STOP

- [ ] **Step 1: Scan** every new or modified file with the pinned scanner. Each must print `{}`:

```bash
.venv/bin/python -B - <<'EOF'
import importlib.util
spec = importlib.util.spec_from_file_location("S", "docs/superpowers/evidence/tactical-candidate-seed-survey/01_seed_survey.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
for f in ("scripts/GPU/alphazero/search_distillation_stage_b.py", "tests/test_search_distillation_stage_b.py",
          "docs/superpowers/evidence/search-distillation-stage-b/01_stage_b_run.py",
          "docs/superpowers/evidence/search-distillation-stage-b/02_stage_b_wrapper.sh",
          "docs/superpowers/evidence/search-distillation-stage-b-seed-survey/01_seed_survey.py"):
    print(f, S.scan(open(f).read()))
EOF
```

- [ ] **Step 2: Report and STOP.** Report both test exit statuses and counts, the static checks, the survey diff hunks and the scans. Leave everything **uncommitted** unless separately approved.

---

## Execution sequence (each step separately approved; none is approved by this plan)

All steps run in a linked worktree on a fresh branch from the commit that holds this build, as Stage A did. The Python and the checkpoint come from the main checkout by absolute path. Any non-zero exit, failed control or timeout stops for review. Nothing is retried, extended or replaced.

| Step | What | Bound |
|---|---|---|
| S1 | Survey (`--mode survey`): exit 0, CLEAN, every control passed. Commit its record only. | `timeout 1200` |
| S2 | Pre-proof (`--mode pre`): exit 0, every control passed. Commit its record only. | `timeout 1200` |
| S3 | Registration: append exactly `(BASE, BASE + 441)` last to ACCOUNTED, and create `plan.json` = `build_plan`. Run `test_seed_selection.py` once. Commit the registry and plan only. | `timeout 60` |
| S4 | Post-proof (`--mode post`): exit 0, STATE HOLDS. Commit its record only. | `timeout 1200` |
| G | Detached launch. Commit `G/` (replays force-added) after termination and the survivor check. | 12 h cap |
| P | Foreground. Commit `P/`. | 5 min cap |
| L1 | Detached. Commit `L1/`. | 85 min cap |
| Q | Foreground, once. Commit `Q/`. If `q.json` says QUALIFICATION NON-PASS, go to K. | 60 s |
| L2 | Detached; refuses unless Q QUALIFIED. Commit `L2/`. | 3.5 h cap |
| T | Foreground; refuses unless Q QUALIFIED. Candidates go to the main checkout's `runs/` (gitignored); their sha256s are in the record. Commit `T/`. | 30 min cap |
| K | Foreground, after the last phase that ran (whatever it ended in). Commit `K/`. | 60 s |
| K2 | Closing edit: append each `closing.json` EXPOSED interval in order, then `(BASE, BASE + 441)` to RETIRED, each last, with comments holding no literal. ACCOUNTED unchanged. Commit the registry only. | — |

**Card caps total:** 17.5 h (G 12 h, P 5 min, L1 85 min, L2 3.5 h, T 30 min). Q and K together add at most 2 minutes.
