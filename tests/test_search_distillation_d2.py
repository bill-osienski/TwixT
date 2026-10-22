"""Search distillation D2, Stage 0: the offline whole-policy trainer (search_distillation_d2.py). CPU, synthetic.

The capture comes from the minimal serial driver with REAL MCTS and the repo's deterministic FakeEvaluator (uniform
priors, value 0) -- no network plays. Training uses a small random-init network with non-default BatchNorm running
statistics so that any reset or update of them is visible. Nothing here measures target quality or strength.

Checks: the reopened records ARE the learner rows (board, legal order, counts, outcome) and batch identically through
trainer.make_padded_batch; training changes the encoder/policy head only (value head and every running statistic
bitwise unchanged) and never the teacher; value drift is reported as ABSOLUTE differences; save/reload is exact;
misuse refuses.
"""
import random

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

import mlx.nn as nn  # noqa: E402

from scripts.GPU.alphazero import search_distillation_d2 as D  # noqa: E402
from scripts.GPU.alphazero import self_play as sp  # noqa: E402
from scripts.GPU.alphazero import training_capture_serial as S  # noqa: E402
from scripts.GPU.alphazero import training_replay_capture as C  # noqa: E402
from scripts.GPU.alphazero.mcts import MCTSConfig  # noqa: E402
from scripts.GPU.alphazero.network import AlphaZeroNetwork  # noqa: E402
from scripts.GPU.alphazero.trainer import make_padded_batch  # noqa: E402
from tests.fpu_search_fixture import FakeEvaluator  # noqa: E402

SEEDS = {0: 11, 1: 12}
CFG = MCTSConfig(n_simulations=8)
DIAG_OFF = {"goal_completion_emit_enabled": False, "goal_completion_record_enabled": False}
RECIPE = D.Recipe(steps=12, batch_size=16, learning_rate=3e-3, anchor_weight=1.0, l2_weight=1e-4, clip_norm=1.0,
                  seed=0)


def _small_net(seed=0):
    """A small network whose BatchNorm running statistics are NOT the defaults (0 / 1)."""
    mx.random.seed(seed)
    net = AlphaZeroNetwork(hidden=16, n_blocks=1)
    rng = np.random.default_rng(seed)
    for _, m in net.named_modules():
        if isinstance(m, nn.BatchNorm):
            n = m.running_mean.shape[0]
            m.running_mean = mx.array(rng.normal(0, 0.3, n).astype(np.float32))
            m.running_var = mx.array(rng.uniform(0.5, 2.0, n).astype(np.float32))
    net.eval()
    return net


def _copy(net, tmp_path, name):
    path = tmp_path / f"{name}.safetensors"
    net.save_weights(str(path))
    return D.reload(str(path), lambda: AlphaZeroNetwork(hidden=16, n_blocks=1))


@pytest.fixture(scope="module")
def phase(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(sp, "_MIRROR_PROB", 0.0)
    net = _small_net()
    ev = FakeEvaluator()
    ev.network = net
    ctx = S.build_context(run_id="test:run", phase_id="test:d2", seeds=SEEDS, model_id="test:model",
                          model_digest=S.network_state_digest(net), model_source={"kind": "random-init test network"},
                          mcts_config=CFG, code={"fixture": "not a source closure"}, phase_bytes=64 << 20,
                          play_kwargs=DIAG_OFF)
    d = tmp_path_factory.mktemp("d2") / "cap"
    out = S.run_serial_capture(evaluator=ev, network=net, capture_dir=str(d), context=ctx, mcts_config=CFG)
    mp.undo()
    return {"dir": d, "games": out["games"]}


def test_THE_REOPENED_RECORDS_ARE_THE_LEARNER_ROWS(phase):
    _ctx, records, ids = D.load_records(str(phase["dir"]))
    learner = [p for g in phase["games"] for p in g.positions]
    assert len(records) == len(learner) == len(ids) > 0
    assert ids == [f"{gid}:{p.ply}" for gid, g in zip(SEEDS, phase["games"]) for p in g.positions]
    for r, p in zip(records, learner):
        assert r.board_tensor.dtype == p.board_tensor.dtype and np.array_equal(r.board_tensor, p.board_tensor)
        assert (r.to_move, r.ply, list(r.legal_moves), r.visit_counts, r.outcome) == \
               (p.to_move, p.ply, list(p.legal_moves), list(p.visit_counts), p.outcome)
    for i in range(0, len(records), 97):                       # the trainer's own batching agrees, array for array
        ours = make_padded_batch(records[i:i + 97], max_moves_cap=576)
        theirs = make_padded_batch(learner[i:i + 97], max_moves_cap=576)
        assert all(np.array_equal(np.array(a), np.array(b)) for a, b in zip(ours, theirs))
    assert max(len(r.legal_moves) for r in records) == 528    # the opening's full width, not truncated


def test_A_ROW_THE_REPLAY_DOES_NOT_REPRODUCE_REFUSES(phase):
    _ctx, envs = C.load_phase(str(phase["dir"]))
    envs[0]["rows"][5]["legal_moves"] = envs[0]["rows"][5]["legal_moves"][::-1]
    with pytest.raises(D.DistillError, match="does not reproduce"):
        D.records_from_envelopes(envs)


@pytest.fixture(scope="module")
def trained(phase, tmp_path_factory):
    _ctx, records, _ids = D.load_records(str(phase["dir"]))
    records = records[::9][:64]                                 # both games, every phase of play
    tmp = tmp_path_factory.mktemp("nets")
    teacher = _small_net(seed=3)
    student = _copy(teacher, tmp, "student")
    original = D.snapshot(teacher)
    tv = D.forward(teacher, records)[1]
    start = D.evaluate(student, records, tv)
    out = D.train_student(student, teacher, records, RECIPE)
    return {"teacher": teacher, "student": student, "records": records, "tv": tv, "start": start, "out": out,
            "original": original, "tmp": tmp}


def test_TRAINING_MOVES_ONLY_THE_ENCODER_AND_POLICY_HEAD(trained):
    before, after = trained["original"], D.snapshot(trained["student"])
    stats = [n for n in before if n.endswith(("running_mean", "running_var"))]
    value = [n for n in before if n.startswith("value_head.")]
    assert len(stats) == 2 * sum(isinstance(m, nn.BatchNorm) for _, m in trained["student"].named_modules()) > 0
    assert all(np.array_equal(before[n], after[n]) for n in stats + value)          # independent of check_after
    assert any(not np.array_equal(before[n], after[n]) for n in before if n.startswith("encoder."))
    assert any(not np.array_equal(before[n], after[n]) for n in before if n.startswith("policy_head."))
    assert set(D.surface_names(trained["student"])) == {n for n in before if n.startswith(("encoder.", "policy_head."))
                                                       and not n.endswith(("running_mean", "running_var"))}
    assert trained["out"]["checks"]["surface_tensors_changed"] > 0
    assert not any(m.training for _, m in trained["student"].named_modules())


def test_THE_TEACHER_IS_NEVER_UPDATED(trained):
    now = D.snapshot(trained["teacher"])
    assert all(np.array_equal(trained["original"][n], now[n]) for n in now)
    assert trained["out"]["teacher_digest"] == C.model_state_digest(trained["original"])
    assert trained["out"]["student_digest"] != trained["out"]["teacher_digest"]


def test_THE_POLICY_MOVES_TOWARD_THE_TARGETS_and_value_drift_is_measured(trained):
    end = D.evaluate(trained["student"], trained["records"], trained["tv"])
    assert end["policy_ce"] < trained["start"]["policy_ce"]
    assert trained["start"]["value_abs_max"] == 0.0                     # the student starts as an exact copy
    assert all(np.isfinite(s["loss"]) for s in trained["out"]["steps"]) and len(trained["out"]["steps"]) == 12
    print("d2-train", {k: round(v, 5) for k, v in trained["start"].items()}, {k: round(v, 5) for k, v in end.items()})


def test_VALUE_DRIFT_IS_ABSOLUTE_so_cancelling_errors_cannot_hide(trained):
    records = trained["records"][:2 * (len(trained["records"]) // 2)]    # an even count
    v = D.forward(trained["teacher"], records)[1]
    shifted = v + np.where(np.arange(len(v)) % 2 == 0, 0.25, -0.25)      # signed mean difference exactly 0
    out = D.evaluate(trained["teacher"], records, shifted)
    assert np.isclose(np.mean(v - shifted), 0.0, atol=1e-9)
    assert out["value_abs_mean"] == pytest.approx(0.25, abs=1e-6) and out["value_abs_max"] == pytest.approx(0.25, abs=1e-6)


def test_SAVE_AND_RELOAD_ARE_EXACT_and_create_only(trained):
    path = trained["tmp"] / "student_out.safetensors"
    saved = D.save_student(trained["student"], str(path))
    back = D.reload(str(path), lambda: AlphaZeroNetwork(hidden=16, n_blocks=1))
    assert D.state_digest(back) == saved["state_digest"] == trained["out"]["student_digest"]
    a, b = D.forward(trained["student"], trained["records"]), D.forward(back, trained["records"])
    assert np.array_equal(a[1], b[1]) and all(np.array_equal(x, y) for x, y in zip(a[0], b[0]))
    with pytest.raises(D.DistillError, match="exists"):
        D.save_student(trained["student"], str(path))
    assert not [p for p in trained["tmp"].iterdir() if p.name.startswith(".")]          # no staging file left


@pytest.mark.parametrize("case", ["same_object", "student_training_mode", "not_a_copy", "teacher_values_shape"])
def test_MISUSE_REFUSES_BEFORE_ANY_STEP(trained, tmp_path, case):
    teacher = _small_net(seed=3)
    student = _copy(teacher, tmp_path, "s")
    records, tv = trained["records"][:8], None
    if case == "same_object":
        student = teacher
    if case == "student_training_mode":
        student.train()
    if case == "not_a_copy":
        student = _small_net(seed=4)
    if case == "teacher_values_shape":
        tv = np.zeros(3)
    before = D.snapshot(student)
    match = {"same_object": "same object", "student_training_mode": "training mode", "not_a_copy": "exact copy",
             "teacher_values_shape": "do not match"}[case]
    with pytest.raises(D.DistillError, match=match):
        D.train_student(student, teacher, records, RECIPE, teacher_values=tv)
    after = D.snapshot(student)
    assert all(np.array_equal(before[n], after[n]) for n in before)


@pytest.mark.parametrize("case", ["value_head", "running_stat"])
def test_CHECK_AFTER_CATCHES_A_FROZEN_TENSOR_THAT_MOVED(trained, case):
    before = trained["original"]
    names = D.surface_names(trained["student"])
    after = dict(D.snapshot(trained["student"]))
    victim = next(n for n in after if (n.startswith("value_head.") if case == "value_head"
                                       else n.endswith("running_mean")))
    after[victim] = after[victim] + 1e-3
    with pytest.raises(D.DistillError, match="off the update surface changed"):
        D.check_after(before, after, names)
