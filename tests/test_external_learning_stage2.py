"""External-engine learning STAGE 2 (training smoke) -- SYNTHETIC checks only.

No incumbent is trained here: the network tests use a freshly initialized network of the incumbent's
architecture on synthetic positions. The only real inputs read are the committed manifest and the stage-1
training rows and logit artifact it binds (hashes and bindings; no forward pass). No held-out row is used.
"""
import ast
import importlib.util
import json
import pathlib
import random
import shutil

import numpy as np
import pytest

from scripts.GPU.alphazero import external_learning as EL
from scripts.GPU.alphazero import external_learning_stage2 as S2
from scripts.GPU.alphazero import external_learning_stage2_authorization as GATE
from scripts.GPU.alphazero.game.twixt_state import TwixtState

SYNTHETIC = 1_000_000                                    # never a 2026-band seed
needs_mlx = pytest.mark.skipif(importlib.util.find_spec("mlx") is None, reason="needs MLX")


def _rows(n, seed=0, plies=(3, 10), chosen=None):
    """Synthetic training rows carrying exactly what `position_record` reads."""
    rng = random.Random(seed)
    rows = []
    for k in range(n):
        st, prefix = TwixtState(max_plies_limit=EL.PLY_CAP), []
        for _ in range(rng.randint(*plies)):
            m = rng.choice(st.legal_moves())
            prefix.append(list(m))
            st = st.apply_move(m)
        legal = [list(m) for m in st.legal_moves()]
        rows.append({"row_id": f"r{k}", "prefix": prefix, "legal_moves": legal, "ply": st.ply, "side": st.to_move,
                     "chosen_index": rng.randrange(len(legal)) if chosen is None else len(legal) + chosen,
                     "game_id": f"g{k % 3}",
                     "split": "train"})
    return rows


def _network(seed=0):
    """A fresh network of the incumbent's architecture with NON-TRIVIAL BN running statistics, so an update
    that touched them would show."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from scripts.GPU.alphazero.network import create_network

    mx.random.seed(seed)
    net = create_network()
    stats = {k: mx.random.uniform(-0.2, 0.2, v.shape) if k.endswith("mean") else mx.random.uniform(0.5, 1.5, v.shape)
             for k, v in tree_flatten(net.parameters()) if k.endswith(("running_mean", "running_var"))}
    net.load_weights(list(stats.items()), strict=False)
    net.eval()
    return net


def _params(net):
    from mlx.utils import tree_flatten

    return {k: np.array(v) for k, v in tree_flatten(net.parameters())}


# ─────────────────────────────── the gate, the manifest, the plan ───────────────────────────────

def test_the_gate_is_closed_and_both_entries_refuse_creating_nothing(tmp_path):
    assert GATE.EXTERNAL_LEARNING_STAGE2_AUTHORIZED is False
    assert S2.main(["run"]) == S2.EXIT_REFUSED
    assert S2.main(["_child", "--run-dir", str(tmp_path)]) == S2.EXIT_REFUSED
    assert not (S2.ROOT / S2.RUN_DIR).exists() and not (S2.ROOT / S2.EVIDENCE_DIR).exists()
    assert list(tmp_path.iterdir()) == []


def test_the_gate_module_holds_only_its_declaration():
    tree = ast.parse(pathlib.Path(GATE.__file__).read_text())
    assert [type(n).__name__ for n in tree.body] == ["Expr", "Assign"]


def test_the_committed_manifest_is_PINNED_and_DERIVED_and_holds_the_cards_budget():
    m = S2.check_manifest()
    assert (m["budget"], m["trainable"]["parameters"]) == ({"steps": 128, "batch": 32, "deadline_s": 1800}, 519)
    assert {k: m["optimizer"][k] for k in ("name", "learning_rate", "grad_clip_global_norm", "weight_decay")} == \
        {"name": "adam", "learning_rate": 0.001, "grad_clip_global_norm": 1.0, "weight_decay": 0.0}
    assert m["trainable"]["names"] == [f"policy_head.{a}.{b}" for a in ("conv", "bn", "fc", "out")
                                       for b in ("weight", "bias")]
    assert m["incumbent"] == {"reference": "calib020_0001",
                              "path": "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors",
                              "sha1": "209cf2d4fd24a48553d259dd71b4954867b9473e"}
    assert m["candidate"] == {"run_path": "runs/external-engine-learning/stage2/candidate.safetensors",
                              "evidence_path": "docs/superpowers/evidence/external-engine-learning-stage2/"
                                               "candidate.safetensors"}
    assert m["stage1"]["rows"]["sha256"].startswith("b59218ea") and m["stage1"]["logits"]["sha256"].startswith("82ff3f2b")
    assert m["control"] == {"beta_init": 1.0, "beta_bounds": [0.0, 100.0], "iterations": 64, "tolerance": 1e-6}
    assert m["training_seed"] == {"slots": 1, "plan": S2.PLAN}


def _evidence_copy(tmp_path):
    for rel in (S2.MANIFEST, EL.EVIDENCE_DIR + "/MANIFEST.sha256", EL.EVIDENCE_DIR + "/record.json",
                EL.EVIDENCE_DIR + "/" + S2.STAGE1_ROWS, EL.EVIDENCE_DIR + "/" + S2.STAGE1_LOGITS):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(S2.ROOT / rel, tmp_path / rel)
    return tmp_path


def test_a_changed_manifest_or_stage1_input_REFUSES(tmp_path):
    root = _evidence_copy(tmp_path)
    m = json.loads((root / S2.MANIFEST).read_text())
    (root / S2.MANIFEST).write_text(json.dumps({**m, "budget": {**m["budget"], "steps": 129}}, indent=1, sort_keys=True) + "\n")
    with pytest.raises(S2.Stop, match="not the pinned bytes"):
        S2.check_manifest(root)
    logits = root / EL.EVIDENCE_DIR / S2.STAGE1_LOGITS
    logits.write_bytes(logits.read_bytes().replace(b'"logits": [', b'"logits":  ['))
    with pytest.raises(S2.Stop, match="training_logits.json is not the file stage 1's manifest hashed"):
        S2.load_inputs(m, root)
    (root / EL.EVIDENCE_DIR / "MANIFEST.sha256").write_text("x\n")
    with pytest.raises(S2.Stop, match="stage-1 evidence manifest is not the bound one"):
        S2.load_inputs(m, root)


def test_the_REAL_cache_binds_to_the_REAL_training_rows_and_nothing_else():
    """The run's own loader on the committed evidence: 128 training rows, each rebuilt in our engine, and
    their pinned logit artifact (rows, legal digests, teacher indices, incumbent). No held-out row returned."""
    train, logits = S2.load_inputs(S2.check_manifest())
    assert len(train) == len(logits) == 128 and {r["split"] for r in train} == {"train"}
    assert [len(z) for z in logits] == [len(r["legal_moves"]) for r in train]


def test_a_REORDERED_legal_list_or_another_incumbent_REFUSES_the_cache(tmp_path):
    train, logits = S2.load_inputs(S2.check_manifest())
    art = S2.ROOT / EL.EVIDENCE_DIR / S2.STAGE1_LOGITS
    sha = EL._sha(art.read_bytes())
    bad = [dict(r) for r in train]
    bad[5] = {**bad[5], "legal_digest": "0" * 64}
    with pytest.raises(S2.Stop, match="not bound to these rows"):
        EL.load_logit_artifact(art, sha256=sha, rows=bad, incumbent_sha1=EL.REFERENCE_SHA1)
    with pytest.raises(S2.Stop, match="not bound to these rows"):
        EL.load_logit_artifact(art, sha256=sha, rows=train, incumbent_sha1="0" * 40)
    swapped = {**train[3], "legal_moves": [train[3]["legal_moves"][1], train[3]["legal_moves"][0],
                                           *train[3]["legal_moves"][2:]]}
    with pytest.raises(S2.Stop, match="legal moves or their order differ"):
        EL.verify_row(swapped)


def test_check_plan_refuses_an_unregistered_seed_and_any_tampering(monkeypatch):
    from scripts.GPU.alphazero import e4_screen_reference as REG

    plan = S2.build_plan(SYNTHETIC, "a" * 64)
    assert plan["manifest_sha256"] == S2.MANIFEST_SHA256 and plan["n_slots"] == 1
    with pytest.raises(S2.Stop, match="not ACCOUNTED"):
        S2.check_plan(plan)                                          # the real registry
    monkeypatch.setattr(REG, "seed_is_accounted", lambda s: True)
    monkeypatch.setattr(REG, "seed_is_unavailable", lambda s: False)
    assert S2.check_plan(json.loads(json.dumps(plan))) == plan
    for bad in ({**plan, "seed": float(SYNTHETIC)}, {**plan, "manifest_sha256": "0" * 64}, {**plan, "n_slots": 2},
                {**plan, "survey_record_sha256": None}, {**plan, "extra": 1}):
        with pytest.raises(S2.Stop):
            S2.check_plan(bad)
    monkeypatch.setattr(REG, "seed_is_unavailable", lambda s: True)
    with pytest.raises(S2.Stop, match="spent"):
        S2.check_plan(plan)


def test_the_schedule_is_uniform_with_replacement_seeded_and_the_FIXED_budget():
    a = S2.sampling_schedule(SYNTHETIC, 128)
    assert a.shape == (128, 32) and a.min() >= 0 and a.max() < 128
    assert np.array_equal(a, S2.sampling_schedule(SYNTHETIC, 128))
    assert not np.array_equal(a, S2.sampling_schedule(SYNTHETIC + 1, 128))
    assert any(len(set(row)) < len(row) for row in a.tolist())                   # with replacement
    counts = np.bincount(a.reshape(-1), minlength=128)
    assert counts.min() > 0 and counts.max() < 4 * counts.mean()                  # every row reachable


# ─────────────────────────────── the surface and the update ───────────────────────────────

@needs_mlx
def test_the_trainable_surface_is_EXACTLY_the_519_head_parameters_and_never_its_BN_statistics():
    from mlx.utils import tree_flatten

    net = _network()
    assert S2.head_only(net) == sorted(S2.TRAINABLE)
    net.freeze()
    net.policy_head.unfreeze()                                       # the trap: unfreeze exposes the statistics
    exposed = sorted(k for k, _ in tree_flatten(net.trainable_parameters()))
    assert "policy_head.bn.running_mean" in exposed and sum(v.size for _, v in tree_flatten(net.trainable_parameters())) == 523
    assert S2.head_only(net) == sorted(S2.TRAINABLE)


@needs_mlx
def test_an_update_changes_ONLY_head_tensors_keeps_every_BN_statistic_and_every_value(monkeypatch):
    monkeypatch.setattr(S2, "STEPS", 4)
    monkeypatch.setattr(S2, "BATCH", 4)
    net, rows = _network(), _rows(6)
    recs = S2.records_of(rows)
    before = _params(net)
    z0, v0 = S2.forward_rows(net, recs)
    log = S2.train_head(net, recs, S2.sampling_schedule(SYNTHETIC, len(rows)))
    after = _params(net)
    changed = sorted(k for k in before if not np.array_equal(before[k], after[k]))
    assert changed and set(changed) <= set(S2.TRAINABLE) and len(log) == 4
    stats = [k for k in before if k.endswith(("running_mean", "running_var"))]
    assert stats and all(np.array_equal(before[k], after[k]) for k in stats)
    z1, v1 = S2.forward_rows(net, recs)
    assert v1 == v0 and z1 != z0
    assert not net.training


@needs_mlx
def test_TRAIN_MODE_would_have_moved_the_BN_statistics__negative_control():
    net, rows = _network(), _rows(4)
    before = _params(net)
    net.train()
    boards, mr, mc, mask, target, _o = S2.batch_of(S2.records_of(rows))
    S2.chosen_ce(net, boards, mr, mc, mask, target)
    after = _params(net)
    assert any(not np.array_equal(before[k], after[k]) for k in before if k.endswith("running_mean"))


@needs_mlx
def test_the_target_is_the_chosen_index_in_OUR_legal_order_even_beyond_slot_512():
    rows = _rows(3, plies=(2, 2), chosen=-1) + _rows(3, seed=1, plies=(3, 3))     # the LAST legal move
    assert all(r["chosen_index"] >= 512 for r in rows[:3])
    assert {r["side"] for r in rows} == {"red", "black"}
    boards, mr, mc, mask, target, _o = S2.batch_of(S2.records_of(rows))
    t, mk = np.array(target), np.array(mask)
    assert [int(i) for i in t.argmax(axis=1)] == [r["chosen_index"] for r in rows]
    assert np.allclose(t.sum(axis=1), 1.0) and [int(x) for x in mk.sum(axis=1)] == [len(r["legal_moves"]) for r in rows]
    assert [[int(np.array(mr)[i, j]), int(np.array(mc)[i, j])] for i, j in enumerate(t.argmax(axis=1))] == \
        [r["legal_moves"][r["chosen_index"]] for r in rows]


@needs_mlx
def test_training_RAISES_the_chosen_move_for_BOTH_colours(monkeypatch):
    """Canonicalization: a black-to-move row's target must line up with the move it names. Trained on one
    red-to-move and one black-to-move row, each chosen move climbs from a near-uniform start (rank ~ n/2 of
    500+) to the top 20 -- a misaligned target would train some OTHER cell and leave its rank random."""
    monkeypatch.setattr(S2, "STEPS", 200)
    monkeypatch.setattr(S2, "BATCH", 2)
    rows = _rows(1, seed=2, plies=(4, 4)) + _rows(1, seed=3, plies=(5, 5))
    assert [r["side"] for r in rows] == ["red", "black"] and min(len(r["legal_moves"]) for r in rows) > 500
    net, recs = _network(1), S2.records_of(rows)
    z0, _ = S2.forward_rows(net, recs)
    S2.train_head(net, recs, S2.sampling_schedule(SYNTHETIC, 2))
    z1, _ = S2.forward_rows(net, recs)
    rank = lambda z, i: sorted(range(len(z)), key=lambda j: -z[j]).index(i)
    for r, a, b in zip(rows, z0, z1):
        assert rank(b, r["chosen_index"]) < 20 < rank(a, r["chosen_index"]), (r["side"], rank(a, r["chosen_index"]))


@needs_mlx
def test_a_NONFINITE_step_stops(monkeypatch):
    import mlx.core as mx

    from scripts.GPU.alphazero import trainer

    monkeypatch.setattr(S2, "STEPS", 2)
    monkeypatch.setattr(S2, "BATCH", 2)
    monkeypatch.setattr(trainer, "clip_grad_norm", lambda g, max_norm: (g, mx.array(float("nan"))))
    rows = _rows(2)
    with pytest.raises(S2.Stop, match="nonfinite loss or gradient"):
        S2.train_head(_network(), S2.records_of(rows), S2.sampling_schedule(SYNTHETIC, 2))


def test_NO_change_and_a_COMMON_SHIFT_of_every_legal_logit_both_STOP():
    """A constant added to every logit of a row leaves its probabilities, order and CE unchanged: no change."""
    rows = [{"chosen_index": 1}, {"chosen_index": 0}]
    z = [[0.0, 1.0, 0.5], [2.0, 0.0]]
    with pytest.raises(S2.Stop, match="no measurable policy change"):
        S2.policy_change(z, [list(x) for x in z], rows)
    for shift in (2.0, -7.5, 30.0):
        shifted = [[v + shift for v in x] for x in z]
        with pytest.raises(S2.Stop, match="no measurable policy change"):
            S2.policy_change(z, shifted, rows)
    per_row = [[v + 2.0 for v in z[0]], [v - 3.0 for v in z[1]]]                    # a different shift per row
    with pytest.raises(S2.Stop, match="no measurable policy change"):
        S2.policy_change(z, per_row, rows)
    tiny = [[z[0][0] + 1e-9, *z[0][1:]], list(z[1])]                                # below the tolerance
    with pytest.raises(S2.Stop, match="no measurable policy change"):
        S2.policy_change(z, tiny, rows)


def test_a_GENUINE_distribution_change_passes_without_any_top1_change():
    rows = [{"chosen_index": 1}, {"chosen_index": 0}]
    z = [[0.0, 1.0, 0.5], [2.0, 0.0]]
    after = [[0.0, 1.0, 0.9], [2.0, 0.0]]                                          # one logit moved; top-1 kept
    out = S2.policy_change(z, after, rows)
    p0, p1 = S2.legal_probs(z[0]), S2.legal_probs(after[0])
    assert out["max_abs_prob_change"] == pytest.approx(max(abs(a - b) for a, b in zip(p0, p1)))
    assert out["max_abs_prob_change"] > S2.POLICY_TOL and out["train_top1_changed"] == 0
    assert out["train_ce_candidate"] > out["train_ce_incumbent"]                     # chosen 1 lost mass to 2
    flipped = S2.policy_change(z, [[1.5, 1.0, 0.5], [2.0, 0.0]], rows)
    assert flipped["train_top1_changed"] == 1
    assert (flipped["train_top1_teacher_incumbent"], flipped["train_top1_teacher_candidate"]) == (2, 1)


# ─────────────────────────────── the control ───────────────────────────────

@needs_mlx
def test_the_control_is_fitted_on_the_cached_TRAINING_logits():
    rows = [{"chosen_index": 0}, {"chosen_index": 0}, {"chosen_index": 0}]          # row 1: NOT its top-1
    logits = [[1.0, 0.0, -1.0], [0.0, 2.0], [0.5, 0.4, 0.3, 0.2]]
    out = S2.fit_control(rows, logits)
    assert out == EL.fit_beta(logits, [0, 0, 0]) and 0 < out["beta"] < 100 and abs(out["derivative"]) < 1e-3


def test_a_FAILED_control_fit_stops_BEFORE_any_network_is_loaded(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(EL, "_committed", lambda rel: True)
    monkeypatch.setattr(S2, "load_plan", lambda: ({"seed": SYNTHETIC, "n_slots": 1}, "p" * 64))
    monkeypatch.setattr(S2, "check_manifest", lambda: {"stage1": {"rows": {"sha256": "r"}, "logits": {"sha256": "l"}}})
    monkeypatch.setattr(S2, "load_inputs", lambda m: ([{"chosen_index": 0}], [[0.0, 1.0]]))

    def fail(train, logits):
        raise S2.Stop("beta", "CE still decreasing at beta = 100")
    monkeypatch.setattr(S2, "fit_control", fail)
    monkeypatch.setattr(S2, "load_incumbent", lambda m: called.append("loaded"))
    rec = {"steps": {}}
    with pytest.raises(S2.Stop, match="CE still decreasing"):
        S2.stage2(tmp_path, rec)
    assert called == [] and "control" not in rec["steps"] and "inputs" in rec["steps"]


# ─────────────────────────────── save and reload ───────────────────────────────

@needs_mlx
def test_save_ONCE_reload_EXACT_and_only_head_tensors_differ_from_the_incumbent_file(monkeypatch, tmp_path):
    monkeypatch.setattr(S2, "STEPS", 3)
    monkeypatch.setattr(S2, "BATCH", 3)
    inc = tmp_path / "incumbent.safetensors"
    net = _network()
    net.save_weights(str(inc))
    rows = _rows(4)
    S2.train_head(net, S2.records_of(rows), S2.sampling_schedule(SYNTHETIC, 4))
    cand = tmp_path / S2.CANDIDATE
    sha = S2.save_candidate(net, cand)
    assert sha == EL._sha(cand.read_bytes())
    with pytest.raises(S2.Stop, match="saved once"):
        S2.save_candidate(net, cand)
    surface = S2.surface_check(inc, cand)
    assert surface["changed"] and set(surface["changed"]) <= set(S2.TRAINABLE)
    assert surface["identical"] == surface["tensors"] - len(surface["changed"])
    reloaded = S2.reload_check(net, cand)
    _z, v_inc = S2.forward_rows(_network(), S2.records_of(rows))
    assert S2.forward_rows(reloaded, S2.records_of(rows)) == S2.forward_rows(net, S2.records_of(rows))
    assert S2.forward_rows(reloaded, S2.records_of(rows))[1] == v_inc


@needs_mlx
def test_a_FORBIDDEN_tensor_change_an_UNCHANGED_head_and_a_WRONG_reload_all_stop(tmp_path):
    import mlx.core as mx

    inc, same, enc = tmp_path / "inc.safetensors", tmp_path / "same.safetensors", tmp_path / "enc.safetensors"
    net = _network()
    net.save_weights(str(inc))
    net.save_weights(str(same))
    with pytest.raises(S2.Stop, match="no policy-head tensor changed"):
        S2.surface_check(inc, same)
    w = dict(mx.load(str(inc)))
    key = next(k for k in w if k.startswith("encoder."))
    w[key] = w[key] + 1.0
    w["policy_head.out.bias"] = w["policy_head.out.bias"] + 1.0
    mx.save_safetensors(str(enc), w)
    with pytest.raises(S2.Stop, match="a forbidden tensor changed"):
        S2.surface_check(inc, enc)
    with pytest.raises(S2.Stop, match="not the trained network"):
        S2.reload_check(net, enc)


# ─────────────────────────────── the whole stage, synthetic ───────────────────────────────

@needs_mlx
def test_the_WHOLE_stage_on_synthetic_inputs_PASSES_and_records_every_step(monkeypatch, tmp_path):
    monkeypatch.setattr(S2, "STEPS", 3)
    monkeypatch.setattr(S2, "BATCH", 4)
    rows = _rows(6)
    inc = tmp_path / "incumbent.safetensors"
    _network().save_weights(str(inc))
    logits = [[float(i % 3) for i in range(len(r["legal_moves"]))] for r in rows]
    monkeypatch.setattr(EL, "_committed", lambda rel: True)
    monkeypatch.setattr(S2, "load_plan", lambda: ({"seed": SYNTHETIC, "n_slots": 1}, "p" * 64))
    monkeypatch.setattr(S2, "check_manifest", lambda: {"stage1": {"rows": {"sha256": "r"}, "logits": {"sha256": "l"}}})
    monkeypatch.setattr(S2, "load_inputs", lambda m: (rows, logits))
    monkeypatch.setattr(S2, "load_incumbent", lambda m: (_network(), inc))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rec = {"steps": {}}
    S2.stage2(run_dir, rec)
    s = rec["steps"]
    assert list(s) == ["plan", "manifest_sha256", "inputs", "control", "surface", "schedule_sha256", "training",
                       "candidate_sha256", "candidate_surface", "values_unchanged", "policy_change"]
    assert s["training"]["steps"] == 3 and s["values_unchanged"] == 6 and (run_dir / S2.CANDIDATE).is_file()
    events = [json.loads(line) for line in (run_dir / "progress.jsonl").read_text().splitlines()]
    assert [e["step"] for e in events if e["event"] == "train_step"] == [0, 1, 2]
