"""Planted-target qualification (card docs/superpowers/2026-10-21-policy-head-capacity-planted-qualification-card.md,
rev 2): the build's focused CPU checks. STUBS ONLY -- synthetic games played through the rules engine, a seeded
random-init network in place of calib020_0001, and random features keyed by state digest. No checkpoint, no real
feature, no fit on real data. The gate stays closed. `trainer` (hence `self_play`) is imported: linked worktree only."""
import itertools
import json
import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from scripts.GPU.alphazero import policy_head_capacity as PHC  # noqa: E402
from scripts.GPU.alphazero import policy_head_capacity_planted as PL  # noqa: E402
from scripts.GPU.alphazero import policy_head_capacity_planted_authorization as GATE  # noqa: E402
from scripts.GPU.alphazero import search_distillation_stage_b as SB  # noqa: E402
from scripts.GPU.alphazero.game import TwixtState  # noqa: E402
from scripts.GPU.alphazero.network import AlphaZeroNetwork  # noqa: E402

N_GAMES, PLIES = 10, 8
SMALL = {"n_train": 48, "n_eval": 12, "steps": 12, "batch": 8,
         "games": N_GAMES, "moves": N_GAMES * PLIES, "eval_games": 2, "eval_moves": 2 * PLIES}
CHANNELS = TwixtState(active_size=24, to_move="red", max_plies_limit=280).to_tensor().shape[0]


def _game(idx, plies=PLIES):
    rng = np.random.default_rng(100 + idx)
    s, moves = TwixtState(active_size=24, to_move="red", max_plies_limit=280), []
    for p in range(plies):
        legal = s.legal_moves()
        mv = legal[int(rng.integers(len(legal)))]
        moves.append({"ply": p, "player": s.to_move, "row": mv[0], "col": mv[1], "n_legal": len(legal)})
        s = s.apply_move(mv)
    return {"game_idx": idx, "board_size": 24, "moves": moves}


GAMES = [("screen-a", _game(i)) for i in range(N_GAMES)]


def _net():
    mx.random.seed(0)
    net = AlphaZeroNetwork()
    net.policy_head.bn.running_mean = mx.array([0.3, -0.2])     # non-default, so a changed statistic is visible
    net.policy_head.bn.running_var = mx.array([1.7, 0.6])
    net.eval()
    return net


def _features(network, states):
    out = np.empty((len(states), 24, 24, PL.HIDDEN), np.float32)
    for i, s in enumerate(states):
        out[i] = np.maximum(np.random.default_rng(int(PL.state_digest(s)[:12], 16)).standard_normal((24, 24, PL.HIDDEN)), 0)
    return out


def _seams(**over):
    base = {"load_games": lambda: GAMES, "checkpoint_sha1": lambda: PL.CHECKPOINT_SHA1,
            "load_network": lambda: (_net(), CHANNELS), "features": _features}
    return PL.Seams(**{**base, **over})


@pytest.fixture(scope="module")
def data():
    """Small training / evaluation splits with stub features, the incumbent logits and both planted targets."""
    train_pool, eval_pool, _ = PL.pools(GAMES)
    t, e = PL.select(train_pool, eval_pool, SMALL)
    T, E = PL._split(t), PL._split(e)
    for x in (T, E):
        x["feats"] = _features(None, x["states"])
    head = _net().policy_head
    i_t, i_e = PL.head_logits(head, T), PL.head_logits(head, E)
    noise = PL.tc_noise(head)
    alpha, got = PL.bisect_alpha(lambda a: PL.centred_scale(
        PL.head_logits(PL.perturbed_head(head, noise, a), T).astype(np.float64) - i_t, T["mask"]))
    tc_head = PL.perturbed_head(head, noise, alpha)
    w1, w2 = PL.tr_teacher()
    s = 1.0 / PL.centred_scale(PL.g_values(T, w1, w2), T["mask"])
    for x, i in ((T, i_t), (E, i_e)):
        x["TC"] = PL.targets(PL.head_logits(tc_head, x), x["mask"])
        x["TR"] = PL.targets(i.astype(np.float64) + s * PL.g_values(x, w1, w2), x["mask"])
    return {"T": T, "E": E, "i_t": i_t, "i_e": i_e, "noise": noise, "alpha": alpha, "tc_scale": got,
            "tc_head": tc_head, "w1": w1, "w2": w2, "s": s}


# ─────────────────────────────── pins, gate, public entry ───────────────────────────────

def test_the_digest_IS_stage_b_s_and_the_card_is_the_pinned_bytes():
    assert PL.state_digest is SB.state_digest
    assert PL._sha((PL.ROOT / PL.CARD).read_bytes()) == PL.CARD_SHA256


def test_the_gate_is_closed_and_the_public_entry_refuses_before_writing_anything(tmp_path):
    assert GATE.POLICY_HEAD_CAPACITY_PLANTED_AUTHORIZED is False
    evidence = PL.ROOT / PL.EVIDENCE_DIR
    assert not evidence.exists()
    with pytest.raises(PL.Refused, match="closed"):
        PL.run(str(tmp_path / "no-such.safetensors"))
    assert PL.main(["run", "--checkpoint", str(tmp_path / "x")]) == PL.EXIT_REFUSED
    assert PL.main([]) == PL.EXIT_USAGE
    assert not evidence.exists()


def test_the_frozen_numbers_are_the_card_s():
    assert PL.FROZEN["steps"] * PL.FROZEN["batch"] == 8 * PL.FROZEN["n_train"]
    assert (PL.FROZEN["n_train"], PL.FROZEN["n_eval"], PL.FROZEN["steps"], PL.FROZEN["batch"]) == (1600, 400, 400, 32)
    assert PL.SEEDS == {"selection": 11, "tc": 12, "tr": 13, "order": 14} and len(set(PL.SEEDS.values())) == 4
    assert (PL.LEARNING_RATE, PL.CLIP_NORM, PL.WEIGHT_DECAY, PL.DELTA, PL.FIT_FRACTION) == (1e-3, 1.0, 0.0, 0.05, 0.25)
    assert PL.ADAM == {"betas": [0.9, 0.999], "eps": 1e-8, "bias_correction": False}
    assert PL.CACHE_DIR.resolve().is_relative_to(PL.ROOT.resolve()) is False
    assert PL.TC_ARRAYS == tuple(sorted(f"{m}.{p}" for m in ("conv", "bn", "fc", "out") for p in ("weight", "bias")))


# ─────────────────────────────── §3 replay, split, leakage, selection, draw order ───────────────────────────────

def test_replay_yields_the_state_before_every_move():
    g = GAMES[0][1]
    out = PL.replay(g)
    assert [p for p, _ in out] == list(range(PLIES)) and out[0][1].ply == 0 and not out[0][1].pegs


@pytest.mark.parametrize("defect, message", [
    (lambda g: g["moves"][3].update(n_legal=g["moves"][3]["n_legal"] + 1), "legal moves, the record says"),
    (lambda g: g["moves"][3].update(row=g["moves"][1]["row"], col=g["moves"][1]["col"]), "is not legal"),
    (lambda g: g["moves"][2].update(ply=5), "not in ply order"),
    (lambda g: g.update(board_size=20), "board_size 20"),
])
def test_THE_REPLAY_CHECK_voids_on_each_mismatch(defect, message):
    g = json.loads(json.dumps(GAMES[1][1]))
    defect(g)
    with pytest.raises(PL.Void, match=message):
        PL.replay(g)


def test_the_split_is_by_game_with_dedupe_and_leakage():
    assert [i for i in range(10) if PL.is_eval_game(i)] == [4, 9]
    train, held, leaked = PL.pools(GAMES)
    assert {p["game_idx"] for p in train}.isdisjoint({4, 9}) and {p["game_idx"] for p in held} <= {4, 9}
    every = {False: set(), True: set()}                       # derived independently, through Stage B's digest
    for _, g in GAMES:
        every[PL.is_eval_game(g["game_idx"])] |= {SB.state_digest(s) for _, s in PL.replay(g)}
    assert [p["digest"] for p in train] == list(dict.fromkeys(p["digest"] for p in train))
    assert {p["digest"] for p in train} == every[False] and len(train) == len(every[False])
    assert {p["digest"] for p in held} == every[True] - every[False] and len(held) == len(every[True] - every[False])
    assert leaked == len(every[True] & every[False]) >= 1
    empty = PL.state_digest(TwixtState(active_size=24, to_move="red", max_plies_limit=280))
    first = [p for p in train if p["digest"] == empty]
    assert len(first) == 1 and (first[0]["game_idx"], first[0]["ply"]) == (0, 0)      # first occurrence kept
    assert empty not in {p["digest"] for p in held}                                   # leaked: left eval, kept in train


def test_counts_and_selection_void_and_selection_is_deterministic():
    with pytest.raises(PL.Void, match="not the frozen"):
        PL.check_counts(GAMES, {**SMALL, "moves": SMALL["moves"] + 1})
    train, held, _ = PL.pools(GAMES)
    with pytest.raises(PL.Void, match="fewer than"):
        PL.select(train, held, {**SMALL, "n_train": len(train) + 1})
    a, b = PL.select(train, held, SMALL), PL.select(train, held, SMALL)
    assert [p["digest"] for p in a[0] + a[1]] == [p["digest"] for p in b[0] + b[1]]


def test_the_draw_order_is_whole_permutations_in_order():
    order = PL.draw_order(48, 12, 8)
    assert order.shape == (12, 8)
    for k in range(2):
        assert sorted(order.reshape(-1)[k * 48:(k + 1) * 48].tolist()) == list(range(48))
    assert np.array_equal(order, PL.draw_order(48, 12, 8)) and not np.array_equal(order, PL.draw_order(48, 12, 8, seed=15))
    with pytest.raises(ValueError):
        PL.draw_order(48, 5, 8)


# ─────────────────────────────── §4 the within-position scale and the teachers ───────────────────────────────

def test_the_within_position_scale_subtracts_each_position_s_mean():
    added = np.array([[1.0, 3.0, 99.0], [10.0, 10.0, 10.0]])
    mask = np.array([[1, 1, 0], [1, 1, 1]], np.float32)
    assert PL.centred_scale(added, mask) == pytest.approx(math.sqrt(2 / 5))        # centred: -1, 1 | 0, 0, 0
    shifted = added + np.array([[7.0], [-3.0]])                                     # a per-position constant
    assert PL.centred_scale(shifted, mask) == pytest.approx(PL.centred_scale(added, mask))


def test_both_teachers_are_planted_at_one_nat(data):
    assert abs(data["tc_scale"] - 1.0) <= PL.SCALE_TOL and 0 < data["alpha"] < 10
    g = PL.g_values(data["T"], data["w1"], data["w2"])
    assert PL.centred_scale(data["s"] * g, data["T"]["mask"]) == pytest.approx(1.0, abs=1e-12)
    for t in ("TC", "TR"):
        p = data["T"][t]
        assert np.allclose(p.sum(1), 1, atol=1e-5) and np.all(p[data["T"]["mask"] < 0.5] == 0)


def test_TC_is_exactly_representable_by_both_arms(data):
    E = data["E"]
    want = PL.head_logits(data["tc_head"], E)
    planted = [(k, v) for k, v in tree_flatten(data["tc_head"].parameters()) if k in PL.TC_ARRAYS]
    for arm in ("C", "R"):
        net = _net()
        PHC.install(net, arm)
        net.policy_head.update(tree_unflatten(planted))
        got = PL.head_logits(net.policy_head, E)
        assert np.array_equal(got, want) and PL.mean_kl(E["TC"], got, E["mask"]) < 1e-6
    assert PL.mean_kl(E["TC"], data["i_e"], E["mask"]) > 0.05                         # the incumbent does not


def test_TR_is_exactly_representable_by_R_only(data):
    E = data["E"]
    net = _net()
    PHC.install(net, "R")
    h = net.policy_head
    h.extra_fc.weight = mx.array(data["w1"].astype(np.float32))
    h.extra_fc.bias = mx.zeros_like(h.extra_fc.bias)
    h.extra_out.weight = mx.array((data["s"] * data["w2"]).astype(np.float32))
    assert PL.mean_kl(E["TR"], PL.head_logits(h, E), E["mask"]) < 1e-6
    assert PL.mean_kl(E["TR"], data["i_e"], E["mask"]) > 0.05


# ─────────────────────────────── §5 the recipe: clipping, guards, temperature ───────────────────────────────

def _one_hot_last(x):
    t = np.zeros_like(x["mask"])
    t[np.arange(len(t)), x["mask"].sum(1).astype(int) - 1] = 1
    return t


def test_CLIPPING_matches_clip_grad_norm_and_is_active(data, monkeypatch):
    from scripts.GPU.alphazero.trainer import clip_grad_norm

    T, E = data["T"], data["E"]
    train = {**T, "target": _one_hot_last(T)}
    order = np.arange(8)[None, :]
    batch = [mx.array(train[k][order[0]]) for k in ("feats", "rows", "cols", "mask", "target")]

    def manual(max_norm):
        ref = _net()
        PHC.install(ref, "C")
        _, grads = nn.value_and_grad(ref, PL.soft_ce)(ref, *batch)
        g, norm = ((grads["policy_head"], None) if max_norm is None
                   else clip_grad_norm(grads["policy_head"], max_norm=max_norm))
        opt = optim.Adam(learning_rate=1e-3, betas=[0.9, 0.999], eps=1e-8, bias_correction=False)
        opt.update(ref.policy_head, g)
        mx.eval(ref.parameters())
        return PL.snapshot(ref), (None if norm is None else float(norm.item()))

    def fitted():
        made = []
        out = PL.fit(lambda: made.append(_net()) or made[-1], "C", train, {**E, "target": E["TC"]}, order,
                     data["i_e"])
        return PL.snapshot(made[0]), out["steps"][0]["grad_norm"]

    def same(a, b):
        return all(np.array_equal(a[k], b[k]) for k in a)

    got, gnorm = fitted()                                         # the frozen 1.0, as stage 2 calls it
    want, norm = manual(1.0)
    assert same(got, want) and gnorm == norm
    monkeypatch.setattr(PL, "CLIP_NORM", 1e-3)                    # force the clip on: not vacuous
    got, gnorm = fitted()
    want, norm = manual(1e-3)
    raw, _ = manual(None)
    assert gnorm == norm > 1e-3 and same(got, want) and not same(got, raw)


@pytest.mark.parametrize("arm", ["C", "R"])
def test_a_fit_changes_exactly_the_arm_s_surface_and_no_bn_statistic(data, arm):
    made = []
    T, E = data["T"], data["E"]
    out = PL.fit(lambda: made.append(_net()) or made[-1], arm, {**T, "target": T["TR"]}, {**E, "target": E["TR"]},
                 PL.draw_order(48, 12, 8), data["i_e"])
    before, after = PL.snapshot(_net()), PL.snapshot(made[0])
    changed = {k for k in before if not np.array_equal(before[k], after[k])}
    assert set(out["surface"]) == set(PHC.SURFACES[arm][0])
    assert "policy_head.fc.weight" in changed and changed <= set(PHC.SURFACES["C"][0])   # no BN statistic, no encoder
    assert made[0].training is False and len(out["steps"]) == 12


def test_THE_GUARDS_void_a_changed_bn_statistic_or_a_parameter_outside_the_surface():
    before = PL.snapshot(_net())
    surface = PHC.SURFACES["C"][0]
    for key, msg in (("policy_head.bn.running_mean", "BatchNorm running statistic"),
                     ("encoder.conv1.weight", "outside the arm's surface")):
        after = {k: v.copy() for k, v in before.items()}
        after[key] = after[key] + 1
        with pytest.raises(PL.Void, match=msg):
            PL.check_after(before, after, surface)
    PL.check_after(before, {k: v.copy() for k, v in before.items()}, surface)


def _fit_small(data, target=None, i_eval=None):
    T, E = data["T"], data["E"]
    return PL.fit(_net, "C", {**T, "target": T["TC"] if target is None else target}, {**E, "target": E["TC"]},
                  PL.draw_order(48, 12, 8), data["i_e"] if i_eval is None else i_eval)


def test_THE_GUARDS_void_a_gradient_outside_the_surface(data, monkeypatch):
    real = PHC.install

    def leaky(network, arm):
        names = real(network, arm)
        network.value_head.unfreeze()
        return names
    monkeypatch.setattr(PL.PHC, "install", leaky)
    with pytest.raises(PL.Void, match="gradient outside"):
        _fit_small(data)


def test_THE_GUARDS_void_leaving_eval_mode_a_wrong_start_and_a_nonfinite_loss(data, monkeypatch):
    with pytest.raises(PL.Void, match="does not start exactly"):
        _fit_small(data, i_eval=data["i_e"] + 1)
    bad = data["T"]["TC"].copy()
    bad[0, 0] = np.nan
    with pytest.raises(PL.Void, match="nonfinite loss or gradient norm"):
        _fit_small(data, target=bad)
    real, calls = PL.snapshot, []

    def then_train(network):
        out = real(network)
        if not calls:
            network.train()
        calls.append(1)
        return out
    monkeypatch.setattr(PL, "snapshot", then_train)
    with pytest.raises(PL.Void, match="left eval mode"):
        _fit_small(data)


def test_the_temperature_fit_recovers_a_planted_temperature():
    rng = np.random.default_rng(3)
    logits = rng.standard_normal((20, 30))
    mask = np.ones((20, 30), np.float32)
    assert PL.fit_temperature(logits, PL.targets(logits / 2.0, mask), mask) == pytest.approx(2.0, rel=1e-4)


# ─────────────────────────────── §6 the rules ───────────────────────────────

PASS = {"C/TC": 0.010, "R/TC": 0.010, "C/TR": 0.200, "R/TR": 0.050, "T/TR": 0.300}
KL0 = {"TC": 0.400, "TR": 0.400}


@pytest.mark.parametrize("change, rule, holds", [                # boundaries are exact in binary (x, x/2, 4x)
    ({}, None, True),
    ({"R/TC": 0.100}, "F1", True), ({"R/TC": 0.1001}, "F1", False), ({"C/TC": 0.1001, "R/TC": 0.090}, "F1", False),
    ({"R/TR": 0.100}, "F2", True), ({"R/TR": 0.1001}, "F2", False),
    ({"C/TR": 0.100}, "S1", True), ({"C/TR": 0.0999}, "S1", False),
    ({"T/TR": 0.100}, "S2", True), ({"T/TR": 0.0999}, "S2", False),
    ({"C/TC": 0.0599}, "S3", True), ({"C/TC": 0.100, "R/TC": 0.050}, "S3", False),     # exactly delta: not < delta
])
def test_THE_FIVE_RULES_on_hand_computed_numbers(change, rule, holds):
    rules, outcome = PL.verdict({**PASS, **change}, KL0)
    if rule is not None:
        assert rules[rule] is holds
    assert all(v for k, v in rules.items() if k != rule)
    assert outcome == ("QUALIFIED" if holds else "NOT_QUALIFIED")


def test_a_nonfinite_kl_voids_the_verdict():
    with pytest.raises(PL.Void, match="nonfinite KL"):
        PL.verdict({**PASS, "R/TR": float("nan")}, KL0)


# ─────────────────────────────── the record boundary, end to end on stubs ───────────────────────────────

def test_the_whole_path_completes_on_stubs_and_records_every_number(tmp_path, monkeypatch):
    real, seen = PL.fit_temperature, []

    def spy(i_train, p_train, mask):
        seen.append(len(i_train))
        return real(i_train, p_train, mask)
    monkeypatch.setattr(PL, "fit_temperature", spy)
    code, rec = PL.conclude(SMALL, _seams(), tmp_path / "ev", tmp_path / "cache")
    assert code == PL.EXIT_DONE and rec["status"] == "COMPLETED", rec.get("void")
    assert rec["verdict"] in ("QUALIFIED", "NOT_QUALIFIED") and set(rec["rules"]) == {"F1", "F2", "S1", "S2", "S3"}
    assert seen == [SMALL["n_train"]]                                            # temperature: training only
    assert abs(rec["teachers"]["tc_scale"] - 1) <= PL.SCALE_TOL and rec["teachers"]["tr_scale"] == pytest.approx(1)
    assert set(rec["fits"]) == {"C/TC", "R/TC", "C/TR", "R/TR"}
    for key, f in rec["fits"].items():
        assert f["starting_kl"] == rec["kl0"][key.split("/")[1]] and f["final_kl"] == rec["kl"][key]
        assert len(f["grad_norms"]) == SMALL["steps"] and 0 <= f["clipped_fraction"] <= 1
    d = rec["descriptive"]
    assert 0 < d["top2_variance_fraction"] <= 1 and d["chance_baseline"] > math.log(400)
    assert json.loads((tmp_path / "ev" / "record.json").read_text()) == json.loads(json.dumps(rec))
    pos = json.loads((tmp_path / "ev" / "positions.json").read_text())
    assert (len(pos["train"]), len(pos["eval"])) == (48, 12) and set(pos["train"][0]) == {
        "digest", "screen", "game_idx", "ply", "to_move", "n_legal"}
    cache = list((tmp_path / "cache").iterdir())
    assert len(cache) == 1 and PL._sha_file(cache[0]) == rec["cache"]["sha256"]
    assert not any(p.suffix == ".npz" for p in (tmp_path / "ev").iterdir())       # the cache is not evidence
    with pytest.raises(FileExistsError):
        PL.conclude(SMALL, _seams(), tmp_path / "ev", tmp_path / "cache")


def _untrained_net():
    net = _net()
    object.__setattr__(net, "eval", lambda: net)
    net.train()
    return net


def _corrupting_top2(real):
    def top2(x):
        for p in PL.CACHE_PROBE.iterdir():
            p.write_bytes(p.read_bytes() + b"x")
        return real(x)
    return top2


@pytest.mark.parametrize("seams, spec, step", [
    ({"checkpoint_sha1": lambda: "0" * 40}, SMALL, "checkpoint"),
    ({"load_network": lambda: (_net(), CHANNELS + 1)}, SMALL, "network"),
    ({"load_network": lambda: (_untrained_net(), CHANNELS)}, SMALL, "network"),
    ({}, {**SMALL, "games": 11}, "counts"),
    ({}, {**SMALL, "n_eval": 40}, "selection"),
    ({"load_games": lambda: GAMES[:3] + [("screen-a", {**GAMES[3][1], "board_size": 20})] + GAMES[4:]}, SMALL, "replay"),
    ({"features": lambda n, s: _features(n, s)[..., :64]}, SMALL, "features"),
    ({"load_games": lambda: (_ for _ in ()).throw(RuntimeError("boom"))}, SMALL, "exception"),
])
def test_EVERY_VOID_PATH_leaves_a_record(tmp_path, seams, spec, step):
    code, rec = PL.conclude(spec, _seams(**seams), tmp_path / "ev", tmp_path / "cache")
    assert code == PL.EXIT_VOID and rec["status"] == "VOID" and rec["void"]["step"] == step, rec.get("void")
    assert json.loads((tmp_path / "ev" / "record.json").read_text())["void"]["step"] == step


def test_THE_CAP_voids_with_a_record(tmp_path):
    clock = itertools.chain([0.0], itertools.repeat(PL.CAP_S))
    code, rec = PL.conclude(SMALL, _seams(), tmp_path / "ev", tmp_path / "cache", clock=lambda: next(clock))
    assert code == PL.EXIT_VOID and rec["void"]["step"] == "cap"
    assert (tmp_path / "ev" / "record.json").exists()


def test_A_CHANGED_CACHE_voids_with_a_record(tmp_path, monkeypatch):
    monkeypatch.setattr(PL, "CACHE_PROBE", tmp_path / "cache", raising=False)
    monkeypatch.setattr(PL, "top2_fraction", _corrupting_top2(PL.top2_fraction))
    code, rec = PL.conclude(SMALL, _seams(), tmp_path / "ev", tmp_path / "cache")
    assert code == PL.EXIT_VOID and rec["void"]["step"] == "cache", rec.get("void")
