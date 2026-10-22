"""The pure capture module (spec 2A checkpoint, extended to the minimal serial path): scripted snapshots of one
fixed 24x24 history, published as a phase and reopened strictly. No engine-generated history, search, model, MLX,
worker or trainer."""
import hashlib
import json
import pathlib
import random
from fractions import Fraction

import numpy as np
import pytest

from scripts.GPU.alphazero import training_replay_capture as C
from scripts.GPU.alphazero.game.twixt_state import TwixtState

ROOT = pathlib.Path(__file__).resolve().parents[1]
RED_CHAIN = [(r, 5 if (r // 2) % 2 == 0 else 6) for r in range(0, 23, 2)]
BLACK_FAR = [(r, 18) for r in range(1, 23, 2)] + [(2, 20)]
HISTORY = [m for pair in zip(RED_CHAIN, BLACK_FAR) for m in pair] + [(23, 4)]
OWN_CROSS = [(5, 5), (10, 10), (7, 6), (10, 12), (5, 6), (12, 10), (7, 5)]
ADJUSTED = 10
TENSORS_A = {"enc.conv.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
             "enc.bn.running_mean": np.zeros(2, np.float32), "enc.bn.running_var": np.ones(2, np.float32),
             "policy_head.fc.bias": np.array([0.5], np.float32)}
TENSORS_B = {**TENSORS_A, "enc.bn.running_var": np.full(2, 2.0, np.float32)}


def _state(moves=()):
    s = TwixtState(active_size=24, to_move="red", max_plies_limit=280)
    for m in moves:
        s = s.apply_move(m)
    return s


def _context():
    files = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
             for p in ("scripts/GPU/alphazero/training_replay_capture.py", "scripts/GPU/alphazero/game/twixt_state.py")}
    return {"schema": C.SCHEMA, "run_id": "fixture:run", "phase_id": "fixture:checkpoint-1",
            "source_kind": "native_self_play", "scheduled_game_ids": [0],
            "games": {"0": {"family_id": "fixture:f0", "start_player": "red", "learner_player": "red",
                            "models_by_colour": {"red": "fixture:model-A", "black": "fixture:model-B"}}},
            "rules": {"board_size": 24, "active_size": 24, "automatic_links": True, "link_removal": False,
                      "own_crossing_forbidden": True, "opponent_crossing_forbidden": True, "swap": False,
                      "max_plies": 280, "start": "empty",
                      "rules_source_sha256": files["scripts/GPU/alphazero/game/twixt_state.py"]},
            "settings": {"selection_adjustment": "closeout_tiebreak", "mirror_prob": 0.0, "position_cap": None},
            "configuration": {"n_simulations": 4},
            "source_identity": {"commit": "fixture:uncommitted-working-tree", "files": files},
            "models": {"fixture:model-A": {"state_digest": C.model_state_digest(TENSORS_A),
                                           "digest_version": C.MODEL_DIGEST_VERSION,
                                           "encoder": {"function": "TwixtState.to_tensor", "in_channels": 30}},
                       "fixture:model-B": {"state_digest": C.model_state_digest(TENSORS_B),
                                           "digest_version": C.MODEL_DIGEST_VERSION,
                                           "encoder": {"function": "TwixtState.to_tensor", "in_channels": 30}}},
            "limits": {"game_bytes": 1 << 20, "phase_bytes": 2 << 20}}


def write_phase(capture_dir, ctx, envelopes):
    """Publish a whole phase through the real writer; the live digests are the context's own."""
    w = C.PhaseWriter(capture_dir, ctx)
    for env in envelopes:
        w.publish_game(env)
    return w.finalize(C.context_model_digests(ctx))


def _scripted(i, legal, move):
    alt = next(m for m in legal if m != move)
    if i == ADJUSTED:
        return {move: 1}, {alt: 5, move: 3}, alt
    return ({move: 1} if i >= 2 else {}), {move: 3 + i % 3, alt: 1}, alt


def _record(rng=None, sources=None, game_ctx=None):
    ctx, sha, gid = game_ctx or (_context(), None, 0)
    sha = sha or C.sha256_hex(C.canonical_bytes(ctx))
    cap = C.DecisionCapture(context=ctx, context_sha256=sha, game_id=gid, game_rng=rng or random.Random(3))
    s, kept = _state(), []
    for i, move in enumerate(HISTORY):
        legal = s.legal_moves()
        before, after_sparse, alt = _scripted(i, legal, move)
        after = {m: after_sparse.get(m, 0) for m in legal}
        cap.begin_decision(state=s, child_visits_before=before, root_visits_before=sum(before.values()))
        cap.after_search(visit_counts=after, root_visits_after=sum(before.values()) + 4, root_value=0.1 * (i % 5))
        if s.to_move == "red":
            obj = object(); cap.note_training_row(obj); kept.append(obj)
        sel, adj = after, None
        if i == ADJUSTED:
            sel = dict(after); sel[move] = max(after.values()) + 1
            adj = {"kind": "closeout_tiebreak", "overrode_to": "endpoint", "target_move": list(move)}
        cap.before_selection(selection_counts=sel, adjustment=adj)
        cap.after_selection(move=move)
        if sources is not None:
            sources.append((before, after, sel))
        s = s.apply_move(move)
    env = cap.finish(move_history=HISTORY, start_player="red", final_state=s, kept_positions=kept)
    return ctx, sha, env, s


def test_fixtures_are_a_completed_game_and_a_blocked_own_crossing():
    final = _state(HISTORY)
    assert (final.ply, final.winner(), len(HISTORY) <= C.MAX_ROWS) == (25, "red", True)
    assert _state(OWN_CROSS).bridges == {((5, 5), (7, 6))}


def test_round_trip_keeps_exact_targets_zeros_inheritance_and_the_rational_policy(tmp_path):
    ctx, sha, env, _ = _record()
    stats = write_phase(str(tmp_path / "cap"), ctx, [env])
    ctx2, (env2,) = C.load_phase(str(tmp_path / "cap"))
    assert ctx2 == ctx and env2 == json.loads(C.canonical_bytes(env))
    row = env2["rows"][ADJUSTED]
    legal = [tuple(m) for m in row["legal_moves"]]
    move, alt = HISTORY[ADJUSTED], next(m for m in legal if m != HISTORY[ADJUSTED])
    expected = [Fraction(5, 8) if m == alt else Fraction(3, 8) if m == move else Fraction(0) for m in legal]
    assert C.policy_target(row) == expected                                   # unchanged by the adjustment
    assert row["child_visits_before"][legal.index(move)] == 1                 # inherited visits kept
    assert row["child_visits_after"].count(0) == len(legal) - 2               # every zero kept
    assert stats["bytes"] <= ctx["limits"]["phase_bytes"] and stats["games"] == 1
    print("checkpoint-measurements", json.dumps(stats))                      # prototype numbers, not throughput


@pytest.mark.parametrize("bad", ["extra_change", "wrong_value", "unknown_kind", "no_record"])
def test_the_adjustment_must_explain_the_selection_change(bad):
    ctx, sha, env, _ = _record()
    row = env["rows"][ADJUSTED]
    if bad == "extra_change":  row["selection_counts"][0] += 1
    if bad == "wrong_value":   row["selection_counts"][row["legal_moves"].index(list(HISTORY[ADJUSTED]))] = 50
    if bad == "unknown_kind":  row["selection_adjustment"]["kind"] = "anything"
    if bad == "no_record":     row["selection_adjustment"] = None
    with pytest.raises(C.CaptureError, match="adjust|selection"):
        C.validate_envelope(C.canonical_bytes(env), ctx, sha)


def test_history_rules_and_attribution_are_reconstructed():
    ctx, sha, env, final = _record()
    out = C.validate_envelope(C.canonical_bytes(env), ctx, sha)
    assert out["outcome"] == {"winner": "red", "reason": "win", "final_ply": 25,
                              "final_state_sha256": C.state_digest(final)}
    assert {r["model_id"] for r in out["rows"] if r["to_move"] == "black"} == {"fixture:model-B"}
    assert [m["source_row_id"] for m in out["buffer_membership"]] == [f"0:{i}" for i in range(0, 25, 2)]


def test_sources_are_copied_and_capture_draws_no_rng():
    rng, sources = random.Random(5), []
    before_state = rng.getstate()
    ctx, sha, env, _ = _record(rng=rng, sources=sources)
    assert rng.getstate() == before_state
    snapshot = C.canonical_bytes(env)
    for before, after, sel in sources:
        before.clear(); after.clear(); sel.clear()
    assert C.canonical_bytes(env) == snapshot


@pytest.mark.parametrize("bad", ["missing_row", "swapped_models", "context_hash", "membership_opponent"])
def test_missing_rows_or_mismatched_identity_fail_and_leave_no_marker(tmp_path, bad):
    ctx, sha, env, _ = _record()
    if bad == "missing_row":         env["rows"].pop(7)
    if bad == "swapped_models":      env["models_by_colour"] = {"red": "fixture:model-B", "black": "fixture:model-A"}
    if bad == "context_hash":        env["context_sha256"] = "0" * 64
    if bad == "membership_opponent": env["buffer_membership"].append({"source_row_id": "0:1", "transform": "identity"})
    with pytest.raises(C.CaptureError):
        write_phase(str(tmp_path / "cap"), ctx, [env])
    assert not (tmp_path / "cap" / "finalized.json").exists()
    assert not (tmp_path / "cap" / "games" / "game_0.json").exists()          # validated BEFORE publication


def test_existing_output_is_refused(tmp_path):
    ctx, sha, env, _ = _record()
    write_phase(str(tmp_path / "cap"), ctx, [env])
    with pytest.raises(C.CaptureError, match="exists"):
        write_phase(str(tmp_path / "cap"), ctx, [env])
    with pytest.raises(C.CaptureError, match="exists"):
        C._publish(str(tmp_path / "cap"), "games/game_0.json", b"x")


def test_a_byte_ceiling_breach_stops_before_the_marker(tmp_path):
    ctx, sha, env, _ = _record()
    ctx["limits"]["phase_bytes"] = 2000
    with pytest.raises(C.CaptureError, match="exceed"):
        write_phase(str(tmp_path / "cap"), ctx, [env])
    assert not (tmp_path / "cap" / "finalized.json").exists()


def test_model_identity_covers_parameters_and_bn_buffers():
    assert C.model_state_digest(TENSORS_A) != C.model_state_digest(TENSORS_B)   # differ only in a BN buffer
    assert C.model_state_digest(dict(reversed(list(TENSORS_A.items())))) == C.model_state_digest(TENSORS_A)


def test_every_row_matches_the_original_scripted_inputs(tmp_path):
    """Independent of the recorder: re-derive each row's legal order and scripted before/after/selection
    counts from the fixture definition and the rules engine, and compare with the REOPENED artifact."""
    ctx, sha, env, _ = _record()
    write_phase(str(tmp_path / "cap"), ctx, [env])
    _, (reopened,) = C.load_phase(str(tmp_path / "cap"))
    s = _state()
    for i, move in enumerate(HISTORY):
        legal = s.legal_moves()
        before, after_sparse, _alt = _scripted(i, legal, move)
        exp_before = [before.get(m, 0) for m in legal]
        exp_after = [after_sparse.get(m, 0) for m in legal]
        exp_sel = list(exp_after)
        if i == ADJUSTED:
            exp_sel[legal.index(move)] = max(exp_after) + 1
        row = reopened["rows"][i]
        assert row["legal_moves"] == [list(m) for m in legal], i
        assert row["child_visits_before"] == exp_before, i
        assert row["child_visits_after"] == exp_after, i
        assert row["selection_counts"] == exp_sel, i
        s = s.apply_move(move)
    assert len(reopened["rows"]) == len(HISTORY)



# ─────────────────────────────── the serial phase: several games, model binding, strict reopen ───────────────────

def _two_games():
    """Games 0 and 1: the same fixed history (game 1 a second, separately identified envelope)."""
    ctx = _context()
    ctx["scheduled_game_ids"] = [0, 1]
    ctx["games"]["1"] = dict(ctx["games"]["0"])
    sha = C.sha256_hex(C.canonical_bytes(ctx))
    envs = []
    for gid in (0, 1):
        _, _, env, _ = _record(game_ctx=(ctx, sha, gid))
        envs.append(env)
    return ctx, envs


def test_a_two_game_phase_round_trips_in_schedule_order(tmp_path):
    ctx, envs = _two_games()
    stats = write_phase(str(tmp_path / "cap"), ctx, envs[::-1])                  # published out of order
    ctx2, out = C.load_phase(str(tmp_path / "cap"))
    assert [e["game_id"] for e in out] == [0, 1] and stats["games"] == 2
    assert out == [json.loads(C.canonical_bytes(e)) for e in envs]
    receipts = (tmp_path / "cap" / "receipts.jsonl").read_text().splitlines()
    assert [json.loads(r)["game_id"] for r in receipts] == [0, 1]


def test_a_missing_game_or_a_changed_model_withholds_the_marker(tmp_path):
    ctx, envs = _two_games()
    w = C.PhaseWriter(str(tmp_path / "a"), ctx)
    w.publish_game(envs[0])
    with pytest.raises(C.CaptureError, match="without a published envelope"):
        w.finalize(C.context_model_digests(ctx))
    w.publish_game(envs[1])
    changed = {**C.context_model_digests(ctx), "fixture:model-A": C.model_state_digest(TENSORS_B)}
    with pytest.raises(C.CaptureError, match="no longer matches"):
        w.finalize(changed)
    assert not (tmp_path / "a" / "finalized.json").exists()
    with pytest.raises(C.CaptureError, match="missing"):
        C.load_phase(str(tmp_path / "a"))                                        # incomplete phase: no reader
    with pytest.raises(C.CaptureError, match="already published"):
        w.publish_game(envs[1])


@pytest.mark.parametrize("bad", ["stray_file", "staging_left", "receipt_edit", "game_bytes_edit"])
def test_the_strict_reader_refuses_a_tampered_phase(tmp_path, bad):
    ctx, envs = _two_games()
    d = tmp_path / "cap"
    write_phase(str(d), ctx, envs)
    if bad == "stray_file":
        (d / "games" / "game_7.json").write_text("{}")
    if bad == "staging_left":
        (d / ".staging-x").write_text("")
    if bad == "receipt_edit":
        r = d / "receipts.jsonl"
        r.chmod(0o644); r.write_text(r.read_text().replace('"rows":25', '"rows":24', 1))
    if bad == "game_bytes_edit":
        g = d / "games" / "game_1.json"
        g.chmod(0o644); g.write_bytes(g.read_bytes().replace(b'"root_value":0.1', b'"root_value":0.2', 1))
    with pytest.raises(C.CaptureError):
        C.load_phase(str(d))


def test_a_root_visit_gain_other_than_the_configured_simulations_is_refused():
    ctx, sha, env, _ = _record()
    env["rows"][3]["root_visits_after"] += 1
    with pytest.raises(C.CaptureError, match="visit gain"):
        C.validate_envelope(C.canonical_bytes(env), ctx, sha)
    cap = C.DecisionCapture(context=ctx, context_sha256=sha, game_id=0, game_rng=random.Random(1))
    s = _state()
    cap.begin_decision(state=s, child_visits_before={}, root_visits_before=0)
    with pytest.raises(C.CaptureError, match="configured 4"):
        cap.after_search(visit_counts={m: (5 if i == 0 else 0) for i, m in enumerate(s.legal_moves())},
                         root_visits_after=5, root_value=0.0)


@pytest.mark.parametrize("field, value", [("source_kind", "fixture:scripted"), ("configuration", {}),
                                          ("limits", {"game_bytes": 9 << 20, "phase_bytes": 1}),
                                          ("scheduled_game_ids", [0, 0])])
def test_an_incomplete_context_is_refused_before_anything_is_written(tmp_path, field, value):
    ctx = _context()
    ctx[field] = value
    with pytest.raises(C.CaptureError, match="context"):
        C.PhaseWriter(str(tmp_path / "cap"), ctx)
    assert not (tmp_path / "cap").exists()
