from collections import Counter
from scripts.GPU.alphazero.build_v16a_neutral_position_manifest import (
    side_to_move_for_ply, bucket_for_ply, candidate_positions, sample_bucket,
    sample_neutral_rows, NEUTRAL_FIELDNAMES)


def _games(specs):
    return [{"game_idx": i, "n_moves": n, "winner": w,
             "replay_path": f"replays/game_{i:06d}.json"} for i, n, w in specs]


def test_side_parity_and_buckets():
    assert side_to_move_for_ply(0) == "red" and side_to_move_for_ply(91) == "black"
    assert bucket_for_ply(0) is None and bucket_for_ply(1) == "opening"
    assert bucket_for_ply(15) == "opening" and bucket_for_ply(16) == "early_mid"
    assert bucket_for_ply(90) == "midgame" and bucket_for_ply(91) == "late"


def test_candidate_positions_range():
    c = candidate_positions(_games([(1, 45, "red")]))
    assert c["midgame"][1][-1] == 44 and 1 not in c["late"]


def test_round_robin_covers_games_first():
    pool = {g: list(range(16, 41)) for g in range(10)}
    sel, _ = sample_bucket(pool, quota=10, cap=2, min_gap=8, seed=1)
    assert len({g for g, _ in sel}) == 10


def test_min_gap_and_cap():
    sel, _ = sample_bucket({1: list(range(16, 41))}, quota=5, cap=2, min_gap=8, seed=1)
    plies = sorted(p for _, p in sel)
    assert len(sel) == 2 and plies[1] - plies[0] >= 8


def test_side_balance():
    sel, sc = sample_bucket({g: list(range(16, 41)) for g in range(100)},
                            quota=100, cap=2, min_gap=8, seed=2)
    assert abs(sc["red"] - sc["black"]) <= 2


def test_prefix_dedup_via_injected_key():
    sel, _ = sample_bucket({g: list(range(1, 16)) for g in range(20)},
                           quota=20, cap=1, min_gap=0, seed=3,
                           state_key_fn=lambda g, p: g % 3)
    assert len(sel) == 3


def test_deterministic_same_seed():
    recs = _games([(i, 60, "red" if i % 2 else "black") for i in range(200)])
    assert (sample_neutral_rows(recs, base_seed=42, source_replay="s")
            == sample_neutral_rows(recs, base_seed=42, source_replay="s"))


def test_no_dup_quota_and_shortfall():
    recs = _games([(i, 60, "red" if i % 2 else "black") for i in range(300)])
    rows, stats = sample_neutral_rows(recs, base_seed=1, source_replay="s")
    keys = [(r["game_idx"], r["position_ply"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert stats["early_mid"]["achieved"] == 100 and stats["midgame"]["achieved"] == 100
    assert stats["late"]["achieved"] == 0 and stats["late"]["requested"] == 100


def test_row_schema_no_a_labels_and_result_passthrough():
    recs = _games([(i, 60, "red" if i % 2 else "black") for i in range(200)]
                  + [(999, 280, "unknown")])
    rows, _ = sample_neutral_rows(recs, base_seed=5, source_replay="src")
    r = rows[0]
    assert set(r.keys()) == set(NEUTRAL_FIELDNAMES)
    assert not any(k in r for k in ("drop_ply", "largest_a_value_drop", "case_rank"))
    assert any(x["game_result"] == "unknown" for x in rows)     # null-winner passthrough
