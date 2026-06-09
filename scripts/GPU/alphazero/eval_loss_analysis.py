"""Pure loss-shape analysis over checkpoint-eval *_games.jsonl rows.

No IO, no MLX: rows in (plain dicts), dicts/lists out (ready to serialize).
Scoring matches eval_summary: A/B keyed off winner_checkpoint, color off
red/black_checkpoint, draws (state_cap/board_full) = 0.5 for both sides.
"""
from __future__ import annotations

from statistics import mean

from .eval_elo import (
    score_rate, elo_diff, score_ci_trinomial, elo_ci, verdict,
)
from .eval_runner import short_id

LENGTH_BUCKETS_DEFAULT = (40, 60, 80, 120, 279, 280)

REQUIRED_KEYS = {
    "task_id", "pairing_id", "game_idx", "red_checkpoint", "black_checkpoint",
    "winner", "winner_checkpoint", "reason", "n_moves", "red_score", "black_score",
}
DRAW_REASONS = {"state_cap", "board_full"}
VALID_REASONS = {"win", "state_cap", "board_full", "unknown_error"}


def _require(i, cond, msg):
    if not cond:
        raise ValueError(f"row {i}: {msg}")


def validate_rows(rows, a_ckpt=None, b_ckpt=None):
    """Fail loud on any row that breaks the eval scoring invariants.

    When a_ckpt and b_ckpt are both given, also require every row to be
    between exactly those two checkpoints (catches a mixed/concatenated
    JSONL of more than one pairing).
    """
    if not rows:
        raise ValueError("no rows to analyze")
    ab = {a_ckpt, b_ckpt} if (a_ckpt is not None and b_ckpt is not None) else None
    for i, r in enumerate(rows):
        missing = REQUIRED_KEYS - r.keys()
        _require(i, not missing, f"missing keys {sorted(missing)}")
        reason = r["reason"]
        _require(i, reason in VALID_REASONS, f"bad reason {reason!r}")
        _require(i, reason != "unknown_error",
                 "reason 'unknown_error' not handled in V1 (none expected in current data)")
        winner = r["winner"]
        _require(i, winner in ("red", "black", None), f"bad winner {winner!r}")
        red, black = r["red_checkpoint"], r["black_checkpoint"]
        if winner == "red":
            _require(i, r["winner_checkpoint"] == red, "winner_checkpoint != red_checkpoint")
            _require(i, r["red_score"] == 1.0 and r["black_score"] == 0.0,
                     "red-win scores not 1.0/0.0")
        elif winner == "black":
            _require(i, r["winner_checkpoint"] == black, "winner_checkpoint != black_checkpoint")
            _require(i, r["red_score"] == 0.0 and r["black_score"] == 1.0,
                     "black-win scores not 0.0/1.0")
        else:  # draw
            _require(i, r["winner_checkpoint"] is None, "draw winner_checkpoint not None")
            _require(i, r["red_score"] == 0.5 and r["black_score"] == 0.5,
                     "draw scores not 0.5/0.5")
            _require(i, reason in DRAW_REASONS, f"draw reason {reason!r} not a draw reason")
        if ab is not None and {red, black} != ab:
            _require(i, False,
                     f"checkpoints {{{short_id(red)}, {short_id(black)}}} != resolved A/B "
                     "— mixed JSONL?")


def score_for_checkpoint(row, ckpt):
    """1.0 if ckpt won, 0.5 on a draw (no winner), else 0.0. Keyed off
    winner_checkpoint — never off color."""
    if row["winner_checkpoint"] == ckpt:
        return 1.0
    if row["winner_checkpoint"] is None:
        return 0.5
    return 0.0


def a_color(row, a_ckpt):
    """Which seat A played this game: 'red' or 'black'."""
    if row["red_checkpoint"] == a_ckpt:
        return "red"
    if row["black_checkpoint"] == a_ckpt:
        return "black"
    raise ValueError(f"A checkpoint {short_id(a_ckpt)} not in row {row['game_idx']}")


def _match_short(ckpts, sid):
    hits = [c for c in ckpts if short_id(c) == sid]
    if len(hits) != 1:
        raise ValueError(f"pairing side {sid!r} matched {len(hits)} checkpoints, expected 1")
    return hits[0]


def _infer_from_pairing(rows, pairing_id):
    pid = pairing_id or rows[0]["pairing_id"]
    if "_vs_" not in pid:
        raise ValueError(f"cannot infer A/B: pairing_id {pid!r} has no '_vs_'")
    a_id, b_id = pid.split("_vs_", 1)
    ckpts = ({r["red_checkpoint"] for r in rows}
             | {r["black_checkpoint"] for r in rows})
    return _match_short(ckpts, a_id), _match_short(ckpts, b_id)


def resolve_checkpoints(rows, pairing_id=None, a_override=None,
                        b_override=None, summary=None):
    """Resolve (A, B) checkpoint paths.

    Precedence: explicit overrides -> sidecar summary checkpoint_a/checkpoint_b
    -> infer from pairing_id + short_id of the row checkpoints. Both resolved
    paths must actually appear across the rows.
    """
    if a_override and b_override:
        a, b = a_override, b_override
    elif summary and summary.get("checkpoint_a") and summary.get("checkpoint_b"):
        a, b = summary["checkpoint_a"], summary["checkpoint_b"]
    else:
        a, b = _infer_from_pairing(rows, pairing_id)
    present = ({r["red_checkpoint"] for r in rows}
               | {r["black_checkpoint"] for r in rows})
    for label, ckpt in (("A", a), ("B", b)):
        if ckpt not in present:
            raise ValueError(f"resolved {label} checkpoint {ckpt!r} not present in rows")
    return a, b


def _ab_stats(rows, a_ckpt):
    """Games / A-wins / B-wins / draws / A-score-rate / avg moves for a subset."""
    n = len(rows)
    a_wins = sum(1 for r in rows if r["winner_checkpoint"] == a_ckpt)
    draws = sum(1 for r in rows if r["winner"] is None)
    b_wins = n - a_wins - draws
    return {
        "games": n,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_score_rate": (score_rate(a_wins, draws, n) if n else None),
        "avg_moves": (round(mean(r["n_moves"] for r in rows), 2) if n else None),
    }


def _bucket_name(edge, buckets):
    idx = buckets.index(edge)
    lo = (buckets[idx - 1] + 1) if idx > 0 else None
    if lo is None:
        return f"<={edge}"
    if lo == edge:
        return f"{edge}"
    return f"{lo}-{edge}"


def _length_bucket_label(n_moves, buckets):
    for edge in buckets:
        if n_moves <= edge:
            return _bucket_name(edge, buckets)
    return f">{buckets[-1]}"


def summarize_by_color(rows, a_ckpt, b_ckpt):
    out = []
    for color in ("red", "black"):
        sub = [r for r in rows if a_color(r, a_ckpt) == color]
        out.append({"a_color": color, **_ab_stats(sub, a_ckpt)})
    return out


def summarize_by_length(rows, a_ckpt, b_ckpt, buckets=LENGTH_BUCKETS_DEFAULT):
    ordered_labels = [_bucket_name(e, buckets) for e in buckets]
    groups = {}
    for r in rows:
        groups.setdefault(_length_bucket_label(r["n_moves"], buckets), []).append(r)
    # bucket order first, then any overflow label (e.g. ">280") last
    labels = ordered_labels + [k for k in groups if k not in ordered_labels]
    out = []
    for lbl in labels:
        sub = groups.get(lbl)
        if sub:  # omit empty buckets
            out.append({"length_bucket": lbl, **_ab_stats(sub, a_ckpt)})
    return out


def summarize_overall(rows, a_ckpt, b_ckpt):
    n = len(rows)
    a_wins = sum(1 for r in rows if r["winner_checkpoint"] == a_ckpt)
    b_wins = sum(1 for r in rows if r["winner_checkpoint"] == b_ckpt)
    wins = sum(1 for r in rows if r["reason"] == "win")
    state_caps = sum(1 for r in rows if r["reason"] == "state_cap")
    board_full = sum(1 for r in rows if r["reason"] == "board_full")
    draws = state_caps + board_full
    rate = score_rate(a_wins, draws, n)
    s_lo, s_hi = score_ci_trinomial(a_wins, draws, b_wins)
    e_lo, e_hi = elo_ci(a_wins, draws, b_wins)

    by_color = summarize_by_color(rows, a_ckpt, b_ckpt)
    rates = {c["a_color"]: c["a_score_rate"] for c in by_color}
    red_rate, black_rate = rates.get("red"), rates.get("black")
    color_gap = (red_rate - black_rate
                 if red_rate is not None and black_rate is not None else None)

    return {
        "games": n,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_score": a_wins + 0.5 * draws,
        "a_score_rate": rate,
        "elo": elo_diff(rate, n),
        "elo_ci95": [e_lo, e_hi],
        "score_rate_ci95": [s_lo, s_hi],
        "verdict": verdict(rate),
        "color_gap": color_gap,
        "termination": {
            "win": wins,
            "state_cap": state_caps,
            "board_full": board_full,
            "unknown_error": 0,
            "draws": draws,
            "state_cap_rate": state_caps / n,
            "board_full_rate": board_full / n,
        },
    }


def _worst_row(r, bucket, a_ckpt):
    return {
        "loss_bucket": bucket,
        "game_idx": r["game_idx"],
        "task_id": r["task_id"],
        "a_color": a_color(r, a_ckpt),
        "winner": r["winner"],
        "reason": r["reason"],
        "n_moves": r["n_moves"],
        "a_score": score_for_checkpoint(r, a_ckpt),
        "red_checkpoint": r["red_checkpoint"],
        "black_checkpoint": r["black_checkpoint"],
    }


def sample_worst_losses(rows, a_ckpt, b_ckpt, limit=50):
    """Up to `limit` rows per bucket: A's shortest decisive losses
    (short_loss), A's longest decisive losses (long_loss), and the
    non-decisive cap/board-full games (draw_cap). short_loss and long_loss
    draw from the same A-loss pool, so they overlap when losses are few."""
    a_losses = [r for r in rows if score_for_checkpoint(r, a_ckpt) == 0.0]
    caps = [r for r in rows if r["winner"] is None]
    short = sorted(a_losses, key=lambda r: r["n_moves"])[:limit]
    long = sorted(a_losses, key=lambda r: -r["n_moves"])[:limit]
    out = []
    for bucket, group in (("short_loss", short), ("long_loss", long),
                          ("draw_cap", caps[:limit])):
        out.extend(_worst_row(r, bucket, a_ckpt) for r in group)
    return out


def analyze_match(rows, a_ckpt, b_ckpt, *, match=None, pairing_id=None,
                  length_buckets=LENGTH_BUCKETS_DEFAULT):
    """Full per-match summary dict (the loss_summary.json payload). The
    worst-loss CSV is produced separately via sample_worst_losses()."""
    overall = summarize_overall(rows, a_ckpt, b_ckpt)
    return {
        "match": match,
        "pairing_id": pairing_id or rows[0]["pairing_id"],
        "a_checkpoint": a_ckpt,
        "b_checkpoint": b_ckpt,
        **overall,
        "by_color": summarize_by_color(rows, a_ckpt, b_ckpt),
        "by_length": summarize_by_length(rows, a_ckpt, b_ckpt, length_buckets),
    }


def combine_branch_summaries(match_summaries):
    """One row per match, sorted descending by a_score_rate (strongest
    branch-vs-anchor first)."""
    rows = [
        {
            "match": s["match"],
            "pairing_id": s["pairing_id"],
            "a_checkpoint": s["a_checkpoint"],
            "b_checkpoint": s["b_checkpoint"],
            "games": s["games"],
            "a_score_rate": s["a_score_rate"],
            "a_wins": s["a_wins"],
            "b_wins": s["b_wins"],
            "draws": s["draws"],
            "elo": s["elo"],
            "verdict": s["verdict"],
        }
        for s in match_summaries
    ]
    # None a_score_rate (a zero-game summary) sorts to the bottom rather than
    # raising on the None-vs-float comparison.
    rows.sort(key=lambda r: (r["a_score_rate"] is not None, r["a_score_rate"] or 0.0),
              reverse=True)
    return rows
