"""Pure replay-aware loss analysis (V2 Phase B) over Phase A capture data.

No IO, no MLX: game rows + replay sidecar dicts in, feature dicts / table
rows / verdict out. The V1 game-level analyzer is untouched; game-row
semantics (scoring, color, validation) live in eval_loss_analysis.

Value-sign convention (confirmed against mcts.py): root_value is negamax,
always from the perspective of the player about to move. A's trajectory uses
A's own plies; B's series is reported in B's own perspective, never merged.

Opening-temperature rule: eval games temperature-sample the first
opening_plies plies, so selected_visit_rank / root_top1_share there reflect
sampling, not confidence. Confidence/diffusion features and rules use
post-opening plies only; value features use all A plies.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean, median, variance

from .eval_elo import score_rate

REPLAY_SCHEMA_VERSION = 1

# Classification constants (spec-locked). The five operator-facing thresholds
# live in Thresholds; these pin down the spec's "many"/"multiple" words.
HEALTHY_START = -0.10        # gradual_decay: the game started healthy
DECAYED_FINAL = -0.40        # gradual_decay: the game ended decayed
DIFFUSE_MEAN_TOP1 = 0.15     # search_diffusion: mean post-opening top1 share
DIFFUSE_PLY_FRACTION = 0.25  # search_diffusion: share of diffuse post plies
LOW_RANK_MEDIAN = 3          # low_visit_selection: median post rank
LOW_RANK_PLY_COUNT = 3       # low_visit_selection: count of low-rank plies
B_ONSET_LOW = 0.25           # B win-onset crossings (B's own perspective)
B_ONSET_HIGH = 0.50
PRIMARY_SHARE = 0.35         # verdict: bar for a primary failure mode
SECONDARY_SHARE = 0.20       # verdict: bar for a secondary signal
MIN_WIN_COHORT = 5           # below this, effect sizes -> insufficient_contrast

COLLAPSE_PRECEDENCE = (
    ("already_bad", "flag_already_bad"),
    ("sharp_value_drop", "flag_sharp"),
    ("gradual_decay", "flag_gradual"),
    ("search_diffusion", "flag_diffusion"),
    ("low_visit_selection", "flag_low_visit"),
)

FAILURE_MODE_GROUPS = {
    "value-drop": ("sharp_value_drop", "gradual_decay"),
    "already-losing": ("already_bad",),
    "diffusion": ("search_diffusion",),
    "low-visit-selection": ("low_visit_selection",),
}

PHASES = ("opening", "early_midgame", "midgame", "late_midgame", "pre_terminal")
MIDGAME_PHASES = PHASES[1:]

CROSSING_KEYS = ("first_a_value_below_0", "first_a_value_below_bad",
                 "first_a_value_below_lost")

REQUIRED_PLY_KEYS = {
    "ply", "player", "row", "col", "root_value", "root_top1_share",
    "selected_visit_rank", "selected_visit_count", "root_total_visits",
    "n_legal",
}

EFFECT_METRICS = ("final_a_value", "largest_a_value_drop", "initial_a_value",
                  "mean_top1_share_post", "median_selected_visit_rank_post")
EFFECT_FORMULA = (
    "cohens_d = (loss_mean - win_mean) / pooled_std(ddof=1); negative d on "
    "value metrics = lower in losses; positive d on visit rank = higher rank "
    "(less confident) in losses")

OPENING_SAMPLING_NOTE = (
    "Plies before opening_plies are temperature-sampled: selected_visit_rank "
    "and root_top1_share there reflect sampling, not confidence. Confidence/"
    "diffusion features and rules use post-opening plies only.")


@dataclass(frozen=True)
class Thresholds:
    bad_value: float = -0.25
    lost_value: float = -0.50
    sharp_drop: float = 0.40
    low_top1_share: float = 0.10
    low_visit_rank: int = 5
    opening_plies: int = 20


def _mean(vals):
    return mean(vals) if vals else None


def _median(vals):
    return median(vals) if vals else None


def side_plies(replay, color):
    """Per-ply records for one side, in game order (spec: a_ply_series)."""
    return [m for m in replay["moves"] if m["player"] == color]


def validate_replay(row, replay):
    """Fail loud if a sidecar contradicts its games.jsonl row."""
    gi = row["game_idx"]
    if replay.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ValueError(
            f"game {gi}: schema_version {replay.get('schema_version')!r} "
            f"!= {REPLAY_SCHEMA_VERSION}")
    for key in ("game_idx", "task_id", "pairing_id", "winner", "winner_checkpoint",
                "reason", "n_moves", "red_checkpoint", "black_checkpoint"):
        if replay.get(key) != row[key]:
            raise ValueError(
                f"game {gi}: replay {key}={replay.get(key)!r} != row {row[key]!r}")
    moves = replay["moves"]
    if len(moves) != row["n_moves"]:
        raise ValueError(
            f"game {gi}: {len(moves)} move records != n_moves {row['n_moves']}")
    for i, m in enumerate(moves):
        missing = REQUIRED_PLY_KEYS - m.keys()
        if missing:
            raise ValueError(f"game {gi} ply {i}: missing keys {sorted(missing)}")
        if m["ply"] != i:
            raise ValueError(f"game {gi} ply {i}: ply field is {m['ply']}")
        expect = "red" if i % 2 == 0 else "black"
        if m["player"] != expect:
            raise ValueError(
                f"game {gi} ply {i}: player {m['player']!r}, expected {expect!r}")


def _crossing(plies, n_moves, pred):
    """First ply where pred(root_value) -> {ply, a_ply, fraction} or None."""
    for i, m in enumerate(plies):
        if pred(m["root_value"]):
            frac = m["ply"] / (n_moves - 1) if n_moves > 1 else 0.0
            return {"ply": m["ply"], "a_ply": i, "fraction": frac}
    return None


def value_features(a_plies, n_moves, th):
    """Value-trajectory features over ALL of A's plies (see module docstring:
    value readings are not temperature-distorted, so the opening is included
    — that is what lets initial_a_value detect already_bad games)."""
    vals = [m["root_value"] for m in a_plies]
    feats = {
        "initial_a_value": _median(vals[:3]),
        "final_a_value": _median(vals[-3:]),
        "mean_a_value": _mean(vals),
        "min_a_value": min(vals) if vals else None,
        "largest_a_value_drop": None,
        "largest_drop_ply": None,
        "largest_drop_a_ply": None,
        "largest_drop_fraction": None,
    }
    if len(vals) >= 2:
        # (delta, index) tuple-min: ties on delta resolve to the earliest ply.
        d, i = min((vals[i] - vals[i - 1], i) for i in range(1, len(vals)))
        ply = a_plies[i]["ply"]
        feats.update(
            largest_a_value_drop=d, largest_drop_ply=ply, largest_drop_a_ply=i,
            largest_drop_fraction=ply / (n_moves - 1) if n_moves > 1 else 0.0)
    for name, thresh in (("first_a_value_below_0", 0.0),
                         ("first_a_value_below_bad", th.bad_value),
                         ("first_a_value_below_lost", th.lost_value)):
        c = _crossing(a_plies, n_moves, lambda v, t=thresh: v <= t)
        feats[f"{name}_ply"] = c["ply"] if c else None
        feats[f"{name}_a_ply"] = c["a_ply"] if c else None
        feats[f"{name}_fraction"] = c["fraction"] if c else None
    return feats


def confidence_features(a_plies, th):
    """Confidence/diffusion features over POST-OPENING A plies only (the
    opening is temperature-sampled — see OPENING_SAMPLING_NOTE). All-null
    when the game has no post-opening A plies."""
    post = [m for m in a_plies if m["ply"] >= th.opening_plies]
    feats = {
        "n_a_plies": len(a_plies),
        "n_a_plies_post": len(post),
        "mean_n_legal": _mean([m["n_legal"] for m in a_plies]),
        "mean_top1_share_post": None,
        "min_top1_share_post": None,
        "median_selected_visit_rank_post": None,
        "max_selected_visit_rank_post": None,
        "mean_selected_visit_share_post": None,
        "low_confidence_ply_count": None,
        "diffuse_ply_fraction": None,
    }
    if post:
        shares = [m["root_top1_share"] for m in post]
        ranks = [m["selected_visit_rank"] for m in post]
        feats.update(
            mean_top1_share_post=mean(shares),
            min_top1_share_post=min(shares),
            median_selected_visit_rank_post=median(ranks),
            max_selected_visit_rank_post=max(ranks),
            mean_selected_visit_share_post=mean(
                [m["selected_visit_count"] / m["root_total_visits"] for m in post]),
            low_confidence_ply_count=sum(r >= th.low_visit_rank for r in ranks),
            diffuse_ply_fraction=(
                sum(s <= th.low_top1_share for s in shares) / len(post)),
        )
    return feats


def opening_key(replay, key_plies):
    """First key_plies moves (both players) as a compact cluster key."""
    return "|".join(f"r{m['row']}c{m['col']}"
                    for m in replay["moves"][:key_plies])


def classify_collapse(f, th):
    """(label, flags) for one game's features. One label via the documented
    precedence; every rule's flag is returned so multi-signal games stay
    visible in the CSVs. Rules with null inputs do not fire."""
    init, fin = f["initial_a_value"], f["final_a_value"]
    drop = f["largest_a_value_drop"]
    sharp = drop is not None and drop <= -th.sharp_drop
    flags = {
        "flag_already_bad": init is not None and init <= th.bad_value,
        "flag_sharp": sharp,
        "flag_gradual": (init is not None and fin is not None
                         and init > HEALTHY_START and fin <= DECAYED_FINAL
                         and not sharp),
        # mean_top1_share_post and diffuse_ply_fraction are co-null (set
        # together in confidence_features), so one guard covers both.
        "flag_diffusion": (
            f["mean_top1_share_post"] is not None
            and (f["mean_top1_share_post"] <= DIFFUSE_MEAN_TOP1
                 or f["diffuse_ply_fraction"] >= DIFFUSE_PLY_FRACTION)),
        # co-null with low_confidence_ply_count, same as above.
        "flag_low_visit": (
            f["median_selected_visit_rank_post"] is not None
            and (f["median_selected_visit_rank_post"] >= LOW_RANK_MEDIAN
                 or f["low_confidence_ply_count"] >= LOW_RANK_PLY_COUNT)),
    }
    label = next((lab for lab, flag in COLLAPSE_PRECEDENCE if flags[flag]),
                 "no_clear_signal")
    return label, flags
