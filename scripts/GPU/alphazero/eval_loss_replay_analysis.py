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
