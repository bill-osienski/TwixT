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


def game_features(row, replay, a_clr, th, key_plies=4):
    """One flat per-game feature dict: identity + value + confidence."""
    a_plies = side_plies(replay, a_clr)
    feats = {
        "game_idx": row["game_idx"], "task_id": row["task_id"],
        "replay_path": row.get("replay_path"), "a_color": a_clr,
        "winner": row["winner"], "n_moves": row["n_moves"],
        "opening_key": opening_key(replay, key_plies),
    }
    feats.update(value_features(a_plies, row["n_moves"], th))
    feats.update(confidence_features(a_plies, th))
    return feats


def b_side_features(replay, b_clr, th, a_first_below_lost_fraction):
    """B's series inside one (lost) game, in B's OWN perspective — kept
    separate from A's series, never sign-flipped or merged."""
    n_moves = replay["n_moves"]
    b_plies = side_plies(replay, b_clr)
    post = [m for m in b_plies if m["ply"] >= th.opening_plies]
    feats = {
        "b_mean_value": _mean([m["root_value"] for m in b_plies]),
        "b_mean_top1_share_post": _mean([m["root_top1_share"] for m in post]),
        "b_median_visit_rank_post": _median(
            [m["selected_visit_rank"] for m in post]),
    }
    for name, t in (("b_first_value_above_025", B_ONSET_LOW),
                    ("b_first_value_above_050", B_ONSET_HIGH)):
        c = _crossing(b_plies, n_moves, lambda v, t=t: v >= t)
        feats[f"{name}_ply"] = c["ply"] if c else None
        feats[f"{name}_fraction"] = c["fraction"] if c else None
    bf = feats["b_first_value_above_050_fraction"]
    feats["b_saw_it_first"] = (bf is not None
                               and a_first_below_lost_fraction is not None
                               and bf < a_first_below_lost_fraction)
    return feats


def cohort_comparison_row(cohort, a_plies_per_game, opening_plies):
    """Ply-pooled aggregates for one cohort (the cohort_comparison.csv row)."""
    plies = [m for g in a_plies_per_game for m in g]
    post = [m for m in plies if m["ply"] >= opening_plies]
    return {
        "cohort": cohort,
        "games": len(a_plies_per_game),
        "plies": len(plies),
        "mean_root_value": _mean([m["root_value"] for m in plies]),
        "median_root_value": _median([m["root_value"] for m in plies]),
        "mean_top1_share_post": _mean([m["root_top1_share"] for m in post]),
        "median_top1_share_post": _median([m["root_top1_share"] for m in post]),
        "mean_selected_visit_rank_post": _mean(
            [m["selected_visit_rank"] for m in post]),
        "median_selected_visit_rank_post": _median(
            [m["selected_visit_rank"] for m in post]),
        "mean_selected_visit_share_post": _mean(
            [m["selected_visit_count"] / m["root_total_visits"] for m in post]),
        "mean_n_legal": _mean([m["n_legal"] for m in plies]),
    }


def phase_of(ply, n_moves, opening_plies):
    """opening = absolute temp-sampled window; the rest splits into four
    equal game-fraction bands."""
    if ply < opening_plies:
        return "opening"
    f = (ply - opening_plies) / (n_moves - opening_plies)
    return MIDGAME_PHASES[min(3, int(f * 4))]


def phase_bucket_rows(cohort, games, opening_plies):
    """games: list of (a_plies, n_moves). Empty phases are omitted."""
    plies_by = {p: [] for p in PHASES}
    games_by = {p: set() for p in PHASES}
    for gi, (a_plies, n_moves) in enumerate(games):
        for m in a_plies:
            p = phase_of(m["ply"], n_moves, opening_plies)
            plies_by[p].append(m)
            games_by[p].add(gi)
    rows = []
    for p in PHASES:
        ms = plies_by[p]
        if not ms:
            continue
        rows.append({
            "cohort": cohort, "phase": p,
            "sampling": "temperature" if p == "opening" else "argmax",
            "games": len(games_by[p]), "plies": len(ms),
            "mean_root_value": _mean([m["root_value"] for m in ms]),
            "median_root_value": _median([m["root_value"] for m in ms]),
            "mean_top1_share": _mean([m["root_top1_share"] for m in ms]),
            "median_top1_share": _median([m["root_top1_share"] for m in ms]),
            "mean_selected_visit_rank": _mean(
                [m["selected_visit_rank"] for m in ms]),
            "median_selected_visit_rank": _median(
                [m["selected_visit_rank"] for m in ms]),
        })
    return rows


def cohens_d(xs, ys):
    """Cohen's d with pooled sample std (ddof=1); None when either side has
    < 2 samples or the pooled variance is zero (degenerate)."""
    if len(xs) < 2 or len(ys) < 2:
        return None
    pooled = sqrt(((len(xs) - 1) * variance(xs) + (len(ys) - 1) * variance(ys))
                  / (len(xs) + len(ys) - 2))
    if pooled == 0:
        return None
    return (mean(xs) - mean(ys)) / pooled


def effect_sizes(loss_feats, win_feats):
    """Loss-vs-win effect sizes per EFFECT_METRICS. Sign convention is fixed
    by EFFECT_FORMULA: d = (loss - win) / pooled_std."""
    metrics = {}
    for name in EFFECT_METRICS:
        xs = [f[name] for f in loss_feats if f[name] is not None]
        ys = [f[name] for f in win_feats if f[name] is not None]
        lm, wm = _mean(xs), _mean(ys)
        metrics[name] = {
            "loss_mean": lm, "win_mean": wm,
            "delta": (lm - wm) if lm is not None and wm is not None else None,
            "d": cohens_d(xs, ys),
        }
    return {"formula": EFFECT_FORMULA, "metrics": metrics}


def collapse_distribution(labels):
    """Counts per collapse label + shares per failure-mode group."""
    n = len(labels)
    counts = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    mode_shares = {mode: sum(counts.get(l, 0) for l in group) / n
                   for mode, group in FAILURE_MODE_GROUPS.items()}
    mode_shares["unexplained"] = counts.get("no_clear_signal", 0) / n
    return {"n": n, "counts": counts, "mode_shares": mode_shares}


def make_verdict(labels, cohort_desc):
    """Deterministic verdict from the loss-cohort collapse labels.

    Primary = the failure-mode group with the largest share, if it reaches
    PRIMARY_SHARE and is not beaten by the unexplained share (a tie goes to
    the explained mode). Secondary = next group at SECONDARY_SHARE+.
    """
    dist = collapse_distribution(labels)
    shares = dist["mode_shares"]
    modes = [(m, s) for m, s in shares.items() if m != "unexplained"]
    modes.sort(key=lambda kv: -kv[1])   # FAILURE_MODE_GROUPS order breaks ties
    top_mode, top_share = modes[0]
    unexplained = shares["unexplained"]
    base = {"mode_shares": shares, "primary_share": top_share}
    if top_share < PRIMARY_SHARE or unexplained > top_share:
        return {**base, "primary": "mixed / no strong single signal",
                "secondary": None, "secondary_share": None,
                "narrative": (
                    f"{cohort_desc} losses show no dominant failure mode "
                    f"(top: {top_mode} {top_share:.0%}, "
                    f"unexplained {unexplained:.0%}).")}
    sec, sec_share = next(((m, s) for m, s in modes[1:] if s >= SECONDARY_SHARE),
                          (None, None))
    tail = (f"; secondary signal: {sec} {sec_share:.0%})." if sec else ").")
    return {**base, "primary": top_mode, "secondary": sec,
            "secondary_share": sec_share,
            "narrative": (f"{cohort_desc} losses are best explained by "
                          f"{top_mode} ({top_share:.0%} of losses{tail}")}


def _pct(vals, q):
    """Linear-interpolated percentile; None on empty input."""
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def timing_distribution(loss_feats):
    """When the loss cohort crosses each value threshold (game fractions)."""
    out = {}
    keys = [(k, f"{k}_fraction") for k in CROSSING_KEYS]
    keys.append(("largest_drop", "largest_drop_fraction"))
    for name, field in keys:
        fracs = [f[field] for f in loss_feats if f[field] is not None]
        out[name] = {"p25": _pct(fracs, 0.25), "p50": _pct(fracs, 0.50),
                     "p75": _pct(fracs, 0.75),
                     "never": len(loss_feats) - len(fracs)}
    return out


def secondary_contrast_summary(loss_feats):
    """A vs B inside the loss cohort. B metrics are in B's own perspective;
    the onset gap asks: did B see the win (>= B_ONSET_HIGH) before A admitted
    the loss (<= lost_value)?"""
    def col(key):
        return [f[key] for f in loss_feats if f.get(key) is not None]

    both = [f for f in loss_feats
            if f.get("b_first_value_above_050_fraction") is not None
            and f.get("first_a_value_below_lost_fraction") is not None]
    gaps = [f["first_a_value_below_lost_fraction"]
            - f["b_first_value_above_050_fraction"] for f in both]
    return {
        "games": len(loss_feats),
        "a_mean_value": _mean(col("mean_a_value")),
        "b_mean_value": _mean(col("b_mean_value")),
        "a_mean_top1_share_post": _mean(col("mean_top1_share_post")),
        "b_mean_top1_share_post": _mean(col("b_mean_top1_share_post")),
        "a_median_visit_rank_post": _median(col("median_selected_visit_rank_post")),
        "b_median_visit_rank_post": _median(col("b_median_visit_rank_post")),
        "b_saw_it_first_share": (
            sum(1 for f in loss_feats if f.get("b_saw_it_first")) / len(loss_feats)
            if loss_feats else None),
        "median_onset_gap_fraction": _median(gaps),
        "onset_gap_games": len(both),
    }
