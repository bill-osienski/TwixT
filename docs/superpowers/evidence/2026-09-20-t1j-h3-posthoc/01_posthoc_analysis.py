#!/usr/bin/env python3
"""H3 POST-HOC ANALYSIS — EXPLORATORY, READ-ONLY, AND NOT A VERDICT.

🔴 WHAT THIS IS NOT. Every comparison below was chosen AFTER the result was
known. None of it was preregistered, none of it is confirmatory, and no number
here revises, supports or qualifies the combined report's interval. The study's
answer is the pinned report and nothing else:

    INCUMBENT STRONGER -- mean 0.7348, n=296 pairs,
    nominal 95% Hoeffding interval [0.6559, 0.8137]

  over legal six-ply positions drawn UNIFORMLY AT RANDOM, structural filters
  only, in the frozen deterministic argmax configuration. It says NOTHING about
  realistic play.

🔴 IT WRITES ONLY INTO ITS OWN DIRECTORY. The combined report, the receipt, the
frozen population, the four segment record sets, the seed registries and every
gate are READ and never touched. Nothing here runs a model, plays a game, draws
a seed or re-derives a walk: the population artifact is read as plain JSON
rather than through the validating loader, because that loader re-runs the PRNG
walk and this analysis must not start a seed-driven process.

🔑 REPRODUCIBLE, AND IT CHECKS ITS OWN INPUTS. Every input is hashed as it is
read and recorded. Re-running against changed inputs REFUSES rather than
quietly analysing something else; re-running against the same inputs produces
byte-identical output.

USAGE:  python 01_posthoc_analysis.py [--out DIR] [--rehash]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]
EVIDENCE = REPO / "docs" / "superpowers" / "evidence"

SEGMENT_DIRS = [
    EVIDENCE / "2026-09-18-t1j-h3-study-segment0-retry",
    EVIDENCE / "2026-09-15-t1j-h3-study-segment1",
    EVIDENCE / "2026-09-15-t1j-h3-study-segment2",
    EVIDENCE / "2026-09-15-t1j-h3-study-segment3",
]
POPULATION = EVIDENCE / "2026-09-17-t1j-h3-study-openings-uniform" / "01_opening_set.json"
COMBINED = (EVIDENCE / "2026-09-20-t1j-h3-study-combined-attempt2"
            / "09_combined_report.json")

BOARD = 24
CENTRE = (BOARD - 1) / 2.0


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ───────────────────────── the frozen scoring rule ──────────────────────────
def points(game) -> float:
    """The incumbent's points for ONE game: 1 win, 0 loss, 0.5 cap.

    🔑 COPIED FROM `h3_study_analysis._points` DELIBERATELY, not imported. This
    script must keep working against the SEALED records without importing the
    live analysis module, and a copy that drifts is caught immediately: the
    pair mean it produces is compared to the pinned report's below, and any
    disagreement ABORTS.
    """
    if game.get("terminal_reason") == "cap":
        return 0.5
    return 1.0 if game.get("winner") == game.get("incumbent_colour") else 0.0


def cohens_d(a, b) -> float:
    """Standardised mean difference. An EFFECT SIZE, not a test: no p-value and
    no interval, because either would invite reading this as confirmatory."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2)
    if pooled <= 0:
        return 0.0
    return (statistics.mean(a) - statistics.mean(b)) / math.sqrt(pooled)


def describe(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    q = statistics.quantiles(xs, n=4) if len(xs) > 1 else [xs[0]] * 3
    return {"n": len(xs), "mean": statistics.mean(xs), "min": xs[0],
            "q1": q[0], "median": q[1], "q3": q[2], "max": xs[-1],
            "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0}


# ───────────────────────────── opening geometry ─────────────────────────────
def opening_features(moves):
    """Geometry only. No engine, no rules, no evaluation -- just where the six
    stones sit, so 'opening characteristics' cannot smuggle in a model."""
    pts = [(int(r), int(c)) for r, c in moves]
    a, b = pts[0::2], pts[1::2]          # first mover's stones, second mover's

    def centrality(ps):
        return statistics.mean(max(abs(r - CENTRE), abs(c - CENTRE)) for r, c in ps)

    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    cross = min(math.dist(p, q) for p in a for q in b)
    return {
        "centrality": centrality(pts),
        "centrality_first": centrality(a),
        "centrality_second": centrality(b),
        "row_extent": max(rows) - min(rows),
        "col_extent": max(cols) - min(cols),
        "bbox_area": (max(rows) - min(rows)) * (max(cols) - min(cols)),
        "min_cross_distance": cross,
        "mean_pairwise_distance": statistics.mean(
            math.dist(pts[i], pts[j])
            for i in range(len(pts)) for j in range(i + 1, len(pts))),
    }


def bucket_by(pairs, key, n=4):
    """Quartile buckets over a continuous feature. Ties go to the lower bucket,
    so bucket sizes are reported rather than assumed equal."""
    vals = sorted(p[key] for p in pairs)
    if not vals:
        return []
    cuts = statistics.quantiles(vals, n=n)
    out = [{"bucket": i, "lo": None, "hi": None, "scores": []} for i in range(n)]
    for p in pairs:
        i = sum(1 for c in cuts if p[key] > c)
        out[i]["scores"].append(p["score"])
    for i, o in enumerate(out):
        o["lo"] = float(vals[0]) if i == 0 else float(cuts[i - 1])
        o["hi"] = float(cuts[i]) if i < len(cuts) else float(vals[-1])
        o["n"] = len(o["scores"])
        o["mean_score"] = statistics.mean(o["scores"]) if o["scores"] else None
        del o["scores"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--rehash", action="store_true",
                    help="record the current input hashes instead of checking "
                         "them. Use ONLY when the inputs legitimately changed.")
    a = ap.parse_args(argv)
    out_dir = pathlib.Path(a.out)

    # ── inputs, hashed as they are read
    inputs = []
    for d in SEGMENT_DIRS:
        inputs.append({"role": "segment_results", "path": str((d / "03_results.jsonl").relative_to(REPO)),
                       "sha256": sha256(d / "03_results.jsonl")})
    inputs.append({"role": "frozen_population", "path": str(POPULATION.relative_to(REPO)),
                   "sha256": sha256(POPULATION)})
    inputs.append({"role": "combined_report", "path": str(COMBINED.relative_to(REPO)),
                   "sha256": sha256(COMBINED)})

    manifest = out_dir / "00_inputs.json"
    if manifest.exists() and not a.rehash:
        recorded = json.loads(manifest.read_text())["inputs"]
        if recorded != inputs:
            print("🔴 REFUSED: an input has changed since these hashes were "
                  "recorded. This analysis describes the SEALED evidence; "
                  "re-run with --rehash only if that change is legitimate.",
                  file=sys.stderr)
            for r, g in zip(recorded, inputs):
                if r != g:
                    print(f"    {r['path']}\n      recorded {r['sha256']}\n"
                          f"      now      {g['sha256']}", file=sys.stderr)
            return 2

    # ── the records
    games = []
    for k, d in enumerate(SEGMENT_DIRS):
        with open(d / "03_results.jsonl", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("record_type") == "task_result":
                    games.append(row)

    by_pair = defaultdict(list)
    for g in games:
        by_pair[g["pair_id"]].append(g)

    population = json.loads(POPULATION.read_text())
    feats = {o["digest"]: opening_features(o["moves"]) for o in population["openings"]}

    pairs = []
    for pid in sorted(by_pair):
        rows = by_pair[pid]
        assert len(rows) == 2, f"pair {pid} has {len(rows)} games"
        assert sorted(g["incumbent_colour"] for g in rows) == ["black", "red"]
        score = sum(points(g) for g in rows) / 2.0
        od = rows[0]["opening_digest"]
        rec = {
            "pair_id": pid, "segment": rows[0]["segment"], "score": score,
            "opening_digest": od,
            "caps": sum(1 for g in rows if g.get("terminal_reason") == "cap"),
            "plies_total": sum(int(g["plies"]) for g in rows),
            "plies_max": max(int(g["plies"]) for g in rows),
            "elapsed_s": sum(float(g["elapsed_s"]) for g in rows),
        }
        rec.update(feats.get(od, {}))
        pairs.append(rec)

    # ── the derivation must agree with the PINNED report, or this aborts
    combined = json.loads(COMBINED.read_text())
    mean_here = statistics.mean(p["score"] for p in pairs)
    if abs(mean_here - combined["primary"]["mean"]) > 1e-12 or \
            len(pairs) != combined["pairs_scored"]:
        print(f"🔴 REFUSED: this script derives mean {mean_here!r} over "
              f"{len(pairs)} pairs; the pinned report says "
              f"{combined['primary']['mean']!r} over "
              f"{combined['pairs_scored']}. The copied scoring rule has "
              f"drifted and nothing below would describe the study.",
              file=sys.stderr)
        return 3

    result = build(pairs, games, combined, mean_here)
    result["inputs"] = inputs

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "00_inputs.json").write_text(
        json.dumps({"inputs": inputs}, indent=1, sort_keys=True) + "\n")
    (out_dir / "02_posthoc.json").write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n")
    (out_dir / "03_posthoc.md").write_text(render(result))
    print(f"wrote {out_dir}/02_posthoc.json and 03_posthoc.md "
          f"({len(pairs)} pairs, {len(games)} games)")
    return 0


def build(pairs, games, combined, mean_here):
    scores = [p["score"] for p in pairs]
    excess = [s - 0.5 for s in scores]

    # by incumbent colour -- PER GAME, since colour is a game-level property
    by_colour = {}
    for colour in ("red", "black"):
        rows = [g for g in games if g["incumbent_colour"] == colour]
        pts = [points(g) for g in rows]
        by_colour[colour] = {
            "games": len(rows), "mean_points": statistics.mean(pts),
            "wins": sum(1 for p in pts if p == 1.0),
            "losses": sum(1 for p in pts if p == 0.0),
            "caps": sum(1 for p in pts if p == 0.5),
        }
    by_colour["difference_red_minus_black"] = (
        by_colour["red"]["mean_points"] - by_colour["black"]["mean_points"])
    by_colour["cohens_d"] = cohens_d(
        [points(g) for g in games if g["incumbent_colour"] == "red"],
        [points(g) for g in games if g["incumbent_colour"] == "black"])

    by_segment = {}
    for k in sorted({p["segment"] for p in pairs}):
        s = [p["score"] for p in pairs if p["segment"] == k]
        by_segment[str(k)] = {"pairs": len(s), "mean_score": statistics.mean(s),
                              "sd": statistics.stdev(s)}

    categories = Counter()
    for p in pairs:
        categories[{0.0: "lost both", 0.25: "lost one, drew one",
                    0.5: "split", 0.75: "won one, drew one",
                    1.0: "won both"}.get(p["score"], f"other {p['score']}")] += 1

    capped = [p["score"] for p in pairs if p["caps"] > 0]
    uncapped = [p["score"] for p in pairs if p["caps"] == 0]

    # concentration: trimmed means, and where the excess comes from
    ordered = sorted(scores, reverse=True)
    trimmed = {}
    for frac in (0.05, 0.10, 0.25):
        k = int(round(len(ordered) * frac))
        trimmed[f"drop_top_{int(frac*100)}pct"] = {
            "dropped": k, "n": len(ordered) - k,
            "mean_score": statistics.mean(ordered[k:])}
    pos = sum(e for e in excess if e > 0)
    neg = sum(-e for e in excess if e < 0)
    ordered_excess = sorted(excess, reverse=True)
    deciles = []
    for i in range(10):
        chunk = ordered_excess[i * len(ordered_excess) // 10:
                               (i + 1) * len(ordered_excess) // 10]
        deciles.append({"decile": i + 1, "n": len(chunk), "excess": sum(chunk)})

    feature_buckets = {}
    for key in ("centrality", "bbox_area", "min_cross_distance",
                "mean_pairwise_distance", "plies_max", "elapsed_s"):
        if all(key in p for p in pairs):
            feature_buckets[key] = bucket_by(pairs, key)

    return {
        "design": "H3_POSTHOC_EXPLORATORY",
        "status": "EXPLORATORY -- chosen after the result was known; not "
                  "preregistered, not confirmatory, and no interval here "
                  "revises the pinned one",
        "pinned_reference": {
            "verdict": combined["verdict"],
            "mean": combined["primary"]["mean"],
            "interval": combined["primary"]["interval"],
            "pairs_scored": combined["pairs_scored"],
            "population_claim": combined["population"]["claim"],
        },
        "derivation_agrees_with_pinned_report": True,
        "n_pairs": len(pairs), "n_games": len(games),
        "mean_pair_score": mean_here,
        "pair_score_distribution": dict(categories),
        "pairs_above_parity": sum(1 for s in scores if s > 0.5),
        "pairs_at_parity": sum(1 for s in scores if s == 0.5),
        "pairs_below_parity": sum(1 for s in scores if s < 0.5),
        "total_excess_over_parity": sum(excess),
        "positive_excess": pos, "negative_excess": neg,
        "excess_by_decile": deciles,
        "trimmed_means": trimmed,
        "by_incumbent_colour": by_colour,
        "by_segment": by_segment,
        "caps": {
            "capped_games": sum(1 for g in games
                                if g.get("terminal_reason") == "cap"),
            "pairs_with_a_cap": len(capped),
            "mean_score_pairs_with_cap": statistics.mean(capped) if capped else None,
            "mean_score_pairs_without_cap": statistics.mean(uncapped),
            "cohens_d_capped_minus_uncapped": cohens_d(capped, uncapped),
        },
        "game_length_plies": describe([int(g["plies"]) for g in games]),
        "elapsed_s_per_game": describe([float(g["elapsed_s"]) for g in games]),
        "feature_buckets": feature_buckets,
        "replay_availability": {
            "h3_moves_persisted": False,
            "note": "H3's result records carry task_id, pair_id, seed, segment, "
                    "stratum, opening_digest, transcript_digest, "
                    "incumbent_colour, winner, terminal_reason, plies and "
                    "elapsed_s; its transcript records carry only n_plies and "
                    "opening_bound. NO MOVE LIST WAS PERSISTED, so EXACT H3 "
                    "VISUAL REPLAY IS UNAVAILABLE and cannot be reconstructed "
                    "from this evidence. The first SIX plies of every game are "
                    "recoverable from the frozen population artifact; nothing "
                    "after ply 6 was recorded. H2 replays exist and may "
                    "ILLUSTRATE engine behaviour, but they are a different "
                    "design on a different population and can neither explain "
                    "nor validate the H3 result.",
        },
    }


def render(r) -> str:
    L = []
    w = L.append
    w("# H3 post-hoc analysis — EXPLORATORY\n")
    w("> 🔴 **Not a verdict, and not preregistered.** Every comparison here was")
    w("> chosen after the result was known. Nothing below revises, supports or")
    w("> qualifies the study's interval, and no causal claim is made.\n")
    p = r["pinned_reference"]
    w(f"The study's answer remains the pinned combined report: **{p['verdict']}**,")
    w(f"mean {p['mean']:.4f} over {p['pairs_scored']} pairs, nominal 95% interval")
    w(f"[{p['interval'][0]:.4f}, {p['interval'][1]:.4f}].\n")
    w(f"**Scope of that claim:** {p['population_claim']}\n")
    w(f"This script re-derives the mean as {r['mean_pair_score']:.6f} from the")
    w("sealed records and ABORTS if it disagrees with the pinned report.\n")

    w("## Is the advantage broad or concentrated?\n")
    w(f"- pairs above parity: **{r['pairs_above_parity']}**, "
      f"at parity: **{r['pairs_at_parity']}**, "
      f"below parity: **{r['pairs_below_parity']}** (of {r['n_pairs']})")
    w(f"- total excess over parity: **{r['total_excess_over_parity']:+.2f}** pair-points"
      f" (positive {r['positive_excess']:.2f}, negative {r['negative_excess']:.2f})\n")
    w("Trimmed means — dropping the best pairs and re-taking the mean:\n")
    w("| dropped | remaining pairs | mean score |")
    w("|---|---|---|")
    for k, v in r["trimmed_means"].items():
        w(f"| {k.replace('_', ' ')} ({v['dropped']}) | {v['n']} | {v['mean_score']:.4f} |")
    w("")
    w("Excess over parity by decile of pair score (decile 1 = strongest):\n")
    w("| decile | pairs | excess |")
    w("|---|---|---|")
    for d in r["excess_by_decile"]:
        w(f"| {d['decile']} | {d['n']} | {d['excess']:+.2f} |")
    w("")

    w("## Paired outcome categories\n")
    w("| category | pairs |")
    w("|---|---|")
    for k, v in sorted(r["pair_score_distribution"].items(),
                       key=lambda kv: -kv[1]):
        w(f"| {k} | {v} |")
    w("")

    c = r["by_incumbent_colour"]
    w("## By incumbent colour (per game, n=592)\n")
    w("| colour | games | mean points | wins | losses | caps |")
    w("|---|---|---|---|---|---|")
    for col in ("red", "black"):
        v = c[col]
        w(f"| {col} | {v['games']} | {v['mean_points']:.4f} | {v['wins']} | "
          f"{v['losses']} | {v['caps']} |")
    w(f"\nDifference (red − black): **{c['difference_red_minus_black']:+.4f}**, "
      f"Cohen's d {c['cohens_d']:+.3f}. *Exploratory.*\n")

    w("## By segment (74 pairs each)\n")
    w("| segment | pairs | mean score | sd |")
    w("|---|---|---|---|")
    for k, v in sorted(r["by_segment"].items()):
        w(f"| {k} | {v['pairs']} | {v['mean_score']:.4f} | {v['sd']:.4f} |")
    w("")

    cp = r["caps"]
    w("## Caps\n")
    w(f"- capped games: **{cp['capped_games']}** of {r['n_games']}; "
      f"pairs containing a cap: **{cp['pairs_with_a_cap']}**")
    w(f"- mean score, pairs with a cap: "
      f"**{cp['mean_score_pairs_with_cap']:.4f}**; without: "
      f"**{cp['mean_score_pairs_without_cap']:.4f}**")
    w(f"- Cohen's d (capped − uncapped): {cp['cohens_d_capped_minus_uncapped']:+.3f}. "
      f"*Exploratory.*\n")

    for label, key in (("Game length (plies, per game)", "game_length_plies"),
                       ("Elapsed seconds per game", "elapsed_s_per_game")):
        d = r[key]
        w(f"## {label}\n")
        w(f"- n {d['n']}, mean {d['mean']:.2f}, sd {d['sd']:.2f}")
        w(f"- min {d['min']:.2f}, q1 {d['q1']:.2f}, median {d['median']:.2f}, "
          f"q3 {d['q3']:.2f}, max {d['max']:.2f}\n")

    w("## Opening characteristics (geometry only)\n")
    w("Quartile buckets over each feature; mean pair score in each. Geometry is")
    w("computed from the six frozen opening stones only — no engine, no rules,")
    w("no evaluation. *All exploratory.*\n")
    for key, buckets in sorted(r["feature_buckets"].items()):
        w(f"**{key}**\n")
        w("| bucket | range | pairs | mean score |")
        w("|---|---|---|---|")
        for b in buckets:
            ms = "—" if b["mean_score"] is None else f"{b['mean_score']:.4f}"
            w(f"| {b['bucket'] + 1} | {b['lo']:.2f} – {b['hi']:.2f} | {b['n']} | {ms} |")
        w("")

    w("## Replay availability\n")
    w(f"{r['replay_availability']['note']}\n")
    w("---\n")
    w("Inputs, by path and sha256, are recorded in `00_inputs.json`; this script")
    w("refuses to run if any of them has changed.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
