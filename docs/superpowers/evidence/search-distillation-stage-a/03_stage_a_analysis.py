"""STAGE A ANALYSIS -- the verdict, from the COMMITTED run record only (read-only).

Card `docs/superpowers/2026-10-11-search-distillation-stage-a-card.md` §2, fixed before any game:
  * game score for sims1600 by AGENT IDENTITY: winner_agent_id sims1600 -> 1, sims400 -> 0, none -> 0.5;
    the winning colour must agree with that colour's bound agent. Checkpoint identity is never used.
  * pair k = games 2k and 2k + 1; pair score = the mean of its two game scores; N = 80 pairs.
  * PASS iff mean - sqrt(ln(40) / (2N)) > 0.5, computed exactly (no rounded threshold); otherwise NON-PASS.
A verdict is computed ONLY for a completed run: outer exit 0, no survivors, verdict COMPLETED, the result-set
check recorded, and the exact 160-game schedule (slot g = game g, pair g // 2, sims1600 red on even g,
budgets 1600/400 bound to the agents, every root visit count = that colour's budget). Anything else: NO
VERDICT. The interval is conditional on independence (card §2). The distinct-transcript count is
descriptive, not a gate.

Inputs, each committed and unmodified: plan.json, outer.json, record.json. One create-only record:
03_stage_a_analysis_run.txt. Exit 0 a verdict (PASS or NON-PASS) / 1 no verdict / 4 refused.
This script holds no seed literal.

    python -B <this file>
"""
import json
import math
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
RUN = "docs/superpowers/evidence/search-distillation-stage-a"
INPUTS = [f"{RUN}/plan.json", f"{RUN}/outer.json", f"{RUN}/record.json"]
OUT = f"{RUN}/03_stage_a_analysis_run.txt"
A, B = "sims1600", "sims400"
BUDGET = {A: 1600, B: 400}
N_PAIRS = 80
N_GAMES = 2 * N_PAIRS
REASONS = {"win", "state_cap", "board_full"}


def schedule_faults(plan, outer, record):
    """Every reason a verdict may not be computed; empty if it may."""
    f = []
    if plan.get("n_slots") != N_GAMES:
        f.append(f"the plan is not {N_GAMES} slots")
    if outer.get("outer_exit") != 0:
        f.append(f"outer exit {outer.get('outer_exit')!r}, not 0")
    if outer.get("survivors", None) != "":
        f.append(f"survivors {outer.get('survivors')!r}: termination not confirmed")
    if record.get("verdict") != "COMPLETED":
        f.append(f"run verdict {record.get('verdict')!r}, not COMPLETED")
    if record.get("result_set_check") != "validate_result_set passed":
        f.append("the result-set check is not recorded as passed")
    games = record.get("games") or []
    if len(games) != N_GAMES:
        f.append(f"{len(games)} games, not {N_GAMES}")
    for i, g in enumerate(games):
        red, black = (A, B) if i % 2 == 0 else (B, A)
        want = {"slot": i, "game_idx": i, "red_agent_id": red, "black_agent_id": black,
                "red_mcts_sims": BUDGET[red], "black_mcts_sims": BUDGET[black]}
        got = {k: g.get(k) for k in want}
        if got != want:
            f.append(f"game {i}: schedule {got} is not {want}")
            continue
        if g.get("root_total_visits") != {"red": [BUDGET[red]], "black": [BUDGET[black]]}:
            f.append(f"game {i}: root visit sets {g.get('root_total_visits')} are not the bound budgets")
        if g.get("reason") not in REASONS:
            f.append(f"game {i}: reason {g.get('reason')!r}")
        w, wa = g.get("winner"), g.get("winner_agent_id")
        if (g.get("reason") == "win") != (w in ("red", "black")):
            f.append(f"game {i}: reason {g.get('reason')!r} with winner {w!r}")
        elif w is None and wa is not None:
            f.append(f"game {i}: no winning colour but winner agent {wa!r}")
        elif w is not None and wa != {"red": red, "black": black}[w]:
            f.append(f"game {i}: winner {w} is bound to {({'red': red, 'black': black})[w]}, record says {wa!r}")
        if not (isinstance(g.get("transcript_sha256"), str) and len(g["transcript_sha256"]) == 64):
            f.append(f"game {i}: no transcript digest")
    return f


def score(game):
    """sims1600's score, by agent identity."""
    return {A: 1.0, B: 0.0, None: 0.5}[game["winner_agent_id"]]


def verdict(games):
    pairs = [(score(games[2 * k]) + score(games[2 * k + 1])) / 2 for k in range(N_PAIRS)]
    mean = sum(pairs) / N_PAIRS
    half = math.sqrt(math.log(40) / (2 * N_PAIRS))
    return {"pairs": pairs, "mean": mean, "half_width": half, "lower": mean - half,
            "verdict": "PASS" if mean - half > 0.5 else "NON-PASS",
            "points": sum(score(g) for g in games),
            "distinct_transcripts": len({g["transcript_sha256"] for g in games})}


def _synthetic(points_a):
    """A completed 160-game record in which sims1600 scores `points_a[i]` in game i."""
    games = []
    for i, s in enumerate(points_a):
        red, black = (A, B) if i % 2 == 0 else (B, A)
        wa = {1.0: A, 0.0: B, 0.5: None}[s]
        w = None if wa is None else ("red" if wa == red else "black")
        games.append({"slot": i, "game_idx": i, "red_agent_id": red, "black_agent_id": black,
                      "red_mcts_sims": BUDGET[red], "black_mcts_sims": BUDGET[black],
                      "root_total_visits": {"red": [BUDGET[red]], "black": [BUDGET[black]]},
                      "reason": "win" if w else "state_cap", "winner": w, "winner_agent_id": wa,
                      "transcript_sha256": f"{i:064x}"})
    return {"verdict": "COMPLETED", "result_set_check": "validate_result_set passed", "games": games}


def _controls():
    plan, outer = {"n_slots": N_GAMES}, {"outer_exit": 0, "survivors": ""}

    def judged(record, o=outer, p=plan):
        faults = schedule_faults(p, o, record)
        return ("NO VERDICT", faults) if faults else (verdict(record["games"])["verdict"], [])
    p104 = [1.0] * 104 + [0.0] * 56                      # 104 points: mean 0.65
    p1045 = [1.0] * 104 + [0.5] + [0.0] * 55             # 104.5 points: mean 0.653125
    cases = [("all 160 wins: PASS", _synthetic([1.0] * 160), "PASS"),
             ("all 160 draws: NON-PASS", _synthetic([0.5] * 160), "NON-PASS"),
             ("104 points (mean 0.65): NON-PASS", _synthetic(p104), "NON-PASS"),
             ("104.5 points (mean 0.653125): PASS", _synthetic(p1045), "PASS")]
    swapped = _synthetic([1.0] * 160)                     # winner by colour, but the record names the other agent
    swapped["games"][7]["winner_agent_id"] = B
    budget = _synthetic([1.0] * 160)
    budget["games"][4]["red_mcts_sims"], budget["games"][4]["black_mcts_sims"] = 400, 1600
    short = _synthetic([1.0] * 160)
    short["games"] = short["games"][:159]
    visits = _synthetic([1.0] * 160)
    visits["games"][9]["root_total_visits"] = {"red": [400], "black": [1599, 1600]}
    out = [(label, judged(rec)[0] == want, f"{judged(rec)[0]}") for label, rec, want in cases]
    out.append(("104 and 104.5 points straddle the exact bound",
                verdict(_synthetic(p104)["games"])["lower"] <= 0.5 < verdict(_synthetic(p1045)["games"])["lower"],
                f"lower bounds {verdict(_synthetic(p104)['games'])['lower']:.6f} / "
                f"{verdict(_synthetic(p1045)['games'])['lower']:.6f}"))
    for label, rec, o in [("159 games: no verdict", short, outer),
                          ("winner agent not the winning colour's agent: no verdict", swapped, outer),
                          ("budgets not bound to the agents: no verdict", budget, outer),
                          ("a root visit count off the budget: no verdict", visits, outer),
                          ("outer exit 124: no verdict", _synthetic([1.0] * 160), {**outer, "outer_exit": 124}),
                          ("survivors: no verdict", _synthetic([1.0] * 160), {**outer, "survivors": "123 "}),
                          ("run verdict STOP_INTEGRITY: no verdict", {**_synthetic([1.0] * 160),
                                                                     "verdict": "STOP_INTEGRITY"}, outer)]:
        got, faults = judged(rec, o)
        out.append((label, got == "NO VERDICT", "; ".join(faults)[:120]))
    return out


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def main():
    if (ROOT / OUT).exists():
        print(f"REFUSED: {OUT} exists (create-only)", file=sys.stderr)
        return 4
    for rel in INPUTS:
        if _git("ls-files", "--error-unmatch", "--", rel).returncode or _git("diff", "--quiet", "HEAD", "--", rel).returncode:
            print(f"REFUSED: {rel} is absent, uncommitted or modified", file=sys.stderr)
            return 4
    plan, outer, record = (json.loads((ROOT / rel).read_text()) for rel in INPUTS)
    dirty = bool(_git("status", "--porcelain").stdout.strip())
    lines = ["SEARCH DISTILLATION STAGE A -- ANALYSIS", "=" * 96,
             f"HEAD {_git('rev-parse', 'HEAD').stdout.strip()}; working tree {'DIRTY' if dirty else 'clean'}",
             *(f"input {rel} (committed, unmodified)" for rel in INPUTS),
             "", "CONTROLS"]
    controls = _controls()
    lines += [f"  {'PASS' if ok else 'FAIL':6s} {label}  [{d}]" for label, ok, d in controls]
    passed = sum(ok for _l, ok, _d in controls)
    lines.append(f"\ncontrols passed: {passed}/{len(controls)}")
    faults = schedule_faults(plan, outer, record) + (["the working tree is dirty"] if dirty else []) + \
        ([] if passed == len(controls) else ["a control failed"])
    if faults:
        lines.append("NO VERDICT: " + "; ".join(faults[:20]) + (f" (+{len(faults) - 20} more)" if len(faults) > 20 else ""))
        code = 1
    else:
        v = verdict(record["games"])
        lines += ["", f"games {N_GAMES}, pairs {N_PAIRS}; sims1600 points {v['points']} of {N_GAMES}",
                  "pair scores: " + " ".join(f"{p:g}" for p in v["pairs"]),
                  f"mean pair score {v['mean']:.6f}; half-width sqrt(ln(40)/(2*{N_PAIRS})) = {v['half_width']:.6f}; "
                  f"lower bound {v['lower']:.6f}",
                  f"distinct transcripts (descriptive): {v['distinct_transcripts']} of {N_GAMES}",
                  "The bound is conditional on the card's independence assumption (§2).",
                  f"VERDICT: {v['verdict']}"]
        code = 0
    print("\n".join(lines))
    with open(ROOT / OUT, "x", encoding="utf-8") as fh:            # create-only
        fh.write("\n".join(lines) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
