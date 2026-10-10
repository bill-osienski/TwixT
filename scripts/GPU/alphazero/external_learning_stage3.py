"""EXTERNAL-ENGINE LEARNING, STAGE 3 -- the preliminary evaluation. PROTOTYPE; PREPARATION ONLY.

Card: `docs/superpowers/2026-10-08-external-engine-learning-viability.md`, design baseline ff1422fb, stage 3.
Nothing here runs on import. The run entry refuses unless the 30th gate,
`external_learning_stage3_authorization.EXTERNAL_LEARNING_STAGE3_AUTHORIZED`, is open -- read at call time,
in the parent AND in the child.

WHAT STAGE 3 DOES, in the card's order (each failure is a STOP; nothing is retried, selected or extended):
  1. the committed seed plan (32 game seeds, ACCOUNTED and unspent), the frozen manifest pinned by sha256 and
     equal to the one this module derives, and the pinned environment (python + `pip freeze`);
  2. the sealed held-out rows of stage 1 (each rebuilt in OUR engine; exactly the manifest's four games);
  3. the incumbent (SHA-1 pinned), its scalar control (stage 2's fitted beta, frozen -- never refitted) and
     the single stage-2 candidate (sha256 pinned) evaluated ONCE on those rows. Per game: mean chosen-move CE
     (all three), raw top-1 agreement and teacher rank (incumbent/candidate), move disagreements; the four
     games weigh EQUALLY. Candidate game-mean CE strictly below BOTH the incumbent and the control, and
     candidate game-mean top-1 strictly above the incumbent -- otherwise STOP_HELDOUT before any game;
  4. 32 fresh candidate-vs-incumbent games through the existing evaluation runner (empty board, stochastic
     opening readout, the card's unchanged research settings, one worker, sequential dual-checkpoint), 16
     colour-balanced pairs, one plan seed per game;
  5. W/L/draw/cap counts and the mean pair score with the conservative 95% Hoeffding interval over the 16
     pair means (unclipped half-width 0.3395). A disaster screen only: VETO when the interval lies wholly
     below 0.50; UNINTERPRETABLE when more than half the games are caps.
A completed screen is not promotion, noninferiority or evidence of stronger play.

    python -m scripts.GPU.alphazero.external_learning_stage3 run
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import external_learning as EL
from . import external_learning_stage2 as S2
from . import external_learning_stage3_authorization as GATE

ROOT = EL.ROOT
MODULE = "scripts.GPU.alphazero.external_learning_stage3"
MANIFEST_KIND = "external-learning-stage3-manifest/1"
PLAN_KIND = "external-learning-stage3-plan/1"
RECORD_KIND = "external-learning-stage3-record/1"
MANIFEST = "docs/superpowers/evidence/external-engine-learning-plans/stage3-manifest.json"
MANIFEST_SHA256 = "f26f2f60e4b9907916abe81984aceeadee899deb0513acb84680ce66c39cffcc"
PLAN = "docs/superpowers/evidence/external-engine-learning-plans/stage3.json"
RUN_DIR = "runs/external-engine-learning/stage3"
EVIDENCE_DIR = "docs/superpowers/evidence/external-engine-learning-stage3"
CLEANUP_FAILED = RUN_DIR + ".cleanup-failed.json"
REPLAYS = "games"

HELDOUT_GAMES = 4
PAIRS = 16
GAMES = 2 * PAIRS                                            # 16 colour-balanced pairs, one seed per game
N_SLOTS = GAMES
EXECUTION_S = 7200                                           # the card's 120 minutes, held-out comparison included
PAIRING = "external-learning-stage3"
CANDIDATE_ID, INCUMBENT_ID = "candidate", "incumbent"
#: the card's evaluation protocol, spelled out -- never the runner's defaults by reference
PROTOCOL = {"board_size": 24, "mcts_sims": 400, "mcts_eval_batch_size": 14, "mcts_stall_flush_sims": 48,
            "selection_mode": "opening_temperature", "opening_temp_plies": 20, "temp_high": 1.0, "temp_low": 0.1,
            "max_moves": 280}
HOEFFDING_ALPHA = 0.05
HALF_WIDTH = math.sqrt(math.log(2 / HOEFFDING_ALPHA) / (2 * PAIRS))   # 0.33953...: pair means in [0, 1]
PARITY = 0.5
VERDICT_EXIT = {"SCREEN_COMPLETED": 0, "STOP_HELDOUT": 1, "STOP": 2}
EXIT_COMPLETED, EXIT_STOP_HELDOUT, EXIT_STOP = 0, 1, 2
EXIT_REFUSED, EXIT_TIMEOUT, EXIT_CLEANUP_UNVERIFIED = EL.EXIT_REFUSED, EL.EXIT_TIMEOUT, EL.EXIT_CLEANUP_UNVERIFIED

Stop, Refused = EL.Stage1Stop, EL.Refused


class HeldoutFailed(Stop):
    """The held-out criteria failed: the stage ends here, before any evaluation-game compute."""


def _require_gate() -> None:
    if GATE.EXTERNAL_LEARNING_STAGE3_AUTHORIZED is not True:
        raise Refused("EXTERNAL_LEARNING_STAGE3_AUTHORIZED is closed: stage 3 is not authorized")


def eval_config():
    from .eval_runner import EvalConfig

    return EvalConfig(**PROTOCOL)


# ─────────────────────────────── the environment ───────────────────────────────

def _freeze_versions(freeze: str) -> Dict[str, str]:
    pins = dict(line.split("==", 1) for line in freeze.splitlines() if "==" in line)
    return {k: pins.get(k) for k in ("mlx", "numpy")}


def recorded_environment(text: str) -> Dict[str, Any]:
    """A stage's `environment.txt` ("python ...", "executable ...", "pip freeze exit 0", then the freeze)."""
    python, _exe, status, freeze = text.split("\n", 3)
    if not python.startswith("python ") or status != "pip freeze exit 0":
        raise Stop("environment", "not a complete environment record")
    return {"python": python[len("python "):], "pip_freeze_sha256": EL._sha(freeze.encode()), **_freeze_versions(freeze)}


def environment_now() -> Dict[str, Any]:
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    if out.returncode:
        raise Stop("environment", f"pip freeze exited {out.returncode}")
    return {"python": sys.version, "pip_freeze_sha256": EL._sha(out.stdout.encode()), **_freeze_versions(out.stdout)}


def check_environment(manifest: Mapping[str, Any], now: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The interpreter and every installed distribution must be the pinned ones."""
    now = dict(now if now is not None else environment_now())
    want = {k: manifest["environment"][k] for k in ("python", "pip_freeze_sha256", "mlx", "numpy")}
    if now != want:
        raise Stop("environment", f"the environment is not the pinned one: {now} != {want}")
    return now


# ─────────────────────────────── the manifest and the seed plan ───────────────────────────────

def _listed(ev: pathlib.Path) -> Dict[str, str]:
    return {name: sha for sha, name in (line.split("  ", 1) for line in (ev / "MANIFEST.sha256").read_text().splitlines())}


def build_manifest(root: pathlib.Path = ROOT) -> Dict[str, Any]:
    """Everything stage 3 is bound to, derived from the committed stage-1/stage-2 evidence and this module."""
    from .eval_runner import cfg_from, readout_from_eval_config
    from .twixtbot_g3_schedule import REFERENCE_CHECKPOINTS

    ev1, ev2 = root / EL.EVIDENCE_DIR, root / S2.EVIDENCE_DIR
    held = [r for r in EL.load_rows(ev1 / S2.STAGE1_ROWS) if r["split"] == "heldout"]
    rec2 = json.loads((ev2 / "record.json").read_bytes())
    cand = ev2 / S2.CANDIDATE
    if EL._sha(cand.read_bytes()) != rec2["steps"]["candidate_sha256"]:
        raise Stop("manifest", "the evidence candidate is not the one stage 2 saved")
    inc = REFERENCE_CHECKPOINTS[EL.REFERENCE]
    setup = root / "docs/superpowers/evidence/external-engine-learning-setup/03_venv_freeze.txt"
    fin1 = json.loads((ev1 / "finalized.json").read_bytes())
    env2 = (ev2 / "environment.txt").read_text(encoding="utf-8")
    cfg = eval_config()
    return {"kind": MANIFEST_KIND, "card": EL.CARD, "baseline": EL.BASELINE,
            "stage1": {"evidence_dir": EL.EVIDENCE_DIR,
                       "manifest_sha256": EL._sha((ev1 / "MANIFEST.sha256").read_bytes()),
                       "rows": {"file": S2.STAGE1_ROWS, "sha256": EL._sha((ev1 / S2.STAGE1_ROWS).read_bytes())},
                       "heldout": {"games": sorted({r["game_id"] for r in held}), "rows": len(held),
                                   "row_ids_sha256": EL._jsha([r["row_id"] for r in held])}},
            "stage2": {"evidence_dir": S2.EVIDENCE_DIR, "stage2_manifest_sha256": S2.MANIFEST_SHA256,
                       "manifest_sha256": EL._sha((ev2 / "MANIFEST.sha256").read_bytes()),
                       "record_sha256": EL._sha((ev2 / "record.json").read_bytes()),
                       "candidate": {"path": f"{S2.EVIDENCE_DIR}/{S2.CANDIDATE}",
                                     "sha256": rec2["steps"]["candidate_sha256"]},
                       "control": {"beta": rec2["steps"]["control"]["beta"],
                                   "training_ce": rec2["steps"]["control"]["ce"]}},
            "incumbent": {"reference": EL.REFERENCE, "path": inc["path"], "sha1": EL.REFERENCE_SHA1},
            "environment": {**recorded_environment(env2), "source": f"{S2.EVIDENCE_DIR}/environment.txt",
                            "source_sha256": EL._sha(env2.encode())},
            "environment_provenance": {
                "stage1": {"python": fin1["python"], **_freeze_versions(setup.read_text(encoding="utf-8")),
                           "freeze_file": "docs/superpowers/evidence/external-engine-learning-setup/03_venv_freeze.txt",
                           "freeze_sha256": EL._sha(setup.read_bytes())},
                "stage2": recorded_environment(env2)},
            "heldout_criteria": {
                "unit": "game", "games": HELDOUT_GAMES, "weights": "each held-out game equally (mean of per-game means)",
                "control": "softmax(beta * incumbent raw logits) over the row's legal moves, beta frozen from stage 2",
                "ce": "per game, mean over its rows of -log p(teacher's chosen move)",
                "top1": "per game, the fraction of rows whose first maximum (recorded legal order) is the chosen move",
                "pass": "candidate game-mean CE < incumbent AND < control (strict), and candidate game-mean top-1 "
                        "> incumbent (strict); otherwise STOP_HELDOUT before any game"},
            "games": {"games": GAMES, "pairs": PAIRS, "workers": 1, "add_noise": False, "eval_config": dict(PROTOCOL),
                      "mcts_config": asdict(cfg_from(cfg)), "readout": asdict(readout_from_eval_config(cfg)),
                      "pairing_id": PAIRING, "agents": [CANDIDATE_ID, INCUMBENT_ID],
                      "layout": "game g: seed = plan base + g, pair g // 2; candidate red on even g, black on odd g"},
            "screen": {"hoeffding_alpha": HOEFFDING_ALPHA, "half_width": HALF_WIDTH, "unit": "pair mean in [0, 1]",
                       "veto": "upper bound (mean + half-width, unclipped) < 0.5",
                       "uninterpretable": f"more than {GAMES // 2} state_cap games"},
            "ceiling_s": EXECUTION_S,
            "game_seeds": {"slots": N_SLOTS, "plan": PLAN}}


def check_manifest(root: pathlib.Path = ROOT) -> Dict[str, Any]:
    raw = (root / MANIFEST).read_bytes()
    if EL._sha(raw) != MANIFEST_SHA256:
        raise Stop("manifest", "the manifest is not the pinned bytes")
    manifest = json.loads(raw)
    if manifest != build_manifest(root):
        raise Stop("manifest", "the manifest is not the one this module derives from the stage-1/2 evidence")
    return manifest


def slot_layout(base: int) -> List[Dict[str, Any]]:
    return [{"game": g, "pair": g // 2, "candidate": "red" if g % 2 == 0 else "black", "seed": base + g}
            for g in range(GAMES)]


def build_plan(base: int, survey_record_sha256: str) -> Dict[str, Any]:
    """Serialized as base + slot count + the layout, never the exclusive endpoint (outside the block)."""
    return {"kind": PLAN_KIND, "card": EL.CARD, "baseline": EL.BASELINE, "manifest_sha256": MANIFEST_SHA256,
            "base": int(base), "n_slots": N_SLOTS, "survey_record_sha256": survey_record_sha256,
            "games": slot_layout(int(base))}


def check_plan(plan: Any) -> Dict[str, Any]:
    """32 game seeds: the plan exactly its frozen form, every seed ACCOUNTED, none exposed, retired or
    consumed by G3."""
    from . import e4_screen_reference as REG
    from .twixtbot_g3_schedule import CONSUMED_SEEDS

    if not isinstance(plan, dict) or plan.get("kind") != PLAN_KIND or type(plan.get("base")) is not int:
        raise Stop("plan", "not a stage-3 plan with an int base")
    sha = plan.get("survey_record_sha256")
    if not (isinstance(sha, str) and len(sha) == 64) or plan != build_plan(plan["base"], sha):
        raise Stop("plan", "the plan is not the frozen form of its base, survey record and manifest")
    seeds = [g["seed"] for g in plan["games"]]
    if [s for s in seeds if not REG.seed_is_accounted(s)]:
        raise Stop("plan", f"seeds not ACCOUNTED in the shared registry: {[s for s in seeds if not REG.seed_is_accounted(s)]}")
    spent = [s for s in seeds if REG.seed_is_unavailable(s) or s in CONSUMED_SEEDS]
    if spent:
        raise Stop("plan", f"seeds already spent (exposed, retired or consumed): {spent}")
    return plan


def load_plan() -> Tuple[Dict[str, Any], str]:
    if not EL._committed(PLAN):
        raise Stop("plan", f"{PLAN} is absent or not committed")
    raw = (ROOT / PLAN).read_bytes()
    return check_plan(json.loads(raw)), EL._sha(raw)


# ─────────────────────────────── the held-out comparison ───────────────────────────────

def load_heldout(manifest: Mapping[str, Any], root: pathlib.Path = ROOT) -> List[Dict[str, Any]]:
    """Stage 1's sealed held-out rows, each rebuilt; exactly the manifest's games and rows."""
    ev = root / manifest["stage1"]["evidence_dir"]
    if EL._sha((ev / "MANIFEST.sha256").read_bytes()) != manifest["stage1"]["manifest_sha256"]:
        raise Stop("inputs", "the stage-1 evidence manifest is not the bound one")
    f = manifest["stage1"]["rows"]
    if _listed(ev).get(f["file"]) != f["sha256"] or EL._sha((ev / f["file"]).read_bytes()) != f["sha256"]:
        raise Stop("inputs", f"{f['file']} is not the file stage 1's manifest hashed")
    rows = [r for r in EL.load_rows(ev / f["file"]) if r["split"] == "heldout"]
    h = manifest["stage1"]["heldout"]
    if sorted({r["game_id"] for r in rows}) != h["games"] or len(h["games"]) != HELDOUT_GAMES or \
            len(rows) != h["rows"] or EL._jsha([r["row_id"] for r in rows]) != h["row_ids_sha256"]:
        raise Stop("inputs", "the held-out rows are not the manifest's sealed games")
    for r in rows:
        EL.verify_row(r)
    return rows


def pins(manifest: Mapping[str, Any], root: pathlib.Path = ROOT) -> Dict[str, Tuple[str, str]]:
    """The only two checkpoints stage 3 may load: absolute path -> (digest kind, digest)."""
    return {str(root / manifest["stage2"]["candidate"]["path"]): ("sha256", manifest["stage2"]["candidate"]["sha256"]),
            str(root / manifest["incumbent"]["path"]): ("sha1", manifest["incumbent"]["sha1"])}


def verify_checkpoint(path: str, pinned: Mapping[str, Tuple[str, str]]) -> None:
    if path not in pinned:
        raise Stop("checkpoint", f"{path} is not a pinned stage-3 checkpoint")
    kind, want = pinned[path]
    raw = pathlib.Path(path).read_bytes()
    got = EL._sha(raw) if kind == "sha256" else hashlib.sha1(raw).hexdigest()
    if got != want:
        raise Stop("checkpoint", f"{pathlib.Path(path).name}: {kind} {got} is not the pinned {want}")


def load_network(path: str, pinned: Mapping[str, Tuple[str, str]]):
    from .probe_eval import load_network_for_scoring

    verify_checkpoint(path, pinned)
    network, *_ = load_network_for_scoring(path, verbose=False)
    network.eval()
    return network


def teacher_rank(z: Sequence[float], i: int) -> int:
    """1 + the number of legal moves with a strictly higher logit than the teacher's choice."""
    return 1 + sum(x > z[i] for x in z)


def heldout_comparison(rows: Sequence[Mapping[str, Any]], incumbent: Sequence[Sequence[float]],
                       candidate: Sequence[Sequence[float]], beta: float, games: Sequence[str]) -> Dict[str, Any]:
    """Per-game metrics of the incumbent, the control (beta * incumbent logits) and the candidate; the
    criteria on game means, each of the frozen held-out games weighing equally."""
    if len(games) != HELDOUT_GAMES or sorted({r["game_id"] for r in rows}) != sorted(games):
        raise Stop("heldout", f"the rows are not exactly the {HELDOUT_GAMES} frozen held-out games")
    if not (len(rows) == len(incumbent) == len(candidate)) or \
            any(len(r["legal_moves"]) != len(a) or len(a) != len(b) for r, a, b in zip(rows, incumbent, candidate)):
        raise Stop("heldout", "a logit row does not cover its legal moves")
    if not all(math.isfinite(x) for zs in (incumbent, candidate) for z in zs for x in z):
        raise Stop("heldout", "a nonfinite held-out logit")
    control = [[beta * x for x in z] for z in incumbent]
    m = {name: EL.game_metrics(rows, zs) for name, zs in
         (("incumbent", incumbent), ("control", control), ("candidate", candidate))}
    per_game = {}
    for gid in sorted(games):
        idx = [i for i, r in enumerate(rows) if r["game_id"] == gid]
        colour = {rows[i]["teacher_colour"] for i in idx}
        rank = lambda zs: sum(teacher_rank(zs[i], rows[i]["chosen_index"]) for i in idx) / len(idx)
        per_game[gid] = {"rows": len(idx), "teacher_colour": colour.pop() if len(colour) == 1 else sorted(colour),
                         **{f"{k}_ce": m[k][gid]["ce"] for k in m},
                         "incumbent_top1": m["incumbent"][gid]["top1"], "candidate_top1": m["candidate"][gid]["top1"],
                         "incumbent_rank": rank(incumbent), "candidate_rank": rank(candidate),
                         "move_disagreements": sum(EL.top1(incumbent[i]) != EL.top1(candidate[i]) for i in idx)}
    mean = lambda key, ids=tuple(per_game): sum(per_game[g][key] for g in ids) / len(ids)
    keys = ("incumbent_ce", "control_ce", "candidate_ce", "incumbent_top1", "candidate_top1", "incumbent_rank",
            "candidate_rank", "move_disagreements")
    colours = {c: [g for g, v in per_game.items() if v["teacher_colour"] == c] for c in ("red", "black")}
    paired = {g: {"candidate_minus_incumbent_ce": v["candidate_ce"] - v["incumbent_ce"],
                  "candidate_minus_control_ce": v["candidate_ce"] - v["control_ce"],
                  "candidate_minus_incumbent_top1": v["candidate_top1"] - v["incumbent_top1"]}
              for g, v in per_game.items()}
    return {"beta": beta, "per_game": per_game, "paired": paired,
            "game_means": {k: mean(k) for k in keys},
            "colour_means_descriptive": {c: {k: mean(k, ids) for k in keys} for c, ids in colours.items() if ids},
            "passes": EL.heldout_passes(m["candidate"], m["incumbent"], m["control"])}


# ─────────────────────────────── the 32 games ───────────────────────────────

def game_tasks(plan: Mapping[str, Any], manifest: Mapping[str, Any], root: pathlib.Path = ROOT):
    """16 colour-balanced pairs from the runner's own agent builder: candidate red on even games. Every
    task's seed and colours must be the plan's slot."""
    from .eval_runner import AgentSpec, build_agent_pairing_tasks, readout_from_eval_config

    rd = readout_from_eval_config(eval_config())
    cand = AgentSpec(CANDIDATE_ID, str(root / manifest["stage2"]["candidate"]["path"]), rd)
    inc = AgentSpec(INCUMBENT_ID, str(root / manifest["incumbent"]["path"]), rd)
    tasks = build_agent_pairing_tasks(PAIRING, cand, inc, GAMES, base_seed=plan["base"], pairing_index=0)
    got = [{"game": t.game_idx, "pair": t.game_idx // 2,
            "candidate": "red" if t.red_agent.agent_id == CANDIDATE_ID else "black", "seed": t.seed} for t in tasks]
    if got != plan["games"]:
        raise Stop("seeds", "the game tasks are not the plan's seed/colour slots")
    return tasks


def pinned_factory(pinned: Mapping[str, Tuple[str, str]]) -> Callable[[str], Any]:
    """The runner's real loader behind the pins: each checkpoint verified before its one load, then cached."""
    from . import eval_runner as ER

    cache: Dict[str, Any] = {}

    def factory(path: str):
        if path not in cache:
            verify_checkpoint(path, pinned)
            cache[path] = ER._default_evaluator_factory(path)
        return cache[path]
    return factory


def play_games(tasks: Sequence[Any], run_dir: pathlib.Path, factory: Callable[[str], Any],
               log: Callable[[Dict[str, Any]], None] = lambda e: None) -> List[Any]:
    """One game at a time through `run_game_tasks` (workers 1), replays captured, each result logged as it
    finishes; any integrity fault raises on the game that produced it."""
    from .eval_runner import run_game_tasks

    results = []
    for t in tasks:
        log({"event": "game_start", "game": t.game_idx, "seed": t.seed, "red": t.red_agent.agent_id})
        [r] = run_game_tasks([t], 1, eval_config(), evaluator_factory=factory, replay_dir=str(run_dir / REPLAYS))
        results.append(r)
        log({"event": "game_end", "game": r.game_idx, "reason": r.reason, "n_moves": r.n_moves,
             "winner_agent_id": r.winner_agent_id})
    return results


def candidate_score(r) -> float:
    return r.red_score if r.red_agent_id == CANDIDATE_ID else r.black_score


def screen_summary(results: Sequence[Any]) -> Dict[str, Any]:
    """The disaster screen over a COMPLETE cohort: 16 pair means of the candidate's score, their mean and
    the unclipped Hoeffding interval; VETO only when it lies wholly below 0.5."""
    if sorted(r.game_idx for r in results) != list(range(GAMES)):
        raise Stop("games", f"incomplete cohort: {len(results)} of {GAMES} games -- no strength rate")
    s = {r.game_idx: candidate_score(r) for r in results}
    pairs = [(s[2 * k] + s[2 * k + 1]) / 2 for k in range(PAIRS)]
    mean = sum(pairs) / PAIRS
    caps = sum(r.reason == "state_cap" for r in results)
    counts = {"candidate_wins": sum(r.winner_agent_id == CANDIDATE_ID for r in results),
              "candidate_losses": sum(r.winner_agent_id == INCUMBENT_ID for r in results),
              "draws_board_full": sum(r.reason == "board_full" for r in results), "caps": caps}
    lo, hi = mean - HALF_WIDTH, mean + HALF_WIDTH
    outcome = "UNINTERPRETABLE" if caps > GAMES // 2 else "VETO" if hi < PARITY else "NOT_VETOED"
    return {**counts, "pair_means": pairs, "mean_pair_score": mean, "half_width": HALF_WIDTH,
            "interval_unclipped": [lo, hi], "outcome": outcome}


# ─────────────────────────────── the stage, in a child process ───────────────────────────────

def stage3(run_dir: pathlib.Path, rec: Dict[str, Any]) -> None:
    from .eval_integrity import IntegrityError, validate_result_set

    steps = rec["steps"] = EL._Steps(run_dir)
    plan, plan_sha = load_plan()
    steps["plan"] = {"sha256": plan_sha, "n_slots": plan["n_slots"]}
    if not EL._committed(MANIFEST):
        raise Stop("manifest", f"{MANIFEST} is absent or not committed")
    manifest = check_manifest()
    steps["manifest_sha256"] = MANIFEST_SHA256
    steps["environment"] = check_environment(manifest)
    rows = load_heldout(manifest)
    steps["heldout_rows"] = {"rows": len(rows), "games": manifest["stage1"]["heldout"]["games"]}
    pinned = pins(manifest)
    records = S2.records_of(rows)
    inc_logits, _ = S2.forward_rows(load_network(str(ROOT / manifest["incumbent"]["path"]), pinned), records)
    cand_logits, _ = S2.forward_rows(load_network(str(ROOT / manifest["stage2"]["candidate"]["path"]), pinned), records)
    cmp = heldout_comparison(rows, inc_logits, cand_logits, manifest["stage2"]["control"]["beta"],
                             manifest["stage1"]["heldout"]["games"])
    steps["heldout"] = cmp
    if not cmp["passes"]:
        raise HeldoutFailed("heldout", "the held-out criteria failed: no evaluation game is played")
    tasks = game_tasks(plan, manifest)
    steps["tasks"] = {"games": len(tasks), "seeds_sha256": EL._jsha([t.seed for t in tasks])}
    try:                                                         # the runner's integrity faults are the card's stops
        results = play_games(tasks, run_dir, pinned_factory(pinned), log=lambda e: EL.progress(run_dir, e))
        validate_result_set(results, tasks, CANDIDATE_ID, INCUMBENT_ID)
    except IntegrityError as e:
        raise Stop("integrity", str(e)) from None
    seed_of = {t.task_id: t.seed for t in tasks}
    steps["games"] = [{**asdict(r), "seed": seed_of[r.task_id]} for r in results]
    steps["screen"] = screen_summary(results)


def _child(run_dir: pathlib.Path) -> int:
    """The record boundary covers the conclusion: record.json is written on every exit."""
    _require_gate()
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "card": EL.CARD, "baseline": EL.BASELINE, "verdict": None,
                           "steps": {}}
    code = EXIT_STOP

    def ceiling(signum, _frame):
        raise Stop("ceiling", f"signal {signum} from the parent's {EXECUTION_S} s ceiling")
    previous = signal.signal(signal.SIGTERM, ceiling)
    try:
        stage3(run_dir, rec)
        rec["verdict"], code = "SCREEN_COMPLETED", EXIT_COMPLETED
    except HeldoutFailed as e:
        rec["stop"] = {"step": e.step, "detail": e.detail}
        rec["verdict"], code = "STOP_HELDOUT", EXIT_STOP_HELDOUT
    except Stop as e:
        rec["stop"] = {"step": e.step, "detail": e.detail}
        rec["verdict"] = "STOP"
    except BaseException as e:                                   # noqa: BLE001 -- recorded below, then re-raised
        rec["stop"] = {"step": "unexpected", "detail": f"{type(e).__name__}: {e}"}
        rec["verdict"] = "STOP"
        EL._write_json(run_dir / "record.json", rec)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
    EL._write_json(run_dir / "record.json", rec)
    return code


# ─────────────────────────────── the parent: claim, ceiling, finalization ───────────────────────────────

def exit_code_for(record: pathlib.Path, status: str, child_exit: Optional[int], reaped: Optional[bool] = True) -> int:
    """ONE answer, as stage 1's: an unverified cleanup first, with its own status; then the recorded verdict
    decides and the child's exit must agree with it."""
    if reaped is not True:
        return EXIT_CLEANUP_UNVERIFIED
    if status == "TIMEOUT":
        return EXIT_TIMEOUT
    verdict = EL._verdict(record)
    if verdict not in VERDICT_EXIT or child_exit != VERDICT_EXIT[verdict]:
        return EXIT_STOP
    return VERDICT_EXIT[verdict]


def run() -> int:
    """THE entry point, stage 2's parent pattern: gate, clean tree, run once; the child under the 120-minute
    ceiling in its own process group; finalized and copied to evidence ONLY once that group is reaped."""
    _require_gate()
    if EL._git("status", "--porcelain"):
        raise Refused("the working tree is dirty: run only from a committed state")
    run_dir, evidence = ROOT / RUN_DIR, ROOT / EVIDENCE_DIR
    if evidence.exists():
        raise Refused(f"{EVIDENCE_DIR} exists: stage 3 runs once")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir()                                          # the claim: exclusive
    except FileExistsError:
        raise Refused(f"{RUN_DIR} exists: stage 3 runs once") from None
    t0 = time.time()
    fin: Dict[str, Any] = {"head": EL._git("rev-parse", "HEAD"), "python": sys.version,
                           "module_sha256": EL._sha(pathlib.Path(__file__).read_bytes()),
                           "gate_sha256": EL._sha(pathlib.Path(GATE.__file__).read_bytes()),
                           "started_unix": t0, "ceiling_s": EXECUTION_S}
    EL._write_json(run_dir / "claim.json", fin)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    EL._write_new(run_dir / "environment.txt", (f"python {sys.version}\nexecutable {sys.executable}\n"
                                                f"pip freeze exit {freeze.returncode}\n{freeze.stdout}").encode())
    status, child_exit, reaped, cleanup_error = "UNKNOWN", None, None, None
    try:
        with open(run_dir / "child.stdout.txt", "xb") as out, open(run_dir / "child.stderr.txt", "xb") as err:
            status, child_exit, reaped, cleanup_error = EL.run_group(
                [sys.executable, "-m", MODULE, "_child", "--run-dir", str(run_dir)],
                stdout=out, stderr=err, timeout=EXECUTION_S)
    finally:
        code = exit_code_for(run_dir / "record.json", status, child_exit, reaped)
        fin.update(status=status, child_exit=child_exit, group_reaped=reaped, cleanup_error=cleanup_error,
                   exit_code=code, elapsed_s=round(time.time() - t0, 3))
        if reaped is not True:
            EL._write_json(ROOT / CLEANUP_FAILED, {**fin, "finalized": False, "run_dir": RUN_DIR,
                                                   "detail": "the process group's end was not verified (not reaped, "
                                                             "the check failed, or it never ran): nothing hashed, copied or "
                                                             "finalized"})
    if reaped is not True:
        return code
    fin.update(verdict=EL._verdict(run_dir / "record.json"),
               files={str(p.relative_to(run_dir)): EL._sha(p.read_bytes())
                      for p in sorted(run_dir.rglob("*")) if p.is_file()})
    EL._write_json(run_dir / "finalized.json", fin)
    shutil.copytree(run_dir, evidence)
    manifest = "".join(f"{EL._sha(p.read_bytes())}  {p.relative_to(evidence)}\n"
                       for p in sorted(evidence.rglob("*")) if p.is_file())
    EL._write_new(evidence / "MANIFEST.sha256", manifest.encode())
    return code


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="External-engine learning, stage 3 (gated).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    p = sub.add_parser("_child")
    p.add_argument("--run-dir", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "run":
            return run()
        return _child(pathlib.Path(a.run_dir))
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
