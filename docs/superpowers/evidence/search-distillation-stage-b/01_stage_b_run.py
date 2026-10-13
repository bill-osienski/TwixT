"""SEARCH DISTILLATION STAGE B -- the pilot's phases, ONE per invocation: G corpus, P selection, L1 held-out
labels, Q the gate, L2 training labels, T training + temperature + held-out verdict, K seed closing.
Card: docs/superpowers/2026-10-12-search-distillation-stage-b-card.md. A copy of Stage A's run script
(evidence/search-distillation-stage-a/01_stage_a_run.py); its diff is limited to: the phase argument and bodies,
the 441-slot plan, reading only COMMITTED outputs of earlier phases that COMPLETED, and the STOP class.

Each phase writes create-only into <PHASE>/ beside this file. Run ONLY through 02_stage_b_wrapper.sh <PHASE>
(the phase's cap and the outer status live there). The testable logic lives in search_distillation_stage_b.
Exit 0 COMPLETED / 1 a repeat run / 2 STOP (NO VERDICT) / 3 STOP_ERROR / 4 REFUSED.

    python -B <this file> --phase G|P|L1|Q|L2|T|K --checkpoint <absolute path> --candidates <absolute dir>
"""
import dataclasses
import hashlib
import json
import pathlib
import platform
import statistics
import subprocess
import sys
import time

from scripts.GPU.alphazero import search_distillation_stage_b as B

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REL = "docs/superpowers/evidence/search-distillation-stage-b"
CARD = "docs/superpowers/2026-10-12-search-distillation-stage-b-card.md"
SHA1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"          # calib020_0001
EXIT = {"COMPLETED": 0, "STOP": 2, "STOP_ERROR": 3, "REFUSED": 4}
PHASES = {"G": (), "P": ("G",), "L1": ("G", "P"), "Q": ("G", "P", "L1"), "L2": ("G", "P", "Q"),
          "T": ("G", "P", "L1", "Q", "L2"), "K": ()}      # each phase's committed, COMPLETED inputs
MODEL = ("G", "L1", "L2", "T")
SIMS = {"t1600": B.SIMS_HI, "t400": B.SIMS_LO, "repeat": B.SIMS_HI}
PHASE = sys.argv[sys.argv.index("--phase") + 1] if "--phase" in sys.argv else None
if PHASE not in PHASES:
    sys.exit(EXIT["REFUSED"])
OUT = HERE / PHASE
T0 = time.monotonic()
log_fh = (OUT / "progress.jsonl").open("x")               # a second run stops here, exit 1


def log(e):
    log_fh.write(json.dumps({**e, "t": round(time.monotonic() - T0, 3)}) + "\n")
    log_fh.flush()


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


class Refused(Exception):
    pass


def write(name, text):
    with (OUT / name).open("x") as fh:
        fh.write(text)


def committed(rel):
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True)
    if tracked.returncode or git("status", "--porcelain", "--", rel):
        raise Refused(f"{rel} is not committed and unmodified")
    return (ROOT / rel).read_text()


def phase_input(phase, name):
    """An earlier phase's output, read only if that phase terminated cleanly with exit 0, COMPLETED, and is committed."""
    d = f"{REL}/{phase}"
    B.require_terminated(json.loads(committed(f"{d}/outer.json")), True)      # exit 0, no survivor
    if json.loads(committed(f"{d}/record.json")).get("verdict") != "COMPLETED":
        raise Refused(f"phase {phase} did not complete")
    return committed(f"{d}/{name}")


def jsonl(text):
    return [json.loads(line) for line in text.splitlines()]


def seed_base():
    from scripts.GPU.alphazero import e4_screen_reference as REG

    plan = json.loads(committed(f"{REL}/plan.json"))
    if plan.get("n_slots") != B.N_SLOTS or type(plan.get("base")) is not int:
        raise Refused(f"the plan is not exactly {B.N_SLOTS} slots from an int base")
    for slot in range(B.N_SLOTS):
        s = plan["base"] + slot
        if not REG.seed_is_accounted(s) or REG.seed_is_unavailable(s):
            raise Refused(f"seed slot {slot} is not ACCOUNTED and available")
    return plan["base"]


def modules_check(rec):
    """Repository modules loaded so far (this script included) must be committed, unmodified, from THIS tree."""
    ours = {n: m.__file__ for n, m in list(sys.modules.items())
            if (n == "__main__" or n.split(".")[0] in ("scripts", "tests")) and getattr(m, "__file__", None)}
    outside = sorted(n for n, f in ours.items() if not pathlib.Path(f).resolve().is_relative_to(ROOT))
    if outside:
        raise Refused(f"repository modules loaded from outside this tree: {outside}")
    rel = sorted(str(pathlib.Path(f).resolve().relative_to(ROOT)) for f in ours.values())
    rec["modules"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in rel}
    if git("status", "--porcelain", "--untracked-files=all", "--", *rel):
        raise Refused("a loaded repository module is uncommitted or differs from HEAD")


def load_games():
    games = json.loads(phase_input("G", "games.json"))
    if [x["game"] for x in games] != list(range(B.N_GAMES)):
        raise B.Stop("games.json is not the 200 scheduled games in slot order")
    return [x["moves"] for x in games]


def evaluator(rec, ckpt):
    from scripts.GPU.alphazero import eval_runner as ER

    t = time.monotonic()
    ev = ER._default_evaluator_factory(str(ckpt))
    rec["load_seconds"] = round(time.monotonic() - t, 3)
    log({"event": "loaded", "seconds": rec["load_seconds"]})
    return ev


def full_row(games, row, keys):
    """A stored label row (already checked by check_label_rows) with its full legal support restored."""
    s = B.replay_state(games[row["game"]][:row["ply"]])
    return {k: B.full_counts(s, row[k], SIMS[k], row["legal_sha"])[1] for k in keys}, s


# ─── the phases ───

def phase_G(rec, base, ckpt):
    """Card §3: 200 games, the incumbent at 400 v itself at 400, the frozen protocol; outcome-blind progress."""
    from scripts.GPU.alphazero import eval_runner as ER
    from scripts.GPU.alphazero.eval_integrity import validate_result_set

    ev = evaluator(rec, ckpt)

    def factory(path):
        assert path == str(ckpt), path
        return ev

    cfg = ER.EvalConfig(**B.PROTOCOL)
    rd = ER.readout_from_eval_config(cfg)
    a, b = (ER.AgentSpec(i, str(ckpt), rd, mcts_sims=B.SIMS_LO) for i in ("a400", "b400"))
    rec["protocol"] = {"PROTOCOL": dict(B.PROTOCOL), "eval_config": dataclasses.asdict(cfg),
                       "agents": [dataclasses.asdict(a), dataclasses.asdict(b)]}
    tasks = ER.build_agent_pairing_tasks("stage-b", a, b, B.N_GAMES, base_seed=base + B.GAME)
    rec["games"], results, games = [], [], []
    for task in tasks:
        slot = task.seed - base
        log({"event": "game_start", "game": task.game_idx, "slot": slot})
        t = time.monotonic()
        [r] = ER.run_game_tasks([task], 1, cfg, evaluator_factory=factory, replay_dir=str(OUT / "replays"))
        secs = round(time.monotonic() - t, 3)
        results.append(r)
        plies = json.loads(pathlib.Path(r.replay_path).read_text())["moves"]
        if [m["ply"] for m in plies] != list(range(r.n_moves)):
            raise B.Stop(f"game {task.game_idx}: the replay is not the game's {r.n_moves} plies in order")
        moves = [[m["row"], m["col"]] for m in plies]
        games.append({"game": task.game_idx, "slot": slot, "moves": moves, "transcript_sha256": B._sha(moves)})
        rec["games"].append({**dataclasses.asdict(r), "slot": slot, "seconds": secs})
        log({"event": "game_end", "game": task.game_idx, "slot": slot, "seconds": secs,      # outcome-blind
             "n_moves": r.n_moves})
    validate_result_set(results, tasks, "a400", "b400")
    rec["result_set_check"] = "validate_result_set passed"
    write("games.json", json.dumps(games, separators=(",", ":")) + "\n")      # move lists only: no outcome


def phase_P(rec, base):
    """Card §4: select_start logged BEFORE each game's draw; then the leakage rule over all 200 games."""
    games = load_games()
    selected = []
    for g in range(B.N_GAMES):
        log({"event": "select_start", "game": g})
        selected.append(B.select_game(games[g], base + B.LABEL + g))
    kept, dropped = B.leakage_filter(dict(enumerate(selected)))
    out = {"selected": selected, "kept": [kept[g] for g in sorted(kept)], "dropped": dropped}
    rec["positions"] = {"selected": sum(map(len, selected)), "heldout_kept": sum(map(len, out["kept"])),
                        "dropped": dropped, "empty_heldout_games": sum(1 for k in out["kept"] if not k)}
    write("selection.json", json.dumps(out, indent=1) + "\n")


def phase_labels(rec, base, ckpt, heldout):
    """Card §5/§11: L1 (held out) or L2 (training; refused unless gate Q QUALIFIED, before the model loads)."""
    q = None if heldout else json.loads(phase_input("Q", "q.json"))
    if not heldout:
        B.require_qualified(q)
    games, sel = load_games(), json.loads(phase_input("P", "selection.json"))["selected"]
    rows = B.label_phase(evaluator(rec, ckpt), games, sel, base, heldout, log, q=q)
    rec["labels"] = len(rows)
    write("labels.jsonl", "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))


def phase_Q(rec):
    """Card §6, once, on the committed L1 labels: Q1, Q2, and the descriptive medians and repeat agreement."""
    games, sel = load_games(), json.loads(phase_input("P", "selection.json"))
    rows = jsonl(phase_input("L1", "labels.jsonl"))
    B.check_label_rows(rows, dict(enumerate(sel["selected"])), games, heldout=True)
    by = {(r["game"], r["ply"]): r for r in rows}
    per_game, d, r_, agree = [], [], [], []
    for h, kept in enumerate(sel["kept"]):
        g = 5 * h + 4
        pos = [full_row(games, by[(g, p["ply"])], ("t1600", "t400", "repeat"))[0] for p in kept]
        for x in pos:
            d.append(B.tv(B.probs(x["t1600"]), B.probs(x["t400"])))
            r_.append(B.tv(B.probs(x["t1600"]), B.probs(x["repeat"])))
            agree.append(bool(B.argmax_set(x["t1600"]) & B.argmax_set(x["repeat"])))
        per_game.append(B.q_game(pos))
    q = {**B.gate_q(per_game), "per_game": per_game, "positions": len(d),
         "median_d": statistics.median(d) if d else None, "median_r": statistics.median(r_) if r_ else None,
         "repeat_agreement": sum(agree) / len(agree) if agree else None}
    rec["q_verdict"] = q["verdict"]
    write("q.json", json.dumps(q, indent=1) + "\n")


def phase_T(rec, base, ckpt, cand_dir):
    """Card §7/§8: both arms (gate Q checked first), surface checks, the soft temperature fit on the TRAINING
    positions' T1600 targets, then the held-out measures and the verdict, once."""
    from scripts.GPU.alphazero import external_learning_stage2 as EL2
    from scripts.GPU.alphazero.probe_eval import load_network_for_scoring

    q = json.loads(phase_input("Q", "q.json"))
    B.require_qualified(q)
    if cand_dir is None or not cand_dir.is_absolute():
        raise Refused("--candidates must name a directory by absolute path")
    games, sel = load_games(), json.loads(phase_input("P", "selection.json"))
    train_rows, held_rows = jsonl(phase_input("L2", "labels.jsonl")), jsonl(phase_input("L1", "labels.jsonl"))
    B.check_label_rows(train_rows, dict(enumerate(sel["selected"])), games, heldout=False)  # before any network
    B.check_label_rows(held_rows, dict(enumerate(sel["selected"])), games, heldout=True)    # or the training seed
    held = {(r["game"], r["ply"]): r for r in held_rows}
    student, control = [], []
    for r in train_rows:
        s = B.replay_state(games[r["game"]][:r["ply"]])
        student.append(B.training_record(s, r["t1600"], B.SIMS_HI, r["legal_sha"]))
        control.append(B.training_record(s, r["t400"], B.SIMS_LO, r["legal_sha"]))

    def load_network():
        if hashlib.sha1(ckpt.read_bytes()).hexdigest() != SHA1:
            raise B.Stop("the incumbent file is no longer the pinned checkpoint")
        net, *_ = load_network_for_scoring(str(ckpt), verbose=False)
        net.eval()
        return net

    schedule, nets = B.train_arms(q, load_network, student, control, base + B.TRAIN, log, EL2.train_head)
    rec["training"] = {"n_train": len(student), "schedule_sha256": B._sha(schedule.tolist()),
                       **{arm: {"first_loss": st[0]["loss"], "last_loss": st[-1]["loss"],
                                "max_grad_norm": max(x["grad_norm"] for x in st)} for arm, (_, st) in nets.items()}}
    cand_dir.mkdir(parents=True, exist_ok=True)
    arms, rec["candidates"] = {}, {}
    for arm, (net, _) in nets.items():
        path = cand_dir / f"stage-b-{arm}.safetensors"
        rec["candidates"][arm] = {"path": str(path), "sha256": EL2.save_candidate(net, path),
                                  "surface": EL2.surface_check(ckpt, path)}
        arms[arm] = EL2.reload_check(net, path)
    arms["incumbent"] = load_network()
    try:
        fit = B.fit_beta_soft(EL2.forward_rows(arms["incumbent"], student)[0],
                              [B.probs(r.visit_counts) for r in student])
    except ValueError as e:
        raise B.Stop(f"temperature fit: {e}")
    per_game = []
    for h, kept in enumerate(sel["kept"]):
        g = 5 * h + 4
        recs = []
        for p in kept:
            r = held[(g, p["ply"])]
            recs.append(B.training_record(B.replay_state(games[g][:p["ply"]]), r["t1600"], B.SIMS_HI, r["legal_sha"]))
        if not recs:
            per_game.append(None)                                        # an empty game (card §8)
            continue
        z = {arm: EL2.forward_rows(net, recs)[0] for arm, net in arms.items()}
        z["temperature"] = [[fit["beta"] * x for x in row] for row in z["incumbent"]]   # legal moves only: re-masked
        per_game.append(B.game_measures([{"target": x.visit_counts, "logits": {a: z[a][i] for a in B.ARMS}}
                                         for i, x in enumerate(recs)]))
    verdict = B.learning_verdict(per_game)
    present = [m["kl"] for m in per_game if m is not None]
    mean_kl = {a: sum(k[a] for k in present) / len(present) if present else None for a in B.ARMS}
    gain = (mean_kl["incumbent"] - mean_kl["student"]) / mean_kl["incumbent"] if present and mean_kl["incumbent"] else None
    rec["verdict_stage_b"] = verdict["verdict"]
    write("verdict.json", json.dumps({"verdict": verdict["verdict"], "learning": verdict, "q": q["verdict"],
                                      "beta": fit, "mean_kl": mean_kl, "relative_kl_gain_vs_incumbent": gain,
                                      "per_game": per_game, "strength_verdict": None}, indent=1) + "\n")


def phase_K(rec, base):
    """Card §9: EXPOSED = the union of slots any phase that ran logged as used (each phase's committed outer
    record must show no survivor; a stopped or timed-out phase is accepted); RETIRED = the whole block."""
    events, ran = [], []
    for ph in ("G", "P", "L1", "Q", "L2", "T"):
        if (HERE / ph / "progress.jsonl").exists():
            B.require_terminated(json.loads(committed(f"{REL}/{ph}/outer.json")), False)   # any exit, no survivor
            events += jsonl(committed(f"{REL}/{ph}/progress.jsonl"))
            ran.append(ph)
    exposed, retired = B.closing_intervals(base, events)
    rec["closing"] = {"phases": ran, "exposed": exposed, "retired": retired}
    write("closing.json", json.dumps(rec["closing"], indent=1) + "\n")


def main(rec, argv):
    arg = lambda k: pathlib.Path(argv[argv.index(k) + 1]) if k in argv else None
    for need in PHASES[PHASE]:
        phase_input(need, "record.json")
    base = None if PHASE == "Q" else seed_base()
    ckpt = None
    if PHASE in MODEL:
        ckpt = arg("--checkpoint")
        if ckpt is None or not ckpt.is_absolute() or not ckpt.is_file():
            raise Refused("--checkpoint must name an existing file by absolute path")
        sha1 = hashlib.sha1(ckpt.read_bytes()).hexdigest()
        if sha1 != SHA1:
            raise Refused(f"checkpoint SHA-1 {sha1} is not the pinned calib020_0001")
        rec["checkpoint"] = {"path": str(ckpt), "sha1_verified": sha1}
        import mlx.core
        import numpy

        rec["environment"] = {"python": sys.version, "platform": platform.platform(), "mlx": mlx.core.__version__,
                              "numpy": numpy.__version__, "executable": sys.executable}
        # every repository module a model phase runs, loaded BEFORE the check
        from scripts.GPU.alphazero import (eval_integrity, eval_runner, external_learning_stage2,  # noqa: F401
                                           local_evaluator, probe_eval, self_play, trainer)
    modules_check(rec)
    if PHASE == "G":
        phase_G(rec, base, ckpt)
    elif PHASE == "P":
        phase_P(rec, base)
    elif PHASE in ("L1", "L2"):
        phase_labels(rec, base, ckpt, PHASE == "L1")
    elif PHASE == "Q":
        phase_Q(rec)
    elif PHASE == "T":
        phase_T(rec, base, ckpt, arg("--candidates"))
    else:
        phase_K(rec, base)
    modules_check(rec)                                    # again: anything loaded lazily
    return "COMPLETED"


def _is_stop(e):
    stops = [B.Stop] + [getattr(sys.modules.get(m), c) for m, c in
                        (("scripts.GPU.alphazero.external_learning", "Stage1Stop"),
                         ("scripts.GPU.alphazero.eval_integrity", "IntegrityError")) if m in sys.modules]
    return isinstance(e, tuple(stops))


if __name__ == "__main__":
    rec = {"kind": "search-distillation-stage-b-record/1", "phase": PHASE, "card": CARD,
           "head": git("rev-parse", "HEAD"), "strength_verdict": None}
    log({"event": "start", "phase": PHASE, "head": rec["head"]})
    try:
        verdict = main(rec, sys.argv)
    except (Refused, B.Refused) as e:
        verdict, rec["reason"] = "REFUSED", str(e)
    except Exception as e:
        verdict, rec["reason"] = ("STOP" if _is_stop(e) else "STOP_ERROR"), f"{type(e).__name__}: {e}"
    rec["verdict"], rec["total_seconds"] = verdict, round(time.monotonic() - T0, 3)
    log({"event": "end", "verdict": verdict})
    write("record.json", json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    sys.exit(EXIT[verdict])
