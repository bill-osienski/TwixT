"""SEARCH DISTILLATION D2 -- STAGE 3: one training run on the Stage 2 corpus. BUILT; run only through its wrapper.

Card: `docs/superpowers/2026-10-22-search-distillation-d2-stage3-card.md`, pinned below by sha256. Nothing runs on
import. `run` refuses unless `search_distillation_d2_train_authorization.SEARCH_DISTILLATION_D2_TRAIN_AUTHORIZED` is
open, read at call time.

WHAT THE RUN DOES: verifies the Stage 2 corpus against the pinned `corpus.json` hash (every phase reopened through the
strict reader, its context and completion marker matching the manifest), rebuilds the training records of the 108
training games and the held-out records of the 12 held-out games (the corpus's own split, unchanged), loads
calib020_0001 twice -- TEACHER and STUDENT, both in evaluation mode, both equal to the pinned state digest -- and:
  1. BASELINE: the teacher on the held-out records (policy cross-entropy; the teacher values for the drift check);
  2. TRAINS the student once with search_distillation_d2.train_student at the frozen recipe (401 updates, batch 64,
     Adam 1e-4 with MLX's resolved defaults read back and checked, clip 1.0, anchor 1.0, L2 1e-4, draw-order seed 0)
     on the encoder and policy head only; the teacher, the value head and every BatchNorm statistic unchanged;
  3. SAVES the final candidate create-only (also when training stopped: a failed candidate is kept as a research
     artifact), reloads it strictly into a fresh network and requires its state digest to be the saved one;
  4. FINAL CHECK on the held-out records with the RELOADED candidate. The held-out records are read only in 1 and 4:
     no intermediate evaluation, selection or retuning.

VERDICT: PASS only if training finished with every check, every metric is finite, held-out policy cross-entropy is
STRICTLY below the teacher's, and absolute value drift from the teacher on the held-out records has mean <= 0.05 and
95th percentile <= 0.15. Otherwise STOP. A PASS establishes fit and bounded value drift, not stronger play. A broken
input, identity or reload, an exception or SIGTERM (the wrapper's 30-minute ceiling) is a VOID. No retry.

    python -m scripts.GPU.alphazero.search_distillation_d2_train run --checkpoint <absolute path to calib020_0001>
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import math
import os
import pathlib
import platform
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from . import search_distillation_d2_measure as M
from . import search_distillation_d2_train_authorization as GATE

ROOT = M.ROOT
RECORD_KIND = "search-distillation-d2-train-record/1"
CARD = "docs/superpowers/2026-10-22-search-distillation-d2-stage3-card.md"
CARD_SHA256 = "cdc77f34e434645e2fd26e205c663e5afe09cf7808be8a290552564071336072"
EVIDENCE_DIR = "docs/superpowers/evidence/search-distillation-d2-train"
CORPUS_JSON = "docs/superpowers/evidence/search-distillation-d2-generate/corpus.json"
CANDIDATE_PARENT = pathlib.Path.home() / "twixt-cache" / "search-distillation-d2"     # outside the repository
ADAM_DEFAULTS = {"betas": [0.9, 0.999], "eps": 1e-8, "bias_correction": False}
FROZEN = {"corpus_sha256": "a9a425f28c296e36ba334e55bcd26c56164e982145f11a09427436f5351edeb0",
          "teacher_digest": "ae0e71fc395ce6b95d24f9e696f02870c7b3bfc6f8ea83f9d5d0a85c4979816b",
          "training_games": 108, "held_out_games": 12, "training_records": 6403, "held_out_records": 617,
          "passes": 4, "steps": 401, "batch_size": 64, "learning_rate": 1e-4, "anchor_weight": 1.0,
          "l2_weight": 1e-4, "clip_norm": 1.0, "seed": 0,
          "value_abs_mean_max": 0.05, "value_abs_p95_max": 0.15}
STEP_LOG_EVERY = 25
EXIT_PASS, EXIT_STOP, EXIT_VOID, EXIT_REFUSED, EXIT_USAGE = 0, 6, 3, 4, 2
Void, Terminated = M.Void, M.Terminated


class Refused(Exception):
    """The run never started: gate closed, card not the pinned bytes, an output exists, or a bad argument."""


def _require_gate() -> None:
    if GATE.SEARCH_DISTILLATION_D2_TRAIN_AUTHORIZED is not True:
        raise Refused("SEARCH_DISTILLATION_D2_TRAIN_AUTHORIZED is closed: training is not authorized")


@dataclasses.dataclass
class Seams:
    """What the run reads from outside this module. TEST seams; `run` passes `_real_seams` only."""
    read_corpus_json: Callable[[], bytes]
    checkpoint_sha1: Callable[[], str]
    load_network: Callable[[], Any]                    # a FRESH load of the incumbent on every call
    make_network_for: Callable[[str], Any]             # a fresh network of the architecture saved at a path
    loaded_modules: Callable[[], Dict[str, str]]


def recipe_for(spec: Mapping[str, Any]):
    """The frozen recipe; its update count must be `passes` passes over the training records at the batch size."""
    from . import search_distillation_d2 as D
    steps = math.ceil(spec["passes"] * spec["training_records"] / spec["batch_size"])
    if steps != spec["steps"]:
        raise Void("recipe", f"{spec['steps']} updates is not {spec['passes']} passes of "
                             f"{spec['training_records']} records at batch {spec['batch_size']} ({steps})")
    return D.Recipe(steps=spec["steps"], batch_size=spec["batch_size"], learning_rate=spec["learning_rate"],
                    anchor_weight=spec["anchor_weight"], l2_weight=spec["l2_weight"], clip_norm=spec["clip_norm"],
                    seed=spec["seed"])


def read_corpus(spec: Mapping[str, Any], raw: bytes) -> Tuple[list, list, Dict[str, Any]]:
    """(training records, held-out records, summary) from the pinned manifest; any disagreement is a VOID."""
    from . import search_distillation_d2 as D
    from . import search_distillation_d2_generate as G
    from . import training_replay_capture as C

    sha = M._sha(raw)
    if sha != spec["corpus_sha256"]:
        raise Void("corpus", f"corpus.json is {sha}, not the pinned {spec['corpus_sha256']}")
    corpus = json.loads(raw)
    if corpus.get("kind") != G.CORPUS_KIND or corpus.get("eligible_for_training") is not True:
        raise Void("corpus", "the manifest is not an eligible D2 corpus")
    if corpus.get("model_digest") != spec["teacher_digest"]:
        raise Void("corpus", "the corpus was not generated by the pinned teacher")
    games = [g for ph in corpus["phases"] for g in ph["games"]]
    train_ids, held_ids = corpus["training_games"], corpus["held_out_games"]
    if (sorted(games) != list(range(len(games))) or sorted(train_ids + held_ids) != sorted(games)
            or held_ids != [g for g in games if G.held_out(g)]):
        raise Void("split", "the manifest's split is not the corpus's slot % 10 == 9 split over all its games")
    if (len(train_ids), len(held_ids)) != (spec["training_games"], spec["held_out_games"]):
        raise Void("split", f"{len(train_ids)} training / {len(held_ids)} held-out games, not the card's")
    held = set(held_ids)
    training, held_out, phases = [], [], []
    cdir = pathlib.Path(corpus["corpus_dir"])
    for ph in corpus["phases"]:
        d = cdir / ph["dir"]
        for f, key in (("context.json", "context_sha256"), ("finalized.json", "finalized_sha256")):
            if M._sha((d / f).read_bytes()) != ph[key]:
                raise Void("corpus", f"{ph['dir']}/{f} differs from the manifest")
        ctx, envelopes = C.load_phase(str(d))                       # strict: receipts and every game file by hash
        if ctx["models"][M.MODEL_ID]["state_digest"] != spec["teacher_digest"]:
            raise Void("corpus", f"{ph['dir']}: the capture context names another model state")
        if [e["game_id"] for e in envelopes] != ph["games"]:
            raise Void("corpus", f"{ph['dir']}: the games are not the manifest's")
        n = 0
        for env in envelopes:
            records, _ids = D.records_from_envelopes([env])
            (held_out if env["game_id"] in held else training).extend(records)
            n += len(records)
        if n != ph["records"]:
            raise Void("corpus", f"{ph['dir']}: {n} records, the manifest says {ph['records']}")
        phases.append({"dir": ph["dir"], "games": len(envelopes), "records": n})
    if (len(training), len(held_out)) != (spec["training_records"], spec["held_out_records"]):
        raise Void("split", f"{len(training)} training / {len(held_out)} held-out records, not the card's")
    return training, held_out, {"corpus_json_sha256": sha, "corpus_dir": str(cdir), "phases": phases,
                                "training_games": train_ids, "held_out_games": held_ids,
                                "training_records": len(training), "held_out_records": len(held_out)}


def judge(spec: Mapping[str, Any], baseline: Mapping[str, float], final: Mapping[str, float] | None,
          training_failure: str | None) -> Dict[str, bool]:
    """The card's criteria, each one named. NaN compares False, so a non-finite metric can never pass."""
    final = final or {}
    values = [*baseline.values(), *final.values()]
    return {"training_finished_with_every_check": training_failure is None,
            "metrics_finite": bool(final) and all(math.isfinite(v) for v in values),
            "held_out_policy_ce_strictly_lower": final.get("policy_ce", math.nan) < baseline["policy_ce"],
            "value_abs_mean_within_limit": final.get("value_abs_mean", math.nan) <= spec["value_abs_mean_max"],
            "value_abs_p95_within_limit": final.get("value_abs_p95", math.nan) <= spec["value_abs_p95_max"]}


def _execute(spec: Mapping[str, Any], seams: Seams, candidate: pathlib.Path, rec: Dict[str, Any], log) -> bool:
    import mlx.core as mx
    import mlx.optimizers as optim

    from . import search_distillation_d2 as D

    recipe = recipe_for(spec)
    expected_adam = D.adam_settings(optim.Adam(learning_rate=recipe.learning_rate))
    if {k: expected_adam[k] for k in ADAM_DEFAULTS} != ADAM_DEFAULTS:
        raise Void("optimizer", f"MLX's Adam resolves to {expected_adam}, not the card's {ADAM_DEFAULTS}")
    rec["recipe"], rec["optimizer"] = dataclasses.asdict(recipe), expected_adam
    rec["device"], rec["mlx"] = str(mx.default_device()), getattr(mx, "__version__", None)

    training, held_out, rec["corpus"] = read_corpus(spec, seams.read_corpus_json())
    log("corpus_verified", training_records=len(training), held_out_records=len(held_out))

    sha1 = seams.checkpoint_sha1()
    if sha1 != M.CHECKPOINT_SHA1:
        raise Void("checkpoint", f"SHA-1 {sha1} is not calib020_0001's")
    rec["checkpoint"] = {"path": M.CHECKPOINT, "sha1": sha1}
    teacher, student = seams.load_network(), seams.load_network()
    teacher.eval()
    student.eval()
    if student is teacher:
        raise Void("networks", "the student and the teacher are one object")
    rec["modes_at_load"] = {"teacher": M._require_eval(teacher, "the teacher after load"),
                            "student": M._require_eval(student, "the student after load")}
    teacher_digest = D.state_digest(teacher)
    if teacher_digest != spec["teacher_digest"] or D.state_digest(student) != teacher_digest:
        raise Void("networks", f"the loaded state {teacher_digest} is not the pinned teacher, or the copies differ")
    rec["teacher_digest_before"] = teacher_digest
    rec["modules"] = seams.loaded_modules()
    log("loaded")

    # 1. BASELINE -- the first of the two reads of the held-out records.
    _, held_values = D.forward(teacher, held_out)
    _, train_values = D.forward(teacher, training)
    rec["baseline_held_out"] = D.evaluate(teacher, held_out, held_values)
    rec["baseline_training"] = D.evaluate(teacher, training, train_values)          # diagnostic only
    log("baseline", policy_ce=rec["baseline_held_out"]["policy_ce"])

    # 2. TRAINING -- the training records only.
    def on_step(entry):
        if entry["step"] % STEP_LOG_EVERY == 0 or entry["step"] == recipe.steps - 1:
            log("step", **entry)

    failure, out = None, None
    t0 = time.monotonic()
    try:
        out = D.train_student(student, teacher, training, recipe, teacher_values=train_values, on_step=on_step)
    except D.DistillError as e:
        failure = str(e)
    rec["training_seconds"] = round(time.monotonic() - t0, 3)
    rec["training_failure"] = failure
    if out is not None:
        if out["optimizer"] != expected_adam:
            raise Void("optimizer", f"training resolved Adam to {out['optimizer']}, not {expected_adam}")
        rec["training"] = {k: out[k] for k in ("checks", "teacher_digest", "student_digest")}
        rec["training_log"] = out["steps"]
    log("trained", failure=failure)

    # 3. SAVE (always) and RELOAD.
    saved = D.save_student(student, str(candidate))
    rec["candidate"] = {**saved, "failed_training": failure is not None}
    log("saved", sha256=saved["sha256"])
    reloaded = D.reload(saved["path"], lambda: seams.make_network_for(saved["path"]))
    rec["candidate"]["reloaded_state_digest"] = D.state_digest(reloaded)
    rec["candidate"]["reload_modes"] = M._require_eval(reloaded, "the reloaded candidate")
    if rec["candidate"]["reloaded_state_digest"] != saved["state_digest"] or \
            (out is not None and saved["state_digest"] != out["student_digest"]):
        raise Void("reload", "the reloaded candidate's state is not the trained student's")
    rec["teacher_digest_after"] = D.state_digest(teacher)
    rec["modes_at_end"] = {"teacher": M._require_eval(teacher, "the teacher at the end")}
    if rec["teacher_digest_after"] != teacher_digest:
        raise Void("teacher", "the teacher changed")

    # 4. FINAL CHECK -- the second and last read of the held-out records, with the RELOADED candidate.
    final = None
    try:
        final = D.evaluate(reloaded, held_out, held_values)
        rec["final_training"] = D.evaluate(reloaded, training, train_values)       # diagnostic only
    except Exception as e:                       # e.g. a failed candidate's non-finite weights; recorded, judged STOP
        rec["final_error"] = f"{type(e).__name__}: {e}"
    rec["final_held_out"] = final
    rec["criteria"] = judge(spec, rec["baseline_held_out"], final, failure)
    rec["verdict"] = "PASS" if all(rec["criteria"].values()) else "STOP"
    rec["stop_reasons"] = [k for k, ok in rec["criteria"].items() if not ok]
    log("judged", verdict=rec["verdict"])
    return rec["verdict"] == "PASS"


def conclude(spec: Mapping[str, Any], seams: Seams, out_dir: pathlib.Path, candidate: pathlib.Path,
             head: str | None = None) -> Tuple[int, Dict[str, Any]]:
    """THE RECORD BOUNDARY (as Stages 1-2's): record.json on every exit; SIGTERM -> VOID at step "terminated"."""
    out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "card": CARD, "card_sha256": CARD_SHA256, "spec": dict(spec),
                           "head": head, "candidate_path": str(candidate), "status": None, "verdict": None,
                           "written_by": "script", "strength_verdict": None,
                           "environment": {"python": sys.version, "platform": platform.platform(),
                                           "executable": sys.executable},
                           "loadavg_before": os.getloadavg()}
    log_fh = open(out_dir / "progress.jsonl", "x")

    def log(event, **kw):
        log_fh.write(json.dumps({"event": event, "t": round(time.monotonic() - t0, 3), **kw}) + "\n")
        log_fh.flush()
        os.fsync(log_fh.fileno())

    def on_term(signum, _frame):
        raise Terminated(signal.Signals(signum).name)

    code = EXIT_VOID
    main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, on_term) if main_thread else None
    try:
        log("start")
        passed = _execute(spec, seams, candidate, rec, log)
        rec["status"] = "COMPLETED"
        code = EXIT_PASS if passed else EXIT_STOP
    except Terminated as t:
        rec.update(status="VOID", void={"step": "terminated", "detail": f"{t} after {time.monotonic() - t0:.1f} s"})
    except Void as v:
        rec.update(status="VOID", void={"step": v.step, "detail": v.detail})
    except Exception as e:                       # any exception is a VOID, and it is recorded
        rec.update(status="VOID", void={"step": "exception", "detail": f"{type(e).__name__}: {e}"})
    except BaseException as e:
        rec.update(status="INTERRUPTED", void={"step": "interrupted", "detail": type(e).__name__})
        raise
    finally:
        if main_thread:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if rec["status"] != "COMPLETED":
                rec["verdict"] = None
            rec["total_seconds"] = round(time.monotonic() - t0, 3)
            rec["loadavg_after"] = os.getloadavg()
            log("end", status=rec["status"], verdict=rec["verdict"])
            log_fh.close()
            M._write_json_new(out_dir / "record.json", rec)
        finally:
            if main_thread:
                signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)
    return code, rec


def _real_seams(checkpoint: pathlib.Path) -> Seams:
    base = M._real_seams(checkpoint)

    def make_network_for(path):
        from .probe_eval import load_network_for_scoring
        return load_network_for_scoring(str(path), verbose=False)[0]

    return Seams(read_corpus_json=lambda: (ROOT / CORPUS_JSON).read_bytes(), checkpoint_sha1=base.checkpoint_sha1,
                 load_network=base.load_network, make_network_for=make_network_for,
                 loaded_modules=base.loaded_modules)


def run(checkpoint: str) -> int:
    """THE PUBLIC ENTRY. No seam: the frozen spec, the real sources, the pinned card."""
    _require_gate()
    if M._sha((ROOT / CARD).read_bytes()) != CARD_SHA256:
        raise Refused(f"{CARD} is not the pinned card")
    ckpt = pathlib.Path(checkpoint)
    if not (ckpt.is_absolute() and ckpt.is_file()):
        raise Refused("--checkpoint must name an existing file by absolute path")
    out = ROOT / EVIDENCE_DIR
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the run is one-shot")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = CANDIDATE_PARENT / f"student-stage3-{stamp}.safetensors"
    if not CANDIDATE_PARENT.is_dir() or os.path.lexists(candidate):
        raise Refused(f"{CANDIDATE_PARENT} is not a directory, or {candidate} exists")
    code, rec = conclude(FROZEN, _real_seams(ckpt), out, candidate, head=M._git("rev-parse", "HEAD"))
    print(json.dumps({"status": rec["status"], "verdict": rec.get("verdict"), "void": rec.get("void"),
                      "stop_reasons": rec.get("stop_reasons"), "head": rec["head"],
                      "candidate": rec.get("candidate", {}).get("path")}))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3 or argv[0] != "run" or argv[1] != "--checkpoint":
        print("usage: python -m scripts.GPU.alphazero.search_distillation_d2_train run --checkpoint <abs path>",
              file=sys.stderr)
        return EXIT_USAGE
    try:
        return run(argv[2])
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
