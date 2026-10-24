"""SEARCH DISTILLATION D2 -- CORRECTED-CONFIGURATION TRIAL, STEP 2: ONE training run and its held-out check, in ONE
fresh process launched with MLX_ENABLE_TF32=0. BUILT; run only through its wrapper.

Card: `docs/superpowers/2026-10-23-search-distillation-d2-corrected-trial-card.md` (§2, §3, §5), pinned below by
sha256, with the reused modules pinned by hash. Nothing runs on import, and importing this module loads no MLX. `run`
refuses unless MLX_ENABLE_TF32 is exactly "0" and no `mlx` module is loaded when it starts, then unless
`search_distillation_d2_trial_authorization.SEARCH_DISTILLATION_D2_TRIAL_AUTHORIZED` is open (read at call time), and
until the new held-out set's record and manifest are committed unchanged.

WHAT THE RUN DOES: a COPY of Stage 3's run (search_distillation_d2_train._execute); the diff: the readers, the setting
and version checks, the scorings ledger and the spike count. Stage 3's recipe, criteria, save and reload are reused
unchanged (train_student, recipe_for, judge, ADAM_DEFAULTS).
  0. READS the 108 original TRAINING games through gradtrace's training-only reader (the pinned corpus.json; each
     phase's context, receipts and completion marker checked; only training games' files opened, each by its receipt's
     hash; the row identities pinned). No original held-out game file is opened, and none of its records is read: the
     record says so, apart from any access to the new set. ADMITS the new held-out set only if its generation record
     is COMPLETED under this card, with the setting "0" at start and end and the teacher's digest before and after,
     and heldout.json is the bytes that record names; its phase is reopened through the strict capture reader.
  1. BASELINE: the teacher on the new held-out records (policy cross-entropy; the teacher values for the drift check).
  2. TRAINS the student once with search_distillation_d2.train_student at Stage 3's frozen recipe (401 updates, batch
     64, Adam 1e-4 with MLX's resolved defaults read back and checked, clip 1.0, anchor 1.0, L2 1e-4, draw-order seed 0)
     on the encoder and policy head only; the teacher, the value head and every BatchNorm statistic unchanged.
  3. SAVES the one candidate create-only (also when training stopped), reloads it strictly into a fresh network and
     requires its state digest to be the saved one.
  4. FINAL CHECK on the new held-out records with the RELOADED candidate. The new set is scored only in 1 and 4; the
     record lists the scorings, which must be exactly baseline, final.

VERDICT (Stage 3's, unchanged): PASS only if training finished with every check, every metric is finite, held-out
policy cross-entropy is STRICTLY below the teacher's, and absolute value drift from the teacher on the held-out records
has mean <= 0.05 and 95th percentile <= 0.15. Otherwise STOP. A PASS establishes fit and bounded value drift, not
stronger play. A broken input, identity, setting, version or reload, an exception or SIGTERM (the wrapper's 30-minute
ceiling) is a VOID. No retry: a STOP or VOID closes the trial.

    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.search_distillation_d2_trial run --checkpoint <calib020_0001>
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import os
import pathlib
import platform
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from . import search_distillation_d2_gradtrace as R
from . import search_distillation_d2_heldout as H
from . import search_distillation_d2_measure as M
from . import search_distillation_d2_train as T
from . import search_distillation_d2_trial_authorization as GATE

ROOT = M.ROOT
RECORD_KIND = "search-distillation-d2-trial-record/1"
CARD = H.CARD
CARD_SHA256 = H.CARD_SHA256
EVIDENCE_DIR = "docs/superpowers/evidence/search-distillation-d2-trial"
CORPUS_JSON = T.CORPUS_JSON
HELDOUT_RECORD = f"{H.EVIDENCE_DIR}/record.json"
HELDOUT_JSON = f"{H.EVIDENCE_DIR}/heldout.json"
CANDIDATE_PARENT = pathlib.Path.home() / "twixt-cache" / "search-distillation-d2"     # outside the repository
SETTING, SETTING_VALUE = H.SETTING, H.SETTING_VALUE
VERSIONS = H.VERSIONS
#: the reused modules, pinned (card §8)
CODE = {"heldout": "scripts/GPU/alphazero/search_distillation_d2_heldout.py",
        "train": "scripts/GPU/alphazero/search_distillation_d2_train.py",
        "gradtrace": "scripts/GPU/alphazero/search_distillation_d2_gradtrace.py",
        "distill": "scripts/GPU/alphazero/search_distillation_d2.py",
        "measure": "scripts/GPU/alphazero/search_distillation_d2_measure.py",
        "capture": "scripts/GPU/alphazero/training_replay_capture.py"}
PINNED = {CODE["heldout"]: "f31579d0f8dd7c103795a19808020e0b450484a67aed94b46f5483bc8375f66d",
          CODE["train"]: "1988e08871a2bdedc4128ca9eef2353aa32f395ba6fadda3558c4157473fcb3f",
          CODE["gradtrace"]: "85b84b074b3359be938fe3e40cc2849ce2b1f7e6db88b909a784b625e3e35c97",
          CODE["distill"]: "bc8fd0c80fc533f50273ebda60c81375c5131d068a70344e07afef9a73617be9",
          CODE["measure"]: "c2ecd81d7939a3879895b8bdfb36221c5cedecc616c345a37d4ef26f03b30669",
          CODE["capture"]: "b58765b13f798133fb685ae4599f4b32d9f262d7659a402f44c10600f8974f32"}
FROZEN = {**{k: T.FROZEN[k] for k in ("corpus_sha256", "teacher_digest", "training_games", "held_out_games",
                                      "training_records", "passes", "steps", "batch_size", "learning_rate",
                                      "anchor_weight", "l2_weight", "clip_norm", "seed", "value_abs_mean_max",
                                      "value_abs_p95_max")},
          "training_ids_sha256": R.FROZEN["training_ids_sha256"], "new_held_out_games": H.FROZEN["games"],
          "spike_norm": 1e4}
SCORINGS = ["baseline", "final"]
STEP_LOG_EVERY = T.STEP_LOG_EVERY
EXIT_PASS, EXIT_STOP, EXIT_VOID, EXIT_REFUSED, EXIT_USAGE = 0, 6, 3, 4, 2
Void, Terminated = M.Void, M.Terminated


class Refused(Exception):
    """The run never started: the setting, MLX already loaded, gate closed, a pin, the held-out set not generated and
    committed, an output exists, or a bad argument."""


@dataclasses.dataclass
class Seams:
    """What the run reads from outside this module. TEST seams; `run` passes `_real_seams` only."""
    read_corpus_json: Callable[[], bytes]
    read_heldout_record: Callable[[], bytes]
    read_heldout_json: Callable[[], bytes]
    checkpoint_sha1: Callable[[], str]
    load_network: Callable[[], Any]                    # a FRESH load of the incumbent on every call
    make_network_for: Callable[[str], Any]             # a fresh network of the architecture saved at a path
    loaded_modules: Callable[[], Dict[str, str]]
    versions: Callable[[], Dict[str, str]]


def read_training(spec: Mapping[str, Any], raw: bytes) -> Tuple[list, Dict[str, Any]]:
    """The 108 original training games through gradtrace's training-only reader; the original held-out games'
    files are never opened, and none of their records is read."""
    from . import training_replay_capture as C

    records, _identities, summary = R.read_training(spec, raw)
    corpus = json.loads(raw)
    held = set(corpus["held_out_games"])
    held_paths = {f"{ph['dir']}/{C._game_path(g)}" for ph in corpus["phases"] for g in ph["games"] if g in held}
    opened = [p for p in summary["files_opened"] if p in held_paths]
    if opened:                                                   # by construction; checked, not assumed
        raise Void("held_out", f"an original held-out game file was opened: {opened[0]}")
    return records, {**summary, "original_held_out": {"games": sorted(held), "files_opened": len(opened),
                                                      "records_read": 0}}


def read_heldout(spec: Mapping[str, Any], record_raw: bytes, manifest_raw: bytes) -> Tuple[list, Dict[str, Any]]:
    """The new held-out records, admitted only from a COMPLETED generation record under this card; any disagreement is
    a VOID. Reading is not scoring: the set is scored only for the baseline and the final check."""
    from . import search_distillation_d2 as D
    from . import training_replay_capture as C

    rec = json.loads(record_raw)
    manifest_sha = M._sha(manifest_raw)
    want = {"kind": H.RECORD_KIND, "status": "COMPLETED", "eligible": True, "card_sha256": CARD_SHA256,
            "setting": {SETTING: SETTING_VALUE}, "setting_at_end": {SETTING: SETTING_VALUE},
            "model_digest_before": spec["teacher_digest"], "model_digest_after": spec["teacher_digest"],
            "heldout_sha256": manifest_sha}
    wrong = sorted(k for k, v in want.items() if rec.get(k) != v)
    if wrong:
        raise Void("held_out_set", f"the generation record does not admit the set: {wrong}")
    manifest = json.loads(manifest_raw)
    games = list(range(spec["new_held_out_games"]))
    if (manifest.get("kind") != H.MANIFEST_KIND or manifest.get("eligible") is not True
            or manifest.get("model_digest") != spec["teacher_digest"] or manifest.get("held_out_games") != games
            or manifest.get("setting") != {SETTING: SETTING_VALUE} or manifest["phase"].get("games") != games):
        raise Void("held_out_set", "the manifest is not an eligible 12-game set of the pinned teacher under the setting")
    ph = manifest["phase"]
    d = pathlib.Path(manifest["heldout_dir"]) / ph["dir"]
    for f, key in (("context.json", "context_sha256"), ("finalized.json", "finalized_sha256")):
        if M._sha((d / f).read_bytes()) != ph[key]:
            raise Void("held_out_set", f"{ph['dir']}/{f} differs from the manifest")
    ctx, envelopes = C.load_phase(str(d))                           # strict: receipts and every game file by hash
    if ctx["models"][M.MODEL_ID]["state_digest"] != spec["teacher_digest"]:
        raise Void("held_out_set", "the capture context names another model state")
    if [e["game_id"] for e in envelopes] != games:
        raise Void("held_out_set", "the games are not the manifest's")
    records, _ids = D.records_from_envelopes(envelopes)
    if len(records) != ph["records"]:
        raise Void("held_out_set", f"{len(records)} records, the manifest says {ph['records']}")
    return records, {"record_sha256": M._sha(record_raw), "manifest_sha256": manifest_sha,
                     "heldout_dir": manifest["heldout_dir"], "games": games, "records": len(records),
                     "seeds": manifest["seeds"]}


def _execute(spec: Mapping[str, Any], seams: Seams, candidate: pathlib.Path, rec: Dict[str, Any], log) -> bool:
    import mlx.core as mx
    import mlx.optimizers as optim

    from . import search_distillation_d2 as D

    rec["versions"] = seams.versions()
    if rec["versions"] != VERSIONS:
        raise Void("versions", f"{rec['versions']} are not the card's {VERSIONS}")
    recipe = T.recipe_for(spec)
    expected_adam = D.adam_settings(optim.Adam(learning_rate=recipe.learning_rate))
    if {k: expected_adam[k] for k in T.ADAM_DEFAULTS} != T.ADAM_DEFAULTS:
        raise Void("optimizer", f"MLX's Adam resolves to {expected_adam}, not the card's {T.ADAM_DEFAULTS}")
    rec["recipe"], rec["optimizer"] = dataclasses.asdict(recipe), expected_adam
    rec["device"], rec["mlx"] = str(mx.default_device()), getattr(mx, "__version__", None)

    training, rec["corpus"] = read_training(spec, seams.read_corpus_json())
    held_out, rec["new_held_out"] = read_heldout(spec, seams.read_heldout_record(), seams.read_heldout_json())
    held = {id(r) for r in held_out}
    scorings: List[str] = []
    rec["new_held_out"]["scorings"] = scorings
    log("inputs_verified", training_records=len(training), new_held_out_records=len(held_out))

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

    # 1. BASELINE -- the first of the two scorings of the new held-out set.
    scorings.append("baseline")
    _, held_values = D.forward(teacher, held_out)
    rec["baseline_held_out"] = D.evaluate(teacher, held_out, held_values)
    _, train_values = D.forward(teacher, training)
    rec["baseline_training"] = D.evaluate(teacher, training, train_values)          # diagnostic only
    log("baseline", policy_ce=rec["baseline_held_out"]["policy_ce"])

    # 2. TRAINING -- the training records only.
    if any(id(r) in held for r in training):
        raise Void("split", "a new held-out record is among the training records")

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
        rec["grad_norm_above_spike_norm"] = sum(s["grad_norm"] > spec["spike_norm"] for s in out["steps"])  # recorded
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

    # 4. FINAL CHECK -- the second and last scoring of the new held-out set, with the RELOADED candidate.
    final = None
    scorings.append("final")
    try:
        final = D.evaluate(reloaded, held_out, held_values)
        rec["final_training"] = D.evaluate(reloaded, training, train_values)       # diagnostic only
    except Exception as e:                       # e.g. a failed candidate's non-finite weights; recorded, judged STOP
        rec["final_error"] = f"{type(e).__name__}: {e}"
    if scorings != SCORINGS:
        raise Void("held_out_set", f"the new set was scored {scorings}, not exactly {SCORINGS}")
    rec["final_held_out"] = final
    rec["criteria"] = T.judge(spec, rec["baseline_held_out"], final, failure)
    rec["verdict"] = "PASS" if all(rec["criteria"].values()) else "STOP"
    rec["stop_reasons"] = [k for k, ok in rec["criteria"].items() if not ok]
    log("judged", verdict=rec["verdict"])
    return rec["verdict"] == "PASS"


def conclude(spec: Mapping[str, Any], seams: Seams, out_dir: pathlib.Path, candidate: pathlib.Path,
             head: str | None = None) -> Tuple[int, Dict[str, Any]]:
    """THE RECORD BOUNDARY (as Stage 3's): record.json on every exit; SIGTERM -> VOID at step "terminated"."""
    out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "card": CARD, "card_sha256": CARD_SHA256, "spec": dict(spec),
                           "head": head, "candidate_path": str(candidate), "status": None, "verdict": None,
                           "written_by": "script", "strength_verdict": None,
                           "setting": {SETTING: os.environ.get(SETTING)},
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
        log("start", setting=rec["setting"])
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
            rec["setting_at_end"] = {SETTING: os.environ.get(SETTING)}
            rec["total_seconds"] = round(time.monotonic() - t0, 3)
            rec["loadavg_after"] = os.getloadavg()
            log("end", status=rec["status"], verdict=rec["verdict"])
            log_fh.close()
            M._write_json_new(out_dir / "record.json", rec)
        finally:
            if main_thread:
                signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)
    return code, rec


def _committed(rel: str) -> bool:
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", rel],
                             capture_output=True).returncode == 0
    return tracked and subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel]).returncode == 0


def _real_seams(checkpoint: pathlib.Path) -> Seams:
    base = T._real_seams(checkpoint)
    return Seams(read_corpus_json=base.read_corpus_json,
                 read_heldout_record=lambda: (ROOT / HELDOUT_RECORD).read_bytes(),
                 read_heldout_json=lambda: (ROOT / HELDOUT_JSON).read_bytes(),
                 checkpoint_sha1=base.checkpoint_sha1, load_network=base.load_network,
                 make_network_for=base.make_network_for, loaded_modules=base.loaded_modules, versions=H.versions)


def run(checkpoint: str) -> int:
    """THE PUBLIC ENTRY. No seam: the setting, the frozen spec, the pinned card and code, the committed held-out set,
    the real sources."""
    if os.environ.get(SETTING) != SETTING_VALUE:
        raise Refused(f"{SETTING} is {os.environ.get(SETTING)!r}, not {SETTING_VALUE!r}: launch a fresh process with it")
    if H._mlx_loaded():
        raise Refused("MLX was imported before the setting was checked: launch a fresh process")
    if GATE.SEARCH_DISTILLATION_D2_TRIAL_AUTHORIZED is not True:
        raise Refused("SEARCH_DISTILLATION_D2_TRIAL_AUTHORIZED is closed: training is not authorized")
    if M._sha((ROOT / CARD).read_bytes()) != CARD_SHA256:
        raise Refused(f"{CARD} is not the pinned card")
    for rel, sha in PINNED.items():
        if M._sha((ROOT / rel).read_bytes()) != sha:
            raise Refused(f"{rel} is not the pinned bytes")
    for rel in (HELDOUT_RECORD, HELDOUT_JSON):
        if not (ROOT / rel).is_file() or not _committed(rel):
            raise Refused(f"{rel} is absent, uncommitted or modified: the held-out set must be generated and committed")
    ckpt = pathlib.Path(checkpoint)
    if not (ckpt.is_absolute() and ckpt.is_file()):
        raise Refused("--checkpoint must name an existing file by absolute path")
    out = ROOT / EVIDENCE_DIR
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the run is one-shot")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = CANDIDATE_PARENT / f"student-tf32off-{stamp}.safetensors"
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
        print("usage: MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.search_distillation_d2_trial run "
              "--checkpoint <abs path>", file=sys.stderr)
        return EXIT_USAGE
    try:
        return run(argv[2])
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
