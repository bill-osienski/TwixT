"""EXTERNAL-ENGINE LEARNING, STAGE 2 -- the training smoke. PROTOTYPE; PREPARATION ONLY.

Card: `docs/superpowers/2026-10-08-external-engine-learning-viability.md`, design baseline ff1422fb, stage 2.
Nothing here runs on import. The run entry refuses unless the 29th gate,
`external_learning_stage2_authorization.EXTERNAL_LEARNING_STAGE2_AUTHORIZED`, is open -- read at call time,
in the parent AND in the child.

WHAT STAGE 2 DOES, in the card's order (each failure is a STOP; nothing is retried, selected or extended):
  1. the committed seed plan (ONE training seed, ACCOUNTED and unspent) and the frozen manifest, pinned by
     sha256 and equal to the one this module derives;
  2. stage 1's committed training rows (each rebuilt in OUR engine) and its logit artifact, through
     `external_learning.load_logit_artifact` -- bound to those rows and the incumbent, never recomputed;
     training rows ONLY: no held-out row is read past its split field;
  3. the scalar-temperature control fitted on the cached training logits (`external_learning.fit_beta`);
     a failed or nonfinite fit stops BEFORE any optimization;
  4. a verified copy of the incumbent (SHA-1 pinned), eval mode, every parameter frozen except the 519 of
     `policy_head.{conv,bn,fc,out}.{weight,bias}` -- BN running statistics stay frozen;
  5. 128 steps, batch 32, uniform with replacement from the training split (one fresh seed), Adam lr 0.001,
     global-norm clipping 1.0, no weight decay; loss = mean -sum(one-hot target * masked log-probs);
  6. the final candidate saved ONCE (create-only) and reloaded: only trainable tensors may differ from the
     incumbent file, every other tensor byte-identical; raw values unchanged; the policy measurably changed.
Held-out rows, stage-3 evaluation and every game stay outside this module.

    python -m scripts.GPU.alphazero.external_learning_stage2 run
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import shutil
import signal
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import external_learning as EL
from . import external_learning_stage2_authorization as GATE

ROOT = EL.ROOT
MODULE = "scripts.GPU.alphazero.external_learning_stage2"
MANIFEST_KIND = "external-learning-stage2-manifest/1"
PLAN_KIND = "external-learning-stage2-plan/1"
RECORD_KIND = "external-learning-stage2-record/1"
MANIFEST = "docs/superpowers/evidence/external-engine-learning-plans/stage2-manifest.json"
MANIFEST_SHA256 = "916224900d6f674d6d93ccc08fc00499ae2bc71f422f57a7a6a5ae17961d831d"
PLAN = "docs/superpowers/evidence/external-engine-learning-plans/stage2.json"
RUN_DIR = "runs/external-engine-learning/stage2"
EVIDENCE_DIR = "docs/superpowers/evidence/external-engine-learning-stage2"
CLEANUP_FAILED = RUN_DIR + ".cleanup-failed.json"
CANDIDATE = "candidate.safetensors"
STAGE1_ROWS, STAGE1_LOGITS = "teacher_rows.jsonl", "training_logits.json"

STEPS, BATCH = 128, 32
LEARNING_RATE, CLIP_NORM, WEIGHT_DECAY = 1e-3, 1.0, 0.0
ADAM = {"betas": [0.9, 0.999], "eps": 1e-8, "bias_correction": False}   # MLX defaults, as the trainer uses
EXECUTION_S = 1800                                           # the card's 30 minutes, control and save included
N_SLOTS = 1
TRAINABLE = tuple(f"policy_head.{m}.{p}" for m in ("conv", "bn", "fc", "out") for p in ("weight", "bias"))
N_TRAINABLE = 519
POLICY_TOL = 1e-6                                            # max |change in a legal probability| must exceed it
SAMPLING = "numpy.random.Generator(PCG64(seed)).integers(0, n_train, size=(steps, batch)): uniform, with replacement"

Stop, Refused = EL.Stage1Stop, EL.Refused
EXIT_PASS, EXIT_STOP, EXIT_REFUSED, EXIT_TIMEOUT = EL.EXIT_PASS, EL.EXIT_STOP, EL.EXIT_REFUSED, EL.EXIT_TIMEOUT


def _require_gate() -> None:
    if GATE.EXTERNAL_LEARNING_STAGE2_AUTHORIZED is not True:
        raise Refused("EXTERNAL_LEARNING_STAGE2_AUTHORIZED is closed: stage 2 is not authorized")


# ─────────────────────────────── the manifest and the seed plan ───────────────────────────────

def build_manifest(root: pathlib.Path = ROOT) -> Dict[str, Any]:
    """Everything stage 2 is bound to, derived from the committed stage-1 evidence and this module."""
    from .twixtbot_g3_schedule import REFERENCE_CHECKPOINTS

    ev = root / EL.EVIDENCE_DIR
    inc = REFERENCE_CHECKPOINTS[EL.REFERENCE]
    return {"kind": MANIFEST_KIND, "card": EL.CARD, "baseline": EL.BASELINE,
            "stage1": {"evidence_dir": EL.EVIDENCE_DIR,
                       "manifest_sha256": EL._sha((ev / "MANIFEST.sha256").read_bytes()),
                       "record_sha256": EL._sha((ev / "record.json").read_bytes()),
                       "rows": {"file": STAGE1_ROWS, "sha256": EL._sha((ev / STAGE1_ROWS).read_bytes())},
                       "logits": {"file": STAGE1_LOGITS, "sha256": EL._sha((ev / STAGE1_LOGITS).read_bytes())}},
            "incumbent": {"reference": EL.REFERENCE, "path": inc["path"], "sha1": EL.REFERENCE_SHA1},
            "candidate": {"run_path": f"{RUN_DIR}/{CANDIDATE}", "evidence_path": f"{EVIDENCE_DIR}/{CANDIDATE}"},
            "budget": {"steps": STEPS, "batch": BATCH, "deadline_s": EXECUTION_S},
            "optimizer": {"name": "adam", "learning_rate": LEARNING_RATE, **ADAM,
                          "grad_clip_global_norm": CLIP_NORM, "weight_decay": WEIGHT_DECAY},
            "loss": "mean over the batch of -sum(one-hot chosen target * compute_masked_log_probs(logits, mask)); "
                    "value output discarded; eval mode (incumbent BN statistics)",
            "sampling": SAMPLING,
            "trainable": {"names": list(TRAINABLE), "parameters": N_TRAINABLE},
            "control": {"beta_init": 1.0, "beta_bounds": list(EL.BETA_BOUNDS),
                        "iterations": EL.BETA_ITERATIONS, "tolerance": EL.BETA_TOLERANCE},
            "training_seed": {"slots": N_SLOTS, "plan": PLAN}}


def check_manifest(root: pathlib.Path = ROOT) -> Dict[str, Any]:
    raw = (root / MANIFEST).read_bytes()
    if EL._sha(raw) != MANIFEST_SHA256:
        raise Stop("manifest", "the manifest is not the pinned bytes")
    manifest = json.loads(raw)
    if manifest != build_manifest(root):
        raise Stop("manifest", "the manifest is not the one this module derives from the stage-1 evidence")
    return manifest


def build_plan(seed: int, survey_record_sha256: str) -> Dict[str, Any]:
    return {"kind": PLAN_KIND, "card": EL.CARD, "baseline": EL.BASELINE, "manifest_sha256": MANIFEST_SHA256,
            "seed": int(seed), "n_slots": N_SLOTS, "survey_record_sha256": survey_record_sha256}


def check_plan(plan: Any) -> Dict[str, Any]:
    """ONE training seed: an int, the plan exactly its frozen form, ACCOUNTED, neither exposed nor retired,
    and not one G3 consumed."""
    from . import e4_screen_reference as REG
    from .twixtbot_g3_schedule import CONSUMED_SEEDS

    if not isinstance(plan, dict) or plan.get("kind") != PLAN_KIND or type(plan.get("seed")) is not int:
        raise Stop("plan", "not a stage-2 plan with an int seed")
    sha = plan.get("survey_record_sha256")
    if not (isinstance(sha, str) and len(sha) == 64) or plan != build_plan(plan["seed"], sha):
        raise Stop("plan", "the plan is not the frozen form of its seed, survey record and manifest")
    s = plan["seed"]
    if not REG.seed_is_accounted(s):
        raise Stop("plan", f"seed {s} is not ACCOUNTED in the shared registry")
    if REG.seed_is_unavailable(s) or s in CONSUMED_SEEDS:
        raise Stop("plan", f"seed {s} is spent (exposed, retired or consumed)")
    return plan


def load_plan() -> Tuple[Dict[str, Any], str]:
    """The committed seed plan (written at registration), checked; and its sha256."""
    if not EL._committed(PLAN):
        raise Stop("plan", f"{PLAN} is absent or not committed")
    raw = (ROOT / PLAN).read_bytes()
    return check_plan(json.loads(raw)), EL._sha(raw)


def sampling_schedule(seed: int, n_train: int):
    """THE fixed budget, read at call time: STEPS draws of BATCH training indices, uniform with replacement."""
    import numpy as np

    return np.random.Generator(np.random.PCG64(seed)).integers(0, n_train, size=(STEPS, BATCH))


# ─────────────────────────────── the inputs: training rows and their cached logits ───────────────────────────────

def load_inputs(manifest: Mapping[str, Any], root: pathlib.Path = ROOT):
    """Stage 1's committed rows (training split only, each rebuilt) and the logit artifact bound to them."""
    ev = root / manifest["stage1"]["evidence_dir"]
    if EL._sha((ev / "MANIFEST.sha256").read_bytes()) != manifest["stage1"]["manifest_sha256"]:
        raise Stop("inputs", "the stage-1 evidence manifest is not the bound one")
    listed = {name: sha for sha, name in (line.split("  ", 1) for line in (ev / "MANIFEST.sha256").read_text().splitlines())}
    for key in ("rows", "logits"):
        f = manifest["stage1"][key]
        if listed.get(f["file"]) != f["sha256"] or EL._sha((ev / f["file"]).read_bytes()) != f["sha256"]:
            raise Stop("inputs", f"{f['file']} is not the file stage 1's manifest hashed")
    train = [r for r in EL.load_rows(ev / STAGE1_ROWS) if r["split"] == "train"]
    for r in train:
        EL.verify_row(r)
    logits = EL.load_logit_artifact(ev / STAGE1_LOGITS, sha256=manifest["stage1"]["logits"]["sha256"], rows=train,
                                    incumbent_sha1=manifest["incumbent"]["sha1"])
    return train, logits


def fit_control(train: Sequence[Mapping[str, Any]], logits: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """The scalar-temperature control, on the cached TRAINING logits only; frozen before stage 3."""
    out = EL.fit_beta(logits, [r["chosen_index"] for r in train])
    if not all(math.isfinite(out[k]) for k in ("beta", "ce", "derivative")):
        raise Stop("beta", "a nonfinite control fit")
    return out


# ─────────────────────────────── the network: incumbent copy, frozen surface, the update ───────────────────────────────

def load_incumbent(manifest: Mapping[str, Any], root: pathlib.Path = ROOT):
    """A verified copy of the incumbent: the file's SHA-1 checked, loaded fresh, eval mode."""
    import hashlib

    from .probe_eval import load_network_for_scoring

    path = root / manifest["incumbent"]["path"]
    if hashlib.sha1(path.read_bytes()).hexdigest() != manifest["incumbent"]["sha1"]:
        raise Stop("incumbent", "the incumbent file is not the pinned checkpoint")
    network, *_ = load_network_for_scoring(str(path), verbose=False)
    network.eval()
    return network, path


def head_only(network) -> List[str]:
    """Freeze everything but the policy head's eight weight/bias arrays. `unfreeze()` would also expose
    the head's BN running statistics, so they are frozen again; the trainable set must be exactly 519."""
    from mlx.utils import tree_flatten

    network.eval()
    network.freeze()
    network.policy_head.unfreeze()
    network.policy_head.bn.freeze(keys=["running_mean", "running_var"])
    flat = tree_flatten(network.trainable_parameters())
    names = sorted(k for k, _ in flat)
    if names != sorted(TRAINABLE) or sum(v.size for _, v in flat) != N_TRAINABLE:
        raise Stop("surface", f"the trainable surface is not the policy head's {N_TRAINABLE}: {names}")
    return names


def records_of(rows: Sequence[Mapping[str, Any]]):
    return [EL.position_record(r) for r in rows]


def batch_of(records: Sequence[Any]):
    """`make_padded_batch(max_moves_cap=576)`; every row's target must keep its whole unit of mass."""
    import numpy as np

    from .trainer import make_padded_batch

    batch = make_padded_batch(list(records), max_moves_cap=EL.MAX_MOVES_CAP)
    if not np.allclose(np.asarray(batch[4]).sum(axis=1), 1.0):
        raise Stop("adapter", "a teacher target lost its mass in batching")
    return batch


def chosen_ce(network, boards, mr, mc, mask, target):
    """Mean chosen-move CE over the batch, policy logits only; the value output is discarded."""
    import mlx.core as mx

    from .trainer import compute_masked_log_probs

    logits, _value, _ = network.forward_padded(boards, mr, mc, mask, active_size=EL.BOARD)
    return -mx.mean(mx.sum(target * compute_masked_log_probs(logits, mask), axis=1))


def train_head(network, records: Sequence[Any], schedule, log: Callable[[Dict[str, Any]], None] = lambda e: None
               ) -> List[Dict[str, float]]:
    """The fixed number of policy-head-only steps, in eval mode. Every step must be finite; a gradient
    outside the policy head, or a training-mode network, stops."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from .trainer import clip_grad_norm

    head_only(network)
    opt = optim.Adam(learning_rate=LEARNING_RATE, betas=ADAM["betas"], eps=ADAM["eps"],
                     bias_correction=ADAM["bias_correction"])
    step_fn = nn.value_and_grad(network, chosen_ce)
    out = []
    for step, idx in enumerate(schedule):
        if network.training:
            raise Stop("training", "the network left eval mode")
        boards, mr, mc, mask, target, _outcome = batch_of([records[int(i)] for i in idx])
        loss, grads = step_fn(network, boards, mr, mc, mask, target)
        leaves = tree_flatten(grads)
        if sorted(k for k, _ in leaves) != sorted(TRAINABLE):
            raise Stop("training", "a gradient outside the policy head's trainable surface")
        head_grads, gnorm = clip_grad_norm(grads["policy_head"], max_norm=CLIP_NORM)
        mx.eval(loss, head_grads, gnorm)
        loss_f, gnorm_f = float(loss.item()), float(gnorm.item())
        if not (math.isfinite(loss_f) and math.isfinite(gnorm_f)):
            raise Stop("training", f"step {step}: nonfinite loss or gradient")
        opt.update(network.policy_head, head_grads)
        mx.eval(network.policy_head.parameters(), opt.state)
        rec = {"step": step, "loss": loss_f, "grad_norm": gnorm_f}
        out.append(rec)
        log({"event": "train_step", **rec})
    if not all(math.isfinite(float(v)) for _, a in tree_flatten(network.policy_head.parameters())
               for v in a.reshape(-1).tolist()):
        raise Stop("training", "a nonfinite policy-head parameter")
    return out


def forward_rows(network, records: Sequence[Any], chunk: int = 32) -> Tuple[List[List[float]], List[float]]:
    """Raw logits over each row's full legal order, and the raw value, through the canonicalized forward."""
    import mlx.core as mx

    logits, values = [], []
    for i in range(0, len(records), chunk):
        part = records[i:i + chunk]
        boards, mr, mc, mask, _t, _o = batch_of(part)
        z, v, _ = network.forward_padded(boards, mr, mc, mask, active_size=EL.BOARD)
        mx.eval(z, v)
        for r, row in zip(part, z.tolist()):
            logits.append([float(x) for x in row[:len(r.legal_moves)]])
        values.extend(float(x) for x in v.reshape(-1).tolist())
    return logits, values


def tensors_of(path: pathlib.Path) -> Dict[str, bytes]:
    import mlx.core as mx
    import numpy as np

    return {k: str(np.asarray(v).dtype).encode() + str(v.shape).encode() + np.asarray(v).tobytes()
            for k, v in mx.load(str(path)).items()}


def surface_check(incumbent_path: pathlib.Path, candidate_path: pathlib.Path) -> Dict[str, Any]:
    """Only trainable tensors may differ; every other tensor (encoder, value head/adapter, EVERY BN running
    statistic) byte-identical to the incumbent file; at least one trainable tensor changed."""
    inc, cand = tensors_of(incumbent_path), tensors_of(candidate_path)
    if set(inc) != set(cand):
        raise Stop("surface", "the candidate's tensors are not the incumbent's names")
    changed = sorted(k for k in inc if inc[k] != cand[k])
    if any(k not in TRAINABLE for k in changed):
        raise Stop("surface", f"a forbidden tensor changed: {[k for k in changed if k not in TRAINABLE]}")
    if not changed:
        raise Stop("change", "no policy-head tensor changed")
    return {"tensors": len(inc), "changed": changed, "identical": len(inc) - len(changed)}


def save_candidate(network, path: pathlib.Path) -> str:
    """Saved ONCE: refuses an existing file."""
    if os.path.lexists(path):
        raise Stop("save", f"{path.name} exists: the candidate is saved once")
    network.save_weights(str(path))
    return EL._sha(path.read_bytes())


def reload_check(network, path: pathlib.Path):
    """The saved candidate, loaded fresh, must hold exactly the trained parameters."""
    import numpy as np
    from mlx.utils import tree_flatten

    from .probe_eval import load_network_for_scoring

    reloaded, *_ = load_network_for_scoring(str(path), verbose=False)
    reloaded.eval()
    mine, theirs = dict(tree_flatten(network.parameters())), dict(tree_flatten(reloaded.parameters()))
    if set(mine) != set(theirs) or any(not np.array_equal(np.asarray(mine[k]), np.asarray(theirs[k])) for k in mine):
        raise Stop("reload", "the reloaded candidate is not the trained network")
    return reloaded


def legal_probs(z: Sequence[float]) -> List[float]:
    """softmax over a row's legal logits -- unchanged by adding any constant to every logit of the row."""
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [x / s for x in e]


def policy_change(before: Sequence[Sequence[float]], after: Sequence[Sequence[float]],
                  rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Training rows only -- a mechanics observation, never strength evidence. The policy changed only if
    some training row's legal-move PROBABILITIES moved by more than POLICY_TOL: a common shift of a row's
    logits leaves its distribution, its order and its CE unchanged, and is no change. Top-1 need not move."""
    diff = max(abs(a - b) for x, y in zip(before, after) for a, b in zip(legal_probs(x), legal_probs(y)))
    if not diff > POLICY_TOL:
        raise Stop("change", f"no measurable policy change on the training rows: max |change in a legal "
                             f"probability| {diff:.3g} <= {POLICY_TOL:g}")
    ce = lambda zs: sum(-math.log(legal_probs(z)[r["chosen_index"]]) for z, r in zip(zs, rows)) / len(rows)
    return {"max_abs_prob_change": diff, "tolerance": POLICY_TOL,
            "train_ce_incumbent": ce(before), "train_ce_candidate": ce(after),
            "train_top1_changed": sum(EL.top1(x) != EL.top1(y) for x, y in zip(before, after)),
            "train_top1_teacher_incumbent": sum(EL.top1(x) == r["chosen_index"] for x, r in zip(before, rows)),
            "train_top1_teacher_candidate": sum(EL.top1(y) == r["chosen_index"] for y, r in zip(after, rows))}


# ─────────────────────────────── the stage, in a child process ───────────────────────────────

def stage2(run_dir: pathlib.Path, rec: Dict[str, Any]) -> None:
    steps = rec["steps"] = EL._Steps(run_dir)
    plan, plan_sha = load_plan()
    steps["plan"] = {"sha256": plan_sha, "n_slots": plan["n_slots"]}
    if not EL._committed(MANIFEST):
        raise Stop("manifest", f"{MANIFEST} is absent or not committed")
    manifest = check_manifest()
    steps["manifest_sha256"] = MANIFEST_SHA256
    train, cached = load_inputs(manifest)
    steps["inputs"] = {"train_rows": len(train), "rows_sha256": manifest["stage1"]["rows"]["sha256"],
                       "logits_sha256": manifest["stage1"]["logits"]["sha256"]}
    steps["control"] = fit_control(train, cached)
    network, inc_path = load_incumbent(manifest)
    steps["surface"] = {"trainable": head_only(network), "parameters": N_TRAINABLE}
    records = records_of(train)
    before, values_before = forward_rows(network, records)
    schedule = sampling_schedule(plan["seed"], len(train))
    steps["schedule_sha256"] = EL._jsha(schedule.tolist())
    log = train_head(network, records, schedule, log=lambda e: EL.progress(run_dir, e))
    if len(log) != STEPS:
        raise Stop("training", f"{len(log)} of {STEPS} steps")
    steps["training"] = {"steps": len(log), "first_loss": log[0]["loss"], "last_loss": log[-1]["loss"],
                         "max_grad_norm": max(s["grad_norm"] for s in log)}
    cand = run_dir / CANDIDATE
    steps["candidate_sha256"] = save_candidate(network, cand)
    steps["candidate_surface"] = surface_check(inc_path, cand)
    reloaded = reload_check(network, cand)
    after, values_after = forward_rows(reloaded, records)
    if values_after != values_before:
        raise Stop("values", "a raw value changed on a training position")
    steps["values_unchanged"] = len(values_after)
    steps["policy_change"] = policy_change(before, after, train)


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
        stage2(run_dir, rec)
        rec["verdict"], code = "PASS", EXIT_PASS
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

def run() -> int:
    """THE entry point, stage 1's parent pattern: gate, clean tree, run once; the child under the 30-minute
    ceiling in its own process group; finalized and copied to evidence ONLY once that group is reaped."""
    import subprocess

    _require_gate()
    if EL._git("status", "--porcelain"):
        raise Refused("the working tree is dirty: run only from a committed state")
    run_dir, evidence = ROOT / RUN_DIR, ROOT / EVIDENCE_DIR
    if evidence.exists():
        raise Refused(f"{EVIDENCE_DIR} exists: stage 2 runs once")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir()                                          # the claim: exclusive
    except FileExistsError:
        raise Refused(f"{RUN_DIR} exists: stage 2 runs once") from None
    t0 = time.time()
    fin: Dict[str, Any] = {"head": EL._git("rev-parse", "HEAD"), "python": sys.version,
                           "module_sha256": EL._sha(pathlib.Path(__file__).read_bytes()),
                           "gate_sha256": EL._sha(pathlib.Path(GATE.__file__).read_bytes()),
                           "started_unix": t0, "ceiling_s": EXECUTION_S}
    EL._write_json(run_dir / "claim.json", fin)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    EL._write_new(run_dir / "environment.txt", (f"python {sys.version}\nexecutable {sys.executable}\n"
                                                f"pip freeze exit {freeze.returncode}\n{freeze.stdout}").encode())
    status, child_exit, reaped = "UNKNOWN", None, None
    try:
        with open(run_dir / "child.stdout.txt", "xb") as out, open(run_dir / "child.stderr.txt", "xb") as err:
            status, child_exit, reaped = EL.run_group([sys.executable, "-m", MODULE, "_child", "--run-dir", str(run_dir)],
                                                      stdout=out, stderr=err, timeout=EXECUTION_S)
    finally:
        code = EL.exit_code_for(run_dir / "record.json", status, child_exit, reaped)
        fin.update(status=status, child_exit=child_exit, group_reaped=reaped, exit_code=code,
                   elapsed_s=round(time.time() - t0, 3))
        if reaped is not True:
            EL._write_json(ROOT / CLEANUP_FAILED, {**fin, "finalized": False, "run_dir": RUN_DIR,
                                                   "detail": "the process group was not reaped (or never ran): "
                                                             "nothing hashed, copied or finalized"})
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

    ap = argparse.ArgumentParser(description="External-engine learning, stage 2 (gated).")
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
