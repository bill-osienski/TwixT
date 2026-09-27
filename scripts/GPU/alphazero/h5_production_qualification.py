"""H5 PRODUCTION-PATH QUALIFICATION -- EARLY and BINDING. GATED; NOT RUN.

Frozen by `docs/superpowers/2026-09-27-t1j-h5-implementation-card.md` §3 and §4.
🔴 THE CARD IS THE AUTHORITY.

ONE module, TWO roles, and the role is DERIVED from `h5_runner.SEED_BLOCKS` --
never chosen by a flag, an argument or the environment:

  empty                                    -> EARLY    finds production-path faults
                                                       before any H5 seed is
                                                       registered; binds NOTHING
  segments 0-3, 148 seeds each, disjoint,  -> BINDING  qualifies the FINAL code
  every seed ACCOUNTED and not unavailable             state; the only record the
                                                       manifests may bind to
  anything else                            -> STOP

No game, no T1j query or replay JVM, no research seed, no outcome. H4's 3P-b
stages under H5's names, with H4's durability rules (claim first; each completed
stage fsynced to `stages.jsonl` before the next; a STOP keeps every earlier
observation), plus `h4_equivalence` (card §4.2): "unchanged from H4" as a CHECK
against the PINNED H4 study manifest, never a claim.
"""
from __future__ import annotations

import ast
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, Optional, Sequence, Tuple

from . import d1_probe as D1
from . import h4_pilot_feasibility as F
from . import h4_production_qualification as Q4
from . import h4_runner as R4
from . import h5_production_qualification_authorization as QAUTH
from . import h5_runner as H5
from . import t1j_adapter as A

DESIGN = H5.STUDY_DESIGN
#: A named SYNTHETIC seed: it reaches one incumbent decision and nothing else.
SEED = -1
COMPILE_DEADLINE_S = Q4.COMPILE_DEADLINE_S
BLOCK_SEEDS = 2 * H5.PAIRS_PER_SEGMENT
EXIT_CLEAN, EXIT_STOP, EXIT_REFUSED, EXIT_UNAUTHORIZED = 0, 2, 4, 5

EARLY, BINDING, STOP = "EARLY", "BINDING", "STOP"
#: Card §3: the ONLY directory a BINDING record may be written to. Fixed and
#: create-only, so a second BINDING run refuses.
BINDING_DIR = "docs/superpowers/evidence/t1j-h5-binding-qualification"
EVIDENCE_ROOT = R4.EVIDENCE_ROOT

#: 🔴 Card §4.1: the PINNED H4 inputs, checked BEFORE their content is parsed.
H4_STUDY_MANIFEST = "docs/superpowers/evidence/t1j-h4-manifests/study_manifest.json"
H4_STUDY_MANIFEST_SHA256 = "24f40b3f35c605e9bb9a158393a5495901fde86771e3127799d856506d147ff1"
H4_MANIFEST_COMMIT = "b720b282dd1653e6275a0051eea6f2506a79ac6e"
REGISTRY = "scripts/GPU/alphazero/e4_screen_reference.py"
H4_REGISTRY_BLOB_SHA256 = "c1021279a2dd37a183c5c7b10577f2a8b541b6714da6c78a6259da4b58228258"
#: Card §4.2 item 2: the ONLY assignments the registry may have changed since H4.
INTERVAL_REGISTRIES = ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS",
                       "RETIRED_SEED_INTERVALS", "TEST_ONLY_SEED_INTERVALS")

#: The failing-stage name for anything after the claim and before `role` begins.
SETUP = "setup"
STAGES = ("role", "incumbent_identity", "checkpoint", "incumbent_move", "compile_a",
          "compile_b", "reproducibility", "study_cli", "h4_equivalence", "bound_identity")

QualificationRefused = Q4.QualificationRefused


class QualificationUnauthorized(QualificationRefused):
    """The gate is closed."""


class _Stop(Exception):
    pass


def _stop(reason: str):
    raise _Stop(reason)


# ─────────────────────────────── the role, derived ───────────────────────────────

def derive_role() -> Tuple[str, Dict[str, Any], Optional[str]]:
    """(role, observation, reason-if-STOP), from `SEED_BLOCKS` and the shared
    registry as they are NOW. Pure: it opens no file and starts no process."""
    blocks = dict(H5.SEED_BLOCKS)
    seen = {"seed_blocks": {str(k): list(v) for k, v in sorted(blocks.items())}}
    if not blocks:
        return EARLY, seen, None
    if sorted(blocks) != list(H5.SEGMENTS):
        return STOP, seen, f"SEED_BLOCKS names segments {sorted(blocks)}, not {list(H5.SEGMENTS)}"
    sizes = {k: hi - lo for k, (lo, hi) in blocks.items()}
    if any(n != BLOCK_SEEDS for n in sizes.values()):
        return STOP, seen, f"block sizes {sizes}, not {BLOCK_SEEDS} each"
    seeds = [s for lo, hi in blocks.values() for s in range(lo, hi)]
    if len(set(seeds)) != len(seeds):
        return STOP, seen, "two H5 blocks overlap"
    faults = R4.seed_registry_faults([{"seed": s} for s in seeds])
    if faults:
        return STOP, seen, f"seeds fail the shared registry: {faults[:3]}"
    return BINDING, seen, None


# ─────────────────────── h4_equivalence, card §4.2 ───────────────────────

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"))


def pinned_h4_manifest() -> Dict[str, Any]:
    """The H4 study manifest, DURABLE and hashing to its pin BEFORE it is parsed."""
    path = os.path.join(R4.REPO_ROOT, H4_STUDY_MANIFEST)
    try:
        F.check_durable([path], _fixture=False)
    except F.H4AnalysisRefused as e:
        _stop(f"the pinned H4 study manifest: {e}")
    with open(path, "rb") as fh:
        raw = fh.read()
    if _sha(raw) != H4_STUDY_MANIFEST_SHA256:
        _stop(f"{H4_STUDY_MANIFEST} does not hash to its pin")
    return json.loads(raw)


def _registry_body(source: str) -> str:
    """The module's AST with the four interval registries REMOVED."""
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not (
        isinstance(n, (ast.Assign, ast.AnnAssign)) and any(
            getattr(t, "id", None) in INTERVAL_REGISTRIES
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target])))]
    return ast.dump(tree)


def h4_registry_source() -> str:
    """The registry as H4 bound it: `git show` at the pinned commit, hash-checked."""
    r = subprocess.run(["git", "-C", str(R4.REPO_ROOT), "show",
                        f"{H4_MANIFEST_COMMIT}:{REGISTRY}"], capture_output=True)
    if r.returncode != 0:
        _stop(f"`git show {H4_MANIFEST_COMMIT[:7]}:{REGISTRY}` failed")
    if _sha(r.stdout) != H4_REGISTRY_BLOB_SHA256:
        _stop(f"the registry at {H4_MANIFEST_COMMIT[:7]} does not hash to its pin")
    return r.stdout.decode("utf-8")


def h4_equivalence(bound: Dict[str, Any]) -> Dict[str, Any]:
    """STOPs unless every condition of card §4.2 holds."""
    m = pinned_h4_manifest()
    entries = m.get("segments") or []
    fields = ("code", "cards", "t1j_runtime", "incumbent_identity")
    if len(entries) != 4 or any(_canon(e.get(f)) != _canon(entries[0].get(f))
                                for e in entries for f in fields):
        _stop("the pinned H4 manifest's four entries do not share code, cards, "
              "runtime and identity")
    h4 = entries[0]
    moved = [p for p, h in h4["code"].items()
             if p != REGISTRY and R4._sha256_or_none(R4.REPO_ROOT / p) != h]      # 1.
    if moved:
        _stop(f"H4 play-path files changed since H4 was bound: {moved[:3]}")
    now = (R4.REPO_ROOT / REGISTRY).read_text(encoding="utf-8")                  # 2.
    if _registry_body(now) != _registry_body(h4_registry_source()):
        _stop(f"{REGISTRY} changed outside its four interval registries since H4")
    cards = [p for p, h in h4["cards"].items() if R4._sha256_or_none(R4.REPO_ROOT / p) != h]
    if cards:                                                                    # 3.
        _stop(f"H4 cards changed since H4 was bound: {cards}")
    if _canon(bound["t1j_runtime"]) != _canon(h4["t1j_runtime"]):                # 4.
        diff = sorted(k for k in set(bound["t1j_runtime"]) | set(h4["t1j_runtime"])
                      if _canon(bound["t1j_runtime"].get(k)) != _canon(h4["t1j_runtime"].get(k)))
        _stop(f"the T1j runtime differs from H4's in {diff}")
    ours, theirs = dict(bound["incumbent_identity"]), dict(h4["incumbent_identity"])  # 5.
    if (ours.pop("design", None), theirs.pop("design", None)) != (DESIGN, "H4_STUDY") \
            or _canon(ours) != _canon(theirs):
        _stop("the incumbent identity differs from H4's in more than `design`")
    return {"h4_manifest_sha256": H4_STUDY_MANIFEST_SHA256,
            "h4_code_files_equal": len(h4["code"]) - 1, "registry_changed_only_in":
            list(INTERVAL_REGISTRIES), "h4_cards_equal": len(h4["cards"]),
            "t1j_runtime_equal": True, "identity_differs_only_in": ["design"]}


# ─────────────────────────────── the stages ───────────────────────────────

def _study_cli_refuses(tmp: str) -> int:
    r = subprocess.run([sys.executable, "-m", "scripts.GPU.alphazero.h5_runner",
                        "--manifest", os.path.join(tmp, "none.json"), "--segment", "0",
                        "--out-dir", os.path.join(tmp, "out"),
                        "--classes", os.path.join(tmp, "cls")],
                       cwd=str(R4.REPO_ROOT), capture_output=True, text=True)
    return r.returncode


def _checks(classes_root: str, log, role: Tuple[str, Dict[str, Any], Optional[str]]) -> None:
    from .game.twixt_state import TwixtState
    log.begin("role")
    if role[0] == STOP:
        _stop(f"role: {role[2]}")
    log.done("role", {"role": role[0], **role[1]})

    log.begin("incumbent_identity")
    config = R4.frozen_argmax_config()
    identity = R4.frozen_incumbent_identity(config, design=DESIGN)
    try:
        R4.check_incumbent_identity(identity, config, design=DESIGN)
    except R4.H4RunError as e:
        _stop(f"incumbent identity: {e}")
    log.done("incumbent_identity", identity)

    log.begin("checkpoint")
    try:
        evaluator = R4.load_production_evaluator(identity)
    except R4.H4RunError as e:
        _stop(f"checkpoint: {e}")
    log.done("checkpoint", {"reference": identity["reference"],
                            "reference_sha1": identity["reference_sha1"]})

    log.begin("incumbent_move")
    task = R4.make_schedule([("h5-production-qualification", SEED, SEED - 1)],
                            mode="fixture", reference=identity)[0]
    state = TwixtState(active_size=A.BOARD_N, to_move="red")
    move = R4.production_incumbent_build(config)(task, evaluator)(state)
    move = tuple(move) if move is not None else None
    if move is None or move not in set(state.legal_moves()):
        _stop(f"the incumbent returned {move}, not a legal empty-board move")
    log.done("incumbent_move", list(move))

    compiled = {}
    for stage, leaf in (("compile_a", "compile-a"), ("compile_b", "compile-b")):
        log.begin(stage)
        paths = R4._verified_t1j_paths(os.path.join(classes_root, leaf))
        R4.check_destinations(os.path.join(classes_root, "unused-out"), paths.classes,
                              mode="study")
        deadline = D1.Deadline(limit_s=COMPILE_DEADLINE_S)
        deadline.start()
        with D1._supervisor(deadline):
            content, local = R4.split_toolchain(D1._default_compile(deadline, paths=paths))
        compiled[stage] = content
        log.done(stage, {"content": content, "local": local})

    log.begin("reproducibility")
    a, b = compiled["compile_a"], compiled["compile_b"]
    if a != b:
        diff = sorted(k for k in a if a[k] != b[k])
        _stop(f"the helper does not compile reproducibly: {diff} differ between two "
              f"compiles, so class hashes cannot be bound in advance")
    log.done("reproducibility", {"identical": True})

    log.begin("study_cli")
    rc = _study_cli_refuses(classes_root)
    if rc != H5.EXIT_UNAUTHORIZED:
        _stop(f"the H5 study CLI exited {rc}, not {H5.EXIT_UNAUTHORIZED}, with its gate shut")
    log.done("study_cli", {"exit": rc})

    bound = H5.bound_identity(identity, a)
    log.begin("h4_equivalence")
    log.done("h4_equivalence", h4_equivalence(bound))

    log.begin("bound_identity")
    log.done("bound_identity", bound)


def _destination(out_dir: str, role: str) -> None:
    """Card §3: BINDING writes ONLY to the fixed directory; anything else never does."""
    binding = os.path.realpath(os.path.join(R4.REPO_ROOT, BINDING_DIR))
    here = os.path.realpath(out_dir)
    if role == BINDING and here != binding:
        raise QualificationRefused(f"a BINDING qualification writes ONLY to {BINDING_DIR}, "
                                   f"not {out_dir}")
    if role != BINDING:
        if here == binding:
            raise QualificationRefused(f"an {role} qualification may never write to the "
                                       f"binding directory {BINDING_DIR}")
        if not R4._inside(pathlib.Path(here), pathlib.Path(EVIDENCE_ROOT)):
            raise QualificationRefused(f"an {role} record goes under the evidence tree, "
                                       f"not {out_dir}")


def _qualify(out_dir: str, classes_root: str) -> Dict[str, Any]:
    """Every refusal BEFORE the claim; the record directory claimed BEFORE the first
    effectful stage; the stage log durable stage by stage."""
    role = derive_role()
    _destination(out_dir, role[0])
    if os.path.lexists(out_dir):
        raise QualificationRefused(f"{out_dir} is occupied; the record is create-only")
    if R4._inside(pathlib.Path(classes_root), R4.REPO_ROOT):
        raise QualificationRefused(f"the class root must be OUTSIDE the repository: "
                                   f"{classes_root}")
    if os.path.lexists(classes_root):
        raise QualificationRefused(f"{classes_root} is occupied; compiles are create-only")
    os.mkdir(out_dir)                             # THE CLAIM: create-only, before any stage
    #: 🔴 EVERYTHING after the claim is inside the recorded-STOP handler -- the stage
    #: log's creation and the class root's included -- so a failure there leaves a
    #: STOP `record.json` naming `setup`, never an occupied directory without one.
    #: (H4's 3P-b created the class root OUTSIDE its handler; it never failed there.)
    log = None
    try:
        try:
            log = Q4._StageLog(os.path.join(out_dir, "stages.jsonl"))
            log.begin(SETUP)
            os.makedirs(classes_root)
            _checks(classes_root, log, role)
            result, failing, reason = "CLEAN", None, None
        except _Stop as e:
            result, failing, reason = "STOP", log.current, str(e)
        except Exception as e:                    # noqa: BLE001 -- any failure is a STOP, recorded
            result, failing, reason = ("STOP", log.current if log else SETUP,
                                       f"{type(e).__name__}: {e}")
    finally:
        if log is not None:
            log.close()
    note = "NOT evidence of strength: no game was played and no outcome exists"
    if role[0] != BINDING:
        note += f". {role[0]}: this record binds NOTHING; the manifest writer refuses it"
    record = {"record": "H5_PRODUCTION_QUALIFICATION", "role": role[0], "result": result,
              "failing_stage": failing, "reason": reason,
              "checks": log.completed if log is not None else {},
              "stages": list(STAGES), "evidence_note": note, "seed": SEED,
              "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    Q4._write_create_only(os.path.join(out_dir, "record.json"), record)
    return record


def qualify(out_dir: str, classes_root: str) -> Dict[str, Any]:
    """PUBLIC ENTRY. The gate is read FIRST, before any file or process."""
    if not QAUTH.H5_PRODUCTION_QUALIFICATION_AUTHORIZED:
        raise QualificationUnauthorized("the H5 production qualification is UNAUTHORIZED. "
                                        "Nothing was loaded, compiled or written.")
    return _qualify(out_dir, classes_root)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """PUBLIC ENTRY."""
    import argparse
    ap = argparse.ArgumentParser(prog="h5_production_qualification",
                                 description="H5 EARLY/BINDING qualification. NOT AUTHORIZED.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--classes-root", required=True)
    a = ap.parse_args(argv)
    if not QAUTH.H5_PRODUCTION_QUALIFICATION_AUTHORIZED:
        print("the H5 production qualification is UNAUTHORIZED. Nothing was loaded, "
              "compiled or written.", file=sys.stderr)
        return EXIT_UNAUTHORIZED
    try:
        rec = qualify(a.out_dir, a.classes_root)
    except QualificationRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"{rec['role']} {rec['result']}: {os.path.join(a.out_dir, 'record.json')}")
    return EXIT_CLEAN if rec["result"] == "CLEAN" else EXIT_STOP


if __name__ == "__main__":                                                        # pragma: no cover
    raise SystemExit(main())
