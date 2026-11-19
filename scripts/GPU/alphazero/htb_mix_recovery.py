"""H-MX recovery amendment 1: one exact replay of the failed fit, with a verified save, behind its own gate.

Amendment: docs/superpowers/2026-11-19-htb-mix-recovery-amendment.md, pinned below by sha256. The card
(revision 1, `d2c9b787…`) and `htb_mix.py` (`79f210cf…`, the code that trained `adba50fb…`) are unchanged.

    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.htb_mix_recovery run replay   # HTB_MIX_RECOVERY_AUTHORIZED

Nothing runs on import, and importing this module loads no MLX or model. The replay (amendment §3) binds the
committed failed fit record (entry 73) by its bytes and runs `htb_mix`'s own steps unchanged: the same pinned
starting weights, the same cached inputs, the same 16 + 16 batch order and the same 7,500 updates. It does not
score either validation set. It saves only if the whole fit record, including the trained-state digest
`adba50fb…`, equals the failed one. It publishes ELIGIBLE (with M1-M6 copied from the failed record) only after the
saved file reloads with that digest. The cache directory is prepared before anything loads (amendment §2).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from . import htb_mix as M
from . import htb_onpolicy as H
from . import htb_pilot as P

AMENDMENT = "docs/superpowers/2026-11-19-htb-mix-recovery-amendment.md"
AMENDMENT_SHA256 = "8d6e386611a5db93ccb6361c92cb00cd51ca67d364af63f4e3d01a6cb09a5a87"   # draft for review
RECORD_KIND = "htb-mix-recovery-record/1"
EV = "docs/superpowers/evidence/htb-mix-recovery"
OUTER = "docs/superpowers/evidence/htb-mix-recovery-run"
GATE = ("htb_mix_recovery_authorization", "HTB_MIX_RECOVERY_AUTHORIZED")
KILL_S = 11_700                                                     # the fit's ceiling (amendment §5)
LEDGER_LINK = "ledger Post-closure entry 73; do-not-repeat #212"

#: the failed fit (entry 73), bound by the bytes of its committed evidence
FAILED = {"record": f"{M.EV['fit']}/record.json",
          "record_sha256": "6507d588e0429a0141a85572d22d5ca06801ebb5c6c5ee8f6d5254351d710ef9",
          "outer": f"{M.OUTER['fit']}/outer.json",
          "outer_sha256": "bb8d768b101ebd743dc5afaa81619fbe663958420b7431e6d596aff25435618a",
          "head": "84ff36b16c208caf0a7c89a6f042f70630f1467c",
          "module_sha256": "79f210cf6025352dd12633772852a2d76f4b7a3b652ff9c2f77a95b5b5ffea94",
          "reference_digest": M.STUDENT["state_digest"],                    # 586dca4f…, the pinned start
          "student_digest": "adba50fb2e241ec0388fa99998b392c3e0b431b55c10676800a546899aee34b2"}
TARGET = FAILED["student_digest"]

#: files under scripts/GPU/alphazero/ that may differ from the failed launch commit (amendment §3): none is imported
#: by the fit
CODE_DIR = "scripts/GPU/alphazero"
MAY_DIFFER = frozenset(f"{CODE_DIR}/{f}" for f in ("htb_mix_fit_authorization.py", "htb_mix_export.py",
                                                     "htb_mix_recovery.py", "htb_mix_recovery_authorization.py"))

#: the runtime the failed run used (the main checkout's .venv, unchanged since 2026-05-09), amendment §3
RUNTIME = {"python": "3.14.10", "machine": "arm64", "mlx": "0.30.3", "numpy": "2.4.1",
           "libmlx_sha256": "281cfddeabcc2987b049ac504ec39706c37636f9b7080a8a6e26c61a358521e0",
           "mlx_core_sha256": "3f96007a255d48f7870519b25eb91b790f29ade90056ecbb23276b9902bc1c86",
           "prefix": str(pathlib.Path.home() / "projects/TwixT_Game/.venv")}       # the failed run's outer.json python

NOT_REPRODUCED = "STOP: the replay did not reproduce the failed fit; nothing is saved"
Stop, Refused = P.Stop, P.Refused
_sha = P._sha


def gate_open(stage: str) -> bool:
    if stage != "recovery":
        return M.gate_open(stage)
    import importlib

    mod = importlib.import_module(f"{__package__}.{GATE[0]}")
    return getattr(mod, GATE[1]) is True


def _code_changes_real(root: pathlib.Path, since: str) -> List[str]:
    out = subprocess.run(["git", "-C", str(root), "diff", "--name-only", since, "HEAD", "--", CODE_DIR],
                         capture_output=True, text=True, check=True).stdout
    return sorted(out.split())


def _runtime_real() -> Dict[str, str]:   # pragma: no cover - reads the installed runtime
    import importlib.metadata as md
    import platform

    import mlx

    base = pathlib.Path(mlx.__path__[0])
    core = sorted(base.glob("core.*.so"))
    return {"python": platform.python_version(), "machine": platform.machine(), "mlx": md.version("mlx"),
            "numpy": md.version("numpy"),
            "libmlx_sha256": _sha((base / "lib" / "libmlx.dylib").read_bytes()),
            "mlx_core_sha256": _sha(core[0].read_bytes()) if len(core) == 1 else f"{len(core)} core extensions",
            "prefix": sys.prefix}


def _reload_real(path: pathlib.Path):   # pragma: no cover - loads a network; never under the build approval
    """The student's own loader (`htb_onpolicy._load_student_real` uses it), strict, in evaluation mode."""
    from .probe_eval import load_network_for_scoring

    net = load_network_for_scoring(str(path), verbose=False)[0]
    net.eval()
    return net


@dataclasses.dataclass
class Config(M.Config):
    """The card's H-MX config plus the recovery's bindings. The defaults ARE the amendment."""
    gate_open: Callable[[str], bool] = gate_open
    amendment_sha256: str = AMENDMENT_SHA256
    failed: Mapping[str, str] = dataclasses.field(default_factory=lambda: FAILED)
    code_changes: Callable[[pathlib.Path, str], List[str]] = _code_changes_real
    fit_module_sha256: Callable[[], str] = lambda: _sha(pathlib.Path(M.__file__).read_bytes())
    runtime: Callable[[], Dict[str, str]] = _runtime_real
    runtime_pins: Mapping[str, str] = dataclasses.field(default_factory=lambda: RUNTIME)

    @property
    def ev_recovery(self) -> pathlib.Path:
        return self.root / EV


# ─────────────────────────────── preconditions and bindings (amendment §1-§3) ───────────────────────────────

def preconditions(cfg: Config) -> None:
    """The recovery gate, the card's and the amendment's pins and commits, then TF32."""
    if cfg.gate_open("recovery") is not True:
        raise Refused(f"{GATE[1]} is closed")
    if _sha((cfg.root / M.CARD).read_bytes()) != M.CARD_SHA256 or not cfg.committed(cfg.root, M.CARD):
        raise Refused("the card is not the pinned, committed revision 1")
    p = cfg.root / AMENDMENT
    if not p.is_file() or _sha(p.read_bytes()) != cfg.amendment_sha256 or not cfg.committed(cfg.root, AMENDMENT):
        raise Refused("the amendment is not the pinned, committed one")
    if cfg.tf32_required and os.environ.get("MLX_ENABLE_TF32") != "0":
        raise Refused("MLX_ENABLE_TF32 must be 0 in this process")


def bind_failed(cfg: Config) -> Dict[str, Any]:
    """The failed fit's committed record and outer.json, by their pinned bytes, and everything the replay relies on:
    FAILED at the save after a full fit of this card, every screen held, the trained-state digest recorded."""
    f = cfg.failed
    raw = {}
    for k in ("record", "outer"):
        rel = f[k]
        if not (cfg.root / rel).is_file() or not cfg.committed(cfg.root, rel):
            raise Refused(f"{rel} is absent, uncommitted or changed")
        raw[k] = (cfg.root / rel).read_bytes()
        if _sha(raw[k]) != f[f"{k}_sha256"]:
            raise Refused(f"{rel} is not the pinned failed {k} (entry 73)")
    rec, outer = json.loads(raw["record"]), json.loads(raw["outer"])
    err = rec.get("error") or {}
    checks = {
        "kind": rec.get("kind") == "htb-mix-fit-record/1" and rec.get("card_sha256") == M.CARD_SHA256,
        "status": rec.get("status") == "FAILED" and err.get("type") == "RuntimeError"
                  and "save_safetensors" in str(err.get("detail")) and "challenger" not in rec,
        "launch": rec.get("head") == f["head"] == outer.get("head") and rec.get("module_sha256") == f["module_sha256"],
        "outer": (outer.get("inner_exit"), outer.get("outer_exit"), outer.get("outcome"), outer.get("gate_restored"),
                  outer.get("survivors"), outer.get("record_by")) == (4, 4, "FAILED", True, "", "script"),
        "fit": (rec.get("fit") or {}).get("student_digest") == f["student_digest"]
               and (rec.get("fit") or {}).get("reference_digest") == f["reference_digest"]
               and (rec.get("fit") or {}).get("updates") == cfg.updates == rec.get("updates"),
        "reproduced_s": (rec.get("s_reproduction") or {}).get("max_relative_difference", 1) <= M.S_REL_TOL,
    }
    try:
        gate = M.gate_verdict(rec["challenger_val"]["op_val"], rec["student_val"]["op_val"],
                              rec["challenger_val"]["tb_val"], rec["student_val"]["tb_val"])
        checks["screens"] = (gate == rec.get("gate") and rec.get("verdict") == gate["verdict"] == "ELIGIBLE"
                             and gate["held"] == ["M1", "M2", "M3", "M4", "M5", "M6"])
    except (KeyError, TypeError):
        checks["screens"] = False
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise Refused(f"the failed fit record does not hold what the replay relies on: {bad}")
    return rec


def code_binding(cfg: Config, failed: Mapping[str, Any]) -> Dict[str, Any]:
    """The fit's code is the failed run's: htb_mix.py byte-identical, and nothing else it can import changed."""
    mod = cfg.fit_module_sha256()
    if mod != failed["module_sha256"]:
        raise Refused(f"htb_mix.py is {mod[:12]}…, not the failed run's {failed['module_sha256'][:12]}…")
    changed = cfg.code_changes(cfg.root, failed["head"])
    extra = sorted(set(changed) - MAY_DIFFER)
    if extra:
        raise Refused(f"code under {CODE_DIR}/ changed since the failed launch commit: {extra}")
    return {"fit_module_sha256": mod, "since": failed["head"], "changed": changed}


def runtime_binding(cfg: Config) -> Dict[str, str]:
    got = cfg.runtime()
    bad = {k: (got.get(k), v) for k, v in cfg.runtime_pins.items() if got.get(k) != v}
    if bad:
        raise Refused(f"the runtime is not the failed run's: {bad}")
    return got


def prepare_cache(cfg: Config) -> Dict[str, Any]:
    """Amendment §2.1: the cache directory exists and is writable before anything loads; no challenger is there."""
    c = pathlib.Path(cfg.cache)
    existed = c.is_dir()
    try:
        c.mkdir(parents=True, exist_ok=True)
        probe = c / f".recovery-probe.{os.getpid()}"
        with open(probe, "xb") as fh:
            fh.write(b"probe\n")
        os.unlink(probe)
    except OSError as e:
        raise Refused(f"the cache directory {c} cannot be prepared: {e}") from e
    if os.path.lexists(cfg.challenger):
        raise Refused(f"{cfg.challenger} already exists")
    return {"path": str(c), "existed_before": existed, "writable": True}


def _norm(o: Any) -> Any:
    """As the record stores it: the comparison is between JSON values."""
    return json.loads(json.dumps(o, sort_keys=True, default=P._jsonable))


def compare_fit(fit: Mapping[str, Any], failed_fit: Mapping[str, Any], target: str) -> Dict[str, Any]:
    a, b = _norm(fit), _norm(failed_fit)
    differs = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return {"student_digest": {"replay": a.get("student_digest"), "failed": b.get("student_digest"),
                               "target": target, "equal": a.get("student_digest") == b.get("student_digest") == target},
            "fit_record_equal": not differs, "differs": differs}


def verified_save(cfg: Config, student, save: Callable[[Any, pathlib.Path], Dict[str, str]],
                  reload: Callable[[pathlib.Path], Any], out: Dict[str, Any], target: str) -> None:
    """Amendment §2.2: the create-only save, then a strict reload whose state digest must be `target`, and the file's
    sha256 recomputed. Fills `out` as it goes (so a STOP still records what was saved); `verified` becomes True last,
    and any mismatch STOPs before a verdict."""
    out["verified"] = False
    saved = save(student, cfg.challenger)
    path = pathlib.Path(saved["path"])
    out.update({"path": str(path), "sha256": saved.get("sha256"), "state_digest": saved.get("state_digest")})
    if path != cfg.challenger or not path.is_file():
        raise Stop("save", f"the save did not write {cfg.challenger}")
    file_sha = _sha(path.read_bytes())
    if file_sha != saved.get("sha256"):
        raise Stop("save", "the saved file's sha256 is not the one the save returned")
    if saved.get("state_digest") != target:
        raise Stop("save", f"the saved network's state digest {saved.get('state_digest')} is not {target}")
    net = reload(path)
    P._require_eval(net, "the reloaded challenger")
    got = out["reload_state_digest"] = P._digest(net)
    if got != target:
        raise Stop("reload", f"the reloaded file's state digest {got} is not {target}")
    if _sha(path.read_bytes()) != file_sha:
        raise Stop("reload", "the challenger file changed during the reload")
    out["verified"] = True


def _record(cfg: Config, **extra) -> Dict[str, Any]:
    return {"kind": RECORD_KIND, "card_sha256": M.CARD_SHA256, "amendment_sha256": cfg.amendment_sha256,
            "stage": "recovery", "links": {"ledger": LEDGER_LINK, "failed_record": cfg.failed["record"],
                                           "failed_record_sha256": cfg.failed["record_sha256"]},
            "head": cfg.head(cfg.root), "pid": os.getpid(), "kill_s": KILL_S,
            "module_sha256": _sha(pathlib.Path(__file__).read_bytes()), **extra}


# ─────────────────────────────── the replay (amendment §3) ───────────────────────────────

def run_replay(cfg: Optional[Config] = None, load_student: Callable[[], Any] = M._load_student_real,
               games: Callable[[M.Config], Mapping[str, Mapping[str, Sequence]]] = M._games_real,
               save: Callable[[Any, pathlib.Path], Dict[str, str]] = P._save_student_real,
               reload: Callable[[pathlib.Path], Any] = _reload_real) -> int:
    cfg = cfg or Config()
    preconditions(cfg)
    P._absent(cfg.ev_recovery, cfg.challenger)
    failed = bind_failed(cfg)                                     # before anything is loaded
    target = cfg.failed["student_digest"]                          # adba50fb… (amendment §3)
    code = code_binding(cfg, cfg.failed)
    runtime = runtime_binding(cfg)
    memory = P._check_available(cfg, cfg.memory_required_bytes, "recovery")
    cache = prepare_cache(cfg)                                    # the repair: before any label or model loads
    rows = M.load_sources(cfg)                                    # both label sets as written, verified
    rec = _record(cfg, code=code, runtime=runtime, memory_at_start=memory,
                  memory_required_bytes=cfg.memory_required_bytes, cache=cache, target_state_digest=target,
                  recipe=dataclasses.asdict(cfg.recipe), epoch_tag=M.FIT_TAG, updates=cfg.updates,
                  validation_scoring="skipped (amendment §3): no validation store is built and nothing is scored")

    def body(rec):
        import mlx.core as mx

        keys = {f"{s}_{a}": [m["key"] for m in rows[s][a][0]] for s in ("op", "tb") for a in ("train", "val")}
        sel = M.mix_selection(keys, cfg.exclusions(cfg.root), cfg.n_op, cfg.n_tb, cfg.tb_pool)
        rec["selection"] = sel["record"]
        if _norm(sel["record"]) != failed["selection"]:
            raise Stop("replay", f"{NOT_REPRODUCED}: the selection differs from the failed record's")
        mx.set_default_device(mx.cpu)
        mv = games(cfg)
        st = {s: P.ExampleStore(*rows[s]["train"], mv[s]) for s in ("op", "tb")}     # training rows only
        plan = M.mixed_plan(st["op"], st["tb"], sel["op_order"], sel["tb_order"], cfg.recipe.rows_per_batch // 2)
        if len(plan) != cfg.updates:
            raise Stop("fit", f"{len(plan)} batches; the card's is {cfg.updates} updates")
        reference, student = load_student(), load_student()
        rec["start"] = M.start_check(reference, plan, n=cfg.start_rows)
        if _norm(rec["start"]) != failed["start"]:
            raise Stop("replay", f"{NOT_REPRODUCED}: the start check differs from the failed record's")
        rec["fit"] = M.fit_mixed(student, reference, plan, cfg.recipe)
        rec["reproduction"] = compare_fit(rec["fit"], failed["fit"], target)
        if not (rec["reproduction"]["student_digest"]["equal"] and rec["reproduction"]["fit_record_equal"]):
            raise Stop("replay", f"{NOT_REPRODUCED}: differs in {rec['reproduction']['differs']}")
        rec["challenger"] = {}
        verified_save(cfg, student, save, reload, rec["challenger"], target)
        rec["screens"] = {"source": f"copied from the failed fit record ({LEDGER_LINK}), applied once there to this "
                                    "trained state; nothing re-scored",
                          "student_val": failed["student_val"], "challenger_val": failed["challenger_val"],
                          "gate": failed["gate"], "descriptive": failed.get("descriptive")}
        rec["verdict"] = "ELIGIBLE"

    return P._terminal(cfg.ev_recovery, rec, body)


# ─────────────────────────────── the export's admission (amendment §6) ───────────────────────────────

def admission(cfg: M.Config) -> Dict[str, Any]:
    """Only a clean, COMPLETED recovery of this card and amendment, bound to the failed record, with a verified
    challenger whose file is still the one saved, is admitted to the export."""
    rec = H.read_committed_json(cfg, f"{EV}/record.json")
    outer = H.read_committed_json(cfg, f"{OUTER}/outer.json")
    for f in H.WRAPPER_FILES[1:]:
        if not cfg.committed(cfg.root, f"{OUTER}/{f}"):
            raise Refused(f"{OUTER}/{f} is absent, uncommitted or changed")
    dirty = cfg.evidence_problems(cfg.root, [EV, OUTER])
    if dirty:
        raise Refused(f"the recovery evidence is not clean: {dirty}")
    if rec.get("status") != "COMPLETED" or rec.get("kind") != RECORD_KIND or rec.get("card_sha256") != M.CARD_SHA256 \
            or rec.get("amendment_sha256") != getattr(cfg, "amendment_sha256", AMENDMENT_SHA256):
        raise Refused(f"{EV}/record.json is not a COMPLETED recovery of this card and amendment")
    bad = [k for k, v in H.OUTER_OK if outer.get(k) != v]
    if bad:
        raise Refused(f"{OUTER}/outer.json is not clean: {[(k, outer.get(k)) for k in bad]}")
    if (outer.get("term_s"), outer.get("grace_s")) != (KILL_S - M.GRACE_S, M.GRACE_S) or rec.get("kill_s") != KILL_S:
        raise Refused(f"the recovery was not under its ceiling: outer TERM {outer.get('term_s')} / grace "
                      f"{outer.get('grace_s')}, record kill_s {rec.get('kill_s')}; the amendment's is {KILL_S} s")
    pgid_rel = f"{OUTER}/pgid.txt"
    if not cfg.committed(cfg.root, pgid_rel) or (cfg.root / pgid_rel).read_text().strip() != str(rec.get("pid")):
        raise Refused(f"{pgid_rel} is not the recorded pid {rec.get('pid')}")
    if outer.get("head") != rec.get("head"):
        raise Refused("the recovery record and its outer.json name different launch commits")
    if (rec.get("links") or {}).get("failed_record_sha256") != FAILED["record_sha256"]:
        raise Refused("the recovery is not bound to the failed fit record (entry 73)")
    if rec.get("verdict") != "ELIGIBLE":
        raise Refused(f"the recovery's verdict is {rec.get('verdict')!r}: only an ELIGIBLE, verified challenger is "
                      "exported")
    ch = rec.get("challenger") or {}
    if ch.get("verified") is not True or ch.get("reload_state_digest") != TARGET or ch.get("state_digest") != TARGET \
            or (rec.get("fit") or {}).get("student_digest") != TARGET:
        raise Refused(f"the recovery's challenger is not verified at {TARGET[:12]}…")
    if not ch.get("path") or not pathlib.Path(ch["path"]).is_file():
        raise Refused("the recovery record names no challenger file that exists")
    if _sha(pathlib.Path(ch["path"]).read_bytes()) != ch.get("sha256"):
        raise Refused("the challenger file is not the one the recovery saved and verified")
    return {"record_sha256": _sha((cfg.root / EV / "record.json").read_bytes()), "head": rec.get("head"),
            "challenger": {"path": ch["path"], "sha256": ch["sha256"], "state_digest": ch["state_digest"]}}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="htb_mix_recovery")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").add_argument("what", choices=("replay",))
    ap.parse_args(argv)
    try:
        return run_replay()
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return P.EXIT["REFUSED"]


if __name__ == "__main__":
    sys.exit(main())
