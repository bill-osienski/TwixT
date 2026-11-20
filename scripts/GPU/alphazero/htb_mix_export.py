"""H-MX export and parity (card docs/superpowers/2026-11-18-htb-mix-distillation-card.md §8), behind its own gate.

    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.htb_mix_export run export     # HTB_MIX_EXPORT_AUTHORIZED

H-OP's §9, unchanged, for the H-MX challenger. Since recovery amendment 1
(docs/superpowers/2026-11-19-htb-mix-recovery-amendment.md §6) the challenger is admitted ONLY from a clean, COMPLETED
recovery whose saved file reloaded at the failed fit's trained-state digest (`htb_mix_recovery.admission`); the
failed fit record (status FAILED, entry 73) is never admissible. Then: export the challenger ONCE with the unmodified
exporter and August's invocation (only the weights path changes), place the pair by content address outside the
product manifest (`~/twixt-cache/htb-mix/export/<id>/`), and qualify it before any game: identical graph ops, inputs,
outputs and initializer shapes to `78c16a31c3dba37f`'s; both runtime canaries; V2-R's S1-S4 with the unmodified tools
and tolerances; V3's mirror-averaged parity. Any failure STOPs; there is no retry, re-export or tolerance change.

Reused, never modified: `htb_onpolicy_export`'s model id, structural signature, manifest builder, export command and
qualification commands. The qualification's V3 tools are H-OP's parameterized copies (`tests/parity/htb_onpolicy_mirror`),
which take the challenger from the command line; their report's free-text "side" label reads "H-OP challenger", and the
record's challenger sha256 and state digest identify the file actually compared.

EXPORT ATTEMPT 2 (docs/superpowers/2026-11-20-htb-mix-export-attempt-2-amendment.md): attempt 1 (`run export`) is VOID
(ledger entry 75; the launch worktree had no node_modules) and its evidence stays as written. `run export-2` runs the
same `run_export` body into fresh, create-only evidence (`htb-mix-export-2/`), after binding attempt 1's committed VOID
record and outer.json by their hashes and checking that onnxruntime-node 1.23.2 is installed in this tree:

    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.htb_mix_export run export-2   # HTB_MIX_EXPORT_AUTHORIZED
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import pathlib
import shutil
import sys
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from . import htb_mix as M
from . import htb_mix_recovery as R
from . import htb_onpolicy_export as X
from . import htb_pilot as P

CURRENT_ID, CURRENT_DIR = X.CURRENT_ID, X.CURRENT_DIR
EXPORT_ROOT = M.CACHE / "export"
SCRATCH = M.CACHE / "export-scratch"
RECORD_KIND = "htb-mix-export-record/1"
Stop, Refused = P.Stop, P.Refused


def build_manifest(template: Mapping[str, Any], mid: str, graph: Mapping[str, Any], data: Mapping[str, Any],
                   source: Mapping[str, str], export_commit: str, runtime_versions: str) -> Dict[str, Any]:
    """H-OP's manifest builder (the current pair's contract, this pair's identity and provenance), with H-MX's texts."""
    m = X.build_manifest(template, mid, graph, data, source, export_commit, runtime_versions)
    m["description"] = ("ONNX export of the H-MX challenger (card docs/superpowers/2026-11-18-htb-mix-distillation-"
                        "card.md §8). Identity and provenance only; held outside the product manifest for the H-MX "
                        "match, never served as hard by this record.")
    m["provenance_note"] = ("Exported once, to scratch, by `htb_mix_export run export`, then placed by content address "
                            "outside the repository. No alternate export was produced.")
    m["claims"] = ("Identity and provenance only. No parity, strength, serving-status or deployment-status claim: the "
                   "H-MX export record and its ledger entry own those conclusions.")
    return m


def recovery_admission(cfg: M.Config) -> Dict[str, Any]:
    """Amendment §6: only a clean, COMPLETED recovery with a verified challenger is exported (card §13's ELIGIBLE, by
    the recovery), and only the very file it saved and reloaded."""
    adm = R.admission(cfg)
    ch = adm["challenger"]
    raw = pathlib.Path(ch["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ch["sha256"]:
        raise Refused("the challenger file is not the one the recovery saved and verified")
    return {**adm, "challenger": {**ch, "sha1": hashlib.sha1(raw).hexdigest()}}


def run_export(cfg: Optional[M.Config] = None, runner: Callable[..., int] = X._run_real,
               py: Optional[pathlib.Path] = None, export_root: pathlib.Path = EXPORT_ROOT,
               scratch: pathlib.Path = SCRATCH, versions: Callable[[pathlib.Path], str] = X.runtime_versions,
               extra: Optional[Mapping[str, Any]] = None) -> int:
    """Card §8. `extra` (attempt 2 only) adds the amendment's fields to the record; nothing else differs."""
    cfg = cfg or M.Config()
    M.preconditions("export", cfg)
    ev = cfg.ev("export")
    P._absent(ev, export_root, scratch)
    adm = recovery_admission(cfg)
    ch = adm["challenger"]
    py = py or (P.main_checkout(cfg.root) / ".venv/bin/python")
    rec = M._record("export", cfg, admission=adm)
    rec["kind"] = RECORD_KIND
    rec["runtime_versions"] = versions(cfg.root)
    rec.update(extra or {})

    def body(rec):
        scratch.mkdir(parents=True)
        if runner("export", X.export_command(py, ch["path"], scratch),
                  {"MLX_ENABLE_TF32": "0", "PYTHONPATH": str(cfg.root)}, cfg.root, ev / "export.log") != 0:
            raise Stop("export", "the exporter failed; see export.log")
        if hashlib.sha256(pathlib.Path(ch["path"]).read_bytes()).hexdigest() != ch["sha256"]:
            raise Stop("export", "the challenger file changed during the export")
        g, d = X._file(scratch / "model.onnx"), X._file(scratch / "model.onnx.data")
        mid = X.model_id(g["sha256"], d["sha256"])
        if mid == CURRENT_ID:
            raise Stop("export", "the challenger's pair is the current pair")
        sig_new = X.graph_signature(scratch / "model.onnx")
        sig_cur = X.graph_signature(cfg.root / CURRENT_DIR / "model.onnx")
        rec["graph_signature_sha256"] = {"challenger": P._jsha(sig_new), "current": P._jsha(sig_cur)}
        if sig_new != sig_cur:
            raise Stop("graph", "the challenger's graph ops, inputs, outputs or initializer shapes differ from the "
                                "current pair's")
        pair_dir = export_root / mid
        pair_dir.mkdir(parents=True)
        for f in ("model.onnx", "model.onnx.data"):
            shutil.copyfile(scratch / f, pair_dir / f)
        template = json.loads((cfg.root / CURRENT_DIR / "manifest.json").read_text())
        manifest = build_manifest(template, mid, g, d, ch, rec["head"], rec["runtime_versions"])
        with open(pair_dir / "manifest.json", "x") as fh:
            fh.write(json.dumps(manifest, indent=2) + "\n")
        rec["challenger"] = {"model_id": mid, "dir": str(pair_dir), "graph_sha256": g["sha256"],
                             "external_data_sha256": d["sha256"], "source_sha256": ch["sha256"],
                             "source_state_digest": ch["state_digest"]}
        rec["steps"] = {}
        for name, argv, env in X.commands(cfg.root, py, ch, pair_dir, scratch, ev):
            rc = runner(name, argv, env, cfg.root, ev / f"{name}.log")
            rec["steps"][name] = rc
            if rc != 0:
                raise Stop(name, f"{name} exited {rc}; see {name}.log (no retry, no tolerance change)")
        for f in ("python_side.json", "node_side.json", "v3_python.json", "v3_node.json"):
            with open(scratch / f, "rb") as src, gzip.open(ev / f"{f}.gz", "xb") as dst:
                shutil.copyfileobj(src, dst)
        s = json.loads((ev / "parity_result.json").read_text())
        v3 = json.loads((ev / "v3_result.json").read_text())
        if s.get("verdict") != "PASS" or v3.get("verdict") != "PASS":
            raise Stop("parity", f"S1-S4 {s.get('verdict')}, V3 {v3.get('verdict')}")
        rec["verdict"] = "PARITY_PASS"

    return P._terminal(ev, rec, body)


# ─────────────── export attempt 2 (docs/superpowers/2026-11-20-htb-mix-export-attempt-2-amendment.md) ───────────────

AMENDMENT2 = "docs/superpowers/2026-11-20-htb-mix-export-attempt-2-amendment.md"
AMENDMENT2_SHA256 = "f8ce7792afc73fc3262ff0fb6aff58925d4b060d6b9a97b38fff94ea7c2ac032"
EV2 = "docs/superpowers/evidence/htb-mix-export-2"
OUTER2 = "docs/superpowers/evidence/htb-mix-export-2-run"
#: attempt 1 (ledger entry 75), preserved: bound by bytes and commit, never written (amendment §1)
ATTEMPT_1 = {"record": f"{M.EV['export']}/record.json",
             "record_sha256": "127f0607cfa0bbd2b6b26b25e786d5f689b8c9cb79546be45d89cc7554f4febb",
             "outer": f"{M.OUTER['export']}/outer.json",
             "outer_sha256": "7dab10720a6c13a6e1176e2b5fe607cd553a33182543914eb100c77a6ff2bcf7",
             "status": "VOID"}
ORT_NODE_VERSION = "1.23.2"
ORT_PACKAGE = "node_modules/onnxruntime-node/package.json"


def node_dependency(root: pathlib.Path) -> Dict[str, str]:
    """Amendment §3, the Python side: a real (not symlinked) node_modules in THIS tree holding onnxruntime-node 1.23.2.
    Refused otherwise, before the record opens and before `runtime_versions` reads the same file."""
    nm = root / "node_modules"
    if nm.is_symlink() or not nm.is_dir():
        raise Refused(f"{nm} is not a real directory in this tree (clone it before the gate opens: amendment §3)")
    pkg = root / ORT_PACKAGE
    if pkg.is_symlink() or not pkg.is_file():
        raise Refused(f"{pkg} is missing: onnxruntime-node is not installed in this tree")
    try:
        version = json.loads(pkg.read_text())["version"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise Refused(f"{pkg} is unreadable: {e}") from None
    if version != ORT_NODE_VERSION:
        raise Refused(f"onnxruntime-node {version} in this tree; the amendment pins {ORT_NODE_VERSION}")
    return {"onnxruntime_node": version, "package": str(pkg)}


@dataclasses.dataclass
class Attempt2Config(M.Config):
    """The card's H-MX config with attempt 2's evidence directory and bindings. The defaults ARE the amendment. (Named
    `export_amendment_sha256`, not `amendment_sha256`: the recovery admission reads the latter as ITS amendment.)"""
    export_amendment_sha256: str = AMENDMENT2_SHA256
    attempt_1: Mapping[str, str] = dataclasses.field(default_factory=lambda: ATTEMPT_1)
    node_dependency: Callable[[pathlib.Path], Dict[str, str]] = node_dependency

    def ev(self, stage: str) -> pathlib.Path:
        return self.root / EV2 if stage == "export" else super().ev(stage)


def attempt_2_bindings(cfg: Attempt2Config) -> Dict[str, Any]:
    """Amendment §2, steps 1-4, in order: the card's preconditions, this amendment's pin, attempt 1's VOID evidence,
    the Node dependency. Returns the record's added fields; refuses before anything is written."""
    M.preconditions("export", cfg)
    if cfg.ev("export") != cfg.root / EV2:
        raise Refused(f"attempt 2 writes only under {EV2}")
    M.read_pinned(cfg, AMENDMENT2, cfg.export_amendment_sha256)
    a1 = cfg.attempt_1
    try:
        rec1 = json.loads(M.read_pinned(cfg, a1["record"], a1["record_sha256"]))
    except ValueError as e:
        raise Refused(f"{a1['record']} is unparseable: {e}") from None
    M.read_pinned(cfg, a1["outer"], a1["outer_sha256"])
    if rec1.get("status") != a1["status"]:
        raise Refused(f"{a1['record']} is not attempt 1's {a1['status']} record")
    dep = cfg.node_dependency(cfg.root)
    return {"attempt": 2, "export_amendment_sha256": cfg.export_amendment_sha256, "attempt_1": dict(a1),
            "node_dependency": dep}


def run_export_2(cfg: Optional[Attempt2Config] = None, runner: Callable[..., int] = X._run_real,
                 py: Optional[pathlib.Path] = None, export_root: pathlib.Path = EXPORT_ROOT,
                 scratch: pathlib.Path = SCRATCH, versions: Callable[[pathlib.Path], str] = X.runtime_versions) -> int:
    """Amendment §2: the bindings, then `run_export` unchanged into attempt 2's evidence directory."""
    cfg = cfg or Attempt2Config()
    extra = attempt_2_bindings(cfg)
    return run_export(cfg, runner=runner, py=py, export_root=export_root, scratch=scratch, versions=versions,
                      extra=extra)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="htb_mix_export")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").add_argument("what", choices=("export", "export-2"))
    args = ap.parse_args(argv)
    try:
        return run_export_2() if args.what == "export-2" else run_export()
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return P.EXIT["REFUSED"]


if __name__ == "__main__":
    sys.exit(main())
