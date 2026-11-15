"""H-OP export and parity (card docs/superpowers/2026-11-14-htb-onpolicy-distillation-card.md §9), behind its own gate.

    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.htb_onpolicy_export run export     # HTB_ONPOLICY_EXPORT_AUTHORIZED

On an ELIGIBLE fit only: export the challenger ONCE with the unmodified exporter and August's invocation (only the
weights path changes), place the pair by content address outside the product manifest (`~/twixt-cache/htb-onpolicy/
export/<id>/`), and qualify it before any game:

* the graph's ops, inputs, outputs and initializer shapes equal the current pair's (`78c16a31c3dba37f`), so one
  evaluation costs the same by construction;
* both runtime canaries pass (Python ORT 1.23.2 in the isolated venv, #186; onnxruntime-node);
* the V2-R surfaces S1-S4 with the unmodified tools and tolerances (`tests/parity/run_python_side.py` in the isolated
  venv, `run_node_side.mjs`, `compare.mjs`);
* V3's mirror-averaged parity through parameterized copies of `tests/parity/htb_mirror/` (derived by
  `tests/htb_onpolicy_v3_spec.py`) and its unmodified `compare.mjs`.

Any failure STOPs; there is no retry, re-export or tolerance change. Nothing is committed under `models/`, and
`HARD_MODEL_ID` is not touched. The record is what the Node match driver admits.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import htb_onpolicy as H
from . import htb_pilot as P

CURRENT_ID = "78c16a31c3dba37f"
CURRENT_DIR = f"models/{CURRENT_ID}"
ISOLATED_PY = pathlib.Path.home() / "twixt-cache" / "ort-canary-1.23.2" / "venv" / "bin" / "python"
EXPORT_ROOT = H.CACHE / "export"
SCRATCH = H.CACHE / "export-scratch"
RECORD_KIND = "htb-onpolicy-export-record/1"
Stop, Refused = P.Stop, P.Refused


def model_id(graph_sha256: str, data_sha256: str) -> str:
    """server/model_manifest.js computeModelId: sha256(graph + ':' + external data), first 16 hex."""
    return hashlib.sha256(f"{graph_sha256}:{data_sha256}".encode()).hexdigest()[:16]


def graph_signature(path: pathlib.Path) -> Dict[str, Any]:
    """What fixes the cost of one evaluation: the node ops in order, the inputs and outputs, and every initializer's
    name, type and shape. Weights' VALUES are not part of it."""
    import onnx

    m = onnx.load(str(path), load_external_data=False)

    def vi(v):
        t = v.type.tensor_type
        return [v.name, int(t.elem_type), [d.dim_value if d.HasField("dim_value") else d.dim_param for d in t.shape.dim]]
    g = m.graph
    return {"ops": [n.op_type for n in g.node], "inputs": [vi(v) for v in g.input], "outputs": [vi(v) for v in g.output],
            "initializers": sorted([i.name, int(i.data_type), list(i.dims)] for i in g.initializer),
            "opset": sorted([o.domain, int(o.version)] for o in m.opset_import)}


def build_manifest(template: Mapping[str, Any], mid: str, graph: Mapping[str, Any], data: Mapping[str, Any],
                   source: Mapping[str, str], export_commit: str, runtime_versions: str) -> Dict[str, Any]:
    """The current pair's manifest with this pair's identity and provenance; the contract is unchanged."""
    m = copy.deepcopy(dict(template))
    m["model_id"] = mid
    m["description"] = ("ONNX export of the H-OP challenger (card docs/superpowers/2026-11-14-htb-onpolicy-distillation-"
                        "card.md §9). Identity and provenance only; held outside the product manifest for the H-OP "
                        "match, never served as hard by this record.")
    m["graph"] = {"filename": "model.onnx", "size_bytes": graph["size_bytes"], "sha256": graph["sha256"]}
    m["external_data"] = {"filename": "model.onnx.data", "size_bytes": data["size_bytes"], "sha256": data["sha256"]}
    m["provenance"] = {
        "source_checkpoint_path": source["path"], "source_checkpoint_sha1": source["sha1"],
        "source_checkpoint_sha256": source["sha256"], "source_state_digest": source["state_digest"],
        "export_commit": export_commit,
        "exporter_config": ("scripts/GPU/alphazero/export_onnx.py unmodified; MLX_ENABLE_TF32=0 .venv/bin/python -m "
                            f"scripts.GPU.alphazero.export_onnx --weights {source['path']} --output <scratch>/model.onnx; "
                            f"August's invocation for calib020_0001 and {CURRENT_ID}'s with only the weights path changed"),
        "runtime_versions": runtime_versions}
    m["provenance_note"] = ("Exported once, to scratch, by `htb_onpolicy_export run export`, then placed by content "
                            "address outside the repository. No alternate export was produced.")
    m["claims"] = ("Identity and provenance only. No parity, strength, serving-status or deployment-status claim: the "
                   "H-OP export record and its ledger entry own those conclusions.")
    return m


def _file(path: pathlib.Path) -> Dict[str, Any]:
    raw = path.read_bytes()
    return {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def commands(root: pathlib.Path, py: pathlib.Path, challenger: Mapping[str, str], pair_dir: pathlib.Path,
             scratch: pathlib.Path, ev: pathlib.Path) -> List[Tuple[str, List[str], Dict[str, str]]]:
    """Every subprocess of the qualification, in order (after the export and the structural check)."""
    tf32 = {"MLX_ENABLE_TF32": "0", "PYTHONPATH": str(root)}
    ipy = str(ISOLATED_PY)
    return [
        ("canary_python", [ipy, "tests/parity/ort_conv_canary.py", str(ev / "canary_python.json")], tf32),
        ("canary_node", ["node", "--test", "server/test_ort_conv_canary.js"], {}),
        ("s_python", [ipy, "tests/parity/run_python_side.py", str(pair_dir), str(scratch / "python_side.json")], tf32),
        ("s_node", ["node", "tests/parity/run_node_side.mjs", str(pair_dir), str(scratch / "node_side.json")], {}),
        ("s_compare", ["node", "tests/parity/compare.mjs", str(scratch / "python_side.json"),
                       str(scratch / "node_side.json"), str(ev / "parity_result.json")], {}),
        ("v3_python", [str(py), "-B", "tests/parity/htb_onpolicy_mirror/python_side.py", str(scratch / "v3_python.json"),
                       challenger["path"], challenger["sha256"], challenger["state_digest"]], tf32),
        ("v3_node", ["node", "tests/parity/htb_onpolicy_mirror/node_side.mjs", str(scratch / "v3_node.json"),
                     str(pair_dir)], {}),
        ("v3_compare", ["node", "tests/parity/htb_mirror/compare.mjs", str(scratch / "v3_python.json"),
                        str(scratch / "v3_node.json"), str(ev / "v3_result.json")], {}),
    ]


def export_command(py: pathlib.Path, weights: str, scratch: pathlib.Path) -> List[str]:
    return [str(py), "-B", "-m", "scripts.GPU.alphazero.export_onnx", "--weights", weights,
            "--output", str(scratch / "model.onnx")]


def _run_real(name: str, argv: Sequence[str], env: Mapping[str, str], cwd: pathlib.Path, log: pathlib.Path) -> int:
    with open(log, "x") as fh:     # pragma: no cover - runs the real tools; never under the build approval
        return subprocess.run(list(argv), cwd=cwd, env={**os.environ, **env}, stdout=fh, stderr=subprocess.STDOUT).returncode


def runtime_versions(root: pathlib.Path) -> str:   # pragma: no cover - reads the live environment
    from importlib import metadata

    def ver(d):
        try:
            return metadata.version(d)
        except metadata.PackageNotFoundError:
            return "absent"
    node = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
    ort = json.loads((root / "node_modules/onnxruntime-node/package.json").read_text())["version"]
    return (f"python {sys.version.split()[0]}; mlx {ver('mlx')}; mlx-metal {ver('mlx-metal')}; torch {ver('torch')}; "
            f"onnx {ver('onnx')}; numpy {ver('numpy')}; node {node}; onnxruntime-node {ort}")


def fit_admission(cfg: H.Config) -> Dict[str, Any]:
    """Only an ELIGIBLE fit is exported (card §14), from its committed record and clean wrapper, and only the very
    challenger file it saved."""
    adm = H.stage_admission(cfg, "fit")
    rec = H.read_committed_json(cfg, f"{H.EV['fit']}/record.json")
    if rec.get("verdict") != "ELIGIBLE":
        raise Refused(f"the fit's verdict is {rec.get('verdict')!r}: only an ELIGIBLE challenger is exported (card §14)")
    ch = rec.get("challenger") or {}
    if not ch.get("path") or not pathlib.Path(ch["path"]).is_file():
        raise Refused("the fit record names no challenger file that exists")
    raw = pathlib.Path(ch["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ch.get("sha256"):
        raise Refused("the challenger file is not the one the fit saved")
    return {**adm, "challenger": {"path": ch["path"], "sha256": ch["sha256"], "state_digest": ch.get("state_digest"),
                                  "sha1": hashlib.sha1(raw).hexdigest()}}


def run_export(cfg: Optional[H.Config] = None, runner: Callable[..., int] = _run_real,
               py: Optional[pathlib.Path] = None, export_root: pathlib.Path = EXPORT_ROOT,
               scratch: pathlib.Path = SCRATCH, versions: Callable[[pathlib.Path], str] = runtime_versions) -> int:
    cfg = cfg or H.Config()
    H.preconditions("export", cfg)
    ev = cfg.ev("export")
    P._absent(ev, export_root, scratch)
    adm = fit_admission(cfg)
    ch = adm["challenger"]
    py = py or (P.main_checkout(cfg.root) / ".venv/bin/python")
    rec = H._record("export", cfg, admission=adm)
    rec["kind"] = RECORD_KIND
    rec["runtime_versions"] = versions(cfg.root)

    def body(rec):
        scratch.mkdir(parents=True)
        if runner("export", export_command(py, ch["path"], scratch), {"MLX_ENABLE_TF32": "0", "PYTHONPATH": str(cfg.root)},
                  cfg.root, ev / "export.log") != 0:
            raise Stop("export", "the exporter failed; see export.log")
        if hashlib.sha256(pathlib.Path(ch["path"]).read_bytes()).hexdigest() != ch["sha256"]:
            raise Stop("export", "the challenger file changed during the export")
        g, d = _file(scratch / "model.onnx"), _file(scratch / "model.onnx.data")
        mid = model_id(g["sha256"], d["sha256"])
        if mid == CURRENT_ID:
            raise Stop("export", "the challenger's pair is the current pair")
        sig_new, sig_cur = graph_signature(scratch / "model.onnx"), graph_signature(cfg.root / CURRENT_DIR / "model.onnx")
        rec["graph_signature_sha256"] = {"challenger": P._jsha(sig_new), "current": P._jsha(sig_cur)}
        if sig_new != sig_cur:
            raise Stop("graph", "the challenger's graph ops, inputs, outputs or initializer shapes differ from the current pair's")
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
        for name, argv, env in commands(cfg.root, py, ch, pair_dir, scratch, ev):
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="htb_onpolicy_export")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").add_argument("what", choices=("export",))
    ap.parse_args(argv)
    try:
        return run_export()
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return P.EXIT["REFUSED"]


if __name__ == "__main__":
    sys.exit(main())
