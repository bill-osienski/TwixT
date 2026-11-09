"""Diagnostic: Python ONNX Runtime on saved inputs. Imports numpy + onnxruntime only, unless --pre-import asks.
usage: ort_run.py <inputs_dir> <out.json> --model PATH [--pre-import mlx|torch|mlx,torch] [--threads N]
       [--first-model PATH] [--after-import mlx|torch]"""
import argparse, hashlib, json, sys
p = argparse.ArgumentParser()
p.add_argument("inputs"); p.add_argument("out"); p.add_argument("--model", required=True)
p.add_argument("--pre-import", default=""); p.add_argument("--after-import", default="")
p.add_argument("--threads", type=int, default=0); p.add_argument("--config", default=""); p.add_argument("--no-arena", action="store_true"); p.add_argument("--no-mempattern", action="store_true"); p.add_argument("--opt", default=""); p.add_argument("--first-model", default="")
a = p.parse_args()
for m in filter(None, a.pre_import.split(",")):
    __import__({"mlx": "mlx.core", "torch": "torch"}[m])
import numpy as np
import onnxruntime as ort
pos = json.load(open(f"{a.inputs}/positions.json"))
def opts():
    so = ort.SessionOptions()
    if a.threads:
        so.intra_op_num_threads = a.threads; so.inter_op_num_threads = a.threads
    for kv in filter(None, a.config.split(",")):
        k, v = kv.split("="); so.add_session_config_entry(k, v)
    if a.no_arena: so.enable_cpu_mem_arena = False
    if a.no_mempattern: so.enable_mem_pattern = False
    if a.opt: so.graph_optimization_level = {"none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC}[a.opt]
    return so
def run(sess):
    out = {}
    for pid, meta in pos.items():
        f = {k: np.load(f"{a.inputs}/{pid}_{k}.npy") for k in ("board", "rows", "cols", "mask")}
        assert hashlib.sha256(f["board"].tobytes()).hexdigest() == meta["board_sha256"]
        lg, v = sess.run(["policy_logits", "value"], {"board": f["board"], "move_rows": f["rows"], "move_cols": f["cols"], "move_mask": f["mask"]})
        out[pid] = {"logits": lg.reshape(-1)[: meta["n_legal"]].astype(float).tolist(), "value": float(np.array(v).reshape(-1)[0])}
    return out
if a.first_model:
    run(ort.InferenceSession(a.first_model, opts(), providers=["CPUExecutionProvider"]))
res = run(ort.InferenceSession(a.model, opts(), providers=["CPUExecutionProvider"]))
for m in filter(None, a.after_import.split(",")):
    __import__({"mlx": "mlx.core", "torch": "torch"}[m])
    res = {"before_import": res, "after_import": run(ort.InferenceSession(a.model, opts(), providers=["CPUExecutionProvider"]))}
loaded = {k: (k in sys.modules) for k in ("mlx", "mlx.core", "torch")}
json.dump({"config": vars(a), "modules_loaded_at_end": loaded, "ort_version": ort.__version__, "numpy": np.__version__, "results": res}, open(a.out, "w"))
print("wrote", a.out, "loaded:", loaded)
