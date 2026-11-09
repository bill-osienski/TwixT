"""Compare an ort_run output against the recorded V2 MLX (python_side) and Node (node_side) outputs."""
import json, sys, numpy as np
X = sys.argv[1]; out = json.load(open(sys.argv[2])); key = sys.argv[3] if len(sys.argv) > 3 else None
py = {p["id"]: p for p in json.load(open(f"{X}/v2/python_side.json"))["positions"]}
nd = {p["id"]: p for p in json.load(open(f"{X}/v2/node_side.json"))["positions"]}
res = out["results"][key] if key else out["results"]
def nl(p): return p["ort_node"] if "ort_node" in p else p
for pid, r in res.items():
    lg = np.array(r["logits"]); m = py[pid]["mlx"]; n = nl(nd[pid])
    print(f"  {pid:18s} vs MLX logit {np.abs(lg-np.array(m['logits'])).max():.3e} value {abs(r['value']-m['value']):.3e} | vs Node logit {np.abs(lg-np.array(n['logits'])).max():.3e} value {abs(r['value']-n['value']):.3e} | vs V2 ort_py logit {np.abs(lg-np.array(py[pid]['ort_py']['logits'])).max():.3e}")
