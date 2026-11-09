"""Diagnostic: run the instrumented graph in Python ORT (numpy+ORT only) and compare every intermediate with Node's."""
import json, re, sys, numpy as np, onnxruntime as ort
D, pid, first = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
if first:
    ort.InferenceSession(first, providers=["CPUExecutionProvider"])
s = ort.InferenceSession(f"{D}/model_instr.onnx", providers=["CPUExecutionProvider"])
feed = {"board": np.fromfile(f"{D}/{pid}_board.bin", "<f4").reshape(1, 30, 24, 24), "move_rows": np.fromfile(f"{D}/{pid}_rows.bin", "<i8"),
        "move_cols": np.fromfile(f"{D}/{pid}_cols.bin", "<i8"), "move_mask": np.fromfile(f"{D}/{pid}_mask.bin", "<f4")}
names = [o.name for o in s.get_outputs()]
outs = dict(zip(names, s.run(names, feed)))
meta = json.load(open(f"{D}/node_out/meta.json"))
order = [o for o, _, _ in json.load(open(f"{D}/order.json"))] + ["policy_logits", "value"]
ops = {o: (t, n) for o, t, n in json.load(open(f"{D}/order.json"))}
first_bad = None
for name in order:
    a = np.asarray(outs[name], dtype=np.float32).reshape(-1)
    b = np.fromfile(f"{D}/node_out/{meta[name]['file']}.bin", "<f4")
    d = float(np.max(np.abs(a - b))) if a.size else 0.0
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    flag = d > 1e-3 * max(1.0, scale)
    if flag and first_bad is None:
        first_bad = name
    print(f"{'**' if flag else '  '} {name:40s} {ops.get(name, ('OUT',''))[0]:14s} n={a.size:7d} maxdiff={d:.3e} max|node|={scale:.3e}")
print("FIRST DIVERGENT:", first_bad, ops.get(first_bad))
