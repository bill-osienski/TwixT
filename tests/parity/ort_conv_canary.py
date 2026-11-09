#!/usr/bin/env python3
"""ONNX Runtime correctness canary: same-shape Convs with an all-zero filter 0 (ledger entries 55-56).

    <python> tests/parity/ort_conv_canary.py [out.json]

Python onnxruntime 1.24.4 was observed to compute such Convs with ONE parameter set, both within a session and across
sessions in a process (docs/superpowers/evidence/htb-integration-g2/diagnostic-1/). This canary qualifies a Python
ONNX Runtime before it may serve as a parity surface for the H-TB student.

Self-contained: synthetic weights from an integer generator (shared with server/test_ort_conv_canary.js), no model,
no student weights, no scratch files. Needs numpy, onnx and onnxruntime only.

Correctness, not difference: every Conv output is compared with ITS OWN float64 NumPy reference.

Cases:
- control_dense: two dense Convs in one session. This must pass on any working runtime, which shows the harness itself
  is sound.
- within_session: three Convs, each with an all-zero filter 0, in one session.
- cross_session: two single-Conv sessions with an all-zero filter 0, created one after the other in this process; the
  second runs first, then the first again.

The references are checked pairwise distinct (each pair differs by more than 1.0), so a conflated runtime cannot pass.
Exit 0 = PASS, 1 = FAIL.
"""
import json
import platform
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

C, H, K = 128, 24, 3
TOL = 1e-3          # observed correct-runtime error ~2e-5; a conflated output is off by ~10-20
DISTINCT = 1.0


def gen(seed, n):
    """Deterministic values in [-0.1, 0.1]; the JS twin is server/test_ort_conv_canary.js:gen."""
    i = np.arange(n, dtype=np.uint64)
    u = ((i ^ np.uint64(seed)) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)
    return ((u % np.uint64(20001)).astype(np.int64) - 10000).astype(np.float64) / 100000


def conv_params(seed, zero_filter0):
    W = gen(seed, C * C * K * K).astype(np.float32).reshape(C, C, K, K)
    if zero_filter0:
        W[0] = 0
    return W, gen(seed + 1000, C).astype(np.float32)


X = (gen(7, C * H * H) * 10).astype(np.float32).reshape(1, C, H, H)


def reference(W, B):
    xp = np.pad(X.astype(np.float64), ((0, 0), (0, 0), (1, 1), (1, 1)))
    r = np.zeros((C, H, H))
    for i in range(K):
        for j in range(K):
            r += np.einsum("oc,chw->ohw", W[:, :, i, j].astype(np.float64), xp[0, :, i:i + H, j:j + H])
    return r + B.astype(np.float64)[:, None, None]


def session(params):
    nodes, inits, outs = [], [], []
    for k, (W, B) in enumerate(params):
        nodes.append(helper.make_node("Conv", ["x", f"W{k}", f"B{k}"], [f"y{k}"], pads=[1, 1, 1, 1]))
        inits += [numpy_helper.from_array(W, f"W{k}"), numpy_helper.from_array(B, f"B{k}")]
        outs.append(helper.make_tensor_value_info(f"y{k}", TensorProto.FLOAT, [1, C, H, H]))
    g = helper.make_graph(nodes, "canary", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, H, H])],
                          outs, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    return lambda: [y[0] for y in s.run([f"y{k}" for k in range(len(params))], {"x": X})]


def check(name, params, outputs, results):
    refs = [reference(W, B) for W, B in params]
    for a in range(len(refs)):
        for b in range(a + 1, len(refs)):
            assert np.abs(refs[a] - refs[b]).max() > DISTINCT, f"{name}: references {a},{b} are not distinct"
    for k, (y, r) in enumerate(zip(outputs, refs)):
        err = float(np.abs(y - r).max())
        results.append({"case": name, "conv": k, "max_abs_err": err, "ok": err <= TOL})


def main():
    results = []
    dense = [conv_params(11, False), conv_params(12, False)]
    check("control_dense", dense, session(dense)(), results)
    within = [conv_params(21, True), conv_params(22, True), conv_params(23, True)]
    check("within_session", within, session(within)(), results)
    a, b = conv_params(31, True), conv_params(32, True)
    run_a = session([a])
    first_a = run_a()[0]
    run_b = session([b])
    y_b = run_b()[0]
    again_a = run_a()[0]
    check("cross_session(first session, before the second exists)", [a], [first_a], results)
    check("cross_session(second session)", [b, a], [y_b], results)
    check("cross_session(first session, after the second)", [a, b], [again_a], results)
    verdict = "PASS" if all(r["ok"] for r in results) else "FAIL"
    report = {"verdict": verdict, "tolerance": TOL, "onnxruntime": ort.__version__, "onnx": onnx.__version__,
              "numpy": np.__version__, "python": platform.python_version(), "executable": sys.executable,
              "results": results}
    for r in results:
        print(f"  {'ok  ' if r['ok'] else 'FAIL'} {r['case']:55s} conv{r['conv']}  max abs err {r['max_abs_err']:.2e}")
    print(f"CANARY {verdict}  (onnxruntime {ort.__version__}, python {platform.python_version()})")
    if len(sys.argv) > 1:
        json.dump(report, open(sys.argv[1], "w"), indent=1)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
