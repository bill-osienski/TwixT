"""One fresh process per case: a graph of N parallel Convs on one input; each output judged against float64 numpy."""
import sys, numpy as np, onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto
D, case = sys.argv[1], sys.argv[2]
w = np.load(f"{D}/w.npz"); rng = np.random.default_rng(0)
x = rng.standard_normal((1, 128, 24, 24)).astype(np.float32)
def rnd(sparse, where=None):
    W = (rng.standard_normal((128, 128, 3, 3)) * 0.1).astype(np.float32)
    if sparse: W[:73] = 0
    if where is not None: W[where] = 0
    return W, (rng.standard_normal(128) * 0.1).astype(np.float32)
cases = {
    "student_W0_W1": [(w["W0"], w["B0"]), (w["W1"], w["B1"])],
    "student_W1_alone": [(w["W1"], w["B1"])],
    "student_W1_W0": [(w["W1"], w["B1"]), (w["W0"], w["B0"])],
    "student_W0_W1_W2": [(w["W0"], w["B0"]), (w["W1"], w["B1"]), (w["W2"], w["B2"])],
    "random_dense_x2": [rnd(False), rnd(False)],
    "random_73zero_filters_x2": [rnd(True), rnd(True)],
    "random_last73_zero_x2": [rnd(False, slice(55, 128)), rnd(False, slice(55, 128))],
    "random_first1_zero_x2": [rnd(False, slice(0, 1)), rnd(False, slice(0, 1))],
    "random_first8_zero_x2": [rnd(False, slice(0, 8)), rnd(False, slice(0, 8))],
    "random_scattered40_zero_x2": [rnd(False, np.arange(0, 128, 3)), rnd(False, np.arange(1, 128, 3))],
    "random_half_elements_zero_x2": [(lambda W, B: (np.where(rng.random(W.shape) < 0.6, 0, W).astype(np.float32), B))(*rnd(False)), (lambda W, B: (np.where(rng.random(W.shape) < 0.6, 0, W).astype(np.float32), B))(*rnd(False))],
}
params = cases[case]
nodes, inits, outs = [], [], []
for k, (W, B) in enumerate(params):
    nodes.append(helper.make_node("Conv", ["x", f"W{k}", f"B{k}"], [f"y{k}"], pads=[1, 1, 1, 1]))
    inits += [numpy_helper.from_array(W, f"W{k}"), numpy_helper.from_array(B, f"B{k}")]
    outs.append(helper.make_tensor_value_info(f"y{k}", TensorProto.FLOAT, [1, 128, 24, 24]))
g = helper.make_graph(nodes, "mini", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 128, 24, 24])], outs, inits)
mm = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
import os
if os.environ.get("SAVE_ONNX"):
    import onnx; onnx.save(mm, f"{D}/{case}.onnx"); np.save(f"{D}/{case}_x.npy", x)
ys = ort.InferenceSession(mm.SerializeToString(), providers=["CPUExecutionProvider"]).run([f"y{k}" for k in range(len(params))], {"x": x})
xp = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (1, 1), (1, 1)))
def ref(W, B):
    r = np.zeros((128, 24, 24))
    for i in range(3):
        for j in range(3):
            r += np.einsum("oc,chw->ohw", W[:, :, i, j].astype(np.float64), xp[0, :, i:i + 24, j:j + 24])
    return r + B.astype(np.float64)[:, None, None]
refs = [ref(W, B) for W, B in params]
for k, y in enumerate(ys):
    errs = [float(np.abs(y[0] - r).max()) for r in refs]
    print(f"  {case:26s} conv{k}: err vs own params {errs[k]:.2e} | best match = params of conv{int(np.argmin(errs))} ({min(errs):.2e})")
