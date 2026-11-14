"""Writes tiny_add.onnx: y = x + 1 on a float32 [1, 4] tensor. A synthetic model for the ONNX Runtime shutdown probe.

Run once with the repo .venv (onnx 1.20.1): .venv/bin/python tests/fixtures/onnx_shutdown/make_tiny_model.py
"""
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

out = pathlib.Path(__file__).with_name("tiny_add.onnx")
one = numpy_helper.from_array(np.ones((1, 4), dtype=np.float32), name="one")
graph = helper.make_graph(
    [helper.make_node("Add", ["x", "one"], ["y"])],
    "tiny_add",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
    initializer=[one],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="twixt-shutdown-probe")
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, out)
print(out)
