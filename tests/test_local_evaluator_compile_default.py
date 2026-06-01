"""Training-safety guardrail for the eval-only `compile` parameter.

`local_evaluator.py` is shared with the training path (trainer.py constructs
`LocalGPUEvaluator(network)` with no `compile` arg). The `compile` param was
added for the eval/Metal-resource fix and MUST default to False so training
behavior stays byte-identical; only eval opts in via `compile=True`.
"""
import inspect

from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator


def test_compile_param_defaults_to_false_signature():
    default = inspect.signature(
        LocalGPUEvaluator.__init__
    ).parameters["compile"].default
    assert default is False


def test_default_instance_does_not_use_compile():
    # __init__ only stores the network reference (never uses it), so passing
    # None is safe and keeps this guardrail GPU-free.
    ev = LocalGPUEvaluator(None)
    assert ev._use_compile is False


def test_compile_true_is_explicit_opt_in():
    ev = LocalGPUEvaluator(None, compile=True)
    assert ev._use_compile is True
