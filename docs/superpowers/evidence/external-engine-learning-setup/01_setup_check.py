"""Bounded setup check (scratch). NO queries, NO games, NO f check, gate untouched.
Through the run's OWN loaders: EL.load_runtime (G3's pinned-clone verifier, TF CPU-only, module-origin import),
the teacher model loaded (constructed, never evaluated), the incumbent loaded and its SHA-1 checked, and a
trivial MLX op on the Metal device AFTER TensorFlow -- the run's import order."""
import os, sys, json, time, platform
ROOT = "/Users/bill/projects/TwixT_Game"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from scripts.GPU.alphazero import external_learning as EL
from scripts.GPU.alphazero import external_learning_authorization as GATE

clone = sys.argv[1]
out = {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
       "gate_closed": GATE.EXTERNAL_LEARNING_STAGE1_AUTHORIZED is False}
assert out["gate_closed"]
out["permission"] = EL.license_notices(clone)
t = time.monotonic()
rt, identity = EL.load_runtime(clone)
out["runtime"] = identity
out["cwd_after_import"] = os.getcwd()
out["module_files"] = {k: getattr(v, "__file__", None) for k, v in rt.items() if hasattr(v, "__file__")}
out["teacher_kwargs"] = {k: str(v) for k, v in EL.teacher_kwargs(rt["ct"]).items()}
import tensorflow as tf
out["tf_visible_gpus"] = [d.name for d in tf.config.get_visible_devices("GPU")]
ev_t = rt["nneval"].NNEvaluater("model/pb")                      # model LOADED; never evaluated
out["teacher_model_loaded"] = type(ev_t).__name__
import mlx.core as mx
out["mlx"] = {"version": mx.__version__, "default_device": str(mx.default_device()),
              "metal_available": bool(mx.metal.is_available())}
a = mx.arange(8, dtype=mx.float32)
mx.eval(a.sum())
out["mlx_trivial_op"] = float(a.sum().item())
from scripts.GPU.alphazero import twixtbot_g3_reference as RF
ev = RF.load_reference_evaluator(EL.REFERENCE, ROOT)
out["incumbent"] = {"reference": EL.REFERENCE, "sha1": getattr(ev, "_g3_sha1", None),
                    "sha1_matches_pin": getattr(ev, "_g3_sha1", None) == EL.REFERENCE_SHA1,
                    "policy_head_params_present": hasattr(ev, "network")}
out["seconds"] = round(time.monotonic() - t, 2)
print(json.dumps(out, indent=1, sort_keys=True))
