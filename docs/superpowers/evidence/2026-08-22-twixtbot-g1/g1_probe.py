"""G1 — install/import/model-load smoke for the twixtbot anchor pilot.
Frozen scope: import backend.nneval, load the pinned SavedModel, run ONE
fixed-position eval_one, record versions/shapes/finiteness. Nothing else.
The engine is imported, never modified."""
import os, sys, json, platform

CLONE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/twixtbot-ui"
os.chdir(CLONE)          # NNEvaluater does os.path.join(os.getcwd(), model)
sys.path.insert(0, CLONE)

out = {"clone": CLONE}
out["python"] = sys.version.split()[0]
out["platform"] = f"{platform.system()} {platform.machine()}"

import numpy as np
out["numpy"] = np.__version__

# The card's "backend.nneval" is src.backend.nneval in this package layout
# (tests use `from src.backend.twixt import Game`).
from src.backend import nneval, naf, twixt          # noqa: E402
from src.backend.point import Point                 # noqa: E402
import tensorflow as tf                             # noqa: E402
out["tensorflow"] = tf.__version__

# ONE fixed position: four central moves on a 24x24 board, own-link crossing off.
MOVES = [(12, 12), (10, 11), (13, 10), (11, 13)]
game = twixt.Game(allow_scl=False)
for x, y in MOVES:
    game.play(Point(x, y))
out["position"] = {"moves": MOVES, "board_size": twixt.Game.SIZE,
                   "allow_scl": game.allow_scl, "turn": int(game.turn)}

ev = nneval.NNEvaluater("model/pb")
out["use_recents"] = bool(ev.use_recents)

pwin, movelogits = ev.eval_one(naf.NetInputs(game))
for name, arr in (("pwin", pwin), ("movelogits", movelogits)):
    a = np.asarray(arr)
    out[name] = {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "all_finite": bool(np.all(np.isfinite(a))),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }
out["pwin_value"] = np.asarray(pwin).ravel().tolist()[:4]

print(json.dumps(out, indent=2))
