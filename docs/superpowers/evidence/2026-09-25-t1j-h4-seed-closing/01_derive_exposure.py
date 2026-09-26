"""H4 CLOSING SEED ACCOUNTING -- exposure DERIVED from the committed records.

Step-4 card §2.1: use of the five H4 blocks was deferred to ONE registry edit
after H4 closed, derived from the preserved records. Exposure = a seed whose game
STARTED (a `game_start` record exists), H3's convention. Read-only: it reads only
`game_start` records -- no outcome field is touched.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.GPU.alphazero import h4_runner as R                     # noqa: E402

RUNS = {("pilot", 0): "docs/superpowers/evidence/t1j-h4-pilot/results.jsonl",
        **{("study", k): f"docs/superpowers/evidence/t1j-h4-study-segment{k}/results.jsonl"
           for k in range(4)}}
ok, total = True, 0
for key, path in RUNS.items():
    recs = [json.loads(l) for l in (ROOT / path).read_text().splitlines() if l.strip()]
    started = [r["seed"] for r in recs if r["record_type"] == "game_start"]
    lo, hi = R.SEED_BLOCKS[key]
    ended = recs[-1]["record_type"]
    exact = sorted(started) == list(range(lo, hi)) and len(started) == len(set(started))
    ok &= exact and ended == "segment_end"
    total += len(started)
    print(f"{key[0]} {key[1]}: block [{lo}, {hi}) -- {len(started)} game_starts, every "
          f"assigned seed started EXACTLY once: {exact}; run ended in {ended}")
print(f"total seeds started: {total} of {sum(hi - lo for lo, hi in R.SEED_BLOCKS.values())}")
print("VERDICT:", "every assigned seed started exactly once -> the five blocks are "
      "EXPOSED WHOLE and RETIRED WHOLE" if ok else "NOT UNIFORM -- no edit")
sys.exit(0 if ok else 1)
