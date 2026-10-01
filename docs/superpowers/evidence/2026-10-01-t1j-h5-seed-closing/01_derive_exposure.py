"""H5 CLOSING SEED ACCOUNTING -- exposure DERIVED from the committed records.

H5 implementation card §7: use-recording of the four H5 blocks was deferred to ONE
registry edit after H5 ended (as H4, step-4 card §2.1). Exposure = a seed whose game
STARTED (a `game_start` record exists), H3's and H4's convention. Read-only: of each
record it uses only the type, a game_start's seed and the final segment_end's
completion fields -- no outcome field is touched. Every input must be committed and
unmodified at HEAD.
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.GPU.alphazero import h4_pilot_feasibility as F                # noqa: E402
from scripts.GPU.alphazero import h5_runner as H5                          # noqa: E402

MANIFEST = "docs/superpowers/evidence/t1j-h5-manifests/study_manifest.json"
RUNS = {k: f"{H5.EVIDENCE_DIRS[k]}/results.jsonl" for k in H5.SEGMENTS}
F.check_durable([str(ROOT / p) for p in (MANIFEST, *RUNS.values())], _fixture=False)
head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
print(f"HEAD {head}; manifest + {len(RUNS)} results files committed and unmodified")
manifest = json.loads((ROOT / MANIFEST).read_text())
ok, total = True, 0
for k, path in RUNS.items():
    lo, hi = H5.SEED_BLOCKS[k]
    assigned = [t["seed"] for t in manifest["segments"][k]["schedule"]]
    types, started, last = [], [], None
    for line in (ROOT / path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            types.append(r["record_type"])
            if r["record_type"] == "game_start":
                started.append(r["seed"])
            last = {f: r.get(f) for f in ("record_type", "complete", "games_completed")}
    complete = last == {"record_type": "segment_end", "complete": True,
                        "games_completed": len(assigned)}
    exact = (assigned == list(range(lo, hi)) and sorted(started) == assigned
             and len(started) == len(set(started)))
    voids = types.count("run_void")
    ok &= exact and complete and voids == 0
    total += len(started)
    print(f"segment {k}: block [{lo}, {hi}) -- manifest assigns {len(assigned)}, "
          f"{len(started)} game_starts, every assigned seed started EXACTLY once: {exact}; "
          f"run_void {voids}; ended in a complete segment_end: {complete}")
print(f"total seeds started: {total} of {sum(hi - lo for lo, hi in H5.SEED_BLOCKS.values())}")
print("VERDICT:", "every assigned seed started exactly once and all four runs completed -> "
      "the four blocks are EXPOSED WHOLE and RETIRED WHOLE" if ok else
      "DOES NOT RECONCILE -- no registry edit")
sys.exit(0 if ok else 1)
