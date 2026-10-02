"""H5 REPLAY EXPORT -- derived viewer files for all 592 H5 games. Run ONCE.

H5 replication card step 15 (optional export); H4 runner card §8's export, by
IMPORT: `h4_runner.load_games` re-derives every game (transcript digests, position
digests from the moves, per-game record checks) and `h4_runner.export_game` builds
each record, refusing rather than guessing. No H4 file is edited.

The ONE difference from H4's output: `export_game` hard-codes three H4 provenance
strings, which would label every H5 file "H4" on the viewer's screen. Exactly those
three are replaced (`relabel`, which refuses if they are not H4's); the verifier
(02) proves nothing else differs.

Each segment's files go to a create-only `replays/` subdirectory of its evidence
directory; the canonical results.jsonl files are read, never written.
"""
import copy
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.GPU.alphazero import h4_pilot_feasibility as F                # noqa: E402
from scripts.GPU.alphazero import h4_runner as R4                          # noqa: E402
from scripts.GPU.alphazero import h5_runner as H5                          # noqa: E402

H4_LABELS = {"config_hash": "h4-recorded-evidence", "mode": "recorded_h4",
             "evidence_note": "derived from the H4 results file; a viewer export, not a result"}
H5_LABELS = {"config_hash": "h5-recorded-evidence", "mode": "recorded_h5",
             "evidence_note": "derived from the H5 results file; a viewer export, not a result"}


def relabel(rec):
    got = {"config_hash": rec["config_hash"], "mode": rec["meta"]["mode"],
           "evidence_note": rec["meta"]["evidence_note"]}
    if got != H4_LABELS:
        raise SystemExit(f"{rec['id']}: the exporter's provenance is not H4's: {got}")
    out = copy.deepcopy(rec)
    out["config_hash"] = H5_LABELS["config_hash"]
    out["meta"]["mode"] = H5_LABELS["mode"]
    out["meta"]["evidence_note"] = H5_LABELS["evidence_note"]
    return out


def main():
    runs = {k: ROOT / H5.EVIDENCE_DIRS[k] / "results.jsonl" for k in H5.SEGMENTS}
    F.check_durable([str(p) for p in runs.values()], _fixture=False)
    total = 0
    for k, results in runs.items():
        export_dir = results.parent / "replays"
        games = R4.load_games(str(results))
        os.makedirs(export_dir)                   # create-only: an existing dir refuses
        for game in games:
            rec = relabel(R4.export_game(game, str(results)))
            fd = os.open(export_dir / f"{game['task_id']}.json",
                         os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=1, sort_keys=True)
        total += len(games)
        print(f"segment {k}: {len(games)} games re-derived by load_games, {len(games)} files "
              f"-> {export_dir.relative_to(ROOT)}")
    print(f"total: {total} files")
    return 0 if total == 592 else 1


if __name__ == "__main__":
    sys.exit(main())
