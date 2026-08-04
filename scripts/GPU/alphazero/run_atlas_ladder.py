"""Atlas ladder CLI -- design sections 2b and 4.

THIS TOOL RUNS NO MEASUREMENT. It emits the replay plan and a runtime
projection; executing the ladder against a real checkpoint is a separate
operator authorization.
"""
from __future__ import annotations

import argparse
import json

from .build_atlas_corpus import _jsonable
from .warm_prefix_replay import LEG_INCREMENTS, NOMINAL_B, project_runtime

_STOP = ("=" * 72 + "\nOPERATOR STOP -- the atlas measurement run is NOT "
         "AUTHORIZED by this tool.\n" + "=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description="Atlas ladder (runs no measurement)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("emit-plan")
    s.add_argument("--corpus-artifact", required=True)

    s = sub.add_parser("project-runtime")
    s.add_argument("--rows", type=int, required=True)
    s.add_argument("--mean-prefix-plies", type=float, required=True)

    args = ap.parse_args()
    if args.cmd == "project-runtime":
        print(json.dumps(_jsonable(
            project_runtime(args.rows, args.mean_prefix_plies)),
            indent=2, sort_keys=True))
        return 0

    print(_STOP)
    print(json.dumps(_jsonable({
        "corpus_artifact": args.corpus_artifact,
        "leg_increments": list(LEG_INCREMENTS),
        "nominal_B": list(NOMINAL_B),
        "boundary": "first flush completion at or after 320 target-search backups",
        "add_noise": False,
        "note": "One random.Random(base_seed + game_idx) per row, continued "
                "across the prefix and all four legs. Never reseeded.",
    }), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
