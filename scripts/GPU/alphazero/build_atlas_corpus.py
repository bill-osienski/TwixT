"""Atlas corpus builder CLI -- design section 3.

THIS TOOL GENERATES NOTHING. It prints generation commands and stops; running
them is a separate operator authorization, exactly as Phase 0's preflight was.

Staged chronology (N does NOT exist before the pilot ladder):
    emit-protocol -> emit-pilot-command -> [generate pilot] -> pilot-gate
    -> [pilot ladder, Stage 3] -> N -> size -> emit-continuation-command
    -> [generate continuation] -> assign

Exit codes: 0 OK, 2 usage/validation, 3 PHASE_GEOMETRY_NO_GO, 4 ASSIGNMENT_SHORTFALL.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from .corpus_geometry import (
    ALLOWED_N,
    MAX_SEED_RANGE_GAMES,
    PILOT_GAMES,
    assign_corpus,
    pilot_geometry_gate,
    size_continuation,
)
from .generate_atlas_reservoir import (
    _sha1_file,
    assert_blocks_agree,
    load_block,
    load_manifest,
)

_STOP = ("=" * 72 + "\nOPERATOR STOP -- reservoir generation is NOT AUTHORIZED by "
         "this tool.\nObtain authorization, then run the command below.\n" + "=" * 72)


def _jsonable(obj):
    """Make tuple-keyed dicts serializable.

    The geometry module returns cells as (split, phase, side) TUPLES, which is
    the natural type there. json.dumps cannot use tuples as keys, and `default=`
    only rescues unserializable VALUES -- not keys. Converting here keeps the
    pure module free of a JSON concern.
    """
    # Dataclasses are the same boundary problem as tuple keys: the atlas rows
    # hold LegResult / BoundaryRecord objects because the read-outs address them
    # by ATTRIBUTE, and only the JSON boundary needs them flattened. Additive --
    # every pre-existing caller's payload contains no dataclasses, so their
    # output is byte-identical.
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else k):
                _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return ["|".join(map(str, x)) if isinstance(x, tuple) else _jsonable(x)
                for x in obj]
    return obj


def _emit(payload) -> None:
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True, default=str))


def _protocol(args) -> dict:
    # The checkpoint is frozen HERE, pre-pilot, because every block must use the
    # same one. Freezing the seed range but not the checkpoint would let a
    # continuation come from a different network under the same seeds.
    sha1 = _sha1_file(args.checkpoint)
    if not sha1:
        raise ValueError(f"checkpoint not found or unreadable: {args.checkpoint}")
    return {
        "base_seed": args.base_seed,
        "checkpoint_path": args.checkpoint,
        "checkpoint_sha1": sha1,
        "seed_range": [args.base_seed, args.base_seed + MAX_SEED_RANGE_GAMES],
        "max_seed_range_games": MAX_SEED_RANGE_GAMES,
        "pilot_games": PILOT_GAMES,
        "sampling_seed": args.sampling_seed,
        "one_position_per_game": True,
        "no_top_up": True,
        "selection_inputs": ["game_id", "phase", "side", "sampling_seed"],
        "note": "N is NOT fixed here -- it is derived from the pilot ladder's "
                "measured class frequencies, and G_total from N. Selection reads "
                "no search result. Generation is separately authorized.",
    }


def _gen_cmd(base_seed, start, n, checkpoint, out_dir) -> str:
    return (f"#   .venv/bin/python -m scripts.GPU.alphazero.generate_atlas_reservoir \\\n"
            f"#     --base-seed {base_seed} --start-index {start} --n-games {n} \\\n"
            f"#     --checkpoint {checkpoint} --out-dir {out_dir} \\\n"
            f"#     --simulations 400 --max-moves 280")


def main() -> int:
    ap = argparse.ArgumentParser(description="Atlas corpus builder (generates nothing)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("emit-protocol")
    s.add_argument("--base-seed", type=int, required=True)
    s.add_argument("--sampling-seed", type=int, required=True)
    s.add_argument("--checkpoint", required=True)

    s = sub.add_parser("emit-pilot-command")
    s.add_argument("--base-seed", type=int, required=True)
    s.add_argument("--sampling-seed", type=int, required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--out-dir", required=True)

    s = sub.add_parser("emit-continuation-command")
    s.add_argument("--base-seed", type=int, required=True)
    s.add_argument("--pilot-dir", required=True)      # to RECOMPUTE the sizing
    s.add_argument("--n-target", type=int, required=True)
    s.add_argument("--size-artifact", required=True)  # compared, never trusted
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--out-dir", required=True)

    s = sub.add_parser("pilot-gate")
    s.add_argument("--sidecar-dir", required=True)
    s.add_argument("--base-seed", type=int, required=True)
    s.add_argument("--sampling-seed", type=int, required=True)

    s = sub.add_parser("size")
    s.add_argument("--sidecar-dir", required=True)
    s.add_argument("--base-seed", type=int, required=True)
    s.add_argument("--n-target", type=int, required=True)

    s = sub.add_parser("assign")
    s.add_argument("--pilot-dir", required=True)
    s.add_argument("--continuation-dir", required=True)
    s.add_argument("--base-seed", type=int, required=True)
    s.add_argument("--n-target", type=int, required=True)
    s.add_argument("--sampling-seed", type=int, required=True)

    args = ap.parse_args()
    if getattr(args, "n_target", None) is not None and args.n_target not in ALLOWED_N:
        print(f"error: --n-target must be one of {ALLOWED_N}", file=sys.stderr)
        return 2

    try:
        if args.cmd == "emit-protocol":
            print(json.dumps(_protocol(args), indent=2, sort_keys=True))
            return 0

        if args.cmd == "emit-pilot-command":
            print(_STOP)
            print(json.dumps(_protocol(args), indent=2, sort_keys=True))
            print("\n# pilot block [0, 24) -- its OWN directory (NOT run here):")
            print(_gen_cmd(args.base_seed, 0, PILOT_GAMES, args.checkpoint,
                           f"{args.out_dir}/pilot"))
            print("\n# N and G_total do not exist yet. Run the pilot gate, then the")
            print("# pilot ladder, then `size`, then emit-continuation-command.")
            return 0

        if args.cmd == "emit-continuation-command":
            # RECOMPUTE, then compare. The artifact is a claim, not an authority:
            # trusting it would let a hand-edited G_total authorize a block the
            # frozen sizing rule never produced -- and the continuation block is
            # the expensive one.
            pilot = load_block(args.pilot_dir, args.base_seed, 0, PILOT_GAMES)
            recomputed = size_continuation(pilot, args.n_target)
            art = json.loads(Path(args.size_artifact).read_text())
            if recomputed.get("verdict") != "OK":
                print(f"error: recomputed sizing is not OK: "
                      f"{recomputed.get('verdict')!r}", file=sys.stderr)
                return 3
            for field in ("verdict", "G_total", "g_cont"):
                if art.get(field) != recomputed.get(field):
                    print(f"error: size artifact does not match a recomputation "
                          f"from the pilot: {field}={art.get(field)!r} vs "
                          f"{recomputed.get(field)!r}", file=sys.stderr)
                    return 2
            # The supplied checkpoint must be the SAME network the pilot used.
            # Checking at assignment instead would reject only after the costly
            # continuation had already been generated.
            pilot_man = load_manifest(args.pilot_dir)
            supplied_sha1 = _sha1_file(args.checkpoint)
            if not supplied_sha1:
                print(f"error: checkpoint not found or unreadable: "
                      f"{args.checkpoint}", file=sys.stderr)
                return 2
            if supplied_sha1 != pilot_man.get("checkpoint_sha1"):
                print(f"error: checkpoint digest {supplied_sha1} does not match "
                      f"the pilot's {pilot_man.get('checkpoint_sha1')}; the "
                      f"continuation would be a different network under the same "
                      f"seed range", file=sys.stderr)
                return 2
            g_total = int(recomputed["G_total"])
            print(_STOP)
            print(f"\n# continuation block [24, {g_total}) -- its OWN directory:")
            print(_gen_cmd(args.base_seed, PILOT_GAMES, g_total - PILOT_GAMES,
                           args.checkpoint, f"{args.out_dir}/continuation"))
            print(f"\n# G_total = {g_total} was RECOMPUTED from the pilot and "
                  f"matched against {args.size_artifact};")
            print("# this tool accepts no free-form --g-total.")
            return 0

        if args.cmd == "pilot-gate":
            metas = load_block(args.sidecar_dir, args.base_seed, 0, PILOT_GAMES)
            r = pilot_geometry_gate(metas, args.sampling_seed)
            _emit(r)
            return 0 if r["verdict"] == "PASS" else 3

        if args.cmd == "size":
            metas = load_block(args.sidecar_dir, args.base_seed, 0, PILOT_GAMES)
            r = size_continuation(metas, args.n_target)
            _emit(r)
            return 0 if r["verdict"] == "OK" else 3

        # assign
        pilot = load_block(args.pilot_dir, args.base_seed, 0, PILOT_GAMES)
        gate = pilot_geometry_gate(pilot, args.sampling_seed)
        if gate["verdict"] != "PASS":
            _emit(gate)
            return 3

        # The continuation interval is DERIVED, never read from the block being
        # validated. Taking it from the block's own manifest would accept any
        # self-consistent size -- and a continuation larger than G_total is an
        # unauthorized top-up in disguise, since the surplus games become extra
        # matching capacity the frozen sizing rule never granted.
        sizing = size_continuation(pilot, args.n_target)
        if sizing["verdict"] != "OK":
            _emit(sizing)
            return 3
        want_n = int(sizing["G_total"]) - PILOT_GAMES
        cont_man = load_manifest(args.continuation_dir)
        if (cont_man["start_index"], cont_man["n_games"]) != (PILOT_GAMES, want_n):
            print(f"error: continuation block is [{cont_man['start_index']}, "
                  f"{cont_man['start_index'] + cont_man['n_games']}); the frozen "
                  f"sizing rule requires exactly [{PILOT_GAMES}, "
                  f"{sizing['G_total']}) for N={args.n_target}", file=sys.stderr)
            return 2
        assert_blocks_agree(load_manifest(args.pilot_dir), cont_man)
        cont = load_block(args.continuation_dir, args.base_seed,
                          PILOT_GAMES, want_n)
        r = assign_corpus(gate["assignment"], cont, args.n_target, args.sampling_seed)
        _emit(r)
        return 0 if r["verdict"] == "OK" else 4

    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
