"""Atlas operator CLI -- design sections 3, 4, 9.

`preflight` and `emit-runbook` are ZERO-GPU. `run-pilot` and `run-final` are the
launchable entry points; they are the only places an evaluator is ever
constructed, and the factory is imported lazily inside those branches so the
other subcommands -- and every test in this stage -- never touch MLX.

THIS TOOL IS NOT AN AUTHORIZATION. Running the atlas is a separate written
decision.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .atlas_artifact import emit, load_run
from .atlas_run import run_final, run_pilot
from .build_atlas_corpus import _jsonable
from .corpus_geometry import PILOT_GAMES
from .generate_atlas_reservoir import (
    assert_blocks_agree, load_block, load_manifest, preflight_source_provenance,
)

EXIT_OK, EXIT_USAGE, EXIT_PROVENANCE, EXIT_ABORTED = 0, 2, 3, 5

_STOP = ("=" * 72 + "\nOPERATOR STOP -- running the atlas is NOT AUTHORIZED by "
         "this tool.\n" + "=" * 72)

# Verdict, the module that raises it, the operator action, and the exit code.
# A read-out verdict is a RESULT the run was asked to produce and exits 0; only
# usage, provenance and abort are nonzero. Conflating the two would make an
# operational no-go look like a crash.
STOP_CONDITIONS = (
    {"verdict": "PHASE_GEOMETRY_NO_GO", "owner": "build_atlas_corpus pilot-gate",
     "exit_code": EXIT_PROVENANCE,
     "action": "Stop before the pilot ladder. No replacement games, no "
               "reassignment, and NOT a smaller pilot."},
    {"verdict": "ASSIGNMENT_SHORTFALL", "owner": "build_atlas_corpus assign",
     "exit_code": 4,
     "action": "Stop. Do not top up, rebalance cells, move pilot rows, or "
               "relax one-position-per-game."},
    {"verdict": "PROJECTED_CAPACITY_NO_GO", "owner": "atlas_labelling.size_from_pilot",
     "exit_code": EXIT_OK,
     "action": "Stop with a projected capacity no-go rather than spending the "
               "full run on a design expected to be underpowered."},
    {"verdict": "CAPACITY_FAILURE", "owner": "atlas_labelling.final_capacity_gate",
     "exit_code": EXIT_OK,
     "action": "Operational capacity failure. Do not weaken labels, move "
               "ambiguous rows, or add games."},
    {"verdict": "INSUFFICIENT_CLASSES", "owner": "atlas_readout_a.evaluate_detector",
     "exit_code": EXIT_OK,
     "action": "Absence of evidence. Report as itself; never read as lateness."},
    {"verdict": "INSUFFICIENT_DISCOVERY_CLASSES",
     "owner": "atlas_readout_a.evaluate_detector", "exit_code": EXIT_OK,
     "action": "Absence of evidence. Report as itself; never read as lateness."},
    {"verdict": "NOT_DEPLOYABLE", "owner": "atlas_readout_a.deployability",
     "exit_code": EXIT_OK,
     "action": "Median `remaining` is zero: Read-out A cannot authorize the "
               "bounded 320+80 prototype. Separation is still reported."},
    {"verdict": "NO_SHAPE_PASSES", "owner": "atlas_readout_c.select_shape",
     "exit_code": EXIT_OK,
     "action": "No widening shape clears the floors. Do NOT invent a third."},
    {"verdict": "PROVENANCE_FAILURE", "owner": "run_atlas preflight / atlas_artifact.emit",
     "exit_code": EXIT_PROVENANCE,
     "action": "The run is not reconstructible. Fix the tree, the checkpoint "
               "or the block, and start over."},
    {"verdict": "ABORTED", "owner": "atlas_run.run_corpus",
     "exit_code": EXIT_ABORTED,
     "action": "The corpus is exactly N assigned positions, so ONE unmeasured "
               "position disqualifies the run. Failures and partial rows are "
               "retained and the read-outs are marked non-authoritative. No "
               "failure-tolerance number exists -- completeness is binary."},
    {"verdict": "UNAVAILABLE", "owner": "atlas_run.run_pilot",
     "exit_code": EXIT_ABORTED,
     "action": "An aborted pilot does not size and does not close widening. "
               "Re-run the pilot; do not carry a partial N."},
)


def measure_provenance(checkpoint: str, *, pilot_dir=None,
                       continuation_dir=None,
                       pilot_artifact: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
    """MEASURE the tree, HEAD and the checkpoint -- never accept them as args.

    Runs before any evaluator exists, because checking afterwards means a dirty
    tree can consume an entire GPU run before anything rejects it.

    SYMMETRIC: the whole chain -- pilot block, continuation block, pilot run,
    final run -- is produced at ONE frozen qualified commit, so both the digest
    and the HEAD must match everywhere. A mismatch means regeneration or
    requalification, not a recorded note.
    """
    def _sources():
        for label, d in (("pilot block", pilot_dir),
                         ("continuation block", continuation_dir)):
            if d:
                yield label, load_manifest(d)
        if pilot_artifact is not None:
            yield "pilot artifact", pilot_artifact["provenance"]

    prov = preflight_source_provenance(checkpoint)       # git + sha1, measured
    for name, recorded in _sources():
        for field in ("checkpoint_sha1", "git_head"):
            if recorded.get(field) != prov[field]:
                raise ValueError(
                    f"{field} mismatch against {name}: measured "
                    f"{prov[field]} != recorded {recorded.get(field)}. The "
                    f"chain must be produced at ONE qualified commit; "
                    f"regenerate or requalify rather than proceeding.")
    if pilot_dir and continuation_dir:
        assert_blocks_agree(load_manifest(pilot_dir),
                            load_manifest(continuation_dir))
    return prov


def launch_wrapper(argv: Sequence[str], *, out_dir) -> str:
    """The detached wrapper body, as a string a shell can run.

    A FUNCTION, not runbook prose, so a test can build a harmless one, EXECUTE
    it, and read the sidecar back. A substring assertion cannot catch a
    redirection-order defect.

    `rc` is captured FIRST, then written, then re-raised as the wrapper's own
    exit status. Nothing about the ordering is left to redirection precedence.
    """
    q = " ".join(shlex.quote(a) for a in argv)
    out = shlex.quote(str(out_dir))
    return (f"{q} > {out}/run.log 2>&1\n"
            f"rc=$?\n"
            f'echo "REAL_EXIT=$rc" > {out}/shell_status\n'
            f"exit $rc\n")


def write_status_sidecar(path, *, verdict: str, exit_code: int,
                         **extra) -> None:
    """Written LAST, after the artifact, so its presence is itself evidence the
    run reached the end. It does NOT replace the wrapper shell's REAL_EXIT: a
    process killed before it can write writes nothing.
    """
    Path(path).write_text(json.dumps(
        {"verdict": verdict, "exit_code": exit_code, **extra},
        indent=2, sort_keys=True))


def _emit_runbook(args) -> int:
    out = args.out_dir or "<out_dir>"
    wrapper = launch_wrapper(
        [".venv/bin/python", "-m", "scripts.GPU.alphazero.run_atlas",
         "run-final", "--pilot-artifact", "<pilot.json>",
         "--corpus-artifact", "<assign.json>", "--pilot-dir", "<pilot_block>",
         "--continuation-dir", "<cont_block>", "--base-seed", "<n>",
         "--checkpoint", "<net>", "--out-dir", str(out)], out_dir=out)
    print(_STOP)
    print(f"""
 A. PILOT  (N is the pilot's OUTPUT, not an input)
 1. Clean tree. Preflight MEASURES it -- do not assert it:
      run_atlas preflight --checkpoint <net> --pilot-dir <pilot_block>
    exit 0 continue | 2 usage | 3 PROVENANCE_FAILURE -- stop, fix, restart.
 2. A PHASE_GEOMETRY_NO_GO means stop, NOT a smaller pilot.
 3. Launch the 24-row pilot (see LAUNCH), then read its artifact:
      .sizing.verdict     OK -> N ; PROJECTED_CAPACITY_NO_GO -> stop
      .early_widening_check.both_fail  true -> close progressive widening
                                       WITHOUT inventing another shape
 B. CONTINUATION
 4. Generate exactly G_total-24 games, then assign:
      build_atlas_corpus assign ... --n-target <N>
    exit 3 PHASE_GEOMETRY_NO_GO | 4 ASSIGNMENT_SHORTFALL -- stop. No top-up,
    no cell rebalance, no moving pilot rows.
 5. Project runtime from the corpus's OBSERVED per-phase ply supply:
      run_atlas_ladder project-runtime --rows <N> --mean-prefix-plies <measured>
    Never scale a smoke to estimate runtime.
 6. Launch the final run with --pilot-artifact: the pilot's 24 discovery rows
    are CARRIED, never re-measured, and 24 + (N-24) must equal N.

 LAUNCH -- always in a shell invocation of its own
 7. `setsid` does not exist on macOS, and a tool timeout SIGTERMs the whole
    process group when the launch and the wait share one call. Launch through a
    DETACHED SHELL WRAPPER so the shell -- not python -- records the exit code:

      OUT={out}; mkdir -p "$OUT"
      cat > "$OUT/launch.sh" <<'EOF'
{wrapper}EOF
      nohup sh "$OUT/launch.sh" > /dev/null 2>&1 &
      disown

    `rc` is captured BEFORE anything else writes, so no redirection can eat it.

 8. In a LATER call, read the two sidecars. Do NOT try to wait on the PID:
    after `disown` a later shell has neither the job table nor a usable job
    spec, which is exactly how Phase 0 lost its exit code. That is why the
    wrapper -- not python, and not the waiting shell -- records it.
      cat "$OUT/shell_status"     # REAL_EXIT=<n>, written by the wrapper shell
      cat "$OUT/status.json"      # verdict + exit_code, written LAST by the run

      both present            -> trust .verdict
      shell_status only       -> python died before reporting; read run.log
      neither                 -> the wrapper never ran; nothing was measured

    `cmd | tail` reports the PIPE's exit code -- redirect to a file instead.
 9. Exit 5 / verdict ABORTED means the corpus was not measured completely. The
    read-outs in the artifact are marked non-authoritative and are for
    diagnosis only.
10. No source edit after preflight, and no commit between generation and
    qualification.

 STOP CONDITIONS""")
    for s in STOP_CONDITIONS:
        print(f"   {s['verdict']:<32} exit {s['exit_code']}  ({s['owner']})")
        print(f"       {s['action']}")
    return EXIT_OK


def _cmd_preflight(args) -> int:
    try:
        prov = measure_provenance(args.checkpoint, pilot_dir=args.pilot_dir,
                                  continuation_dir=args.continuation_dir)
    except (RuntimeError, ValueError) as e:
        print(json.dumps({"verdict": "PROVENANCE_FAILURE", "reason": str(e)},
                         indent=2, sort_keys=True))
        return EXIT_PROVENANCE
    except (FileNotFoundError, OSError, KeyError) as e:
        print(json.dumps({"verdict": "USAGE", "reason": str(e)},
                         indent=2, sort_keys=True))
        return EXIT_USAGE
    if args.corpus_artifact:
        try:
            json.loads(Path(args.corpus_artifact).read_text())
        except (OSError, ValueError) as e:
            print(json.dumps({"verdict": "USAGE", "reason": str(e)},
                             indent=2, sort_keys=True))
            return EXIT_USAGE
    print(json.dumps(_jsonable({"verdict": "OK", "provenance": prov}),
                     indent=2, sort_keys=True))
    return EXIT_OK


def _finish(out_dir, name: str, doc: Dict[str, Any]) -> int:
    """Write the artifact, THEN the status sidecar -- in that order, so the
    sidecar's presence is evidence the run reached the end."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(emit(doc))
    code = EXIT_OK if doc["verdict"] == "OK" else EXIT_ABORTED
    write_status_sidecar(out / "status.json", verdict=doc["verdict"],
                         exit_code=code, measured=doc.get("measured"),
                         assigned=doc.get("assigned"))
    return code


def _cmd_run_pilot(args) -> int:
    """Launchable. Stage 5 writes it, drives it once against a patched factory,
    and never executes it for real."""
    prov = measure_provenance(args.checkpoint, pilot_dir=args.pilot_dir)
    games = load_block(args.pilot_dir, args.base_seed, 0, PILOT_GAMES)
    from .eval_runner import _default_evaluator_factory      # lazy: MLX
    evaluator = _default_evaluator_factory(args.checkpoint)  # ONCE per run
    doc = run_pilot(evaluator, games, sampling_seed=args.sampling_seed,
                    base_seed=args.base_seed,
                    move_histories=_histories(args.pilot_dir, games),
                    provenance=prov)
    return _finish(args.out_dir, "pilot_artifact.json", doc)


def _cmd_run_final(args) -> int:
    """Same shape, plus the pilot artifact as an INPUT."""
    pilot_doc = load_run(Path(args.pilot_artifact))
    prov = measure_provenance(args.checkpoint, pilot_dir=args.pilot_dir,
                              continuation_dir=args.continuation_dir,
                              pilot_artifact=pilot_doc)
    pilot_games = load_block(args.pilot_dir, args.base_seed, 0, PILOT_GAMES)
    cont_manifest = load_manifest(args.continuation_dir)
    cont_games = load_block(args.continuation_dir, args.base_seed,
                            cont_manifest["start_index"],
                            cont_manifest["n_games"])
    assignment = json.loads(Path(args.corpus_artifact).read_text())["rows"]
    from .eval_runner import _default_evaluator_factory      # lazy: MLX
    evaluator = _default_evaluator_factory(args.checkpoint)  # ONCE per run
    doc = run_final(evaluator, pilot_doc=pilot_doc, pilot_games=pilot_games,
                    continuation_games=cont_games,
                    assignment_rows=assignment, base_seed=args.base_seed,
                    move_histories=_histories(args.continuation_dir,
                                              cont_games),
                    provenance=prov)
    return _finish(args.out_dir, "atlas_artifact.json", doc)


def _histories(block_dir, games) -> Dict[int, List[Any]]:
    """Move histories from the block's sidecars, keyed by game id."""
    out = {}
    for g in games:
        d = json.loads((Path(block_dir) /
                        f"game_{g.game_id:06d}.json").read_text())
        out[g.game_id] = [tuple(m) for m in d["move_history"]]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Atlas operator CLI (this tool authorizes nothing)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("emit-runbook")
    s.add_argument("--out-dir")

    s = sub.add_parser("preflight")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--pilot-dir")
    s.add_argument("--continuation-dir")
    s.add_argument("--corpus-artifact")

    # NO --active-size / --prefix-sims / --increments / --n-target: the board,
    # the replay budget, the ladder and N are frozen, and a flag that can
    # change them is a protocol change with a command-line interface.
    s = sub.add_parser("run-pilot")
    s.add_argument("--pilot-dir", required=True)
    s.add_argument("--base-seed", type=int, required=True)
    s.add_argument("--sampling-seed", type=int, required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--out-dir", required=True)

    s = sub.add_parser("run-final")
    s.add_argument("--pilot-artifact", required=True)
    s.add_argument("--corpus-artifact", required=True)
    s.add_argument("--pilot-dir", required=True)
    s.add_argument("--continuation-dir", required=True)
    s.add_argument("--base-seed", type=int, required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--out-dir", required=True)

    args = ap.parse_args(argv)
    return {"emit-runbook": _emit_runbook, "preflight": _cmd_preflight,
            "run-pilot": _cmd_run_pilot, "run-final": _cmd_run_final}[
                args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
