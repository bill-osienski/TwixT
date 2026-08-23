#!/usr/bin/env python3
"""POST-RUN (2026-08-23). Not part of the G1 run; added during review.

WHY THIS EXISTS. g1_probe.py RECORDED shapes and finiteness but never ASSERTED
them, so it would have exited 0 on non-finite values or a wrong movelogits
shape. Exit 0 is therefore evidence that the process completed, NOT that the G1
criteria held. This file supplies the missing binding by checking the criteria
against the RECORDED outputs. It re-runs nothing: no model is loaded, no engine
is executed, no TensorFlow is imported.

usage: 07_POSTRUN_verify_criteria.py <path-to-twixtbot-ui-clone>
exit 0 = criteria hold; exit 1 = criteria violated; exit 2 = cannot verify.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PWIN_WIDTH = 3          # three-class head
BOARD = 24


def derived_move_width(clone):
    """Move-space width from the RULES ENGINE, not from the recorded output.

    Asserting `movelogits == 528` against the number we observed would be
    self-confirming. twixt.Game builds open_pegs as the cells a colour may play:
    x not in (0, SIZE-1), i.e. 22*24 = 528. Importing twixt pulls no TF.
    """
    sys.path.insert(0, clone)
    from src.backend.twixt import Game
    g = Game(allow_scl=False)
    widths = {len(g.open_pegs[c]) for c in (Game.BLACK, Game.WHITE)}
    if len(widths) != 1:
        raise AssertionError(f"colours disagree on move-space width: {widths}")
    if Game.SIZE != BOARD:
        raise AssertionError(f"board is {Game.SIZE}, expected {BOARD}")
    return widths.pop()


def check(rec, exit_code, move_width):
    """Every G1 pass criterion, as an assertion. Raises on violation."""
    assert exit_code == 0, f"probe exit was {exit_code}, not 0"
    assert rec["pwin"]["all_finite"] is True, "pwin contains non-finite values"
    assert rec["movelogits"]["all_finite"] is True, "movelogits contains non-finite values"
    assert rec["pwin"]["shape"] == [1, PWIN_WIDTH], f'pwin shape {rec["pwin"]["shape"]}'
    assert rec["movelogits"]["shape"] == [1, move_width], \
        f'movelogits shape {rec["movelogits"]["shape"]} != [1, {move_width}]'
    assert rec["position"]["board_size"] == BOARD, "board size is not 24"
    assert rec["position"]["allow_scl"] is False, "allow_scl was not False"


def negative_self_test(rec, exit_code, move_width):
    """A checker that has never rejected anything is not known to bind.

    Mutate each criterion in turn and confirm `check` raises. If any mutation
    slips through, this file is decorative and says so.
    """
    import copy
    cases = []
    for path, bad in [
        (("pwin", "all_finite"), False),
        (("movelogits", "all_finite"), False),
        (("pwin", "shape"), [1, 2]),
        (("movelogits", "shape"), [1, move_width + 1]),
        (("position", "board_size"), 19),
        (("position", "allow_scl"), True),
    ]:
        m = copy.deepcopy(rec)
        m[path[0]][path[1]] = bad
        try:
            check(m, exit_code, move_width)
        except AssertionError:
            cases.append((f"{path[0]}.{path[1]}={bad!r}", "rejected"))
        else:
            cases.append((f"{path[0]}.{path[1]}={bad!r}", "SLIPPED THROUGH"))
    try:
        check(rec, 1, move_width)
    except AssertionError:
        cases.append(("exit=1", "rejected"))
    else:
        cases.append(("exit=1", "SLIPPED THROUGH"))
    return cases


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-2])
        return 2
    clone = sys.argv[1]
    if not os.path.isdir(clone):
        print(f"FAIL: clone not found at {clone}; cannot derive the move width "
              f"independently, so the shape criterion cannot be verified.")
        return 2

    rec = json.load(open(os.path.join(HERE, "02_probe.stdout.txt")))
    exit_code = int(open(os.path.join(HERE, "04_exit.txt")).read().strip().split("=")[1])
    move_width = derived_move_width(clone)
    print(f"move-space width derived from the rules engine: {move_width}")

    print("\nnegative self-test (each mutation must be rejected):")
    slipped = 0
    for label, verdict in negative_self_test(rec, exit_code, move_width):
        print(f"  {verdict:16} {label}")
        slipped += verdict != "rejected"
    if slipped:
        print(f"\nFAIL: {slipped} mutation(s) slipped through; this checker does not bind.")
        return 1

    print("\nG1 criteria against the recorded outputs:")
    try:
        check(rec, exit_code, move_width)
    except AssertionError as e:
        print(f"  VIOLATED: {e}")
        return 1
    print(f"  exit code            0")
    print(f"  pwin                 shape {rec['pwin']['shape']}, all finite")
    print(f"  movelogits           shape {rec['movelogits']['shape']}, all finite")
    print(f"  board / allow_scl    {rec['position']['board_size']} / {rec['position']['allow_scl']}")
    print("\nPASS: every G1 criterion holds on the recorded outputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
