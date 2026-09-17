"""🔴 THE RETIRED CO-PRODUCED GENERATION PATH — its tests, kept with it.

These exercise `h3_coproduced_generator_retired`, `h3_generation_command` (the
JVM supervisor) and `h3_generation_preflight`. ALL THREE ARE RETIRED: the
co-produced stratum closed on 2026-09-17 after two VOIDed attempts, the second
because T1j cannot move at the plies the protocol required.

WHY THEY STILL RUN. The machinery worked -- it caught both failures cleanly,
restored its gate every time and left a durable record of each. Deleting the
tests would leave the preserved code unexercised, and unexercised code in a tree
rots into something that LOOKS live. They stay green so the retirement is honest.

🔴 NOTHING HERE IS AN ACTIVE STUDY PATH. `test_h3_uniform_population.py` asserts
that no live module imports any of these three.
"""
import os
import sys
import json
import subprocess
import textwrap

import pytest

from scripts.GPU.alphazero import h3_coproduced_generator_retired as GEN
from scripts.GPU.alphazero import h3_generation_command as GCMD
from scripts.GPU.alphazero import h3_generation_preflight as PF
from scripts.GPU.alphazero import h3_study_rules as R
from scripts.GPU.alphazero import h3_study_runner as RUN


OK_SUP = {"timed_out": False, "interrupted": False, "group_cleared": True,
          "exit_code": 0}




def test_the_generation_wrapper_has_NO_gate_or_source_override():
    help_text = GCMD._parser().format_help()
    for flag in ("--authorize", "--force", "--gate", "--runner-source"):
        assert flag not in help_text, flag
    assert "os.environ" not in open(GCMD.__file__, encoding="utf-8").read()

def test_restore_gate_REWRITES_an_open_gate_and_VERIFIES_it(tmp_path):
    src = tmp_path / "gen.py"
    src.write_text("x = 1\nH3_GENERATION_AUTHORIZED = True\ny = 2\n")
    assert GCMD.restore_gate(str(src)) is True
    back = src.read_text()
    assert "H3_GENERATION_AUTHORIZED = False" in back
    assert "H3_GENERATION_AUTHORIZED = True" not in back

def test_restore_gate_is_FALSE_when_it_cannot_verify(tmp_path):
    missing = tmp_path / "nope.py"
    assert GCMD.restore_gate(str(missing)) is False
    two = tmp_path / "two.py"
    two.write_text("H3_GENERATION_AUTHORIZED = True\nH3_GENERATION_AUTHORIZED = True\n")
    assert GCMD.restore_gate(str(two)) is False


@pytest.mark.parametrize("r,want", [
    ({"timed_out": False, "interrupted": False, "group_cleared": True,
      "exit_code": 0}, 0),
    ({"timed_out": True, "interrupted": False, "group_cleared": True,
      "exit_code": 6}, 6),
    ({"timed_out": False, "interrupted": True, "group_cleared": True,
      "exit_code": 9}, 9),
    # 🔴 A SURVIVING DESCENDANT OUTRANKS EVERYTHING, even exit 0
    ({"timed_out": False, "interrupted": False, "group_cleared": False,
      "exit_code": 0}, 8),
    ({"timed_out": True, "interrupted": False, "group_cleared": False,
      "exit_code": 6}, 8),
])
def test_every_supervisor_outcome_gets_ITS_OWN_exit_code(monkeypatch, tmp_path,
                                                         r, want):
    # 🔴 THE RECEIPT PATH MUST BE REDIRECTED. Without this the test wrote a real
    # 00_launch_receipt.json into the RUN'S OWN DESTINATION -- a test artifact
    # sitting exactly where the authorized generation is meant to write, which
    # would then have refused the real launch as "already exists".
    monkeypatch.setattr(GCMD, "RECEIPT", str(tmp_path / "receipt.json"))
    monkeypatch.setattr(GCMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(GCMD, "supervise", lambda *a, **k: dict(r))
    monkeypatch.setattr(GCMD, "restore_gate", lambda *a, **k: True)
    code = GCMD.main(["--out", str(tmp_path / "o.json"),
                      "--trace", str(tmp_path / "t.jsonl")])
    assert code == want

def test_the_wrapper_REFUSES_BEFORE_SPAWNING_when_the_destination_exists(
        monkeypatch, tmp_path):
    spawned = []
    monkeypatch.setattr(GCMD, "RECEIPT", str(tmp_path / "receipt.json"))
    monkeypatch.setattr(GCMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(GCMD, "supervise",
                        lambda *a, **k: spawned.append(1) or {
                            "timed_out": False, "interrupted": False,
                            "group_cleared": True, "exit_code": 0})
    monkeypatch.setattr(GCMD, "restore_gate", lambda *a, **k: True)
    (tmp_path / "o.json").write_text("{}")
    code = GCMD.main(["--out", str(tmp_path / "o.json"),
                      "--trace", str(tmp_path / "t.jsonl")])
    assert code == GCMD.EXIT_REFUSED
    assert spawned == [], "nothing may be spawned once the destination is taken"

def test_the_destination_INSIDE_A_SPENT_DIRECTORY_is_refused(monkeypatch):
    monkeypatch.setattr(GCMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(GCMD, "restore_gate", lambda *a, **k: True)
    spent = GCMD.SPENT_OUT_DIRS[0]
    with pytest.raises(GEN.H3GenerationError, match="SPENT"):
        GCMD._check_destination(f"{spent}/o.json", f"{spent}/t.jsonl")

def test_THE_WORKER_REFUSES_without_the_supervisors_CAPABILITY(capsys):
    assert GCMD.worker_main(["--worker"]) == GCMD.EXIT_REFUSED
    assert "UNSUPERVISED" in capsys.readouterr().err

def test_the_capability_is_a_PIPE_and_DELETES_NOTHING(tmp_path):
    fd = GCMD._make_capability()
    assert GCMD._consume_capability(fd) is True
    assert GCMD._consume_capability(None) is False
    victim = tmp_path / "precious.txt"
    victim.write_text("x" * (GCMD.CAPABILITY_BYTES * 2))
    assert GCMD._consume_capability(str(victim)) is False
    assert victim.exists(), "the capability check must never delete a file"

def test_THE_FINALLY_PATH_also_restores_the_generation_gate(monkeypatch, tmp_path):
    """🔑 A DIFFERENT PATH FROM THE REFUSAL'S. With the gate SHUT the wrapper
    returns before the try/finally, so only a run that gets past the gate
    exercises the `finally`'s restoration — and a control that removes it is
    invisible to the refusal test."""
    monkeypatch.setattr(GCMD, "RECEIPT", str(tmp_path / "receipt.json"))
    monkeypatch.setattr(GCMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(GCMD, "supervise", lambda *a, **k: {
        "timed_out": False, "interrupted": False, "group_cleared": True,
        "exit_code": 0})
    monkeypatch.setattr(GCMD, "restore_gate", lambda *a, **k: False)
    code = GCMD.main(["--out", str(tmp_path / "o.json"),
                      "--trace", str(tmp_path / "t.jsonl")])
    assert code == GCMD.EXIT_GATE_NOT_RESTORED, (
        "a failed restoration SUPERSEDES the worker's own exit 0")

def test_GENERATION_HAS_ITS_OWN_PREREGISTERED_DEADLINE():
    """🔴 It was reusing SEGMENT_DEADLINE_S — a cap frozen for 148 GAMES at the
    pilot's 41.22 s/game, which says nothing about 148 six-ply generation walks.
    Same number, different quantity: a bound borrowed from something else."""
    assert R.GENERATION_DEADLINE_S == 10800
    assert "GENERATION_DEADLINE_S" in open(
        GEN.__file__, encoding="utf-8").read()
    src = open(GCMD.__file__, encoding="utf-8").read()
    assert "GENERATION_DEADLINE_S" in src
    assert "SEGMENT_DEADLINE_S" not in src, (
        "the generation wrapper must not borrow the match segment's cap")

def _launch(monkeypatch, tmp_path, *, sup=None, restore=True, gate=True):
    """Drive the wrapper's parent path with the supervisor stubbed."""
    monkeypatch.setattr(GCMD, "RECEIPT", str(tmp_path / "00_launch_receipt.json"))
    monkeypatch.setattr(GCMD, "gate_is_open", lambda: gate)
    monkeypatch.setattr(GCMD, "restore_gate", lambda *a, **k: restore)
    if sup is not None:
        monkeypatch.setattr(GCMD, "supervise", lambda *a, **k: dict(sup))
    code = GCMD.main(["--out", str(tmp_path / "o.json"),
                      "--trace", str(tmp_path / "t.jsonl")])
    import json
    path = tmp_path / "00_launch_receipt.json"
    doc = json.loads(path.read_text()) if path.exists() else None
    return code, doc

def test_NO_RECEIPT_WHEN_THE_GATE_WAS_SHUT(monkeypatch, tmp_path):
    """Nothing was attempted and nothing consumed; a receipt would claim a launch
    that never happened."""
    code, doc = _launch(monkeypatch, tmp_path, gate=False)
    assert code == GCMD.EXIT_UNAUTHORIZED and doc is None

def test_THE_RECEIPT_EXISTS_EVEN_WHEN_THE_WORKER_WROTE_NO_TRACE(monkeypatch,
                                                                tmp_path):
    """🔑 THE CASE THE WORKER'S OWN `finally` CANNOT COVER: killed before it ever
    opened its trace."""
    code, doc = _launch(monkeypatch, tmp_path,
                        sup={"timed_out": True, "interrupted": False,
                             "group_cleared": True, "exit_code": -9})
    assert code == GCMD.EXIT_TIMEOUT
    assert doc is not None
    assert doc["outcome"] == "TIMEOUT" and doc["timed_out"] is True
    assert doc["trace_exists"] is False and doc["artifact_exists"] is False
    assert doc["worker_exit"] == -9

@pytest.mark.parametrize("sup,restore,outcome,code_name", [
    (OK_SUP, True, "COMPLETED", "EXIT_COMPLETED"),
    ({**OK_SUP, "timed_out": True, "exit_code": 6}, True, "TIMEOUT", "EXIT_TIMEOUT"),
    ({**OK_SUP, "interrupted": True, "exit_code": 9}, True, "INTERRUPTED",
     "EXIT_INTERRUPTED"),
    ({**OK_SUP, "group_cleared": False}, True, "CLEANUP_FAILED",
     "EXIT_CLEANUP_FAILED"),
    (OK_SUP, False, "GATE_NOT_RESTORED", "EXIT_GATE_NOT_RESTORED"),
])
def test_THE_RECEIPT_RECORDS_THE_SUPERSEDING_OUTCOME(monkeypatch, tmp_path, sup,
                                                     restore, outcome, code_name):
    code, doc = _launch(monkeypatch, tmp_path, sup=sup, restore=restore)
    assert code == getattr(GCMD, code_name)
    assert doc["outcome"] == outcome
    assert doc["exit_code"] == code
    assert doc["gate_restored"] is restore
    assert doc["group_cleared"] is sup["group_cleared"]

def test_A_SURVIVING_DESCENDANT_SUPERSEDES_THE_WORKERS_OWN_SUCCESS(monkeypatch,
                                                                   tmp_path):
    """The worker exited 0; a descendant lived. The receipt says CLEANUP_FAILED
    and keeps the worker's 0 beside it, so neither fact is lost."""
    code, doc = _launch(monkeypatch, tmp_path,
                        sup={**OK_SUP, "group_cleared": False, "exit_code": 0})
    assert code == GCMD.EXIT_CLEANUP_FAILED
    assert doc["outcome"] == "CLEANUP_FAILED"
    assert doc["worker_exit"] == 0, "the worker's own result is still recorded"


def test_a_REFUSAL_BEFORE_SPAWNING_still_leaves_a_receipt(monkeypatch, tmp_path):
    """An authorization was spent on a run that could not start -- H2's lesson,
    where exactly that went unrecorded."""
    (tmp_path / "o.json").write_text("{}")
    code, doc = _launch(monkeypatch, tmp_path, sup=OK_SUP)
    assert code == GCMD.EXIT_REFUSED
    assert doc["outcome"] == "REFUSED" and doc["supervised"] is False
    assert "already exists" in doc["note"]

def test_THE_RECEIPT_IS_CREATE_ONLY_and_a_SECOND_LAUNCH_is_refused(monkeypatch,
                                                                   tmp_path):
    code, doc = _launch(monkeypatch, tmp_path, sup=OK_SUP)
    assert code == GCMD.EXIT_COMPLETED and doc is not None
    # the receipt now occupies the destination: a second launch must refuse
    code2, _ = _launch(monkeypatch, tmp_path, sup=OK_SUP)
    assert code2 == GCMD.EXIT_REFUSED

def test_O_EXCL_refuses_the_receipt_EVEN_WITH_THE_PRECHECK_DISABLED(monkeypatch,
                                                                    tmp_path):
    """🔑 THE PRECHECK CATCHES THE SECOND LAUNCH FIRST, so the create-only flag on
    the write itself is never exercised by that path — and a control removing
    `O_EXCL` went NOT CAUGHT. This reaches it alone: with `_check_destination`
    neutralised, the write must STILL refuse rather than overwrite a receipt that
    is the durable record of an earlier launch."""
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"outcome": "AN EARLIER LAUNCH"}')
    monkeypatch.setattr(GCMD, "RECEIPT", str(receipt))
    monkeypatch.setattr(GCMD, "_check_destination", lambda *a, **k: None)
    monkeypatch.setattr(GCMD, "gate_is_open", lambda: True)
    monkeypatch.setattr(GCMD, "restore_gate", lambda *a, **k: True)
    monkeypatch.setattr(GCMD, "supervise", lambda *a, **k: dict(OK_SUP))
    code = GCMD.main(["--out", str(tmp_path / "o.json"),
                      "--trace", str(tmp_path / "t.jsonl")])
    import json
    assert json.loads(receipt.read_text())["outcome"] == "AN EARLIER LAUNCH", (
        "the earlier launch's receipt was OVERWRITTEN")
    assert code == GCMD.EXIT_UNEXPECTED, (
        "a receipt that could not be written must not report success")


class _Ctx:
    """The move log a T1jAgent requires — and `reset`, because the REAL
    `IntegrationContext` has it and the walk must call it before either agent can
    move. A double without `reset` would have hidden the very defect that VOIDed
    the one authorized generation attempt."""

    def __init__(self):
        self.moves = []
        self.task_id = None
        self.stats = {}

    def reset(self, task_id, opening):
        assert isinstance(task_id, str) and task_id, task_id
        self.task_id = task_id
        self.moves = [tuple(m) for m in opening]
        self.stats.setdefault(task_id, {"binds": 0, "t1j_queries": 0,
                                        "searched_binds": 0})


def _inert_movers(config=None):
    """FACTORIES, not move functions — because the card gives each opening ONE
    agent whose streams advance across its plies, and a per-move function cannot
    have streams at all."""
    calls = {"built_incumbent": 0, "built_t1j": 0, "moves": 0, "seeds": [],
             "ctxs": []}

    def incumbent_agent(*, seed, colour):
        calls["built_incumbent"] += 1
        calls["seeds"].append(seed)
        assert colour in ("red", "black")
        state = {"n": 0}

        def agent(st):
            # 🔑 STREAM-LIKE: the move depends on the seed AND on how many moves
            # this agent has already made, exactly as a persisting generator does.
            calls["moves"] += 1
            state["n"] += 1
            legal = sorted(st.legal_moves())
            return legal[(seed + state["n"]) % len(legal)]

        return agent

    def t1j_agent(*, colour, ctx):
        calls["built_t1j"] += 1
        calls["ctxs"].append(ctx)
        assert colour in ("red", "black")

        def agent(st):
            calls["moves"] += 1
            assert len(ctx.moves) == st.ply, (len(ctx.moves), st.ply)
            return sorted(st.legal_moves())[-1]

        return agent

    return {"incumbent_agent": incumbent_agent, "t1j_agent": t1j_agent,
            "new_context": _Ctx,
            "config": config or GEN.generation_config(), "calls": calls}


def test_the_incumbent_FACTORY_TAKES_A_SEED_and_t1j_does_not():
    """🔑 The asymmetry IS the finding: only the incumbent supplies entropy."""
    import inspect as _i
    m = _inert_movers()
    assert "seed" in _i.signature(m["incumbent_agent"]).parameters
    assert "seed" not in _i.signature(m["t1j_agent"]).parameters


def test_THE_RETIRED_PREFLIGHT_CANNOT_RUN_BECAUSE_ITS_RANGE_IS_SPENT():
    """🔴 THE RETIREMENT IS ENFORCED BY ARITHMETIC, NOT ONLY BY A DOCSTRING.

    These two tests used to prove the preflight built both movers through the
    real path and refused an argmax generating config. Neither can be exercised
    now, and the reason is the point: attempt 2's range is in
    `SPENT_GENERATION_RANGES`, so `attempt_seed` refuses before a mover exists.

    A retirement that rests on a comment is a retirement someone reopens. This
    one rests on the spent-range check, which is the same check that would refuse
    any re-use of a drawn range.
    """
    with pytest.raises(R.H3StudyError, match="SPENT"):
        R.attempt_seed(GEN.GEN_SEED_CO_PRODUCED, 0, 0)
    with pytest.raises(R.H3StudyError, match="SPENT"):
        R.attempt_seed(GEN.GEN_SEED_CO_PRODUCED_ATTEMPT1, 0, 0)
    # ... and the preflight, which needs that seed, therefore cannot proceed
    with pytest.raises(R.H3StudyError, match="SPENT"):
        PF.preflight_movers()


def test_THE_RETIREMENT_IS_NOT_A_SWITCH_BECAUSE_THERE_IS_NO_SWITCH():
    """🔴 RETIREMENT MUST NOT LOOK LIKE A CLOSED GATE.

    A `False` constant invites someone to set it True after a review. No review
    can make T1j search at ply 1, so there is nothing to review and nothing to
    flip: the constant is GONE and `check_gate` refuses unconditionally.

    Leaving it at False while `check_gate` no longer read it would have been
    worse than either -- a control in appearance only.
    """
    assert not hasattr(GEN, "H3_GENERATION_AUTHORIZED")
    with pytest.raises(GEN.H3GenerationError, match="RETIRED"):
        GEN.check_gate()
    GEN.H3_GENERATION_AUTHORIZED = True      # nothing reads it, so nothing changes
    try:
        with pytest.raises(GEN.H3GenerationError, match="RETIRED"):
            GEN.check_gate()
    finally:
        del GEN.H3_GENERATION_AUTHORIZED


def test_THE_SUPERVISOR_CAN_NO_LONGER_OPEN_ANYTHING():
    """🔴 WHAT REPLACED THE GATE-RESTORATION TESTS, AND WHY.

    Two tests here used to prove the supervisor restored
    `H3_GENERATION_AUTHORIZED` in the source after every exit, and read it back
    from the file rather than from memory. They were good tests of a real
    control -- and their subject no longer exists: the constant is gone, because
    a gate `check_gate` had stopped reading was a control in appearance only.

    There is nothing left to restore, so the invariant is now the stronger one:
    the supervisor cannot open the path at all. Kept rather than deleted, so the
    retirement is asserted and not just described.
    """
    assert GCMD.gate_is_open() is False
    assert not hasattr(GEN, "H3_GENERATION_AUTHORIZED")
    # the constant it used to read is gone from the retired generator's source
    import pathlib as _p
    gen_src = _p.Path(GEN.__file__).read_text(encoding="utf-8")
    assert "\nH3_GENERATION_AUTHORIZED" not in gen_src
    with pytest.raises(GEN.H3GenerationError, match="RETIRED"):
        GEN.check_gate()


def test_NO_TEST_MAY_WRITE_INTO_THE_RUNS_OWN_DESTINATION():
    """🔴 A test DID. `test_every_supervisor_outcome_gets_ITS_OWN_exit_code` did
    not redirect `RECEIPT`, so running the suite created a real
    `00_launch_receipt.json` in the destination the authorized generation is
    meant to write — which would then have refused the real launch as "already
    exists". An authorization spent because a TEST occupied the destination is
    exactly H2's defect, arriving by a new road.

    Every wrapper test that reaches the receipt must patch the path first.
    """
    import ast
    import inspect as _i
    src = open(__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    offenders = []
    reaching = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        body = ast.get_source_segment(src, fn) or ""
        # only a test that FORCES THE GATE OPEN can reach the receipt at all:
        # with the gate shut the wrapper returns before the try/finally.
        reaches_receipt = ("GCMD.main(" in body
                           and 'GCMD, "gate_is_open", lambda: True' in body)
        redirects = 'GCMD, "RECEIPT"' in body or "_launch(" in body
        if reaches_receipt:
            reaching.append(fn.name)
            if not redirects:
                offenders.append(fn.name)
    assert offenders == [], offenders
    # 🔴 NON-VACUITY. This test used to live in `test_h3_study_runner.py` and
    # scanned that file. When the supervisor tests moved here it kept passing --
    # against a file with nothing left to police. A scan of the wrong file is
    # indistinguishable from a clean one, so it now proves it saw its subject.
    assert reaching, (
        "this scan found no test that reaches the receipt at all; it is "
        "scanning the wrong file and would pass however dirty that file got")
    # 🔑 AND THE INVARIANT ITSELF CHANGED WHEN THE DESTINATION WAS CONSUMED.
    # It used to assert the receipt was ABSENT, which was right while the
    # destination was unspent: a test creating one would have refused the real
    # launch as "already exists". Attempt 2 then CONSUMED it, so absence is now
    # the wrong thing to want -- that receipt is the VOID's own record and must
    # stay exactly as the run left it.
    import json
    import os
    assert os.path.lexists(GCMD.RECEIPT), "the VOID's receipt must be preserved"
    receipt = json.load(open(GCMD.RECEIPT, encoding="utf-8"))
    assert receipt["outcome"] == "UNEXPECTED" and receipt["worker_exit"] == 4, (
        "the preserved receipt no longer records attempt 2's VOID; a test has "
        "overwritten spent evidence")
    assert receipt["artifact_exists"] is False and receipt["gate_restored"] is True
