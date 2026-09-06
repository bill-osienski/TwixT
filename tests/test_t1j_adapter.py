"""Pure-Python checks for the E3b-qualified T1j adapter.

No Java, no T1j, no subprocess: these cover the parts of the adapter that are
ours -- the coordinate transforms, the external cap semantics, and the dump
parser. The lockstep behaviour against the real engine is covered by the E3b
qualification run, whose evidence is committed under docs/superpowers/evidence/.
"""
import subprocess

import pytest

from scripts.GPU.alphazero import t1j_adapter as A


@pytest.mark.parametrize("transform", A.SURVIVING_TRANSFORMS)
def test_transforms_round_trip_everywhere(transform):
    """to_ours is the exact inverse of to_t1j, over the whole board."""
    for row in range(A.BOARD_N):
        for col in range(A.BOARD_N):
            x, y = A.to_t1j(row, col, transform=transform)
            assert 0 <= x < A.BOARD_N and 0 <= y < A.BOARD_N
            assert A.to_ours(x, y, transform=transform) == (row, col)


@pytest.mark.parametrize("transform", A.SURVIVING_TRANSFORMS)
def test_transforms_are_bijections(transform):
    images = {A.to_t1j(r, c, transform=transform)
              for r in range(A.BOARD_N) for c in range(A.BOARD_N)}
    assert len(images) == A.BOARD_N * A.BOARD_N


def test_canonical_is_a_survivor_and_players_are_paired():
    assert A.CANONICAL in A.SURVIVING_TRANSFORMS
    assert A.to_t1j(3, 7) == (7, 3) and A.to_ours(7, 3) == (3, 7)
    assert A.PLAYER_TO_T1J == {"red": "Y", "black": "X"}
    assert {A.T1J_TO_PLAYER[v]: v for v in ("Y", "X")} == {"red": "Y", "black": "X"}


def test_unknown_transform_rejected():
    with pytest.raises(ValueError):
        A.to_t1j(0, 0, transform="rotate_37_degrees")
    with pytest.raises(ValueError):
        A.to_ours(0, 0, transform="rotate_37_degrees")


def test_terminal_with_cap_is_natural_or_capped():
    # below the cap and not naturally terminal -> not terminal
    assert A.terminal_with_cap(5, False, ply_cap=10) is False
    # naturally terminal below the cap -> terminal
    assert A.terminal_with_cap(5, True, ply_cap=10) is True
    # exactly at the cap -> terminal even without a natural win
    assert A.terminal_with_cap(10, False, ply_cap=10) is True
    # past the cap -> terminal
    assert A.terminal_with_cap(11, False, ply_cap=10) is True
    # cap 0 makes even the empty position terminal
    assert A.terminal_with_cap(0, False, ply_cap=0) is True


def test_ply_cap_is_required_keyword_only():
    with pytest.raises(TypeError):
        A.terminal_with_cap(1, False)              # type: ignore[call-arg]
    with pytest.raises(TypeError):
        A.terminal_with_cap(1, False, 10)          # type: ignore[misc]
    with pytest.raises(TypeError):
        # timeout_s supplied, so this isolates the MISSING ply_cap; without it
        # the call would raise for two reasons and prove neither.
        A.replay([], java="j", jar="j", classes="c", timeout_s=1)  # type: ignore[call-arg]


def test_negative_cap_rejected():
    with pytest.raises(ValueError):
        A.terminal_with_cap(0, False, ply_cap=-1)


def bits(*cells):
    """A full-width legal map with exactly `cells` set."""
    on = {r * A.BOARD_N + c for (r, c) in cells}
    return "".join("1" if i in on else "0" for i in range(A.LEGAL_BITS))


DUMP = """SIZE x=24 y=24
CAP 280
PLY 0 mover=- move=- next=Y moveNr=0 termY=false termX=false pegs=0 bridges=0
  PEGS
  BRIDGES
  HIST
  LEGAL {full}
PLY 1 mover=Y move=10,10 next=X moveNr=1 termY=false termX=true pegs=1 bridges=0
  PEGS 10,10,Y
  BRIDGES 1,1|2,3|Y
  HIST 10,10
  LEGAL {one}
""".format(full=bits((0, 0), (0, 1)), one=bits((0, 1)))


def test_parse_dump_reads_every_field():
    plies = A.parse_dump(DUMP)
    assert len(plies) == 2
    first, second = plies
    assert first.ply == 0 and first.next_player == "Y"
    assert first.pegs == set() and first.bridges == set() and first.history == ()
    assert first.legal == {(0, 0), (0, 1)}
    assert first.winner is None
    assert second.ply == 1
    assert second.pegs == {"10,10,Y"} and second.bridges == {"1,1|2,3|Y"}
    assert second.history == ((10, 10),)
    assert second.legal == {(0, 1)}
    assert second.term_x is True and second.winner == "X"


def test_parse_dump_winner_prefers_y_then_x_then_none():
    def one(ty, tx):
        return A.parse_dump(
            f"PLY 0 next=Y moveNr=0 termY={ty} termX={tx}\n"
            f"  PEGS\n  BRIDGES\n  HIST\n  LEGAL {bits()}\n"
        )[0].winner
    assert one("true", "false") == "Y"
    assert one("false", "true") == "X"
    assert one("false", "false") is None


def test_our_snapshot_maps_pegs_and_bridges():
    class FakeState:
        pegs = {(1, 2): "red", (3, 4): "black"}
        bridges = {((1, 2), (3, 4))}
    pegs, bridges = A.our_snapshot(FakeState())
    # identity transform sends (row, col) -> (col, row)
    assert pegs == {"2,1,Y", "4,3,X"}
    assert bridges == {"2,1|4,3|Y"}


# --- the serialized legal map must be exactly BOARD_N**2 wide ---------------

def test_legal_bits_round_trip_full_width():
    cells = {(0, 0), (5, 7), (23, 23)}
    assert A.parse_legal_bits(bits(*cells)) == cells


def test_truncated_legal_map_rejected_even_though_the_set_is_identical():
    """The control is only meaningful because truncation is otherwise invisible."""
    cells = {(0, 0), (5, 7)}
    full = bits(*cells)
    short = full.rstrip("0")
    assert len(short) < A.LEGAL_BITS - A.BOARD_N   # a whole column of tail dropped
    # decoded leniently, the truncated form is the SAME set -- only width catches it
    lenient = {(i // A.BOARD_N, i % A.BOARD_N) for i, b in enumerate(short) if b == "1"}
    assert lenient == cells
    with pytest.raises(ValueError):
        A.parse_legal_bits(short)


def test_extended_legal_map_rejected():
    with pytest.raises(ValueError):
        A.parse_legal_bits(bits((0, 0)) + "0")


def test_non_binary_legal_map_rejected():
    with pytest.raises(ValueError):
        A.parse_legal_bits("2" + bits((0, 0))[1:])


def test_parse_dump_rejects_a_truncated_legal_line():
    tampered = DUMP.replace("  LEGAL " + bits((0, 1)), "  LEGAL " + bits((0, 1)).rstrip("0"))
    assert tampered != DUMP
    with pytest.raises(ValueError):
        A.parse_dump(tampered)


# --- the Java helper ships with the adapter --------------------------------

def test_committed_java_sources_are_present():
    assert len(A.JAVA_SOURCES) == 3
    for src in A.JAVA_SOURCES:
        assert src.is_file(), src
    names = {p.name for p in A.JAVA_SOURCES}
    assert names == {"ScratchPrefs.java", "ScratchPrefsFactory.java", "E3bDump.java"}


def test_compile_helper_refuses_when_a_source_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "JAVA_SOURCES", A.JAVA_SOURCES + (tmp_path / "Absent.java",))
    with pytest.raises(FileNotFoundError):
        A.compile_helper("javac", "t1j.jar", str(tmp_path / "out"))


# --- E4 preflight: the generic fixed-position query path --------------------

QUERY_LINE = (
    "QUERY q=1 requested_depth=5 mdFixedPly=true mdPly=5 move_x=15 move_y=15 "
    "to_move=Y usealphabeta=true currentMaxPly=6 completed_depth=5 completed=true "
    "legal=true null_sentinel=false moveNr=6 eval_regime=early_moveNr_lt_8 elapsed_us=1234"
)


def test_parse_queries_reads_every_field_and_maps_the_move():
    (r,) = A.parse_queries("PROC pid=1\n" + QUERY_LINE + "\nPOSTCOND x=1\n")
    assert (r.q, r.requested_depth, r.to_move) == (1, 5, "Y")
    assert r.move == A.to_ours(15, 15)          # identity transform -> (15, 15)
    assert r.usealphabeta and r.completed and r.legal
    assert (r.current_max_ply, r.completed_depth) == (6, 5)
    assert not r.null_sentinel
    assert (r.move_nr, r.eval_regime, r.elapsed_us) == (6, "early_moveNr_lt_8", 1234)


def test_parse_queries_null_sentinel_has_no_move():
    line = QUERY_LINE.replace("move_x=15 move_y=15", "move_x=-1 move_y=-1") \
                     .replace("null_sentinel=false", "null_sentinel=true")
    (r,) = A.parse_queries(line)
    assert r.move is None and r.null_sentinel


def test_parse_queries_rejects_a_missing_field():
    with pytest.raises(ValueError):
        A.parse_queries(QUERY_LINE.replace(" completed_depth=5", ""))


def test_parse_queries_ignores_non_query_lines():
    assert A.parse_queries("PROC pid=7\nPOSTCOND failures=0\n") == []


def test_query_rejects_a_depth_below_the_deepening_floor():
    for bad in (0, 1, 2):
        with pytest.raises(ValueError):
            A.query([], depth=bad, java="j", jar="j", classes="c")


def test_preflight_sources_extend_the_e3b_set_without_changing_it():
    assert len(A.JAVA_SOURCES) == 3                      # E3b's qualified set, untouched
    assert A.PREFLIGHT_SOURCES[:3] == A.JAVA_SOURCES
    assert A.PREFLIGHT_SOURCES[3].name == "E4Preflight.java"
    for src in A.PREFLIGHT_SOURCES:
        assert src.is_file(), src
    assert A.PREFLIGHT_MAIN == "net.schwagereit.t1j.E4Preflight"


def test_compile_helper_still_defaults_to_the_e3b_set(tmp_path, monkeypatch):
    seen = {}

    def fake_run(args, **kw):
        seen["args"] = args
        class R: returncode = 0; stdout = ""; stderr = ""
        return R()

    monkeypatch.setattr(A.subprocess, "run", fake_run)
    A.compile_helper("javac", "t1j.jar", str(tmp_path))
    assert sum(a.endswith(".java") for a in seen["args"]) == 3
    A.compile_helper("javac", "t1j.jar", str(tmp_path), sources=A.PREFLIGHT_SOURCES)
    assert sum(a.endswith(".java") for a in seen["args"]) == 4


PROC_LINE = ("PROC pid=1234 java_version=17.0.20.1 vm=OpenJDK_64-Bit_Server_VM "
             "headless=true prefs_factory=e2probe.ScratchPrefs")


def test_parse_procs_reads_identity_and_counts_processes():
    (p,) = A.parse_procs(PROC_LINE + "\n" + QUERY_LINE + "\n")
    assert p.pid == 1234 and p.java_version == "17.0.20.1"
    assert p.prefs_factory == "e2probe.ScratchPrefs" and p.headless == "true"
    assert len(A.parse_procs(PROC_LINE + "\n" + PROC_LINE.replace("1234", "9") + "\n")) == 2
    assert A.parse_procs("QUERY q=1\n") == []


def test_parse_procs_rejects_a_missing_field():
    with pytest.raises(ValueError):
        A.parse_procs(PROC_LINE.replace(" headless=true", ""))


POSTCOND_LINE = ("POSTCOND no_throw=true windows=0 frames=0 headless=true prefs_ok=true "
                 "refl_ok=true refl_n=3 failures=0")


def test_parse_postconds_reads_the_safety_surface():
    (p,) = A.parse_postconds(POSTCOND_LINE + "\n")
    assert p.clean and p.refl_n == 3 and p.windows == 0 and p.prefs_ok


@pytest.mark.parametrize("field,bad", [
    ("no_throw=true", "no_throw=false"),
    ("windows=0", "windows=1"),
    ("frames=0", "frames=2"),
    ("headless=true", "headless=false"),
    ("prefs_ok=true", "prefs_ok=false"),
    ("refl_ok=true", "refl_ok=false"),
    ("failures=0", "failures=1"),
])
def test_a_dirty_postcond_is_not_clean(field, bad):
    (p,) = A.parse_postconds(POSTCOND_LINE.replace(field, bad) + "\n")
    assert not p.clean


def test_parse_postconds_rejects_a_missing_field():
    with pytest.raises(ValueError):
        A.parse_postconds(POSTCOND_LINE.replace(" prefs_ok=true", ""))


# ═══ the helper's OWN preference observations (2026-09-05 reporting gap 1) ═══
#
# The H1 match VOIDED on `prefs_ok=false`, and nothing recorded WHAT the failing
# JVM had compared: a Python sample at run start and another after the failure
# can miss a transient Java read error entirely. The helper now emits its own
# before/after values on the POSTCOND line, additively. The check is unchanged.

SHA_A = "6cb3a052650f90de53f34a8eb25455c470c6254c5f0fcac3f80c3ca9e8d0128d"
SHA_B = "0f9b9e9d3a6e1c2b4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4"
PREFS_OBS = (" prefs_before={b} prefs_after={a} count_before={cb} count_after={ca}")
POSTCOND_LINE_V2 = (POSTCOND_LINE.replace("prefs_ok=true", "prefs_ok=false")
                    + PREFS_OBS.format(b=SHA_A, a="ERROR", cb=527, ca=-1))


def test_parse_postconds_reads_the_helpers_OWN_prefs_observations():
    (p,) = A.parse_postconds(POSTCOND_LINE_V2 + "\n")
    assert (p.prefs_before, p.prefs_after) == (SHA_A, "ERROR")
    assert (p.count_before, p.count_after) == (527, -1)
    assert type(p.count_before) is int and type(p.count_after) is int
    assert not p.prefs_ok and not p.clean


def test_parse_postconds_tolerates_the_EARLIER_source_without_observations():
    """E4, L0, D1, the low-ply qualification and the 2026-09-05 H1 match all ran
    against the earlier E4Preflight source, and E3bDump still emits this shape.
    Those transcripts must stay parseable: the observations are additive."""
    (p,) = A.parse_postconds(POSTCOND_LINE + "\n")
    assert (p.prefs_before, p.prefs_after, p.count_before, p.count_after) == (
        None, None, None, None)
    assert p.clean


@pytest.mark.parametrize("ok,b,a,cb,ca", [
    ("true", SHA_A, SHA_B, 527, 527),      # values differ, yet prefs_ok=true
    ("true", SHA_A, SHA_A, 527, 526),      # counts differ, yet prefs_ok=true
    ("false", SHA_A, SHA_A, 527, 527),     # nothing differs, yet prefs_ok=false
])
def test_a_POSTCOND_line_that_DISAGREES_WITH_ITSELF_is_refused(ok, b, a, cb, ca):
    """🔑 TWO CHANNELS ON ONE LINE MUST SAY ONE THING. `prefs_ok` is the verdict
    the helper computed from exactly these four values; a line where they
    disagree is an unreadable instrument, not an observation."""
    line = (POSTCOND_LINE.replace("prefs_ok=true", f"prefs_ok={ok}")
            + PREFS_OBS.format(b=b, a=a, cb=cb, ca=ca))
    with pytest.raises(A.HelperOutputError, match="disagrees with itself") as exc:
        A.parse_postconds(line + "\n")
    assert exc.value.stdout == line + "\n"


@pytest.mark.parametrize("ok,b,a,cb,ca", [
    ("false", SHA_A, SHA_B, 527, 527),
    ("false", SHA_A, SHA_A, 527, -1),
    ("false", "null", SHA_A, 527, 527),    # plistBefore was never taken
    ("false", "ABSENT", SHA_A, 526, 527),
    ("true", SHA_A, SHA_A, 527, 527),
])
def test_a_SELF_CONSISTENT_line_is_accepted(ok, b, a, cb, ca):
    line = (POSTCOND_LINE.replace("prefs_ok=true", f"prefs_ok={ok}")
            + PREFS_OBS.format(b=b, a=a, cb=cb, ca=ca))
    (p,) = A.parse_postconds(line + "\n")
    assert p.prefs_ok is (ok == "true")


def test_the_prefs_observation_is_readable_from_a_BOUNDED_EXCERPT():
    """The query path raises `AbortError` carrying only `helper_failure_excerpt`
    -- FAIL lines and the POSTCOND line joined by " | " -- so the observation has
    to be recoverable from that text, where POSTCOND does not start a line.
    It carries `prefs_ok` too, so the reader does not need the excerpt to know
    whether the check failed."""
    excerpt = A.helper_failure_excerpt(
        "PROC pid=1\nFAIL preference surfaces unchanged\n" + POSTCOND_LINE_V2 + "\n")
    assert excerpt.startswith("FAIL preference")
    assert A.postcond_prefs_observation(excerpt) == {
        "prefs_ok": False, "prefs_before": SHA_A, "prefs_after": "ERROR",
        "count_before": 527, "count_after": -1}
    # the earlier source: a COMPLETE line with no observation = four Nones
    assert A.postcond_prefs_observation("FAIL x | " + POSTCOND_LINE) == {
        "prefs_ok": True, "prefs_before": None, "prefs_after": None,
        "count_before": None, "count_after": None}
    assert A.postcond_prefs_observation("no postcond here") is None


@pytest.mark.parametrize("text", [
    # TRUNCATED by the excerpt bound: the marker, or a base field cut off
    "FAIL x | " + POSTCOND_LINE_V2[:-6] + "...",
    "FAIL x | " + POSTCOND_LINE_V2.split(" count_after=")[0],
    "FAIL x | " + POSTCOND_LINE.split(" failures=")[0],
    # a COMPLETE-looking legacy line with the marker: the cut may have fallen
    # exactly before the observation fields -- "older helper" is not knowable
    "FAIL x | " + POSTCOND_LINE + "...",
    # PARTIAL: some observation fields, not all four
    "FAIL x | " + POSTCOND_LINE + " prefs_before=ABSENT prefs_after=ERROR",
])
def test_an_INCOMPLETE_segment_reads_as_UNKNOWN_not_as_the_earlier_source(text):
    """🔴 MISSING FIELDS CANNOT IDENTIFY AN OLDER HELPER. A segment the excerpt
    bound cut short, or a partial observation, used to read as four Nones --
    the same answer as "the helper ran the earlier source". Unknown is None."""
    assert A.postcond_prefs_observation(text) is None


@pytest.mark.parametrize("present", [
    ("prefs_before",), ("prefs_after",), ("count_before",), ("count_after",),
    ("prefs_before", "prefs_after"),
    ("prefs_before", "prefs_after", "count_before"),
    ("prefs_after", "count_before", "count_after"),
])
def test_a_PARTIAL_observation_is_REFUSED_by_the_parser(present):
    """🔴 REPRODUCED BY REVIEW: prefs_ok=true, prefs_before=ABSENT,
    prefs_after=ERROR, count_after missing parsed as CLEAN, because the
    self-agreement check required all four fields and a partial line skipped it.
    Either NONE (the earlier source) or ALL FOUR; anything between is an
    unreadable instrument."""
    values = {"prefs_before": "ABSENT", "prefs_after": "ERROR",
              "count_before": "527", "count_after": "527"}
    line = POSTCOND_LINE + "".join(f" {k}={values[k]}" for k in present)
    with pytest.raises(A.HelperOutputError, match="partial") as exc:
        A.parse_postconds(line + "\n")
    assert exc.value.stdout == line + "\n"


# ═══════ the replay hole 12.9 recorded: no timeout parameter existed at all ═══
#
# `query` already took `timeout_s` and defaulted it to None -- protection present
# but switched off. `replay` was worse: it had no parameter, so no caller could
# bound it through the adapter's API at all, and `subprocess.run(timeout=None)`
# waits forever. These assert at the PROCESS BOUNDARY, never at the call site.

def _spy_run(monkeypatch):
    calls = []

    def fake(args, **kw):
        calls.append(kw)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake)
    return calls


def test_replay_requires_a_timeout_and_refuses_an_unbounded_one():
    with pytest.raises(TypeError, match="timeout_s"):
        A.replay([(1, 1)], ply_cap=280, java="j", jar="j", classes="c")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="timeout_s"):
        A.replay([(1, 1)], ply_cap=280, java="j", jar="j", classes="c", timeout_s=None)


def test_the_replay_timeout_reaches_subprocess_run(monkeypatch):
    calls = _spy_run(monkeypatch)
    A.replay([(1, 1)], ply_cap=280, java="j", jar="j", classes="c", timeout_s=42)
    assert calls, "no subprocess call was observed; the assertion below is vacuous"
    assert [c.get("timeout") for c in calls] == [42]


def test_the_query_timeout_still_reaches_subprocess_run(monkeypatch):
    calls = _spy_run(monkeypatch)
    A.query([(1, 1)], depth=3, java="j", jar="j", classes="c", timeout_s=7)
    assert calls and [c.get("timeout") for c in calls] == [7]
