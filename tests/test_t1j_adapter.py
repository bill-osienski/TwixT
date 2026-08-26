"""Pure-Python checks for the E3b-qualified T1j adapter.

No Java, no T1j, no subprocess: these cover the parts of the adapter that are
ours -- the coordinate transforms, the external cap semantics, and the dump
parser. The lockstep behaviour against the real engine is covered by the E3b
qualification run, whose evidence is committed under docs/superpowers/evidence/.
"""
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
        A.replay([], java="j", jar="j", classes="c")   # type: ignore[call-arg]


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
