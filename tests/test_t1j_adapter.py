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


DUMP = """SIZE x=24 y=24
CAP 280
PLY 0 mover=- move=- next=Y moveNr=0 termY=false termX=false pegs=0 bridges=0
  PEGS
  BRIDGES
  HIST
  LEGAL 11
PLY 1 mover=Y move=10,10 next=X moveNr=1 termY=false termX=true pegs=1 bridges=0
  PEGS 10,10,Y
  BRIDGES 1,1|2,3|Y
  HIST 10,10
  LEGAL 01
"""


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
            f"PLY 0 next=Y moveNr=0 termY={ty} termX={tx}\n  PEGS\n  BRIDGES\n  HIST\n  LEGAL 0\n"
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
