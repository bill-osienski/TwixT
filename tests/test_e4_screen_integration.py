"""Fail-closed checks for the E4 integration wiring. No Java, no model, no seed.

The three checks this file exists for are the ones a passing run would otherwise
not distinguish from a broken one: the position the SEARCH jvm reconstructed, the
identity of the JDK, and the exact reflection count.
"""
import json

import pytest

from scripts.GPU.alphazero import e4_screen_integration as I
from scripts.GPU.alphazero import t1j_adapter as A
from scripts.GPU.alphazero.e4_screen_runner import AbortError, PHASE_BIND, PHASE_MOVE, PHASE_PRECONDITION
from scripts.GPU.alphazero.game.twixt_state import TwixtState

OPENING = [(11, 11), (12, 13), (13, 12), (10, 13), (12, 10), (14, 14)]
CLEAN_POST = ("POSTCOND no_throw=true windows=0 frames=0 headless=true prefs_ok=true "
              "refl_ok=true refl_n={n} failures=0")


def state_after(moves):
    s = TwixtState(active_size=24, to_move="red")
    for mv in moves:
        s = s.apply_move(tuple(mv))
    return s


def ply_state_for(state, moves, **override):
    """A PlyState that AGREES with `state`, unless an override breaks it."""
    pegs, bridges = A.our_snapshot(state)
    base = dict(ply=state.ply, next_player=A.PLAYER_TO_T1J[state.to_move],
                term_y=state.winner() == "red", term_x=state.winner() == "black",
                pegs=pegs, bridges=bridges,
                history=tuple(A.to_t1j(*m) for m in moves),
                legal={A.to_t1j(r, c) for (r, c) in state.legal_moves()})
    base.update(override)
    return A.PlyState(**base)


# --- the shared comparison -------------------------------------------------

def test_an_agreeing_ply_state_has_no_divergences():
    s = state_after(OPENING)
    assert I.compare_state(s, ply_state_for(s, OPENING), OPENING) == []


@pytest.mark.parametrize("field,value,label", [
    ("pegs", {"1,1,Y"}, "pegs"),
    ("bridges", {"1,1|2,3|Y"}, "bridges"),
    ("next_player", "X", "side to move"),
    ("ply", 99, "ply"),
    ("legal", {(0, 0)}, "legal set"),
    ("term_y", True, "terminal"),
    ("history", ((1, 1),), "history"),
])
def test_every_bound_observable_is_compared(field, value, label):
    s = state_after(OPENING)
    div = I.compare_state(s, ply_state_for(s, OPENING, **{field: value}), OPENING)
    assert div and any(label.split()[0] in d for d in div), div


# --- the SEARCH jvm's position is re-bound ---------------------------------

class _Ctx(I.IntegrationContext):
    def __init__(self, moves):
        super().__init__()
        self.reset("t", moves)


def fake_query(*, dump_state, dump_moves, refl_n=3, move=(15, 13), completed=True,
               depth=3, n_dumps=1, rc=0):
    rec = A.QueryRecord(q=1, requested_depth=depth, move=move, to_move="Y",
                        usealphabeta=True, current_max_ply=depth + 1,
                        completed_depth=depth if completed else depth - 1,
                        completed=completed, legal=True, null_sentinel=False,
                        move_nr=dump_state.ply, eval_regime="normal", elapsed_us=1)
    dumps = [ply_state_for(dump_state, dump_moves)] * n_dumps
    return lambda moves, **kw: ([rec], dumps, rc, CLEAN_POST.format(n=refl_n))


def agent_for(state, moves, **kw):
    ctx = _Ctx(moves)
    rt = I.T1jRuntime(java="j", jar="j", classes="c", ply_cap=280)
    return I.T1jAgent(runtime=rt, ctx=ctx, depth=3, colour=state.to_move,
                      _query=fake_query(dump_state=state, dump_moves=moves, **kw))


def test_the_agent_accepts_a_move_when_the_searched_position_agrees():
    s = state_after(OPENING)
    a = agent_for(s, OPENING)
    assert a(s) == (15, 13)
    assert a.ctx.stats["t"]["searched_binds"] == 1


def test_a_DIFFERENT_searched_position_is_refused():
    """The search jvm rebuilt something else. The replay binder would not see it."""
    s = state_after(OPENING)
    other = state_after(OPENING[:-1] + [(20, 20)])       # same ply, different position
    ctx = _Ctx(OPENING)
    rt = I.T1jRuntime(java="j", jar="j", classes="c", ply_cap=280)
    a = I.T1jAgent(runtime=rt, ctx=ctx, depth=3, colour=s.to_move,
                   _query=fake_query(dump_state=other, dump_moves=OPENING[:-1] + [(20, 20)]))
    with pytest.raises(AbortError) as e:
        a(s)
    assert e.value.phase == PHASE_MOVE
    assert "SEARCH jvm reconstructed a different position" in e.value.message
    assert ctx.stats["t"]["searched_binds"] == 0


def test_more_than_one_searched_dump_is_refused():
    s = state_after(OPENING)
    with pytest.raises(AbortError):
        agent_for(s, OPENING, n_dumps=2)(s)


def test_an_incomplete_depth_is_refused():
    s = state_after(OPENING)
    with pytest.raises(AbortError):
        agent_for(s, OPENING, completed=False)(s)


def test_a_move_illegal_in_our_engine_is_refused():
    s = state_after(OPENING)
    with pytest.raises(AbortError):
        agent_for(s, OPENING, move=(11, 11))(s)          # already occupied


# --- the reflection COUNT is enforced --------------------------------------

def test_the_exact_reflection_count_is_required():
    for n in (0, 1, 2, 4, 6):
        with pytest.raises(AbortError) as e:
            I.check_postcond(CLEAN_POST.format(n=n), expected_refl=3, where="w", phase=PHASE_MOVE)
        assert "reflective accesses" in e.value.message
    I.check_postcond(CLEAN_POST.format(n=3), expected_refl=3, where="w", phase=PHASE_MOVE)
    I.check_postcond(CLEAN_POST.format(n=1), expected_refl=1, where="w", phase=PHASE_BIND)


def test_a_wrong_reflection_count_stops_the_agent():
    s = state_after(OPENING)
    with pytest.raises(AbortError) as e:
        agent_for(s, OPENING, refl_n=4)(s)               # authorized NAMES, wrong COUNT
    assert "expected exactly 3" in e.value.message


def test_a_dirty_postcond_is_refused_even_with_the_right_count():
    dirty = CLEAN_POST.format(n=3).replace("prefs_ok=true", "prefs_ok=false")
    with pytest.raises(AbortError):
        I.check_postcond(dirty, expected_refl=3, where="w", phase=PHASE_MOVE)


def test_more_than_one_postcond_line_is_refused():
    two = CLEAN_POST.format(n=3) + "\n" + CLEAN_POST.format(n=3)
    with pytest.raises(AbortError):
        I.check_postcond(two, expected_refl=3, where="w", phase=PHASE_MOVE)


# --- the JDK is bound by identity, not by presence --------------------------

def test_jdk_identity_accepts_matching_components(tmp_path):
    pinned = {}
    for rel, body in (("bin/java", b"J"), ("release", b"R")):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
        pinned[rel] = I._sha256(str(p))
    assert I.verify_jdk_identity(str(tmp_path), pinned) == pinned


def test_a_WRONG_jdk_is_refused_even_at_the_right_path(tmp_path):
    p = tmp_path / "bin" / "java"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"some other jvm")
    with pytest.raises(AbortError) as e:
        I.verify_jdk_identity(str(tmp_path), {"bin/java": "0" * 64})
    assert e.value.phase == PHASE_PRECONDITION and "sha256" in e.value.message


def test_a_MISSING_jdk_component_is_refused(tmp_path):
    with pytest.raises(AbortError) as e:
        I.verify_jdk_identity(str(tmp_path), {"lib/modules": "0" * 64})
    assert "missing" in e.value.message


def test_all_four_components_are_pinned():
    assert set(I.PINNED_JDK) == {"bin/java", "bin/javac", "lib/modules", "release"}
    assert all(len(v) == 64 for v in I.PINNED_JDK.values())


# --- per-task stats never reset --------------------------------------------

def test_stats_are_per_task_and_never_reset():
    ctx = I.IntegrationContext()
    for tid in ("a", "b", "c"):
        ctx.reset(tid, OPENING)
        ctx.bump("binds"); ctx.bump("binds"); ctx.bump("t1j_queries")
    assert set(ctx.stats) == {"a", "b", "c"}
    assert ctx.total("binds") == 6 and ctx.total("t1j_queries") == 3
    assert all(v["binds"] == 2 for v in ctx.stats.values())
