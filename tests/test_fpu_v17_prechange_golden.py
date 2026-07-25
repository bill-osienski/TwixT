"""v17 Task 1 -- the pre-change identity golden must reproduce exactly.

Captured from UNMODIFIED `mcts.py` BEFORE any v17 source edit, per plan Task 1
("Gate: no MCTS source edit before goldens exist") and design §2.2's required
identity proofs.

After Task 3 wires the v17 selection branch, these same assertions must still
hold with `fpu_shipped_policy_mass_reduction` defaulting to `None`, and Task 3
adds the `0.0`-path re-run of `cpu_search_signature`. A failure here after a
Task 2/3 edit means the exact-zero identity was broken.
"""
import json

from tests.fpu_v17_prechange_fixture import (
    BATCHING,
    FPU_TRACE_POINTS,
    cpu_search_signature,
    prechange_goldens,
    synthetic_selection_trace,
    v17_config,
)

from scripts.GPU.alphazero.mcts import MCTSConfig

GOLDEN_PATH = "tests/golden/fpu_v17_prechange_search.json"


def _golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def test_prechange_goldens_match():
    assert prechange_goldens() == _golden()


def test_prechange_goldens_reproduce_twice_byte_identically():
    """Plan Task 1: 'scientific-result goldens reproduce twice
    byte-identically'. Compares serialized bytes, not just objects."""
    dumps = [json.dumps(prechange_goldens(), sort_keys=True, indent=1)
             for _ in range(2)]
    assert dumps[0] == dumps[1]
    with open(GOLDEN_PATH) as f:
        assert dumps[0] + "\n" == f.read()


def test_golden_is_pinned_to_the_frozen_batching_triple():
    """Design §2.4. `stall_flush_sims` must be the explicitly derived 48, NOT
    the `MCTSConfig` standalone default of 16 -- that substitution is the exact
    silent-drift this pin exists to catch."""
    assert BATCHING == {"eval_batch_size": 14, "stall_flush_sims": 48,
                        "pending_virtual_visits": 8}
    assert MCTSConfig().stall_flush_sims == 16          # the value NOT used
    cfg = v17_config()
    assert [cfg.eval_batch_size, cfg.stall_flush_sims, cfg.pending_virtual_visits] \
        == _golden()["cpu_search"]["batching"] == [14, 48, 8]


def test_cpu_search_signature_is_non_vacuous():
    """A golden that would match anything proves nothing. The CPU search must
    actually have searched: a full visit budget, a real tree, and one callback
    per completed simulation."""
    sig = _golden()["cpu_search"]
    n_sims = sig["search"]["n_sims"]
    assert sig["search"]["root_visit_count"] == n_sims == 200
    assert sum(v for _m, v in sig["search"]["visits"]) == n_sims
    assert len(sig["tree"]) > 1                          # branched, not a chain
    assert [c[0] for c in sig["callbacks"]] == list(range(1, n_sims + 1))
    assert sig["selected_move"] is not None


def test_synthetic_trace_actually_discriminates_fpu():
    """The synthetic sweep is only a useful identity basis if `fpu_value`
    changes the selected move somewhere inside it -- i.e. it brackets the
    pinned tree's X/Y decision boundary rather than returning one constant."""
    trace = _golden()["synthetic_selection_trace"]
    assert len(trace) == len(FPU_TRACE_POINTS)
    chosen = [move for _fpu_hex, move in trace]
    assert len(set(chosen)) == 2, chosen
    # monotone: once the unvisited move wins it keeps winning as fpu_value rises
    assert chosen == sorted(chosen, key=lambda m: chosen.index(m))
    assert synthetic_selection_trace() == trace


def test_cpu_search_signature_alone_matches_golden_block():
    """Task 3 re-runs exactly this entry point on the `0.0` path."""
    assert cpu_search_signature() == _golden()["cpu_search"]
