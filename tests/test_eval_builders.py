import pytest

from scripts.GPU.alphazero.eval_checkpoint_match import build_match_tasks
from scripts.GPU.alphazero.eval_checkpoint_tournament import (
    build_tournament_tasks, short_id, resolve_checkpoint,
)


def test_match_balanced_colors():
    tasks = build_match_tasks("A.sft", "B.sft", games=4, base_seed=1000,
                              pairing_id="A_vs_B")
    reds = [t.red_checkpoint for t in tasks]
    assert reds == ["A.sft", "B.sft", "A.sft", "B.sft"]
    assert sum(1 for t in tasks if t.red_checkpoint == "A.sft") == 2


def test_match_seed_is_task_derived_and_stable():
    t1 = build_match_tasks("A", "B", games=4, base_seed=1000, pairing_id="p")
    t2 = build_match_tasks("A", "B", games=4, base_seed=1000, pairing_id="p")
    assert [t.seed for t in t1] == [t.seed for t in t2]
    assert [t.seed for t in t1] == [1000, 1001, 1002, 1003]


def test_match_rejects_too_few_games():
    with pytest.raises(ValueError):
        build_match_tasks("A", "B", games=1, base_seed=0, pairing_id="p")


def test_match_rejects_odd_games():
    # Odd count would give one model an extra red game -> color imbalance.
    with pytest.raises(ValueError, match="even"):
        build_match_tasks("A", "B", games=3, base_seed=0, pairing_id="p")


def test_tournament_flat_list_unique_task_ids():
    pairings = [("A", "B"), ("A", "C")]
    tasks = build_tournament_tasks(pairings, games=4, base_seed=500)
    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids)) == 8


def test_tournament_pairing_ids_and_grouping():
    pairings = [("A", "B"), ("A", "C")]
    tasks = build_tournament_tasks(pairings, games=2, base_seed=0)
    pids = {t.pairing_id for t in tasks}
    assert pids == {"A_vs_B", "A_vs_C"}


def test_tournament_seeds_independent_across_pairings():
    # Pairing 0 and pairing 1 must not share seeds (offset by stride).
    tasks = build_tournament_tasks([("A", "B"), ("A", "C")], games=2, base_seed=0)
    seeds = [t.seed for t in tasks]
    assert len(seeds) == len(set(seeds))


def test_short_id_from_path():
    assert short_id("checkpoints/x/model_iter_0419.safetensors") == "0419"
    assert short_id("0419") == "0419"


def test_resolve_checkpoint_short_id_uses_dir():
    path = resolve_checkpoint("0419", "checkpoints/alphazero-v2-staged")
    assert path == "checkpoints/alphazero-v2-staged/model_iter_0419.safetensors"


def test_resolve_checkpoint_passthrough_full_path():
    p = "checkpoints/x/model_iter_0419.safetensors"
    assert resolve_checkpoint(p, "ignored") == p
