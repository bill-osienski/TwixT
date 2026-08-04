"""Atlas Stage 2 -- reservoir producer: seeding, preflight, block validation."""
import json
from pathlib import Path

import pytest

from scripts.GPU.alphazero.corpus_geometry import GameMeta
from scripts.GPU.alphazero.generate_atlas_reservoir import (
    game_meta_from_sidecar,
    generate_block,
    load_block,
    seed_for_index,
    validate_source_provenance,
)

from tests.eval_fakes import FakeEvaluator

BASE = 20400000

FAKE_PROV = {
    "git_head": "deadbeef" * 5, "worktree_clean": True,
    "checkpoint_path": "fake://evaluator", "checkpoint_sha1": "0" * 40,
}

PROD_FIELDS = {"active_size": 24, "n_simulations": 400, "max_moves": 280,
               "batching": [14, 48, 8]}


def _block(d, start_index, n_games, n_moves_cfg=6, provenance=None):
    return generate_block(
        evaluator=FakeEvaluator(value=0.0), base_seed=BASE,
        start_index=start_index, n_games=n_games, out_dir=Path(d),
        provenance=dict(provenance or FAKE_PROV),
        n_simulations=8, max_moves=n_moves_cfg, active_size=6,
    )


def _fake_production(d):
    """Rewrite a tiny block's manifest to claim production settings, so the
    NON-settings checks can be exercised in isolation."""
    man_path = Path(d) / "block_manifest.json"
    man = json.loads(man_path.read_text())
    man.update(PROD_FIELDS)
    man_path.write_text(json.dumps(man))


# ------------------------------------------------------------- provenance --

def test_validate_source_provenance_rejects_a_CONSTRUCTED_dirty_tree():
    """The dirty case is CONSTRUCTED, never observed from ambient worktree
    state. v18's provenance test built its negative out of a dirty tree and so
    passed only while the tree was dirty -- failing at the clean HEAD the
    protocol actually required."""
    with pytest.raises(RuntimeError, match="dirty"):
        validate_source_provenance(
            porcelain=" M scripts/GPU/alphazero/mcts.py\n",
            git_head="a" * 40, checkpoint_path="ck", checkpoint_sha1="0" * 40)


def test_validate_source_provenance_requires_checkpoint_identity():
    with pytest.raises(ValueError, match="checkpoint identity"):
        validate_source_provenance(porcelain="", git_head="a" * 40,
                                   checkpoint_path="missing.safetensors",
                                   checkpoint_sha1="")


def test_validate_source_provenance_passes_when_clean():
    got = validate_source_provenance(porcelain="", git_head="a" * 40,
                                     checkpoint_path="ck", checkpoint_sha1="0" * 40)
    assert got["worktree_clean"] is True and got["checkpoint_sha1"] == "0" * 40


# ----------------------------------------------------------------- seeding --

def test_seed_is_base_plus_index_exactly():
    assert seed_for_index(BASE, 0) == BASE
    assert seed_for_index(BASE, 24) == BASE + 24
    assert seed_for_index(BASE, 479) == BASE + 479


def test_continuation_block_is_disjoint_and_offset(tmp_path):
    pilot = _block(tmp_path / "pilot", 0, 3)
    cont = _block(tmp_path / "cont", 3, 3)
    assert [g["game_idx"] for g in pilot] == [0, 1, 2]
    assert [g["seed"] for g in cont] == [BASE + 3, BASE + 4, BASE + 5]
    assert not ({g["seed"] for g in pilot} & {g["seed"] for g in cont})


def test_a_single_index_reproduces_exactly(tmp_path):
    """The whole point: index 4 is the same game whether produced in a block
    starting at 0 or a continuation starting at 4."""
    from_zero = _block(tmp_path / "x", 0, 6)[4]
    from_four = _block(tmp_path / "y", 4, 1)[0]
    assert from_zero["seed"] == from_four["seed"] == BASE + 4
    assert from_zero["start_player"] == from_four["start_player"]
    assert from_zero["n_moves"] == from_four["n_moves"]
    assert from_zero["move_history"] == from_four["move_history"]


def test_manifest_records_the_block_and_the_checkpoint(tmp_path):
    _block(tmp_path / "m", 24, 2)
    man = json.loads((tmp_path / "m" / "block_manifest.json").read_text())
    assert man["base_seed"] == BASE
    assert man["start_index"] == 24 and man["n_games"] == 2
    assert man["seed_range"] == [BASE + 24, BASE + 26]
    assert man["add_noise"] is True          # shipped generation keeps noise ON
    assert man["checkpoint_path"] == "fake://evaluator"
    assert man["checkpoint_sha1"] == "0" * 40
    assert man["worktree_clean"] is True


def test_refuses_a_directory_that_already_holds_a_block(tmp_path):
    _block(tmp_path / "dup", 0, 2)
    with pytest.raises(FileExistsError):
        _block(tmp_path / "dup", 0, 2)


def test_block_may_not_exceed_the_frozen_seed_range(tmp_path):
    with pytest.raises(ValueError):
        _block(tmp_path / "over", 470, 20)      # 470 + 20 > 480


# ------------------------------------------------------------ load_block ----

def test_load_block_validates_the_frozen_seed_identity(tmp_path):
    d = tmp_path / "v"
    _block(d, 0, 3)
    metas = load_block(d, base_seed=BASE, start_index=0, n_games=3,
                       require_production=False)
    assert [m.game_id for m in metas] == [0, 1, 2]
    assert isinstance(metas[0], GameMeta)

    side = json.loads((d / "game_000001.json").read_text())
    side["seed"] = BASE + 999
    (d / "game_000001.json").write_text(json.dumps(side))
    with pytest.raises(ValueError, match="seed"):
        load_block(d, base_seed=BASE, start_index=0, n_games=3,
                   require_production=False)


def test_load_block_rejects_gaps_and_wrong_blocks(tmp_path):
    d = tmp_path / "g"
    _block(d, 0, 3)
    (d / "game_000001.json").unlink()                       # gap
    with pytest.raises(ValueError, match="filename"):
        load_block(d, base_seed=BASE, start_index=0, n_games=3,
                   require_production=False)

    d2 = tmp_path / "e"
    _block(d2, 0, 3)
    with pytest.raises(ValueError, match="manifest"):        # wrong block requested
        load_block(d2, base_seed=BASE, start_index=24, n_games=3,
                   require_production=False)


def test_load_block_rejects_a_block_from_a_dirty_tree(tmp_path):
    """Dirty provenance is injected via the manifest, not produced by dirtying
    the worktree."""
    d = tmp_path / "dirty"
    _block(d, 0, 2, provenance={**FAKE_PROV, "worktree_clean": False})
    _fake_production(d)
    with pytest.raises(ValueError, match="dirty worktree"):
        load_block(d, base_seed=BASE, start_index=0, n_games=2)


def test_load_block_rejects_missing_checkpoint_identity(tmp_path):
    d = tmp_path / "nock"
    _block(d, 0, 2, provenance={**FAKE_PROV, "checkpoint_sha1": ""})
    _fake_production(d)
    with pytest.raises(ValueError, match="checkpoint identity"):
        load_block(d, base_seed=BASE, start_index=0, n_games=2)


def test_load_block_rejects_a_non_production_block(tmp_path):
    """A tiny smoke block passes every interval and seed check. Only the frozen
    production settings distinguish it from a corpus block."""
    d = tmp_path / "smoke"
    _block(d, 0, 2)                       # active_size=6, 8 sims, max_moves=6
    with pytest.raises(ValueError, match="production settings"):
        load_block(d, base_seed=BASE, start_index=0, n_games=2)


def test_load_block_rejects_a_duplicate_index_behind_another_filename(tmp_path):
    d = tmp_path / "dup2"
    _block(d, 0, 3)
    side = json.loads((d / "game_000002.json").read_text())
    side["game_idx"] = 1                  # same index, different filename
    (d / "game_000002.json").write_text(json.dumps(side))
    with pytest.raises(ValueError, match="disagree|duplicate"):
        load_block(d, base_seed=BASE, start_index=0, n_games=3,
                   require_production=False)


def test_sidecar_carries_everything_GameMeta_needs(tmp_path):
    _block(tmp_path / "s", 0, 2)
    side = json.loads((tmp_path / "s" / "game_000000.json").read_text())
    for k in ("game_idx", "seed", "start_player", "n_moves"):
        assert k in side
    meta = game_meta_from_sidecar(side)
    assert isinstance(meta, GameMeta)
    assert meta.game_id == 0 and meta.seed == BASE
