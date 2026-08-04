"""Atlas reservoir producer -- design section 3 source protocol and section 2b seeding.

Exists because the shipped generator cannot satisfy the frozen per-game seed
identity: play_games derives each game RNG from a MASTER stream
(random.Random(rng.randint(...))), so game i's seed depends on every preceding
draw -- there is no start offset, a continuation block is unreachable, and the
emitted GameRecord carries neither index nor seed.

Here: game_seed = base_seed + game_idx, exactly. A block is fully determined by
(base_seed, start_index, n_games), and any single index reproduces independently
of the block it was produced in.

How a game is PLAYED is unchanged: same play_game, shipped settings, Dirichlet
noise ON, and start_player derived by the same leading draw play_games uses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .corpus_geometry import MAX_SEED_RANGE_GAMES, GameMeta
from .mcts import MCTSConfig
from .self_play import play_game

MANIFEST = "block_manifest.json"

# The frozen production settings a corpus block MUST have been generated under
# (design section 3). A block missing any of these is not a corpus block -- an
# active_size=6 smoke, a noise-off run, or wrong batching would otherwise pass
# every interval and seed check and become the corpus.
PRODUCTION_SETTINGS = {
    "active_size": 24,
    "n_simulations": 400,
    "max_moves": 280,
    "batching": [14, 48, 8],
    "add_noise": True,
}


def seed_for_index(base_seed: int, game_idx: int) -> int:
    """The frozen identity. No master stream, no offset arithmetic elsewhere."""
    return base_seed + game_idx


def game_meta_from_sidecar(d: Dict[str, Any]) -> GameMeta:
    return GameMeta(game_id=d["game_idx"], seed=d["seed"],
                    n_moves=d["n_moves"], start_player=d["start_player"])


def _git(args: Sequence[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _sha1_file(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    return hashlib.sha1(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Source provenance -- a PREFLIGHT, not a post-hoc record
# ---------------------------------------------------------------------------

def validate_source_provenance(porcelain: str, git_head: str,
                               checkpoint_path: str,
                               checkpoint_sha1: str) -> Dict[str, Any]:
    """PURE. Raises unless the source is clean and the checkpoint is identified.

    Pure so the dirty case can be CONSTRUCTED in a test rather than observed from
    ambient worktree state -- the v18 lesson, where a provenance test built its
    negative out of a dirty tree and therefore passed only while the tree was
    dirty, failing at the clean HEAD the protocol required.
    """
    if porcelain.strip():
        raise RuntimeError(
            "refusing to generate: the worktree is dirty. Source provenance must "
            "be clean BEFORE the run, not recorded as unclean after it.\n"
            f"{porcelain.strip()}")
    if not checkpoint_sha1 or len(checkpoint_sha1) != 40:
        raise ValueError(
            f"checkpoint identity missing or invalid for {checkpoint_path!r}: "
            f"sha1={checkpoint_sha1!r}")
    return {
        "git_head": git_head, "worktree_clean": True,
        "checkpoint_path": checkpoint_path, "checkpoint_sha1": checkpoint_sha1,
    }


def preflight_source_provenance(checkpoint_path: str) -> Dict[str, Any]:
    """MUST run BEFORE evaluator construction and before the first game.

    Checking afterwards means a dirty tree can consume an entire GPU reservoir
    run before anything rejects it.
    """
    return validate_source_provenance(
        porcelain=_git(["status", "--porcelain"]),
        git_head=_git(["rev-parse", "HEAD"]),
        checkpoint_path=checkpoint_path,
        checkpoint_sha1=_sha1_file(checkpoint_path),
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_block(evaluator, base_seed: int, start_index: int, n_games: int,
                   out_dir: Path, provenance: Dict[str, Any],
                   n_simulations: int = 400, max_moves: int = 280,
                   active_size: int = 24) -> List[Dict[str, Any]]:
    """Generate games [start_index, start_index + n_games) into a FRESH dir.

    `provenance` MUST come from `preflight_source_provenance`, run before the
    evaluator was built. This function does not compute it: source validity is a
    precondition of spending the run, not a fact discovered at the end of it.
    """
    if start_index < 0 or n_games <= 0:
        raise ValueError("start_index must be >= 0 and n_games > 0")
    if start_index + n_games > MAX_SEED_RANGE_GAMES:
        raise ValueError(
            f"block [{start_index}, {start_index + n_games}) exceeds the frozen "
            f"{MAX_SEED_RANGE_GAMES}-game seed range")
    out_dir = Path(out_dir)
    if (out_dir / MANIFEST).exists() or any(out_dir.glob("game_*.json")):
        raise FileExistsError(
            f"{out_dir} already holds a block. Each block writes its OWN "
            f"directory -- sharing one would overwrite the other's manifest "
            f"and games. Refusing to proceed.")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = MCTSConfig(n_simulations=n_simulations, eval_batch_size=14,
                     stall_flush_sims=48, pending_virtual_visits=8)
    rows: List[Dict[str, Any]] = []
    for game_idx in range(start_index, start_index + n_games):
        seed = seed_for_index(base_seed, game_idx)
        game_rng = random.Random(seed)
        # Same leading draw play_games consumes, in the same order, so a game is
        # shipped-identical GIVEN its RNG. Only the RNG's provenance changes.
        start_player = "red" if game_rng.random() < 0.5 else "black"
        rec = play_game(
            evaluator=evaluator, mcts_config=cfg, rng=game_rng,
            max_moves=max_moves, add_noise=True, active_size=active_size,
            start_player=start_player, game_id=game_idx,
        )
        row = {
            "game_idx": game_idx, "seed": seed, "start_player": start_player,
            "n_moves": rec.n_moves, "winner": rec.winner,
            "draw_reason": rec.draw_reason,
            "move_history": [list(m) for m in rec.move_history],
        }
        (out_dir / f"game_{game_idx:06d}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True))
        rows.append(row)

    (out_dir / MANIFEST).write_text(json.dumps({
        "base_seed": base_seed, "start_index": start_index, "n_games": n_games,
        "seed_range": [seed_for_index(base_seed, start_index),
                       seed_for_index(base_seed, start_index + n_games)],
        "n_simulations": n_simulations, "max_moves": max_moves,
        "active_size": active_size,
        "batching": [cfg.eval_batch_size, cfg.stall_flush_sims,
                     cfg.pending_virtual_visits],
        # Shipped generation keeps root Dirichlet noise ON. This is NOT the
        # atlas ladder's add_noise=False; the two must not be conflated.
        "add_noise": True,
        **provenance,      # preflighted BEFORE the run, never recomputed here
    }, indent=2, sort_keys=True))
    return rows


# ---------------------------------------------------------------------------
# Strict loading
# ---------------------------------------------------------------------------

def load_manifest(block_dir) -> Dict[str, Any]:
    man_path = Path(block_dir) / MANIFEST
    if not man_path.exists():
        raise ValueError(f"no {MANIFEST} in {block_dir}")
    return json.loads(man_path.read_text())


def assert_blocks_agree(pilot_manifest: Dict[str, Any],
                        cont_manifest: Dict[str, Any]) -> None:
    """Pilot and continuation must come from the SAME checkpoint and source.

    A continuation generated from a different checkpoint, or from a dirty tree,
    is a different reservoir wearing the same seed range.
    """
    for field in ("base_seed", "checkpoint_path", "checkpoint_sha1", "git_head"):
        if pilot_manifest.get(field) != cont_manifest.get(field):
            raise ValueError(
                f"pilot/continuation disagree on {field}: "
                f"{pilot_manifest.get(field)!r} vs {cont_manifest.get(field)!r}")
    for name, man in (("pilot", pilot_manifest), ("continuation", cont_manifest)):
        if not man.get("worktree_clean", False):
            raise ValueError(f"{name} block was generated from a dirty worktree")


def load_block(block_dir, base_seed: int, start_index: int, n_games: int,
               *, require_production: bool = True) -> List[GameMeta]:
    """Load a block, verifying the FROZEN identity and provenance before use.

    Validates, in order: manifest agreement; frozen production settings and clean
    provenance (unless require_production=False, for tiny-scale producer tests
    only -- the CLI never disables it); exact filenames with no duplicate index
    behind a second name; exact index coverage; and seed == base_seed + game_idx.

    Without all of these, a tampered, mixed, under-settings or partial directory
    silently becomes the corpus.
    """
    block_dir = Path(block_dir)
    man = load_manifest(block_dir)
    if (man["base_seed"], man["start_index"], man["n_games"]) != (
            base_seed, start_index, n_games):
        raise ValueError(
            f"manifest describes block base={man['base_seed']} "
            f"start={man['start_index']} n={man['n_games']}, requested "
            f"base={base_seed} start={start_index} n={n_games}")

    if require_production:
        for field, want in PRODUCTION_SETTINGS.items():
            got = man.get(field)
            if got != want:
                raise ValueError(
                    f"block was not generated under production settings: "
                    f"{field}={got!r}, frozen value is {want!r}")
        if man.get("worktree_clean") is not True:
            raise ValueError(
                f"block in {block_dir} was generated from a dirty worktree; "
                f"its source provenance is not reconstructible")
        sha1 = man.get("checkpoint_sha1")
        if not sha1 or len(sha1) != 40:
            raise ValueError(
                f"block in {block_dir} has missing or invalid checkpoint "
                f"identity: checkpoint_sha1={sha1!r}")

    expected = set(range(start_index, start_index + n_games))
    files = sorted(block_dir.glob("game_*.json"))
    want_names = {f"game_{i:06d}.json" for i in expected}
    got_names = {f.name for f in files}
    if got_names != want_names:
        raise ValueError(
            f"filename set mismatch: missing={sorted(want_names - got_names)} "
            f"unexpected={sorted(got_names - want_names)}")

    sides: Dict[int, Dict[str, Any]] = {}
    for f in files:
        d = json.loads(f.read_text())
        idx = d["game_idx"]
        if idx in sides:
            raise ValueError(f"duplicate game_idx {idx} across filenames")
        if f.name != f"game_{idx:06d}.json":
            raise ValueError(
                f"{f.name} declares game_idx {idx}; filename and index disagree")
        sides[idx] = d
    if set(sides) != expected:
        missing, extra = sorted(expected - set(sides)), sorted(set(sides) - expected)
        raise ValueError(f"index set mismatch: missing={missing} extra={extra}")

    metas = []
    for idx in sorted(expected):
        d = sides[idx]
        want = seed_for_index(base_seed, idx)
        if d["seed"] != want:
            raise ValueError(
                f"game {idx}: seed {d['seed']} != base_seed + game_idx ({want}); "
                f"the frozen replay seed identity is violated")
        metas.append(game_meta_from_sidecar(d))
    return metas


def main() -> int:
    ap = argparse.ArgumentParser(description="Atlas reservoir producer")
    ap.add_argument("--base-seed", type=int, required=True)
    ap.add_argument("--start-index", type=int, required=True)
    ap.add_argument("--n-games", type=int, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--simulations", type=int, default=400)
    ap.add_argument("--max-moves", type=int, default=280)
    args = ap.parse_args()

    # PREFLIGHT FIRST -- before the evaluator is built and before any game runs.
    # A dirty tree or an unidentifiable checkpoint must cost nothing.
    provenance = preflight_source_provenance(args.checkpoint)

    # One long-lived evaluator, shared across every game in the block. Rebuilding
    # a compiled evaluator per unit of work is the documented MLX trap.
    from .eval_runner import _default_evaluator_factory
    rows = generate_block(
        evaluator=_default_evaluator_factory(args.checkpoint),
        base_seed=args.base_seed, start_index=args.start_index,
        n_games=args.n_games, out_dir=Path(args.out_dir),
        provenance=provenance,
        n_simulations=args.simulations, max_moves=args.max_moves,
    )
    print(f"generated {len(rows)} games, indices "
          f"[{args.start_index}, {args.start_index + args.n_games})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
