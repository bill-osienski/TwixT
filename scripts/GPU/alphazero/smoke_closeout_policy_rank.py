"""One-shot smoke for the closeout-diagnostic policy-rank fix.

Plays a handful of self-play games with the iter_0129 checkpoint and reports
how many ranking blocks now have a non-null best_policy_rank. Before the fix
the answer was 0/N; after the fix it should be ~N/N.

Run:
    .venv/bin/python -m scripts.GPU.alphazero.smoke_closeout_policy_rank
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.mcts import MCTSConfig
    from scripts.GPU.alphazero.self_play import play_game
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator

    weights = "checkpoints/alphazero-v2-staged/model_iter_0129.safetensors"
    n_games = 5
    n_sims = 200
    max_moves = 280

    print(f"Loading network (hidden=128, blocks=6) and weights={weights}")
    network = create_network(hidden=128, n_blocks=6)
    network.load_weights(weights)
    evaluator = LocalGPUEvaluator(network)

    mcts_config = MCTSConfig(n_simulations=n_sims)

    n_blocks_total = 0
    n_blocks_pol_nonnull = 0
    n_blocks_vis_nonnull = 0
    n_diag_records = 0
    games_with_diag = 0

    t0 = time.perf_counter()
    for i in range(n_games):
        gt0 = time.perf_counter()
        game = play_game(
            evaluator,
            mcts_config=mcts_config,
            max_moves=max_moves,
            active_size=24,
            game_id=i,
        )
        diag = list(game.goal_completion_diagnostics or [])
        if diag:
            games_with_diag += 1
            n_diag_records += len(diag)
        for r in diag:
            for k in ("endpoint_completion_ranking", "distance_reducing_ranking"):
                blk = r.get(k)
                if blk is None:
                    continue
                n_blocks_total += 1
                if blk.get("best_policy_rank") is not None:
                    n_blocks_pol_nonnull += 1
                if blk.get("best_visit_rank") is not None:
                    n_blocks_vis_nonnull += 1
        winner = game.winner or "draw"
        print(
            f"  game {i+1}/{n_games}: winner={winner}, n_moves={game.n_moves}, "
            f"diag_records={len(diag)}, time={time.perf_counter()-gt0:.1f}s"
        )

    elapsed = time.perf_counter() - t0
    print()
    print(f"=== smoke result (elapsed {elapsed:.1f}s) ===")
    print(f"games_with_diagnostics:     {games_with_diag}/{n_games}")
    print(f"diag records (per-ply):     {n_diag_records}")
    print(f"ranking blocks total:       {n_blocks_total}")
    print(f"  with non-null policy_rank: {n_blocks_pol_nonnull}")
    print(f"  with non-null visit_rank:  {n_blocks_vis_nonnull}")
    if n_blocks_total == 0:
        print()
        print("WARN: no ranking blocks captured. Games may have ended before any "
              "closeout-window ply, or emit_threshold wasn't reached. Try more "
              "games or different seeds — this isn't necessarily a regression.")
        return 1
    pol_rate = 100.0 * n_blocks_pol_nonnull / n_blocks_total
    vis_rate = 100.0 * n_blocks_vis_nonnull / n_blocks_total
    print()
    print(f"non-null policy_rank rate: {pol_rate:.1f}%   "
          f"(expected ~100% after fix; was 0% before)")
    print(f"non-null visit_rank rate:  {vis_rate:.1f}%   "
          f"(unchanged by fix)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
