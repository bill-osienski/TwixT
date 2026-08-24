"""The seeded reference agent for G3 — our MLX side of an external match.

Pilot card: docs/superpowers/2026-08-22-twixtbot-anchor-pilot-card.md.
PREPARATION ONLY. Constructing an agent plays no game and consumes no seed.

WHY THIS MODULE EXISTS. The harness previously took `reference_fn` as an
unconstrained callable and merely *recorded* `task["seed"]`. Nothing bound the
scheduled seed, the checkpoint, 400 simulations, batch size 14, stall-flush 48 or
the opening-temperature readout to the moves our side actually played. A callable
that ignored all of them would have satisfied the old signature silently.

Everything here is taken from `eval_runner.play_eval_game`, which is the frozen
research evaluation path used for the `0379` benchmarks, so the reference plays
in G3 exactly as it plays in our own matches:

* search RNG   `random.Random(seed ^ 0xA5A5A5)` as red, `^ 0x5A5A5A` as black
* readout RNG  `random.Random(seed ^ 0xC3C3C3)` as red, `^ 0x3C3C3C` as black
  -- a SEPARATE stream, because MCTS shares one `self.rng` across prior shuffle,
  PUCT tie-break and readout, so drawing readout numbers from it would change the
  generator state entering every later search.
* search       `search_with_root(state, add_noise=False)`
* readout      `eval_readout.select(...)`, never `mcts.select_move`

One agent instance per game. It is stateful by design: the two RNG streams must
advance across the game exactly as they do in `play_eval_game`.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from .twixtbot_g3_schedule import OUR_SETTINGS, REFERENCE_CHECKPOINTS


class ReferenceError(Exception):
    """The reference agent could not be built or produced an unusable move."""


def eval_config():
    """The frozen research evaluation configuration, as an EvalConfig."""
    from .eval_runner import EvalConfig

    cfg = EvalConfig(
        board_size=24,
        mcts_sims=OUR_SETTINGS["mcts_sims"],
        mcts_eval_batch_size=OUR_SETTINGS["mcts_eval_batch_size"],
        mcts_stall_flush_sims=OUR_SETTINGS["mcts_stall_flush_sims"],
        selection_mode=OUR_SETTINGS["selection_mode"],
        opening_temp_plies=OUR_SETTINGS["opening_temp_plies"],
        temp_high=OUR_SETTINGS["temp_high"],
        temp_low=OUR_SETTINGS["temp_low"],
        max_moves=OUR_SETTINGS["max_moves"],
    )
    for field, want in (("mcts_sims", 400), ("mcts_eval_batch_size", 14),
                        ("mcts_stall_flush_sims", 48), ("max_moves", 280)):
        if getattr(cfg, field) != want:
            raise ReferenceError(f"{field} is {getattr(cfg, field)}, expected {want}")
    return cfg


def load_reference_evaluator(reference: str, repo_root: str):
    """Load a pinned reference checkpoint into an evaluator. Verifies the SHA-1."""
    import hashlib
    import os

    if reference not in REFERENCE_CHECKPOINTS:
        raise ReferenceError(f"unknown reference {reference!r}")
    meta = REFERENCE_CHECKPOINTS[reference]
    path = os.path.join(repo_root, meta["path"])
    if not os.path.isfile(path):
        raise ReferenceError(f"checkpoint missing: {path}")

    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != meta["sha1"]:
        raise ReferenceError(
            f"{reference} sha1 {h.hexdigest()} != pinned {meta['sha1']}"
        )

    from .local_evaluator import LocalGPUEvaluator
    from .probe_eval import load_network_for_scoring

    network, *_ = load_network_for_scoring(path)
    network.eval()
    return LocalGPUEvaluator(network)


class SeededReferenceAgent:
    """One game's worth of our reference play, bound to a scheduled seed.

    Stateful on purpose: both RNG streams advance across the game exactly as in
    `eval_runner.play_eval_game`. One instance per game, never shared.
    """

    #: The XOR masks from eval_runner.play_eval_game. Not parameters.
    SEARCH_MASK = {"red": 0xA5A5A5, "black": 0x5A5A5A}
    READOUT_MASK = {"red": 0xC3C3C3, "black": 0x3C3C3C}

    def __init__(self, *, evaluator, colour: str, seed: int, config=None):
        import random

        from . import eval_readout
        from .eval_runner import cfg_from, readout_from_eval_config
        from .mcts import MCTS

        if colour not in ("red", "black"):
            raise ReferenceError(f"colour must be red or black, got {colour!r}")
        self.colour = colour
        self.seed = int(seed)
        self.config = config or eval_config()
        self._eval_readout = eval_readout

        self.mcts = MCTS(evaluator, cfg_from(self.config),
                         random.Random(self.seed ^ self.SEARCH_MASK[colour]))
        self.readout_rng = random.Random(self.seed ^ self.READOUT_MASK[colour])
        self.readout = readout_from_eval_config(self.config)
        self.moves_made = 0

    def __call__(self, state) -> Tuple[int, int]:
        """One move, by the frozen research path. Raises on anything unusable."""
        from .eval_runner import root_child_stats, validate_ply

        if state.to_move != self.colour:
            raise ReferenceError(
                f"asked to move as {self.colour} but {state.to_move} is to move"
            )
        counts, root_value, root = self.mcts.search_with_root(state, add_noise=False)
        top2 = self._eval_readout.top_two(root_child_stats(counts, root))
        validate_ply(state.ply, self.config.mcts_sims, root.visit_count, root_value, top2)
        move, _overrode = self._eval_readout.select(
            counts, state.ply, self.readout, self.readout_rng, top2=top2
        )
        self.moves_made += 1
        return (int(move[0]), int(move[1]))


def build_reference_agent(*, task: dict, evaluator, colour: str, config=None) -> SeededReferenceAgent:
    """The one construction path. Binds the SCHEDULED seed, not an arbitrary one."""
    if "seed" not in task:
        raise ReferenceError("task carries no seed")
    return SeededReferenceAgent(
        evaluator=evaluator, colour=colour, seed=task["seed"], config=config
    )
