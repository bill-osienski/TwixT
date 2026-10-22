"""Training replay capture -- the MINIMAL SERIAL DRIVER (spec 2026-10-11, serial path only).

Plays the scheduled games one after another exactly as train()'s sequential loop does -- game_rng =
random.Random(seed); start_player = "red" if game_rng.random() < 0.5 else "black"; play_game(evaluator,
mcts_config=..., rng=game_rng, max_moves=280, add_noise=True, active_size=24, start_player, game_id) -- with a
DecisionCapture riding in play_game, and publishes the phase through training_replay_capture.PhaseWriter.

Model identity: the named-tensors/1 digest of the LIVE network's parameters() (every weight and every BatchNorm
running statistic) is bound in the context before the first game and must be unchanged after the last, or the phase
is not finalized. The evaluator must serve that very network object.

play_game's keyword defaults are train()'s; the only other keywords allowed are the two goal-completion
DIAGNOSTIC switches (DIAGNOSTIC_FLAGS), recorded in the context. They cost most of a game's CPU time with a cheap
evaluator and feed only the refused closeout features; that they leave play unchanged is not asserted here.

Ordinary self-play only (one model for both colours, no learner filter); V1 refuses mirror augmentation, the
position cap, resign, adjudication, the closeout tie-break and forced visits (play_game's capture guard). Seeds are
supplied by the caller: seed accounting belongs to the card that authorizes a generation run. Nothing runs on import.
"""
from __future__ import annotations

import dataclasses
import hashlib
import platform
import random
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from . import self_play as sp
from . import training_replay_capture as C

ROOT_FILES = ("scripts/GPU/alphazero/training_replay_capture.py", "scripts/GPU/alphazero/training_capture_serial.py",
              "scripts/GPU/alphazero/self_play.py", "scripts/GPU/alphazero/mcts.py",
              "scripts/GPU/alphazero/game/twixt_state.py", "scripts/GPU/alphazero/network.py",
              "scripts/GPU/alphazero/local_evaluator.py")
DIAGNOSTIC_FLAGS = ("goal_completion_emit_enabled", "goal_completion_record_enabled")


def network_state_digest(network) -> str:
    """named-tensors/1 over network.parameters(): weights AND BatchNorm running statistics (MLX keeps those in
    parameters(); frozen or not)."""
    from mlx.utils import tree_flatten
    tensors = {k: np.array(v) for k, v in tree_flatten(network.parameters())}
    if not any(k.endswith("running_mean") for k in tensors):
        raise C.CaptureError("the network exposes no BatchNorm running statistics to hash")
    return C.model_state_digest(tensors)


def start_player_for(seed: int) -> str:
    """train()'s start-player draw, made on a COPY of the game's stream (the real stream draws it again)."""
    return "red" if random.Random(seed).random() < 0.5 else "black"


def code_identity(root) -> Dict[str, str]:
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in ROOT_FILES}


def build_context(*, run_id: str, phase_id: str, seeds: Mapping[int, int], model_id: str, model_digest: str,
                  model_source: Mapping[str, Any], mcts_config, code: Mapping[str, str],
                  game_bytes: int = C.ENVELOPE_MAX_BYTES, phase_bytes: int,
                  play_kwargs: Optional[Mapping[str, bool]] = None) -> dict:
    ids = list(seeds)
    if any(type(s) is not int or s < 0 for s in seeds.values()):
        raise C.CaptureError("seeds must be non-negative ints")
    games = {str(g): {"seed": seeds[g], "seed_use": "train()-serial: random.Random(seed), one start-player draw",
                      "start_player": start_player_for(seeds[g]), "learner_player": None,
                      "models_by_colour": {"red": model_id, "black": model_id}} for g in ids}
    return {"schema": C.SCHEMA, "run_id": run_id, "phase_id": phase_id, "source_kind": "native_self_play",
            "execution_mode": "serial", "scheduled_game_ids": ids, "games": games,
            "rules": {"board_size": C.BOARD, "active_size": C.BOARD, "automatic_links": True, "link_removal": False,
                      "own_crossing_forbidden": True, "opponent_crossing_forbidden": True, "swap": False,
                      "max_plies": C.CAP_PLIES, "start": "empty"},
            "configuration": {"n_simulations": mcts_config.n_simulations, "mcts": dataclasses.asdict(mcts_config),
                              "add_noise": True, "mirror_prob": sp._MIRROR_PROB, "position_cap": None,
                              "play_kwargs": dict(play_kwargs or {})},
            "models": {model_id: {"digest_version": C.MODEL_DIGEST_VERSION, "state_digest": model_digest,
                                  "source": dict(model_source)}},
            "code": dict(code),
            "runtime": {"python": platform.python_version(), "platform": platform.platform()},
            "limits": {"game_bytes": game_bytes, "phase_bytes": phase_bytes}}


def run_serial_capture(*, evaluator, network, capture_dir: str, context: dict, mcts_config,
                       play_game=None) -> Dict[str, Any]:
    """Play and capture every scheduled game in order; finalize only if the model is unchanged.
    Returns the GameRecords (the learner rows play_game produced, unchanged by capture) and the phase stats."""
    play_game = play_game or sp.play_game
    if getattr(evaluator, "network", None) is not network:
        raise C.CaptureError("the evaluator does not serve the network whose identity is bound")
    if sp._MIRROR_PROB != 0.0:
        raise C.CaptureError(f"mirror augmentation is on (_MIRROR_PROB {sp._MIRROR_PROB}); V1 needs 0")
    (model_id, model), = context["models"].items()
    if network_state_digest(network) != model["state_digest"]:
        raise C.CaptureError("the live network does not match the context's model digest")
    conf = context["configuration"]
    if conf["n_simulations"] != mcts_config.n_simulations:
        raise C.CaptureError("the context's n_simulations is not the search configuration's")
    if conf.get("mcts") != dataclasses.asdict(mcts_config):
        raise C.CaptureError("the context's recorded MCTS configuration is not the search configuration")
    if (conf.get("add_noise"), conf.get("mirror_prob"), conf.get("position_cap")) != (True, 0.0, None):
        raise C.CaptureError("the context records noise, mirror or position-cap settings V1 does not run")
    forced = [f for f in ("closeout_td1_visit_forcing_enabled", "closeout_selection_tiebreak_enabled")
              if getattr(mcts_config, f)]
    if forced:
        raise C.CaptureError(f"V1 refuses forced visits and the closeout tie-break: {forced}")
    extra = dict(conf.get("play_kwargs") or {})
    if set(extra) - set(DIAGNOSTIC_FLAGS) or any(type(v) is not bool for v in extra.values()):
        raise C.CaptureError(f"play_kwargs may only switch {DIAGNOSTIC_FLAGS}, got {extra}")
    writer = C.PhaseWriter(capture_dir, context)
    games: List[Any] = []
    for gid in writer.context["scheduled_game_ids"]:
        g = writer.context["games"][str(gid)]
        game_rng = random.Random(g["seed"])
        start_player = "red" if game_rng.random() < 0.5 else "black"        # train()'s draw, on the real stream
        if start_player != g["start_player"]:
            raise C.CaptureError(f"game {gid}: start player {start_player} is not the context's")
        cap = C.DecisionCapture(context=writer.context, context_sha256=writer.context_sha256, game_id=gid,
                                game_rng=game_rng)
        games.append(play_game(evaluator, mcts_config=mcts_config, rng=game_rng, max_moves=C.CAP_PLIES,
                               add_noise=True, active_size=C.BOARD, start_player=start_player, game_id=gid,
                               training_capture=cap, **extra))
        writer.publish_game(cap.envelope)
    stats = writer.finalize({model_id: network_state_digest(network)})
    return {"games": games, "stats": stats, "context_sha256": writer.context_sha256}
