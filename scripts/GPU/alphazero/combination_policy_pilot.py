"""THE COMBINATION POLICY PILOT -- the throughput check, the corpus generation, the CPU training with its check, and
the build's CPU timing. BUILT with its gates closed; nothing here has run.

Card: `docs/superpowers/2026-10-28-combination-policy-pilot-card.md`, pinned below by sha256, with the four frozen
modules it reuses. Nothing runs on import, and importing this module loads no MLX. Every subcommand refuses unless
MLX_ENABLE_TF32 is exactly "0" and no `mlx` module is loaded when it starts (spawned workers inherit the setting and
report it); `run` then refuses unless its kind's gate is open (read at call time) and the earlier steps' committed
records admit it.

THE STEPS (card §5; each a separate approval, each in ONE fresh process under its wrapper's hard ceiling):
* `run throughput`: 144 slots on a registered block, n_workers 18, arbiter cap 14 rows per call. Its games are never
  used: no rows are written. ALL 144 must finish inside the 15-minute ceiling or the run is VOID and the pilot is
  closed. Records games/h, the arbiter's mean rows per call and the generation ceiling (1.5x the projection, <= 6 h).
* `run generate`: 1,000 slots on its own registered block; each game's rows go to one create-only file OUTSIDE the
  repository (CACHE_DIR), named with its sha256 in the committed games.jsonl. Any incomplete run is VOID.
Both: slot g plays seed plan.base + g (combination_selfplay); a `game_start` event (slot, worker) is logged BEFORE
each game and the final claim count after the last; background_compute is read before loading and after every game
(BOINC detected, or a failed check, is a VOID); eval mode at load and at the end.
* `run train`: weights-only calib020_0001 (SHA-1 checked), on the CPU; arm R installed with init_seed 20261028 and
  checked bit-equal to the parent; the corpus verified file by file and target by target; then card §4's one
  objective -- the MIRROR-AVERAGED policy, log p_bar = logaddexp(log p(s), log p(Ms) mapped back) - log 2 -- for the
  pre-fit KL, the loss and the held-out check; 3,000 Adam updates of batch 64 in the order from PCG64(20261029); then
  card §5's STOP rules in order. The verdict is ELIGIBLE (the screen may be proposed) or STOP with its reason.
* `time`: the build's CPU timing (card §5 step 1) -- 20 synchronized updates at the real architecture on synthetic
  inputs with random weights, no checkpoint -- and the training ceiling it prices.

Any failure, exception or SIGTERM inside the record boundary is a VOID with record.json written by the script; a
STOP rule is a COMPLETED run with verdict STOP. One shot each; no retry, top-up or extension.

    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.combination_policy_pilot run throughput|generate|train --checkpoint <abs path>
    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.combination_policy_pilot time
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import os
import pathlib
import platform
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import combination_policy_generate_authorization as GENERATE_GATE
from . import combination_policy_throughput_authorization as THROUGHPUT_GATE
from . import combination_policy_train_authorization as TRAIN_GATE
from . import combination_selfplay as CS
from . import halving_policy_target as T
from . import mirror_halving_combination as K
from . import search_distillation_d2_measure as M

ROOT = M.ROOT
RECORD_KIND = "combination-policy-pilot-record/1"
CARD = "docs/superpowers/2026-10-28-combination-policy-pilot-card.md"
CARD_SHA256 = "237424bd90bc8273ebab246fe87f19188f2e264f347ccd002352e930a5a1e389"
SETTING, SETTING_VALUE = "MLX_ENABLE_TF32", "0"
VERSIONS = {"mlx": "0.30.3", "numpy": "2.4.1", "python": "3.14.9"}
#: the frozen components, imported and never modified (card §3)
PINNED = {"scripts/GPU/alphazero/mirror_halving_combination.py":
          "907ce4342f95c28f30f71dfaf50406994a93cfaf70929c89f0a1c0bedca1a176",
          "scripts/GPU/alphazero/mirror_averaging.py": "1f23ac6adeb6ea0a33b7fe5500edeb2c9929a9781089106d5925a99817340aef",
          "scripts/GPU/alphazero/root_sequential_halving.py":
          "0bec4ef1914b25541f5247c5dd6c0002cea171b2902c629e52cfc034e9402790",
          "scripts/GPU/alphazero/policy_head_capacity.py":
          "beca0ec7e182b010185473b2550c1edd2a6245dab2b6b4d9b3222db2b4750476"}
CACHE_DIR = pathlib.Path.home() / "twixt-cache" / "combination-policy-pilot"     # the corpus: outside the repo
N_WORKERS, MAX_BATCH_ROWS = 18, 14              # entry 20's measured starting settings; not established optima here
KINDS = {"throughput": {"slots": 144, "keep_rows": False, "gate": (THROUGHPUT_GATE, "COMBINATION_POLICY_THROUGHPUT_AUTHORIZED"),
                        "evidence": "docs/superpowers/evidence/combination-policy-throughput",
                        "plan": "docs/superpowers/evidence/combination-policy-throughput-run/plan.json"},
         "generate": {"slots": 1000, "keep_rows": True, "gate": (GENERATE_GATE, "COMBINATION_POLICY_GENERATE_AUTHORIZED"),
                      "evidence": "docs/superpowers/evidence/combination-policy-generate",
                      "plan": "docs/superpowers/evidence/combination-policy-generate-run/plan.json"},
         "train": {"gate": (TRAIN_GATE, "COMBINATION_POLICY_TRAIN_AUTHORIZED"),
                   "evidence": "docs/superpowers/evidence/combination-policy-train"}}
TIMING_EVIDENCE = "docs/superpowers/evidence/combination-policy-timing"
#: what each step needs committed first: (evidence dir, record field, value)
ADMISSION = {"throughput": [(TIMING_EVIDENCE, "status", "COMPLETED")],
             "generate": [(KINDS["throughput"]["evidence"], "status", "COMPLETED")],
             "train": [(KINDS["generate"]["evidence"], "status", "COMPLETED"), (TIMING_EVIDENCE, "status", "COMPLETED")]}
GENERATE_CEILING = {"factor": 1.5, "max_seconds": 6 * 3600}
#: card §4: the one recipe. Not a tuning point.
RECIPE = {"updates": 3000, "batch": 64, "lr": 1e-3, "l2": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
          "bias_correction": False, "init_seed": 20261028, "order_seed": 20261029, "min_kl": 0.01,
          "heldout_mod": 10, "heldout_rem": 9, "from_ply": 20, "arch": {"in_channels": 30, "hidden": 128,
                                                                         "n_blocks": 6}}
TIMING = {"updates": 20, "warmup": 2, "forward_batches": 5, "seed": 20261030, "n_legal": 500, "tensor_calls": 200,
          "projected_rows": 37000, "projected_heldout_rows": 3700, "ceiling_factor": 1.5}
BOARD, M_PAD = 24, CS.M_PAD
EXIT_COMPLETED, EXIT_USAGE, EXIT_VOID, EXIT_REFUSED = 0, 2, 3, 4
Void, Terminated = M.Void, M.Terminated


class Refused(Exception):
    """The run never started: the setting, MLX already loaded, a gate closed, a pin, an output exists, admission, or
    a bad argument."""


# ─────────────────────────────── shared ───────────────────────────────

def _sha_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_npz_new(path: pathlib.Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Create-only, flushed and fsynced; returns the file's sha256."""
    buf = io.BytesIO()
    np.savez(buf, **{k: np.asarray(v) for k, v in arrays.items()})
    data = buf.getvalue()
    with open(path, "xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    return hashlib.sha256(data).hexdigest()


def _watch(seams: Any, rec: Dict[str, Any], log, when: str) -> None:
    """The background-compute check, logged every time; BOINC detected or a failed check is a VOID."""
    obs = seams.background()
    rec["background_checks"] = rec.get("background_checks", 0) + 1
    log("bg_check", when=when, ok=obs.get("ok") is True, n_procs=obs.get("n_procs"), hits=obs.get("hits"),
        exempt=obs.get("exempt"), error=obs.get("error"))
    if obs.get("ok") is not True:
        raise Void("background_compute", f"{when}: " + (f"the check failed: {obs.get('error')}" if obs.get("error")
                                                         else f"BOINC detected: {obs.get('hits')}"))


def _row_keys(game: Mapping[str, Any]) -> List[str]:
    """Each row's position, keyed by the move sequence that reached it (transpositions are not merged)."""
    return [hashlib.sha256(json.dumps(game["moves"][:p]).encode()).hexdigest() for p in game["row_plies"]]


def target_stats(rows: Mapping[str, np.ndarray]) -> Dict[str, float]:
    """Sums over one game's rows (card §8.3): entropy of pi, KL(pi || stored prior), pi's mass on the played move and
    on the candidates eliminated in the first round."""
    out = {"rows": 0, "entropy": 0.0, "kl_prior": 0.0, "mass_played": 0.0, "mass_round1_out": 0.0}
    for i in range(len(rows["ply"])):
        n = int(rows["n_legal"][i])
        pi, p = rows["target"][i, :n], rows["prior"][i, :n].astype(np.float64)
        nz = pi > 0
        out["rows"] += 1
        out["entropy"] += float(-np.sum(pi[nz] * np.log(pi[nz])))
        out["kl_prior"] += float(np.sum(pi[nz] * (np.log(pi[nz]) - np.log(p[nz]))))
        out["mass_played"] += float(pi[int(rows["choice"][i])])
        c = rows["cand"][i]
        out1 = [int(j) for j, r in zip(c, rows["rounds"][i]) if j >= 0 and r == 1 and rows["n_rounds"][i] > 1]
        out["mass_round1_out"] += float(np.sum(pi[out1])) if out1 else 0.0
    return out


# ─────────────────────────────── the throughput check and generation (card §5 steps 2-3) ───────────────────────────────

@dataclasses.dataclass
class GenSeams:
    """What a generation run reads from outside this module. TEST seams; `run` passes `_real_gen_seams` only."""
    versions: Callable[[], Dict[str, str]]
    read_plan: Callable[[str], Mapping[str, Any]]
    seed_ok: Callable[[int], bool]
    checkpoint_sha1: Callable[[], str]
    load_evaluator: Callable[[], Any]
    require_eval: Callable[[Any, str], Dict[str, Any]]
    loaded_modules: Callable[[], Dict[str, str]]
    background: Callable[[], Dict[str, Any]]
    generate: Callable[..., Dict[str, Any]]
    rows_dir: pathlib.Path


def _generate(kind: str, seams: GenSeams, out_dir: pathlib.Path, rec: Dict[str, Any], log) -> None:
    from . import root_sequential_halving as RSH

    spec = KINDS[kind]
    n = spec["slots"]
    rec["versions"] = seams.versions()
    if rec["versions"] != VERSIONS:
        raise Void("versions", f"{rec['versions']} are not the card's {VERSIONS}")
    rec["variant"] = K.check_variant(RSH)
    plan = seams.read_plan(spec["plan"])
    if plan.get("n_slots") != n or type(plan.get("base")) is not int:
        raise Void("seeds", f"the plan is not exactly {n} slots from an int base: {plan}")
    bad = [plan["base"] + s for s in range(n) if not seams.seed_ok(plan["base"] + s)]
    if bad:
        raise Void("seeds", f"{len(bad)} seeds not ACCOUNTED, available and schedulable, first {bad[0]}")
    rec["seeds"] = {"base": plan["base"], "n_slots": n}
    sha1 = seams.checkpoint_sha1()
    if sha1 != M.CHECKPOINT_SHA1:
        raise Void("checkpoint", f"SHA-1 {sha1} is not calib020_0001's")
    rec["checkpoint_sha1"] = sha1
    _watch(seams, rec, log, "before load")
    t = time.monotonic()
    evaluator = seams.load_evaluator()
    rec["load_seconds"] = round(time.monotonic() - t, 3)
    rec["mode_at_load"] = seams.require_eval(evaluator, "after load")
    rec["modules"] = seams.loaded_modules()
    rows_dir = seams.rows_dir / kind
    if spec["keep_rows"]:
        rows_dir.mkdir(parents=True, exist_ok=False)
        rec["rows_dir"] = str(rows_dir)
    totals = {"games": 0, "plies": 0, "wins": 0, "state_caps": 0, "other_draws": 0, "rows": 0, "entropy": 0.0,
              "kl_prior": 0.0, "mass_played": 0.0, "mass_round1_out": 0.0}
    keys: List[str] = []
    games_fh = open(out_dir / "games.jsonl", "x")

    def on_start(g: int, w: int) -> None:
        log("game_start", slot=g, worker=w)

    def on_game(g: int, w: int, game: Mapping[str, Any], rows: Mapping[str, np.ndarray], secs: float) -> None:
        if game["seed"] != plan["base"] + g:
            raise Void("seeds", f"slot {g} played seed {game['seed']}, not {plan['base'] + g}")
        if len(rows["ply"]) != game["n_rows"] or list(rows["ply"]) != game["row_plies"] or \
                any(p < RECIPE["from_ply"] for p in game["row_plies"]):
            raise Void("rows", f"slot {g}: the rows do not match the game's halving plies")
        row = {"slot": g, "worker": w, "seconds": round(secs, 3), **game}
        if spec["keep_rows"]:
            name = f"slot_{g:04d}.npz"
            row["rows_file"], row["rows_sha256"] = name, _write_npz_new(rows_dir / name, rows)
        games_fh.write(json.dumps(row) + "\n")
        games_fh.flush()
        os.fsync(games_fh.fileno())
        st = target_stats(rows)
        totals["games"] += 1
        totals["plies"] += game["n_moves"]
        totals["wins"] += game["winner"] is not None
        totals["state_caps"] += game["reason"] == "state_cap"
        totals["other_draws"] += game["winner"] is None and game["reason"] != "state_cap"
        for k in ("rows", "entropy", "kl_prior", "mass_played", "mass_round1_out"):
            totals[k] += st[k]
        keys.extend(_row_keys(game))
        log("game_end", slot=g, worker=w, n_moves=game["n_moves"], reason=game["reason"], n_rows=game["n_rows"])
        _watch(seams, rec, log, f"after slot {g}")

    t = time.monotonic()
    try:
        tele = seams.generate(evaluator=evaluator, n_slots=n, slot_base=plan["base"], n_workers=N_WORKERS,
                              max_batch_rows=MAX_BATCH_ROWS, on_start=on_start, on_game=on_game)
    except CS.GenerationFailed as e:
        raise Void("generation", str(e)) from None
    finally:
        games_fh.close()
    secs = time.monotonic() - t
    log("claims", claims=tele["claims"])
    if tele["claims"] != n or tele["started"] != list(range(n)) or tele["ended"] != list(range(n)):
        raise Void("games", f"claims {tele['claims']}, started {len(tele['started'])}, ended {len(tele['ended'])} "
                            f"of {n} slots")
    _watch(seams, rec, log, "before the summary")
    rec["mode_at_end"] = seams.require_eval(evaluator, "at the end")
    rows_total = totals["rows"]
    server = tele["server"]
    rec["generation_seconds"] = round(secs, 3)
    rec["telemetry"] = {**tele, "hellos": {str(k): v for k, v in tele["hellos"].items()}, "started": len(tele["started"]),
                        "ended": len(tele["ended"])}
    rec["throughput"] = {"games_per_hour": n / secs * 3600.0, "arbiter_rows": server["rows"],
                         "arbiter_calls": server["batches"],
                         "mean_rows_per_call": server["rows"] / server["batches"] if server["batches"] else None,
                         "n_workers": N_WORKERS, "max_batch_rows": MAX_BATCH_ROWS}
    rec["summary"] = {**{k: totals[k] for k in ("games", "plies", "wins", "state_caps", "other_draws", "rows")},
                      "mean_entropy": totals["entropy"] / rows_total if rows_total else None,
                      "mean_kl_target_prior": totals["kl_prior"] / rows_total if rows_total else None,
                      "mean_mass_played": totals["mass_played"] / rows_total if rows_total else None,
                      "mean_mass_round1_out": totals["mass_round1_out"] / rows_total if rows_total else None,
                      "duplicate_rows": len(keys) - len(set(keys)),
                      "duplicates_note": "rows whose move sequence repeats an earlier row's; transpositions not merged"}
    if kind == "throughput":
        projected = KINDS["generate"]["slots"] / rec["throughput"]["games_per_hour"] * 3600.0
        rec["generate_ceiling"] = {"projected_seconds": projected,
                                   "ceiling_seconds": min(math.ceil(GENERATE_CEILING["factor"] * projected),
                                                          GENERATE_CEILING["max_seconds"]),
                                   **GENERATE_CEILING}


def _real_gen_seams(checkpoint: pathlib.Path) -> GenSeams:
    from . import background_compute as BG
    from . import e4_screen_reference as REG

    def seed_ok(s):
        return REG.seed_is_accounted(s) and not REG.seed_is_unavailable(s) and not REG.seed_is_test_only(s)

    def load_evaluator():
        from .local_evaluator import LocalGPUEvaluator
        from .probe_eval import load_network_for_scoring
        network = load_network_for_scoring(str(checkpoint), verbose=False)[0]
        network.eval()
        return LocalGPUEvaluator(network)          # compile=False: the self-play arbiter's evaluator

    return GenSeams(versions=_versions, read_plan=lambda rel: json.loads((ROOT / rel).read_text()), seed_ok=seed_ok,
                    checkpoint_sha1=lambda: hashlib.sha1(checkpoint.read_bytes()).hexdigest(),
                    load_evaluator=load_evaluator, require_eval=lambda ev, when: M._require_eval(ev.network, when),
                    loaded_modules=M._real_seams(checkpoint).loaded_modules, background=BG.observe,
                    generate=CS.generate, rows_dir=CACHE_DIR)


# ─────────────────────────────── the objective (card §4) ───────────────────────────────

def mirrored(boards: np.ndarray, cols: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The left-right reflection MirrorAveragingEvaluator uses: boards via _mirror_position_lr, columns S-1-c."""
    from .self_play import _mirror_position_lr
    m_boards = np.stack([_mirror_position_lr(b, [], [], BOARD)[0] for b in boards]).astype(np.float32)
    return m_boards, np.where(mask > 0, BOARD - 1 - cols, cols).astype(np.int32)


def features(network, boards: np.ndarray, rows: np.ndarray, cols: np.ndarray, mask: np.ndarray):
    """The frozen encoder's output on the canonicalized input, exactly as forward_padded computes it, evaluated."""
    import mlx.core as mx

    from .network import canonicalize_batch
    b, r, c, m = canonicalize_batch(mx.array(boards), mx.array(rows), mx.array(cols), mx.array(mask), BOARD)
    f = network.encoder(b)
    mx.eval(f, r, c, m)
    return f, r, c, m


def log_policy(head, f, r, c, m):
    import mlx.core as mx
    z = head.forward_padded(f, r, c, m)
    return z - mx.logsumexp(z, axis=-1, keepdims=True)


def log_averaged(head, direct, mirror):
    """log of the mirror-averaged prior: log((p(s) + p(Ms) mapped back) / 2), index-aligned as the evaluator's."""
    import mlx.core as mx
    return mx.logaddexp(log_policy(head, *direct), log_policy(head, *mirror)) - math.log(2.0)


def batch_inputs(network, data: Mapping[str, np.ndarray], idx: np.ndarray):
    """(direct features, mirrored features) for the rows idx."""
    b, r, c, m = (data[k][idx] for k in ("boards", "rows", "cols", "mask"))
    mb, mc = mirrored(b, c, m)
    return features(network, b, r, c, m), features(network, mb, r, mc, m)


def loss_fn(network, direct, mirror, target, l2):
    """Card §4: CE(pi, p_bar) over legal moves, mean over rows, + l2 * sum of squares of the trainable arrays."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    ce = -mx.sum(target * log_averaged(network.policy_head, direct, mirror)) / target.shape[0]
    return ce + l2 * sum(mx.sum(v * v) for _, v in tree_flatten(network.trainable_parameters()))


def draw_order(n: int, updates: int, batch: int, seed: int) -> np.ndarray:
    """Fresh permutations of the training rows from PCG64(seed), concatenated and cut into batches."""
    rng = np.random.Generator(np.random.PCG64(seed))
    total, parts = updates * batch, []
    while sum(len(p) for p in parts) < total:
        parts.append(rng.permutation(n))
    return np.concatenate(parts)[:total].reshape(updates, batch)


def measure(network, data: Mapping[str, np.ndarray], idx: np.ndarray, chunk: int = 64) -> Dict[str, Any]:
    """Over rows idx, with the current head: mean CE(pi, p_bar), mean KL(pi || p_bar), and the largest probability
    discrepancy |p_bar - stored prior| over legal moves (descriptive: it does not bound the KL)."""
    import mlx.core as mx
    ce = kl = 0.0
    disc = 0.0
    for i in range(0, len(idx), chunk):
        j = idx[i:i + chunk]
        lp = log_averaged(network.policy_head, *batch_inputs(network, data, j))
        mx.eval(lp)
        lp = np.asarray(lp, dtype=np.float64)
        for k, row in enumerate(j):
            n = int(data["n_legal"][row])
            pi, lq = data["target64"][row, :n], lp[k, :n]
            if not np.all(np.isfinite(lq)):
                raise FloatingPointError(f"row {row}: non-finite log p_bar")
            nz = pi > 0
            ce -= float(np.sum(pi[nz] * lq[nz]))
            kl += float(np.sum(pi[nz] * (np.log(pi[nz]) - lq[nz])))
            disc = max(disc, float(np.max(np.abs(np.exp(lq) - data["prior"][row, :n]))))
    return {"rows": int(len(idx)), "ce": ce / len(idx), "kl": kl / len(idx), "probability_discrepancy": disc}


def values(network, data: Mapping[str, np.ndarray], idx: np.ndarray, chunk: int = 64) -> np.ndarray:
    """The network's raw value outputs (forward_padded) on rows idx."""
    import mlx.core as mx
    out = []
    for i in range(0, len(idx), chunk):
        j = idx[i:i + chunk]
        v = network.forward_padded(*(mx.array(data[k][j]) for k in ("boards", "rows", "cols", "mask")), BOARD)[1]
        mx.eval(v)
        out.append(np.array(v))
    return np.concatenate(out)


def snapshot(network) -> Dict[str, np.ndarray]:
    from mlx.utils import tree_flatten
    return {k: np.array(v) for k, v in tree_flatten(network.parameters())}


def digest(arrays: Mapping[str, np.ndarray], prefix: str) -> str:
    h = hashlib.sha256()
    for k in sorted(a for a in arrays if a.startswith(prefix)):
        a = np.ascontiguousarray(arrays[k])
        h.update(f"{k}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def changed_outside(before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray],
                    surface: Sequence[str]) -> List[str]:
    """Arrays that changed but are not trainable, or are BatchNorm running statistics."""
    if set(before) != set(after):
        return ["<the parameter set changed>"]
    changed = {k for k in before if not np.array_equal(before[k], after[k])}
    return sorted(k for k in changed if k not in surface or k.endswith(("running_mean", "running_var")))


def _require_eval_mode(network, where: str) -> None:
    if any(m.training for _, m in network.named_modules()):
        raise Void("mode", f"{where}: the network left eval mode")


# ─────────────────────────────── the corpus (card §4) ───────────────────────────────

def build_corpus(games: Sequence[Mapping[str, Any]], read_rows: Callable[[Mapping[str, Any]], Mapping[str, np.ndarray]],
                 from_ply: int) -> Dict[str, np.ndarray]:
    """Replay every game; for each row, the board and legal moves at its ply, with the stored prior and target. VOID
    unless every row's legal count, prior and target agree with the replay and with halving_target recomputed."""
    from .game.twixt_state import TwixtState

    cols = {k: [] for k in ("boards", "rows", "cols", "mask", "prior", "target64", "n_legal", "slot", "ply")}
    for game in games:
        rows = read_rows(game)
        plies = [int(p) for p in rows["ply"]]
        if plies != list(game["row_plies"]) or any(p < from_ply for p in plies):
            raise Void("corpus", f"slot {game['slot']}: row plies {plies[:5]}... are not the game's halving plies")
        state = TwixtState(active_size=BOARD, to_move="red", max_plies_limit=280)
        want = dict(zip(plies, range(len(plies))))
        found = 0
        for ply, move in enumerate(game["moves"]):
            if ply in want:
                i = want[ply]
                legal = state.legal_moves()
                n = int(rows["n_legal"][i])
                if len(legal) != n or n > M_PAD:
                    raise Void("corpus", f"slot {game['slot']} ply {ply}: {len(legal)} legal moves, {n} stored")
                prior = rows["prior"][i].astype(np.float32)
                cand = [int(c) for c in rows["cand"][i] if c >= 0]
                q = [float(v) for v in rows["q"][i][:len(cand)]]
                if not np.array_equal(T.halving_target(prior[:n], cand, q, int(rows["n_max"][i])),
                                      rows["target"][i, :n]) or np.any(rows["target"][i, n:]):
                    raise Void("corpus", f"slot {game['slot']} ply {ply}: the stored target is not halving_target's")
                r = np.zeros(M_PAD, np.int32)
                c = np.zeros(M_PAD, np.int32)
                m = np.zeros(M_PAD, np.float32)
                r[:n], c[:n], m[:n] = [a for a, _ in legal], [b for _, b in legal], 1.0
                cols["boards"].append(state.to_tensor().transpose(1, 2, 0).astype(np.float32))
                for k, v in (("rows", r), ("cols", c), ("mask", m), ("prior", prior), ("target64", rows["target"][i]),
                             ("n_legal", n), ("slot", game["slot"]), ("ply", ply)):
                    cols[k].append(v)
                found += 1
            state = state.apply_move(tuple(move))
        if found != len(plies):
            raise Void("corpus", f"slot {game['slot']}: a row's ply is beyond the game")
    out = {k: np.stack(v) if k in ("boards", "rows", "cols", "mask", "prior", "target64") else np.asarray(v)
           for k, v in cols.items()}
    out["target"] = out["target64"].astype(np.float32)
    return out


# ─────────────────────────────── training and the check (card §4, §5 step 4) ───────────────────────────────

@dataclasses.dataclass
class TrainSeams:
    """What the training run reads from outside this module. TEST seams; `run` passes `_real_train_seams` only."""
    versions: Callable[[], Dict[str, str]]
    read_record: Callable[[str], Mapping[str, Any]]
    read_games: Callable[[], List[Mapping[str, Any]]]
    read_rows: Callable[[Mapping[str, Any]], Mapping[str, np.ndarray]]
    checkpoint_sha1: Callable[[], str]
    load_network: Callable[[], Tuple[Any, Dict[str, int]]]
    loaded_modules: Callable[[], Dict[str, str]]


def _train(recipe: Mapping[str, Any], seams: TrainSeams, out_dir: pathlib.Path, rec: Dict[str, Any], log) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from . import policy_head_capacity as PHC

    mx.set_default_device(mx.cpu)
    if mx.default_device() != mx.cpu:
        raise Void("device", f"the default device is {mx.default_device()}, not the CPU")
    rec["device"] = str(mx.default_device())
    rec["recipe"] = dict(recipe)
    rec["versions"] = seams.versions()
    if rec["versions"] != VERSIONS:
        raise Void("versions", f"{rec['versions']} are not the card's {VERSIONS}")
    timing = seams.read_record(TIMING_EVIDENCE)
    if timing.get("arch") != recipe["arch"]:
        raise Void("timing", f"the timed architecture {timing.get('arch')} is not the recipe's {recipe['arch']}")
    sha1 = seams.checkpoint_sha1()
    if sha1 != M.CHECKPOINT_SHA1:
        raise Void("checkpoint", f"SHA-1 {sha1} is not calib020_0001's")
    rec["checkpoint_sha1"] = sha1

    log("phase", name="corpus")
    games = seams.read_games()
    data = build_corpus(games, seams.read_rows, recipe["from_ply"])
    held = data["slot"] % recipe["heldout_mod"] == recipe["heldout_rem"]
    i_train, i_held = np.flatnonzero(~held), np.flatnonzero(held)
    rec["corpus"] = {"games": len(games), "rows": int(len(data["slot"])), "train_rows": int(len(i_train)),
                     "heldout_rows": int(len(i_held)), "heldout_rule": f"slot % {recipe['heldout_mod']} == "
                                                                        f"{recipe['heldout_rem']}"}
    if not len(i_train) or not len(i_held):
        raise Void("corpus", "an empty training or held-out set")

    log("phase", name="load")
    network, arch = seams.load_network()            # weights only; eval mode
    rec["arch"] = arch
    if arch != recipe["arch"]:
        raise Void("checkpoint", f"the network's architecture {arch} is not the recipe's {recipe['arch']}")
    _require_eval_mode(network, "after load")
    rec["modules"] = seams.loaded_modules()
    probe = i_held[:min(len(i_held), 32)]
    parent_logits = np.array(network.policy_head.forward_padded(*batch_inputs(network, data, probe)[0]))
    v_parent = values(network, data, i_held)
    surface = PHC.install(network, "R", init_seed=recipe["init_seed"])
    rec["surface"] = {"names": surface, "parameters": PHC.SURFACES["R"][1]}
    rec["r_init_sha256"] = PHC.residual_sha256(PHC.residual_arrays(network.policy_head))
    if rec["r_init_sha256"] != PHC.residual_sha256(PHC.residual_init(recipe["init_seed"])):
        raise Void("install", "arm R's residual path does not start at its seed's arrays")
    if not np.array_equal(np.array(network.policy_head.forward_padded(*batch_inputs(network, data, probe)[0])),
                          parent_logits):
        raise Void("install", "arm R does not start bit-equal to the parent's policy head")
    _require_eval_mode(network, "after install")
    before = snapshot(network)
    rec["digests_before"] = {p: digest(before, p) for p in ("encoder.", "value_head.", "policy_head.")}

    def stop(reason: str, detail: str) -> None:
        rec["verdict"], rec["stop"] = "STOP", {"rule": reason, "detail": detail}
        log("stop", rule=reason, detail=detail)

    log("phase", name="prefit")
    try:
        rec["prefit_train"] = measure(network, data, i_train)
        rec["prefit_heldout"] = measure(network, data, i_held)
    except FloatingPointError as e:
        return stop("nonfinite", str(e))
    if rec["prefit_train"]["kl"] < recipe["min_kl"]:
        return stop("prefit_kl", f"mean KL(pi || p_bar) {rec['prefit_train']['kl']:.6g} < {recipe['min_kl']}: "
                                 "nothing to learn")

    log("phase", name="fit")
    order = draw_order(len(i_train), recipe["updates"], recipe["batch"], recipe["order_seed"])
    opt = optim.Adam(learning_rate=recipe["lr"], betas=list(recipe["betas"]), eps=recipe["eps"],
                     bias_correction=recipe["bias_correction"])
    step_fn = nn.value_and_grad(network, loss_fn)
    losses = []
    t = time.monotonic()
    for step, pos in enumerate(order):
        idx = i_train[pos]
        direct, mirror = batch_inputs(network, data, idx)
        loss, grads = step_fn(network, direct, mirror, mx.array(data["target"][idx]), recipe["l2"])
        flat = tree_flatten(grads)
        if sorted(k for k, _ in flat) != sorted(surface):
            raise Void("fit", f"step {step}: a gradient outside the arm's surface")
        mx.eval(loss, grads)
        loss_f = float(loss.item())
        if not math.isfinite(loss_f) or not all(bool(mx.all(mx.isfinite(g)).item()) for _, g in flat):
            rec["losses"] = losses
            return stop("nonfinite", f"step {step}: non-finite loss or gradient")
        opt.update(network, grads)
        mx.eval(network.parameters(), opt.state)
        losses.append(loss_f)
        if step % 100 == 0:
            log("progress", step=step, loss=loss_f, seconds=round(time.monotonic() - t, 1))
    rec["fit_seconds"] = round(time.monotonic() - t, 3)
    rec["losses"] = losses
    _require_eval_mode(network, "after training")

    log("phase", name="check")
    after = snapshot(network)
    rec["digests_after"] = {p: digest(after, p) for p in ("encoder.", "value_head.", "policy_head.")}
    outside = changed_outside(before, after, surface)
    v_after = values(network, data, i_held)
    rec["value_check"] = {"rows": int(len(i_held)), "bitwise_equal": bool(np.array_equal(v_parent, v_after)),
                          "max_abs_diff": float(np.max(np.abs(v_parent - v_after)))}
    if outside or rec["digests_after"]["encoder."] != rec["digests_before"]["encoder."] or \
            rec["digests_after"]["value_head."] != rec["digests_before"]["value_head."] or \
            not rec["value_check"]["bitwise_equal"]:
        return stop("value_changed", f"arrays outside the surface changed {outside}; value outputs equal: "
                                     f"{rec['value_check']['bitwise_equal']}")
    try:
        rec["heldout_after"] = measure(network, data, i_held)
    except FloatingPointError as e:
        return stop("nonfinite", str(e))
    head = {k: v for k, v in after.items() if k.startswith("policy_head.")}
    rec["head_file"], rec["head_sha256"] = "head.npz", _write_npz_new(out_dir / "head.npz", head)
    if not rec["heldout_after"]["ce"] < rec["prefit_heldout"]["ce"]:
        return stop("heldout_not_improved", f"held-out CE {rec['heldout_after']['ce']:.6g} is not below "
                                            f"{rec['prefit_heldout']['ce']:.6g}")
    rec["verdict"] = "ELIGIBLE"


def _real_train_seams(checkpoint: pathlib.Path) -> TrainSeams:
    corpus = KINDS["generate"]["evidence"]

    def read_games():
        rel = f"{corpus}/games.jsonl"
        if M._git("status", "--porcelain", "--untracked-files=all", "--", rel):
            raise Void("corpus", f"{rel} is uncommitted or differs from HEAD")
        return [json.loads(line) for line in (ROOT / rel).read_text().splitlines() if line.strip()]

    def read_rows(game):
        path = CACHE_DIR / "generate" / game["rows_file"]
        if _sha_file(path) != game["rows_sha256"]:
            raise Void("corpus", f"{path} is not the file generation wrote")
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    def load_network():
        from .probe_eval import load_network_for_scoring
        network, in_channels, hidden, n_blocks = load_network_for_scoring(str(checkpoint), verbose=False)
        network.eval()
        return network, {"in_channels": in_channels, "hidden": hidden, "n_blocks": n_blocks}

    return TrainSeams(versions=_versions, read_record=_read_record, read_games=read_games, read_rows=read_rows,
                      checkpoint_sha1=lambda: hashlib.sha1(checkpoint.read_bytes()).hexdigest(),
                      load_network=load_network, loaded_modules=M._real_seams(checkpoint).loaded_modules)


# ─────────────────────────────── the build's CPU timing (card §5 step 1) ───────────────────────────────

def synthetic_batch(rng: np.random.Generator, batch: int, n_legal: int) -> Dict[str, np.ndarray]:
    """Shapes as the corpus's, values synthetic: sparse binary boards, either side to move, n_legal of the 576 cells
    legal in a random order. No game, no checkpoint."""
    from .network import CH_TO_MOVE
    boards = (rng.random((batch, BOARD, BOARD, RECIPE["arch"]["in_channels"])) < 0.05).astype(np.float32)
    boards[:, :, :, CH_TO_MOVE] = rng.integers(0, 2, batch)[:, None, None].astype(np.float32)
    cells = np.array([(r, c) for r in range(BOARD) for c in range(BOARD)], np.int32)
    rows, cols = np.zeros((batch, M_PAD), np.int32), np.zeros((batch, M_PAD), np.int32)
    for i in range(batch):
        perm = cells[rng.permutation(len(cells))]
        rows[i], cols[i] = perm[:, 0], perm[:, 1]
    mask = np.zeros((batch, M_PAD), np.float32)
    mask[:, :n_legal] = 1.0
    target = rng.random((batch, M_PAD)) * mask
    return {"boards": boards, "rows": rows, "cols": cols, "mask": mask,
            "target": (target / target.sum(1, keepdims=True)).astype(np.float32)}


def _timing(spec: Mapping[str, Any], recipe: Mapping[str, Any], out_dir: pathlib.Path, rec: Dict[str, Any],
            log) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from . import policy_head_capacity as PHC
    from .game.twixt_state import TwixtState
    from .network import create_network

    mx.set_default_device(mx.cpu)
    if mx.default_device() != mx.cpu:
        raise Void("device", f"the default device is {mx.default_device()}, not the CPU")
    rec["device"] = str(mx.default_device())
    rec["versions"] = _versions()
    rec["arch"] = dict(recipe["arch"])
    rec["spec"] = dict(spec)
    mx.random.seed(spec["seed"])
    network = create_network(hidden=recipe["arch"]["hidden"], n_blocks=recipe["arch"]["n_blocks"],
                             in_channels=recipe["arch"]["in_channels"])
    network.eval()
    surface = PHC.install(network, "R", init_seed=recipe["init_seed"])
    rec["weights"] = "random (create_network under mx.random.seed), arm R installed; no checkpoint read"
    rng = np.random.Generator(np.random.PCG64(spec["seed"]))
    opt = optim.Adam(learning_rate=recipe["lr"], betas=list(recipe["betas"]), eps=recipe["eps"],
                     bias_correction=recipe["bias_correction"])
    step_fn = nn.value_and_grad(network, loss_fn)
    all_idx = np.arange(recipe["batch"])

    def update() -> float:
        data = synthetic_batch(rng, recipe["batch"], spec["n_legal"])
        t = time.perf_counter()
        direct, mirror = batch_inputs(network, data, all_idx)
        loss, grads = step_fn(network, direct, mirror, mx.array(data["target"]), recipe["l2"])
        opt.update(network, grads)
        mx.eval(loss, network.parameters(), opt.state)        # synchronized: the update has completed
        secs = time.perf_counter() - t
        if not math.isfinite(float(loss.item())):
            raise Void("timing", "a non-finite loss")
        return secs

    for _ in range(spec["warmup"]):
        update()
    log("phase", name="updates")
    upd = [update() for _ in range(spec["updates"])]
    log("phase", name="forward")
    fwd = []
    for _ in range(spec["forward_batches"]):
        data = synthetic_batch(rng, recipe["batch"], spec["n_legal"])
        t = time.perf_counter()
        lp = log_averaged(network.policy_head, *batch_inputs(network, data, all_idx))
        mx.eval(lp)
        fwd.append(time.perf_counter() - t)
    state = TwixtState(active_size=BOARD, to_move="red", max_plies_limit=280)
    t = time.perf_counter()
    for _ in range(spec["tensor_calls"]):
        state.to_tensor()
    per_tensor = (time.perf_counter() - t) / spec["tensor_calls"]
    per_update = math.fsum(upd) / len(upd)
    per_row_pair = math.fsum(fwd) / len(fwd) / recipe["batch"]
    rows, held = spec["projected_rows"], spec["projected_heldout_rows"]
    projected = {"updates_seconds": per_update * recipe["updates"],
                 "measure_seconds": per_row_pair * (rows + held),         # the pre-fit pass and the held-out check
                 "value_seconds": per_row_pair * held,                     # two single-orientation passes
                 "corpus_tensor_seconds": per_tensor * rows}
    total = math.fsum(projected.values())
    rec["timing"] = {"update_seconds": upd, "mean_update_seconds": per_update, "max_update_seconds": max(upd),
                     "forward_seconds": fwd, "per_row_pair_forward_seconds": per_row_pair,
                     "per_to_tensor_seconds": per_tensor, "surface": surface}
    rec["projection"] = {**projected, "total_seconds": total,
                         "ceiling_seconds": int(math.ceil(spec["ceiling_factor"] * total / 60.0) * 60),
                         "ceiling_factor": spec["ceiling_factor"],
                         "note": "projected rows are the card's estimates (about 37,000 rows, 3,700 held out); "
                                 "loading and replay are not timed"}


# ─────────────────────────────── the record boundary and the public entry ───────────────────────────────

def conclude(kind: str, execute: Callable[[pathlib.Path, Dict[str, Any], Callable[..., None]], None],
             out_dir: pathlib.Path, admission: Mapping[str, str], head: Optional[str] = None) -> Tuple[int, Dict]:
    """THE RECORD BOUNDARY: record.json on every exit this survives; SIGTERM -> VOID at step "terminated"."""
    out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "run": kind, "card": CARD, "card_sha256": CARD_SHA256, "head": head,
                           "admission": dict(admission), "status": None, "verdict": None, "written_by": "script",
                           "setting": {SETTING: os.environ.get(SETTING)},
                           "environment": {"python": sys.version, "platform": platform.platform(),
                                           "executable": sys.executable},
                           "loadavg_before": os.getloadavg()}
    log_fh = open(out_dir / "progress.jsonl", "x")

    def log(event, **kw):
        log_fh.write(json.dumps({"event": event, "t": round(time.monotonic() - t0, 3), **kw}) + "\n")
        log_fh.flush()
        os.fsync(log_fh.fileno())

    def on_term(signum, _frame):
        raise Terminated(signal.Signals(signum).name)

    code = EXIT_VOID
    main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, on_term) if main_thread else None
    try:
        log("start", setting=rec["setting"])
        execute(out_dir, rec, log)
        rec["status"] = "COMPLETED"
        code = EXIT_COMPLETED
    except Terminated as t:
        rec.update(status="VOID", verdict=None, void={"step": "terminated",
                                                      "detail": f"{t} after {time.monotonic() - t0:.1f} s"})
    except Void as v:
        rec.update(status="VOID", verdict=None, void={"step": v.step, "detail": v.detail})
    except Exception as e:                       # any exception is a VOID, and it is recorded
        rec.update(status="VOID", verdict=None, void={"step": "exception", "detail": f"{type(e).__name__}: {e}"})
    except BaseException as e:
        rec.update(status="INTERRUPTED", verdict=None, void={"step": "interrupted", "detail": type(e).__name__})
        raise
    finally:
        if main_thread:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            rec["setting_at_end"] = {SETTING: os.environ.get(SETTING)}
            rec["total_seconds"] = round(time.monotonic() - t0, 3)
            rec["loadavg_after"] = os.getloadavg()
            log("end", status=rec["status"], verdict=rec.get("verdict"))
            log_fh.close()
            M._write_json_new(out_dir / "record.json", rec)
        finally:
            if main_thread:
                signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)
    return code, rec


def _versions() -> Dict[str, str]:
    import mlx.core as mx
    return {"mlx": mx.__version__, "numpy": np.__version__, "python": platform.python_version()}


def _read_record(rel: str) -> Mapping[str, Any]:
    """A committed record: it must exist and be unchanged from HEAD."""
    path = f"{rel}/record.json"
    if not (ROOT / path).is_file():
        raise FileNotFoundError(path)
    if M._git("status", "--porcelain", "--untracked-files=all", "--", path):
        raise Refused(f"{path} is uncommitted or differs from HEAD")
    return json.loads((ROOT / path).read_text())


def admitted(kind: str, read_record: Callable[[str], Mapping[str, Any]]) -> Dict[str, str]:
    """{record: value} for every earlier step this kind needs, or Refused."""
    seen = {}
    for rel, field, want in ADMISSION[kind]:
        try:
            rec = read_record(rel)
        except FileNotFoundError:
            raise Refused(f"{rel}/record.json does not exist: {kind} is not admitted") from None
        if rec.get("status") != "COMPLETED" or rec.get(field) != want or rec.get("card_sha256") != CARD_SHA256:
            raise Refused(f"{rel}/record.json is not COMPLETED with {field} {want} under this card: {kind} is not "
                          f"admitted")
        seen[rel] = want
    return seen


def _preflight() -> None:
    """Every subcommand: the setting in a fresh process, the pinned card and the frozen modules."""
    if os.environ.get(SETTING) != SETTING_VALUE:
        raise Refused(f"{SETTING} is {os.environ.get(SETTING)!r}, not {SETTING_VALUE!r}: launch a fresh process with it")
    if _mlx_loaded():
        raise Refused("MLX was imported before the setting was checked: launch a fresh process")
    if M._sha((ROOT / CARD).read_bytes()) != CARD_SHA256:
        raise Refused(f"{CARD} is not the pinned card")
    for rel, sha in PINNED.items():
        if M._sha((ROOT / rel).read_bytes()) != sha:
            raise Refused(f"{rel} is not the pinned bytes")


def run(kind: str, checkpoint: str) -> int:
    """THE PUBLIC ENTRY for the three gated steps. No seam: the setting, the gate, the pins, admission, real sources."""
    if kind not in KINDS:
        raise Refused(f"unknown run {kind!r}")
    _preflight()
    gate, name = KINDS[kind]["gate"]
    if getattr(gate, name) is not True:
        raise Refused(f"{name} is closed: the {kind} step is not authorized")
    admission = admitted(kind, _read_record)
    ckpt = pathlib.Path(checkpoint)
    if not (ckpt.is_absolute() and ckpt.is_file()):
        raise Refused("--checkpoint must name an existing file by absolute path")
    out = ROOT / KINDS[kind]["evidence"]
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the run is one-shot")
    if kind == "train":
        seams = _real_train_seams(ckpt)
        execute = lambda o, r, lg: _train(RECIPE, seams, o, r, lg)    # noqa: E731
    else:
        if KINDS[kind]["keep_rows"] and os.path.lexists(CACHE_DIR / kind):
            raise Refused(f"{CACHE_DIR / kind} exists: the corpus is written once")
        seams = _real_gen_seams(ckpt)
        execute = lambda o, r, lg: _generate(kind, seams, o, r, lg)   # noqa: E731
    code, rec = conclude(kind, execute, out, admission, head=M._git("rev-parse", "HEAD"))
    print(json.dumps({"status": rec["status"], "void": rec.get("void"), "run": kind, "verdict": rec.get("verdict"),
                      "stop": rec.get("stop"), "throughput": rec.get("throughput")}))
    return code


def run_timing() -> int:
    """THE BUILD'S CPU TIMING: approved with the build; no gate, no checkpoint, no GPU; one shot. It times the CPU, so
    it refuses (consuming nothing) while BOINC runs, and is VOID if BOINC appears by the end."""
    from . import background_compute as BG

    _preflight()
    out = ROOT / TIMING_EVIDENCE
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the timing is one-shot")
    first = BG.observe()
    if first.get("ok") is not True:
        raise Refused(f"{BG.describe(first)}: the CPU is shared; quit BOINC before the timing")

    def execute(o, r, lg):
        r["background_before"] = first
        _timing(TIMING, RECIPE, o, r, lg)
        r["background_after"] = after = BG.observe()
        if after.get("ok") is not True:
            raise Void("background_compute", f"by the end of the timing: {BG.describe(after)}")

    code, rec = conclude("time", execute, out, {}, head=M._git("rev-parse", "HEAD"))
    print(json.dumps({"status": rec["status"], "void": rec.get("void"), "projection": rec.get("projection")}))
    return code


def _mlx_loaded() -> bool:
    return any(m == "mlx" or m.startswith("mlx.") for m in sys.modules)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    usage = ("usage: MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.combination_policy_pilot "
             "run throughput|generate|train --checkpoint <abs path> | time")
    try:
        if argv == ["time"]:
            return run_timing()
        if len(argv) == 4 and argv[0] == "run" and argv[1] in KINDS and argv[2] == "--checkpoint":
            return run(argv[1], argv[3])
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(usage, file=sys.stderr)
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
