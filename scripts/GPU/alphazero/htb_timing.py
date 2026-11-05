"""H-TB BOUNDED TIMING -- card §10 step 3; plan docs/superpowers/2026-11-04-htb-bounded-timing-plan.md (pinned below).

Two entries, each run once under its own wrapper, behind the closed gate `HTB_TIMING_AUTHORIZED`:

    <teacher venv python> -m scripts.GPU.alphazero.htb_timing time label --clone-root <pinned checkout>
    .venv/bin/python      -m scripts.GPU.alphazero.htb_timing time fit

They reuse the pilot's own code paths (`htb_pilot`): its refusals, its terminal-record boundary, `select`,
`label_game`, `write_label_shards`, `load_label_shards`, `build_examples`, `fit` and `score`.

**Train arm only.** Games come from `eligible_games(arms=("train",))`, and an arm guard STOPs on any other game.
`select` is called on train games only, with `n_val=0`, so no validation or test position is replayed or keyed.

**Time only.** Teacher outputs, labels, predictions, losses and metrics are discarded in memory. A recursive check
STOPs the run if the record would carry any of those fields. The fit's examples use synthetic uniform targets and zero
values, and its shard reload uses synthetic zero shards.

**Projections price the complete runs (plan §5).** Each term is measured or derived from a measurement. Anything left
unpriced is listed by name under `unpriced`. A proposed ceiling is 1.5 × the total, rounded up to 10 minutes. The
fit's memory estimate is set against physical and available memory.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import platform
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from . import htb_pilot as H
from . import htb_timing_authorization as GATE
from .capacity_c0_inventory import split_of

PLAN = "docs/superpowers/2026-11-04-htb-bounded-timing-plan.md"
PLAN_SHA256 = "ca147b4ee3bf1da469f05e17ce9bb5dd43c424cb7da1a07aee576432bbfa8596"
EVIDENCE = "docs/superpowers/evidence/htb-timing"
CACHE = pathlib.Path.home() / "twixt-cache" / "htb-timing"

# the user's exact ceilings (2026-11-04): TERM / KILL / cleanup observed through
CEILINGS = {"label": {"term_s": 1190, "kill_s": 1200, "cleanup_through_s": 1210},
            "fit": {"term_s": 1790, "kill_s": 1800, "cleanup_through_s": 1810}}
EXECUTION_ALLOWANCE_S = CEILINGS["label"]["kill_s"] + CEILINGS["fit"]["kill_s"]          # 3,000 s = 50 min
FACTOR, ROUND_S = 1.5, 600

#: fields that would carry teacher or model OUTPUTS; their presence anywhere in a record STOPs the run
FORBIDDEN_KEYS = frozenset({"target", "target576", "value", "value_id", "value_ref", "values", "loss", "policy",
                            "kl", "top1", "value_mse", "grad_norm", "log", "probs", "logits", "verdict"})


def _counts_from_c0(root: pathlib.Path) -> Dict[str, Dict[str, int]]:
    inv = json.loads((root / H.C0_INVENTORY).read_text())
    return {a: {"games": inv["split"]["arms"][a]["games"], "positions": inv["split"]["arms"][a]["positions"]}
            for a in H.ARMS}


@dataclasses.dataclass
class TimingConfig:
    """The defaults ARE the plan; tests substitute fakes and smaller sizes."""
    root: pathlib.Path = H.ROOT
    data_root: pathlib.Path = dataclasses.field(default_factory=H.main_checkout)  # corpus + checkpoint
    cache: pathlib.Path = CACHE
    committed: Callable[[pathlib.Path, str], bool] = H._committed_git
    games_train: Callable[[pathlib.Path], List[Dict[str, Any]]] = \
        lambda root: H.eligible_games(root / H.CORPUS, arms=("train",))
    load_teacher: Callable[[str], Any] = H._load_teacher_real
    load_reference: Callable[[pathlib.Path], Any] = H._load_reference_real
    save_student: Callable[[Any, pathlib.Path], Dict[str, str]] = H._save_student_real
    available_memory: Callable[[], Dict[str, int]] = lambda: H.available_memory()
    forbid_modules: Sequence[str] = ("mlx",)            # the label process must never import MLX
    # plan sizes
    key_games: int = 200
    teacher_games: int = 40
    teacher_ply_mod: int = 5
    warmup_calls: int = 20
    fit_first_game: int = 40
    fit_rows: int = 640
    fit_ply_mod: int = 5
    fit_ply_rem: int = 1
    memory_rows: int = 64
    updates: int = 60
    warmup_updates: int = 10
    reload_shards: int = 2
    # what the full runs will do (the card)
    n_train: int = H.N_TRAIN
    n_val: int = H.N_VAL
    shard_rows: int = H.SHARD_ROWS
    start_rows: int = H.START_ROWS
    full_updates: int = H.FROZEN_UPDATES
    rows_per_batch: int = H.FROZEN_RECIPE.rows_per_batch

    def evidence(self, kind: str) -> pathlib.Path:
        return self.root / EVIDENCE / kind

    def cache_dir(self, kind: str) -> pathlib.Path:
        return self.cache / kind


def _preconditions(cfg: TimingConfig, kind: str) -> None:
    H.preconditions(GATE, "HTB_TIMING_AUTHORIZED", cfg)
    if H._sha((cfg.root / PLAN).read_bytes()) != PLAN_SHA256:
        raise H.Refused("the timing plan is not the pinned one")
    if not cfg.committed(cfg.root, PLAN):
        raise H.Refused("the timing plan is not committed")
    H._absent(cfg.evidence(kind), cfg.cache_dir(kind))


def arm_guard(games: Sequence[Mapping[str, Any]]) -> None:
    other = [g["id"] for g in games if split_of(g["id"]) != "train"]
    if other:
        raise H.Stop("arm", f"{len(other)} non-train games reached the timing, e.g. {other[:3]}")


def assert_no_outputs(obj: Any, path: str = "record") -> None:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if k in FORBIDDEN_KEYS:
                raise H.Stop("outputs", f"{path}.{k} would carry an output")
            assert_no_outputs(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_outputs(v, f"{path}[{i}]")


def _ceiling(total_s: float) -> Dict[str, Any]:
    c = int(math.ceil(FACTOR * total_s / ROUND_S) * ROUND_S)
    return {"projected_s": round(total_s, 1), "factor": FACTOR, "kill_s": c, "term_s": c - 10, "cleanup_through_s": c + 10}


class _TimedTeacher:
    """Delegates to the teacher; times each evaluate() call. Outputs pass through to label_game and are discarded."""

    def __init__(self, teacher):
        self.t, self.durations = teacher, []

    def new_game(self):
        return self.t.new_game()

    def play(self, tb, rc):
        self.t.play(tb, rc)

    def divergences(self, our, tb):
        return self.t.divergences(our, tb)

    def evaluate(self, tb, colour, legal):
        t0 = time.perf_counter()
        out = self.t.evaluate(tb, colour, legal)
        self.durations.append(time.perf_counter() - t0)
        return out


# ─────────────────────────────── projections (pure) ───────────────────────────────

def project_label(m: Mapping[str, Any], c0: Mapping[str, Mapping[str, int]], cfg: TimingConfig) -> Dict[str, Any]:
    pos = c0["train"]["positions"] + c0["val"]["positions"]
    rows = cfg.n_train + cfg.n_val
    calls = 2 * rows
    d = m["teacher_calls"]
    shards = math.ceil(cfg.n_train / cfg.shard_rows) + math.ceil(cfg.n_val / cfg.shard_rows)
    terms = {
        "teacher_load_s": m["teacher_load_s"],
        "corpus_auth_s": m["corpus_auth_s"],
        "selection_s": m["selection_s"] / m["selection_positions"] * pos,
        "teacher_calls_s": d["warmup_s"] + (calls - d["warmup_calls"]) * d["steady_mean_s"],
        "replay_overhead_s": m["replay_overhead_s"] / m["plies_replayed"] * 2 * pos,
        "shard_writes_s": m["shard_write_s"] * shards,
        "selection_serialization_s": m["selection_serialization_s"],
    }
    return {"terms": {k: round(v, 1) for k, v in terms.items()}, **_ceiling(sum(terms.values())),
            "basis": {"positions_train_val": pos, "teacher_calls": calls, "shards": shards,
                      "replay": "upper bound: every train and val position replayed in both orientations"},
            "unpriced": ["record, manifest and started.json writes (KB-scale JSON)",
                         "the selection record's file write beyond its in-memory serialization"]}


def project_fit(m: Mapping[str, Any], cfg: TimingConfig) -> Dict[str, Any]:
    rows = cfg.n_train + cfg.n_val
    shards = math.ceil(cfg.n_train / cfg.shard_rows) + math.ceil(cfg.n_val / cfg.shard_rows)
    u = m["updates"]
    terms = {
        "corpus_auth_s": m["corpus_auth_s"],
        "label_reload_s": m["reload_s_per_shard"] * shards,
        "examples_s": m["examples_s_per_row"] * rows,
        "model_loads_s": m["load_s"][0] + m["load_s"][1],
        "start_check_s": m["score_s_per_row_network"] * cfg.start_rows,
        "validation_scoring_s": m["score_s_per_row_network"] * 2 * cfg.n_val,
        "updates_s": u["warmup_s"] + (cfg.full_updates - u["warmup_updates"]) * u["steady_mean_s"],
        "save_s": m["save_s"],
    }
    return {"terms": {k: round(v, 1) for k, v in terms.items()}, **_ceiling(sum(terms.values())),
            "basis": {"rows": rows, "shards": shards, "updates": cfg.full_updates},
            "unpriced": ["check_label_binding: reading record.json and selection.json (tens of MB of JSON)",
                         "the fit's step log and record writes"]}


def memory_estimate(m: Mapping[str, Any], cfg: TimingConfig) -> Dict[str, Any]:
    rows = cfg.n_train + cfg.n_val
    boards_only = 2 * m["board_bytes"] * rows
    examples = m["example_bytes_traced"] * rows
    labels = m["label_row_bytes"] * rows
    est = examples + labels + m["networks_bytes"]
    return {"rows": rows, "boards_only_bytes": boards_only, "examples_bytes": examples, "label_rows_bytes": labels,
            "networks_bytes": m["networks_bytes"], "estimate_bytes": est,
            "physical_bytes": m["memory_before"]["physical_bytes"], "available_bytes": m["memory_before"]["available_bytes"],
            "fits_available": est < m["memory_before"]["available_bytes"],
            "fits_physical": est < m["memory_before"]["physical_bytes"],
            "lazy_boards_s_per_row": m["lazy_board_s_per_row"],
            "lazy_boards_projected_s": m["lazy_board_s_per_row"] * cfg.full_updates * cfg.rows_per_batch,
            "note": "boards are float32 (H, W, 30) for both orientations of every row, materialized at once by "
                    "build_examples; the lazy figure prices rebuilding each batch's boards per update instead (the measured "
                    "per-row time already builds both orientations, so it is multiplied by rows, not by 2 x rows)."}


# ─────────────────────────────── the two entries ───────────────────────────────

def time_label(clone_root: str, cfg: Optional[TimingConfig] = None) -> int:
    cfg = cfg or TimingConfig()
    _preconditions(cfg, "label")
    rec: Dict[str, Any] = {"kind": "htb-timing-label-record/1", "card_sha256": H.CARD_SHA256,
                           "plan_sha256": PLAN_SHA256, "ceiling": CEILINGS["label"]}

    def body(rec):
        m: Dict[str, Any] = {}
        rec["environment"] = {"python": sys.version.split()[0], "executable": sys.executable,
                              "numpy": np.__version__, "platform": platform.platform()}
        t = time.perf_counter()
        teacher, identity = cfg.load_teacher(clone_root)
        m["teacher_load_s"] = time.perf_counter() - t
        rec["runtime"] = identity
        t = time.perf_counter()
        games = cfg.games_train(cfg.data_root)
        m["corpus_auth_s"] = time.perf_counter() - t
        arm_guard(games)
        games = sorted(games, key=lambda g: g["id"])
        rec["train_games_available"] = len(games)

        sub = games[:cfg.key_games]                                       # L2: replay, keys, dedup, order preparation
        t = time.perf_counter()
        sel = H.select(sub, n_train=1, n_val=0)
        m["selection_s"] = time.perf_counter() - t
        m["selection_positions"] = sel["positions"]["train"]
        del sel

        timed, plies, total = _TimedTeacher(teacher), 0, 0.0              # L3: label_game, both orientations
        for g in games[:cfg.teacher_games]:
            rows = [{"ply": k, "key": H.state_key(s)} for k, s in H.positions(g["moves"]) if k % cfg.teacher_ply_mod == 0]
            plies += 2 * (max(r["ply"] for r in rows) + 1)
            t = time.perf_counter()
            out = H.label_game(g["id"], g["moves"], rows, timed)
            total += time.perf_counter() - t
            del out
        d = timed.durations
        w = min(cfg.warmup_calls, len(d))
        m["teacher_calls"] = {"calls": len(d), "warmup_calls": w, "warmup_s": float(sum(d[:w])),
                              "steady_mean_s": float(np.mean(d[w:])) if len(d) > w else float(np.mean(d)),
                              "steady_p95_s": float(np.quantile(d[w:], 0.95)) if len(d) > w else None}
        m["label_game_total_s"] = total
        m["replay_overhead_s"] = total - float(sum(d))                    # teacher time counted ONCE, not here
        m["plies_replayed"] = plies

        syn = [{"game_id": "synthetic", "ply": 0, "key": "0" * 64, "legal": [], "target": [], "value": 0.0,
                "value_id": 0.0, "value_ref": 0.0} for _ in range(cfg.shard_rows)]       # L4: zeros, not outputs
        t = time.perf_counter()
        H.write_label_shards(syn, cfg.cache_dir("label") / "shard-write", cfg.shard_rows)
        m["shard_write_s"] = time.perf_counter() - t
        del syn

        rows_full = [{"game_id": "iter_0000_game_000", "ply": 0, "key": "0" * 64}] * (cfg.n_train + cfg.n_val)
        t = time.perf_counter()                                           # L5: selection.json, in memory only
        json.dumps({"rows": rows_full}, sort_keys=True, indent=1)
        m["selection_serialization_s"] = time.perf_counter() - t

        loaded = [x for x in cfg.forbid_modules if x in sys.modules]
        if loaded:
            raise H.Stop("isolation", f"the label process imported {loaded}")
        rec["measurements"] = m
        rec["projection"] = project_label(m, _counts_from_c0(cfg.root), cfg)
        assert_no_outputs(rec)

    return H._terminal(cfg.evidence("label"), rec, body)


def _fit_rows(games, cfg):
    labelled, by_game, seen = [], {}, set()
    for g in games[cfg.fit_first_game:]:
        for k, s in H.positions(g["moves"]):
            if k % cfg.fit_ply_mod != cfg.fit_ply_rem:
                continue
            key = H.state_key(s)
            if key in seen:                                              # distinct states only, as the selection
                continue
            seen.add(key)
            t = np.zeros(H.PADDED)
            legal = list(s.legal_moves())
            for r, c in legal:
                t[H.A.padded_index(r, c)] = 1.0 / len(legal)               # synthetic uniform target
            labelled.append({"game_id": g["id"], "ply": k, "key": key, "value": 0.0, "target576": t})
            by_game[g["id"]] = g["moves"]
            if len(labelled) == cfg.fit_rows:
                return labelled, by_game
    raise H.Stop("rows", f"only {len(labelled)} timing rows available, {cfg.fit_rows} needed")


def time_fit(cfg: Optional[TimingConfig] = None) -> int:
    cfg = cfg or TimingConfig()
    _preconditions(cfg, "fit")
    rec: Dict[str, Any] = {"kind": "htb-timing-fit-record/1", "card_sha256": H.CARD_SHA256,
                           "plan_sha256": PLAN_SHA256, "ceiling": CEILINGS["fit"]}

    def body(rec):
        import tracemalloc

        import mlx.core as mx
        from mlx.utils import tree_flatten

        from .self_play import _mirror_position_lr

        mx.set_default_device(mx.cpu)
        m: Dict[str, Any] = {"memory_before": cfg.available_memory()}
        rec["environment"] = {"python": sys.version.split()[0], "executable": sys.executable, "numpy": np.__version__,
                              "mlx": mx.__version__, "device": str(mx.default_device()), "platform": platform.platform()}
        t = time.perf_counter()
        games = cfg.games_train(cfg.data_root)
        m["corpus_auth_s"] = time.perf_counter() - t
        arm_guard(games)
        games = sorted(games, key=lambda g: g["id"])

        syn = [{"game_id": "synthetic", "ply": 0, "key": "0" * 64, "legal": [], "target": [], "value": 0.0,
                "value_id": 0.0, "value_ref": 0.0} for _ in range(cfg.reload_shards * cfg.shard_rows)]
        d = cfg.cache_dir("fit") / "reload"
        H.write_label_shards(syn, d, cfg.shard_rows)
        msha = H._sha((d / "manifest.json").read_bytes())
        t = time.perf_counter()
        back = H.load_label_shards(d, msha)
        m["reload_s_per_shard"] = (time.perf_counter() - t) / cfg.reload_shards
        m["label_row_bytes"] = int(back[0]["target576"].nbytes + 200)    # the array plus a dict/meta allowance
        del syn, back

        labelled, by_game = _fit_rows(games, cfg)
        t = time.perf_counter()
        ex = H.build_examples(labelled, by_game)
        m["examples_s_per_row"] = (time.perf_counter() - t) / len(ex)
        m["board_bytes"] = int(ex[0]["record"].board_tensor.nbytes)
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        held = H.build_examples(labelled[:cfg.memory_rows], by_game)
        m["example_bytes_traced"] = int((tracemalloc.get_traced_memory()[0] - base) / len(held))
        tracemalloc.stop()
        del held
        t = time.perf_counter()                                          # prices rebuilding boards lazily per batch
        for row in labelled:
            s = dict(H.positions(by_game[row["game_id"]][:row["ply"] + 1]))[row["ply"]]
            b = H.board_hwc(s)
            _mirror_position_lr(b, list(s.legal_moves()), [0.0] * len(s.legal_moves()), H.BOARD)
        m["lazy_board_s_per_row"] = (time.perf_counter() - t) / len(labelled)

        loads = []
        for _ in range(2):
            t = time.perf_counter()
            net = cfg.load_reference(cfg.data_root)
            loads.append(time.perf_counter() - t)
            if len(loads) == 1:
                reference = net
            else:
                student = net
        m["load_s"] = loads
        m["networks_bytes"] = int(2 * sum(np.array(v).nbytes for _, v in tree_flatten(reference.parameters())))

        per_epoch = cfg.fit_rows // cfg.rows_per_batch
        if cfg.fit_rows % cfg.rows_per_batch or cfg.updates % per_epoch:
            raise H.Stop("shape", f"{cfg.updates} updates do not divide into epochs of {per_epoch}")
        recipe = dataclasses.replace(H.FROZEN_RECIPE, epochs=cfg.updates // per_epoch,
                                     rows_per_batch=cfg.rows_per_batch)
        stamps: List[float] = []
        t0 = time.perf_counter()
        H.fit(student, reference, ex, recipe, update_times=stamps)       # its return carries losses: discarded
        gaps = np.diff([t0] + stamps)
        w = cfg.warmup_updates
        m["updates"] = {"updates": len(gaps), "warmup_updates": w, "warmup_s": float(gaps[:w].sum()),
                        "steady_mean_s": float(gaps[w:].mean()), "steady_p95_s": float(np.quantile(gaps[w:], 0.95))}

        t = time.perf_counter()
        H.score(reference, ex)                                           # metrics discarded
        m["score_s_per_row_network"] = (time.perf_counter() - t) / len(ex)
        t = time.perf_counter()
        cfg.save_student(student, cfg.cache_dir("fit") / "student-timing.safetensors")
        m["save_s"] = time.perf_counter() - t
        m["memory_after"] = cfg.available_memory()

        rec["measurements"] = m
        rec["projection"] = project_fit(m, cfg)
        rec["memory_estimate"] = memory_estimate(m, cfg)
        assert_no_outputs(rec)

    return H._terminal(cfg.evidence("fit"), rec, body)


def main(argv: Optional[Sequence[str]] = None, cfg: Optional[TimingConfig] = None) -> int:
    ap = argparse.ArgumentParser(prog="htb_timing")
    sub = ap.add_subparsers(dest="cmd", required=True)
    tm = sub.add_parser("time")
    tm.add_argument("what", choices=("label", "fit"))
    tm.add_argument("--clone-root")
    a = ap.parse_args(argv)
    try:
        if a.what == "label":
            if not a.clone_root:
                raise H.Refused("time label needs --clone-root")
            return time_label(a.clone_root, cfg)
        return time_fit(cfg)
    except H.Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return H.EXIT["REFUSED"]


if __name__ == "__main__":
    sys.exit(main())
