"""H-MX -- mixed-data distillation from the current student (card revision 1, frozen 2026-11-18).

Card: docs/superpowers/2026-11-18-htb-mix-distillation-card.md, pinned below by sha256. This module is the card's
model-free build: nothing runs on import, and importing it loads no MLX, TensorFlow or model. Every runtime entry is
behind its own closed gate, read at call time:

    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.htb_mix run fit       # HTB_MIX_FIT_AUTHORIZED
    python -m scripts.GPU.alphazero.htb_mix openings                        # pure: the frozen match openings

The fit (card §6): 7,500 CPU updates from the current student under H-OP's recipe, every 32-row batch 16 H-OP rows
(all 120,000, one pass, in `htb-mix-fit/1|op` order) then 16 H-TB rows (the first 120,000 of the 239,989 H-TB
training rows left after the E1-E3 drop, in `htb-mix-fit/1|tb` order). Both cached label sets are read as written,
through their committed records and manifests. The one-time fit screens M1-M6 (card §7) score S and C once on both
cached validation sets; S must reproduce H-OP's recorded S scores first.

Reused, never modified: `htb_pilot` (the key, label storage, examples, scoring, the terminal-record boundary) and
`htb_onpolicy` (openings rule, exclusions E1-E3, generation games, stage admission pattern, the fit loop it mirrors).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from . import htb_onpolicy as H
from . import htb_pilot as P

ROOT = P.ROOT
CARD = "docs/superpowers/2026-11-18-htb-mix-distillation-card.md"
CARD_SHA256 = "d2c9b787ff3b45f55d4a4bb3a6fa7029d83273885c71f64bf7118b887f3c936f"   # revision 1, frozen
NS = "htb-mix/20261118"
SIMS = 200

# ── card §6: the mix ──
FIT_TAG = "htb-mix-fit/1"
RECIPE = P.Recipe()                         # H-OP's (the first fit's): lr 1e-4, value 1.0, L2 1e-4, clip 1.0, 32 rows
HALF = RECIPE.rows_per_batch // 2           # 16 rows from each source per batch
N_OP = 120_000                              # every H-OP training row, one pass
N_TB = 120_000                              # the first 120,000 of the H-TB pool, one pass
TB_ROWS, TB_DROPPED = 240_000, 11           # the H-TB training rows, and the E1-E3 hits dropped (card §5)
TB_POOL = TB_ROWS - TB_DROPPED              # 239,989
UPDATES = N_OP // HALF                      # 7,500
START_ROWS, START_KL_FLOOR = P.START_ROWS, P.START_KL_FLOOR      # 2,048 rows = the first 64 mixed batches

# ── card §7: the one-time fit screens ──
M1_FACTOR, M4_FACTOR, M5_DROP, M6_FACTOR = 0.90, 1.05, 0.010, 1.15
S_REL_TOL = 1e-9                            # S must reproduce H-OP's recorded S scores (card §7)
GATE_MISSED = "STOP: the mixed challenger missed the fit gate"

STUDENT = H.STUDENT                         # 0e1cb621..., state digest 586dca4f...
CACHE = pathlib.Path.home() / "twixt-cache" / "htb-mix"           # outside the repository

# ── the cached inputs, pinned by the bytes of their committed records (card §3, §4) ──
SOURCES = {
    "tb": {"label_record": "docs/superpowers/evidence/htb-label/record.json",
           "label_record_sha256": "b71c785a6fe8c0750a69785d0ab85710e150f68cc200e5b5121929df727de6fd",
           "selection": "docs/superpowers/evidence/htb-label/selection.json",
           "selection_sha256": "161babe33e7cb32371a0d830ece02077e81051f08d66399e1623d3d1fe32ed3d",
           "manifests": {"train": "ec28e768bb57ea25ad0426caada4446219fb02695919bd38226dae88eea89282",
                         "val": "5d3dcee9f862d77744f3cef68415a4c154906d7be88e9dc904fbc0ca4a1b2c5c"},
           "counts": {"train": 240_000, "val": 24_000},
           "cache": P.CACHE},
    "op": {"label_record": "docs/superpowers/evidence/htb-onpolicy-label/record.json",
           "label_record_sha256": "d58351ad5cc82ee6c4795541a2cb393dcccd59f78382b90530dd43df4644eefa",
           "selection": "docs/superpowers/evidence/htb-onpolicy-label/selection.json",
           "selection_sha256": "ecde66378fbd971ba44bed19cc91b7ad2085fc97ee98b85fdb8c12a8135375f3",
           "manifests": {"train": "b0ea21cb0a579705dd318e6662f314b7f2933e195e589d3884d54fc07f9ce933",
                         "val": "66cfcaf2acd92e09ca6b089c376dbc455d346f7b777b32aa4de23927eab0cd59"},
           "counts": {"train": 120_000, "val": 12_000},
           "cache": H.CACHE},
}
OP_GAMES = "docs/superpowers/evidence/htb-onpolicy-gen/games.jsonl"
OP_GAMES_SHA256 = "391d40e8814ef570e6d409d87eda7dd26d6aa0582e24e6d315831594de26f39d"
#: H-OP's fit record: S's scores on both validation sets, which S must reproduce, and (descriptive only) H-OP's
#: challenger's scores beside ours
OP_FIT_RECORD = "docs/superpowers/evidence/htb-onpolicy-fit/record.json"
OP_FIT_RECORD_SHA256 = "8fd862d8ca06b1e79aa757024b50c87326fa4f7a62f9d672fbec38335645d23d"

# ── card §9: openings; the H-OP schedules they must avoid ──
OP_OPENINGS = {H.OPENINGS_REL: "ca4e3bba05bfc6abddf782b1a1f68fc474bc3600410c7cbd4308073d306d3915",
               H.GEN_EXTENSION_REL: "578d34f371b238a9919a6eeb9f71c3b30eb286b7b08e95d1722e1fc17f270664"}
PAIRS = {"match": 400, "match smoke": 2}

# evidence (create-only, one run each)
EV = {"openings": "docs/superpowers/evidence/htb-mix-openings",
      "fit": "docs/superpowers/evidence/htb-mix-fit",
      "export": "docs/superpowers/evidence/htb-mix-export"}
OUTER = {"fit": "docs/superpowers/evidence/htb-mix-fit-run",
         "export": "docs/superpowers/evidence/htb-mix-export-run"}
OPENINGS_REL = f"{EV['openings']}/openings.json"
OPENINGS_SHA256 = "f4ae110ff9e7547e413b15fff29ca0ff94d34c5f21d5d8b7da76af6577ee3563"   # frozen at build

GATES = {"fit": ("htb_mix_fit_authorization", "HTB_MIX_FIT_AUTHORIZED"),
         "export": ("htb_mix_export_authorization", "HTB_MIX_EXPORT_AUTHORIZED")}
GRACE_S = H.GRACE_S
KILL_S = {"fit": 11_700, "export": 3_600}                        # card §11
FIT_MEMORY_REQUIRED = P.FIT_MEMORY_REQUIRED                       # 12 GiB, H-OP's (card §4)

Stop, Refused = P.Stop, P.Refused
_sha, _jsha = P._sha, P._jsha


# ─────────────────────────────── openings and seeds (card §9) ───────────────────────────────

def op_openings(root: pathlib.Path = ROOT) -> List[List[List[int]]]:
    """Every H-OP opening: generation (frozen list and extension), generation smoke, match and match smoke, from the
    committed files, each refused unless it is the pinned bytes."""
    out = []
    for rel, want in OP_OPENINGS.items():
        raw = (root / rel).read_bytes()
        if _sha(raw) != want:
            raise Refused(f"{rel} is not the pinned H-OP opening file")
        d = json.loads(raw)
        for k in ("gen", "gen_smoke", "match", "match_smoke", "gen_extension"):
            out += [e["moves"] for e in d.get(k, [])]
    return out


def forbidden_openings(root: pathlib.Path = ROOT) -> List[List[List[int]]]:
    """Card §9: the benchmark's and G3's openings and every H-OP opening."""
    return [*H.bench_openings(), *H.g3_openings(root), *op_openings(root)]


def build_openings(root: pathlib.Path = ROOT, n_match: int = PAIRS["match"],
                   n_smoke: int = PAIRS["match smoke"]) -> Dict[str, Any]:
    """The H-MX match and match-smoke openings, frozen at build time. Exclusions, cumulative: match <- benchmark + G3 +
    every H-OP list; match smoke <- + match."""
    bench, g3, op = H.bench_openings(), H.g3_openings(root), op_openings(root)
    excl = [*bench, *g3, *op]
    match = H.opening_list(NS, "match opening", n_match, excl)
    excl += [o["moves"] for o in match]
    smoke = H.opening_list(NS, "match smoke opening", n_smoke, excl)
    return {"kind": "htb-mix-openings/1", "card_sha256": CARD_SHA256, "namespace": NS,
            "rule": "teacher_bench openingAt: rows and columns 4-19, sorted legal moves, sha256[:12] mod n; distinct "
                    "directly or reflected from every earlier list and from the exclusions",
            "excluded": {"benchmark_openings": len(bench), "g3_openings": len(g3), "h_op_openings": len(op),
                         "benchmark_sha256": _jsha(bench), "g3_sha256": _jsha(g3), "h_op_sha256": _jsha(op)},
            "match": match, "match_smoke": smoke}


def dump_openings(o: Mapping[str, Any]) -> bytes:
    return (json.dumps(o, sort_keys=True, indent=1) + "\n").encode()


def seed_strings() -> Iterable[str]:
    """The Node driver's `${NS}/${match|match smoke}/pair p/game g/ply k` for every ply a search can be asked for."""
    for path, pairs in PAIRS.items():
        for p in range(pairs):
            for g in (0, 1):
                for k in H.SEED_PLIES:
                    yield f"{NS}/{path}/pair {p}/game {g}/ply {k}"


def seed_uniqueness() -> Dict[str, int]:
    pairs: Set[str] = set()
    values: Set[str] = set()
    n = repeats = 0
    for s in seed_strings():
        h = _sha(s.encode())
        pairs.add(h[:24])
        for v in (h[:12], h[12:24]):
            repeats += v in values
            values.add(v)
        n += 1
    return {"derivations": n, "distinct_pairs": len(pairs), "values": 2 * n, "repeated_values": repeats}


# ─────────────────────────────── gates, preconditions, admission ───────────────────────────────

def gate_open(stage: str) -> bool:
    import importlib

    mod_name, name = GATES[stage]
    mod = importlib.import_module(f"{__package__}.{mod_name}")
    return getattr(mod, name) is True


@dataclasses.dataclass
class Config:
    """Everything an entry touches. The defaults ARE the card; tests substitute fakes and smaller counts."""
    root: pathlib.Path = ROOT
    data_root: pathlib.Path = dataclasses.field(default_factory=P.main_checkout)   # the C0 corpus lives here
    cache: pathlib.Path = CACHE
    sources: Mapping[str, Mapping[str, Any]] = dataclasses.field(default_factory=lambda: SOURCES)
    n_op: int = N_OP
    n_tb: int = N_TB
    tb_pool: int = TB_POOL
    recipe: P.Recipe = RECIPE
    start_rows: int = START_ROWS
    committed: Callable[[pathlib.Path, str], bool] = H.committed
    evidence_problems: Callable[[pathlib.Path, Sequence[str]], List[str]] = H.evidence_problems
    head: Callable[[pathlib.Path], str] = H.head_of
    gate_open: Callable[[str], bool] = gate_open
    exclusions: Callable[[pathlib.Path], H.Exclusions] = H.build_exclusions
    pins: Mapping[str, str] = dataclasses.field(default_factory=lambda: {OP_GAMES: OP_GAMES_SHA256,
                                                                          OP_FIT_RECORD: OP_FIT_RECORD_SHA256})
    tf32_required: bool = True
    memory_required_bytes: Optional[int] = FIT_MEMORY_REQUIRED
    available_memory: Callable[[], Dict[str, int]] = lambda: P.available_memory()

    def ev(self, stage: str) -> pathlib.Path:
        return self.root / EV[stage]

    @property
    def challenger(self) -> pathlib.Path:
        return self.cache / "challenger.safetensors"

    @property
    def updates(self) -> int:
        return self.n_op // (self.recipe.rows_per_batch // 2)


def preconditions(stage: str, cfg: Config) -> None:
    """Refuse before anything heavy: the gate, then the card's pin and commit, then the environment."""
    if cfg.gate_open(stage) is not True:
        raise Refused(f"{GATES[stage][1]} is closed")
    if _sha((cfg.root / CARD).read_bytes()) != CARD_SHA256:
        raise Refused("the card is not the pinned one")
    if not cfg.committed(cfg.root, CARD):
        raise Refused("the card is not committed")
    if cfg.tf32_required and os.environ.get("MLX_ENABLE_TF32") != "0":
        raise Refused("MLX_ENABLE_TF32 must be 0 in this process")


def read_pinned(cfg: Config, rel: str, sha256: str) -> bytes:
    """A committed file, refused unless it is exactly the pinned bytes."""
    p = cfg.root / rel
    if not p.is_file() or not cfg.committed(cfg.root, rel):
        raise Refused(f"{rel} is absent, uncommitted or changed")
    raw = p.read_bytes()
    if _sha(raw) != sha256:
        raise Refused(f"{rel} is not the pinned file")
    return raw


def stage_admission(cfg: Config, stage: str) -> Dict[str, Any]:
    """H-OP's handoff check (the repair of 2026-11-14) for an H-MX stage: a COMPLETED record of this card, committed
    and clean wrapper evidence, a clean outer completion under the stage's own ceiling, bound by launch commit and pid."""
    rec = H.read_committed_json(cfg, f"{EV[stage]}/record.json")
    outer = H.read_committed_json(cfg, f"{OUTER[stage]}/outer.json")
    for f in H.WRAPPER_FILES[1:]:
        if not cfg.committed(cfg.root, f"{OUTER[stage]}/{f}"):
            raise Refused(f"{OUTER[stage]}/{f} is absent, uncommitted or changed")
    dirty = cfg.evidence_problems(cfg.root, [EV[stage], OUTER[stage]])
    if dirty:
        raise Refused(f"the {stage} evidence is not clean: {dirty}")
    bad = [k for k, v in H.OUTER_OK if outer.get(k) != v]
    if rec.get("status") != "COMPLETED" or rec.get("card_sha256") != CARD_SHA256:
        raise Refused(f"{EV[stage]}/record.json is not a COMPLETED record of this card")
    if bad:
        raise Refused(f"{OUTER[stage]}/outer.json is not clean: {[(k, outer.get(k)) for k in bad]}")
    if (outer.get("term_s"), outer.get("grace_s")) != (KILL_S[stage] - GRACE_S, GRACE_S) or \
            rec.get("kill_s") != KILL_S[stage]:
        raise Refused(f"the {stage} run was not under its ceiling: outer TERM {outer.get('term_s')} / grace "
                      f"{outer.get('grace_s')}, record kill_s {rec.get('kill_s')}; the card's is {KILL_S[stage]} s")
    pgid_rel = f"{OUTER[stage]}/pgid.txt"
    if not cfg.committed(cfg.root, pgid_rel) or (cfg.root / pgid_rel).read_text().strip() != str(rec.get("pid")):
        raise Refused(f"{pgid_rel} is not the recorded pid {rec.get('pid')}")
    if outer.get("head") != rec.get("head"):
        raise Refused(f"the {stage} record and its outer.json name different launch commits")
    return {"record_sha256": _sha((cfg.root / EV[stage] / "record.json").read_bytes()), "head": rec.get("head")}


def _record(stage: str, cfg: Config, **extra) -> Dict[str, Any]:
    return {"kind": f"htb-mix-{stage}-record/1", "card_sha256": CARD_SHA256, "stage": stage,
            "head": cfg.head(cfg.root), "pid": os.getpid(), "kill_s": KILL_S[stage],
            "module_sha256": _sha(pathlib.Path(__file__).read_bytes()), **extra}


# ─────────────────────────────── the cached inputs (card §4, §5) ───────────────────────────────

def _rows_of(sel: Mapping[str, Any], arm: str) -> List[Tuple[str, int, str]]:
    return [(r["game_id"], int(r["ply"]), r["key"]) for r in sel["rows"][arm]]


def load_sources(cfg: Config) -> Dict[str, Dict[str, Tuple]]:
    """Both label sets as written (card §4): each source's committed label record and selection, pinned by bytes; its
    manifests the record's and the pinned ones; every shard and row list verified by the frozen loader; the rows the
    selection's, in order; the exact counts. Returns {source: {arm: (meta, target, value)}}."""
    out: Dict[str, Dict[str, Tuple]] = {}
    for s, src in cfg.sources.items():
        rec = json.loads(read_pinned(cfg, src["label_record"], src["label_record_sha256"]))
        sel = json.loads(read_pinned(cfg, src["selection"], src["selection_sha256"]))
        if rec.get("status") != "COMPLETED" or rec.get("selection_sha256") != src["selection_sha256"]:
            raise Refused(f"the {s} label record is not COMPLETED over its pinned selection")
        man = {e["arm"]: e["manifest_sha256"] for e in rec.get("labels", [])}
        if man != dict(src["manifests"]) or rec.get("counts") != dict(src["counts"]):
            raise Refused(f"the {s} label record's manifests or counts are not the pinned ones")
        out[s] = {}
        for arm in ("train", "val"):
            meta, target, value = P.load_label_compact(pathlib.Path(src["cache"]) / f"labels-{arm}", man[arm])
            got = [(m["game_id"], int(m["ply"]), m["key"]) for m in meta]
            if got != _rows_of(sel, arm):
                raise Refused(f"the {s} {arm} labels are not the selection's rows, in order")
            if len(got) != src["counts"][arm] or len({k for _, _, k in got}) != len(got):
                raise Refused(f"the {s} {arm} labels do not hold {src['counts'][arm]} distinct keys")
            out[s][arm] = (meta, target, value)
    return out


def mix_selection(keys: Mapping[str, Sequence[str]], ex: H.Exclusions, n_op: int, n_tb: int, tb_pool: int
                  ) -> Dict[str, Any]:
    """Card §5-§6, from the four key lists: every check, then the two fixed orders.

    STOPs on a repeated key, on any key shared between two of the four sets (the key is mirror-canonical), on an E1-E3
    key in the H-OP rows or either validation set, or on an H-TB pool or H-OP count other than the card's. Returns the
    indices into each training list in batch order and what the record keeps."""
    names = ("op_train", "op_val", "tb_train", "tb_val")
    sets = {n: set(keys[n]) for n in names}
    for n in names:
        if len(sets[n]) != len(keys[n]):
            raise Stop("overlap", f"{n} repeats a key")
    overlap = {f"{a}&{b}": len(sets[a] & sets[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    if any(overlap.values()):
        raise Stop("overlap", f"key overlap between the cached sets: { {k: v for k, v in overlap.items() if v} }")
    e123 = {n: sum(ex.train_excluded(k) for k in keys[n]) for n in names}
    if e123["op_train"] or e123["op_val"] or e123["tb_val"]:
        raise Stop("exclusions", f"E1-E3 keys outside the H-TB training rows: {e123}")
    pool = [i for i, k in enumerate(keys["tb_train"]) if not ex.train_excluded(k)]
    if len(pool) != tb_pool:
        raise Stop("exclusions", f"the H-TB pool is {len(pool)} rows after the E1-E3 drop; the card's is {tb_pool}")
    if len(keys["op_train"]) != n_op:
        raise Stop("selection", f"{len(keys['op_train'])} H-OP training rows; the card's is {n_op}")
    if len(pool) < n_tb:
        raise Stop("selection", f"the H-TB pool ({len(pool)}) is smaller than {n_tb}")
    op_order = sorted(range(len(keys["op_train"])), key=lambda i: _sha(f"{FIT_TAG}|op|{keys['op_train'][i]}".encode()))
    tb_order = sorted(pool, key=lambda i: _sha(f"{FIT_TAG}|tb|{keys['tb_train'][i]}".encode()))[:n_tb]
    dropped = [keys["tb_train"][i] for i in range(len(keys["tb_train"])) if ex.train_excluded(keys["tb_train"][i])]
    return {"op_order": op_order, "tb_order": tb_order,
            "record": {"overlap": overlap, "e1_e3_hits": e123, "tb_dropped": len(dropped),
                       "tb_dropped_keys_sha256": _jsha(sorted(dropped)), "tb_pool": len(pool),
                       "op": {"n": len(op_order), "order_sha256": _jsha([keys["op_train"][i] for i in op_order])},
                       "tb": {"n": len(tb_order), "order_sha256": _jsha([keys["tb_train"][i] for i in tb_order])}}}


# ─────────────────────────────── the mixed fit (card §6) ───────────────────────────────

Plan = List[List[Tuple[Any, List[int]]]]       # per update: [(store, indices), ...] in batch order


def mixed_plan(op_store, tb_store, op_order: Sequence[int], tb_order: Sequence[int], half: int) -> Plan:
    """Batch b = H-OP rows op_order[half*b : half*(b+1)], then H-TB rows tb_order[half*b : half*(b+1)]."""
    if len(op_order) != len(tb_order) or len(op_order) % half:
        raise Stop("fit", f"{len(op_order)} H-OP and {len(tb_order)} H-TB rows do not make equal batches of {half}")
    return [[(op_store, list(op_order[i:i + half])), (tb_store, list(tb_order[i:i + half]))]
            for i in range(0, len(op_order), half)]


def single_plan(store, recipe: P.Recipe, tag: str) -> Plan:
    """`htb_onpolicy.fit`'s own batches (epochs over one source in its epoch order), for the equivalence test."""
    n = len(store)
    return [[(store, H.epoch_order(store.keys, e, tag)[b:b + recipe.rows_per_batch])]
            for e in range(recipe.epochs) for b in range(0, n, recipe.rows_per_batch)]


def start_check(reference, plan: Plan, n: int = START_ROWS) -> Dict[str, float]:
    """KL(T || reference_MA) >= 0.01 on the first n rows of the mixed order (card §6)."""
    rows: List[Mapping[str, Any]] = []
    for batch in plan:
        for store, idx in batch:
            rows += store.take(idx)
        if len(rows) >= n:
            break
    s = P.score(reference, rows[:n])
    if not s["kl"] >= START_KL_FLOOR:
        raise Stop("start", f"KL(T || reference_MA) {s['kl']} on the first {n} mixed rows is below {START_KL_FLOOR}")
    return s


def fit_mixed(student, reference, plan: Plan, recipe: P.Recipe = RECIPE, tag: str = FIT_TAG,
              on_step: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    """`htb_onpolicy.fit` line for line, except that each update's rows come from `plan` (card §6). Given
    `single_plan(store, recipe, tag)` it reproduces `htb_onpolicy.fit(..., tag=tag)` bitwise (a model-free test)."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from .trainer import clip_grad_norm, compute_masked_log_probs

    if student is reference:
        raise Stop("fit", "the student and the reference are the same object")
    for net, who in ((student, "the student"), (reference, "the reference")):
        P._require_eval(net, who)
    ref_digest = P._digest(reference)
    if P._digest(student) != ref_digest:
        raise Stop("fit", "the student does not start as an exact copy of the reference")
    if not plan or any(sum(len(i) for _, i in batch) != recipe.rows_per_batch for batch in plan):
        raise Stop("fit", f"a batch does not hold {recipe.rows_per_batch} rows")
    seen: Dict[int, Set[int]] = {}
    for batch in plan:
        for store, idx in batch:
            s = seen.setdefault(id(store), set())
            if s & set(idx) or len(set(idx)) != len(idx):
                raise Stop("fit", "a row repeats within the plan")
            s.update(idx)
    names = P.surface_names(student)
    before = P._snapshot(student)
    opt = optim.Adam(learning_rate=recipe.learning_rate)
    resolved = {"betas": [float(b) for b in opt.betas], "eps": float(opt.eps), "bias_correction": bool(opt.bias_correction)}
    if resolved != P.ADAM_DEFAULTS:
        raise Stop("optimizer", f"MLX's Adam resolves to {resolved}, not {P.ADAM_DEFAULTS}")

    def loss_fn(model, boards, rows, cols, mask, target, tv):
        logits, values, _ = model.forward_padded(boards, rows, cols, mask, active_size=P.BOARD)
        policy = mx.mean(-mx.sum(target * compute_masked_log_probs(logits, mask), axis=1))
        value = mx.mean((values - tv) ** 2)
        l2 = sum(mx.sum(p ** 2) for _, p in tree_flatten(model.trainable_parameters()))
        return policy + recipe.value_weight * value + recipe.l2_weight * l2, policy, value

    grad_fn = nn.value_and_grad(student, loss_fn)
    log = []
    for step, batch in enumerate(plan):
        recs, tv = [], []
        for store, idx in batch:
            for e in store.take(idx):
                recs += [e["record"], e["mirror"]]
                tv += [e["value"]] * 2
        boards, rows, cols, mask, target = P._batch(recs)
        (loss, policy, value), grads = grad_fn(student, boards, rows, cols, mask, target,
                                               mx.array(np.asarray(tv, dtype=np.float32)))
        grads, norm = clip_grad_norm(grads, recipe.clip_norm)
        opt.update(student, grads)
        mx.eval(student.parameters(), opt.state, loss, policy, value, norm)
        del recs, boards, rows, cols, mask, target, grads
        vals = (float(loss), float(policy), float(value), float(norm))
        if not all(np.isfinite(vals)):
            raise Stop("fit", f"update {step}: non-finite loss or gradient norm {vals}")
        if step % P.STEP_LOG_EVERY == 0:
            log.append({"update": step, "loss": vals[0], "policy": vals[1], "value": vals[2], "grad_norm": vals[3]})
            if on_step is not None:
                on_step(log[-1])
    P._require_eval(student, "the student after the fit")
    if P._digest(reference) != ref_digest:
        raise Stop("fit", "the reference changed during the fit")
    after = P._snapshot(student)
    stats = sorted(k for k in before if k.endswith(("running_mean", "running_var")))
    moved_stats = [k for k in stats if not np.array_equal(before[k], after[k])]
    if moved_stats:
        raise Stop("batchnorm", f"running statistics changed: {moved_stats[:5]}")
    moved = [k for k in names if not np.array_equal(before[k], after[k])]
    groups = {g: any(k.startswith(g) for k in moved) for g in ("encoder.", "policy_head.", "value_head.")}
    if not all(groups.values()):
        raise Stop("surface", f"a part of the network never moved: {groups}")
    return {"recipe": dataclasses.asdict(recipe), "optimizer": {"name": "Adam", **resolved}, "updates": len(plan),
            "rows_per_source_per_batch": [len(i) for _, i in plan[0]], "epoch_tag": tag, "log": log,
            "surface_tensors": len(names), "surface_tensors_changed": len(moved),
            "running_statistics_unchanged": len(stats), "reference_digest": ref_digest,
            "student_digest": P._digest(student)}


# ─────────────────────────────── the one-time fit screens (card §7) ───────────────────────────────

def s_reproduction(s: Mapping[str, Mapping[str, float]], want: Mapping[str, Mapping[str, float]],
                   tol: float = S_REL_TOL) -> Dict[str, Any]:
    """S's scores on both sets against H-OP's recorded S scores: n exact, each metric within `tol` relative."""
    diffs = {}
    for arm in ("op_val", "tb_val"):
        if s[arm]["n"] != want[arm]["n"]:
            raise Stop("reproduction", f"S scored {s[arm]['n']} {arm} rows; H-OP recorded {want[arm]['n']}")
        for m in ("kl", "top1", "value_mse"):
            d = abs(s[arm][m] - want[arm][m]) / max(abs(want[arm][m]), 1e-300)
            diffs[f"{arm}.{m}"] = d
            if not d <= tol:
                raise Stop("reproduction", f"S {arm} {m} {s[arm][m]!r} does not reproduce H-OP's {want[arm][m]!r} "
                                           f"(relative {d:.3e} > {tol:g}): the scoring path or the data changed")
    return {"tolerance_relative": tol, "max_relative_difference": max(diffs.values()), "relative": diffs}


def _movement(c: Mapping[str, float], s: Mapping[str, float]) -> Dict[str, Any]:
    lower_is_toward = {"kl": True, "top1": False, "value_mse": True}
    out = {}
    for m, lower in lower_is_toward.items():
        sv, cv = float(s[m]), float(c[m])
        d = cv - sv
        direction = "unchanged" if d == 0 else ("toward the teacher" if (d < 0) == lower else "away from the teacher")
        out[m] = {"S": sv, "C": cv, "C_minus_S": d, "C_over_S": (cv / sv) if sv else None, "direction": direction}
    return out


def gate_verdict(c_op: Mapping[str, float], s_op: Mapping[str, float], c_tb: Mapping[str, float],
                 s_tb: Mapping[str, float]) -> Dict[str, Any]:
    """M1-M6 of card §7, and every metric's movement on both sets (reported either way)."""
    m = {"M1": c_op["kl"] <= M1_FACTOR * s_op["kl"],
         "M2": c_op["top1"] > s_op["top1"],
         "M3": c_op["value_mse"] <= s_op["value_mse"],
         "M4": c_tb["kl"] <= M4_FACTOR * s_tb["kl"],
         "M5": c_tb["top1"] >= s_tb["top1"] - M5_DROP,
         "M6": c_tb["value_mse"] <= M6_FACTOR * s_tb["value_mse"]}
    held = [k for k, v in m.items() if v]
    bars = {"M1": M1_FACTOR * s_op["kl"], "M2": s_op["top1"], "M3": s_op["value_mse"],
            "M4": M4_FACTOR * s_tb["kl"], "M5": s_tb["top1"] - M5_DROP, "M6": M6_FACTOR * s_tb["value_mse"]}
    return {**m, "held": held, "failed": [k for k in m if k not in held], "bars": bars,
            "movement": {"op_val": _movement(c_op, s_op), "tb_val": _movement(c_tb, s_tb)},
            "factors": {"M1": M1_FACTOR, "M4": M4_FACTOR, "M5_drop": M5_DROP, "M6": M6_FACTOR},
            "verdict": "ELIGIBLE" if len(held) == 6 else GATE_MISSED}


def op_fit_scores(cfg: Config) -> Dict[str, Any]:
    """H-OP's recorded scores (pinned record): S's, which ours must reproduce, and its challenger's (descriptive)."""
    r = json.loads(read_pinned(cfg, OP_FIT_RECORD, cfg.pins[OP_FIT_RECORD]))
    return {"S": {"op_val": r["student_val"], "tb_val": r["retention"]["student"]},
            "H_OP_challenger": {"op_val": r["challenger_val"], "tb_val": r["retention"]["challenger"]}}


# ─────────────────────────────── the fit run ───────────────────────────────

def _load_student_real():   # pragma: no cover - loads the student; never under the build approval
    return H._load_student_real()


def _games_real(cfg: Config) -> Dict[str, Dict[str, Sequence]]:   # pragma: no cover - reads the corpus and H-OP games
    """The move lists the rows replay from: the C0 corpus (digest-checked by the pool) and H-OP's committed games."""
    read_pinned(cfg, OP_GAMES, cfg.pins[OP_GAMES])
    tb = {g["id"]: g["moves"] for g in P.eligible_games(cfg.data_root / P.CORPUS, arms=("train", "val"))}
    op = {H.game_id(g["index"]): g["moves"] for g in H.read_games(cfg.root / OP_GAMES)}
    return {"tb": tb, "op": op}


def run_fit(cfg: Optional[Config] = None, load_student: Callable[[], Any] = _load_student_real,
            games: Callable[[Config], Mapping[str, Mapping[str, Sequence]]] = _games_real,
            save: Callable[[Any, pathlib.Path], Dict[str, str]] = P._save_student_real) -> int:
    cfg = cfg or Config()
    preconditions("fit", cfg)
    P._absent(cfg.ev("fit"), cfg.challenger)
    want = op_fit_scores(cfg)                                     # pinned before anything is loaded
    memory = P._check_available(cfg, cfg.memory_required_bytes, "fit")
    rows = load_sources(cfg)                                      # both label sets as written, verified
    rec = _record("fit", cfg, recipe=dataclasses.asdict(cfg.recipe), epoch_tag=FIT_TAG, updates=cfg.updates,
                  memory_at_start=memory, memory_required_bytes=cfg.memory_required_bytes,
                  sources={s: {k: cfg.sources[s][k] for k in ("label_record_sha256", "selection_sha256", "manifests")}
                           for s in cfg.sources},
                  s_reference=want["S"])

    def body(rec):
        import mlx.core as mx

        # the overlap and E1-E3 checks inside the record boundary: a failure is a recorded STOP (card §5, §13)
        keys = {f"{s}_{a}": [m["key"] for m in rows[s][a][0]] for s in ("op", "tb") for a in ("train", "val")}
        sel = mix_selection(keys, cfg.exclusions(cfg.root), cfg.n_op, cfg.n_tb, cfg.tb_pool)
        rec["selection"] = sel["record"]
        mx.set_default_device(mx.cpu)
        mv = games(cfg)
        st = {f"{s}_{a}": P.ExampleStore(*rows[s][a], mv[s]) for s in ("op", "tb") for a in ("train", "val")}
        plan = mixed_plan(st["op_train"], st["tb_train"], sel["op_order"], sel["tb_order"],
                          cfg.recipe.rows_per_batch // 2)
        if len(plan) != cfg.updates:
            raise Stop("fit", f"{len(plan)} batches; the card's is {cfg.updates} updates")
        reference, student = load_student(), load_student()
        rec["start"] = start_check(reference, plan, n=cfg.start_rows)
        s = {"op_val": P.score(reference, st["op_val"]), "tb_val": P.score(reference, st["tb_val"])}
        rec["student_val"] = s
        rec["s_reproduction"] = s_reproduction(s, want["S"])      # before any update: a cheap stop
        rec["fit"] = fit_mixed(student, reference, plan, cfg.recipe)
        c = {"op_val": P.score(student, st["op_val"]), "tb_val": P.score(student, st["tb_val"])}
        rec["challenger_val"] = c
        rec["gate"] = gate_verdict(c["op_val"], s["op_val"], c["tb_val"], s["tb_val"])
        rec["verdict"] = rec["gate"]["verdict"]
        rec["descriptive"] = {"note": "never gated (card §7): H-OP's challenger's recorded scores on the same rows",
                              "H_OP_challenger": want["H_OP_challenger"]}
        rec["challenger"] = save(student, cfg.challenger)

    return P._terminal(cfg.ev("fit"), rec, body)


# ─────────────────────────────── CLI ───────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="htb_mix")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").add_argument("what", choices=("fit",))
    sub.add_parser("openings")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "openings":                       # pure: re-derive and compare with the committed list
            raw = dump_openings(build_openings())
            same = raw == (ROOT / OPENINGS_REL).read_bytes()
            print(f"{OPENINGS_REL}: {'re-derives byte-equal' if same else 'DIFFERS'} ({_sha(raw)})")
            o = json.loads(raw)
            probs = H.opening_problems({"match": o["match"], "match_smoke": o["match_smoke"]}, forbidden_openings())
            seeds = seed_uniqueness()
            ok = seeds["distinct_pairs"] == seeds["derivations"] and seeds["repeated_values"] == 0
            print(f"openings: {'distinct' if not probs else probs[:5]}; seeds: {seeds}")
            return 0 if same and not probs and ok else 3
        return run_fit()
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return P.EXIT["REFUSED"]


if __name__ == "__main__":
    sys.exit(main())
