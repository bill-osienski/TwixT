> **HISTORICAL RECORD (labeled 2026-07-21):** this is the implementation plan as approved 2026-07-18 and executed 2026-07-18/21 (with in-flight amendments recorded in `docs/updated-v16a-ledger.md` and `.superpowers/sdd/progress.md`). It is not current operational guidance — that lives in `docs/fpu-v2-repair-operator-guide.md`.

# FPU v2 Role-Feasibility Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the v16 policy-mass FPU corpus pipeline after reservoir protocol v1's post-screen GATE-FAIL: make the allocation config-authoritative (schema v2), move targets to late-only, convert the expected capacity failure from a traceback into a controlled gate report, and prove the new profile on the immutable v1 discovery screen — all before any new game generation.

**Architecture:** One new validated `AllocationProfile` object becomes the single source of truth for every result-determining allocation constant; every selection/qualification function gains an `alloc` parameter defaulting to a legacy profile built from the frozen module constants (schema-1 behavior stays byte-identical, the existing 2,183-test suite stays green). A new pure `post-screen-qualify` stage writes an immutable PASS/GATE_FAIL report that `select` requires. Protocol v2 adds `run_kind` (production vs tooling_smoke) fingerprinted through the whole chain. Two new pure commands (`analyze-screen-feasibility`, `sizing-analysis`) run the exact qualifier+selector against the immutable 24,378-row v1 screen as discovery evidence.

**Tech Stack:** Python stdlib only (json/csv/dataclasses/argparse/hashlib/random), pytest. No new dependencies.

## Global Constraints

Copied from the approved spec (docs plan doc dated 2026-07-18) — every task implicitly includes these:

- **reservoir_v1 is an immutable POST-SCREEN GATE-FAIL.** Never edit, delete, top up, reclassify, or select from `logs/eval/fpu_v16_policy_mass_v2/reservoir_v1/` as confirmatory data. It is discovery evidence only. Never overwrite that artifact root.
- **Locked science:** absolute flat/diffuse target definition unchanged (`normalized_entropy >= 0.90 AND top1_prior <= 0.025`); targets required only in `late`; controls stay in all four phases as pooled collateral coverage; NO phase-relative quantile targets; selected-A stays a separate tuning-only gate.
- **Observed capacities (regression fixture):** kept target capacity opening 0 / early_mid 0 / midgame 0 / late 136 (155 kept late-target rows across 86 games, ≤2/game). b400_plus kept late targets: 12 rows in 12 distinct games (7 black / 5 red).
- **Production profile (120 rows):** `target|late` 40 tuning + 20 frozen_check; `control|<each of 4 phases>` 10 tuning + 5 frozen_check. Totals: 80 tuning / 40 frozen_check, 60 target / 60 control. Late-target band minima totals: b400_plus ≥ 8, b300_399 ≥ 12, b200_299 ≥ 12; candidate per-split minima tuning {4, 8, 8} / frozen_check {4, 5, 5} — NOT frozen until the exact selector produces a witness on the v1 screen (Task 14); if infeasible, STOP for a science decision — never silently lower.
- **Frozen_check late-target = 20 is load-bearing:** `DEV_BAND_MIN_N = 20` (`diagnose_fpu_policy_mass.py:99`) is the per-stratum activation minimum.
- **v1 stays byte-identical:** `build_fpu_dev_corpus.py` untouched; schema-1 configs and all existing tests keep exact current behavior. Legacy constants (`SPLIT_ALLOC_V2`, `CORPUS_SIZE`, `LATE_TARGET_FLOORS`, `MAX_PER_GAME`, `MIN_PLY_GAP`, `SIDE_TOL`) survive ONLY as the schema-1 legacy defaults; no schema-2 decision may read them behind the config's back.
- **Import discipline:** `fpu_dev_corpus_v2.py` must NOT top-level-import `fpu_dev_reservoir_protocol` (cycle) — lazy imports inside function bodies only, matching existing sites.
- **Exit codes (shared vocabulary, already defined in `fpu_dev_reservoir_protocol.py:614-617`):** 0 = OK/PASS, 2 = usage/IO, 3 = identity or artifact mismatch, 4 = GATE_FAIL. An expected capacity failure must never traceback.
- **`run_kind` ∈ {`production`, `tooling_smoke`}**; the production diagnostic entry point hard-rejects `tooling_smoke`, and every schema-2 artifact (protocol, config, report, manifest meta, fingerprints) names its `run_kind` so any future consumer can check. The guarantee covers the paths guarded in this plan (Tasks 7–9, 12), not every conceivable consumer — matching the Definition of Done. Smoke output must never be used to select a coefficient, pass a safety gate, justify a strength match, or enter self-play.
- **Board size is 24, explicitly** (final review edit 3): every quantitative claim in this plan (`n_legal = 528 − ply`, the band/ply ranges, the late-only target geometry, all measured capacities) holds on the 24×24 board — the only size played. The generic tooling may accept other board sizes, but v16 production artifacts must pin `board_size: 24` (Task 16's fresh protocol carries it; the smoke protocol copies it from v1). No multi-board-size runs.
- **Do not hardcode `new_collapse_stratum == "ply_bucket"`**; with late-only targets the phase target sub-gate equals the pooled target gate. Never imply a band rate gate ran when its minimum sample was unmet.
- **Post-screen `PASS` means the exact selector succeeded** (review correction 1): the report may only say PASS after a complete dry-run witness from the real deterministic selector exists — capacity bounds alone are necessary, not sufficient. A select failure after PASS is then always corruption or a code defect (raw traceback is correct there).
- **Discovery analyses authenticate their input** (review correction 2): `analyze-screen-feasibility` and `sizing-analysis` require the screen's own config and run the full identity chain (identities, rows-vs-meta, config rederivation) before analyzing. Authentication proves the INPUT is the qualified v1 artifact; it does not make the OUTPUT confirmatory — reports stay `discovery_only`.
- **Schema-1 byte identity is pinned by GOLDENS captured before any change** (review correction 4): Task 0 captures pre-repair sampler and verdict artifacts; every new field (stats profile, fingerprint `run_kind`, stratum census) is emitted for schema 2 only.
- Test command: `.venv/bin/python -m pytest -p no:cacheprovider`. Full-suite baseline: 2183 passed / 0 failed.
- Git: feature branch off `main`, FF-merge when done (user preference: linear history, no `--no-ff`). Commit style: `feat(fpu-v2): ...` / `fix(fpu-v2): ...`.
- **No GPU/operator run is authorized by this plan.** Tasks 15–16 are GATED and require explicit user authorization at execution time.

## File Structure

- Modify: `scripts/GPU/alphazero/fpu_dev_corpus_v2.py` — profile object, threading, report, new CLI modes (Tasks 1–7, 10–11)
- Modify: `scripts/GPU/alphazero/fpu_dev_reservoir_protocol.py` — protocol v2, `run_kind`, schema-2 `derive_config` (Task 8)
- Modify: `scripts/GPU/alphazero/diagnose_fpu_policy_mass.py` — smoke rejection, fingerprint, inactive-gate labeling (Task 9)
- Create: `tests/test_fpu_v2_repair.py` — all new tests (kept in one new file; existing test files edited only where a pinned message/type genuinely changed)
- Create: `tests/goldens/fpu_v2_schema1_sampler_golden.json`, `tests/goldens/fpu_v2_schema1_verdict_golden.json` — pre-repair schema-1 captures (Task 0)
- Create: `docs/fpu-v2-repair-operator-guide.md` — operator guide (Task 13)
- Create at runtime (Task 14): `logs/eval/fpu_v16_policy_mass_v2/analysis/` — discovery-only reports (a NEW artifact root; nothing under `reservoir_v1/` is written)

---

### Task 0: Capture schema-1 goldens (on the UNMODIFIED tree, before any code change)

Comparing `alloc=None` against `AllocationProfile.legacy()` only proves two NEW paths agree with each other. The byte-identity authority must be an artifact captured BEFORE the repair exists.

**Files:**
- Create: `tests/goldens/fpu_v2_schema1_sampler_golden.json`, `tests/goldens/fpu_v2_schema1_verdict_golden.json`
- Create: `tests/test_fpu_v2_repair.py` (golden tests + the shared `_golden_pool()` builder)

- [ ] **Step 1: Build the capture inputs.** `_golden_pool()` must be deterministic and live IN the test file (so the golden test re-derives the same input forever). Reuse the existing suite's feasible-pool builder if one is importable (`grep -n "def make.*pool\|def _make.*kept" tests/test_fpu_dev_corpus_v2.py`); otherwise copy the smallest feasible legacy pool construction from that file inline. For the verdict golden, copy the row/config fixtures the existing `dev_safety_verdict` tests use (`grep -rn "dev_safety_verdict" tests/`).

- [ ] **Step 2: Capture, via a throwaway script in the session scratchpad (NOT committed):** run `v2.sample_v2_rows(_golden_pool(), seed=3)` and dump `{"rows": rows, "stats": stats}` with `json.dumps(..., sort_keys=True)` to the sampler golden; run `diag.dev_safety_verdict(<fixture rows>, <ref>, <cand>)` and dump `verdict.metrics` to the verdict golden.

- [ ] **Step 3: Write the golden tests:**

```python
def test_schema1_sampler_output_matches_pre_repair_golden():
    golden = json.loads(Path(
        "tests/goldens/fpu_v2_schema1_sampler_golden.json").read_text())
    rows, stats = v2.sample_v2_rows(_golden_pool(), seed=3)
    assert json.loads(json.dumps(
        {"rows": rows, "stats": stats}, sort_keys=True)) == golden


def test_schema1_verdict_metrics_match_pre_repair_golden():
    golden = json.loads(Path(
        "tests/goldens/fpu_v2_schema1_verdict_golden.json").read_text())
    verdict = diag.dev_safety_verdict(_golden_verdict_rows(),
                                      _golden_ref(), _golden_cand())
    assert json.loads(json.dumps(verdict.metrics, sort_keys=True)) == golden
```

- [ ] **Step 4: Run (PASS on the unmodified tree), commit:**

```bash
git add tests/goldens tests/test_fpu_v2_repair.py
git commit -m "test(fpu-v2): schema-1 golden pins (pre-repair capture)"
```

Every later task keeps these two tests green — they are the schema-1 byte-identity guard.

---

### Task 1: `AllocationProfile` — the one validated profile object

**Files:**
- Modify: `scripts/GPU/alphazero/fpu_dev_corpus_v2.py` (insert after the `MAX_PER_CELL_PER_GAME` block, i.e. after line ~230, before `proposal_cell_of`)
- Test: `tests/test_fpu_v2_repair.py` (new file)

**Interfaces:**
- Produces: `AllocationProfile` frozen dataclass with fields `schema_version: int`, `run_kind: str`, `allocation: Dict[Tuple[str,str], Dict[str,int]]`, `band_minima_total: Dict[str,int]`, `band_minima_per_split: Dict[str, Dict[str,int]]`, `max_per_game: int`, `min_ply_gap: int`, `side_tol: int`; properties `corpus_size`, `cell_order`, `split_totals`, `quota_by_phase`; method `fingerprint() -> Dict[str, Any]`; classmethod `legacy() -> AllocationProfile`.
- Produces: `parse_allocation_profile(raw: Mapping, *, source: str) -> AllocationProfile` and module constant `PROFILE_RUN_KINDS = ("production", "tooling_smoke")`.
- Consumes: existing constants `SPLIT_ALLOC_V2`, `LATE_TARGET_FLOORS`, `MAX_PER_GAME`, `MIN_PLY_GAP`, `SIDE_TOL`, `PHASES`, `SPLITS`, `LATE_TARGET_CELL`, `LATE_CELL_BANDS`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the v2 role-feasibility repair (schema-2 allocation authority)."""
import copy
import json

import pytest

from scripts.GPU.alphazero import fpu_dev_corpus_v2 as v2


# The spec's production profile, as the schema-2 JSON fields carry it.
PRODUCTION_PROFILE_RAW = {
    "config_schema_version": 2,
    "run_kind": "production",
    "phase_allocation": {
        "target|late":       {"tuning": 40, "frozen_check": 20},
        "control|opening":   {"tuning": 10, "frozen_check": 5},
        "control|early_mid": {"tuning": 10, "frozen_check": 5},
        "control|midgame":   {"tuning": 10, "frozen_check": 5},
        "control|late":      {"tuning": 10, "frozen_check": 5},
    },
    "late_floors": {"b400_plus": 8, "b300_399": 12, "b200_299": 12},
    "late_target_band_minima": {
        "tuning":       {"b400_plus": 4, "b300_399": 8, "b200_299": 8},
        "frozen_check": {"b400_plus": 4, "b300_399": 5, "b200_299": 5},
    },
    "max_per_game": 2,
    "min_ply_gap": 12,
    "side_tol": 2,
    "corpus_size": 120,
}


def test_parse_production_profile_totals():
    p = v2.parse_allocation_profile(PRODUCTION_PROFILE_RAW, source="test")
    assert p.corpus_size == 120
    assert p.split_totals == {"tuning": 80, "frozen_check": 40}
    assert p.quota_by_phase == {
        "opening": 15, "early_mid": 15, "midgame": 15, "late": 75}
    assert p.allocation[("target", "late")] == {"tuning": 40, "frozen_check": 20}
    assert p.run_kind == "production"


def test_legacy_profile_mirrors_module_constants():
    p = v2.AllocationProfile.legacy()
    assert p.schema_version == 1
    assert p.allocation == {c: dict(a) for c, a in v2.SPLIT_ALLOC_V2.items()}
    assert p.corpus_size == v2.CORPUS_SIZE == 240
    assert p.band_minima_total == dict(v2.LATE_TARGET_FLOORS)
    assert p.band_minima_per_split == {}
    assert (p.max_per_game, p.min_ply_gap, p.side_tol) == (
        v2.MAX_PER_GAME, v2.MIN_PLY_GAP, v2.SIDE_TOL)


@pytest.mark.parametrize("mutate, needle", [
    (lambda r: r.__setitem__("config_schema_version", 3), "config_schema_version"),
    (lambda r: r.__setitem__("run_kind", "experiment"), "run_kind"),
    (lambda r: r["phase_allocation"].__setitem__(
        "targetlate", {"tuning": 1, "frozen_check": 1}), "role|phase"),
    (lambda r: r["phase_allocation"].__setitem__(
        "hero|late", {"tuning": 1, "frozen_check": 1}), "role"),
    (lambda r: r["phase_allocation"].__setitem__(
        "target|endgame", {"tuning": 1, "frozen_check": 1}), "phase"),
    (lambda r: r["phase_allocation"]["target|late"].__setitem__("tuning", -1),
     "negative"),
    (lambda r: r["phase_allocation"]["target|late"].__setitem__("tuning", 40.5),
     "integer"),
    (lambda r: r.__setitem__("corpus_size", 121), "corpus_size"),
    (lambda r: r["late_target_band_minima"]["tuning"].__setitem__(
        "b400_plus", 99), "minima"),
    (lambda r: r["late_floors"].__setitem__("b100", 1), "band"),
    # Review correction 5: an incomplete per-split map (e.g. frozen_check
    # silently omitted) must be rejected, not silently accepted.
    (lambda r: r["late_target_band_minima"].pop("frozen_check"), "every split"),
])
def test_parse_rejects_malformed_profiles(mutate, needle):
    raw = copy.deepcopy(PRODUCTION_PROFILE_RAW)
    mutate(raw)
    with pytest.raises(ValueError, match="(?i)" + needle):
        v2.parse_allocation_profile(raw, source="test")


def test_per_split_minima_must_cover_totals():
    raw = copy.deepcopy(PRODUCTION_PROFILE_RAW)
    raw["late_target_band_minima"]["tuning"]["b300_399"] = 3   # 3+5=8 < total 12
    with pytest.raises(ValueError, match="b300_399"):
        v2.parse_allocation_profile(raw, source="test")


def test_fingerprint_covers_the_complete_effective_profile():
    p = v2.parse_allocation_profile(PRODUCTION_PROFILE_RAW, source="test")
    fp = p.fingerprint()
    assert fp["run_kind"] == "production"
    assert fp["allocation"]["target|late"] == {"tuning": 40, "frozen_check": 20}
    assert fp["band_minima_per_split"]["frozen_check"]["b300_399"] == 5
    assert fp["corpus_size"] == 120
    assert fp["max_per_game"] == 2 and fp["min_ply_gap"] == 12 and fp["side_tol"] == 2
    json.dumps(fp, sort_keys=True)   # must be JSON-serializable as-is
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_v2_repair.py -v`
Expected: FAIL / ERROR with `AttributeError: ... has no attribute 'parse_allocation_profile'`

- [ ] **Step 3: Implement**

Insert into `fpu_dev_corpus_v2.py` (after `MAX_PER_CELL_PER_GAME`, before `proposal_cell_of`). Note `_ROLES` and `PROFILE_RUN_KINDS` are new module constants:

```python
# ---------------------------------------------------------------------------
# AllocationProfile -- the ONE validated, schema-2 config-authoritative
# allocation object (repair plan Sec 6). Every result-determining function
# accepts `alloc: AllocationProfile`; `None` means the schema-1 LEGACY profile
# built from the frozen module constants above (v1-era behavior, byte-identical).
# ---------------------------------------------------------------------------

_ROLES: Tuple[str, ...] = ("target", "control")
PROFILE_RUN_KINDS: Tuple[str, ...] = ("production", "tooling_smoke")


@dataclasses.dataclass(frozen=True)
class AllocationProfile:
    schema_version: int
    run_kind: str
    allocation: Dict[Tuple[str, str], Dict[str, int]]
    band_minima_total: Dict[str, int]
    band_minima_per_split: Dict[str, Dict[str, int]]
    max_per_game: int
    min_ply_gap: int
    side_tol: int

    @property
    def corpus_size(self) -> int:
        return sum(a["tuning"] + a["frozen_check"] for a in self.allocation.values())

    @property
    def cell_order(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(self.allocation.keys())

    @property
    def split_totals(self) -> Dict[str, int]:
        return {s: sum(a[s] for a in self.allocation.values()) for s in SPLITS}

    @property
    def quota_by_phase(self) -> Dict[str, int]:
        q: Dict[str, int] = {}
        for (_role, phase), a in self.allocation.items():
            q[phase] = q.get(phase, 0) + a["tuning"] + a["frozen_check"]
        return q

    def fingerprint(self) -> Dict[str, Any]:
        """The COMPLETE effective profile, JSON-shaped -- what reports, manifest
        meta and diagnostic fingerprints record (never merely a file hash)."""
        return {
            "schema_version": self.schema_version,
            "run_kind": self.run_kind,
            "allocation": {f"{r}|{p}": dict(a)
                           for (r, p), a in self.allocation.items()},
            "band_minima_total": dict(self.band_minima_total),
            "band_minima_per_split": {s: dict(m) for s, m
                                      in self.band_minima_per_split.items()},
            "max_per_game": self.max_per_game,
            "min_ply_gap": self.min_ply_gap,
            "side_tol": self.side_tol,
            "corpus_size": self.corpus_size,
        }

    @classmethod
    def legacy(cls) -> "AllocationProfile":
        """Schema-1 profile = the frozen module constants, verbatim. The ONLY
        place the legacy constants are consumed on behalf of selection."""
        return cls(
            schema_version=1, run_kind="production",
            allocation={c: dict(a) for c, a in SPLIT_ALLOC_V2.items()},
            band_minima_total=dict(LATE_TARGET_FLOORS),
            band_minima_per_split={},
            max_per_game=MAX_PER_GAME, min_ply_gap=MIN_PLY_GAP,
            side_tol=SIDE_TOL)


def _profile_int(raw: Any, name: str, source: str, *, minimum: int = 0) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{source}: {name} must be an integer, got {raw!r}")
    if raw < minimum:
        raise ValueError(
            f"{source}: {name} must be >= {minimum} (never negative), got {raw}")
    return raw


def parse_allocation_profile(raw: Mapping[str, Any], *,
                             source: str) -> AllocationProfile:
    """Validate + build the schema-2 profile (repair plan Sec 6's rejection
    list). `source` names the config/profile file in every error."""
    schema = raw.get("config_schema_version")
    if schema != 2:
        raise ValueError(f"{source}: unsupported config_schema_version "
                         f"{schema!r} for an allocation profile (only 2)")
    run_kind = raw.get("run_kind")
    if run_kind not in PROFILE_RUN_KINDS:
        raise ValueError(f"{source}: unsupported run_kind {run_kind!r} "
                         f"(must be one of {PROFILE_RUN_KINDS})")

    allocation: Dict[Tuple[str, str], Dict[str, int]] = {}
    for key, counts in raw["phase_allocation"].items():
        parts = str(key).split("|")
        if len(parts) != 2:
            raise ValueError(f"{source}: malformed role|phase key {key!r}")
        role, phase = parts
        if role not in _ROLES:
            raise ValueError(f"{source}: unknown role {role!r} in {key!r}")
        if phase not in PHASES:
            raise ValueError(f"{source}: unknown phase {phase!r} in {key!r}")
        if set(counts) != set(SPLITS):
            raise ValueError(f"{source}: {key!r} must have exactly the splits "
                             f"{sorted(SPLITS)}, got {sorted(counts)}")
        allocation[(role, phase)] = {
            s: _profile_int(counts[s], f"{key}.{s}", source) for s in SPLITS}
    if not allocation:
        raise ValueError(f"{source}: phase_allocation is empty")

    declared = _profile_int(raw["corpus_size"], "corpus_size", source, minimum=1)
    total = sum(a["tuning"] + a["frozen_check"] for a in allocation.values())
    if declared != total:
        raise ValueError(f"{source}: corpus_size {declared} inconsistent with "
                         f"the allocation total {total}")

    late_alloc = allocation.get(LATE_TARGET_CELL)

    def _band_map(m: Mapping[str, Any], name: str) -> Dict[str, int]:
        out = {}
        for band, n in m.items():
            if band not in LATE_CELL_BANDS:
                raise ValueError(f"{source}: unknown band {band!r} in {name}")
            out[str(band)] = _profile_int(n, f"{name}[{band}]", source)
        return out

    band_minima_total = _band_map(raw["late_floors"], "late_floors")
    band_minima_per_split: Dict[str, Dict[str, int]] = {}
    for split, m in raw["late_target_band_minima"].items():
        if split not in SPLITS:
            raise ValueError(f"{source}: unknown split {split!r} in "
                             f"late_target_band_minima")
        band_minima_per_split[split] = _band_map(
            m, f"late_target_band_minima[{split}]")
    if band_minima_per_split and set(band_minima_per_split) != set(SPLITS):
        raise ValueError(
            f"{source}: late_target_band_minima must name every split "
            f"({sorted(SPLITS)}) or be empty -- a silently omitted split "
            f"would carry no minima at all; got "
            f"{sorted(band_minima_per_split)}")

    if band_minima_total or band_minima_per_split:
        if late_alloc is None:
            raise ValueError(f"{source}: band minima require a "
                             f"{LATE_TARGET_CELL} allocation cell")
        if sum(band_minima_total.values()) > sum(late_alloc.values()):
            raise ValueError(
                f"{source}: late_floors total {sum(band_minima_total.values())} "
                f"exceeds the late-target allocation {sum(late_alloc.values())} "
                f"(minima larger than the associated target allocation)")
        for split, m in band_minima_per_split.items():
            if sum(m.values()) > late_alloc[split]:
                raise ValueError(
                    f"{source}: late_target_band_minima[{split}] total "
                    f"{sum(m.values())} exceeds that split's late-target "
                    f"allocation {late_alloc[split]} (minima larger than the "
                    f"associated target allocation)")
        if band_minima_per_split:
            for band, floor in band_minima_total.items():
                covered = sum(m.get(band, 0)
                              for m in band_minima_per_split.values())
                if covered < floor:
                    raise ValueError(
                        f"{source}: per-split minima for band {band} sum to "
                        f"{covered} < the required total {floor}")

    return AllocationProfile(
        schema_version=2, run_kind=run_kind, allocation=allocation,
        band_minima_total=band_minima_total,
        band_minima_per_split=band_minima_per_split,
        max_per_game=_profile_int(raw["max_per_game"], "max_per_game", source,
                                  minimum=1),
        min_ply_gap=_profile_int(raw["min_ply_gap"], "min_ply_gap", source),
        side_tol=_profile_int(raw["side_tol"], "side_tol", source))
```

- [ ] **Step 4: Run the new tests, then the full suite**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_v2_repair.py -v` → all PASS
Run: `.venv/bin/python -m pytest -p no:cacheprovider -q` → 2183 + new, 0 failed

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/fpu_dev_corpus_v2.py tests/test_fpu_v2_repair.py
git commit -m "feat(fpu-v2): AllocationProfile -- validated schema-2 allocation authority + legacy profile"
```

---

### Task 2: Schema-2 config loading + `profile_for(config)`

**Files:**
- Modify: `scripts/GPU/alphazero/fpu_dev_corpus_v2.py` — `V2Config` (line ~1842), `_V2_CONFIG_REQUIRED_KEYS` (line ~1821), `load_v2_config` (line ~1927)
- Test: `tests/test_fpu_v2_repair.py`

**Interfaces:**
- Produces: `V2Config` gains optional fields (all default `None`): `run_kind`, `late_target_band_minima`, `max_per_game`, `min_ply_gap`, `side_tol`, `corpus_size`, `post_screen_report_out`. New constant `_V2_CONFIG_REQUIRED_KEYS_SCHEMA2` (the 7 new keys). New function `profile_for(config: V2Config) -> AllocationProfile`.
- Consumes: Task 1's `parse_allocation_profile`, `AllocationProfile.legacy`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_fpu_v2_repair.py`)

```python
SCHEMA1_CONFIG_KEYS = {
    "source_index_path": "idx.jsonl", "seed_range": [0, 4800],
    "selection_seed": 7, "phase_allocation": {}, "late_floors": {},
    "enumerator_params": {}, "new_collapse_stratum": "ply_bucket",
    "checkpoint": "ck.safetensors", "forbidden_manifests": [],
    "screen_out": "s.csv", "select_out": "m.csv",
    "expected_fingerprints": {}, "config_schema_version": 1,
    "protocol_path": "p.json", "match_summary_path": "ms.json",
    "replay_dir": "replays", "report_out": "r.json",
}


def _write_config(tmp_path, extra):
    cfg = dict(SCHEMA1_CONFIG_KEYS)
    cfg.update(extra)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def test_schema1_config_loads_and_yields_legacy_profile(tmp_path):
    cfg = v2.load_v2_config(_write_config(tmp_path, {}))
    assert cfg.run_kind is None            # schema-1 carries none of the new keys
    p = v2.profile_for(cfg)
    assert p.schema_version == 1
    assert p.corpus_size == 240


def test_schema2_config_missing_new_keys_is_rejected(tmp_path):
    path = _write_config(tmp_path, {"config_schema_version": 2})
    with pytest.raises(ValueError, match="run_kind"):
        v2.load_v2_config(path)


def test_schema2_config_round_trips_into_a_profile(tmp_path):
    extra = dict(PRODUCTION_PROFILE_RAW)
    extra["post_screen_report_out"] = "psq.json"
    cfg = v2.load_v2_config(_write_config(tmp_path, extra))
    p = v2.profile_for(cfg)
    assert p.schema_version == 2 and p.corpus_size == 120
    assert cfg.post_screen_report_out == "psq.json"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_fpu_v2_repair.py -v -k schema`
Expected: FAIL (`V2Config` has no `run_kind`, `profile_for` undefined)

- [ ] **Step 3: Implement**

(a) Append to the `V2Config` dataclass (after `stall_flush_sims`, keeping defaults last):

```python
    # Schema-2 (repair plan) fields -- None on a schema-1 config. `profile_for`
    # is the ONLY consumer; the loader enforces presence when
    # config_schema_version >= 2.
    run_kind: Optional[str] = None
    late_target_band_minima: Optional[Dict[str, Any]] = None
    max_per_game: Optional[int] = None
    min_ply_gap: Optional[int] = None
    side_tol: Optional[int] = None
    corpus_size: Optional[int] = None
    post_screen_report_out: Optional[str] = None
```

(b) After the `_V2_CONFIG_REQUIRED_KEYS` tuple add:

```python
# The ADDITIONAL top-level keys a schema-2 (repair plan Sec 6) config must
# carry. Enforced by load_v2_config only when config_schema_version >= 2, so
# schema-1 configs (reservoir_v1's generation) keep loading exactly as before.
_V2_CONFIG_REQUIRED_KEYS_SCHEMA2: Tuple[str, ...] = (
    "run_kind", "late_target_band_minima", "max_per_game",
    "min_ply_gap", "side_tol", "corpus_size", "post_screen_report_out")
```

(c) In `load_v2_config`, after the existing `missing` check, add:

```python
    if int(raw.get("config_schema_version", 0)) >= 2:
        missing2 = sorted(k for k in _V2_CONFIG_REQUIRED_KEYS_SCHEMA2
                          if k not in raw)
        if missing2:
            raise ValueError(
                f"load_v2_config: {path} declares config_schema_version "
                f"{raw['config_schema_version']} but is missing required "
                f"schema-2 key(s): {', '.join(missing2)}")
```

and extend the `V2Config(...)` construction:

```python
        run_kind=raw.get("run_kind"),
        late_target_band_minima=raw.get("late_target_band_minima"),
        max_per_game=raw.get("max_per_game"),
        min_ply_gap=raw.get("min_ply_gap"),
        side_tol=raw.get("side_tol"),
        corpus_size=raw.get("corpus_size"),
        post_screen_report_out=raw.get("post_screen_report_out"),
```

(d) After `load_v2_config` add:

```python
def profile_for(config: V2Config) -> AllocationProfile:
    """The config's effective AllocationProfile. Schema 1 -> the frozen legacy
    constants (byte-identical v1-era behavior); schema 2 -> parsed + validated
    from the config's own fields. THE one bridge from config to allocation --
    no production decision reads SPLIT_ALLOC_V2/CORPUS_SIZE/LATE_TARGET_FLOORS/
    MAX_PER_GAME/MIN_PLY_GAP/SIDE_TOL behind this function's back."""
    if config.config_schema_version < 2:
        return AllocationProfile.legacy()
    return parse_allocation_profile({
        "config_schema_version": config.config_schema_version,
        "run_kind": config.run_kind,
        "phase_allocation": config.phase_allocation,
        "late_floors": config.late_floors,
        "late_target_band_minima": config.late_target_band_minima,
        "max_per_game": config.max_per_game,
        "min_ply_gap": config.min_ply_gap,
        "side_tol": config.side_tol,
        "corpus_size": config.corpus_size,
    }, source=config.config_path)
```

- [ ] **Step 4: Run tests + full suite** (same commands as Task 1 Step 4)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/fpu_dev_corpus_v2.py tests/test_fpu_v2_repair.py
git commit -m "feat(fpu-v2): schema-2 config keys + profile_for(config) bridge"
```

---

### Task 3: Pure qualification report + the real-failure regression fixture

**Files:**
- Modify: `scripts/GPU/alphazero/fpu_dev_corpus_v2.py` — extract `_capacity_shortfalls` from `_capacity_precheck` (line 525); add `post_screen_qualification_report` next to `post_screen_qualification` (line 2835)
- Test: `tests/test_fpu_v2_repair.py`

**Interfaces:**
- Produces: `_capacity_shortfalls(games_profile, alloc) -> List[str]` (pure; the two upper bounds, returning failure strings instead of raising); `post_screen_qualification_report(kept_rows, alloc) -> Dict` with keys `status` (`"PASS"`/`"GATE_FAIL"`), `binding_constraint`, `failures`, `cells` (per `role|phase`: demand/capacity/n_rows/n_games/red/black), `global_realizable_capacity`, `late_target_bands` (per band: minimum_total/minimum_per_split/capacity/n_games/red/black), `profile` (the fingerprint).
- Consumes: Task 1's `AllocationProfile`; existing `_games_and_profile`, `LATE_TARGET_CELL`.

- [ ] **Step 1: Write the failing tests** (append; this fixture is reused by Tasks 4–5)

```python
def _kept_row(game_idx, ply, side, phase, band, role, tag=""):
    return {
        "game_idx": game_idx, "ply": ply, "side": side, "phase": phase,
        "band": band, "role": role, "n_legal": 528 - ply,
        "canonical_sha1": f"sha-{game_idx}-{ply}-{side}{tag}",
        "root_value_stm": 0.0, "normalized_entropy": 0.5, "top1_prior": 0.1,
        "top4_mass": 0.4, "top8_mass": 0.6,
    }


def make_gate_fail_fixture():
    """COUNT- AND GEOMETRY-FAITHFUL analogue of the OBSERVED reservoir_v1
    kept pool (review correction 5): 155 late-target rows in 86 games
    (36x1-row + 31x2-row + 19x3-row), realizable exactly 136 under <=2/game;
    bands 12 b400_plus (12 games, ONE row each, 7 black / 5 red) /
    52 b300_399 / 91 b200_299; every ply lies in its band's geometric range
    on a 24 board (n_legal = 528 - ply: b400_plus needs ply <= 128, b300_399
    ply 129-228, b200_299 ply 229-328; late phase needs ply >= 91). Targets
    0/0/0 outside late; controls ample in all four phases."""
    rows, gi = [], 0
    # The real b400 scarcity: 12 single-row games, sides 7 black / 5 red.
    for i in range(12):
        rows.append(_kept_row(gi, 91 + i, "black" if i < 7 else "red",
                              "late", "b400_plus", "target"))
        gi += 1
    # Deal the remaining 143 rows (52 b300 + 91 b200) into 24 one-row +
    # 31 two-row + 19 three-row games (per-game overlap represented).
    bands = ["b300_399"] * 52 + ["b200_299"] * 91
    BAND_BASE_PLY = {"b300_399": 150, "b200_299": 240}
    it = iter(bands)
    for size in [1] * 24 + [2] * 31 + [3] * 19:
        for k in range(size):
            band = next(it)
            rows.append(_kept_row(
                gi, BAND_BASE_PLY[band] + 13 * k,      # >=12-ply spacing
                "red" if (gi + k) % 2 == 0 else "black",
                "late", band, "target"))
        gi += 1
    # Ample controls: 40 games per phase, one red+one black control each
    # (ply in the opening/early_mid/midgame/late ranges; n_legal consistent).
    for phase, base_ply in (("opening", 2), ("early_mid", 20),
                            ("midgame", 50), ("late", 95)):
        for _ in range(40):
            rows.append(_kept_row(gi, base_ply, "red", phase, "b400_plus",
                                  "control"))
            rows.append(_kept_row(gi, base_ply + 12, "black", phase,
                                  "b400_plus", "control"))
            gi += 1
    return rows


def test_gate_fail_fixture_is_faithful_to_the_measured_screen():
    late_targets = [r for r in make_gate_fail_fixture()
                    if (r["role"], r["phase"]) == ("target", "late")]
    assert len(late_targets) == 155
    assert len({r["game_idx"] for r in late_targets}) == 86
    per_game = Counter(r["game_idx"] for r in late_targets)
    assert sum(min(2, n) for n in per_game.values()) == 136
    b400 = [r for r in late_targets if r["band"] == "b400_plus"]
    assert len(b400) == 12 == len({r["game_idx"] for r in b400})
    assert Counter(r["side"] for r in b400) == {"black": 7, "red": 5}
    assert Counter(r["band"] for r in late_targets) == {
        "b400_plus": 12, "b300_399": 52, "b200_299": 91}
    for r in late_targets:                     # geometry: band matches n_legal
        n = 528 - r["ply"]
        assert r["band"] == ("b400_plus" if n >= 400
                             else "b300_399" if n >= 300 else "b200_299")


def test_old_allocation_gate_fails_naming_target_opening():
    report = v2.post_screen_qualification_report(
        make_gate_fail_fixture(), v2.AllocationProfile.legacy())
    assert report["status"] == "GATE_FAIL"
    assert "target|opening" in report["binding_constraint"]
    assert "capacity 0" in report["binding_constraint"]
    assert "demand 45" in report["binding_constraint"]
    assert report["cells"]["target|opening"]["capacity"] == 0
    assert report["cells"]["target|late"]["capacity"] == 136


def test_new_production_profile_passes_capacity_on_the_fixture():
    alloc = v2.parse_allocation_profile(PRODUCTION_PROFILE_RAW, source="test")
    report = v2.post_screen_qualification_report(
        make_gate_fail_fixture(), alloc)
    assert report["status"] == "PASS"
    assert report["binding_constraint"] is None
    assert report["late_target_bands"]["b400_plus"]["capacity"] == 12
    assert report["late_target_bands"]["b400_plus"]["n_games"] == 12
    assert report["profile"]["run_kind"] == "production"


def test_band_capacity_below_total_minimum_gate_fails():
    rows = [r for r in make_gate_fail_fixture()
            if not (r["role"] == "target" and r["band"] == "b400_plus")]
    alloc = v2.parse_allocation_profile(PRODUCTION_PROFILE_RAW, source="test")
    report = v2.post_screen_qualification_report(rows, alloc)
    assert report["status"] == "GATE_FAIL"
    assert any("b400_plus" in f for f in report["failures"])
```

- [ ] **Step 2: Run to verify failure** — `post_screen_qualification_report` undefined.

- [ ] **Step 3: Implement**

(a) Rework `_capacity_precheck` (line 525) into a thin raiser over a new pure helper; the helper takes `alloc` (this also pre-wires Task 4's threading — `_capacity_precheck` gains the `alloc` parameter now):

```python
def _capacity_shortfalls(
        games_profile: Mapping[Any, Mapping[Tuple[str, str], int]],
        alloc: AllocationProfile) -> List[str]:
    """The two GENUINE-infeasibility upper bounds, as failure strings (empty =
    both hold). Single-sourced: `_capacity_precheck` raises on the first,
    `post_screen_qualification_report` records them all."""
    failures: List[str] = []
    capacity: Counter = Counter()
    for prof in games_profile.values():
        for cell, n in prof.items():
            if cell in alloc.allocation:
                capacity[cell] += min(n, alloc.max_per_game)
    for cell, a in alloc.allocation.items():
        demand = a["tuning"] + a["frozen_check"]
        have = capacity.get(cell, 0)
        if have < demand:
            failures.append(f"cell {cell} capacity {have} < demand {demand}")
    global_capacity = sum(
        min(sum(n for cell, n in prof.items() if cell in alloc.allocation),
            alloc.max_per_game)
        for prof in games_profile.values())
    if global_capacity < alloc.corpus_size:
        failures.append(
            f"global capacity {global_capacity} < corpus size "
            f"{alloc.corpus_size} under the global <={alloc.max_per_game} "
            f"per-game rule ({len(games_profile)} games)")
    return failures


def _capacity_precheck(
        games_profile: Mapping[Any, Mapping[Tuple[str, str], int]],
        *, where: str = "assign_split_v2",
        alloc: Optional[AllocationProfile] = None) -> None:
    # (docstring: keep the existing one, add one line: "`alloc` None = the
    # schema-1 legacy profile.")
    alloc = alloc if alloc is not None else AllocationProfile.legacy()
    failures = _capacity_shortfalls(games_profile, alloc)
    if failures:
        raise ValueError(f"{where}: {failures[0]}")
```

The existing raise messages are preserved verbatim (`"{where}: cell {cell} capacity {have} < demand {demand}"` and the global one) — do NOT reword them; existing tests may pin them.

(b) Add after `post_screen_qualification` (line ~2897):

```python
def post_screen_qualification_report(
        kept_rows: List[dict], alloc: AllocationProfile) -> Dict[str, Any]:
    """The CONTROLLED post-screen qualification verdict (repair plan Sec 7):
    every configured (role, phase) cell's capacity vs demand, the global
    <=max_per_game bound, and the late-target band capacities vs the TOTAL
    minima -- as a JSON-shaped report, never a raise. Pure. NECESSARY bounds
    only: per-SPLIT band minima are provable only by the exact selector."""
    _games, gprofile = _games_and_profile(kept_rows)
    mpg = alloc.max_per_game
    failures = _capacity_shortfalls(gprofile, alloc)
    # Re-key the shortfalls for the report's cells table (same numbers).
    cells: Dict[str, Any] = {}
    for (role, phase), a in alloc.allocation.items():
        contributing = {gi: prof[(role, phase)] for gi, prof in
                        gprofile.items() if prof.get((role, phase))}
        rows = [r for r in kept_rows if (r["role"], r["phase"]) == (role, phase)]
        sides = Counter(r["side"] for r in rows)
        cells[f"{role}|{phase}"] = {
            "demand": a["tuning"] + a["frozen_check"],
            "capacity": sum(min(n, mpg) for n in contributing.values()),
            "n_rows": len(rows), "n_games": len(contributing),
            "red": sides.get("red", 0), "black": sides.get("black", 0)}
    global_capacity = sum(
        min(sum(n for cell, n in prof.items() if cell in alloc.allocation), mpg)
        for prof in gprofile.values())

    late_rows = [r for r in kept_rows
                 if (r["role"], r["phase"]) == LATE_TARGET_CELL]
    by_game_band: Dict[Any, Counter] = defaultdict(Counter)
    for r in late_rows:
        by_game_band[r["game_idx"]][r["band"]] += 1
    bands: Dict[str, Any] = {}
    for band, minimum in alloc.band_minima_total.items():
        band_capacity = sum(min(c[band], mpg) for c in by_game_band.values())
        sides = Counter(r["side"] for r in late_rows if r["band"] == band)
        bands[band] = {
            "minimum_total": minimum,
            "minimum_per_split": {
                s: alloc.band_minima_per_split.get(s, {}).get(band, 0)
                for s in SPLITS},
            "capacity": band_capacity,
            "n_games": sum(1 for c in by_game_band.values() if c[band]),
            "red": sides.get("red", 0), "black": sides.get("black", 0)}
        if band_capacity < minimum:
            failures.append(
                f"late-target band {band} capacity {band_capacity} < "
                f"total minimum {minimum}")

    # The report's binding constraint uses the cells-table naming for the
    # first per-cell failure ("role|phase: ..." reads better in a report than
    # the raise's tuple), but keeps the raise-compatible substrings.
    binding = None
    if failures:
        binding = failures[0].replace(
            "cell ('", "").replace("', '", "|").replace("')", "") \
            if failures[0].startswith("cell (") else failures[0]
    return {
        "status": "PASS" if not failures else "GATE_FAIL",
        "binding_constraint": binding,
        "failures": failures,
        "cells": cells,
        "global_realizable_capacity": global_capacity,
        "late_target_bands": bands,
        "per_split_minima_note": (
            "per-split band minima are proven only by the exact selector "
            "witness, not by this capacity bound"),
        "profile": alloc.fingerprint(),
    }
```

Note on `binding_constraint`: after the transform, a per-cell failure reads `target|opening capacity 0 < demand 45` — which satisfies the regression test's three substring asserts (`"target|opening"`, `"capacity 0"`, `"demand 45"`). If the inline transform proves fragile in practice, build the string directly in the cells loop instead — behavior over cleverness.

- [ ] **Step 4: Run tests + full suite.** All existing `_capacity_precheck` callers still pass no `alloc` → legacy → identical messages; the full suite must stay green with zero edits to existing tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/fpu_dev_corpus_v2.py tests/test_fpu_v2_repair.py
git commit -m "feat(fpu-v2): controlled post-screen qualification report + 0/0/0/136 regression fixture"
```

---

### Task 4: Thread `alloc` through greedy/sampler/qualification/select

**Files:**
- Modify: `scripts/GPU/alphazero/fpu_dev_corpus_v2.py` — `_greedy_assign_v2` (410), `assign_split_v2` (578), `_pickable` (604), `_select_manifest` (682), `sample_v2_rows` (984), `post_screen_qualification` (2835), `select_final_manifest` (2972)
- Test: `tests/test_fpu_v2_repair.py`

**Interfaces:**
- Produces: every listed function gains keyword `alloc: Optional[AllocationProfile] = None` (None → `AllocationProfile.legacy()`), except `_select_manifest` and `_pickable`/`_greedy_assign_v2` where it is a required positional-after-existing parameter (internal). `select_final_manifest` computes `alloc = profile_for(config)` itself and passes it down — later tasks rely on that.
- Consumes: Tasks 1–3.

- [ ] **Step 1: Write the failing tests** (append)

```python
def make_feasible_120_pool():
    """A pool the PRODUCTION profile can exactly fill: 70 late-target pair
    games (bands 14/28/28) + 40 control pair-games per phase. Pairs are
    side-opposed, >=12 plies apart, and BAND-CONSISTENT with their own plies
    (n_legal = 528 - ply stays inside the pair's band). Deliberately more
    generous than the real screen (2 b400 rows/game) -- real-scarcity
    tightness is exercised by the faithful gate-fail fixture and the Task 14
    run against the actual screen."""
    BAND_PAIR_PLIES = {"b400_plus": (100, 114), "b300_399": (150, 164),
                       "b200_299": (240, 254)}
    rows = []
    gi = 0
    for band in (["b400_plus"] * 14 + ["b300_399"] * 28 + ["b200_299"] * 28):
        p0, p1 = BAND_PAIR_PLIES[band]
        rows.append(_kept_row(gi, p0, "red", "late", band, "target"))
        rows.append(_kept_row(gi, p1, "black", "late", band, "target"))
        gi += 1
    for phase, base_ply in (("opening", 2), ("early_mid", 20),
                            ("midgame", 50), ("late", 95)):
        for _ in range(40):
            rows.append(_kept_row(gi, base_ply, "red", phase, "b400_plus",
                                  "control"))
            rows.append(_kept_row(gi, base_ply + 12, "black", phase,
                                  "b400_plus", "control"))
            gi += 1
    return rows


def _production_alloc():
    return v2.parse_allocation_profile(PRODUCTION_PROFILE_RAW, source="test")


def test_sampler_fills_the_production_profile_exactly():
    rows, stats = v2.sample_v2_rows(make_feasible_120_pool(), seed=11,
                                    alloc=_production_alloc())
    assert stats["n_rows"] == 120
    assert stats["cell_counts"]["target|late|tuning"] == 40
    assert stats["cell_counts"]["target|late|frozen_check"] == 20
    assert stats["cell_counts"]["control|opening|tuning"] == 10
    splits = Counter(r["split"] for r in rows)
    assert splits == {"tuning": 80, "frozen_check": 40}
    roles = Counter(r["role"] for r in rows)
    assert roles == {"target": 60, "control": 60}
    # Whole-game split isolation, global <=2/game, >=12-ply gap, no dup hashes.
    split_by_game, plies_by_game = {}, {}
    for r in rows:
        assert split_by_game.setdefault(r["game_idx"], r["split"]) == r["split"]
        plies_by_game.setdefault(r["game_idx"], []).append(r["ply"])
    for plies in plies_by_game.values():
        assert len(plies) <= 2
        if len(plies) == 2:
            assert abs(plies[0] - plies[1]) >= 12
    hashes = [r["canonical_sha1"] for r in rows]
    assert len(hashes) == len(set(hashes))


def test_sampler_is_deterministic_for_a_profile():
    a = v2.sample_v2_rows(make_feasible_120_pool(), seed=11,
                          alloc=_production_alloc())
    b = v2.sample_v2_rows(make_feasible_120_pool(), seed=11,
                          alloc=_production_alloc())
    assert a == b


def test_mutating_module_constants_cannot_change_a_schema2_result(monkeypatch):
    alloc = _production_alloc()
    before = v2.sample_v2_rows(make_feasible_120_pool(), seed=11, alloc=alloc)
    monkeypatch.setattr(v2, "CORPUS_SIZE", 999)
    monkeypatch.setattr(v2, "MAX_PER_GAME", 1)
    monkeypatch.setattr(v2, "MIN_PLY_GAP", 100)
    monkeypatch.setattr(v2, "SIDE_TOL", 0)
    monkeypatch.setattr(v2, "SPLIT_ALLOC_V2",
                        {("target", "opening"): {"tuning": 1,
                                                 "frozen_check": 1}})
    after = v2.sample_v2_rows(make_feasible_120_pool(), seed=11, alloc=alloc)
    assert after == before


def test_qualification_raise_matches_report_verdict():
    fixture = make_gate_fail_fixture()
    with pytest.raises(ValueError, match="target.*opening"):
        v2.post_screen_qualification(fixture)          # legacy default
    v2.post_screen_qualification(fixture, alloc=_production_alloc())  # no raise
```

- [ ] **Step 2: Run to verify failure** — `sample_v2_rows` rejects the `alloc` kwarg.

- [ ] **Step 3: Implement.** Mechanical threading; the AllocationProfile parameter is named `alloc` everywhere (never `profile` — that name is taken by the per-game counts mapping):

  1. `_greedy_assign_v2(games_profile, seed, attempt, alloc)` — new required 4th param. Inside: `SPLIT_ALLOC_V2` → `alloc.allocation` (line 462 `need = ...`), `MAX_PER_GAME` → `alloc.max_per_game`, any `SPLIT_TOTALS` read → `alloc.split_totals`. Read the function body fully before editing; replace every constant read.
  2. `assign_split_v2(games_profile, seed, *, attempt=0, alloc=None)` — resolve `alloc = alloc or AllocationProfile.legacy()`; pass to `_capacity_precheck(..., alloc=alloc)` and `_greedy_assign_v2(..., alloc)`.
  3. `_pickable(rows_of_game, cell, band, used_sha1, chosen_plies, min_gap)` — new required `min_gap` param replacing the `MIN_PLY_GAP` read at line 619; update both call sites.
  4. `_select_manifest(games, profile, split_of, alloc)` — new required 4th param. Replace: `CELL_ORDER_V2` (869) → `alloc.cell_order`; `SPLIT_ALLOC_V2[cell][split]` (870) → `alloc.allocation[cell][split]`; `LATE_TARGET_FLOORS` (887, 908) → `alloc.band_minima_total`; `MAX_PER_GAME` (741) → `alloc.max_per_game`; `MIN_PLY_GAP` (743, and via `_pickable`) → `alloc.min_ply_gap`; `SIDE_TOL` (932) → `alloc.side_tol`; stats `for (role, phase) in SPLIT_ALLOC_V2` (950) → `alloc.allocation`.
  5. `sample_v2_rows(kept, *, seed, alloc=None)` — resolve legacy default once at the top; pass `alloc` to `_capacity_precheck`, `_greedy_assign_v2`, `_select_manifest`.
  6. `post_screen_qualification(kept_rows, alloc=None)` — resolve legacy default; pass `alloc=` to `_capacity_precheck`; replace the floor loop's `LATE_TARGET_FLOORS`/`MAX_PER_GAME` reads with `alloc.band_minima_total`/`alloc.max_per_game`.
  7. `select_final_manifest` — after step 5 (forbidden check), insert `alloc = profile_for(config)`; change step 7 to `post_screen_qualification(kept, alloc=alloc)` and step 8 to `sample_v2_rows(kept, seed=config.selection_seed, alloc=alloc)`; and — **schema 2 ONLY** (review correction 4: schema-1 artifact bytes must not change; Task 0's goldens are the guard):

```python
    if config.config_schema_version >= 2:
        stats["allocation_profile"] = alloc.fingerprint()
        stats["run_kind"] = alloc.run_kind
```

- [ ] **Step 4: Run tests + full suite.** The legacy default must keep every existing sampler/qualification/select test green without edits. If any existing test constructed `_select_manifest`/`_greedy_assign_v2` calls directly (they are private; grep `tests/` for `_select_manifest(` and `_greedy_assign_v2(`), update those call sites to pass `v2.AllocationProfile.legacy()` — that is a signature accommodation, not a behavior change.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/fpu_dev_corpus_v2.py tests/test_fpu_v2_repair.py
git commit -m "feat(fpu-v2): thread AllocationProfile through greedy/sampler/qualification/select (schema-1 legacy default)"
```

---

### Task 5: Per-split late-target band minima in the sampler

**Files:**
- Modify: `scripts/GPU/alphazero/fpu_dev_corpus_v2.py` — `_select_manifest` (682–962)
- Test: `tests/test_fpu_v2_repair.py`

**Interfaces:**
- Produces: `_select_manifest` enforces `alloc.band_minima_per_split` exactly (floor pass + selected-rows verification); stats gains `"late_target_band_count_by_split"`.
- Consumes: Task 4's threaded `_select_manifest(games, profile, split_of, alloc)`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_per_split_band_minima_hold_on_selected_rows():
    rows, stats = v2.sample_v2_rows(make_feasible_120_pool(), seed=11,
                                    alloc=_production_alloc())
    by_split = {s: Counter() for s in ("tuning", "frozen_check")}
    for r in rows:
        if (r["role"], r["phase"]) == ("target", "late"):
            by_split[r["split"]][r["band"]] += 1
    assert by_split["tuning"]["b400_plus"] >= 4
    assert by_split["tuning"]["b300_399"] >= 8
    assert by_split["tuning"]["b200_299"] >= 8
    assert by_split["frozen_check"]["b400_plus"] >= 4
    assert by_split["frozen_check"]["b300_399"] >= 5
    assert by_split["frozen_check"]["b200_299"] >= 5
    assert stats["late_target_band_count_by_split"]["tuning"]["b400_plus"] >= 4


@pytest.mark.parametrize("starve_band", ["b400_plus", "b300_399", "b200_299"])
def test_insufficient_band_capacity_fails_by_name(starve_band):
    pool = [r for r in make_feasible_120_pool()
            if not (r["role"] == "target" and r["band"] == starve_band)]
    with pytest.raises(ValueError, match=starve_band):
        v2.sample_v2_rows(pool, seed=11, alloc=_production_alloc())


def test_legacy_profile_selection_is_unchanged_by_the_split_minima_code():
    # Golden guard: the legacy path must not change. Reuse an existing
    # feasible legacy pool builder from tests/test_fpu_dev_corpus_v2.py --
    # import the module and call its pool fixture/helper (grep for the
    # helper the existing sampler tests use, e.g. a make-*-pool function),
    # select with alloc=None and with alloc=AllocationProfile.legacy(),
    # and assert the two (rows, stats) results are equal.
    import tests.test_fpu_dev_corpus_v2 as legacy_tests
    pool = legacy_tests.make_realistic_feasible_pool()  # adjust to the real name
    assert (v2.sample_v2_rows(pool, seed=3) ==
            v2.sample_v2_rows(pool, seed=3,
                              alloc=v2.AllocationProfile.legacy()))
```

(For the third test: the existing test file has pool builders for the sampler suite — find the real helper name with `grep -n "def make.*pool\|def _make.*kept" tests/test_fpu_dev_corpus_v2.py` and use it; if none is importable, build the pool inline by copying the smallest existing feasible-pool construction from that file.)

- [ ] **Step 2: Run to verify failure** — per-split assertion fails / `late_target_band_count_by_split` missing.

- [ ] **Step 3: Implement** in `_select_manifest`:

  1. Add running state next to `floor_count` (line 723):

```python
    floor_count_by_split: Dict[str, Counter] = {s: Counter() for s in SPLITS}
```

  2. In `take` (line 769), after `floor_count[r["band"]] += 1` add:

```python
                floor_count_by_split[split][r["band"]] += 1
```

  3. Replace the floor-pass budget (lines 886–890) so the pass draws while EITHER the split's own minimum or the global total is unmet (legacy `band_minima_per_split == {}` reduces `need_split` to 0 — behavior byte-identical):

```python
            if cell == LATE_TARGET_CELL:
                floor_bands = dict(alloc.band_minima_total)
                for m in alloc.band_minima_per_split.values():
                    for b in m:
                        floor_bands.setdefault(b, 0)
                for band, floor in floor_bands.items():
                    def floor_budget(_band=band, _floor=floor, _split=split,
                                     _ord=ordinary_budget):
                        need_total = _floor - floor_count[_band]
                        need_split = (alloc.band_minima_per_split
                                      .get(_split, {}).get(_band, 0)
                                      - floor_count_by_split[_split][_band])
                        return min(_ord(), max(need_total, need_split, 0))
                    fill(cand_games, cell, split, band, floor_budget)
```

  4. After the existing total-floor verification (lines 905–913), add the per-split verification and stats, both counted FROM THE SELECTED ROWS:

```python
    late_by_split: Dict[str, Counter] = {s: Counter() for s in SPLITS}
    for r in selected:
        if (r["role"], r["phase"]) == LATE_TARGET_CELL:
            late_by_split[r["split"]][r["band"]] += 1
    for split, minima in alloc.band_minima_per_split.items():
        for band, m in minima.items():
            if late_by_split[split][band] < m:
                raise ValueError(
                    f"per-split late-target band minimum unmet: split {split} "
                    f"band {band} has {late_by_split[split][band]} of the "
                    f"required {m}")
```

and in `stats`:

```python
        "late_target_band_count_by_split": {
            s: dict(sorted(late_by_split[s].items())) for s in SPLITS},
```

- [ ] **Step 4: Run tests + full suite** (legacy-equality test is the guard that the floor-pass rewrite is byte-identical for schema-1).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/fpu_dev_corpus_v2.py tests/test_fpu_v2_repair.py
git commit -m "feat(fpu-v2): per-split late-target band minima -- floor pass + selected-rows verification"
```

---

### Task 6: Geometric preflight — per-phase quotas (non-uniform, odd-safe)

**Files:**
- Modify: `scripts/GPU/alphazero/fpu_dev_corpus_v2.py` — `v2_geometry_feasibility` (1422), `_build_v2_witness` (1246), `v2_preflight_source` (1554), `run_screen` preflight call (~2155)
- Test: `tests/test_fpu_v2_repair.py`

**Interfaces:**
- Produces: `v2_geometry_feasibility(..., quota_per_phase: Union[int, Mapping[str,int]] = QUOTA_PER_PHASE, split_totals: Optional[Mapping[str,int]] = None, ...)`; `v2_preflight_source(records, alloc=None)`; `run_screen` passes the config's profile. `V2PreflightReport.quota_per_phase` becomes the normalized per-phase dict.
- Consumes: Task 2's `profile_for`.

Background (why this is not a constant swap): the current witness assumes uniform, even, exactly-tiling quotas (`quota // 2` pairs per phase; split placement checks `!=` equality, lines 1318/1412). The production profile's quotas are 15/15/15/75 — non-uniform AND odd — so the witness must (a) take per-phase quotas, (b) use ceil pair counts (over-reserve by ≤1 row per phase, strictly conservative), and (c) verify split budgets as COVERAGE (`>=`), because ceil over-selection can exceed the exact totals. All three changes preserve the legacy result for the uniform/even/tiling legacy profile (ceil == floor for even quotas; over-selection never occurs, so `>=` coverage passes exactly where `==` did).

- [ ] **Step 1: Write the failing tests** (append)

```python
def _pair_proposals(gi, phase, ply, band):
    cell = (phase, None) if phase != "late" else ("late", band)
    return [
        {"game_idx": gi, "ply": ply, "side": "red", "phase": phase,
         "n_legal": 528 - ply, "band": band, "proposal_cell": cell},
        {"game_idx": gi, "ply": ply + 12, "side": "black", "phase": phase,
         "n_legal": 528 - ply - 12, "band": band, "proposal_cell": cell},
    ]


def make_production_geometry():
    """Whole-pair candidate geometry ample for quotas 15/15/15/75 + the
    candidate floors 8/12/12."""
    by_game, gi = {}, 0
    for phase, ply in (("opening", 2), ("early_mid", 20), ("midgame", 50)):
        for _ in range(12):                       # 12 pairs = 24 >= 15
            by_game[gi] = _pair_proposals(gi, phase, ply, "b400_plus"); gi += 1
    for band, n in (("b400_plus", 10), ("b300_399", 20), ("b200_299", 20)):
        for _ in range(n):                        # 50 pairs = 100 >= 75
            by_game[gi] = _pair_proposals(gi, "late", 100, band); gi += 1
    return by_game


def test_preflight_accepts_per_phase_quota_mapping():
    alloc = _production_alloc()
    report = v2.v2_geometry_feasibility(
        make_production_geometry(),
        quota_per_phase=alloc.quota_by_phase,
        late_candidate_floors=alloc.band_minima_total,
        max_per_game=alloc.max_per_game, min_gap=alloc.min_ply_gap,
        side_tol=alloc.side_tol,
        split_totals=alloc.split_totals)
    assert report.feasible, report.binding_constraint
    assert report.quota_per_phase == alloc.quota_by_phase


def test_preflight_names_the_starved_phase_under_a_mapping():
    geometry = {gi: rows for gi, rows in make_production_geometry().items()
                if rows[0]["phase"] != "opening"}
    alloc = _production_alloc()
    report = v2.v2_geometry_feasibility(
        geometry, quota_per_phase=alloc.quota_by_phase,
        late_candidate_floors=alloc.band_minima_total,
        split_totals=alloc.split_totals)
    assert not report.feasible
    assert "opening" in report.binding_constraint


def test_scalar_quota_legacy_path_unchanged():
    # The existing suite's own preflight fixtures keep passing untouched --
    # this is just the direct scalar-call sanity check on the new signature.
    geometry = make_production_geometry()
    report = v2.v2_geometry_feasibility(geometry, quota_per_phase=24)
    assert report.quota_per_phase == {p: 24 for p in v2.PHASES}
```

- [ ] **Step 2: Run to verify failure** — mapping is rejected / `split_totals` unknown kwarg.

- [ ] **Step 3: Implement**

  1. `v2_geometry_feasibility`: change the signature to `quota_per_phase: Union[int, Mapping[str, int]] = QUOTA_PER_PHASE` and add `split_totals: Optional[Mapping[str, int]] = None`. First lines of the body:

```python
    quota_by_phase: Dict[str, int] = (
        {p: int(quota_per_phase.get(p, 0)) for p in phases}
        if isinstance(quota_per_phase, Mapping)
        else {p: int(quota_per_phase) for p in phases})
    split_totals = dict(split_totals) if split_totals is not None \
        else dict(SPLIT_TOTALS)
```

  Replace every scalar use: capacity check (1512) `quota_per_phase` → `quota_by_phase[phase]`; side-aliasing (1530, 1533) pair demand → `-(-quota_by_phase[phase] // PAIR_POSITIONS)` (ceil); `_report`'s `quota_per_phase=` field → `dict(quota_by_phase)`; the witness call (1546–1548) passes `quota_by_phase` and `split_totals`.
  2. `V2PreflightReport.quota_per_phase` type becomes `Dict[str, int]`. Grep the existing test file for pins: `grep -n "quota_per_phase" tests/test_fpu_dev_corpus_v2.py` — update any `== 60` pin to `== {p: 60 for p in PHASES}` (mechanical; the report now records per-phase demand).
  3. `_build_v2_witness(proposals_by_game, phases, quota_by_phase, late_candidate_floors, max_per_game, min_gap, side_tol, split_totals)`: line 1318 → `pairs_per_phase = {p: -(-quota_by_phase[p] // PAIR_POSITIONS) for p in phases}`; lines 1361–1371 use `pairs_per_phase[phase]` and report `quota_by_phase[phase]`; the split-placement loop (1389–1399) keeps the first-fit rule but on overflow places the game with the LARGEST remaining need instead of failing (over-selection is now legitimate):

```python
        if not placed:
            best = max(SPLITS, key=lambda s: split_totals[s] - filled[s])
            split_of[gi] = best
            filled[best] += n
```

  and the exact-equality verification (1411–1414) becomes coverage:

```python
        if per_split_pos[split] < split_totals[split]:
            return None, (f"split-budget:{split} (realized {per_split_pos[split]} "
                          f"< {split_totals[split]})")
```

  4. `v2_preflight_source(records, alloc: Optional[AllocationProfile] = None)` — when `alloc` is given, forward `quota_per_phase=alloc.quota_by_phase, late_candidate_floors=alloc.band_minima_total, max_per_game=alloc.max_per_game, min_gap=alloc.min_ply_gap, side_tol=alloc.side_tol, split_totals=alloc.split_totals` to `v2_geometry_feasibility`.
  5. `run_screen` (line ~2155): `report = v2_preflight_source(records, alloc=profile_for(config))`.

- [ ] **Step 4: Run tests + full suite** (the suite's preflight/witness tests are the legacy guard; only the `quota_per_phase` report-field pins may need the mechanical dict update from step 3.2).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/fpu_dev_corpus_v2.py tests/test_fpu_v2_repair.py tests/test_fpu_dev_corpus_v2.py
git commit -m "feat(fpu-v2): per-phase preflight quotas (non-uniform, odd-safe ceil pairs, coverage split check)"
```

---

### Task 7: `post-screen-qualify` CLI mode + select requires a PASS report

**Files:**
- Modify: `scripts/GPU/alphazero/fpu_dev_corpus_v2.py` — `_parse_v2_args` (3198), `main` (3299); new module constants + two functions near `_v2_cli_hard_stop`
- Test: `tests/test_fpu_v2_repair.py`

**Interfaces:**
- Produces: CLI `--mode post-screen-qualify --config C --screen S` writing `config.post_screen_report_out` (immutable, canonical JSON) and exiting 0 (PASS) / 4 (GATE_FAIL) / 2 (usage/IO) / 3 (identity or artifact mismatch); module constants `EXIT_OK = 0, EXIT_USAGE = 2, EXIT_MISMATCH = 3, EXIT_GATE_FAIL = 4` (documented mirror of `fpu_dev_reservoir_protocol`'s — same values, no import, Sec-6 cycle); functions `run_post_screen_qualify(config, screen_csv_path) -> Tuple[int, Dict]` and `require_pass_report(config, screen_csv_path, alloc, *, screen_csv_sha1, config_sha1) -> Dict` (final review edit 1: the report records and the gatekeeper verifies `protocol_sha1`, `config_sha1`, `selection_seed`, and `run_kind`, alongside screen bytes + profile — the exact witness depends on `selection_seed`).
- **`PASS` semantics (review correction 1): PASS means the EXACT deterministic selector succeeded, not merely that capacity bounds held.** `run_post_screen_qualify` runs the capacity report AND a dry run of `sample_v2_rows(kept, seed=config.selection_seed, alloc=alloc)` — the same selector, same seed, same kept rows select will later use (the screen sha1 binding guarantees the same input, so determinism guarantees the same outcome). The report records the complete witness (counts by cell, split, band, side, and rows-per-game). Either stage failing → GATE_FAIL, exit 4, with the failing stage's reason. A select failure AFTER a PASS report is therefore always corruption or a code defect — the raw traceback there is correct and intended.
- Behavior: schema-2 `--mode select` refuses to run without a matching PASS report (missing file → 2; status GATE_FAIL / stale `screen_csv_sha1` / mismatched profile fingerprint → 3). Schema-1 select is UNCHANGED. `select_final_manifest` still re-checks everything itself.

- [ ] **Step 1: Write the failing tests** (append). The full identity chain needs a real reservoir, so these tests exercise the two new FUNCTIONS directly (report document + gatekeeper); CLI exit-code mapping is tested through `main` only for the argument-validation paths:

```python
def test_post_screen_report_document_binds_screen_and_profile(tmp_path):
    alloc = v2.AllocationProfile.legacy()
    kept = make_gate_fail_fixture()
    report = v2.post_screen_qualification_report(kept, alloc)
    doc = v2.build_post_screen_report_document(
        report, selector_witness=None, selector_error=None,
        screen_csv_sha1="abc123", config=None, alloc=alloc)
    assert doc["status"] == "GATE_FAIL"
    assert doc["screen_csv_sha1"] == "abc123"
    assert doc["no_manifest_written"] is True
    assert doc["profile"] == alloc.fingerprint()


def test_pass_requires_a_complete_selector_witness():
    # Review correction 1: capacity PASS + no witness must NOT yield PASS.
    alloc = _production_alloc()
    kept = make_feasible_120_pool()
    report = v2.post_screen_qualification_report(kept, alloc)
    assert report["status"] == "PASS"                  # necessary bounds hold
    no_witness = v2.build_post_screen_report_document(
        report, selector_witness=None,
        selector_error="synthetic: selector refused", screen_csv_sha1="s",
        config=None, alloc=alloc)
    assert no_witness["status"] == "GATE_FAIL"
    assert "selector" in no_witness["binding_constraint"]
    rows, stats = v2.sample_v2_rows(kept, seed=20260718, alloc=alloc)
    witness = v2.build_selector_witness(rows, stats)
    with_witness = v2.build_post_screen_report_document(
        report, selector_witness=witness, selector_error=None,
        screen_csv_sha1="s", config=None, alloc=alloc)
    assert with_witness["status"] == "PASS"
    assert with_witness["selector_witness"]["n_rows"] == 120
    assert with_witness["selector_witness"]["cell_counts"][
        "target|late|frozen_check"] == 20


def test_require_pass_report_rejects_missing_failed_stale_mismatched(tmp_path):
    alloc = _production_alloc()
    report_path = tmp_path / "psq.json"

    class Cfg:   # duck-typed: only the fields the gatekeeper reads
        post_screen_report_out = str(report_path)
        config_path = "cfg.json"
        selection_seed = 20260718
        expected_fingerprints = {"protocol_sha1": "proto1"}

    kw = dict(screen_csv_sha1="s1", config_sha1="cfg1")
    with pytest.raises(FileNotFoundError):
        v2.require_pass_report(Cfg(), "screen.csv", alloc, **kw)
    # Final review edit 1: the PASS report binds the COMPLETE config --
    # protocol_sha1, config_sha1, selection_seed (the witness depends on it),
    # and run_kind, alongside screen bytes + profile.
    ok = {"status": "PASS", "screen_csv_sha1": "s1",
          "profile": alloc.fingerprint(), "protocol_sha1": "proto1",
          "config_sha1": "cfg1", "selection_seed": 20260718,
          "run_kind": "production"}
    for bad, needle in [
            (dict(ok, status="GATE_FAIL"), "GATE_FAIL"),
            (dict(ok, screen_csv_sha1="other"), "stale"),
            (dict(ok, profile=v2.AllocationProfile.legacy().fingerprint()),
             "profile"),
            (dict(ok, protocol_sha1="other"), "protocol_sha1"),
            (dict(ok, config_sha1="other"), "config_sha1"),
            (dict(ok, selection_seed=1), "selection_seed"),
            (dict(ok, run_kind="tooling_smoke"), "run_kind"),
    ]:
        report_path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=needle):
            v2.require_pass_report(Cfg(), "screen.csv", alloc, **kw)
    report_path.write_text(json.dumps(ok))
    assert v2.require_pass_report(
        Cfg(), "screen.csv", alloc, **kw)["status"] == "PASS"


def test_cli_post_screen_qualify_requires_screen_and_schema2(tmp_path):
    cfg_path = _write_config(tmp_path, {})       # schema 1
    assert v2.main(["--mode", "post-screen-qualify", "--config", cfg_path,
                    "--screen", str(tmp_path / "nope.csv")]) == 2
```

- [ ] **Step 2: Run to verify failure** — the three new names are undefined; the CLI rejects the new mode choice.

- [ ] **Step 3: Implement**

  (a) Constants near `_v2_cli_hard_stop`:

```python
# Shared exit-code vocabulary (repair plan Sec 7) -- the SAME values as
# fpu_dev_reservoir_protocol.EXIT_* (design Sec 3). Mirrored, not imported:
# Sec 6 forbids the top-level import (that module imports FROM here).
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_MISMATCH = 3
EXIT_GATE_FAIL = 4
```

  (b) The report document + gatekeeper functions (place after `post_screen_qualification_report`):

```python
def build_selector_witness(rows: List[dict],
                           stats: Dict[str, Any]) -> Dict[str, Any]:
    """The dry-run selector's COMPLETE witness (review correction 1):
    selected counts by cell, split, band, side, and game -- what a PASS
    report certifies actually exists."""
    return {
        "n_rows": len(rows),
        "cell_counts": stats["cell_counts"],
        "late_target_band_count": stats["late_target_band_count"],
        "late_target_band_count_by_split":
            stats["late_target_band_count_by_split"],
        "side_count": stats["side_count"],
        "rows_per_game": {str(gi): n for gi, n in sorted(
            Counter(r["game_idx"] for r in rows).items())},
        "assignment_attempt": stats["assignment_attempt"],
    }


def build_post_screen_report_document(
        report: Dict[str, Any], *, selector_witness: Optional[Dict[str, Any]],
        selector_error: Optional[str], screen_csv_sha1: str,
        config: Optional[V2Config], alloc: AllocationProfile,
        config_sha1: Optional[str] = None) -> Dict[str, Any]:
    """The persisted post_screen_qualification_report.json body. PASS iff the
    capacity report passed AND a complete selector witness exists (review
    correction 1) -- necessary bounds alone never certify; per-split floors,
    spacing, side balance and whole-game assignment are proved by the witness.
    `no_manifest_written` is a recorded FACT: this stage never writes a
    manifest, in either outcome."""
    if report["status"] == "PASS" and selector_witness is not None:
        status, binding = "PASS", None
    elif report["status"] != "PASS":
        status, binding = "GATE_FAIL", report["binding_constraint"]
    else:
        status = "GATE_FAIL"
        binding = f"selector dry-run failed: {selector_error}"
    return {
        "status": status,
        "binding_constraint": binding,
        "report": report,
        "selector_witness": selector_witness,
        "selector_error": selector_error,
        "profile": alloc.fingerprint(),
        "screen_csv_sha1": screen_csv_sha1,
        "protocol_sha1": (config.expected_fingerprints.get("protocol_sha1")
                          if config is not None else None),
        "config_path": (config.config_path if config is not None else None),
        "config_sha1": config_sha1,
        "selection_seed": (config.selection_seed
                           if config is not None else None),
        "run_kind": alloc.run_kind,
        "no_manifest_written": True,
    }


def require_pass_report(config, screen_csv_path: str,
                        alloc: AllocationProfile, *,
                        screen_csv_sha1: str,
                        config_sha1: str) -> Dict[str, Any]:
    """select's gatekeeper (repair plan Sec 7): a matching PASS report must
    exist BEFORE sampling, bound to the COMPLETE config (final review edit 1):
    screen bytes, allocation profile, protocol_sha1, config_sha1,
    selection_seed (the exact witness depends on it), and run_kind. Missing
    file propagates FileNotFoundError (CLI -> EXIT_USAGE); every mismatch
    raises ValueError naming the reason (CLI -> EXIT_MISMATCH)."""
    doc = json.loads(Path(config.post_screen_report_out).read_text())
    if doc.get("status") != "PASS":
        raise ValueError(
            f"post-screen report {config.post_screen_report_out} has status "
            f"{doc.get('status')!r} (GATE_FAIL or unknown) -- select refuses")
    if doc.get("screen_csv_sha1") != screen_csv_sha1:
        raise ValueError(
            f"post-screen report is stale: it binds screen sha1 "
            f"{doc.get('screen_csv_sha1')!r}, but {screen_csv_path} hashes to "
            f"{screen_csv_sha1!r}")
    if doc.get("profile") != alloc.fingerprint():
        raise ValueError(
            "post-screen report was produced for a DIFFERENT allocation "
            "profile than this config's -- profile fingerprint mismatch")
    expected = {
        "protocol_sha1": config.expected_fingerprints.get("protocol_sha1"),
        "config_sha1": config_sha1,
        "selection_seed": config.selection_seed,
        "run_kind": alloc.run_kind,
    }
    for key, want in expected.items():
        if doc.get(key) != want:
            raise ValueError(
                f"post-screen report {key} mismatch: report has "
                f"{doc.get(key)!r}, this config requires {want!r}")
    return doc
```

  (c) `run_post_screen_qualify(config, screen_csv_path)` — mirrors select's steps 1–4 then reports instead of raising. Reuse the existing helpers exactly as `select_final_manifest` does (same order, same lazy import):

```python
def run_post_screen_qualify(config: V2Config,
                            screen_csv_path: str) -> Tuple[int, Dict[str, Any]]:
    """The CONTROLLED post-screen qualification stage (repair plan Sec 7).
    PURE (no evaluator). Returns (exit_code, report_document); writes
    config.post_screen_report_out (immutable canonical JSON; byte-identical
    re-runs are idempotently accepted)."""
    from .fpu_dev_reservoir_protocol import (canonical_json_bytes,
                                             rederive_and_assert_config_unchanged,
                                             write_atomic)
    from . import fpu_provenance
    screen_meta = json.loads(Path(screen_csv_path + ".meta.json").read_text())
    validate_screen_identities(
        screen_meta, config, forbidden_paths=config.forbidden_manifests,
        screen_csv_path=screen_csv_path)
    rederive_and_assert_config_unchanged(config)
    screen_rows = read_screen_csv(screen_csv_path)
    validate_screen_rows_against_meta(screen_rows, screen_meta)
    alloc = profile_for(config)
    kept = kept_rows_from_screen(screen_rows)
    report = post_screen_qualification_report(kept, alloc)
    # Review correction 1: PASS requires the EXACT deterministic selector to
    # succeed on the same (kept, seed, alloc) select will later use.
    selector_witness, selector_error = None, None
    if report["status"] == "PASS":
        try:
            sel_rows, sel_stats = sample_v2_rows(
                kept, seed=config.selection_seed, alloc=alloc)
        except ValueError as exc:
            selector_error = str(exc)
        else:
            selector_witness = build_selector_witness(sel_rows, sel_stats)
    doc = build_post_screen_report_document(
        report, selector_witness=selector_witness,
        selector_error=selector_error,
        screen_csv_sha1=fpu_provenance.file_sha1(screen_csv_path),
        config=config, alloc=alloc,
        config_sha1=fpu_provenance.file_sha1(config.config_path))
    write_atomic(config.post_screen_report_out, canonical_json_bytes(doc))
    return (EXIT_OK if doc["status"] == "PASS" else EXIT_GATE_FAIL), doc
```

  (Check the actual import names first: `grep -n "write_atomic\|canonical_json_bytes" scripts/GPU/alphazero/fpu_dev_corpus_v2.py` — if other sites import them differently, match that style. `fpu_provenance.file_sha1` is the hash helper the identity chain already uses; confirm with `grep -n "file_sha1" scripts/GPU/alphazero/fpu_dev_corpus_v2.py` and reuse the module's existing import form.)

  (d) CLI wiring in `_parse_v2_args`: add `"post-screen-qualify"` to `--mode` choices; it requires `--screen` (extend the existing `if args.mode == "select"` requirement to `in ("select", "post-screen-qualify")`). In `main`, add before the select branch:

```python
    if args.mode == "post-screen-qualify":
        if config.config_schema_version < 2:
            return _v2_cli_hard_stop(
                "[fpu-dev-corpus-v2] post-screen-qualify requires a schema-2 "
                "config (the controlled report is a repair-plan feature)")
        try:
            code, doc = run_post_screen_qualify(config, args.screen)
        except (OSError, json.JSONDecodeError) as exc:
            return _v2_cli_hard_stop(
                f"[fpu-dev-corpus-v2] post-screen-qualify STOPPED (I/O): {exc}")
        except ValueError as exc:
            # Identity/rederive/meta mismatch, or an immutable-report
            # conflict: artifact mismatch -> 3 (repair plan Sec 7).
            print(f"[fpu-dev-corpus-v2] post-screen-qualify MISMATCH: {exc}")
            return EXIT_MISMATCH
        print(f"[fpu-dev-corpus-v2] post-screen-qualify: {doc['status']} "
              f"(binding: {doc['binding_constraint']}) -> "
              f"{config.post_screen_report_out}")
        return code
```

  (e) In the select branch of `main`, immediately before the `select_final_manifest` call, add (schema-2 only):

```python
        if config.config_schema_version >= 2:
            from . import fpu_provenance
            alloc = profile_for(config)
            try:
                require_pass_report(
                    config, args.screen, alloc,
                    screen_csv_sha1=fpu_provenance.file_sha1(args.screen),
                    config_sha1=fpu_provenance.file_sha1(config.config_path))
            except FileNotFoundError as exc:
                return _v2_cli_hard_stop(
                    f"[fpu-dev-corpus-v2] select STOPPED: no post-screen "
                    f"qualification report -- run --mode post-screen-qualify "
                    f"first ({exc})")
            except ValueError as exc:
                print(f"[fpu-dev-corpus-v2] select MISMATCH (post-screen "
                      f"report): {exc}")
                return EXIT_MISMATCH
```

  (f) Scope correction (smoke isolation): in the select branch's `write_screen_meta` dict (line ~3375), add `"run_kind": config.run_kind` **when `config.config_schema_version >= 2`** (schema-1 meta bytes unchanged; Task 0 goldens guard). Every schema-2 manifest then names its `run_kind` in its own meta, and its `stats["allocation_profile"]`/`stats["run_kind"]` (Task 4) bind it to the config — so any future consumer can check, and the production diagnostic (Task 9) does.

- [ ] **Step 4: Run tests + full suite** (schema-1 select paths untouched → suite green).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/fpu_dev_corpus_v2.py tests/test_fpu_v2_repair.py
git commit -m "feat(fpu-v2): controlled post-screen-qualify stage -- immutable PASS/GATE_FAIL report, exit 0/2/3/4, select requires PASS"
```

---

### Task 8: Protocol v2 + `run_kind` + schema-2 `derive_config`

**Files:**
- Modify: `scripts/GPU/alphazero/fpu_dev_reservoir_protocol.py` — `PROTOCOL_SCHEMA_KEYS` block, `build_protocol` (813), `_validate_protocol_shape` (1269), `derive_config` (2109), `run_qualify` preflight wiring (2383), `default_preflight` (1935)
- Test: `tests/test_fpu_v2_repair.py`

**Interfaces:**
- Produces: `PROTOCOL_SCHEMA_KEYS_V2 = PROTOCOL_SCHEMA_KEYS + ("run_kind", "late_target_band_minima", "max_per_game", "min_ply_gap", "side_tol", "corpus_size", "post_screen_report_out")`; `protocol_schema_keys_for(params_or_protocol) -> Tuple[str, ...]` selecting by `protocol_version` (1 → legacy keys, 2 → V2 keys, else ValueError); `build_protocol` validates v2 protocols (`run_kind` ∈ `PROFILE_RUN_KINDS`, `config_schema_version == 2`, and the allocation fields parse via `fpu_dev_corpus_v2.parse_allocation_profile` — reuse, don't restate); `derive_config` carries the 7 new fields for schema-2 (26 top-level keys); qualify's geometric preflight uses the protocol's own allocation.
- Consumes: `fpu_dev_corpus_v2.parse_allocation_profile`, `PROFILE_RUN_KINDS`, `v2_geometry_feasibility` per-phase form (this module already imports from `fpu_dev_corpus_v2` — the legal direction).

- [ ] **Step 1: Write the failing tests** (append)

```python
from scripts.GPU.alphazero import fpu_dev_reservoir_protocol as proto


def _v2_protocol_params():
    # Minimal complete v2 params: legacy keys (copy the existing protocol
    # fixture from tests/test_fpu_dev_reservoir_protocol.py -- grep for the
    # params builder its build_protocol tests use) + the v2 additions.
    params = dict(V1_PROTOCOL_PARAMS_FIXTURE)          # from the existing tests
    params.update({
        "protocol_version": 2, "config_schema_version": 2,
        "run_kind": "production",
        "phase_allocation": PRODUCTION_PROFILE_RAW["phase_allocation"],
        "late_floors": PRODUCTION_PROFILE_RAW["late_floors"],
        "late_target_band_minima":
            PRODUCTION_PROFILE_RAW["late_target_band_minima"],
        "max_per_game": 2, "min_ply_gap": 12, "side_tol": 2,
        "corpus_size": 120, "post_screen_report_out": "psq.json",
    })
    return params


def test_build_protocol_v2_requires_and_validates_run_kind():
    params = _v2_protocol_params()
    p = proto.build_protocol(params)
    assert p["run_kind"] == "production"
    params["run_kind"] = "experiment"
    with pytest.raises(ValueError, match="run_kind"):
        proto.build_protocol(params)
    del params["run_kind"]
    with pytest.raises(ValueError, match="run_kind"):
        proto.build_protocol(params)


def test_v1_protocol_schema_is_untouched():
    p = proto.build_protocol(V1_PROTOCOL_PARAMS_FIXTURE)
    assert "run_kind" not in p


def test_derive_config_v2_carries_run_kind_and_profile_fields(monkeypatch):
    # Reuse the existing derive_config measurement fixture from
    # tests/test_fpu_dev_reservoir_protocol.py (grep for the
    # ReservoirMeasurements builder its derive_config tests use).
    protocol = proto.build_protocol(_v2_protocol_params())
    cfg = proto.derive_config(protocol, MEASUREMENTS_FIXTURE,
                              protocol_path="p.json")
    for key in ("run_kind", "late_target_band_minima", "max_per_game",
                "min_ply_gap", "side_tol", "corpus_size",
                "post_screen_report_out"):
        assert key in cfg, key
    assert cfg["run_kind"] == "production"
    assert cfg["config_schema_version"] == 2
```

(`V1_PROTOCOL_PARAMS_FIXTURE` / `MEASUREMENTS_FIXTURE`: the existing test file `tests/test_fpu_dev_reservoir_protocol.py` builds both for its own `build_protocol`/`derive_config` tests — import or copy the smallest existing builder rather than inventing new ones; adjust the two names to the real ones found there.)

- [ ] **Step 2: Run to verify failure** — v2 params: `build_protocol` currently drops the unknown keys silently → `p["run_kind"]` KeyError.

- [ ] **Step 3: Implement**

  1. After `PROTOCOL_SCHEMA_KEYS` add:

```python
PROTOCOL_SCHEMA_KEYS_V2: Tuple[str, ...] = PROTOCOL_SCHEMA_KEYS + (
    "run_kind", "late_target_band_minima", "max_per_game", "min_ply_gap",
    "side_tol", "corpus_size", "post_screen_report_out")


def protocol_schema_keys_for(doc: Mapping[str, Any]) -> Tuple[str, ...]:
    version = int(doc.get("protocol_version", 1))
    if version == 1:
        return PROTOCOL_SCHEMA_KEYS
    if version == 2:
        return PROTOCOL_SCHEMA_KEYS_V2
    raise ValueError(f"unsupported protocol_version {version}")
```

  2. `build_protocol`: replace the fixed `PROTOCOL_SCHEMA_KEYS` reads with `keys = protocol_schema_keys_for(params)`; after assembling a v2 protocol, validate:

```python
    if int(protocol.get("protocol_version", 1)) >= 2:
        from .fpu_dev_corpus_v2 import PROFILE_RUN_KINDS, parse_allocation_profile
        if protocol["run_kind"] not in PROFILE_RUN_KINDS:
            raise ValueError(f"build_protocol: unsupported run_kind "
                             f"{protocol['run_kind']!r} "
                             f"(must be one of {PROFILE_RUN_KINDS})")
        if int(protocol["config_schema_version"]) != 2:
            raise ValueError("build_protocol: protocol_version 2 requires "
                             "config_schema_version 2")
        parse_allocation_profile(protocol, source="protocol")   # full Sec-6 validation
```

  (`parse_allocation_profile` reads exactly the keys the protocol carries: `config_schema_version`, `run_kind`, `phase_allocation`, `late_floors`, `late_target_band_minima`, `max_per_game`, `min_ply_gap`, `side_tol`, `corpus_size` — no adapter needed.)
  3. `_validate_protocol_shape`: same `protocol_schema_keys_for(protocol)` substitution.
  4. `derive_config`: after the existing dict literal, add:

```python
    if int(protocol.get("protocol_version", 1)) >= 2:
        config.update({
            "run_kind": protocol["run_kind"],
            "late_target_band_minima": protocol["late_target_band_minima"],
            "max_per_game": protocol["max_per_game"],
            "min_ply_gap": protocol["min_ply_gap"],
            "side_tol": protocol["side_tol"],
            "corpus_size": protocol["corpus_size"],
            "post_screen_report_out": protocol["post_screen_report_out"],
        })
    return config
```

  (rename the literal to `config = { ... }` first; `run_kind` thereby lands inside `protocol_sha1`'s hash — the fingerprint requirement is free.)
  5. Preflight wiring: `default_preflight(measurements, alloc=None)` — pass `alloc` through to the `v2_geometry_feasibility` call (per-phase quotas etc., Task 6's `v2_preflight_source`-style forwarding). In `run_qualify`, where the preflight is injected/defaulted, build the profile for v2 protocols:

```python
    alloc = None
    if int(protocol.get("protocol_version", 1)) >= 2:
        from .fpu_dev_corpus_v2 import parse_allocation_profile
        alloc = parse_allocation_profile(protocol, source="protocol")
```

  and thread it into the `default_preflight` call site (`preflight=lambda m: default_preflight(m, alloc=alloc)` or an explicit parameter — match `qualify_core`'s existing injection shape, visible at `qualify_core` line 1973).

- [ ] **Step 4: Run tests + full suite.** The existing key-set golden tests (`_DERIVED_CONFIG_TOP_LEVEL_KEYS` etc. in `tests/test_fpu_dev_reservoir_protocol.py`) pin the V1 protocol path, which is unchanged — they must pass untouched.

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/fpu_dev_reservoir_protocol.py tests/test_fpu_v2_repair.py
git commit -m "feat(fpu-v2): protocol v2 -- run_kind + allocation authority carried and fingerprinted through derive_config + qualify preflight"
```

---

### Task 9: Diagnostic — smoke rejection, run_kind fingerprint, honest inactive gates

**Files:**
- Modify: `scripts/GPU/alphazero/diagnose_fpu_policy_mass.py` — the `--dev-corpus-config` consistency check (~lines 900–940), `build_run_fingerprint` (641), `dev_safety_verdict` (223–316)
- Test: `tests/test_fpu_v2_repair.py`

**Interfaces:**
- Produces: production diagnostic hard-rejects `run_kind="tooling_smoke"` configs (SystemExit with a message naming the rule); `build_run_fingerprint(..., run_kind: Optional[str] = None)` records `"run_kind"` ONLY when a value is passed (review correction 4: v1 fingerprint bytes unchanged — call sites pass `config.run_kind` only for schema-2 dev-corpus configs); `dev_safety_verdict(..., include_stratum_census: bool = False)` — when True, metrics gain `f"{stratum_key}_stratum_sizes"` (every stratum's target-row count) and `f"{stratum_key}_inactive_strata"` (strata with `n < DEV_BAND_MIN_N`, i.e. whose rate gate did NOT run); default False keeps v1 gate-JSON bytes identical (Task 0's verdict golden is the guard). The v2 operator path (schema-2 `--dev-corpus-config`) passes True.

- [ ] **Step 1: Write the failing tests** (append)

```python
from scripts.GPU.alphazero import diagnose_fpu_policy_mass as diag


def test_dev_safety_verdict_names_inactive_strata():
    # 25 band-A target rows (active) + 5 band-B rows (inactive): the gate for
    # B must be reported as NOT having run, with its sample size.
    def target_row(band, i, collapsed):
        return {"role": "target", "band": band, "ply_bucket": "late",
                "new_collapse": collapsed, "lock_in": False,
                "mover_delta": 0.0, "effective_children_reduction": 0.0,
                "top_share_increase": 0.0}
    rows = ([target_row("b300_399", i, False) for i in range(25)]
            + [target_row("b200_299", i, False) for i in range(5)])
    verdict = diag.dev_safety_verdict(rows, REF_CFG_FIXTURE, CAND_CFG_FIXTURE,
                                      include_stratum_census=True)
    assert verdict.metrics["band_stratum_sizes"] == {
        "b300_399": 25, "b200_299": 5}
    assert verdict.metrics["band_inactive_strata"] == ["b200_299"]
    # Default OFF: v1 gate-JSON metrics unchanged (Task 0 golden also pins this).
    plain = diag.dev_safety_verdict(rows, REF_CFG_FIXTURE, CAND_CFG_FIXTURE)
    assert "band_stratum_sizes" not in plain.metrics
    assert "band_inactive_strata" not in plain.metrics
```

(`REF_CFG_FIXTURE`/`CAND_CFG_FIXTURE` and the exact row schema: copy the row/config builders the existing `dev_safety_verdict` tests use — `grep -n "dev_safety_verdict" tests/test_fpu_*.py` — and match their field names exactly; the test above shows the intent, the existing fixture shows the true row shape.)

```python
def test_production_diagnostic_rejects_tooling_smoke(tmp_path):
    extra = dict(PRODUCTION_PROFILE_RAW)
    extra["run_kind"] = "tooling_smoke"
    extra["post_screen_report_out"] = "psq.json"
    cfg = v2.load_v2_config(_write_config(tmp_path, extra))
    with pytest.raises(SystemExit, match="tooling_smoke"):
        diag.require_production_run_kind(cfg)


def test_run_fingerprint_records_run_kind_only_when_given(tmp_path):
    manifest = tmp_path / "m.csv"
    manifest.write_text("x\n")
    fp = diag.build_run_fingerprint(
        dev_manifest=str(manifest), checkpoint=CHECKPOINT_FIXTURE,
        base_cfg=BASE_CFG_FIXTURE, run_kind="production")
    assert fp["run_kind"] == "production"
    legacy = diag.build_run_fingerprint(
        dev_manifest=str(manifest), checkpoint=CHECKPOINT_FIXTURE,
        base_cfg=BASE_CFG_FIXTURE)
    assert "run_kind" not in legacy      # review correction 4: v1 bytes intact
```

(same fixture-reuse note for `build_run_fingerprint`'s existing test inputs.)

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

  1. New guard next to the `--dev-corpus-config` consistency block (~line 900):

```python
def require_production_run_kind(config) -> None:
    """Repair plan Sec 3: smoke artifacts prove plumbing only. The PRODUCTION
    diagnostic refuses them outright -- they can never select a coefficient,
    pass a safety gate, justify a strength match, or enter self-play."""
    if getattr(config, "run_kind", None) == "tooling_smoke":
        raise SystemExit(
            "[fpu-policy-mass] --dev-corpus-config has run_kind=tooling_smoke: "
            "smoke artifacts are REJECTED by the production diagnostic "
            "(repair plan Sec 3)")
```

  Call it immediately after the config is loaded in the `--dev-corpus-config` path (the block at ~line 925 that checks `config.select_out`).
  2. `build_run_fingerprint(..., run_kind: Optional[str] = None)` — add the keyword; include `"run_kind": run_kind` in the returned dict ONLY when `run_kind is not None` (v1 fingerprint bytes unchanged). At the two call sites (lines ~1192, ~1301) pass `run_kind=config.run_kind` only when a schema-2 dev-corpus config is loaded in that scope; otherwise pass nothing.
  3. `dev_safety_verdict`: inside the closure that computes `_new_collapse_rates_by(key)` (line ~253), the per-value grouping already exists (`srows` per stratum value). Add, mirroring the same grouping expression over the same target rows:

```python
        sizes = {value: len(srows) for value, srows in grouped}   # same iterable
        metrics[f"{key}_stratum_sizes"] = dict(sorted(sizes.items()))
        metrics[f"{key}_inactive_strata"] = sorted(
            v for v, n in sizes.items() if n < DEV_BAND_MIN_N)
```

  Apply it both for the gated `stratum_key` and for the always-reported `band` summary (the `if stratum_key != "band":` branch at line 281 already computes band summaries — give it the same two metrics keys under the `band_` prefix). **Both additions are gated behind the new `include_stratum_census: bool = False` keyword** (review correction 4): with the default False, `metrics` is byte-identical to today (Task 0's verdict golden pins this); the schema-2 operator path (`--dev-corpus-config` with `config_schema_version >= 2`) passes `True`.

- [ ] **Step 4: Run tests + full suite.**

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/diagnose_fpu_policy_mass.py tests/test_fpu_v2_repair.py
git commit -m "feat(fpu-v2): diagnostic honesty -- smoke rejection, run_kind fingerprint, inactive rate-gate labeling"
```

---

### Task 10: `analyze-screen-feasibility` — authenticated discovery command

**Files:**
- Modify: `scripts/GPU/alphazero/fpu_dev_corpus_v2.py` — new functions + CLI mode
- Test: `tests/test_fpu_v2_repair.py`

**Interfaces:**
- Produces: pure core `_analyze_screen_kept(kept: List[dict], alloc: AllocationProfile, selection_seed: int) -> Dict` + authenticated wrapper `analyze_screen_feasibility(config: V2Config, screen_csv_path: str, profile_json: str) -> Dict`, and CLI `--mode analyze-screen-feasibility --config C --screen S --profile-json P --out R`. **Review correction 2: `--config` (the screen's OWN config — schema-1 for reservoir_v1) is REQUIRED, and the wrapper runs the full evidence chain before analyzing:** `validate_screen_identities` (incl. the screen artifact's bytes and forbidden manifests), `rederive_and_assert_config_unchanged` (re-measures the reservoir — allow minutes of hashing on the real 4,800-replay run), `validate_screen_rows_against_meta`. A freshly computed CSV hash alone proves nothing about WHICH file was analyzed; the identity chain proves it is the qualified v1 artifact. Authentication authenticates the INPUT — the OUTPUT stays `discovery_only` and is never production evidence. Exit 0 when qualification AND the selector witness both pass; 4 otherwise; 3 identity/artifact mismatch; 2 usage/IO. Deterministic: identical inputs → byte-identical `--out`.
- Profile JSON = the `PRODUCTION_PROFILE_RAW` shape plus `"selection_seed": <int>`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_analyze_core_pass_with_witness():
    doc = v2._analyze_screen_kept(make_feasible_120_pool(),
                                  _production_alloc(), 20260718)
    assert doc["status"] == "PASS" and doc["discovery_only"] is True
    assert doc["selector_witness"]["n_rows"] == 120
    assert doc["qualification"]["status"] == "PASS"


def test_analyze_core_old_allocation_fails():
    legacy_raw = {
        "config_schema_version": 2, "run_kind": "production",
        "phase_allocation": {f"{r}|{p}": dict(a) for (r, p), a
                             in v2.SPLIT_ALLOC_V2.items()},
        "late_floors": dict(v2.LATE_TARGET_FLOORS),
        "late_target_band_minima": {},
        "max_per_game": 2, "min_ply_gap": 12, "side_tol": 2,
        "corpus_size": 240,
    }
    alloc = v2.parse_allocation_profile(legacy_raw, source="test")
    doc = v2._analyze_screen_kept(make_gate_fail_fixture(), alloc, 20260718)
    assert doc["status"] == "GATE_FAIL"
    assert "target|opening" in doc["qualification"]["binding_constraint"]


def test_analyze_core_is_deterministic():
    a = v2._analyze_screen_kept(make_feasible_120_pool(),
                                _production_alloc(), 20260718)
    b = v2._analyze_screen_kept(make_feasible_120_pool(),
                                _production_alloc(), 20260718)
    assert a == b
```

(The authenticated wrapper + CLI exit codes — including the mismatch-→-3 path — are exercised end to end by Task 12's fabricated-artifact integration test, which is the only place a complete identity chain exists in tests.)

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

```python
def _analyze_screen_kept(kept: List[dict], alloc: AllocationProfile,
                         selection_seed: int) -> Dict[str, Any]:
    """Pure analysis core (repair plan Sec 9): the exact schema-2 qualifier +
    selector over an ALREADY-AUTHENTICATED kept pool. Discovery evidence only."""
    qualification = post_screen_qualification_report(kept, alloc)
    witness, selector_error = None, None
    if qualification["status"] == "PASS":
        try:
            rows, stats = sample_v2_rows(kept, seed=selection_seed, alloc=alloc)
        except ValueError as exc:
            selector_error = str(exc)
        else:
            witness = {
                "n_rows": len(rows), "stats": stats,
                "rows": [{k: r[k] for k in
                          ("game_idx", "ply", "side", "phase", "band",
                           "role", "split", "canonical_sha1")}
                         for r in rows]}
    return {
        "discovery_only": True,
        "profile": alloc.fingerprint(),
        "selection_seed": selection_seed,
        "qualification": qualification,
        "selector_witness": witness,
        "selector_error": selector_error,
        "status": "PASS" if witness is not None else "GATE_FAIL",
    }


def analyze_screen_feasibility(config: V2Config, screen_csv_path: str,
                               profile_json: str) -> Dict[str, Any]:
    """Authenticated wrapper (review correction 2): prove the screen IS the
    qualified artifact its config describes -- identities (incl. the CSV's
    own bytes), config rederivation (re-measures the reservoir), rows-vs-meta
    -- BEFORE analyzing. Authentication authenticates the INPUT; the output
    stays discovery_only. Zero-GPU. Writes no manifest, rebinds no protocol."""
    from .fpu_dev_reservoir_protocol import rederive_and_assert_config_unchanged
    from . import fpu_provenance
    screen_meta = json.loads(Path(screen_csv_path + ".meta.json").read_text())
    verified = validate_screen_identities(
        screen_meta, config, forbidden_paths=config.forbidden_manifests,
        screen_csv_path=screen_csv_path)
    rederive_and_assert_config_unchanged(config)
    screen_rows = read_screen_csv(screen_csv_path)
    validate_screen_rows_against_meta(screen_rows, screen_meta)
    raw = json.loads(Path(profile_json).read_text())
    alloc = parse_allocation_profile(raw, source=profile_json)
    doc = _analyze_screen_kept(kept_rows_from_screen(screen_rows), alloc,
                               int(raw["selection_seed"]))
    doc.update({
        "screen_csv": screen_csv_path,
        "screen_csv_sha1": fpu_provenance.file_sha1(screen_csv_path),
        "analyzed_config_path": config.config_path,
        "verified_screen_provenance": verified,
    })
    return doc
```

CLI: add the mode to choices; add `--profile-json` and `--out` arguments (both `default=None`; `_parse_v2_args` requires them plus `--screen` for this mode; `--config` stays required for EVERY mode). `main` branch:

```python
    if args.mode == "analyze-screen-feasibility":
        try:
            doc = analyze_screen_feasibility(config, args.screen,
                                             args.profile_json)
        except (OSError, json.JSONDecodeError) as exc:
            return _v2_cli_hard_stop(
                f"[fpu-dev-corpus-v2] analyze-screen-feasibility STOPPED "
                f"(I/O): {exc}")
        except ValueError as exc:
            # Identity / rederive / rows-vs-meta / profile-parse mismatch.
            print(f"[fpu-dev-corpus-v2] analyze-screen-feasibility MISMATCH: "
                  f"{exc}")
            return EXIT_MISMATCH
        from .fpu_dev_reservoir_protocol import canonical_json_bytes
        Path(args.out).write_bytes(canonical_json_bytes(doc))
        print(f"[fpu-dev-corpus-v2] analyze-screen-feasibility: "
              f"{doc['status']} -> {args.out}")
        return EXIT_OK if doc["status"] == "PASS" else EXIT_GATE_FAIL
```

- [ ] **Step 4: Run tests + full suite.**

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/fpu_dev_corpus_v2.py tests/test_fpu_v2_repair.py
git commit -m "feat(fpu-v2): analyze-screen-feasibility -- pure discovery witness over an existing screen"
```

---

### Task 11: `sizing-analysis` — deterministic whole-game resampling

**Files:**
- Modify: `scripts/GPU/alphazero/fpu_dev_corpus_v2.py` — new function + CLI mode
- Test: `tests/test_fpu_v2_repair.py`

**Interfaces:**
- Produces: pure core `sizing_analysis_core(kept, all_game_ids, alloc, selection_seed, *, game_counts, trials, seed) -> Dict`, helper `_binomial_lower_bound(k, n, alpha) -> float`, authenticated wrapper `sizing_analysis(config, screen_csv_path, profile_json, *, game_counts, trials, seed) -> Dict`, and CLI `--mode sizing-analysis --config C --screen S --profile-json P --game-counts 1200,2400,3600 --trials 299 --seed 1 --out R` (authenticated like Task 10 — review correction 2). Report per count: `n_trials`, `n_successes`, `success_rate`, `lower_bound_95` (exact Clopper-Pearson), `meets_criterion` (lower bound ≥ 0.99), `degenerate_full_reservoir` flag, `failure_reasons` histogram, capacity distributions.
- **Methodology (review correction 3):** the sampling universe is the COMPLETE reservoir from the qualified source index — zero-kept-row games included; whole games preserved as units; the full-reservoir count runs exactly ONE trial (every draw is the same set) and is flagged degenerate; the preregistered criterion is an exact one-sided 95% binomial lower bound ≥ 0.99 (all-success needs 299 trials — `0.05^(1/299) = 0.99003`), not an observed point rate; the report labels itself finite-reservoir subsampling, which cannot independently certify a fresh reservoir.
- Consumes: Task 10's profile-JSON shape and authenticated-wrapper pattern.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_binomial_lower_bound_pins_the_299_rule():
    assert v2._binomial_lower_bound(299, 299, 0.05) >= 0.99
    assert v2._binomial_lower_bound(298, 298, 0.05) < 0.99
    assert v2._binomial_lower_bound(0, 100, 0.05) == 0.0
    # One failure in 299 must drop the bound below the criterion.
    assert v2._binomial_lower_bound(298, 299, 0.05) < 0.99


def test_sizing_core_universe_includes_zero_yield_games():
    kept = make_feasible_120_pool()                    # games 0..229
    universe = list(range(250))                        # + 20 zero-yield games
    alloc = _production_alloc()
    kw = dict(game_counts=[100, 250], trials=8, seed=5)
    a = v2.sizing_analysis_core(kept, universe, alloc, 20260718, **kw)
    b = v2.sizing_analysis_core(kept, universe, alloc, 20260718, **kw)
    assert a == b                                      # deterministic
    assert a["n_games_available"] == 250
    assert a["n_zero_yield_games"] == 20
    full = a["by_game_count"]["250"]
    assert full["degenerate_full_reservoir"] is True
    assert full["n_trials"] == 1                       # not 8 identical draws
    small = a["by_game_count"]["100"]
    assert small["n_trials"] == 8
    assert small["n_successes"] + sum(
        small["failure_reasons"].values()) == 8
    assert 0.0 <= small["lower_bound_95"] <= small["success_rate"] or (
        small["success_rate"] == 0.0)
    assert a["cannot_certify_beyond"] == 250
    assert a["method"] == "finite-reservoir whole-game subsampling"
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

```python
def _binomial_lower_bound(k: int, n: int, alpha: float) -> float:
    """Exact one-sided (Clopper-Pearson) lower confidence bound for a
    binomial proportion: the largest p with P(X >= k | n, p) <= alpha.
    Stdlib-only bisection; 0.0 when k == 0. All-success closed form:
    alpha ** (1/n) -- e.g. 299 all-success trials give a 95% lower bound of
    0.99003 >= 0.99, while 298 give 0.98999 (the preregistered 299 rule)."""
    if k == 0:
        return 0.0
    def tail_ge_k(p: float) -> float:
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                   for i in range(k, n + 1))
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if tail_ge_k(mid) < alpha:
            lo = mid
        else:
            hi = mid
    return lo


def sizing_analysis_core(kept: List[dict], all_game_ids: List[Any],
                         alloc: AllocationProfile, selection_seed: int, *,
                         game_counts: List[int], trials: int,
                         seed: int) -> Dict[str, Any]:
    """Repair plan Sec 11 + review correction 3: FINITE-RESERVOIR
    SUBSAMPLING. Estimates how reliably smaller whole-game subsets of THIS
    discovery reservoir support the profile; it does NOT independently
    certify a fresh reservoir of any size.

    `all_game_ids` is the COMPLETE reservoir universe (from the qualified
    source index), including games that yielded ZERO kept rows -- excluding
    them would bias success upward. Whole games are the sampling unit."""
    by_game: Dict[Any, List[dict]] = {gi: [] for gi in all_game_ids}
    for r in kept:
        by_game[r["game_idx"]].append(r)     # KeyError = row outside universe
    games = sorted(by_game)
    by_count: Dict[str, Any] = {}
    for count in sorted(set(int(c) for c in game_counts)):
        if count > len(games):
            by_count[str(count)] = {"skipped": f"only {len(games)} games "
                                               f"in the reservoir"}
            continue
        # At the full count every draw is the SAME set -- one trial, flagged.
        n_trials = 1 if count == len(games) else trials
        successes, reasons = 0, Counter()
        capacity_samples: Dict[str, List[int]] = defaultdict(list)
        for t in range(n_trials):
            rng = random.Random(f"sizing:{seed}:{count}:{t}")
            subset = set(rng.sample(games, count))
            sub_kept = [r for gi in subset for r in by_game[gi]]
            rep = post_screen_qualification_report(sub_kept, alloc)
            for cell, info in rep["cells"].items():
                capacity_samples[cell].append(info["capacity"])
            for band, info in rep["late_target_bands"].items():
                capacity_samples[f"band:{band}"].append(info["capacity"])
            if rep["status"] != "PASS":
                reasons[f"qualify: {rep['binding_constraint']}"] += 1
                continue
            try:
                sample_v2_rows(sub_kept, seed=selection_seed, alloc=alloc)
            except ValueError as exc:
                reasons[f"select: {str(exc).splitlines()[0][:120]}"] += 1
            else:
                successes += 1
        lower = _binomial_lower_bound(successes, n_trials, 0.05)
        by_count[str(count)] = {
            "n_trials": n_trials, "n_successes": successes,
            "success_rate": successes / n_trials,
            "lower_bound_95": lower,
            "meets_criterion": lower >= 0.99,
            "degenerate_full_reservoir": count == len(games),
            "failure_reasons": dict(sorted(reasons.items())),
            # Repair plan Sec 11: role/phase + band capacity DISTRIBUTIONS.
            "capacity_min_median_max": {
                cell: [min(vals), sorted(vals)[len(vals) // 2], max(vals)]
                for cell, vals in sorted(capacity_samples.items())}}
    return {
        "discovery_only": True,
        "method": "finite-reservoir whole-game subsampling",
        "confidence_criterion": {
            "method": "exact one-sided (Clopper-Pearson) lower bound",
            "alpha": 0.05, "target_reliability": 0.99,
            "all_success_trials_required": 299,
            "note": ("estimates smaller subsets of THIS discovery reservoir; "
                     "does not independently certify a fresh reservoir"),
        },
        "profile": alloc.fingerprint(), "selection_seed": selection_seed,
        "analysis_seed": seed, "n_trials_per_count": trials,
        "n_games_available": len(games),
        "n_zero_yield_games": sum(1 for gi in games if not by_game[gi]),
        "cannot_certify_beyond": len(games),
        "by_game_count": by_count,
    }
```

The authenticated wrapper `sizing_analysis(config, screen_csv_path, profile_json, *, game_counts, trials, seed)` mirrors Task 10's exactly: identities → rederivation → rows-vs-meta, then **builds the universe from the qualified source index** — `all_game_ids = [rec["game_idx"] for rec in load_game_index(config.source_index_path)]` (the same 4,800-record index the identity chain just hard-matched) — and calls the core; it stamps `screen_csv`/`screen_csv_sha1`/`analyzed_config_path`/`verified_screen_provenance` like Task 10's wrapper.

CLI: same pattern as Task 10 (`--config` required; `--game-counts` comma-split ints, `--trials` int, `--seed` int required for this mode); write via `canonical_json_bytes`; exit 0 on completed analysis (it is a measurement, not a gate), 3 identity/artifact mismatch, 2 usage/IO.
`math` and `random` must be imported at module top if not already (`grep -n "^import math\|^import random" scripts/GPU/alphazero/fpu_dev_corpus_v2.py`; add what's missing).

- [ ] **Step 4: Run tests + full suite.**

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/fpu_dev_corpus_v2.py tests/test_fpu_v2_repair.py
git commit -m "feat(fpu-v2): sizing-analysis -- deterministic whole-game resampling over an existing screen"
```

---

### Task 12: Zero-GPU schema-2 CLI integration test (fabricated artifacts)

The 400-game smoke must NOT be the first full traversal of the new CLI/evidence path (review correction 6). One test drives the REAL CLI end to end on a fabricated reservoir, with the evaluator faked at exactly its two seams — every artifact reader, writer, hash, config derivation, CLI route, and the selector execute for real.

**Files:**
- Test: `tests/test_fpu_v2_repair.py`

**Interfaces:**
- Consumes: the faithful-fixture builders the existing suites already use to fabricate a complete on-disk reservoir (replays + sidecars + index + match summary) for `run_qualify` end-to-end tests — find them with `grep -n "def.*fixture\|def _make.*reservoir\|def.*fabricat" tests/test_fpu_dev_reservoir_protocol.py` and reuse the smallest one that drives the real `run_qualify`; the two evaluator seams `fpu_dev_corpus_v2._build_v2_anchor_search_fn` and `build_teacher_calibration_manifest._teacher_infer` (the lazy import inside `run_screen` resolves the module attribute at call time, so `monkeypatch.setattr` on the source module works).

- [ ] **Step 1: Write the integration test**

```python
def test_schema2_cli_end_to_end_on_fabricated_artifacts(tmp_path, monkeypatch):
    # (1) Fabricate a reservoir + emit a v2 protocol (run_kind=tooling_smoke,
    # a small allocation the fabricated pool can fill) and run the REAL
    # run_qualify -> PASS report + derived schema-2 config on disk.
    cfg_path = _fabricate_qualified_v2_reservoir(tmp_path)   # reuses existing
    #                                                          suite builders
    # (2) Fake ONLY the evaluator seams, deterministically: roles/values
    # chosen so enough proposals classify target/control and anchor-qualify.
    monkeypatch.setattr(v2, "_build_v2_anchor_search_fn",
                        _fake_anchor_search_fn)
    from scripts.GPU.alphazero import build_teacher_calibration_manifest as btcm
    monkeypatch.setattr(btcm, "_teacher_infer", _fake_teacher_infer)
    # (3) Drive the REAL CLI through the whole evidence path.
    assert v2.main(["--mode", "screen", "--config", cfg_path]) == 0
    cfg = v2.load_v2_config(cfg_path)
    assert v2.main(["--mode", "post-screen-qualify", "--config", cfg_path,
                    "--screen", cfg.screen_out]) == 0
    report_bytes = Path(cfg.post_screen_report_out).read_bytes()
    assert v2.main(["--mode", "select", "--config", cfg_path,
                    "--screen", cfg.screen_out]) == 0
    manifest_bytes = Path(cfg.select_out).read_bytes()
    # (4) Idempotency: re-running is a clean accept, byte-identical artifacts.
    assert v2.main(["--mode", "post-screen-qualify", "--config", cfg_path,
                    "--screen", cfg.screen_out]) == 0
    assert Path(cfg.post_screen_report_out).read_bytes() == report_bytes
    assert v2.main(["--mode", "select", "--config", cfg_path,
                    "--screen", cfg.screen_out]) == 0
    assert Path(cfg.select_out).read_bytes() == manifest_bytes
    # (5) Report tampering: missing -> 2, edited (status flipped) -> 3.
    saved = Path(cfg.post_screen_report_out)
    saved.unlink()
    assert v2.main(["--mode", "select", "--config", cfg_path,
                    "--screen", cfg.screen_out]) == 2
    doc = json.loads(report_bytes)
    doc["status"] = "GATE_FAIL"
    saved.write_text(json.dumps(doc))
    assert v2.main(["--mode", "select", "--config", cfg_path,
                    "--screen", cfg.screen_out]) == 3
    saved.write_bytes(report_bytes)
    # (6) Smoke isolation: the production diagnostic rejects this config.
    with pytest.raises(SystemExit, match="tooling_smoke"):
        diag.require_production_run_kind(cfg)
```

Write `_fabricate_qualified_v2_reservoir`, `_fake_anchor_search_fn`, and `_fake_teacher_infer` in the test file, built on the existing suite's fabrication helpers (the fakes' shapes must match what `run_screen` actually calls — read its evaluator-call sites first; the fake teacher returns priors concentrated or flat per position so both roles appear, and the fake anchor returns `root_value_stm` within ±0.25 so rows anchor-qualify).

- [ ] **Step 2: Run it** — this is a characterization-style integration test written AFTER the features exist, so it should pass immediately; if it fails, the failure is a real wiring bug found before GPU day. Then run the full suite.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fpu_v2_repair.py
git commit -m "test(fpu-v2): fabricated-artifact schema-2 CLI integration -- qualify to select, idempotency, smoke rejection"
```

---

### Task 13: Documentation + operator-message correction

**Files:**
- Create: `docs/fpu-v2-repair-operator-guide.md`
- Modify: `scripts/GPU/alphazero/fpu_dev_reservoir_protocol.py` (qualify PASS message), `scripts/GPU/alphazero/fpu_dev_corpus_v2.py` (module docstring pipeline summary)

- [ ] **Step 1: Correct the qualify operator message.** Find the qualify PASS print (`grep -n "PASS\|qualified" scripts/GPU/alphazero/fpu_dev_reservoir_protocol.py | grep -i print`) and make the success line read exactly:

```text
reservoir geometry qualified; raw-policy role and anchor qualification remain pending
```

(Spec Sec 2: reservoir qualification and the geometric preflight are role-agnostic; only the screen learns roles. If a test pins the old message, update that pin — it is the message under correction.)

- [ ] **Step 2: Write `docs/fpu-v2-repair-operator-guide.md`** containing: the pipeline order (`emit-protocol → emit-gen-command → generate → qualify → screen → post-screen-qualify → select`), the four exit codes, the production profile JSON (verbatim from Global Constraints), the smoke profile JSON (verbatim from Task 15), the run_kind rules (smoke can never be production evidence), the pooled-control semantics sentence ("phase-stratified collateral coverage feeding pooled control gates — not four independent phase hard gates"), the inactive-band-gate honesty rule with `DEV_BAND_MIN_N = 20`, and the two discovery commands with example invocations. Also record the §2 correction: the conditional mechanism note that "policy-mass ≈ absolute in concentrated openings" is a hypothesis requiring `Q_parent ≈ 0`, not a measured fact.

- [ ] **Step 3: Update the `fpu_dev_corpus_v2.py` module docstring** (lines 1–130 banner): add the `post-screen-qualify` stage and the two discovery modes to the pipeline summary. Per the surface-descriptor audit rule (memory), grep for stale stage lists: `grep -rn "screen.*select" scripts/GPU/alphazero/fpu_dev_corpus_v2.py | grep -i "never the same\|two stages"` and update the `--mode` help text written in Task 7/10/11 to name all five modes.

- [ ] **Step 4: Run the full suite** (docstring/message pins).

- [ ] **Step 5: Commit**

```bash
git add docs/fpu-v2-repair-operator-guide.md scripts/GPU/alphazero/fpu_dev_reservoir_protocol.py scripts/GPU/alphazero/fpu_dev_corpus_v2.py
git commit -m "docs(fpu-v2): operator guide + corrected qualify message + five-mode CLI descriptors"
```

**User-owned (not this plan's edits):** ledger updates per spec §12 — mark reservoir protocol v1 `POST-SCREEN GATE-FAIL` in `docs/updated-v16a-ledger.md` (or its successor), record 0/0/0/136 + threshold ranges, record that no FPU coefficient ran. Flag these to the user at review.

---

### Task 14: CHECKPOINT — zero-GPU proof on the real v1 screen (mandatory; runnable now)

No GPU, no new games — pure commands over the immutable discovery screen. **STOP after this task and present results for user review; per the spec, the per-split band minima are not frozen until this witness exists.**

- [ ] **Step 1: Full suite green** — `.venv/bin/python -m pytest -p no:cacheprovider -q` → 0 failed.

- [ ] **Step 2: Write the two profile JSONs** to `logs/eval/fpu_v16_policy_mass_v2/analysis/`:
  - `production_profile.json` = `PRODUCTION_PROFILE_RAW` + `"selection_seed": 20260718`
  - `old_allocation_profile.json` = the legacy 240-row allocation as schema-2 JSON (the `legacy_raw` literal from Task 10's test, same seed)

- [ ] **Step 3: Run the proof**

```bash
.venv/bin/python -m scripts.GPU.alphazero.fpu_dev_corpus_v2 \
  --mode analyze-screen-feasibility \
  --config logs/eval/fpu_v16_policy_mass_v2/reservoir_v1/fpu_dev_corpus_v2_config.json \
  --screen logs/eval/fpu_v16_policy_mass_v2/reservoir_v1/fpu_dev_source_screen.csv \
  --profile-json logs/eval/fpu_v16_policy_mass_v2/analysis/production_profile.json \
  --out logs/eval/fpu_v16_policy_mass_v2/analysis/production_feasibility.json
```

(`--config` is the v1 screen's OWN schema-1 config — the identity chain authenticates the discovery input, review correction 2; expect the rederivation to re-hash the 4,800 replays, i.e. minutes, on the first run.)

Expected: exit 0; report `status=PASS`; witness `n_rows=120`, cell counts exactly the production allocation, per-split band minima satisfied (b400_plus ≥ 4/4 — the known-tight constraint: only 12 candidate rows in 12 games exist), sides within tolerance. Run twice → byte-identical.

```bash
.venv/bin/python -m scripts.GPU.alphazero.fpu_dev_corpus_v2 \
  --mode analyze-screen-feasibility \
  --screen logs/eval/fpu_v16_policy_mass_v2/reservoir_v1/fpu_dev_source_screen.csv \
  --profile-json logs/eval/fpu_v16_policy_mass_v2/analysis/old_allocation_profile.json \
  --out logs/eval/fpu_v16_policy_mass_v2/analysis/old_allocation_feasibility.json
```

Expected: exit 4; qualification binding constraint names `target|opening` capacity 0 < demand 45.

- [ ] **Step 4: If the production witness FAILS on the per-split band minima: STOP.** Do not lower any minimum. Report the binding constraint to the user — that is the spec's mandated scientific-allocation decision point.

- [ ] **Step 5: Commit the two profile JSONs** (reports under `logs/` stay untracked unless the user says otherwise) and present: suite count, both report statuses, the witness's band/side composition, and the b400 margin.

---

### Task 15: GATED — 400-game tooling smoke (requires explicit user authorization)

**Do not start without the user's go-ahead in the session log.** Prereqs: Tasks 0–14 complete, Task 14 witness PASS, user has reviewed.

- [ ] **Step 1: Author the smoke protocol params** — copy every generation knob verbatim from `logs/eval/fpu_v16_policy_mass_v2/reservoir_v1/reservoir_protocol.json` (same checkpoints, board 24, 400 sims, eval batch 14, stall flush 48, opening-temperature settings, max moves 280, 4 workers, replay capture), changing ONLY: `protocol_version: 2`, `config_schema_version: 2`, `run_kind: "tooling_smoke"`, `games: 400`, `base_seed: <v1 base_seed + 10000>` (fresh, non-overlapping — verify against v1's `seed_range`), all artifact paths rooted at `logs/eval/fpu_v16_policy_mass_v2/smoke_v1/`, and the smoke allocation:

```json
{
  "phase_allocation": {
    "target|late":       {"tuning": 4, "frozen_check": 2},
    "control|opening":   {"tuning": 2, "frozen_check": 1},
    "control|early_mid": {"tuning": 2, "frozen_check": 1},
    "control|midgame":   {"tuning": 2, "frozen_check": 1},
    "control|late":      {"tuning": 2, "frozen_check": 1}
  },
  "late_floors": {},
  "late_target_band_minima": {},
  "max_per_game": 2, "min_ply_gap": 12, "side_tol": 2,
  "corpus_size": 18
}
```

- [ ] **Step 2: Run the full chain, no manual artifact edits:** `emit-protocol` → `emit-gen-command` → generate 400 games (GPU) → `qualify` → `--mode screen` (GPU) → `--mode post-screen-qualify` → `--mode select` → verify manifest/meta/fingerprints. Expected manifest: 18 rows, 12 tuning / 6 frozen_check, 6 target / 12 control, `run_kind=tooling_smoke` in every artifact.

- [ ] **Step 3: Verify smoke isolation:** run the production diagnostic against the smoke config and confirm it exits with the Task 9 rejection message. Verify idempotency: re-run `post-screen-qualify` and `select` → byte-identical artifacts / clean accept.

- [ ] **Step 4: Record the smoke's technical PASS/FAIL in the operator guide** — explicitly not a scientific result.

---

### Task 16: GATED — production sizing + fresh production protocol (requires user review)

- [ ] **Step 1: Run sizing** on the immutable v1 screen with the production profile:

```bash
.venv/bin/python -m scripts.GPU.alphazero.fpu_dev_corpus_v2 \
  --mode sizing-analysis \
  --config logs/eval/fpu_v16_policy_mass_v2/reservoir_v1/fpu_dev_corpus_v2_config.json \
  --screen logs/eval/fpu_v16_policy_mass_v2/reservoir_v1/fpu_dev_source_screen.csv \
  --profile-json logs/eval/fpu_v16_policy_mass_v2/analysis/production_profile.json \
  --game-counts 1200,1800,2400,3000,3600,4200,4800 --trials 299 --seed 20260718 \
  --out logs/eval/fpu_v16_policy_mass_v2/analysis/sizing_report.json
```

(`--trials 299` is the preregistered all-success requirement for the 95% lower bound ≥ 0.99; start with `--trials 50` once to measure runtime, then run the real 299. The universe is the full 4,800-game reservoir from the qualified source index — zero-yield games included; the 4,800 entry is a single flagged degenerate trial.)

- [ ] **Step 2: Freeze the size — PREREGISTERED margin rule (final review edit 2, fixed BEFORE looking at results):** production game count = **the next larger tested tier above the smallest tier with `meets_criterion` true** (exact one-sided 95% binomial lower bound ≥ 0.99). Concretely, on the tier ladder 1200/1800/2400/3000/3600/4200/4800: if 3,600 is the smallest qualifying tier, production uses 4,200; if 4,200 qualifies first, production uses 4,800. If the smallest qualifying tier is 4,800 itself (no larger tier exists), production uses 4,800. There is no post-results choice to make. This is finite-reservoir subsampling — it estimates subsets of THIS reservoir, not a fresh one; if no count ≤ 4,800 meets the criterion, STOP — going beyond the discovery screen is a user decision.

- [ ] **Step 3: Emit the fresh production protocol** (protocol v2, `run_kind=production`, new artifact root, fresh seed range, the game count from Step 2's preregistered rule, and **`board_size: 24`** — final review edit 3: the generic tooling may support other boards, but the v16 production protocol MUST carry board size 24; no multi-board-size runs) — and stop. Generating the production reservoir is a separate authorization.

---

## Definition of Done (spec §14)

- [ ] protocol v1 failure recorded, artifacts preserved untouched
- [ ] schema-1 artifacts byte-identical, pinned by pre-repair goldens (Task 0)
- [ ] production + smoke profiles immutable, config-driven, fingerprinted (Tasks 1–2, 8)
- [ ] old allocation fails in regression (Task 3) and on the real screen (Task 14)
- [ ] post-screen PASS = exact-selector witness, never capacity bounds alone (Task 7)
- [ ] new production profile passes exact selection on the real discovery screen (Task 14)
- [ ] per-split band minima have a constructive witness (Task 14)
- [ ] discovery analyses authenticate their input via the full identity chain (Tasks 10–11, 14)
- [ ] expected gate failures never traceback (Task 7)
- [ ] focused + full suites pass (every task), incl. the fabricated-artifact CLI integration path (Task 12)
- [ ] 400-game smoke completes through manifest selection (Task 15, gated)
- [ ] smoke artifacts rejected by the production diagnostic entry point, and every schema-2 artifact names its `run_kind` so any future consumer can check (Tasks 7–9, 12, 15 — the guarantee covers the paths guarded here, not every conceivable consumer)
- [ ] production game count justified by finite-reservoir subsampling with the preregistered 95%-lower-bound ≥ 0.99 criterion (Task 16, gated)
- [ ] operator commands emitted from the frozen protocol, never hand-assembled (Task 15)
- [ ] no result-determining source or parameter implicit (Task 4's constant-mutation test)
