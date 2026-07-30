"""v18 broad non-A SOURCE UNIVERSE -- established, verified and frozen.

Spec: docs/superpowers/specs/2026-07-29-v18-depth2-provisional-backup-design.md
Sec 2.2 / 2.2.3.  Plan: Task 4.

This pool supplies EVERY numeric threshold the preflight applies, so its
independence from A and from the four established acceptance gates is the
load-bearing property. Exclusions are COMPUTED and counted here, never asserted.

WHAT THIS TASK CANNOT DO. A replay move carries only
`col, n_legal, player, ply, root_top1_share, root_total_visits, root_value,
row, selected_visit_count, selected_visit_rank` -- no tree, no depth-2
population, no residuals. And `root_value` belongs to whichever checkpoint held
that colour, so in a `0379_vs_calib020_0001` pool it is one checkpoint's value
on one colour only. Therefore NO value-based rule and NO near-even rule may be
applied here; both wait for the remeasured census (Task 7) and the precommitted
matcher (Task 4b).

`freeze_source_universe` SELECTS NO COHORT. It freezes what may be measured.

Two exclusions, in a frozen ORDER (`UNIVERSE["order_of_operations"]`):
  * GAME, by the replay's content SHA-1, before the 800 are chosen;
  * POSITION, by canonical state SHA-1, after -- so a game is never dropped for
    holding one forbidden position, which would defeat zero-yield retention.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from . import fpu_provenance
from .diagnose_fpu_baseline_policy_mass import game_identities
from .fpu_dev_reservoir_protocol import canonical_json_bytes
from .fpu_state_hash import canonical_state_sha1
from .position_probe_cases import position_state
from .v18_preflight_criteria import CENSUS, PHASE_WINDOWS, UNIVERSE

# `game_identities` is re-exported deliberately: identity is the replay's
# content hash, and v17 already learned that `(replay_dir, game_idx)` "both
# invents overlaps and misses a copied game that was renumbered". Import the
# helper; never re-derive it.
__all__ = [
    "FORBIDDEN_SOURCES", "CANDIDATE_UNIVERSES", "game_identities",
    "forbidden_canonical_hashes", "forbidden_game_content_sha1s",
    "enumerate_census", "apply_exclusions", "freeze_source_universe",
    "report_universe", "UNIVERSE", "CENSUS", "PHASE_WINDOWS",
]

LOGS = "logs/eval"

# ---------------------------------------------------------------------------
# Forbidden sources -- every consumed or acceptance population.
#
# `total_rows` and `distinct_positions` are MEASURED values, pinned here and
# re-verified against the file at read time; a mismatch raises rather than
# silently shrinking an exclusion set.
# ---------------------------------------------------------------------------

FORBIDDEN_SOURCES = (
    {
        "name": "gate_A",
        "path": f"{LOGS}/calib020_0001_black_loss_post_opening_predrop_probe/position_probe_cases.csv",
        "kind": "probe_cases_csv",
        "total_rows": 90,
        "distinct_positions": 30,
        "why": "the reach population; must not also supply thresholds",
    },
    {
        "name": "gate_B",
        # The gate_B CASES csv carries no `replay_path`, so positions are taken
        # from the INPUT manifest that generated it. Verified to cover exactly
        # the same 18 (game_idx, position_ply) pairs.
        "path": f"{LOGS}/loss_analysis_v2_1/goal_line_trigger_probe_manifest.json",
        "kind": "goal_line_manifest_json",
        "cases_csv": f"{LOGS}/black_predrop_calib010_goal_line/goal_line_trigger_probe_cases.csv",
        "cases_csv_rows": 54,
        "total_rows": 18,
        "distinct_positions": 18,
        "why": "established acceptance positions (spec Sec 2.2)",
    },
    {
        "name": "gate_C",
        "path": f"{LOGS}/calib020_post_opening_sweep/position_probe_cases.csv",
        "kind": "probe_cases_csv",
        "total_rows": 240,
        "distinct_positions": 30,
        "rejected_as_control_source": True,
        "rejection_reason": (
            "revision 1 named this the CONTROL source. It is 240 rows over only "
            "30 distinct positions, repeated across checkpoints 0001 0003 0005 "
            "0008 0010 0015 0379 0409 -- and those 30 positions ARE gate C. "
            "Excluding them canonically leaves zero controls; keeping only "
            "checkpoint=0001 retains duplicate evaluations of the same consumed "
            "positions rather than independent discovery controls"),
        "why": "established acceptance positions (spec Sec 2.2)",
    },
    {
        "name": "gate_D",
        "path": f"{LOGS}/calib020_0001_red_loss_post_opening_predrop_probe/position_probe_cases.csv",
        "kind": "probe_cases_csv",
        "total_rows": 90,
        "distinct_positions": 30,
        "why": "established acceptance positions (spec Sec 2.2)",
    },
    {
        "name": "v16a_neutral_consumed",
        "path": f"{LOGS}/v16a_fpu_unbiased/neutral_position_manifest.csv",
        "kind": "probe_cases_csv",
        "total_rows": 324,
        "distinct_positions": 324,
        "why": "consumed; do-not-repeat #42 forbids tuning against it",
    },
    {
        "name": "v17_development_selected",
        "path": f"{LOGS}/fpu_v17_baseline_policy_mass/development/fpu_dev_corpus_v2_manifest.csv",
        "kind": "corpus_manifest_csv",
        "total_rows": 32,
        "distinct_positions": 32,
        "why": "consumed selection records (spec Sec 2.1)",
    },
    {
        "name": "v16_production_selected",
        "path": f"{LOGS}/fpu_v16_policy_mass_v2/production_v2_b400amend_4000g_seed20300000/fpu_dev_corpus_v2_manifest.csv",
        "kind": "corpus_manifest_csv",
        "total_rows": 120,
        "distinct_positions": 120,
        "why": "consumed selection records (spec Sec 2.1)",
    },
    {
        "name": "a_replay_games",
        "path": f"{LOGS}/calib020_0001_vs_0379_800g_w4_seed20115_replays",
        "kind": "replay_dir",
        "n_games": 800,
        "total_rows": 0,
        "distinct_positions": 0,
        "summary": f"{LOGS}/calib020_0001_vs_0379_800g_w4_seed20115_replay.json",
        "jsonl": f"{LOGS}/calib020_0001_vs_0379_800g_w4_seed20115_replay_games.jsonl",
        "summary_sha1": "bf1e3701ca8591295bd1e70b2a88a84087fad316",
        "jsonl_sha1": "fb0944ae0333b951a817d0393919b45f2a12fd78",
        "replay_data_sha1": "427d4ab669a81fe409de7da6d7c458056aff306e",
        "why": "the A rows' game source -- GAME-level exclusion",
    },
)

# Byte identities of every forbidden evidence file, pinned so a REPLACEMENT file
# with the same row count cannot pass. Counts alone authenticate nothing: an
# edited CSV with 240 rows still has 240 rows.
FORBIDDEN_SOURCE_SHA1S = {
    "gate_A": "175c73ef2c761df83ccf5f5cd935152093f8dfb1",
    "gate_B": "00a3a4220e593791eb4c9eec7973392e5906b0b9",
    "gate_B_cases_csv": "091e80ef3b5e2dbaaf9bb76ce77ef4b8597e90ed",
    "gate_C": "592a624b088fd39565518bb9560d41401432b648",
    "gate_D": "5f26ec1aae22be62a8bfe272e4787294e8887f0e",
    "v16a_neutral_consumed": "bf7a00ad7ca524ef3aa778b9e0decc11218a2f7d",
    "v16_production_selected": "84cdd4b45e089a2ebb292491c146ba00bff17ea9",
    "v17_development_selected": "15b0228edc1ed605fea799694d4ca0eda3e3468b",
}

# The replay RESERVOIRS the forbidden evidence points into. Authenticating the
# probe CSVs is not enough: the canonical exclusion hashes are reconstructed
# from these sidecars, so a replay that drifts changes the exclusion SET while
# every evidence-file hash in the record stays unchanged. Each reservoir is
# bound by game count and by the established length-delimited aggregate, checked
# BEFORE the exclusions are derived and again BEFORE anything is written.
FORBIDDEN_REPLAY_RESERVOIRS = (
    {
        "name": "seed20115",
        "dir": f"{LOGS}/calib020_0001_vs_0379_800g_w4_seed20115_replays",
        "n_games": 800,
        "replay_data_sha1": "427d4ab669a81fe409de7da6d7c458056aff306e",
        "referenced_by": ("gate_A", "gate_D", "v16a_neutral_consumed",
                          "a_replay_games"),
    },
    {
        "name": "seed35791",
        "dir": f"{LOGS}/eps035_0399_vs_0379_800g_w4_seed35791_replays",
        "n_games": 800,
        "replay_data_sha1": "d36b01c0993095e07785666316028f0c875eed7b",
        "referenced_by": ("gate_B",),
    },
    {
        "name": "seed40937",
        "dir": f"{LOGS}/lr0003_0409_vs_0379_800g_w4_seed40937_replays",
        "n_games": 800,
        "replay_data_sha1": "80aa2068319cdbe0429100b736d293f5b8bc437e",
        "referenced_by": ("gate_C",),
    },
)

# ---------------------------------------------------------------------------
# The SELECTED source universe, bound in TRACKED CODE.
#
# The Step 5 decision is evidence, so it lives here rather than in an operator's
# shell history. `freeze_source_universe` refuses any other REAL universe name:
# a source chosen after seeing the data is not a preregistered source.
# ---------------------------------------------------------------------------

SELECTED_UNIVERSE = {
    "name": "seed20116",
    "n_games": 800,
    "summary_sha1": "18a015fa804fc0d3866feb42e2d637ce11e87930",
    "jsonl_sha1": "789ab890f606aebe87dead98b2207d2dc4760c65",
    "replay_data_sha1": "13e6b3d6414be580bef2b9ff1b02d2f3a29ba445",
    # The pairing is COLOUR-BALANCED: each checkpoint plays black in 400 games
    # and red in the other 400, so the authenticated identity is the unordered
    # PAIR plus the split -- not a fixed black/red assignment. Checking only the
    # first filename-sorted replay would have concluded a fixed assignment and
    # then rejected game 1.
    "checkpoint_pair": (
        "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors",
        "checkpoints/alphazero-v2-staged/model_iter_0379.safetensors"),
    "checkpoint_sha1s": {
        "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors":
            "209cf2d4fd24a48553d259dd71b4954867b9473e",
        "checkpoints/alphazero-v2-staged/model_iter_0379.safetensors":
            "8ad62ac432c35c6ea9b0630b8a2b8c572a0b03a1",
    },
    "anchor_checkpoint":
        "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors",
    "games_per_colour": 400,
    "exact_800_has_no_margin": (
        "the source survives the game exclusion at EXACTLY 800 games. Any "
        "future whole-game exclusion, or any drift in these hashes, drops it "
        "below the frozen minimum and INVALIDATES this selection -- it requires "
        "a new source decision, never a silent fallback to candidate 2 or 3"),
}

# ---------------------------------------------------------------------------
# Candidate universes, in preference order. Each is VERIFIED at run time.
# ---------------------------------------------------------------------------

CANDIDATE_UNIVERSES = (
    {
        "name": "seed20116",
        "preference": 1,
        "replay_dir": f"{LOGS}/0379_vs_calib020_0001_800g_w4_seed20116_replays",
        "summary": f"{LOGS}/0379_vs_calib020_0001_800g_w4_seed20116_replay.json",
        "jsonl": f"{LOGS}/0379_vs_calib020_0001_800g_w4_seed20116_replay_games.jsonl",
        "min_distinct_games": 800,
        "note": (
            "800 games under the same calib020_0001 anchor, seed range disjoint "
            "from A's seed20115, from v16 production (20300000+) and from v17 "
            "development (20310000+). Retired at the corpus-composition stage "
            "for branching-band geometry, never consumed as evidence"),
    },
    {
        "name": "v17_development",
        "preference": 2,
        "replay_dir": f"{LOGS}/fpu_v17_baseline_policy_mass/development/calib020_0001_vs_0379_1600g_w4_seed20310000_replays",
        "summary": f"{LOGS}/fpu_v17_baseline_policy_mass/development/calib020_0001_vs_0379_1600g_w4_seed20310000.json",
        "jsonl": f"{LOGS}/fpu_v17_baseline_policy_mass/development/calib020_0001_vs_0379_1600g_w4_seed20310000_games.jsonl",
        "min_distinct_games": 800,
        "note": "authenticated v17 development reservoir, minus its 32 selected rows",
    },
    {
        "name": "v16_production",
        "preference": 3,
        "replay_dir": f"{LOGS}/fpu_v16_policy_mass_v2/production_v2_b400amend_4000g_seed20300000/calib020_0001_vs_0379_4000g_w4_seed20300000_replays",
        "summary": f"{LOGS}/fpu_v16_policy_mass_v2/production_v2_b400amend_4000g_seed20300000/calib020_0001_vs_0379_4000g_w4_seed20300000.json",
        "jsonl": f"{LOGS}/fpu_v16_policy_mass_v2/production_v2_b400amend_4000g_seed20300000/calib020_0001_vs_0379_4000g_w4_seed20300000_games.jsonl",
        "min_distinct_games": 800,
        "note": "v16 production reservoir, minus its 120 selected rows",
    },
)

PHASE_ORDER = ("opening", "early_mid", "midgame", "late")


def phase_of_ply(ply: int) -> str:
    for name in PHASE_ORDER:
        low, high = PHASE_WINDOWS[name]
        if ply >= low and (high is None or ply <= high):
            return name
    raise ValueError(f"ply {ply} matched no phase window")


def side_to_move_for_ply(ply: int) -> str:
    """Red opens, so even plies are red to move. Verified against the replay's
    own `player` field, which `census_for_game` cross-checks."""
    return "red" if ply % 2 == 0 else "black"


# ---------------------------------------------------------------------------
# Test-only universes. Small inline fixtures, so the unit tests need no replay
# data and no GPU. Named with dunders so they can never collide with a real one.
# ---------------------------------------------------------------------------


def synthetic_game(n_moves: int, game_idx: int) -> Dict:
    """A replay-shaped dict with the real move schema. Moves walk the board in
    a fixed pattern; only `ply`, `player` and `n_legal` are read by the census."""
    moves = [
        {"ply": p, "player": side_to_move_for_ply(p),
         "row": 2 + (p // 20) % 20, "col": 2 + p % 20,
         "n_legal": 528 - p, "root_value": 0.0, "root_top1_share": 0.3,
         "root_total_visits": 400, "selected_visit_count": 120,
         "selected_visit_rank": 1}
        for p in range(n_moves)
    ]
    return {"game_idx": game_idx, "board_size": 24, "n_moves": n_moves,
            "moves": moves, "winner": "black", "reason": "win",
            "black_checkpoint": "fixture/black.safetensors",
            "red_checkpoint": "fixture/red.safetensors",
            "seed": 20260729, "schema_version": 1, "pairing_id": 0,
            "task_id": 0, "winner_checkpoint": "fixture/black.safetensors"}


_FIXTURE_UNIVERSES = {
    # Three games with positions, plus one that yields nothing at all, so
    # zero-yield retention is exercised rather than assumed.
    "__fixture__": {"lengths": [40, 120, 95, 0], "min_distinct_games": 3},
    "__empty__": {"lengths": [], "min_distinct_games": 1},
    # The revision-1 failure shape: many rows, far too few distinct games.
    "__degenerate__": {"lengths": [200, 200], "min_distinct_games": 30},
    # 799 survivors against a floor of 800.
    "__short__": {"lengths": [120] * 3, "min_distinct_games": 800},
}


def _universe_spec(name: str) -> Dict:
    for spec in CANDIDATE_UNIVERSES:
        if spec["name"] == name:
            return spec
    if name in _FIXTURE_UNIVERSES:
        fixture = _FIXTURE_UNIVERSES[name]
        return {"name": name, "preference": 0, "fixture": True,
                "lengths": fixture["lengths"],
                "min_distinct_games": fixture["min_distinct_games"]}
    raise ValueError(f"unknown universe {name!r}")


def _fixture_games(spec: Dict) -> List[Tuple[str, str, Dict]]:
    out = []
    for idx, length in enumerate(spec["lengths"]):
        game = synthetic_game(length, idx)
        raw = canonical_json_bytes(game)
        out.append((hashlib.sha1(raw).hexdigest(), f"__fixture__/game_{idx:06d}.json",
                    game))
    return out


# ---------------------------------------------------------------------------
# Exclusion sets -- computed, never assumed.
# ---------------------------------------------------------------------------


def _parse_csv(raw: bytes) -> List[Dict[str, str]]:
    return list(csv.DictReader(raw.decode().splitlines()))


def _snapshot(label: str, path: str, expected_sha1: str) -> Tuple[bytes, str]:
    """Read ONCE, hash THOSE bytes, and hand them back for parsing.

    AUTHENTICATED BYTES MUST BE THE BYTES PARSED. Hashing a path and then
    reopening it is not authentication: a file that changes between the two
    reads yields census rows derived from bytes the record does not describe.
    Every consumer below parses the buffer returned here and never reopens.
    """
    raw = Path(path).read_bytes()
    actual = hashlib.sha1(raw).hexdigest()
    if actual != expected_sha1:
        raise ValueError(
            f"{label}: {path} hashes {actual}, pinned {expected_sha1}. A "
            f"replacement file with an unchanged row count would pass a count "
            f"check; refusing to build evidence from an artifact that moved")
    return raw, actual


def _authenticate_file(label: str, path: str, expected_sha1: str) -> str:
    """Hash-only authentication, for artifacts that are never parsed (the
    checkpoint weights). Anything whose CONTENT is consumed uses `_snapshot`."""
    actual = fpu_provenance.file_sha1(path)
    if actual != expected_sha1:
        raise ValueError(
            f"{label}: {path} hashes {actual}, pinned {expected_sha1}")
    return actual


def authenticate_forbidden_sources(
        sources: Sequence[Dict] = FORBIDDEN_SOURCES) -> Tuple[Dict[str, str],
                                                              Dict[str, bytes]]:
    """Byte-authenticate every forbidden evidence file.

    Returns `(verified_hashes, payloads)`. The payloads ARE the authenticated
    bytes; `forbidden_canonical_hashes` parses them and never reopens a path.
    """
    verified: Dict[str, str] = {}
    payloads: Dict[str, bytes] = {}
    for source in sources:
        name = source["name"]
        if source["kind"] == "replay_dir":
            _, verified[f"{name}.summary_sha1"] = _snapshot(
                name, source["summary"], source["summary_sha1"])
            _, verified[f"{name}.jsonl_sha1"] = _snapshot(
                name, source["jsonl"], source["jsonl_sha1"])
            paths = sorted(str(p) for p in Path(source["path"]).glob("game_*.json"))
            actual = fpu_provenance.replay_data_sha1(paths)
            if actual != source["replay_data_sha1"]:
                raise ValueError(
                    f"{name}: replay_data_sha1 {actual} != pinned "
                    f"{source['replay_data_sha1']}")
            verified[f"{name}.replay_data_sha1"] = actual
            continue
        payloads[name], verified[name] = _snapshot(
            name, source["path"], FORBIDDEN_SOURCE_SHA1S[name])
        if source["kind"] == "goal_line_manifest_json":
            payloads[f"{name}.cases_csv"], verified[f"{name}.cases_csv"] = _snapshot(
                name, source["cases_csv"],
                FORBIDDEN_SOURCE_SHA1S["gate_B_cases_csv"])
    return verified, payloads


def authenticate_selected_universe(spec: Dict) -> Dict[str, str]:
    """Cross-authenticate the whole source chain, not four hashes side by side.

    Hashing the summary and the JSONL proves each file is unchanged; it proves
    nothing about whether they DESCRIBE the replays being measured. This walks
    the chain: JSONL rows 1:1 onto replay sidecars, every scalar cross-checked,
    and the checkpoint pair verified on EVERY selected replay -- checking only
    the first filename-sorted replay cannot establish uniform identity.
    """
    replay_paths = sorted(str(p) for p in Path(spec["replay_dir"]).glob("game_*.json"))
    n_expected = SELECTED_UNIVERSE["n_games"]
    if len(replay_paths) != n_expected:
        raise ValueError(
            f"selected universe has {len(replay_paths)} replays, {n_expected} pinned")

    summary_raw, summary_sha1 = _snapshot(
        "selected.summary", spec["summary"], SELECTED_UNIVERSE["summary_sha1"])
    jsonl_raw, jsonl_sha1 = _snapshot(
        "selected.jsonl", spec["jsonl"], SELECTED_UNIVERSE["jsonl_sha1"])
    replay_data = fpu_provenance.replay_data_sha1(replay_paths)
    if replay_data != SELECTED_UNIVERSE["replay_data_sha1"]:
        raise ValueError(
            f"selected replay_data_sha1 {replay_data} != pinned "
            f"{SELECTED_UNIVERSE['replay_data_sha1']}")

    summary = json.loads(summary_raw.decode())
    if summary["games"] != n_expected:
        raise ValueError(
            f"summary reports {summary['games']} games, {n_expected} pinned")
    pinned_pair = set(SELECTED_UNIVERSE["checkpoint_pair"])
    if {summary["checkpoint_a"], summary["checkpoint_b"]} != pinned_pair:
        raise ValueError("summary checkpoint pair contradicts SELECTED_UNIVERSE")

    rows = [json.loads(line) for line in
            jsonl_raw.decode().splitlines() if line.strip()]
    if len(rows) != n_expected:
        raise ValueError(f"JSONL has {len(rows)} rows, {n_expected} pinned")

    by_idx = {}
    for row in rows:
        if row["game_idx"] in by_idx:
            raise ValueError(f"JSONL repeats game_idx {row['game_idx']}")
        by_idx[row["game_idx"]] = row
    if len(by_idx) != n_expected:
        raise ValueError("JSONL game_idx set is not 1:1 with the pinned count")

    anchor = SELECTED_UNIVERSE["anchor_checkpoint"]
    opponent = next(c for c in SELECTED_UNIVERSE["checkpoint_pair"] if c != anchor)
    games_by_colour = {"black": {"anchor": 0, "opponent": 0},
                       "red": {"anchor": 0, "opponent": 0}}

    seen_paths = set()
    # ONE read per replay. The parsed object below and the per-file digest come
    # from the same buffer, so no sidecar can change between them.
    replays: Dict[str, Dict] = {}
    replay_sha1s: Dict[str, str] = {}
    for path in replay_paths:
        raw = Path(path).read_bytes()
        replay = json.loads(raw.decode())
        replays[path] = replay
        replay_sha1s[path] = hashlib.sha1(raw).hexdigest()
        idx = int(replay["game_idx"])
        row = by_idx.get(idx)
        if row is None:
            raise ValueError(f"{path}: game_idx {idx} absent from the JSONL")
        if str(Path(row["replay_path"])) != str(Path(path)):
            raise ValueError(
                f"game {idx}: JSONL replay_path {row['replay_path']!r} does not "
                f"name the sidecar {path!r} it is paired with")
        seen_paths.add(str(Path(row["replay_path"])))
        for field in ("winner", "reason", "n_moves", "black_checkpoint",
                      "red_checkpoint", "winner_checkpoint", "pairing_id",
                      "task_id"):
            if row[field] != replay[field]:
                raise ValueError(
                    f"game {idx}: JSONL {field}={row[field]!r} contradicts the "
                    f"replay's {replay[field]!r}")
        if len(replay["moves"]) != replay["n_moves"]:
            raise ValueError(
                f"game {idx}: {len(replay['moves'])} moves recorded but "
                f"n_moves={replay['n_moves']}")
        # EVERY replay, not merely the first filename-sorted one. The pairing is
        # colour-balanced, so the invariant is the unordered PAIR.
        if {replay["black_checkpoint"], replay["red_checkpoint"]} != pinned_pair:
            raise ValueError(
                f"game {idx}: checkpoint pair "
                f"({replay['black_checkpoint']}, {replay['red_checkpoint']}) "
                f"contradicts the authenticated pair {sorted(pinned_pair)}")
        games_by_colour["black"][
            "anchor" if replay["black_checkpoint"] == anchor else "opponent"] += 1
        games_by_colour["red"][
            "anchor" if replay["red_checkpoint"] == anchor else "opponent"] += 1
    if len(seen_paths) != n_expected:
        raise ValueError(
            f"JSONL replay_path set covers {len(seen_paths)} sidecars, "
            f"{n_expected} required -- the 1:1 chain is incomplete")

    # The COLOUR SCHEDULE is part of the identity: an unbalanced pairing is a
    # different experiment even with the same two checkpoints. Both colours are
    # counted for both roles, so an imbalance cannot hide in the colour the
    # earlier one-sided count never looked at.
    each = SELECTED_UNIVERSE["games_per_colour"]
    expected_schedule = {"black": {"anchor": each, "opponent": each},
                         "red": {"anchor": each, "opponent": each}}
    if games_by_colour != expected_schedule:
        raise ValueError(
            f"colour schedule {games_by_colour} is not the pinned "
            f"{expected_schedule}")
    a_is_anchor = summary["checkpoint_a"] == anchor
    a_role = "anchor" if a_is_anchor else "opponent"
    if games_by_colour["black"][a_role] != summary["a_as_black"]["games"]:
        raise ValueError(
            f"replay colour schedule contradicts the summary: checkpoint_a "
            f"played black in {games_by_colour['black'][a_role]} replays, "
            f"summary reports {summary['a_as_black']['games']}")
    if games_by_colour["red"][a_role] != summary["a_as_red"]["games"]:
        raise ValueError(
            f"replay colour schedule contradicts the summary: checkpoint_a "
            f"played red in {games_by_colour['red'][a_role]} replays, "
            f"summary reports {summary['a_as_red']['games']}")

    checkpoint_sha1s = {
        path: _authenticate_file(f"checkpoint {path}", path, expected)
        for path, expected in SELECTED_UNIVERSE["checkpoint_sha1s"].items()
    }
    return {
        "summary_sha1": summary_sha1,
        "jsonl_sha1": jsonl_sha1,
        "replay_data_sha1": replay_data,
        "checkpoint_sha1s": checkpoint_sha1s,
        "anchor_checkpoint": anchor,
        "opponent_checkpoint": opponent,
        "games_by_colour": games_by_colour,
        "jsonl_rows_matched_1to1": n_expected,
        # The authenticated snapshot the census consumes. Nothing downstream
        # reopens a sidecar.
        "replays": replays,
        "replay_sha1s": replay_sha1s,
    }


def _reverify_replay_data(spec: Dict) -> None:
    """Closing re-authentication of the SELECTED universe: the aggregate must
    still be the pinned value AFTER the census has consumed the snapshot."""
    paths = sorted(str(p) for p in Path(spec["replay_dir"]).glob("game_*.json"))
    actual = fpu_provenance.replay_data_sha1(paths)
    if actual != SELECTED_UNIVERSE["replay_data_sha1"]:
        raise ValueError(
            f"replay data changed during the freeze: {actual} != pinned "
            f"{SELECTED_UNIVERSE['replay_data_sha1']}. No artifact written")


def reverify_all_replay_sources(spec: Dict) -> None:
    """Closing check over EVERY replay reservoir the record depends on -- the
    selected universe AND the three forbidden reservoirs that define the
    exclusion set. Runs after all derivation and before any write, so drift
    anywhere in the evidence chain leaves no artifact behind."""
    _reverify_replay_data(spec)
    authenticate_forbidden_reservoirs(when="closing")


def _verify_counts(source: Dict, total_rows: int, distinct: int) -> None:
    if total_rows != source["total_rows"] or distinct != source["distinct_positions"]:
        raise ValueError(
            f"{source['name']}: measured ({total_rows} rows, {distinct} distinct "
            f"positions) contradicts the pinned "
            f"({source['total_rows']}, {source['distinct_positions']}). Refusing "
            f"to build an exclusion set from an artifact that moved")


def _verify_reservoir(reservoir: Dict, when: str) -> str:
    paths = sorted(str(p) for p in Path(reservoir["dir"]).glob("game_*.json"))
    if len(paths) != reservoir["n_games"]:
        raise ValueError(
            f"forbidden reservoir {reservoir['name']}: {len(paths)} replays, "
            f"{reservoir['n_games']} pinned ({when})")
    actual = fpu_provenance.replay_data_sha1(paths)
    if actual != reservoir["replay_data_sha1"]:
        raise ValueError(
            f"forbidden reservoir {reservoir['name']} changed during the freeze "
            f"({when}): {actual} != pinned {reservoir['replay_data_sha1']}. "
            f"No artifact written")
    return actual


def authenticate_forbidden_reservoirs(when: str = "opening") -> Dict[str, str]:
    """Bind every replay reservoir the forbidden evidence points into.

    The exclusion set is DERIVED from these sidecars, so binding only the probe
    CSVs would leave the set itself unauthenticated.
    """
    return {r["name"]: _verify_reservoir(r, when)
            for r in FORBIDDEN_REPLAY_RESERVOIRS}


def _reservoir_for(replay_path: str) -> Dict:
    """Every referenced replay must live in a PINNED reservoir. A gate source
    that starts pointing somewhere new fails loudly rather than contributing
    silently unauthenticated exclusions."""
    parent = str(Path(replay_path).parent)
    for reservoir in FORBIDDEN_REPLAY_RESERVOIRS:
        if parent == reservoir["dir"]:
            return reservoir
    raise ValueError(
        f"replay {replay_path} lies outside every pinned forbidden reservoir "
        f"({[r['name'] for r in FORBIDDEN_REPLAY_RESERVOIRS]}); its bytes "
        f"cannot be authenticated, so it may not define an exclusion")


def forbidden_canonical_hashes(payloads: Dict[str, bytes],
                               sources: Sequence[Dict] = FORBIDDEN_SOURCES) -> Set[str]:
    """Every consumed or acceptance POSITION, as a canonical state SHA-1.

    Parses the AUTHENTICATED payloads handed in by
    `authenticate_forbidden_sources`; it never reopens a pinned path, so the
    bytes that produced these exclusions are exactly the bytes the record
    hashes. Corpus manifests already record `canonical_position_sha1` from this
    same function, so their column is used directly -- verified equal on
    sampled rows.
    """
    hashes: Set[str] = set()
    for source in sources:
        kind = source["kind"]
        name = source["name"]
        if kind == "replay_dir":
            continue                      # game-level exclusion, not positional
        if name not in payloads:
            raise ValueError(
                f"{name}: no authenticated payload supplied; exclusions may "
                f"only be derived from authenticated bytes")
        if kind == "corpus_manifest_csv":
            rows = _parse_csv(payloads[name])
            distinct = {r["canonical_position_sha1"] for r in rows}
            _verify_counts(source, len(rows), len(distinct))
            hashes |= distinct
            continue
        if kind == "goal_line_manifest_json":
            cases = json.loads(payloads[name].decode())["cases"]
            rows = [{"replay_path": c["replay_path"],
                     "position_ply": c["position_ply"],
                     "side_to_move": c["side_to_move"]} for c in cases]
        elif kind == "probe_cases_csv":
            rows = _parse_csv(payloads[name])
        else:
            raise ValueError(f"{name}: unknown kind {kind!r}")

        seen = set()
        for row in rows:
            seen.add((row["replay_path"], int(float(row["position_ply"]))))
        _verify_counts(source, len(rows), len(seen))
        for replay_path, ply in sorted(seen):
            # The reservoir was authenticated before this call and is
            # re-authenticated before anything is written, so these bytes are
            # bound at both ends of the derivation.
            _reservoir_for(replay_path)
            replay = json.loads(Path(replay_path).read_bytes().decode())
            state = position_state(replay, ply, side_to_move_for_ply(ply))
            hashes.add(canonical_state_sha1(state))
    return hashes


def forbidden_game_content_sha1s(
        sources: Sequence[Dict] = FORBIDDEN_SOURCES) -> Set[str]:
    """Replay CONTENT SHA-1s of every forbidden GAME, via the imported helper."""
    out: Set[str] = set()
    for source in sources:
        if source["kind"] != "replay_dir":
            continue
        # Identity is derived INSIDE the authenticated window: the reservoir is
        # checked before this call and re-checked before any write.
        _reservoir_for(str(Path(source["path"]) / "game_000000.json"))
        paths = sorted(Path(source["path"]).glob("game_*.json"))
        if source.get("n_games") is not None and len(paths) != source["n_games"]:
            raise ValueError(
                f"{source['name']}: {len(paths)} replays found, {source['n_games']} "
                f"pinned. Refusing to build a game exclusion set from a directory "
                f"that moved")
        out |= set(game_identities([{"replay_path": str(p)} for p in paths]))
    return out


def apply_exclusions(rows: Sequence[Dict], forbidden_hashes: Set[str],
                     forbidden_games: Set[str]) -> Tuple[List[Dict], Dict]:
    """Drop rows whose canonical position OR whose game is forbidden.

    The report is emitted even when both counts are zero: a pool whose
    exclusions removed nothing has not been verified, and a zero must be
    VISIBLE rather than implied.
    """
    kept, by_hash, by_game = [], 0, 0
    for row in rows:
        if row.get("game_content_sha1") in forbidden_games:
            by_game += 1
            continue
        if row.get("canonical_sha1") in forbidden_hashes:
            by_hash += 1
            continue
        kept.append(row)
    return kept, {
        "input_rows": len(rows),
        "excluded_by_canonical_hash": by_hash,
        "excluded_by_game": by_game,
        "kept_rows": len(kept),
    }


# ---------------------------------------------------------------------------
# Census enumeration -- replay-only predicates.
# ---------------------------------------------------------------------------

SELECTION_INPUTS = {
    "fields_used": ("ply", "player", "n_legal", "reason", "winner", "n_moves"),
    "phase_windows": "PHASE_WINDOWS, by ply",
    "position_rule": CENSUS["position_rule"],
    "quantiles": CENSUS["quantiles"],
    "value_based_rule": "NONE -- replay root_value is checkpoint-contaminated",
    "near_even_rule": "NONE -- cannot be applied pre-search",
    "clip_statistic_rule": "NONE -- does not exist pre-search",
}


def assert_census_within_ceiling(total: int) -> None:
    """Abort BEFORE the evaluator loads rather than silently truncating."""
    ceiling = CENSUS["max_total_searches"]
    if total > ceiling:
        raise ValueError(
            f"census of {total} positions exceeds the frozen ceiling {ceiling} "
            f"({UNIVERSE['n_games']} games x {CENSUS['positions_per_game']}); "
            f"aborting before the evaluator loads")
    return None


def census_for_game(replay: Dict, game_content_sha1: str,
                    replay_path: str) -> Tuple[List[Dict], Dict[str, int]]:
    """The per-game phase-stratified census. Returns `(rows, per_phase_counts)`.

    A phase with no qualifying ply contributes ZERO and is reported; it is never
    backfilled from another phase, and short late supply collapses without
    replacement rather than emitting the same ply repeatedly.
    """
    by_phase: Dict[str, List[int]] = {name: [] for name in PHASE_ORDER}
    for move in replay["moves"]:
        ply = int(move["ply"])
        if move["player"] != side_to_move_for_ply(ply):
            raise ValueError(
                f"{replay_path}: move ply {ply} has player {move['player']!r}, "
                f"contradicting the red-opens convention")
        by_phase[phase_of_ply(ply)].append(ply)

    rows: List[Dict] = []
    counts = {name: 0 for name in PHASE_ORDER}
    for name in PHASE_ORDER:
        plies = sorted(set(by_phase[name]))
        if not plies:
            continue                      # contributes zero, reported below
        chosen: List[int] = []
        for q in CENSUS["quantiles"][name]:
            rank = max(1, math.ceil(q * len(plies)))
            ply = plies[rank - 1]
            if ply not in chosen:         # without replacement
                chosen.append(ply)
        for ply in chosen:
            move = next(m for m in replay["moves"] if int(m["ply"]) == ply)
            rows.append({
                "game_content_sha1": game_content_sha1,
                "game_idx": int(replay["game_idx"]),
                "replay_path": replay_path,
                "position_ply": ply,
                "side_to_move": side_to_move_for_ply(ply),
                "phase": name,
                "n_legal": int(move["n_legal"]),
                "game_reason": replay["reason"],
                "game_winner": replay["winner"],
                "game_n_moves": int(replay["n_moves"]),
            })
        counts[name] = len(chosen)
    return rows, counts


def _iter_games(spec: Dict, snapshot: Dict = None) -> Iterable[Tuple[str, str, Dict]]:
    if spec.get("fixture"):
        yield from _fixture_games(spec)
        return
    if snapshot is not None:
        # Consume the AUTHENTICATED buffers; never reopen a sidecar.
        for path in sorted(snapshot["replays"]):
            yield snapshot["replay_sha1s"][path], path, snapshot["replays"][path]
        return
    for path in sorted(Path(spec["replay_dir"]).glob("game_*.json")):
        raw = path.read_bytes()
        yield hashlib.sha1(raw).hexdigest(), str(path), json.loads(raw)


def enumerate_census(universe_spec: Dict, *,
                     forbidden_games: Set[str] = frozenset(),
                     with_canonical: bool = True,
                     snapshot: Dict = None) -> Dict:
    """Steps 2-4 of the frozen order: game exclusions, the 800 cut, then the
    per-game census INSIDE those games. Position exclusions are step 5 and are
    applied by the caller, so they can never remove a game."""
    surviving: List[Tuple[str, str, Dict]] = []
    excluded_games = 0
    total_games = 0
    for sha1, path, replay in _iter_games(universe_spec, snapshot):
        total_games += 1
        if sha1 in forbidden_games:
            excluded_games += 1
            continue
        surviving.append((sha1, path, replay))

    # Step 3: content-SHA ascending, take exactly the first N. Deterministic and
    # independent of filesystem order.
    surviving.sort(key=lambda item: item[0])
    minimum = universe_spec["min_distinct_games"]
    chosen = surviving[:minimum]

    rows: List[Dict] = []
    per_phase = {name: 0 for name in PHASE_ORDER}
    per_game_phases: Dict[str, Dict[str, int]] = {}
    for sha1, path, replay in chosen:
        game_rows, counts = census_for_game(replay, sha1, path)
        rows.extend(game_rows)
        per_game_phases[sha1] = counts
        for name in PHASE_ORDER:
            per_phase[name] += counts[name]

    if with_canonical:
        parsed = {sha1: replay for sha1, _p, replay in _iter_games(
            universe_spec, snapshot)} if not universe_spec.get("fixture") else {}
        for row in rows:
            replay = parsed.get(row["game_content_sha1"])
            if replay is None:
                # Fixture rows are hashed from the synthetic state directly.
                row["canonical_sha1"] = hashlib.sha1(
                    f"{row['game_content_sha1']}:{row['position_ply']}".encode()
                ).hexdigest()
            else:
                state = position_state(replay, row["position_ply"],
                                       row["side_to_move"])
                row["canonical_sha1"] = canonical_state_sha1(state)

    return {
        "total_games_seen": total_games,
        "excluded_games": excluded_games,
        "surviving_games": len(surviving),
        "all_game_ids": [sha1 for sha1, _p, _r in chosen],
        "census_positions": rows,
        "per_phase": per_phase,
        "per_game_phases": per_game_phases,
    }


# ---------------------------------------------------------------------------
# Report (Step 5) and freeze (Execution Phase step 3).
# ---------------------------------------------------------------------------


def report_universe(universe_name: str) -> Dict:
    """READ-ONLY enumeration. Writes NO artifact and binds nothing.

    Read-only is not the same as unauthenticated. This runs the SAME
    opening/closing authentication as the freeze -- forbidden payloads, all
    three forbidden reservoirs, and the selected universe's snapshot, which is
    handed to `enumerate_census` rather than letting it reread the sidecars.
    A report derived from unverified bytes is not a report, it is a guess.

    Real reports accept ONLY the selected source. Step 5 chose `seed20116`;
    leaving a route to enumerate candidates 2 and 3 afterwards would be a
    post-selection inspection path, which is exactly what preregistration
    exists to close. Fixtures stay available for the unit tests.
    """
    spec = _universe_spec(universe_name)
    fixture = spec.get("fixture", False)
    if not fixture and universe_name != SELECTED_UNIVERSE["name"]:
        raise ValueError(
            f"universe {universe_name!r} is not the selected source "
            f"{SELECTED_UNIVERSE['name']!r}. The Step 5 decision is bound in "
            f"tracked code; enumerating an alternative source afterwards is a "
            f"post-selection inspection route, not a report")

    if fixture:
        forbidden_games, forbidden_hashes, snapshot = set(), set(), None
    else:
        _verified, payloads = authenticate_forbidden_sources()
        authenticate_forbidden_reservoirs("opening")
        snapshot = authenticate_selected_universe(spec)
        forbidden_games = forbidden_game_content_sha1s()
        forbidden_hashes = forbidden_canonical_hashes(payloads)

    enumerated = enumerate_census(spec, forbidden_games=forbidden_games,
                                  snapshot=snapshot)
    kept, exclusion_report = apply_exclusions(
        enumerated["census_positions"], forbidden_hashes, set())

    sides: Dict[str, int] = {}
    phases: Dict[str, int] = {name: 0 for name in PHASE_ORDER}
    games_with_rows = set()
    for row in kept:
        sides[row["side_to_move"]] = sides.get(row["side_to_move"], 0) + 1
        phases[row["phase"]] += 1
        games_with_rows.add(row["game_content_sha1"])

    if not fixture:
        # Closing check before the numbers are handed to a human: figures
        # derived from bytes that drifted mid-enumeration are not a report.
        reverify_all_replay_sources(spec)

    return {
        "universe": universe_name,
        "authenticated": not fixture,
        "preference": spec.get("preference"),
        "total_games_seen": enumerated["total_games_seen"],
        "excluded_games_by_content_sha1": enumerated["excluded_games"],
        "surviving_games": enumerated["surviving_games"],
        "n_games": len(enumerated["all_game_ids"]),
        "meets_minimum": (enumerated["surviving_games"]
                          >= spec["min_distinct_games"]),
        "min_distinct_games": spec["min_distinct_games"],
        "census_positions_before_position_exclusions":
            len(enumerated["census_positions"]),
        "census_positions_after": len(kept),
        "exclusion_report": exclusion_report,
        "per_phase_before": enumerated["per_phase"],
        "per_phase_after": phases,
        "side_balance": sides,
        "games_contributing_at_least_one_row": len(games_with_rows),
        "zero_yield_games": len(enumerated["all_game_ids"]) - len(games_with_rows),
        "selection_inputs": SELECTION_INPUTS,
    }


def freeze_source_universe(out_path: str, universe_name: str, seed: int, *,
                           extra_forbidden_hashes: Set[str] = frozenset()) -> Dict:
    """Apply both exclusions in the frozen order, then bind the survivors.

    Refuses rather than shrinking: an empty result is `collapsed`, and fewer
    DISTINCT GAMES than the minimum is a stop. Distinct games -- not rows -- is
    the binding supply, because the cohort takes at most one position per game.
    """
    spec = _universe_spec(universe_name)
    fixture = spec.get("fixture", False)
    if not fixture and universe_name != SELECTED_UNIVERSE["name"]:
        raise ValueError(
            f"universe {universe_name!r} is not the selected source "
            f"{SELECTED_UNIVERSE['name']!r}. The Step 5 decision is bound in "
            f"tracked code; a source chosen after seeing the data is not a "
            f"preregistered source")

    if fixture:
        verified_forbidden, payloads, verified_chain = {}, {}, {}
        verified_reservoirs = {}
        forbidden_games, forbidden_hashes = set(), set()
    else:
        verified_forbidden, payloads = authenticate_forbidden_sources()
        # Opening check on every reservoir the exclusions are derived FROM,
        # before a single canonical hash or game identity is computed.
        verified_reservoirs = authenticate_forbidden_reservoirs("opening")
        verified_chain = authenticate_selected_universe(spec)
        forbidden_games = forbidden_game_content_sha1s()
        forbidden_hashes = forbidden_canonical_hashes(payloads)
    forbidden_hashes = set(forbidden_hashes) | set(extra_forbidden_hashes)

    enumerated = enumerate_census(
        spec, forbidden_games=forbidden_games,
        snapshot=None if fixture else verified_chain)
    if not enumerated["all_game_ids"]:
        raise ValueError(
            f"universe {universe_name!r} collapsed to zero games after the "
            f"content-SHA game exclusion")
    if enumerated["surviving_games"] < spec["min_distinct_games"]:
        raise ValueError(
            f"universe {universe_name!r} supplies {enumerated['surviving_games']} "
            f"distinct games, fewer than the required "
            f"{spec['min_distinct_games']}; the cohort takes at most one "
            f"position per game, so distinct games is the binding supply")

    kept, exclusion_report = apply_exclusions(
        enumerated["census_positions"], forbidden_hashes, set())
    assert_census_within_ceiling(len(kept))

    games_with_rows = {row["game_content_sha1"] for row in kept}
    per_phase_after = {name: 0 for name in PHASE_ORDER}
    for row in kept:
        per_phase_after[row["phase"]] += 1

    # Geometry that RECONCILES with the rows actually carried. Recording only
    # the pre-exclusion phase counts beside a post-exclusion row list makes the
    # record describe two different populations at once, and drops the
    # game-exclusion count entirely.
    census_geometry = {
        "positions_before_position_exclusions":
            len(enumerated["census_positions"]),
        "positions_after_position_exclusions": len(kept),
        "per_phase_before": enumerated["per_phase"],
        "per_phase_after": per_phase_after,
        "games_seen": enumerated["total_games_seen"],
        "games_excluded_by_content_sha1": enumerated["excluded_games"],
        "games_surviving_game_exclusion": enumerated["surviving_games"],
        "games_selected": len(enumerated["all_game_ids"]),
        "positions_excluded_by_canonical_hash":
            exclusion_report["excluded_by_canonical_hash"],
    }

    record = {
        "run_kind": "shipped_only_preflight_source_universe",
        "scientific_interpretation_forbidden": True,
        "selection_is_independent_of_residual_exposure": True,
        "universe": universe_name,
        "seed": seed,
        "n_games": len(enumerated["all_game_ids"]),
        "all_game_ids": enumerated["all_game_ids"],
        "zero_yield_games": len(enumerated["all_game_ids"]) - len(games_with_rows),
        "census_positions": kept,
        "census_geometry": census_geometry,
        "per_phase": per_phase_after,
        "exclusion_report": exclusion_report,
        "selection_inputs": SELECTION_INPUTS,
        "forbidden_sources": [s["name"] for s in FORBIDDEN_SOURCES],
        "verified_forbidden_source_sha1s": verified_forbidden,
        "verified_forbidden_reservoirs": verified_reservoirs,
        "selected_universe": SELECTED_UNIVERSE if not fixture else "fixture",
        "summary_sha1": verified_chain.get("summary_sha1", "fixture"),
        "jsonl_sha1": verified_chain.get("jsonl_sha1", "fixture"),
        # The ESTABLISHED length-delimited helper over the exact selected
        # replays -- never a local re-derivation.
        "replay_data_sha1": verified_chain.get("replay_data_sha1", "fixture"),
        "checkpoint_sha1s": verified_chain.get("checkpoint_sha1s", "fixture"),
        "anchor_checkpoint": verified_chain.get("anchor_checkpoint", "fixture"),
        # Both colours for both roles. There is NO fixed black/red assignment in
        # this source, so `black_checkpoint_sha1` / `red_checkpoint_sha1` would
        # be false claims and are deliberately absent.
        "games_by_colour": verified_chain.get("games_by_colour", "fixture"),
        "jsonl_rows_matched_1to1": verified_chain.get("jsonl_rows_matched_1to1", 0),
        "git_commit": fpu_provenance.git_commit(),
        "worktree_clean": fpu_provenance.worktree_clean(),
    }
    _assert_record_reconciles(record)
    if not fixture:
        # Closing re-authentication BEFORE any bytes are written: a sidecar that
        # changed while the census or the exclusions consumed it leaves no
        # partial artifact. Covers the selected universe AND all three
        # forbidden reservoirs.
        reverify_all_replay_sources(spec)
    raw = canonical_json_bytes(record)
    record_out = dict(record)
    record_out["universe_sha1"] = hashlib.sha1(raw).hexdigest()
    with open(out_path, "wb") as handle:
        handle.write(raw)
    return record_out


def _selected_replay_paths(spec: Dict, enumerated: Dict) -> List[str]:
    """The exact replay files backing `all_game_ids`, for the established
    `fpu_provenance.replay_data_sha1` helper."""
    if spec.get("fixture"):
        return []
    return sorted(str(p) for p in Path(spec["replay_dir"]).glob("game_*.json"))


def _assert_record_reconciles(record: Dict) -> None:
    """The record must agree with the rows it carries.

    A frozen artifact whose geometry describes a different population than its
    row list is not authenticated, however many hashes it holds.
    """
    geometry = record["census_geometry"]
    kept = record["census_positions"]

    if sum(geometry["per_phase_after"].values()) != len(kept):
        raise ValueError(
            f"per_phase_after sums to {sum(geometry['per_phase_after'].values())} "
            f"but census_positions holds {len(kept)} rows")
    if geometry["positions_after_position_exclusions"] != len(kept):
        raise ValueError("positions_after does not equal the kept row count")
    if (geometry["positions_before_position_exclusions"]
            - geometry["positions_excluded_by_canonical_hash"] != len(kept)):
        raise ValueError(
            "position exclusions do not reconcile: before - excluded != after")
    if (geometry["games_seen"] - geometry["games_excluded_by_content_sha1"]
            != geometry["games_surviving_game_exclusion"]):
        raise ValueError(
            "game exclusions do not reconcile: seen - excluded != surviving")
    if geometry["games_selected"] != len(record["all_game_ids"]):
        raise ValueError("games_selected does not equal len(all_game_ids)")
    if geometry["games_selected"] != record["n_games"]:
        raise ValueError("n_games does not equal games_selected")
    if len(set(record["all_game_ids"])) != len(record["all_game_ids"]):
        raise ValueError("all_game_ids contains a duplicate identity")
    # Zero-yield games are RETAINED, so every kept row's game must be present
    # in all_game_ids and the count must include the games contributing nothing.
    yielding = {row["game_content_sha1"] for row in kept}
    if not yielding <= set(record["all_game_ids"]):
        raise ValueError("a kept row names a game absent from all_game_ids")
    if record["zero_yield_games"] != len(record["all_game_ids"]) - len(yielding):
        raise ValueError("zero_yield_games does not reconcile with all_game_ids")


def _main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true",
                        help="READ-ONLY enumeration; writes no artifact")
    parser.add_argument("--universe", default="seed20116")
    args = parser.parse_args(argv)
    if not args.report:
        parser.error("only --report is available from the CLI; the binding "
                     "freeze runs in the Execution Phase at a clean HEAD")
    out = dict(report_universe(args.universe))
    out.pop("selection_inputs", None)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
