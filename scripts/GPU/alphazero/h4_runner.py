"""H4 GAME RUNNER AND CANONICAL PERSISTENCE. GATED; NOT AUTHORIZED.

Frozen by `docs/superpowers/2026-09-22-t1j-h4-runner-persistence-card.md`
(amended 176ed89). 🔴 THE CARD IS THE AUTHORITY: where this module and the card
disagree, the card wins and the disagreement is an amendment, not a code fix.

Plays EMPTY-BOARD, COLOUR-REVERSED PAIRS through the real harness `play_task`
and the §4B-qualified H4 adapter, and persists every game COMPLETELY: every
move, who made it, which T1j routine answered, every JVM's process record with
its coherence digests, and a transcript digest RECOMPUTED from the moves.
Identical games are data: nothing is ever merged, collapsed or dropped.

🔴 THE HARNESS FLATTENS EXCEPTION TYPES. `play_task` turns every non-AbortError
raised by an agent or the binder into a bare `AbortError(... ) from None`, which
would make a timeout, unreadable output and an adapter refusal indistinguishable.
The runner's thin agent and binder wrappers therefore KEEP the ORIGINAL exception
before the harness converts it, and the `run_void` classification is decided by
THAT exception's type -- never by reading a message (card §3.4).

🔴 OUTCOME BLINDING (card §7, as amended). Winners and points appear in the
canonical results file and in exports DERIVED from it after the run; stdout,
stderr and the progress trace never carry them, and the runner computes no score.

🔴 FIXTURE MODE: negative synthetic seeds, a stub incumbent, header
`H4_SYNTHETIC_FIXTURE` with `evidence: false`.

🔴 PILOT AND STUDY MODE (step 3P-a, card §12): the PRODUCTION incumbent -- built
ONCE from the frozen argmax sources, its identity read off that object -- with no
injected seam of any kind; the schedule READ FROM THE COMMITTED MANIFEST; the
header bound to the manifest entry twice, before any destination is claimed and
again after compiling, before the first game. The gate lives in
`h4_pilot_authorization.py`, OUTSIDE this hashed file (card §12.6.2).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import e4_screen_integration as INT
from . import e4_screen_runner as HARNESS
from . import h2_match_rules as H2R
from . import h4_repair_qualification as H4RQ
from . import l0_match_rules as L0R
from . import t1j_adapter as A
from .d1_probe import D1Error, D1VoidError, Deadline, T1jPaths, _supervisor
from .d1_probe import _default_compile as _compile_helper_verified
from .e4_screen_runner import AbortError, PHASE_BIND, PHASE_MOVE

#: THE H4 PILOT IS NOT AUTHORIZED. The gate is `AUTH.H4_PILOT_EXECUTION_AUTHORIZED`,
#: in its own unhashed module (card §12.6.2), read AS AN ATTRIBUTE AT CALL TIME at
#: BOTH public entries -- `run_games` and `main` -- in EVERY mode, fixture
#: included: a fixture run with a real process boundary would launch real JVMs.
#: No override exists: not argv, not the environment, not a file, not an import hook.
from . import h4_pilot_authorization as AUTH

#: Card §2: the §4B-qualified runtime, READ, never retyped.
DEPTH = H4RQ.DEPTH
QUERY_TIMEOUT_S = H4RQ.PER_CALL_TIMEOUT_S
REPLAY_TIMEOUT_S = H4RQ.PER_CALL_TIMEOUT_S
PLY_CAP = L0R.PLY_CAP

#: Card §10.1: the mode decides the design label, the seed sign and the evidence flag.
DESIGNS = {"fixture": "H4_SYNTHETIC_FIXTURE", "pilot": "H4_PILOT", "study": "H4_STUDY"}
FIXTURE_DESIGN = DESIGNS["fixture"]
FIXTURE_NOTE = "SYNTHETIC FIXTURE — NOT A GAME"

#: Card §1. (incumbent colour, t1j colour); T1j's colour is the harness's anchor.
ARMS = {"A": ("red", "black"), "B": ("black", "red")}
EMPTY_BOARD = "empty_board"          # the state factory's key -- not an opening set

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
EVIDENCE_ROOT = REPO_ROOT / "docs" / "superpowers" / "evidence"
#: Card §3.1 (amended after step 3): the header hashes FOUR cards -- the analysis
#: card binds a results file to a manifest carrying all four -- and the FIVE
#: Python modules that play. The analysis reads both tuples from here.
CARDS = ("docs/superpowers/2026-09-22-t1j-h4-runner-persistence-card.md",
         "docs/superpowers/2026-09-22-t1j-h4-4b-acceptance-qualification-card.md",
         "docs/superpowers/2026-09-21-t1j-h4-replacement-card.md",
         "docs/superpowers/2026-09-23-t1j-h4-analysis-card.md",
         "docs/superpowers/2026-09-24-t1j-h4-step4-seed-card.md")
#: 🔴 Card §12.6.1: EVERY first-party file reachable by import at any depth from
#: this runner and the 3P-b qualification, package `__init__`s included, minus the
#: two gate modules. Explicit and reviewed; a test recomputes the closure from
#: source and requires equality with this tuple AND with the card's list.
CODE = (
    "scripts/GPU/__init__.py",
    "scripts/GPU/alphazero/__init__.py",
    "scripts/GPU/alphazero/d0_postmortem.py",
    "scripts/GPU/alphazero/d1_probe.py",
    "scripts/GPU/alphazero/d1_selection.py",
    "scripts/GPU/alphazero/e4_screen_command.py",
    "scripts/GPU/alphazero/e4_screen_integration.py",
    "scripts/GPU/alphazero/e4_screen_reference.py",
    "scripts/GPU/alphazero/e4_screen_rules.py",
    "scripts/GPU/alphazero/e4_screen_runner.py",
    "scripts/GPU/alphazero/eval_integrity.py",
    "scripts/GPU/alphazero/eval_readout.py",
    "scripts/GPU/alphazero/eval_replay.py",
    "scripts/GPU/alphazero/eval_runner.py",
    "scripts/GPU/alphazero/evaluator.py",
    "scripts/GPU/alphazero/fpu_state_hash.py",
    "scripts/GPU/alphazero/game/__init__.py",
    "scripts/GPU/alphazero/game/twixt_state.py",
    "scripts/GPU/alphazero/h2_match_plan.py",
    "scripts/GPU/alphazero/h2_match_rules.py",
    "scripts/GPU/alphazero/h2_match_runner.py",
    "scripts/GPU/alphazero/h4_4a_characterization.py",
    "scripts/GPU/alphazero/h4_production_qualification.py",
    "scripts/GPU/alphazero/h4_repair_qualification.py",
    "scripts/GPU/alphazero/h4_runner.py",
    "scripts/GPU/alphazero/l0_match_plan.py",
    "scripts/GPU/alphazero/l0_match_rules.py",
    "scripts/GPU/alphazero/local_evaluator.py",
    "scripts/GPU/alphazero/mcts.py",
    "scripts/GPU/alphazero/network.py",
    "scripts/GPU/alphazero/opening_diagnostics.py",
    "scripts/GPU/alphazero/probe_eval.py",
    "scripts/GPU/alphazero/t1j_adapter.py",
    "scripts/GPU/alphazero/t1j_toolchain.py",
    "scripts/GPU/alphazero/twixtbot_g3_reference.py",
    "scripts/GPU/alphazero/twixtbot_g3_schedule.py",
    "scripts/GPU/alphazero/void_trace.py",
)
#: Card §12.6.2: the ONLY files on the path left out of `code`.
GATE_MODULES = ("scripts/GPU/alphazero/h4_pilot_authorization.py",
                "scripts/GPU/alphazero/h4_production_qualification_authorization.py")

#: Analysis card §0.1: the header fields a manifest entry binds.
MANIFEST_BOUND = ("schedule_digest", "seeds", "incumbent_identity", "t1j_runtime", "cards",
                  "code")
#: Card §12.1 item 5, §12.6.3: the compile identity, split EXHAUSTIVELY into what
#: ran (bound) and where it ran (recorded in `t1j_local`, never bound).
TOOLCHAIN_CONTENT = ("jar_sha256", "jdk_components", "sources", "classes", "main_class")
TOOLCHAIN_LOCAL = ("toolchain", "jar", "jdk_home", "classes_dir")
#: Replacement card §5.4: the per-segment hard deadline.
SEGMENT_DEADLINE_S = 14_400

#: 🔴 Step-4 card §2 (BOUND): the five registered blocks, one seed per game.
#: Registered in `e4_screen_reference.ACCOUNTED_SEED_INTERVALS` after collision
#: proof v16 was CLEAN (evidence/2026-09-24-t1j-h4-seed-registration/).
SEED_BLOCKS = {("pilot", 0): (202_632_000, 202_632_032),
               ("study", 0): (202_634_000, 202_634_148),
               ("study", 1): (202_636_000, 202_636_148),
               ("study", 2): (202_638_000, 202_638_148),
               ("study", 3): (202_640_000, 202_640_148)}
#: Step-4 card §2.1 / amendment 3: FIXED, repo-relative, create-only; the one-shot
#: use record of each run until H4 closes.
EVIDENCE_DIRS = {("pilot", 0): "docs/superpowers/evidence/t1j-h4-pilot",
                 **{("study", k): f"docs/superpowers/evidence/t1j-h4-study-segment{k}"
                    for k in range(4)}}

#: Card §3.4: the §6 categories, and nothing else.
CLASSIFICATIONS = ("timeout", "adapter_refusal", "unreadable_output", "replay_mismatch",
                   "coherence_mismatch", "missing_record", "digest_failure",
                   "process_count", "illegal_move", "leaked_process", "unexpected")

#: Card §7: fields that may never reach stdout, stderr or the progress trace.
BLIND_FIELDS = ("winner", "points", "t1j_points", "incumbent_points", "score",
                "pair_score")


class H4RunError(Exception):
    """Refused before anything ran. Never a statement about either engine."""


class H4RunVoid(H4RunError):
    """An OPERATIONAL FAILURE (card §6). The run is VOID."""

    def __init__(self, classification: str, message: str):
        if classification not in CLASSIFICATIONS:
            raise ValueError(f"unknown classification {classification!r}")
        super().__init__(f"[{classification}] {message}")
        self.classification = classification
        self.message = message


class H4IllegalMove(AbortError):
    """An agent returned an illegal move. SUBCLASSES AbortError, so the harness
    passes it through with its TYPE intact rather than flattening it."""


class H4IncumbentError(AbortError):
    """The incumbent raised. Kept typed for the same reason."""


# ─────────────────────────────── the schedule (§1) ───────────────────────────────

def make_schedule(pairs: Sequence[Tuple[str, int, int]], *, mode: str,
                  reference: Optional[Mapping[str, str]] = None) -> List[Dict[str, Any]]:
    """`(pair_id, seed_A, seed_B)` -> adjacent tasks, Arm A then Arm B."""
    tasks: List[Dict[str, Any]] = []
    for pair_id, seed_a, seed_b in pairs:
        for arm, seed in (("A", seed_a), ("B", seed_b)):
            inc, t1j = ARMS[arm]
            task = {"task_id": f"{pair_id}-{arm}", "pair_id": pair_id, "arm": arm,
                    "seed": seed, "opening": EMPTY_BOARD, "incumbent_colour": inc,
                    "t1j_colour": t1j, "anchor_colour": t1j, "reference_colour": inc,
                    "t1j_mdPly": DEPTH}
            if reference is not None:
                task.update(reference=reference["reference"],
                            reference_sha1=reference["reference_sha1"])
            tasks.append(task)
    check_schedule(tasks, mode=mode)
    return tasks


def check_schedule(tasks: Sequence[Mapping[str, Any]], *, mode: str) -> None:
    """Pairs adjacent A-then-B, unique ids and seeds, and the seed SIGN of the mode."""
    if mode not in DESIGNS:
        raise H4RunError(f"unknown mode {mode!r}; one of {sorted(DESIGNS)}")
    if not tasks or len(tasks) % 2:
        raise H4RunError(f"{len(tasks)} tasks: a schedule is whole pairs")
    for i in range(0, len(tasks), 2):
        a, b = tasks[i], tasks[i + 1]
        if (a.get("arm"), b.get("arm")) != ("A", "B") or a.get("pair_id") != b.get("pair_id"):
            raise H4RunError(f"tasks {i},{i + 1} are not one adjacent A-then-B pair")
    ids = [t["task_id"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise H4RunError("a task_id repeats: that is an integrity fault, not a replicate")
    seeds = [t["seed"] for t in tasks]
    if len(set(seeds)) != len(seeds):
        raise H4RunError("a seed repeats: that is an integrity fault, not a replicate")
    for s in seeds:
        if type(s) is not int:
            raise H4RunError(f"seed {s!r} is not an int")
        if mode == "fixture" and s >= 0:
            raise H4RunError(f"fixture mode refuses the non-negative seed {s}: research "
                             f"seeds are positive, synthetic ones negative (card §10.1)")
        if mode != "fixture" and s < 0:
            raise H4RunError(f"{mode} mode refuses the negative seed {s}: that is a "
                             f"SYNTHETIC fixture seed (card §10.1)")


def schedule_digest(tasks: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(list(tasks), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _inside(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def check_destinations(out_dir: str, classes_dir: str, *, mode: str) -> None:
    out, cls = pathlib.Path(out_dir), pathlib.Path(classes_dir)
    if mode == "fixture" and _inside(out, EVIDENCE_ROOT):
        raise H4RunError(f"fixture mode refuses a destination under the evidence tree: "
                         f"{out_dir} (card §10.1)")
    if _inside(cls, REPO_ROOT):
        raise H4RunError(f"the class directory must be OUTSIDE the repository: "
                         f"{classes_dir} (card §2)")


# ─────────────────────── the canonical results file (§3) ───────────────────────

class Results:
    """Create-only JSONL, fsynced record by record: a kill leaves a truthful prefix."""

    def __init__(self, path: str):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        self._f = os.fdopen(fd, "w", encoding="utf-8")
        self.path = path

    def emit(self, record: Mapping[str, Any]) -> None:
        self._f.write(json.dumps(record, sort_keys=True) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self) -> None:
        self._f.close()


def read_records(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def completed_counts(path: str) -> Dict[str, int]:
    """Card §3.4: RECOMPUTED from the records already written, never tallied aside."""
    recs = read_records(path)
    kinds = [r.get("record_type") for r in recs]
    return {"games": kinds.count("game_result"), "ply": kinds.count("ply"),
            "process": kinds.count("process"), "game_result": kinds.count("game_result")}


def blinding_violations(text: str) -> List[str]:
    """Card §7: the blind fields found in console or trace text."""
    return [f for f in BLIND_FIELDS if f'"{f}"' in text or f"{f}=" in text]


def transcript_digest_of(plies: Sequence[Mapping[str, Any]], result: Mapping[str, Any]) -> str:
    """Card §3.2: the frozen H2 definition, applied to the WHOLE game (opening_bound 0)."""
    t = H2R.transcript([{"ply": p["ply"], "mover": p["mover"], "move": p["move"]}
                        for p in plies],
                       {"plies": result["plies"], "terminal_reason": result["terminal_reason"],
                        "winner": result["winner"]}, opening_bound=0)
    return H2R.transcript_digest(t)


# ─────────────────────── the header and its binding (§3.1, §12) ───────────────────────

def split_toolchain(identity: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(content, local). The compile identity must have EXACTLY these fields."""
    want = set(TOOLCHAIN_CONTENT) | set(TOOLCHAIN_LOCAL)
    if set(identity) != want:
        raise H4RunError(f"the compile identity's fields {sorted(identity)} are not exactly "
                         f"the classified {sorted(want)} (card §12.6.3)")
    return ({k: identity[k] for k in TOOLCHAIN_CONTENT},
            {k: identity[k] for k in TOOLCHAIN_LOCAL})


def bound_identity(incumbent_identity: Mapping[str, Any],
                   toolchain_content: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The bound fields that do not depend on the schedule. `code` and `cards` are
    hashed from the files NOW -- the state that is about to play."""
    return {"incumbent_identity": dict(incumbent_identity),
            "t1j_runtime": {"toolchain": (dict(toolchain_content)
                                          if toolchain_content is not None else None),
                            "depth": DEPTH, "query_timeout_s": QUERY_TIMEOUT_S,
                            "replay_timeout_s": REPLAY_TIMEOUT_S, "ply_cap": PLY_CAP,
                            "h4_acceptance": True},
            "cards": {c: _sha256_or_none(REPO_ROOT / c) for c in CARDS},
            "code": {m: _sha256_or_none(REPO_ROOT / m) for m in CODE}}


def header_candidate(*, mode: str, segment: int, schedule: Sequence[Mapping[str, Any]],
                     incumbent_identity: Mapping[str, Any],
                     toolchain_content: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return {"record_type": "header", "design": DESIGNS[mode], "evidence": mode != "fixture",
            "segment": segment, "schedule_digest": schedule_digest(schedule),
            "seeds": [t["seed"] for t in schedule],
            **bound_identity(incumbent_identity, toolchain_content)}


def _canon(v: Any) -> str:
    """Type-strict comparison form: 1 and 1.0, True and 1 differ."""
    return json.dumps(v, sort_keys=True, separators=(",", ":"))


def binding_differences(header: Mapping[str, Any], entry: Mapping[str, Any], *,
                        stage: str, with_toolchain: bool) -> List[str]:
    """The fields in which the header does not equal its manifest entry."""
    bad = [f for f, want in (("design", stage), ("segment", entry.get("segment")))
           if header.get(f) != want]
    for f in MANIFEST_BOUND:
        got, want = header.get(f), entry.get(f)
        if f == "t1j_runtime" and not with_toolchain:
            got = {k: v for k, v in (got or {}).items() if k != "toolchain"}
            want = {k: v for k, v in want.items() if k != "toolchain"} \
                if isinstance(want, dict) else want
        if _canon(got) != _canon(want):
            bad.append(f)
    return bad


def canonical_schedule(mode: str, segment: int,
                       reference: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Step-4 card §2: the ONLY schedule a pilot or study entry may carry -- pair k
    of the block gets lo+2k (Arm A) and lo+2k+1 (Arm B)."""
    if (mode, segment) not in SEED_BLOCKS:
        raise H4RunError(f"no registered block for {mode} segment {segment}")
    lo, hi = SEED_BLOCKS[(mode, segment)]
    tag = "h4p" if mode == "pilot" else f"h4s{segment}"
    pairs = [(f"{tag}-{k:02d}", lo + 2 * k, lo + 2 * k + 1) for k in range((hi - lo) // 2)]
    return make_schedule(pairs, mode=mode, reference=reference)


def seed_registry_faults(schedule: Sequence[Mapping[str, Any]]) -> List[str]:
    """Every seed ACCOUNTED, and none EXPOSED, RETIRED, TEST_ONLY or CONSUMED --
    read from the shared registry at call time (step-4 card §3, 4b)."""
    from . import e4_screen_reference as REF
    bad = []
    for t in schedule:
        st = REF.seed_status(t["seed"])
        why = ([] if st["accounted"] else ["not accounted"]) + \
              [k for k in ("exposed", "retired", "test_only") if st[k]] + \
              (["consumed"] if t["seed"] in REF.CONSUMED_SEEDS else [])
        if why:
            bad.append(f"{t['seed']}: {', '.join(why)}")
    return bad


def claim_directory(path: str) -> None:
    """Step-4 card amendment 1: the ONE-SHOT use record. Create-only (`os.mkdir`,
    never `exist_ok`), parent fsynced; an occupied directory refuses, whatever left
    it there -- a VOID included."""
    try:
        os.mkdir(path)
    except FileExistsError:
        raise H4RunError(f"{path} is occupied: a run's evidence directory is claimed "
                         f"once, and an occupied one -- a VOID's included -- refuses") from None
    fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def manifest_entry(manifest: str, *, mode: str, segment: int) -> Dict[str, Any]:
    try:
        with open(manifest, encoding="utf-8") as fh:
            m = json.load(fh)
    except (OSError, ValueError) as e:
        raise H4RunError(f"manifest {manifest} is unreadable: {e}") from None
    if not isinstance(m, dict) or m.get("stage") != DESIGNS[mode]:
        raise H4RunError(f"{manifest} is not a {DESIGNS[mode]} manifest")
    entries = m.get("segments")
    if not isinstance(entries, list) or not 0 <= segment < len(entries) \
            or entries[segment].get("segment") != segment:
        raise H4RunError(f"{manifest} has no entry for segment {segment}")
    return entries[segment]


# ───────────────────── the production incumbent (§2, §12.1) ─────────────────────

def frozen_argmax_config():
    """THE ONE construction of the configuration the incumbent plays under, from
    the upstream sources -- the qualified reference path's eval config with H2's
    selection mode. The seam carries THIS object; the identity is read off it."""
    from . import twixtbot_g3_reference as G3
    cfg = G3.eval_config()
    return cfg.__class__(**{**cfg.__dict__, "selection_mode": H2R.SELECTION_MODE})


def frozen_incumbent_identity(config, *, design: str) -> Dict[str, Any]:
    """H2's frozen argmax identity, read off THE OBJECT THAT WILL PLAY."""
    from . import e4_screen_reference as REF
    from . import h2_match_runner as H2RUN
    ident = dict(H2RUN.frozen_incumbent_identity())
    ident["argmax_config"] = {f: getattr(config, f)
                              for f in REF.frozen_settings()["eval_config"]}
    ident["design"] = design
    return ident


def check_incumbent_identity(identity: Mapping[str, Any], config, *, design: str) -> None:
    """The object against the qualified path, `selection_mode`, then the whole
    recorded identity against the frozen one -- type-strictly."""
    from . import h2_match_runner as H2RUN
    want = frozen_incumbent_identity(config, design=design)
    try:
        H2RUN._same(want["argmax_config"],
                    {**want["eval_config"], **want["inert_under_argmax"]}, "argmax_config")
    except H2RUN.H2VoidError as e:
        raise H4RunError(f"the configuration that would play disagrees with the "
                         f"qualified path's frozen settings: {e}") from None
    if (identity.get("eval_config") or {}).get("selection_mode") != H2R.SELECTION_MODE:
        raise H4RunError(f"the incumbent identity's selection_mode is not "
                         f"{H2R.SELECTION_MODE!r}")
    try:
        H2RUN._same(dict(identity), want, "incumbent_identity")
    except H2RUN.H2VoidError as e:
        raise H4RunError(str(e)) from None


def load_production_evaluator(identity: Mapping[str, Any]):
    """The qualified loader (it recomputes the checkpoint's sha1 against its pin),
    then the loaded object's tags against THIS identity."""
    from . import e4_screen_command as SCREEN_CMD
    ev = SCREEN_CMD._default_load_evaluator(str(REPO_ROOT))
    got = (getattr(ev, "_g3_reference", None), getattr(ev, "_g3_sha1", None))
    if got != (identity["reference"], identity["reference_sha1"]):
        raise H4RunError(f"the loaded evaluator is {got}, not the identity's "
                         f"({identity['reference']}, {identity['reference_sha1']})")
    return ev


def production_incumbent_build(config) -> Callable:
    """Builds the incumbent for a task from THE config object -- never a fresh one."""
    from . import e4_screen_reference as REF
    from . import twixtbot_g3_reference as G3

    def build(task, evaluator=None):
        return G3.build_reference_agent(task=task, evaluator=evaluator,
                                        colour=REF.reference_colour(task), config=config)
    build.config = config
    return build


def _verified_t1j_paths(classes: str) -> T1jPaths:
    """The VERIFIED toolchain's java and jar; the class directory is an output."""
    from . import t1j_toolchain as TC
    tc = TC.verified_paths()
    return T1jPaths(java=os.path.join(tc["jdk_home"], "bin", "java"), jar=tc["jar"],
                    classes=classes, ply_cap=PLY_CAP)


# ───────────────────────── classification (§3.4, §6) ─────────────────────────

def classify(exc: BaseException) -> str:
    """By TYPE, never by message."""
    if isinstance(exc, H4RunVoid):
        return exc.classification
    if isinstance(exc, subprocess.TimeoutExpired):
        return "timeout"
    if isinstance(exc, D1VoidError):
        return "timeout"                         # the whole-run SIGALRM deadline
    if isinstance(exc, H4IllegalMove):
        return "illegal_move"
    if isinstance(exc, H4IncumbentError):
        return "unexpected"
    if isinstance(exc, AbortError):
        return "replay_mismatch" if exc.phase == PHASE_BIND else "adapter_refusal"
    if isinstance(exc, (ValueError, KeyError)):   # incl. HelperOutputError, unconverted
        return "unreadable_output"
    return "unexpected"


def _stdout_of(exc: BaseException) -> Optional[str]:
    for e in (exc, exc.__cause__):
        out = getattr(e, "stdout", None)
        if isinstance(out, str):
            return out
    return None


# ──────────────────────────────── one game ────────────────────────────────

class _Stash:
    """The ORIGINAL exception, kept before the harness flattens it."""

    def __init__(self):
        self.exc: Optional[BaseException] = None

    def keep(self, e: BaseException) -> None:
        if self.exc is None:
            self.exc = e


class _Live:
    """The recorder `play_task` writes to. Streams each ply's process records, then
    the enriched ply record -- durable as the game goes, not after it."""

    def __init__(self, results: Results, ctx, task, notes: List[Dict[str, Any]],
                 written: Dict[str, List[Dict[str, Any]]]):
        self.results, self.ctx, self.task, self.notes = results, ctx, task, notes
        self.written = written
        self._flushed = 0

    def flush_processes(self) -> None:
        bucket = self.ctx.processes.get(self.task["task_id"], [])
        for obs in bucket[self._flushed:]:
            rec = {"record_type": "process", "task_id": self.task["task_id"],
                   "pair_id": self.task["pair_id"], "arm": self.task["arm"], **obs}
            self.results.emit(rec)
            self.written["process"].append(rec)
        self._flushed = len(bucket)

    def emit(self, record: Mapping[str, Any]) -> None:
        self.flush_processes()
        if record.get("record_type") != "ply":
            return                                # opening_bound: its replay is recorded
        note = self.notes.pop(0)
        if [note["mover"], note["move"]] != [record["mover"], list(record["move"])]:
            raise H4RunVoid("missing_record", f"ply {record['ply']}: the harness applied "
                            f"{record['move']} but the agent returned {note['move']}")
        rec = {"record_type": "ply", "task_id": self.task["task_id"],
               "pair_id": self.task["pair_id"], "arm": self.task["arm"],
               "ply": record["ply"], "mover": record["mover"], "move": list(record["move"]),
               **{k: v for k, v in note.items() if k not in ("mover", "move")}}
        self.results.emit(rec)
        self.written["ply"].append(rec)


def _wrap_agents(base_factory: Callable, ctx, task, notes, stash: _Stash,
                 now: Callable[[], float]) -> Callable:
    def agent_for(t, mover):
        inner = base_factory(t, mover)
        actor = "t1j" if mover == t["t1j_colour"] else "incumbent"

        def call(state):
            t0 = now()
            try:
                mv = inner(state)
            except AbortError as e:
                stash.keep(e)
                raise
            except BaseException as e:            # noqa: BLE001 -- kept, re-raised typed
                stash.keep(e)
                if actor == "incumbent":
                    raise H4IncumbentError(PHASE_MOVE, f"incumbent raised {e!r}") from e
                raise
            elapsed = now() - t0
            mv = tuple(mv) if mv is not None else None
            if mv is None or mv not in set(state.legal_moves()):
                err = H4IllegalMove(PHASE_MOVE, f"{actor} returned {mv}, not legal")
                stash.keep(err)
                raise err
            note = {"mover": mover, "actor": actor, "move": list(mv), "elapsed_s": elapsed}
            if actor == "t1j":
                q = [o for o in ctx.processes.get(t["task_id"], []) if o["role"] == "query"][-1]
                note.update(source=q["source"],
                            t1j_elapsed_us=(q["telemetry"] or {}).get("elapsed_us"),
                            query_ordinal=q["ordinal"])
            notes.append(note)
            return mv
        return call
    return agent_for


def _wrap_binder(binder: Callable, stash: _Stash) -> Callable:
    def bind(task, state, ply, move=None):
        try:
            return binder(task, state, ply, move)
        except BaseException as e:                # noqa: BLE001 -- kept, re-raised
            stash.keep(e)
            raise
    return bind


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                               # something holds that pid
    return True


def check_game(task: Mapping[str, Any], plies: Sequence[Mapping[str, Any]],
               procs: Sequence[Mapping[str, Any]], result: Mapping[str, Any], *,
               check_pids: bool = True) -> None:
    """Card §2.1 and §6, asserted FROM THE RECORDS. The reader re-applies it
    without the pid-liveness check, which only means something during the run."""
    if result["terminal_reason"] not in ("win", "cap"):
        raise H4RunVoid("unexpected", f"terminal reason {result['terminal_reason']!r}")
    if [p["ply"] for p in plies] != list(range(1, result["plies"] + 1)):
        raise H4RunVoid("missing_record", "the ply records are not contiguous 1..plies")
    for r in list(plies) + list(procs):
        if (r.get("task_id"), r.get("pair_id"), r.get("arm")) != (
                task["task_id"], task["pair_id"], task["arm"]):
            raise H4RunVoid("missing_record", f"a record carries another game's identity: "
                            f"{r.get('task_id')}/{r.get('pair_id')}/{r.get('arm')}")
    t1j_moves = sum(1 for p in plies if p["actor"] == "t1j")
    queries = [o for o in procs if o["role"] == "query"]
    replays = [o for o in procs if o["role"] == "replay"]
    pids = [o["proc"]["pid"] for o in procs if o.get("proc")]
    for ok, msg in ((len(queries) == t1j_moves,
                     f"{len(queries)} query records for {t1j_moves} T1j moves"),
                    (len(replays) == result["plies"] + 1,
                     f"{len(replays)} replay records for {result['plies']} plies"),
                    (all(o["outcome"] == "accepted" for o in procs),
                     "a process record is not accepted"),
                    (len(pids) == len(procs) == len(set(pids)),
                     f"{len(set(pids))} distinct pids for {len(procs)} process records")):
        if not ok:
            raise H4RunVoid("process_count", msg)
    for o in procs:
        if o["t1j_position_digest"] != o["expected_position_digest"]:
            raise H4RunVoid("coherence_mismatch",
                            f"{o['role']} ordinal {o['ordinal']}: T1j's position digest "
                            f"differs from ours")
    alive = [p for p in pids if check_pids and _pid_alive(p)]
    if alive:
        raise H4RunVoid("leaked_process", f"recorded pids still alive: {alive}")


def play_game(*, task, ctx, binder, base_factory, state_factory, results: Results,
              written, stash: _Stash, now) -> Dict[str, Any]:
    """ONE game through the real harness. Returns the game_result record."""
    start = {"record_type": "game_start", "task_id": task["task_id"],
             "pair_id": task["pair_id"], "arm": task["arm"],
             "incumbent_colour": task["incumbent_colour"], "t1j_colour": task["t1j_colour"],
             "seed": task["seed"], "game_index": task["game_index"]}
    results.emit(start)
    n_ply0, n_proc0 = len(written["ply"]), len(written["process"])
    notes: List[Dict[str, Any]] = []
    live = _Live(results, ctx, task, notes, written)
    t0 = now()
    try:
        out = HARNESS.play_task(task=dict(task),
                                agent_for=_wrap_agents(base_factory, ctx, task, notes,
                                                       stash, now),
                                state_factory=state_factory,
                                binder=_wrap_binder(binder, stash), rec=live,
                                ply_cap=PLY_CAP)
    finally:
        live.flush_processes()                    # a refused call's record survives
    elapsed = now() - t0
    plies = written["ply"][n_ply0:]
    procs = written["process"][n_proc0:]
    t1j_pts = out["t1j_points"]
    result = {"record_type": "game_result", "task_id": task["task_id"],
              "pair_id": task["pair_id"], "arm": task["arm"], "winner": out["winner"],
              "terminal_reason": out["terminal_reason"], "plies": out["plies"],
              "t1j_points": t1j_pts, "incumbent_points": 1.0 - t1j_pts,
              "elapsed_s": elapsed,
              "queries": sum(1 for o in procs if o["role"] == "query"),
              "replays": sum(1 for o in procs if o["role"] == "replay"),
              "distinct_pids": len({o["proc"]["pid"] for o in procs if o.get("proc")})}
    try:
        result["transcript_digest"] = transcript_digest_of(plies, result)
    except H2R.H2RulesError as e:
        raise H4RunVoid("missing_record", f"the transcript would not form: {e}") from None
    check_game(task, plies, procs, result)
    results.emit(result)
    written["game_result"].append(result)
    return result


# ─────────────────────────────── the run (§3, §6) ───────────────────────────────

def _trace(fh, obj: Mapping[str, Any]) -> None:
    fh.write(json.dumps(obj, sort_keys=True) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def run_games(*, mode: str, out_dir: str, deadline_s: float,
              paths: Optional[T1jPaths] = None, classes: Optional[str] = None,
              schedule: Optional[Sequence[Mapping[str, Any]]] = None,
              manifest: Optional[str] = None,
              incumbent_build: Optional[Callable] = None, evaluator: Any = None,
              incumbent_identity: Optional[Mapping[str, Any]] = None, segment: int = 0,
              _compile: Optional[Callable] = None,
              _now: Callable[[], float] = time.monotonic) -> Dict[str, Any]:
    """PUBLIC ENTRY. Refuses while the gate is shut, in EVERY mode."""
    if not AUTH.H4_PILOT_EXECUTION_AUTHORIZED:
        raise H4RunError("the H4 pilot is UNAUTHORIZED. Nothing has been built, "
                         "compiled, played or written.")
    if mode not in DESIGNS:
        raise H4RunError(f"unknown mode {mode!r}; one of {sorted(DESIGNS)}")
    if mode != "fixture":
        return _production(mode=mode, out_dir=out_dir, classes=classes,
                           deadline_s=deadline_s, manifest=manifest, segment=segment,
                           _now=_now,
                           injected={"paths": paths, "schedule": schedule,
                                     "incumbent_build": incumbent_build,
                                     "evaluator": evaluator,
                                     "incumbent_identity": incumbent_identity,
                                     "_compile": _compile})
    if manifest is not None or classes is not None or paths is None:
        raise H4RunError("fixture mode takes stub paths and binds to no manifest")
    check_schedule(schedule, mode=mode)
    check_destinations(out_dir, paths.classes, mode=mode)
    if incumbent_build is None or incumbent_identity is None:
        raise H4RunError("fixture mode needs the STUB incumbent and its identity")
    if _compile is None:
        raise H4RunError("fixture mode never compiles for real; pass the stub compile")
    return _run_unguarded(mode=mode, schedule=schedule, out_dir=out_dir, paths=paths,
                          deadline_s=deadline_s, incumbent_build=incumbent_build,
                          evaluator=evaluator, incumbent_identity=incumbent_identity,
                          segment=segment, _compile=_compile, _now=_now)


def _production(*, mode, out_dir, classes, deadline_s, manifest, segment, _now, injected):
    """Card §12.1: pilot and study mode. NO injected seam -- not even the toolchain
    paths, which are resolved HERE from the verified toolchain; the schedule comes
    from the committed manifest; every bound field that needs no compile is checked
    BEFORE any destination is claimed, the toolchain resolved or the checkpoint
    loaded."""
    seams = sorted(k for k, v in injected.items() if v is not None)
    if seams:
        raise H4RunError(f"{mode} mode accepts no injected seam, got {seams}: an entry "
                         f"that accepts its own seams is not a gate (card §12.1)")
    if manifest is None or classes is None:
        raise H4RunError(f"{mode} mode plays only a committed manifest's schedule, "
                         f"into a named class directory")
    entry = manifest_entry(manifest, mode=mode, segment=segment)
    schedule = entry.get("schedule")
    check_schedule(schedule, mode=mode)
    config = frozen_argmax_config()
    identity = frozen_incumbent_identity(config, design=DESIGNS[mode])
    check_incumbent_identity(identity, config, design=DESIGNS[mode])
    wrong_ref = [t["task_id"] for t in schedule
                 if (t.get("reference"), t.get("reference_sha1"))
                 != (identity["reference"], identity["reference_sha1"])]
    if wrong_ref:
        raise H4RunError(f"tasks {wrong_ref[:3]} name another incumbent reference")
    if _canon(list(schedule)) != _canon(canonical_schedule(mode, segment, identity)):
        raise H4RunError(f"the manifest's schedule is not the CANONICAL {mode} segment "
                         f"{segment} schedule of the step-4 card (block, pair ids, arm "
                         f"seeds, reference)")
    faults = seed_registry_faults(schedule)
    if faults:
        raise H4RunError(f"seeds fail the shared registry: {faults[:3]}")
    want_dir = EVIDENCE_DIRS[(mode, segment)]
    if entry.get("evidence_dir") != want_dir or os.path.realpath(out_dir) != \
            os.path.realpath(os.path.join(REPO_ROOT, want_dir)):
        raise H4RunError(f"{mode} segment {segment} writes ONLY to its manifest's "
                         f"evidence_dir {want_dir!r}, not {out_dir!r}")
    if os.path.lexists(out_dir):
        raise H4RunError(f"{out_dir} is occupied: this entry has already been run, or "
                         f"a VOID left it -- a run is one-shot (step-4 card §2.1)")
    head = header_candidate(mode=mode, segment=segment, schedule=schedule,
                            incumbent_identity=identity, toolchain_content=None)
    bad = binding_differences(head, entry, stage=DESIGNS[mode], with_toolchain=False)
    if bad:
        raise H4RunError(f"the run would not bind to its manifest: {bad} differ. Nothing "
                         f"was claimed, loaded, compiled or played.")
    paths = _verified_t1j_paths(classes)
    check_destinations(out_dir, paths.classes, mode=mode)
    evaluator = load_production_evaluator(identity)
    return _run_unguarded(mode=mode, schedule=schedule, out_dir=out_dir, paths=paths,
                          deadline_s=deadline_s,
                          incumbent_build=production_incumbent_build(config),
                          evaluator=evaluator, incumbent_identity=identity,
                          segment=segment, _compile=None, _now=_now, entry=entry)


def _run_unguarded(*, mode, schedule, out_dir, paths, deadline_s, incumbent_build,
                   evaluator, incumbent_identity, segment, _compile, _now, entry=None):
    t_start = _now()
    claim_directory(out_dir)                      # THE claim: before any draw or JVM
    results = Results(os.path.join(out_dir, "results.jsonl"))
    trace = os.fdopen(os.open(os.path.join(out_dir, "trace.jsonl"),
                              os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "w")
    written: Dict[str, List[Dict[str, Any]]] = {"ply": [], "process": [], "game_result": []}
    stash = _Stash()
    where: Dict[str, Any] = {"stage": "setup", "task": None}
    deadline = Deadline(limit_s=deadline_s)
    try:
        ctx = INT.IntegrationContext()
        runtime = INT.T1jRuntime(java=paths.java, jar=paths.jar, classes=paths.classes,
                                 ply_cap=PLY_CAP, timeout_s=REPLAY_TIMEOUT_S,
                                 h4_acceptance=True)
        binder = INT.make_binder(runtime, ctx)
        base_factory = INT.make_agent_factory(runtime=runtime, ctx=ctx, evaluator=evaluator,
                                              reference_build=incumbent_build,
                                              t1j_timeout_s=QUERY_TIMEOUT_S)
        state_factory = INT.make_state_factory({EMPTY_BOARD: []}, ctx)
        deadline.start()
        with _supervisor(deadline):
            where["stage"] = "compile"
            compile_fn = _compile or (lambda d: _compile_helper_verified(d, paths=paths))
            content, local = split_toolchain(compile_fn(deadline))
            header = header_candidate(mode=mode, segment=segment, schedule=schedule,
                                      incumbent_identity=incumbent_identity,
                                      toolchain_content=content)
            if entry is not None:
                bad = binding_differences(header, entry, stage=DESIGNS[mode],
                                          with_toolchain=True)
                if bad:
                    raise H4RunVoid("unexpected", f"after compiling, the header does not "
                                    f"bind to its manifest: {bad} differ. No game was "
                                    f"played (card §12.6.3)")
            results.emit({**header, "t1j_local": local})
            setup_s = _now() - t_start
            _trace(trace, {"event": "run_start", "n_games": len(schedule),
                           "design": DESIGNS[mode]})
            for i, task in enumerate(schedule):
                where.update(stage="game", task=task)
                deadline.check(f"before game {i}")
                _trace(trace, {"event": "game_start", "index": i,
                               "task_id": task["task_id"]})
                res = play_game(task={**task, "game_index": i}, ctx=ctx, binder=binder,
                                base_factory=base_factory, state_factory=state_factory,
                                results=results, written=written, stash=stash, now=_now)
                _trace(trace, {"event": "game_done", "index": i,
                               "terminal_reason": res["terminal_reason"],
                               "plies": res["plies"], "elapsed_s": res["elapsed_s"]})
            where.update(stage="post_game_check", task=None)
            end = {"record_type": "segment_end", "games_completed": len(schedule),
                   "complete": True, "setup_s": setup_s, "total_s": _now() - t_start}
            results.emit(end)
            _trace(trace, {"event": "run_end", "games_completed": len(schedule)})
            return {"games_completed": len(schedule), "results": results.path,
                    "complete": True}
    except BaseException as raised:              # noqa: BLE001 -- recorded, re-raised
        original = stash.exc or raised
        if isinstance(raised, H4RunVoid):
            original = raised
        stage = where["stage"]
        if isinstance(original, D1VoidError):
            stage = "deadline"
        task = where["task"] or {}
        void = {"record_type": "run_void", "stage": stage,
                "task_id": task.get("task_id"), "pair_id": task.get("pair_id"),
                "arm": task.get("arm"),
                "ply": (written["ply"][-1]["ply"] if task and written["ply"]
                        and written["ply"][-1]["task_id"] == task.get("task_id") else None),
                "classification": classify(original),
                "exception": type(original).__name__, "reason": str(original),
                "stdout": _stdout_of(original),
                "completed": completed_counts(results.path),
                "elapsed_s": _now() - t_start}
        try:
            results.emit(void)
            _trace(trace, {"event": "run_void", "classification": void["classification"]})
        except Exception:                          # noqa: BLE001 -- never mask the cause
            pass
        if isinstance(raised, (KeyboardInterrupt, SystemExit)):
            raise
        raise H4RunVoid(void["classification"], str(original)) from original
    finally:
        results.close()
        trace.close()


def _sha256_or_none(p: pathlib.Path) -> Optional[str]:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


# ───────────────────────── reading and exporting (§3, §4, §8) ─────────────────────────

def load_games(path: str, *, allow_fixture: bool = False) -> List[Dict[str, Any]]:
    """Every complete game, RE-DERIVED: transcript digests recomputed, expected
    position digests recomputed from the moves alone. Refuses rather than guesses."""
    from .game.twixt_state import TwixtState
    recs = read_records(path)
    if not recs or recs[0].get("record_type") != "header":
        raise H4RunError(f"{path}: the first record is not a header")
    header = recs[0]
    if header.get("design") == FIXTURE_DESIGN and not allow_fixture:
        raise H4RunError(f"{path}: {FIXTURE_NOTE} -- fixture data is never a result")
    last = recs[-1].get("record_type")
    if last == "run_void":
        raise H4RunError(f"{path}: the run is VOID; its partial games are not results")
    if last != "segment_end":
        raise H4RunError(f"{path}: ends in neither segment_end nor run_void (killed)")
    games: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for r in recs[1:-1]:
        tid = r.get("task_id")
        g = games.setdefault(tid, {"start": None, "plies": [], "processes": [], "result": None})
        if tid not in order:
            order.append(tid)
        kind = r["record_type"]
        if kind == "game_start":
            g["start"] = r
        elif kind == "ply":
            g["plies"].append(r)
        elif kind == "process":
            g["processes"].append(r)
        elif kind == "game_result":
            g["result"] = r
        else:
            raise H4RunError(f"{path}: unexpected record type {kind!r}")
    out = []
    for tid in order:
        g = games[tid]
        if g["start"] is None or g["result"] is None:
            raise H4RunError(f"{tid}: a game_start or game_result record is missing")
        res = g["result"]
        try:
            got = transcript_digest_of(g["plies"], res)
            check_game(g["start"], g["plies"], g["processes"], res, check_pids=False)
        except (H2R.H2RulesError, H4RunVoid) as e:
            raise H4RunError(f"{tid}: {e}") from None
        if got != res.get("transcript_digest"):
            raise H4RunError(f"{tid}: transcript digest does not recompute")
        moves = [tuple(p["move"]) for p in g["plies"]]
        states = [TwixtState(active_size=A.BOARD_N, to_move="red")]
        for mv in moves:
            states.append(states[-1].apply_move(mv))
        for o in g["processes"]:
            p = o["board_ply"]
            exp = INT.position_digest(INT.expected_payload(states[p], moves[:p]))
            if exp != o["expected_position_digest"] or exp != o["t1j_position_digest"]:
                raise H4RunError(f"{tid}: process ordinal {o['ordinal']}: the position "
                                 f"digest does not recompute from the moves")
        final = states[-1]
        expect = res["winner"] if res["terminal_reason"] == "win" else None
        if final.winner() != expect:
            raise H4RunError(f"{tid}: the moves produce winner {final.winner()!r}, not "
                             f"the recorded {res['terminal_reason']} result")
        out.append({"task_id": tid, "header": header, **g})
    return out


def export_game(game: Mapping[str, Any], results_path: str) -> Dict[str, Any]:
    """A Replay.html-compatible record, DERIVED and hash-bound. Refuses, never guesses."""
    res, start, plies = game["result"], game["start"], game["plies"]
    fixture = game["header"].get("design") == FIXTURE_DESIGN
    tid = res["task_id"]
    if any(r.get("task_id") != tid for r in [start, *plies]):
        raise H4RunError(f"{tid}: mixed task ids in one game")
    if transcript_digest_of(plies, res) != res["transcript_digest"]:
        raise H4RunError(f"{tid}: transcript digest does not recompute")
    if [p["ply"] for p in plies] != list(range(1, res["plies"] + 1)):
        raise H4RunError(f"{tid}: the move sequence is not contiguous 1..plies")
    moves = [{"turn": p["ply"], "player": p["mover"], "row": p["move"][0],
              "col": p["move"][1], "bridges_created": [], "heuristics": {},
              "search_score": None, "root_top1_share": None} for p in plies]
    with open(results_path, "rb") as fh:
        source_sha = hashlib.sha256(fh.read()).hexdigest()
    who = {start["t1j_colour"]: "T1j (theirs)", start["incumbent_colour"]: "Incumbent (ours)"}
    note = ("derived from the H4 results file; a viewer export, not a result")
    return {"id": tid, "config_hash": "h4-recorded-evidence", "depth": None,
            "seed": start["seed"],
            "winner": res["winner"] if res["winner"] is not None else "draw",
            "starting_player": "red", "moves": moves,
            "meta": {"board_size": A.BOARD_N, "mode": "recorded_h4",
                     "reason": res["terminal_reason"], "n_moves": len(moves),
                     "task_id": tid, "pair_id": res["pair_id"], "arm": res["arm"],
                     "players": who, "transcript_digest": res["transcript_digest"],
                     "source_sha256": source_sha,
                     "evidence_note": (f"{FIXTURE_NOTE}. " if fixture else "") + note}}


def write_exports(results_path: str, export_dir: str, *,
                  allow_fixture: bool = False) -> List[str]:
    """After the run, from the results file: a create-only export per game."""
    os.makedirs(export_dir)                       # create-only: an existing dir refuses
    written = []
    for game in load_games(results_path, allow_fixture=allow_fixture):
        path = os.path.join(export_dir, f"{game['task_id']}.json")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(export_game(game, results_path), fh, indent=1, sort_keys=True)
        written.append(path)
    return written


# ─────────────────────────────────── the CLI ───────────────────────────────────

EXIT_OK = 0
EXIT_VOID = 3
EXIT_REFUSED = 4
EXIT_UNAUTHORIZED = 5


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI. Refuses while the gate is shut, BEFORE touching anything."""
    import argparse
    ap = argparse.ArgumentParser(prog="h4_runner",
                                 description="H4 pilot/study runner. IT IS NOT AUTHORIZED.")
    ap.add_argument("--mode", choices=("pilot", "study"), required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--segment", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--classes", required=True)
    a = ap.parse_args(argv)
    if not AUTH.H4_PILOT_EXECUTION_AUTHORIZED:
        print("the H4 pilot is UNAUTHORIZED. No JVM was started, nothing was played, "
              "and no file was written.", file=sys.stderr)
        return EXIT_UNAUTHORIZED
    try:
        out = run_games(mode=a.mode, manifest=a.manifest, segment=a.segment,
                        out_dir=a.out_dir, classes=a.classes,
                        deadline_s=SEGMENT_DEADLINE_S)
    except H4RunVoid as e:
        print(f"VOID: {e}", file=sys.stderr)
        return EXIT_VOID
    except (H4RunError, D1Error) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"complete: {out['games_completed']} games -> {out['results']}")
    return EXIT_OK


if __name__ == "__main__":                                                        # pragma: no cover
    raise SystemExit(main())
