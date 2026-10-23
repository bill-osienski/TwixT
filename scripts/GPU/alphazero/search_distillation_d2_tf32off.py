"""SEARCH DISTILLATION D2 -- FULL-FLOAT32 SETTING VALIDATION (MLX_ENABLE_TF32=0): the unchanged synthetic probe and
entry 27's four original saved linear-operation cases, in ONE fresh process launched with the setting. BUILT; run
only through its wrapper.

Card: `docs/superpowers/2026-10-23-search-distillation-d2-tf32off-validation-card.md`, pinned below by sha256, as are
everything it reads. Nothing runs on import, and importing this module imports no MLX: the setting is read by MLX once
per process, so `run` refuses unless MLX_ENABLE_TF32 is exactly "0" and no `mlx` module is loaded when it starts. It
then refuses unless the gate
`search_distillation_d2_tf32off_authorization.SEARCH_DISTILLATION_D2_TF32OFF_AUTHORIZED` is open (read at call time),
every pin matches and no output exists.
  SET A: `mlx_matmul_rows_probe.execute` with its frozen SPEC, unchanged, into `probe/`.
  SET B: fc and out at S0 and S21, operands from entry 27's saved `<case>_original.npz` (each checked against entry
    27's committed artifacts.jsonl, and its saved expected arrays against NumPy's recomputation), entry 27's three
    implementations (`search_distillation_d2_linrepro.compute`), the GPU three times and the CPU once.
  JUDGED, both sets, by the probe's `judge` and `scales` (its SPEC tolerances). Per set, over GPU evaluations:
    T (any element outside the tolerance), G (any gross element), R (repeats not bitwise identical). A set passes
    only if T = G = R = 0; CANDIDATE only if both pass, otherwise NOT_A_WORKAROUND. The CPU's counts are reported and
    never enter the verdict.
The verdict, the counts and the comparisons with entries 27 and 30 are computed inside the record boundary. A
difference is an observation; an incomplete set A, a fixture failing its checks, a non-finite result, an exception or
SIGTERM (the wrapper's 10-minute ceiling) are a VOID. No retry. No full outputs are saved.

    MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.search_distillation_d2_tf32off run
"""
from __future__ import annotations

import datetime
import hashlib
import importlib
import io
import json
import os
import pathlib
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from . import mlx_matmul_rows_probe as P
from . import search_distillation_d2_tf32off_authorization as GATE

ROOT = pathlib.Path(__file__).resolve().parents[3]
RECORD_KIND = "search-distillation-d2-tf32off-record/1"
CARD = "docs/superpowers/2026-10-23-search-distillation-d2-tf32off-validation-card.md"
CARD_SHA256 = "9704ce36ac7669941450ddc85dc62477dce7f23d0dae04cffdb42cbc2f786d67"
GENERATOR = "scripts/GPU/alphazero/mlx_matmul_rows_probe.py"
GENERATOR_SHA256 = "1217bc8e48589b077bbe0962564ffb5c0298326f96de6658ae39a83cf739bb57"
LINREPRO = "scripts/GPU/alphazero/search_distillation_d2_linrepro.py"
LINREPRO_SHA256 = "76fe5eff60a89aa6ca6e33f1e935d1dc617cd0d2f7152b18deab833cef53b8f0"
FIXTURE_ARTIFACTS = "docs/superpowers/evidence/search-distillation-d2-linrepro/artifacts.jsonl"
FIXTURE_ARTIFACTS_SHA256 = "668cae70a3b8ef85b8e9d90c857d37f98e58e52d38050162752235bb3e7abc88"
ENTRY27_CASES = "docs/superpowers/evidence/search-distillation-d2-linrepro/cases.jsonl"
ENTRY27_CASES_SHA256 = "35583162d8ce79f2fd01faf3b342e2a5050526386cdfa580cb25c3b239c2f617"
ENTRY30_CASES = "docs/superpowers/evidence/search-distillation-d2-mlxrows/cases.jsonl"
ENTRY30_CASES_SHA256 = "4ad1bdf538f3e79bd78a0fe0b237b92fa22a362acb7fbd760a83271dfe223d3d"
FIXTURE_DIR = pathlib.Path.home() / "twixt-cache" / "search-distillation-d2" / "linrepro-20261023T041634Z"
EVIDENCE_DIR = "docs/superpowers/evidence/search-distillation-d2-tf32off"
OUTPUT_PARENT = pathlib.Path.home() / "twixt-cache" / "search-distillation-d2"          # outside the repository
SETTING, SETTING_VALUE = "MLX_ENABLE_TF32", "0"
FROZEN = {"fixture_cases": ["fc_S0_original", "fc_S21_original", "out_S0_original", "out_S21_original"],
          "implementations": ["linear_autodiff", "explicit_autodiff", "explicit_products"],
          "outputs": ["Y", "dX", "dW", "db"], "gpu_repeats": 3, "cpu_repeats": 1,
          "ungated": {"outputs": ["dW", "db"], "layer_outputs": [["out", "Y"]]}, "key_cap": 100}
EXIT_COMPLETED, EXIT_USAGE, EXIT_VOID, EXIT_REFUSED = P.EXIT_COMPLETED, P.EXIT_USAGE, P.EXIT_VOID, P.EXIT_REFUSED
Void, Terminated = P.Void, P.Terminated


class Refused(Exception):
    """The run never started: the setting, MLX already loaded, gate closed, a pin, or an output exists."""


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _mlx_loaded() -> bool:
    return any(m == "mlx" or m.startswith("mlx.") for m in sys.modules)


def _read_fixture(name: str) -> bytes:
    return (FIXTURE_DIR / f"{name}.npz").read_bytes()


# ─────────────────────────────── set B: the saved cases ───────────────────────────────

def _fixture_entries() -> Dict[str, Dict[str, Any]]:
    raw = (ROOT / FIXTURE_ARTIFACTS).read_bytes()
    if _sha(raw) != FIXTURE_ARTIFACTS_SHA256:
        raise Void("fixtures", f"{FIXTURE_ARTIFACTS} is not the pinned bytes")
    return {e["name"]: e for e in map(json.loads, raw.decode().splitlines())}


def fixture_operands(name: str, entries) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """(operands, expected, source) for one saved case; VOID unless the bytes are entry 27's and the saved expected
    arrays equal NumPy's recomputation from the saved operands."""
    entry = entries.get(name)
    if entry is None or entry.get("path") != str(FIXTURE_DIR / f"{name}.npz"):
        raise Void("fixtures", f"{name}: not recorded at {FIXTURE_DIR} in {FIXTURE_ARTIFACTS}")
    raw = _read_fixture(name)
    if _sha(raw) != entry["sha256"]:
        raise Void("fixtures", f"{name}.npz is not the recorded file")
    with np.load(io.BytesIO(raw)) as z:
        o = {k: z[f"operand.{k}"] for k in ("X", "W", "b", "C", "legal")}
        saved = {k: z[f"expected.{k}"] for k in FROZEN["outputs"]}
    for k in ("X", "W", "b", "C"):
        if o[k].dtype != np.float32 or not np.all(np.isfinite(o[k])):
            raise Void("fixtures", f"{name}: operand {k} is not finite float32")
    ref = P.expected(o)
    for k in FROZEN["outputs"]:
        if not np.array_equal(ref[k], saved[k]):
            raise Void("fixtures", f"{name}: the saved expected {k} differs from the recomputation")
    return o, ref, {"path": entry["path"], "sha256": entry["sha256"]}


def fixture_case(name: str, entries, L) -> Dict[str, Any]:
    o, ref, source = fixture_operands(name, entries)
    sc = P.scales(o, ref)
    layer, state = name.split("_")[:2]
    result: Dict[str, Any] = {"id": name, "layer": layer, "state": state, "source": source, "scales": sc,
                              "rows": int(o["X"].shape[0]), "implementations": {}}
    for impl in FROZEN["implementations"]:
        per = {}
        for dev, reps in (("gpu", FROZEN["gpu_repeats"]), ("cpu", FROZEN["cpu_repeats"])):
            runs = []
            with P._On(dev):
                for i in range(reps):
                    got = L.compute(impl, o)
                    for k, v in got.items():
                        if not np.all(np.isfinite(v)):
                            raise Void("non_finite", f"{name} {impl} on {dev}, repeat {i}: {k} is not finite")
                    runs.append(got)
            per[dev] = {
                "outputs": {k: [P.judge(k, r[k], ref[k], sc[k], o, P.SPEC) for r in runs] for k in FROZEN["outputs"]},
                "repeats_bitwise": {k: [bool(np.array_equal(r[k], runs[0][k])) for r in runs[1:]]
                                    for k in FROZEN["outputs"]},
                "repeats_max_abs_diff": {k: [float(np.abs(r[k].astype(np.float64) - runs[0][k]).max())
                                             for r in runs[1:]] for k in FROZEN["outputs"]}}
        result["implementations"][impl] = per
    return result


# ─────────────────────────────── counts, verdict, comparisons ───────────────────────────────

def _evaluations(set_a: List[Dict[str, Any]], set_b: List[Dict[str, Any]]):
    """{set: [(case, layer, impl, device, per-device dict)]}: the probe's cases as implementation "probe"."""
    return {"synthetic": [(c["id"], c["layer"], "probe", dev, d) for c in set_a for dev, d in c["devices"].items()],
            "fixtures": [(c["id"], c["layer"], impl, dev, d) for c in set_b
                         for impl, per in c["implementations"].items() for dev, d in per.items()]}


def counts(rows) -> Dict[str, Dict[str, Any]]:
    """Per device: T, G, R (and the keys behind each), and the tolerance-only rows of Y and dX."""
    cap = FROZEN["key_cap"]
    out: Dict[str, Dict[str, Any]] = {}
    for case, _layer, impl, dev, d in rows:
        c = out.setdefault(dev, {"evaluations": 0, "T": 0, "G": 0, "R": 0, "T_keys": [], "G_keys": [], "R_keys": [],
                                 "tolerance_only_row_evaluations": 0, "tolerance_only_rows": 0})
        for k, judged in d["outputs"].items():
            for i, j in enumerate(judged):
                key = f"{case}/{impl}/{k}/{i}"
                c["evaluations"] += 1
                for name, hit in (("T", j["outside_tol"] != 0), ("G", j["gross"] != 0)):
                    if hit:
                        c[name] += 1
                        if len(c[f"{name}_keys"]) < cap:
                            c[f"{name}_keys"].append(key)
                if "rows_outside_tol" in j:
                    extra = j["rows_outside_tol"] - j["rows_gross"]
                    c["tolerance_only_rows"] += extra
                    c["tolerance_only_row_evaluations"] += extra > 0
            if not all(d["repeats_bitwise"][k]):
                c["R"] += 1
                if len(c["R_keys"]) < cap:
                    c["R_keys"].append(f"{case}/{impl}/{k}")
    return out


def verdict(per_set: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """A set passes only if the GPU's T = G = R = 0; CANDIDATE only if every set passes. The CPU never enters."""
    failed = {s: [n for n in ("T", "G", "R") if c["gpu"][n]] for s, c in per_set.items()}
    passed = {s: not f for s, f in failed.items()}
    return {"sets_pass": passed, "failed_counts": failed,
            "verdict": "CANDIDATE" if passed and all(passed.values()) else "NOT_A_WORKAROUND",
            "gross_removed_but_tolerance_failing": {s: c["gpu"]["G"] == 0 and c["gpu"]["T"] > 0
                                                    for s, c in per_set.items()}}


def _group(layer: str, dev: str, k: str) -> str:
    if dev == "cpu":
        return "cpu"
    ungated = k in FROZEN["ungated"]["outputs"] or [layer, k] in FROZEN["ungated"]["layer_outputs"]
    return "gpu_ungated" if ungated else "gpu_gated"


def _prior(path: str, sha: str) -> Dict[str, Any]:
    raw = (ROOT / path).read_bytes()
    if _sha(raw) != sha:
        raise Void("prior_record", f"{path} is not the pinned bytes")
    return {c["id"]: c for c in map(json.loads, raw.decode().splitlines())}


def compare(rows, prior_max_abs) -> Dict[str, Dict[str, Any]]:
    """Per group (cpu, gpu_ungated, gpu_gated): how many largest absolute errors equal the earlier record's exactly,
    and the keys of those that differ or are missing there."""
    out: Dict[str, Dict[str, Any]] = {}
    for case, layer, impl, dev, d in rows:
        for k, judged in d["outputs"].items():
            for i, j in enumerate(judged):
                g = out.setdefault(_group(layer, dev, k), {"equal": 0, "different": 0, "different_keys": []})
                before = prior_max_abs(case, impl, dev, k, i)
                if before is not None and before == j["max_abs"]:
                    g["equal"] += 1
                else:
                    g["different"] += 1
                    if len(g["different_keys"]) < FROZEN["key_cap"]:
                        g["different_keys"].append(f"{case}/{impl}/{dev}/{k}/{i}")
    return out


def _entry30_lookup(prior):
    def f(case, _impl, dev, k, i):
        try:
            return prior[case]["devices"][dev]["outputs"][k][i]["max_abs"]
        except (KeyError, IndexError):
            return None
    return f


def _entry27_lookup(prior):
    def f(case, impl, dev, k, i):
        try:
            return prior[case]["implementations"][impl][dev]["outputs"][k][i]["max_abs"]
        except (KeyError, IndexError):
            return None
    return f


# ─────────────────────────────── the record boundary ───────────────────────────────

def conclude(out: pathlib.Path, example: pathlib.Path) -> Tuple[int, Dict[str, Any]]:
    """THE RECORD BOUNDARY: record.json on every exit this survives; SIGTERM -> VOID at step "terminated"."""
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    writer = P._Writer(out)
    rec: Dict[str, Any] = {"kind": RECORD_KIND, "card": CARD, "card_sha256": CARD_SHA256, "frozen": FROZEN,
                           "status": None, "written_by": "script", "setting": {SETTING: os.environ.get(SETTING)},
                           "example_dir": str(example), "probe": None, "fixture_cases_done": [],
                           "loadavg_before": os.getloadavg()}

    def log(event, **kw):
        writer.line("progress.jsonl", {"event": event, "t": round(time.monotonic() - t0, 3), **kw})

    def on_term(signum, _frame):
        raise Terminated(signal.Signals(signum).name)

    code = EXIT_VOID
    main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, on_term) if main_thread else None
    try:
        log("start", setting=rec["setting"])
        code_a, rec_a = P.execute(out / "probe", example, P.SPEC)
        rec["probe"] = {k: rec_a.get(k) for k in ("status", "void", "example", "total_seconds",
                                                  "cases_with_gpu_gross", "cases_with_cpu_gross",
                                                  "cases_gpu_nonrepeatable")}
        log("set_a", status=rec_a["status"])
        if code_a != EXIT_COMPLETED:
            raise Void("probe", f"set A ended {rec_a['status']}: {rec_a.get('void')}")
        set_a = [json.loads(line) for line in (out / "probe" / "cases.jsonl").read_text().splitlines()]
        entries = _fixture_entries()
        L = importlib.import_module(f"{__package__}.search_distillation_d2_linrepro")
        if pathlib.Path(L.__file__).resolve() != ROOT / LINREPRO:
            raise Void("linrepro", f"{L.__file__} is not {LINREPRO}")
        t_b = time.monotonic()
        set_b = []
        for name in FROZEN["fixture_cases"]:
            case = fixture_case(name, entries, L)
            writer.line("fixture_cases.jsonl", case)
            set_b.append(case)
            rec["fixture_cases_done"].append(name)
            log("set_b_case", id=name)
        rec["fixture_seconds"] = round(time.monotonic() - t_b, 3)
        rows = _evaluations(set_a, set_b)
        rec["counts"] = {s: counts(r) for s, r in rows.items()}
        rec.update(verdict(rec["counts"]))
        rec["comparisons"] = {
            "synthetic_vs_entry30": compare(rows["synthetic"], _entry30_lookup(_prior(ENTRY30_CASES,
                                                                                      ENTRY30_CASES_SHA256))),
            "fixtures_vs_entry27": compare(rows["fixtures"], _entry27_lookup(_prior(ENTRY27_CASES,
                                                                                    ENTRY27_CASES_SHA256)))}
        rec["status"] = "COMPLETED"
        code = EXIT_COMPLETED
    except Terminated as t:
        rec.update(status="VOID", void={"step": "terminated", "detail": f"{t} after {time.monotonic() - t0:.1f} s"})
    except Void as v:
        rec.update(status="VOID", void={"step": v.step, "detail": v.detail})
    except Exception as e:                       # any exception is a VOID, and it is recorded
        rec.update(status="VOID", void={"step": "exception", "detail": f"{type(e).__name__}: {e}"})
    except BaseException as e:
        rec.update(status="INTERRUPTED", void={"step": "interrupted", "detail": type(e).__name__})
        raise
    finally:
        if main_thread:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            rec["setting_at_end"] = {SETTING: os.environ.get(SETTING)}
            rec["total_seconds"] = round(time.monotonic() - t0, 3)
            rec["loadavg_after"] = os.getloadavg()
            log("end", status=rec["status"], verdict=rec.get("verdict"))
            writer.close()
            with open(out / "record.json", "x") as fh:
                json.dump(rec, fh, indent=1)
                fh.write("\n")
        finally:
            if main_thread:
                signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)
    return code, rec


def run() -> int:
    """THE PUBLIC ENTRY. No seam: the setting, the pinned card and inputs, the two create-only directories."""
    if os.environ.get(SETTING) != SETTING_VALUE:
        raise Refused(f"{SETTING} is {os.environ.get(SETTING)!r}, not {SETTING_VALUE!r}: launch a fresh process with it")
    if _mlx_loaded():
        raise Refused("MLX was imported before the setting was checked: launch a fresh process")
    if GATE.SEARCH_DISTILLATION_D2_TF32OFF_AUTHORIZED is not True:
        raise Refused("SEARCH_DISTILLATION_D2_TF32OFF_AUTHORIZED is closed: the validation is not authorized")
    if _sha((ROOT / CARD).read_bytes()) != CARD_SHA256:
        raise Refused(f"{CARD} is not the pinned card")
    if _sha(pathlib.Path(P.__file__).read_bytes()) != GENERATOR_SHA256 or \
            pathlib.Path(P.__file__).resolve() != ROOT / GENERATOR:
        raise Refused(f"{GENERATOR} is not the pinned generator")
    for rel, sha in ((LINREPRO, LINREPRO_SHA256), (FIXTURE_ARTIFACTS, FIXTURE_ARTIFACTS_SHA256),
                     (ENTRY27_CASES, ENTRY27_CASES_SHA256), (ENTRY30_CASES, ENTRY30_CASES_SHA256)):
        if _sha((ROOT / rel).read_bytes()) != sha:
            raise Refused(f"{rel} is not the pinned bytes")
    if not FIXTURE_DIR.is_dir():
        raise Refused(f"{FIXTURE_DIR} is not a directory")
    out = ROOT / EVIDENCE_DIR
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the run is one-shot")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    example = OUTPUT_PARENT / f"tf32off-{stamp}"
    if not OUTPUT_PARENT.is_dir() or os.path.lexists(example):
        raise Refused(f"{OUTPUT_PARENT} is not a directory, or {example} exists")
    code, rec = conclude(out, example)
    print(json.dumps({"status": rec["status"], "void": rec.get("void"), "verdict": rec.get("verdict"),
                      "failed_counts": rec.get("failed_counts"),
                      "gpu": {s: {n: c["gpu"][n] for n in ("T", "G", "R")} for s, c in rec.get("counts", {}).items()},
                      "example": (rec.get("probe") or {}).get("example")}))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv != ["run"]:
        print("usage: MLX_ENABLE_TF32=0 python -m scripts.GPU.alphazero.search_distillation_d2_tf32off run",
              file=sys.stderr)
        return EXIT_USAGE
    try:
        return run()
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
