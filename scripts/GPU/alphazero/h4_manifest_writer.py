"""H4 MANIFEST WRITER -- the step-4d tool, built in 4b. NOT RUN in 4b.

Frozen by `docs/superpowers/2026-09-24-t1j-h4-step4-seed-card.md` §4 (bound) and
the analysis card §0.1. 🔴 THE CARDS ARE THE AUTHORITY.

Writes the pilot manifest (one entry) and the study manifest (four entries) from
the COMMITTED 4c re-qualification record, re-deriving every field and COMPARING
it -- never copying on trust:

  schedule / seeds / digest  the CANONICAL schedule of each registered block,
                             every seed passing the shared registry
  code / cards               sha256 of the files at HEAD, committed and unmodified,
                             equal to the record's bound identity -- else something
                             moved after the qualification: REFUSE
  t1j_runtime                the record's toolchain content, confirmed by a FRESH
                             toolchain verification now
  incumbent_identity         re-derived by the production path, the pilot's equal
                             to the record's
  evidence_dir               the fixed name, not yet existing

Each manifest is validated by the ANALYSIS module's own reader before anything is
written; both files are create-only in a create-only directory.

OUTSIDE `code`: nothing on the play path imports this module (the closure test in
`tests/test_h4_production_path.py` would see it).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any, Dict, Optional, Sequence, Tuple

from . import h4_pilot_feasibility as F
from . import h4_runner as R

MANIFEST_DIR = "docs/superpowers/evidence/t1j-h4-manifests"
PILOT_FILE, STUDY_FILE = "pilot_manifest.json", "study_manifest.json"
EXIT_OK, EXIT_REFUSED = 0, 4


def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"))


def _fresh_toolchain_content() -> Dict[str, Any]:
    """The toolchain as it verifies NOW -- jar sha256 and JDK components."""
    from . import e4_screen_integration as INT
    from . import t1j_toolchain as TC
    tc = TC.verified_paths()
    return {"jar_sha256": INT._sha256(tc["jar"]),
            "jdk_components": INT.verify_jdk_identity(tc["jdk_home"])}


def build_manifests(record: str, *, _fixture: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(pilot manifest, study manifest), every field re-derived and compared."""
    stages = os.path.join(os.path.dirname(os.path.abspath(record)), "stages.jsonl")
    F.check_durable([record, stages], _fixture=_fixture)
    if not _fixture:
        # 🔴 the files whose hashes the manifests carry must BE the committed ones
        F.check_durable([str(R.REPO_ROOT / p) for p in (*R.CODE, *R.CARDS)],
                        _fixture=False)
    rec = F._load_json(record, "qualification record")
    if not isinstance(rec, dict) or rec.get("record") != "H4_PRODUCTION_QUALIFICATION" \
            or rec.get("result") != "CLEAN":
        F.refuse(f"{record} is not a CLEAN H4 production qualification record")
    bound = (rec.get("checks") or {}).get("bound_identity")
    if not isinstance(bound, dict):
        F.refuse(f"{record} carries no bound identity")
    content = bound["t1j_runtime"]["toolchain"]

    config = R.frozen_argmax_config()
    identities = {}
    for mode in ("pilot", "study"):
        ident = R.frozen_incumbent_identity(config, design=R.DESIGNS[mode])
        try:
            R.check_incumbent_identity(ident, config, design=R.DESIGNS[mode])
        except R.H4RunError as e:
            F.refuse(f"incumbent identity: {e}")
        identities[mode] = ident
    now = R.bound_identity(identities["pilot"], content)
    for field in ("code", "cards", "incumbent_identity", "t1j_runtime"):
        if _canon(now[field]) != _canon(bound.get(field)):
            F.refuse(f"`{field}` at HEAD differs from the qualification record: something "
                     f"moved after 4c, so a new qualification is needed first")
    fresh = _fresh_toolchain_content()
    for k, v in fresh.items():
        if _canon(v) != _canon(content.get(k)):
            F.refuse(f"the toolchain verifies differently NOW: `{k}` differs from the "
                     f"qualified content")

    manifests = {}
    for mode, segments in (("pilot", (0,)), ("study", (0, 1, 2, 3))):
        entries = []
        for seg in segments:
            schedule = R.canonical_schedule(mode, seg, identities[mode])
            faults = R.seed_registry_faults(schedule)
            if faults:
                F.refuse(f"{mode} segment {seg}: seeds fail the shared registry: {faults[:3]}")
            evidence_dir = R.EVIDENCE_DIRS[(mode, seg)]
            if os.path.lexists(os.path.join(R.REPO_ROOT, evidence_dir)):
                F.refuse(f"{evidence_dir} already exists: that entry has been claimed")
            head = R.header_candidate(mode=mode, segment=seg, schedule=schedule,
                                      incumbent_identity=identities[mode],
                                      toolchain_content=content)
            entries.append({"segment": seg, "schedule": schedule,
                            **{k: head[k] for k in R.MANIFEST_BOUND},
                            "evidence_dir": evidence_dir})
        manifests[mode] = {"stage": R.DESIGNS[mode], "segments": entries,
                           "qualification_record_sha256": F.sha256_file(record)}
    with tempfile.TemporaryDirectory() as tmp:          # the ANALYSIS reader's verdict
        for mode, stage in (("pilot", F.PILOT), ("study", F.STUDY)):
            path = os.path.join(tmp, f"{mode}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(manifests[mode], fh)
            F.load_manifest(path, stage, _fixture=False)
    return manifests["pilot"], manifests["study"]


def _write(record: str, out_dir: str, *, _fixture: bool) -> Tuple[str, str]:
    if os.path.lexists(out_dir):
        F.refuse(f"{out_dir} is occupied; the manifests are written once")
    pilot, study = build_manifests(record, _fixture=_fixture)
    os.mkdir(out_dir)
    paths = (os.path.join(out_dir, PILOT_FILE), os.path.join(out_dir, STUDY_FILE))
    F.write_create_only(paths[0], pilot)
    F.write_create_only(paths[1], study)
    return paths


def write_manifests(record: str, out_dir: str = MANIFEST_DIR) -> Tuple[str, str]:
    """PUBLIC ENTRY. No argument relaxes durability or the fixture refusal."""
    return _write(record, out_dir, _fixture=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """PUBLIC ENTRY."""
    import argparse
    ap = argparse.ArgumentParser(prog="h4_manifest_writer",
                                 description="H4 step-4d manifest writer.")
    ap.add_argument("--record", required=True)
    ap.add_argument("--out-dir", default=MANIFEST_DIR)
    a = ap.parse_args(argv)
    try:
        paths = write_manifests(a.record, a.out_dir)
    except (F.H4AnalysisRefused, R.H4RunError) as e:
        print(f"REFUSED, nothing written: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print("manifests written: " + ", ".join(paths))
    return EXIT_OK


if __name__ == "__main__":                                                        # pragma: no cover
    raise SystemExit(main())
