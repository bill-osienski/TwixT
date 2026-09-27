"""H5 MANIFEST WRITER -- the step-8 tool, built in step 3. NOT RUN in step 3.

Frozen by `docs/superpowers/2026-09-27-t1j-h5-implementation-card.md` §3 and §5.
🔴 THE CARD IS THE AUTHORITY.

Writes the ONE H5 study manifest (four entries) from the COMMITTED BINDING
qualification record, re-deriving every field and COMPARING it -- never copying
on trust. 🔴 An EARLY record can never bind, and three independent refusals see
to it (card §3): its ROLE is not BINDING; it is not at the fixed BINDING path; and
its bound `code` predates the step-6 registration, so it differs from HEAD.

The manifest is validated by the H5 ANALYSIS module's own reader before anything
is written; the file is create-only in a create-only directory. OUTSIDE `code`:
nothing on the play path imports this module.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any, Dict, Optional, Sequence

from . import h4_manifest_writer as W4
from . import h4_pilot_feasibility as F
from . import h4_runner as R4
from . import h5_confirmatory_analysis as C5
from . import h5_production_qualification as Q
from . import h5_runner as H5

MANIFEST_DIR = "docs/superpowers/evidence/t1j-h5-manifests"
STUDY_FILE = "study_manifest.json"
EXIT_OK, EXIT_REFUSED = 0, 4


def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"))


def _fresh_toolchain_content() -> Dict[str, Any]:
    """The toolchain as it verifies NOW -- H4's function, by import."""
    return W4._fresh_toolchain_content()


def build_manifest(record: str, *, _fixture: bool) -> Dict[str, Any]:
    """The study manifest, every field re-derived and compared."""
    stages = os.path.join(os.path.dirname(os.path.abspath(record)), "stages.jsonl")
    F.check_durable([record, stages], _fixture=_fixture)
    if not _fixture:
        # 🔴 the files whose hashes the manifest carries must BE the committed ones
        F.check_durable([str(H5.REPO_ROOT / p) for p in (*H5.CODE, *H5.CARDS)],
                        _fixture=False)
    rec = F._load_json(record, "qualification record")
    if not isinstance(rec, dict) or rec.get("record") != "H5_PRODUCTION_QUALIFICATION" \
            or rec.get("result") != "CLEAN":
        F.refuse(f"{record} is not a CLEAN H5 production qualification record")
    if rec.get("role") != Q.BINDING:                                  # refusal 1: role
        F.refuse(f"{record} is an {rec.get('role')!r} record: only a BINDING "
                 f"qualification may bind a manifest")
    want = os.path.realpath(os.path.join(H5.REPO_ROOT, Q.BINDING_DIR, "record.json"))
    if os.path.realpath(record) != want:                              # refusal 2: path
        F.refuse(f"{record} is not the BINDING record at {Q.BINDING_DIR}/record.json")
    bound = (rec.get("checks") or {}).get("bound_identity")
    if not isinstance(bound, dict):
        F.refuse(f"{record} carries no bound identity")
    content = bound["t1j_runtime"]["toolchain"]

    config = R4.frozen_argmax_config()
    ident = R4.frozen_incumbent_identity(config, design=H5.STUDY_DESIGN)
    try:
        R4.check_incumbent_identity(ident, config, design=H5.STUDY_DESIGN)
    except R4.H4RunError as e:
        F.refuse(f"incumbent identity: {e}")
    now = H5.bound_identity(ident, content)
    for field in ("code", "cards", "incumbent_identity", "t1j_runtime"):   # refusal 3
        if _canon(now[field]) != _canon(bound.get(field)):
            F.refuse(f"`{field}` at HEAD differs from the qualification record: something "
                     f"moved after the BINDING qualification, so a new one is needed first")
    fresh = _fresh_toolchain_content()
    for k, v in fresh.items():
        if _canon(v) != _canon(content.get(k)):
            F.refuse(f"the toolchain verifies differently NOW: `{k}` differs from the "
                     f"qualified content")

    entries = []
    for seg in H5.SEGMENTS:
        try:
            schedule = H5.canonical_schedule(seg, ident)
        except R4.H4RunError as e:
            F.refuse(f"segment {seg}: {e}")
        faults = R4.seed_registry_faults(schedule)
        if faults:
            F.refuse(f"segment {seg}: seeds fail the shared registry: {faults[:3]}")
        evidence_dir = H5.EVIDENCE_DIRS[seg]
        if os.path.lexists(os.path.join(H5.REPO_ROOT, evidence_dir)):
            F.refuse(f"{evidence_dir} already exists: that segment has been claimed")
        head = H5.header_candidate(mode="study", segment=seg, schedule=schedule,
                                   incumbent_identity=ident, toolchain_content=content)
        entries.append({"segment": seg, "schedule": schedule,
                        **{k: head[k] for k in R4.MANIFEST_BOUND},
                        "evidence_dir": evidence_dir})
    manifest = {"stage": H5.STUDY_DESIGN, "segments": entries,
                "qualification_record_sha256": F.sha256_file(record)}
    with tempfile.TemporaryDirectory() as tmp:          # the ANALYSIS reader's verdict
        path = os.path.join(tmp, STUDY_FILE)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        C5.load_manifest(path, _fixture=False)
    return manifest


def _write(record: str, out_dir: str, *, _fixture: bool) -> str:
    if os.path.lexists(out_dir):
        F.refuse(f"{out_dir} is occupied; the manifest is written once")
    manifest = build_manifest(record, _fixture=_fixture)
    os.mkdir(out_dir)
    path = os.path.join(out_dir, STUDY_FILE)
    F.write_create_only(path, manifest)
    return path


def write_manifest(record: str, out_dir: str = MANIFEST_DIR) -> str:
    """PUBLIC ENTRY. No argument relaxes durability or the fixture refusal."""
    return _write(record, out_dir, _fixture=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """PUBLIC ENTRY."""
    import argparse
    ap = argparse.ArgumentParser(prog="h5_manifest_writer",
                                 description="H5 step-8 manifest writer.")
    ap.add_argument("--record", required=True)
    ap.add_argument("--out-dir", default=MANIFEST_DIR)
    a = ap.parse_args(argv)
    try:
        path = write_manifest(a.record, a.out_dir)
    except (F.H4AnalysisRefused, R4.H4RunError) as e:
        print(f"REFUSED, nothing written: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"manifest written: {path}")
    return EXIT_OK


if __name__ == "__main__":                                                        # pragma: no cover
    raise SystemExit(main())
