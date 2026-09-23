"""H4 PROCEED/STOP ARTIFACT -- the only door from the pilot to aggregation.

Frozen by `docs/superpowers/2026-09-23-t1j-h4-analysis-card.md` (amended
6490055), §2, §2.1 and §4.1. 🔴 THE CARD IS THE AUTHORITY.

Its OWN module (card amendment 1). It imports nothing from confirmatory analysis
and computes no outcome: the decision is the §1.4 rules RECOMPUTED from the pilot
results, never typed by a person and never copied from the report. The stored
feasibility report is read back and must equal that recomputation, so a report
edited after it was written -- or written from another pilot -- is refused.

A decision about files that are not durable is not durable either: the manifest,
the pilot results and the feasibility report must each be tracked at HEAD and
unmodified before anything is written.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional, Sequence

from . import h4_pilot_feasibility as F

#: Card §4.1.
EXIT = {"PROCEED": 0, "STOP_RUNTIME": 2, "STOP_CAP": 2, "STOP_COLLAPSE": 2, "VOID": 3}
EXIT_REFUSED = 4
#: Card §2: this card, the runner card and the replacement card.
ARTIFACT_CARDS = (F.CARD, F.RUNNER_CARD, F.REPLACEMENT_CARD)


def _decide(manifest: str, results: str, report: str, out: str, *,
            _fixture: bool) -> Dict[str, Any]:
    F.check_free(out)
    if os.path.dirname(os.path.abspath(out)) != os.path.dirname(os.path.abspath(results)):
        F.refuse(f"{out}: the artifact is written into the pilot's evidence directory")
    F.check_durable([manifest, results, report])
    stored = F._load_json(report, "feasibility report")
    if not isinstance(stored, dict):
        F.refuse(f"{report} is not a report")
    fresh = F.build_report(manifest, results, _fixture=_fixture)
    if {k: v for k, v in stored.items() if k != "written_at"} != fresh:
        F.refuse(f"{report} does not re-verify against the pilot results: it is not "
                 f"the report those results produce")
    artifact = {"decision": fresh["verdict"], "rules_fired": fresh["rules_fired"],
                "pilot_results_sha256": F.sha256_file(results),
                "feasibility_report_sha256": F.sha256_file(report),
                "cards": {c: F.sha256_file(os.path.join(F.R.REPO_ROOT, c))
                          for c in ARTIFACT_CARDS},
                "written_at": F.now_iso()}
    F.write_create_only(out, artifact)
    return artifact


def write_decision(manifest: str, results: str, report: str, out: str) -> Dict[str, Any]:
    """PUBLIC ENTRY. Fixture data is refused here: no argument relaxes it."""
    return _decide(manifest, results, report, out, _fixture=False)


def _main(argv: Optional[Sequence[str]], *, _fixture: bool) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="h4_pilot_decision",
                                 description="H4 proceed/stop artifact.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        art = _decide(a.manifest, a.results, a.report, a.out, _fixture=_fixture)
    except F.H4AnalysisRefused as e:
        print(f"REFUSED, nothing written: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"decision artifact written: {a.out} ({art['decision']})")
    return EXIT[art["decision"]]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """PUBLIC ENTRY."""
    return _main(argv, _fixture=False)


if __name__ == "__main__":                                                        # pragma: no cover
    raise SystemExit(main())
