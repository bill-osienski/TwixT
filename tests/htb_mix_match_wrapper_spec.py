"""The two H-MX match wrappers, DERIVED from H-OP's reviewed match-smoke and match wrappers.

Card docs/superpowers/2026-11-18-htb-mix-distillation-card.md §9, §11, §14. Each wrapper is its H-OP parent's text from
the `set -u` line on, with literal substitutions that must each match exactly once, under a header written here. The
frozen H-MX analyser's sha256 is pinned at derivation time (read from the file). Run once to write them (create-only);
tests/htb_integration/test_mix_match.mjs re-derives both and compares bytes.

    .venv/bin/python -m tests.htb_mix_match_wrapper_spec
"""
import hashlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
EV = "docs/superpowers/evidence"
ANALYSER = "tests/htb_integration/mix_match_analyse.mjs"
OP_ANALYSER_SHA256 = "1876706130de56fe81ea44c8ef513fe2802440c0fa3d2c77eafce8ca07e6f7b7"
OP_SMOKE = f"{EV}/htb-onpolicy-match-smoke"
MX_SMOKE = f"{EV}/htb-mix-match-smoke"

#: stage -> (H-OP parent's dir, H-MX run dir, H-OP gate, H-MX gate)
STAGES = {
    "match-smoke": ("htb-onpolicy-match-smoke", "htb-mix-match-smoke", "ONPOLICY_MATCH_SMOKE_AUTHORIZED",
                    "MIX_MATCH_SMOKE_AUTHORIZED"),
    "match": ("htb-onpolicy-match", "htb-mix-match", "ONPOLICY_MATCH_AUTHORIZED", "MIX_MATCH_AUTHORIZED"),
}

HEADERS = {
    "match-smoke": (
        "# H-MX MATCH, TIMING SMOKE: TWO COLOUR-REVERSED PAIRS (4 GAMES, UNSCORED) -- OUTER WRAPPER\n"
        "# (card docs/superpowers/2026-11-18-htb-mix-distillation-card.md, revision 1, §9, §11). The challenger (the export\n"
        "# qualified by the committed H-MX export record) against the current hard 78c16a31c3dba37f, both through\n"
        "# protocolMove at 200 simulations, under the product's game-end rule. CEILING (card §11): TERM at 1790 s, KILL 10 s\n"
        "# later, 1800 s in all. Idle waiting (idle_gate) is not part of the ceiling. A run the ceiling stops is VOID, with no\n"
        "# automatic retry. BUILT ONLY: launching it needs the user's launch review.\n"),
    "match": (
        "# H-MX MATCH: 400 COLOUR-REVERSED PAIRS (800 SCORED GAMES) -- OUTER WRAPPER\n"
        "# (card docs/superpowers/2026-11-18-htb-mix-distillation-card.md, revision 1, §9, §11, §13). The challenger (the\n"
        "# export qualified by the committed H-MX export record) against the current hard 78c16a31c3dba37f, both through\n"
        "# protocolMove at 200 simulations, under the product's game-end rule. CEILING FIXED IN THE CARD (§11): TERM at 64790\n"
        "# s, KILL 10 s later, 64800 s (18 h) in all, never raised. Launch only if the committed smoke timing.json re-derives\n"
        "# byte-equal with the frozen analyser and says LAUNCH_OK. Idle waiting (idle_gate) is not part of the ceiling. A run\n"
        "# the ceiling stops is VOID, with no automatic retry. BUILT ONLY: launching it needs the user's launch review.\n"),
}
DERIVED_NOTE = (
    "# DERIVED by tests/htb_mix_match_wrapper_spec.py from {parent} (its text from `set -u` on), with exact-once\n"
    "# substitutions only: the paths, the card, the gate (a line of tests/htb_integration/mix_authorization.mjs), the\n"
    "# driver (mix_match.mjs), the frozen H-MX analyser and its sha256 pinned HERE, the H-MX export record, the record\n"
    "# kind and the card's section numbers{smoke_note}. Everything else is H-OP's, unchanged: the idle gate, the BOINC\n"
    "# check, the one-line gate-open commit, the ONNX Runtime shutdown test (ledger entry 64), the onnxruntime-node pin,\n"
    "# the perl process-group supervisor, outer.json, and gate restoration.\n")


def parent(stage: str) -> str:
    return f"{EV}/{STAGES[stage][0]}-run/01_wrapper.sh"


def analyser_sha256() -> str:
    return hashlib.sha256((ROOT / ANALYSER).read_bytes()).hexdigest()


def derive(stage: str) -> str:
    op_dir, mx_dir, op_gate, mx_gate = STAGES[stage]
    text = (ROOT / parent(stage)).read_text()
    body = text[text.index("set -u\n"):]
    subs = [
        (f"OUT_REL={EV}/{op_dir}-run\n", f"OUT_REL={EV}/{mx_dir}-run\n"),
        (f"RUN_REL={EV}/{op_dir}\n", f"RUN_REL={EV}/{mx_dir}\n"),
        ("PLAN_REL=docs/superpowers/2026-11-14-htb-onpolicy-distillation-card.md\n",
         "PLAN_REL=docs/superpowers/2026-11-18-htb-mix-distillation-card.md\n"),
        ("ANALYSER=tests/htb_integration/onpolicy_match_analyse.mjs\n", f"ANALYSER={ANALYSER}\n"),
        (f"ANALYSER_SHA256={OP_ANALYSER_SHA256}\n", f"ANALYSER_SHA256={analyser_sha256()}\n"),
        (f"EXPORT_REL={EV}/htb-onpolicy-export/record.json\n", f"EXPORT_REL={EV}/htb-mix-export/record.json\n"),
        ("GATE_REL=tests/htb_integration/onpolicy_authorization.mjs\n",
         "GATE_REL=tests/htb_integration/mix_authorization.mjs\n"),
        (f"GATE={op_gate}\n", f"GATE={mx_gate}\n"),
        ("DRIVER=tests/htb_integration/onpolicy_match.mjs\n", "DRIVER=tests/htb_integration/mix_match.mjs\n"),
        ("RECORD_KIND=htb-onpolicy-match-record/1\n", "RECORD_KIND=htb-mix-match-record/1\n"),
        ("# THE EXPORT RECORD (card §9) must be committed", "# THE EXPORT RECORD (card §8) must be committed"),
        ("# THE PINNED RUNTIMES (card §4),", "# THE PINNED RUNTIMES (card §3),"),
    ]
    if stage == "match":
        subs += [
            (f"for f in {OP_SMOKE}/record.json {OP_SMOKE}/progress.jsonl \\\n",
             f"for f in {MX_SMOKE}/record.json {MX_SMOKE}/progress.jsonl \\\n"),
            (f"         {OP_SMOKE}-run/outer.json {OP_SMOKE}-run/pgid.txt; do\n",
             f"         {MX_SMOKE}-run/outer.json {MX_SMOKE}-run/pgid.txt; do\n"),
            (f'[ "$(git ls-files -- {OP_SMOKE}/games | wc -l', f'[ "$(git ls-files -- {MX_SMOKE}/games | wc -l'),
            (f"TIMING_REL={OP_SMOKE}/timing.json\n", f"TIMING_REL={MX_SMOKE}/timing.json\n"),
        ]
    for old, new in subs:
        if body.count(old) != 1:
            raise ValueError(f"{stage}: {old!r} occurs {body.count(old)} times in {parent(stage)}")
        body = body.replace(old, new)
    smoke_note = ", and the committed smoke evidence and its timing" if stage == "match" else ""
    header = ("#!/bin/bash\n" + HEADERS[stage]
              + DERIVED_NOTE.format(parent=parent(stage), smoke_note=smoke_note))
    return header + body


def wrapper_path(stage: str) -> pathlib.Path:
    return ROOT / EV / f"{STAGES[stage][1]}-run" / "01_wrapper.sh"


def main() -> None:
    for stage in STAGES:
        p = wrapper_path(stage)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "x") as fh:
            fh.write(derive(stage))
        p.chmod(0o755)
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
