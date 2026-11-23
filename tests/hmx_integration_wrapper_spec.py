"""The H-MX integration check wrapper, DERIVED from the H-MX match's reviewed wrapper.

Proposal docs/superpowers/2026-11-22-htb-mix-integration-proposal.md, revision 1, §5.1, B7. The wrapper is the match
wrapper's text from the `set -u` line on, with literal substitutions that must each match exactly once, under a header
written here. Run once to write it (create-only); tests/test_hmx_integration_wrapper.py re-derives it and compares
bytes.

    .venv/bin/python -m tests.hmx_integration_wrapper_spec
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
EV = "docs/superpowers/evidence"
PARENT = f"{EV}/htb-mix-match-run/01_wrapper.sh"
OUT_REL = f"{EV}/hmx-integration-check-run"
RUN_REL = f"{EV}/hmx-integration-check"
GATE_REL = "tests/htb_integration/hmx_integration_authorization.mjs"
GATE = "HMX_INTEGRATION_CHECK_AUTHORIZED"
DRIVER = "tests/htb_integration/hmx_integration_check.mjs"
PRELOAD = "tests/htb_integration/hmx_seed_preload.mjs"
MATCH_RECORD = f"{EV}/htb-mix-match/record.json"
MATCH_RECORD_SHA256 = "6bfa8bb6082da5e30b6816728d0a6ba1e6b04ea801fa9e26c85fee9dc6017371"
TERM_S, KILL_S = 14390, 14400

HEADER = (
    "#!/bin/bash\n"
    "# H-MX INTEGRATION CHECK: ONE GATED RUN (G0, S1, S2, A0-A2, S4, R1, R2, T) -- OUTER WRAPPER\n"
    "# (proposal docs/superpowers/2026-11-22-htb-mix-integration-proposal.md, revision 1, §5). The staged tree (this\n"
    "# one: the build with commit (ii), HARD_MODEL_ID 627c8d774df838a8) against the reference tree <this tree>-ref\n"
    "# (today's product) and the revert tree <this tree>-revert (the build with the swap reverted). Loads models and\n"
    "# starts servers; plays no new game. CEILING (proposal §5.1): TERM at 14390 s, KILL 10 s later, 14400 s (4 h) in\n"
    "# all. Idle waiting (idle_gate) is not part of the ceiling. A run the ceiling stops is VOID, with no automatic retry.\n"
    "# BUILT ONLY: launching it needs the user's approval of the gated checks.\n"
    f"# DERIVED by tests/hmx_integration_wrapper_spec.py from {PARENT} (its text from `set -u` on), with\n"
    "# exact-once substitutions only: the paths, the proposal, the gate (a line of\n"
    f"# {GATE_REL}), the driver and its arguments (the two trees), the stage and ceiling,\n"
    "# the record kind, the match evidence in place of the export record and the smoke evidence, and the seed preload in\n"
    "# place of the analyser. Everything else is the match wrapper's, unchanged: the idle gate, the BOINC check, the\n"
    "# one-line gate-open commit, the ONNX Runtime shutdown test (ledger entry 64), the onnxruntime-node pin, the perl\n"
    "# process-group supervisor (the servers the driver starts are in its group), outer.json, and gate restoration.\n"
)

#: the match wrapper's export-record block (card §8) -> the two trees this check reads
EXPORT_BLOCK = (
    "# THE EXPORT RECORD (card §8) must be committed; the challenger directory is the one it names.\n"
    'git ls-files --error-unmatch "$EXPORT_REL" > /dev/null 2>&1 || refuse "$EXPORT_REL is not committed"\n'
    "CHALLENGER_DIR=$(\"$NODE\" -e 'const r = JSON.parse(require(\"fs\").readFileSync(process.argv[1], \"utf8\")); "
    "if (r.status !== \"COMPLETED\" || r.verdict !== \"PARITY_PASS\" || !r.challenger?.dir) process.exit(1); "
    "console.log(r.challenger.dir);' \"$EXPORT_REL\") || refuse \"$EXPORT_REL is not a COMPLETED PARITY_PASS naming a "
    'challenger directory"\n'
    '[ -d "$CHALLENGER_DIR" ] || refuse "$CHALLENGER_DIR (the exported challenger) is missing"\n'
)
TREES_BLOCK = (
    "# THE TWO OTHER TREES (proposal §5.1) must exist, each with a real node_modules; the driver checks their commits,\n"
    "# cleanliness and runtime before anything loads.\n"
    'REF_ROOT="$W-ref"\n'
    'REVERT_ROOT="$W-revert"\n'
    'for t in "$REF_ROOT" "$REVERT_ROOT"; do\n'
    '  [ -d "$t/node_modules" ] && [ ! -L "$t/node_modules" ] || refuse "$t has no real node_modules"\n'
    "done\n"
)
SMOKE_BLOCK_START = "# THE SMOKE EVIDENCE must be committed (and, the tree being clean, unchanged and complete).\n"
SMOKE_BLOCK_END = ("|| refuse \"$TIMING_REL is not LAUNCH_OK at the fixed 64800 s cap\"\n")
MATCH_BLOCK = (
    "# THE MATCH EVIDENCE the replays read must be committed and unchanged.\n"
    'git ls-files --error-unmatch "$MATCH_RECORD" > /dev/null 2>&1 || refuse "$MATCH_RECORD is not committed"\n'
    '[ "$(shasum -a 256 "$MATCH_RECORD" | cut -d\' \' -f1)" = "$MATCH_RECORD_SHA256" ] || refuse "$MATCH_RECORD is not the '
    'committed match record"\n'
)


def _smoke_block(body: str) -> str:
    i = body.index(SMOKE_BLOCK_START)
    j = body.index(SMOKE_BLOCK_END, i) + len(SMOKE_BLOCK_END)
    return body[i:j]


def derive() -> str:
    text = (ROOT / PARENT).read_text()
    body = text[text.index("set -u\n"):]
    subs = [
        (f"OUT_REL={EV}/htb-mix-match-run\n", f"OUT_REL={OUT_REL}\n"),
        (f"RUN_REL={EV}/htb-mix-match\n", f"RUN_REL={RUN_REL}\n"),
        ("PLAN_REL=docs/superpowers/2026-11-18-htb-mix-distillation-card.md\n",
         "PLAN_REL=docs/superpowers/2026-11-22-htb-mix-integration-proposal.md\n"),
        ("ANALYSER=tests/htb_integration/mix_match_analyse.mjs\n", f"PRELOAD={PRELOAD}\n"),
        ("ANALYSER_SHA256=80e741d01463acf51a19db23b0c862b5caf605eb9d0dc97bafeb14f09e528d59\n",
         f"MATCH_RECORD={MATCH_RECORD}\nMATCH_RECORD_SHA256={MATCH_RECORD_SHA256}\n"),
        (f"EXPORT_REL={EV}/htb-mix-export-2/record.json\n", ""),
        ("GATE_REL=tests/htb_integration/mix_authorization.mjs\n", f"GATE_REL={GATE_REL}\n"),
        ("GATE=MIX_MATCH_AUTHORIZED\n", f"GATE={GATE}\n"),
        ("DRIVER=tests/htb_integration/mix_match.mjs\n", f"DRIVER={DRIVER}\n"),
        ("STAGE=match\n", "STAGE=check\n"),
        ("TERM_S=64790\n", f"TERM_S={TERM_S}\n"),
        ("KILL_S=64800\n", f"KILL_S={KILL_S}\n"),
        ("RECORD_KIND=htb-mix-match-record/1\n", "RECORD_KIND=hmx-integration-check-record/1\n"),
        ('[ "$(shasum -a 256 "$ANALYSER" | cut -d\' \' -f1)" = "$ANALYSER_SHA256" ] || refuse "$ANALYSER is not the frozen '
         'analyser (sha256 $ANALYSER_SHA256)"\n',
         'git ls-files --error-unmatch "$PRELOAD" > /dev/null 2>&1 || refuse "$PRELOAD is not committed"\n'),
        (EXPORT_BLOCK, TREES_BLOCK),
        ("# THE PINNED RUNTIMES (card §3),", "# THE PINNED RUNTIMES (proposal §5.1),"),
        ("env -u MODEL_MANIFEST -u HARD_MODEL_MANIFEST -u HARD_MODE \"$NODE\" \"$DRIVER\" run \"$STAGE\" "
         "--challenger-dir \"$CHALLENGER_DIR\" --kill-s \"$KILL_S\" \\\n",
         "env -u MODEL_MANIFEST -u HARD_MODEL_MANIFEST -u HARD_MODE -u NODE_OPTIONS -u HMX_SEED_QUEUE \"$NODE\" \"$DRIVER\" "
         "run \"$STAGE\" --ref-root \"$REF_ROOT\" --revert-root \"$REVERT_ROOT\" --kill-s \"$KILL_S\" \\\n"),
    ]
    for old, new in subs:
        if body.count(old) != 1:
            raise ValueError(f"{old!r} occurs {body.count(old)} times in {PARENT}")
        body = body.replace(old, new)
    smoke = _smoke_block(body)
    if body.count(smoke) != 1:
        raise ValueError("the smoke block is not unique")
    body = body.replace(smoke, MATCH_BLOCK)
    return HEADER + body


def wrapper_path() -> pathlib.Path:
    return ROOT / OUT_REL / "01_wrapper.sh"


def main() -> None:
    p = wrapper_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "x") as fh:
        fh.write(derive())
    p.chmod(0o755)
    print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
