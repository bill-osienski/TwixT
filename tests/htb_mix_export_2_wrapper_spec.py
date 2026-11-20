"""The H-MX export attempt-2 wrapper, DERIVED from attempt 1's export wrapper (itself derived from H-OP's).

Amendment docs/superpowers/2026-11-20-htb-mix-export-attempt-2-amendment.md §2-§3. The wrapper is the parent's text
from the `set -u` line on, with literal substitutions that must each match exactly once and ONE inserted block (the
Node dependency preflight, before the idle gate), under a header written here. Only the two evidence paths, the
subcommand (`run export-2`) and the preflight change. The ceiling (TERM 3,590 s, grace 10 s), the gate, the module, the
record kind, the input checked (the current student file), the idle gate, the background-compute check, the perl
process-group supervisor, pgid.txt, survivors, the records, outer.json and the gate restoration are the parent's,
unchanged. Run once to write it (create-only); tests/test_htb_mix_export_2.py re-derives it and compares bytes.

    .venv/bin/python -m tests.htb_mix_export_2_wrapper_spec
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARENT = "docs/superpowers/evidence/htb-mix-export-run/01_wrapper.sh"
TERM_S = 3590
ORT_NODE_VERSION = "1.23.2"
SUBS = [("OUT_REL=docs/superpowers/evidence/htb-mix-export-run\n", "OUT_REL=docs/superpowers/evidence/htb-mix-export-2-run\n"),
        ("RUN_REL=docs/superpowers/evidence/htb-mix-export\n", "RUN_REL=docs/superpowers/evidence/htb-mix-export-2\n"),
        ('-m "$MODULE" run export \\\n', '-m "$MODULE" run export-2 \\\n')]
ANCHOR = "# THE IDLE GATE (last precondition)"
PREFLIGHT = (
    "# THE NODE DEPENDENCY (export attempt-2 amendment §3), after the one-shot checks and before the idle gate: a real\n"
    "# node_modules in THIS tree, from which Node (resolving as server/inference.js does) finds onnxruntime-node at\n"
    f"# exactly $W/node_modules and at version {ORT_NODE_VERSION}. A miss refuses: nothing is written or launched.\n"
    'NODE=$(command -v node) || refuse "no node on PATH"\n'
    '[ -d node_modules ] && [ ! -L node_modules ] || refuse "node_modules is not a real directory in this tree (clone it before the gate opens: amendment §3)"\n'
    "ORT_P=$(\"$NODE\" -e 'console.log(require(\"module\").createRequire(process.argv[1]).resolve(\"onnxruntime-node/package.json\"))' \"$W/server/inference.js\" 2>/dev/null) ||\n"
    '  refuse "Node cannot resolve onnxruntime-node from this tree"\n'
    '[ "$ORT_P" = "$W/node_modules/onnxruntime-node/package.json" ] || refuse "onnxruntime-node resolves to $ORT_P, not this tree\'s node_modules"\n'
    "ORT_V=$(\"$NODE\" -e 'console.log(require(process.argv[1]).version)' \"$ORT_P\" 2>/dev/null) || refuse \"cannot read the onnxruntime-node version\"\n"
    f'[ "$ORT_V" = "{ORT_NODE_VERSION}" ] || refuse "onnxruntime-node $ORT_V, the amendment pins {ORT_NODE_VERSION}"\n')


def derive() -> str:
    text = (ROOT / PARENT).read_text()
    body = text[text.index("set -u\n"):]
    for old, new in SUBS:
        if body.count(old) != 1:
            raise ValueError(f"{old!r} occurs {body.count(old)} times in {PARENT}")
        body = body.replace(old, new)
    if body.count(ANCHOR) != 1:
        raise ValueError(f"the idle-gate anchor occurs {body.count(ANCHOR)} times in {PARENT}")
    body = body.replace(ANCHOR, PREFLIGHT + ANCHOR)
    for want in (f"TERM_S={TERM_S}\n", "GRACE_S=10\n", "GATE=HTB_MIX_EXPORT_AUTHORIZED\n",
                 "MODULE=scripts.GPU.alphazero.htb_mix_export\n", "RECORD_KIND=htb-mix-export-record/1\n"):
        if want not in body:
            raise ValueError(f"the parent is not attempt 1's export wrapper: {want!r} missing")
    kill = TERM_S + 10
    header = (
        "#!/bin/bash\n"
        "# H-MX `htb_mix_export run export-2` -- OUTER WRAPPER, EXPORT ATTEMPT 2 (export amendment 1,\n"
        "# docs/superpowers/2026-11-20-htb-mix-export-attempt-2-amendment.md §2-§3). Fixed ceiling: TERM at "
        f"{TERM_S} s, KILL 10 s\n"
        f"# later, {kill} s in all; cleanup observation bounded through {kill + 10} s. Idle waiting before launch (idle_gate)\n"
        "# is NOT part of the ceiling. Runs the card's §8 export and parity of the verified challenger (entry 74) into fresh,\n"
        "# create-only evidence; attempt 1 (entry 75, VOID) stays as written. MLX on the CPU.\n"
        "# BUILT ONLY (the user's amendment-and-build approval, 2026-11-20): launching it is a separate approval. No\n"
        "# automatic retry; a run the ceiling stops is VOID and nothing from it is used.\n"
        f"# DERIVED by tests/htb_mix_export_2_wrapper_spec.py from {PARENT} (its text from\n"
        "# `set -u` on), with exact-once substitutions only (the two evidence paths and the subcommand, run export-2) and\n"
        "# one inserted block, the Node dependency preflight before the idle gate. The ceiling, the gate, the module, the\n"
        "# record kind, the input checked (the current student file), the idle gate, the background-compute check, the\n"
        "# perl process-group supervisor, pgid.txt, survivors, the records, outer.json and the gate restoration are the\n"
        "# parent's, unchanged.\n")
    return header + body


def wrapper_path() -> pathlib.Path:
    return ROOT / "docs/superpowers/evidence/htb-mix-export-2-run/01_wrapper.sh"


def main() -> None:
    p = wrapper_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "x") as fh:
        fh.write(derive())
    p.chmod(0o755)
    print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
