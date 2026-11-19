"""The H-MX recovery wrapper, DERIVED from the H-MX fit wrapper (itself derived from H-OP's reviewed fit wrapper).

Amendment docs/superpowers/2026-11-19-htb-mix-recovery-amendment.md §5. The wrapper is the parent's text from the
`set -u` line on, with literal substitutions that must each match exactly once, under a header written here. Only the
paths, the gate, the module, the record kind and the subcommand (`run replay`) change. The fit's ceiling (TERM 11,690 s,
grace 10 s), the input checked (the current student file), the idle gate, the background-compute check, the perl
process-group supervisor, pgid.txt, survivors, the records, outer.json and the gate restoration are the parent's,
unchanged. Run once to write it (create-only); tests/test_htb_mix_recovery.py re-derives it and compares bytes.

    .venv/bin/python -m tests.htb_mix_recovery_wrapper_spec
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARENT = "docs/superpowers/evidence/htb-mix-fit-run/01_wrapper.sh"
TERM_S = 11690
SUBS = [("OUT_REL=docs/superpowers/evidence/htb-mix-fit-run\n", "OUT_REL=docs/superpowers/evidence/htb-mix-recovery-run\n"),
        ("RUN_REL=docs/superpowers/evidence/htb-mix-fit\n", "RUN_REL=docs/superpowers/evidence/htb-mix-recovery\n"),
        ("GATE_REL=scripts/GPU/alphazero/htb_mix_fit_authorization.py\n",
         "GATE_REL=scripts/GPU/alphazero/htb_mix_recovery_authorization.py\n"),
        ("GATE=HTB_MIX_FIT_AUTHORIZED\n", "GATE=HTB_MIX_RECOVERY_AUTHORIZED\n"),
        ("MODULE=scripts.GPU.alphazero.htb_mix\n", "MODULE=scripts.GPU.alphazero.htb_mix_recovery\n"),
        ("RECORD_KIND=htb-mix-fit-record/1\n", "RECORD_KIND=htb-mix-recovery-record/1\n"),
        ('-m "$MODULE" run fit \\\n', '-m "$MODULE" run replay \\\n')]


def derive() -> str:
    text = (ROOT / PARENT).read_text()
    body = text[text.index("set -u\n"):]
    for old, new in SUBS:
        if body.count(old) != 1:
            raise ValueError(f"{old!r} occurs {body.count(old)} times in {PARENT}")
        body = body.replace(old, new)
    if f"TERM_S={TERM_S}\n" not in body or "GRACE_S=10\n" not in body:
        raise ValueError("the parent's ceiling is not the amendment's")
    kill = TERM_S + 10
    header = (
        "#!/bin/bash\n"
        "# H-MX `htb_mix_recovery run replay` -- OUTER WRAPPER (recovery amendment 1,\n"
        "# docs/superpowers/2026-11-19-htb-mix-recovery-amendment.md §5). Fixed ceiling: TERM at "
        f"{TERM_S} s, KILL 10 s later, {kill} s\n"
        f"# in all; cleanup observation bounded through {kill + 10} s. Idle waiting before launch (idle_gate) is NOT part of\n"
        "# the ceiling. Runs the one exact replay of the failed H-MX fit (entry 73): the same pinned starting weights,\n"
        "# cached inputs, 16 + 16 batch order and 7,500 updates, no validation scoring; a save only if the trained-state\n"
        "# digest is adba50fb..., and ELIGIBLE only after the saved file reloads at that digest. MLX on the CPU.\n"
        "# BUILT ONLY (the user's amendment-and-build approval, 2026-11-19): launching it is a separate approval. No\n"
        "# automatic retry; a run the ceiling stops is VOID and nothing from it is used.\n"
        f"# DERIVED by tests/htb_mix_recovery_wrapper_spec.py from {PARENT} (its text from\n"
        "# `set -u` on), with exact-once substitutions only: the paths, the gate, the module (htb_mix_recovery), the\n"
        "# record kind and the subcommand (run replay); the ceiling, the input checked (the current student file), the\n"
        "# idle gate, the background-compute check, the perl process-group supervisor, pgid.txt, survivors, the\n"
        "# records, outer.json and the gate restoration are the parent's, unchanged.\n")
    return header + body


def wrapper_path() -> pathlib.Path:
    return ROOT / "docs/superpowers/evidence/htb-mix-recovery-run/01_wrapper.sh"


def main() -> None:
    p = wrapper_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "x") as fh:
        fh.write(derive())
    p.chmod(0o755)
    print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
