"""The two H-MX Python-stage wrappers, DERIVED from H-OP's reviewed fit and export wrappers.

Card docs/superpowers/2026-11-18-htb-mix-distillation-card.md §11, §14. Each wrapper is its parent's text from the
`set -u` line on, with literal substitutions that must each match exactly once, under a header written here. Only the
paths, the gate, the module and the record kind change: the ceilings (TERM 11,690 / 3,590 s, grace 10 s), the input
checked (the current student file), the idle gate, the background-compute check, the perl process-group supervisor,
pgid.txt, survivors, the records, outer.json and the gate restoration are the parent's, unchanged. Run once to write
them (create-only); tests/test_htb_mix.py re-derives both and compares bytes.

    .venv/bin/python -m tests.htb_mix_wrapper_spec
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: stage -> (parent, H-OP stage stem, H-MX module, TERM_S, what it runs)
STAGES = {
    "fit": ("docs/superpowers/evidence/htb-onpolicy-fit-run/01_wrapper.sh", "fit", "htb_mix", 11690,
            "the mixed fit from the current student (7,500 updates, 16 H-OP + 16 H-TB cached rows per batch, card §6), "
            "S's reproduction of H-OP's recorded scores and the one-time fit screens M1-M6 (card §7), MLX on the CPU"),
    "export": ("docs/superpowers/evidence/htb-onpolicy-export-run/01_wrapper.sh", "export", "htb_mix_export", 3590,
               "on an ELIGIBLE fit only, one export of the challenger with the unmodified exporter, the structural "
               "check, both canaries, S1-S4 and V3 (card §8); module htb_mix_export"),
}


def derive(stage: str) -> str:
    parent, s, module, term_s, what = STAGES[stage]
    text = (ROOT / parent).read_text()
    body = text[text.index("set -u\n"):]
    op_module = "htb_onpolicy_export" if stage == "export" else "htb_onpolicy"
    subs = [(f"OUT_REL=docs/superpowers/evidence/htb-onpolicy-{s}-run\n", f"OUT_REL=docs/superpowers/evidence/htb-mix-{s}-run\n"),
            (f"RUN_REL=docs/superpowers/evidence/htb-onpolicy-{s}\n", f"RUN_REL=docs/superpowers/evidence/htb-mix-{s}\n"),
            (f"GATE_REL=scripts/GPU/alphazero/htb_onpolicy_{s}_authorization.py\n",
             f"GATE_REL=scripts/GPU/alphazero/htb_mix_{s}_authorization.py\n"),
            (f"GATE=HTB_ONPOLICY_{s.upper()}_AUTHORIZED\n", f"GATE=HTB_MIX_{s.upper()}_AUTHORIZED\n"),
            (f"MODULE=scripts.GPU.alphazero.{op_module}\n", f"MODULE=scripts.GPU.alphazero.{module}\n"),
            (f"RECORD_KIND=htb-onpolicy-{s}-record/1\n", f"RECORD_KIND=htb-mix-{s}-record/1\n")]
    for old, new in subs:
        if body.count(old) != 1:
            raise ValueError(f"{stage}: {old!r} occurs {body.count(old)} times in {parent}")
        body = body.replace(old, new)
    if f"TERM_S={term_s}\n" not in body or f'-m "$MODULE" run {stage} \\\n' not in body:
        raise ValueError(f"{stage}: the parent's ceiling or command is not the card's")
    kill = term_s + 10
    header = (
        "#!/bin/bash\n"
        f"# H-MX `run {stage}` -- OUTER WRAPPER (card docs/superpowers/2026-11-18-htb-mix-distillation-card.md,\n"
        f"# revision 1, §11, §14). Fixed ceiling: TERM at {term_s} s, KILL 10 s later, {kill} s in all; cleanup\n"
        f"# observation bounded through {kill + 10} s. Idle waiting before launch (idle_gate) is NOT part of the ceiling.\n"
        f"# Runs {what}.\n"
        "# BUILT ONLY (the user's build approval, 2026-11-18): launching it is a separate approval. No automatic retry; a\n"
        "# run the ceiling stops is VOID and nothing from it is scored or used.\n"
        f"# DERIVED by tests/htb_mix_wrapper_spec.py from {parent} (its text from `set -u` on), with exact-once\n"
        f"# substitutions only: the paths, the gate, the module ({module}) and the record kind; the ceiling, the input\n"
        "# checked (the current student file) and the command are the parent's. The idle gate, the background-compute\n"
        "# check, the perl process-group supervisor, pgid.txt, survivors, the records, outer.json and the gate\n"
        "# restoration are the parent's, unchanged.\n")
    return header + body


def wrapper_path(stage: str) -> pathlib.Path:
    return ROOT / "docs/superpowers/evidence" / f"htb-mix-{stage}-run" / "01_wrapper.sh"


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
