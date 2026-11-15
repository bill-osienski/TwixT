"""The four H-OP Python-stage wrappers, DERIVED from the first H-TB fit's and label's reviewed wrappers.

Card docs/superpowers/2026-11-14-htb-onpolicy-distillation-card.md §12, §15. Each wrapper is its parent's text from the
`set -u` line on, with literal substitutions that must each match exactly once, under a header written here. Run once
to write them (create-only); tests/test_htb_onpolicy.py re-derives every wrapper and compares bytes.

    .venv/bin/python -m tests.htb_onpolicy_wrapper_spec
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIT_PARENT = "docs/superpowers/evidence/htb-fit-run/01_wrapper.sh"
LABEL_PARENT = "docs/superpowers/evidence/htb-label-run/01_wrapper.sh"
CKPT_LINE = 'CKPT="$MAIN/checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"\n'
STUDENT_LINE = 'CKPT="$HOME/twixt-cache/htb-pilot/student.safetensors"     # the current student (card §3); no checkpoint\n'

#: stage -> (parent, OUT_REL, RUN_REL, gate stem, gate, TERM_S, kind, (old command, new command), what it runs)
STAGES = {
    "gen-smoke": (FIT_PARENT, "htb-onpolicy-gen-smoke-run", "htb-onpolicy-gen-smoke", "htb_onpolicy_gen_smoke",
                  "HTB_ONPOLICY_GEN_SMOKE_AUTHORIZED", 3590, "htb-onpolicy-gen-smoke-record/1",
                  ('"$MODULE" run fit \\\n', '"$MODULE" run gen-smoke \\\n'),
                  "the 40-game generation smoke (20/20 alternating) on the GPU research path; then, outside this "
                  "wrapper, `htb_onpolicy yield` writes the frozen yield verdict (card §4.4)"),
    "generate": (FIT_PARENT, "htb-onpolicy-gen-run", "htb-onpolicy-gen", "htb_onpolicy_generate",
                 "HTB_ONPOLICY_GENERATE_AUTHORIZED", 143990, "htb-onpolicy-generate-record/1",
                 ('"$MODULE" run fit \\\n', '"$MODULE" run generate \\\n'),
                 "the generation run (9:1, stopping at 1.15 x N or G_MAX 4,000) on the GPU research path; the module "
                 "admits it only on a committed smoke whose yield verdict re-derives byte-equal and is LAUNCH_OK"),
    "label": (LABEL_PARENT, "htb-onpolicy-label-run", "htb-onpolicy-label", "htb_onpolicy_label",
              "HTB_ONPOLICY_LABEL_AUTHORIZED", 6590, "htb-onpolicy-label-record/1", None,
              "selection (120,000 + 12,000 rows, card §5) and the teacher's labels with the benchmark's lockstep "
              "check (card §6), in the TEACHER venv"),
    "fit": (FIT_PARENT, "htb-onpolicy-fit-run", "htb-onpolicy-fit", "htb_onpolicy_fit",
            "HTB_ONPOLICY_FIT_AUTHORIZED", 11690, "htb-onpolicy-fit-record/1", None,
            "the fit from the current student (7,500 updates, card §7), the imitation gate I1-I3 (card §8) and the "
            "descriptive retention scores, MLX on the CPU"),
    "export": (FIT_PARENT, "htb-onpolicy-export-run", "htb-onpolicy-export", "htb_onpolicy_export",
               "HTB_ONPOLICY_EXPORT_AUTHORIZED", 3590, "htb-onpolicy-export-record/1",
               ('"$MODULE" run fit \\\n', '"$MODULE" run export \\\n'),
               "on an ELIGIBLE fit only, one export of the challenger with the unmodified exporter, the structural "
               "check, both canaries, S1-S4 and V3 (card §9); module htb_onpolicy_export"),
}
MODULES = {"export": "scripts.GPU.alphazero.htb_onpolicy_export"}


def derive(stage: str) -> str:
    parent, out_rel, run_rel, stem, gate, term_s, kind, cmd, what = STAGES[stage]
    text = (ROOT / parent).read_text()
    body = text[text.index("set -u\n"):]
    old_out, old_run = ("htb-fit-run", "htb-fit") if parent == FIT_PARENT else ("htb-label-run", "htb-label")
    old_gate = "HTB_FIT_AUTHORIZED" if parent == FIT_PARENT else "HTB_LABEL_AUTHORIZED"
    old_stem = "htb_fit" if parent == FIT_PARENT else "htb_label"
    old_term = "TERM_S=18590\n" if parent == FIT_PARENT else "TERM_S=13190\n"
    old_kind = "RECORD_KIND=htb-fit-record/1\n" if parent == FIT_PARENT else "RECORD_KIND=htb-label-record/1\n"
    subs = [(f"OUT_REL=docs/superpowers/evidence/{old_out}\n", f"OUT_REL=docs/superpowers/evidence/{out_rel}\n"),
            (f"RUN_REL=docs/superpowers/evidence/{old_run}\n", f"RUN_REL=docs/superpowers/evidence/{run_rel}\n"),
            (f"GATE_REL=scripts/GPU/alphazero/{old_stem}_authorization.py\n",
             f"GATE_REL=scripts/GPU/alphazero/{stem}_authorization.py\n"),
            (f"GATE={old_gate}\n", f"GATE={gate}\n"),
            ("MODULE=scripts.GPU.alphazero.htb_pilot\n",
             f"MODULE={MODULES.get(stage, 'scripts.GPU.alphazero.htb_onpolicy')}\n"),
            (old_term, f"TERM_S={term_s}\n"), (old_kind, f"RECORD_KIND={kind}\n")]
    if parent == FIT_PARENT:
        subs.append((CKPT_LINE, STUDENT_LINE))
    if cmd:
        subs.append(cmd)
    for old, new in subs:
        if body.count(old) != 1:
            raise ValueError(f"{stage}: {old!r} occurs {body.count(old)} times in {parent}")
        body = body.replace(old, new)
    kill = term_s + 10
    header = (
        "#!/bin/bash\n"
        f"# H-OP `run {stage}` -- OUTER WRAPPER (card docs/superpowers/2026-11-14-htb-onpolicy-distillation-card.md,\n"
        f"# revision 3, §12, §15). Fixed ceiling: TERM at {term_s} s, KILL 10 s later, {kill} s in all; cleanup\n"
        f"# observation bounded through {kill + 10} s. Idle waiting before launch (idle_gate) is NOT part of the ceiling.\n"
        f"# Runs {what}.\n"
        "# BUILT ONLY (the user's build approval, 2026-11-14): launching it is a separate approval. No automatic retry; a\n"
        "# run the ceiling stops is VOID and nothing from it is scored or used.\n"
        f"# DERIVED by tests/htb_onpolicy_wrapper_spec.py from {parent} (its text from `set -u` on), with exact-once\n"
        f"# substitutions only: the paths, the gate, the module ({MODULES.get(stage, 'scripts.GPU.alphazero.htb_onpolicy').split('.')[-1]}), TERM_S, the record kind"
        + (", the input checked (the current student file, not a checkpoint)" if parent == FIT_PARENT else "")
        + (" and the command" if cmd else "") + ".\n"
        "# The idle gate, the background-compute check, the perl process-group supervisor, pgid.txt, survivors, the\n"
        "# records, outer.json and the gate restoration are the parent's, unchanged.\n")
    return header + body


def wrapper_path(stage: str) -> pathlib.Path:
    return ROOT / "docs/superpowers/evidence" / STAGES[stage][1] / "01_wrapper.sh"


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
