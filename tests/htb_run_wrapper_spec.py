"""The H-TB FULL-RUN wrappers (`htb_pilot run label` / `run fit`) as exact derivations of the pinned cycle-2 training
wrapper, the reviewed pattern the timing wrappers also use (the user's review, 2026-11-05). Used by the generator
(once) and by tests/test_htb_pilot.py (every run)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = "docs/superpowers/evidence/combination-policy-cycle2-train-run/01_wrapper.sh"
SOURCE_SHA256 = "249ead56a496092b3ba5432b6d3431cfb5b6d3d916335b50e177105b9a27db19"
TARGETS = {"label": "docs/superpowers/evidence/htb-label-run/01_wrapper.sh",
           "fit": "docs/superpowers/evidence/htb-fit-run/01_wrapper.sh"}
#: the user's adopted ceilings (2026-11-05): TERM; KILL is TERM + 10 s; cleanup is observed through TERM + 20 s
TERM_S = {"label": 13190, "fit": 18590}
RUN_REL = {"label": "docs/superpowers/evidence/htb-label", "fit": "docs/superpowers/evidence/htb-fit"}
GATE = {"label": "HTB_LABEL_AUTHORIZED", "fit": "HTB_FIT_AUTHORIZED"}


def header(kind):
    term = TERM_S[kind]
    env = {"label": "the TEACHER venv ($HOME/twixt-cache/htb-teacher/teacher-venv/bin/python) with --clone-root "
                    "$HOME/twixt-cache/htb-teacher/twixtbot-ui",
           "fit": "the main checkout's project .venv, MLX on the CPU"}[kind]
    what = {"label": "selects the 240,000 + 24,000 rows (red-first, card §14) and labels them with the teacher, "
                     "compact as produced (card §16)",
            "fit": "fits calib020_0001 to the committed labels, examples built per batch (card §15), and scores F1-F3"}[kind]
    return (f"#!/bin/bash\n"
            f"# H-TB PILOT, FULL `run {kind}` -- OUTER WRAPPER (card docs/superpowers/2026-11-03-htb-twixtbot-distillation-\n"
            f"# pilot-card.md §10, §13-§16). The user's adopted ceiling (2026-11-05): TERM at {term} s, KILL 10 s later,\n"
            f"# {term + 10} s in all; cleanup observation bounded through {term + 20} s. Idle waiting before launch (idle_gate, at\n"
            f"# most its MAX_WAIT_S) is NOT part of that ceiling. The ceiling comes from a projection that is an ESTIMATE.\n"
            f"# Runs `htb_pilot run {kind}` ONCE in {env}: it {what}.\n"
            f"# BUILT ONLY (ledger, 2026-11-05): launching it is a separate approval. No automatic retry.\n"
            f"# A copy of combination-policy-cycle2-train-run/01_wrapper.sh (whose header follows, from its line 2); the diff: the\n"
            f"# paths, the gate ({GATE[kind]}), the module and its `run {kind}` command, the record kind, the interpreter and\n"
            f"# inputs it checks, the inner-exit table (htb codes: 0 COMPLETED, 2 REFUSED, 3 STOP, 4 FAILED, 5 INTERRUPTED),\n"
            f"# TERM_S and this header. The idle gate, background-compute check, supervisor, survivors, records and gate\n"
            f"# restoration are unchanged.\n"
            f"#\n")


def replacements(kind):
    gate_file = {"label": "htb_label_authorization.py", "fit": "htb_fit_authorization.py"}[kind]
    common = [
        ("OUT_REL=docs/superpowers/evidence/combination-policy-cycle2-train-run\n",
         f"OUT_REL=docs/superpowers/evidence/htb-{kind}-run\n"),
        ("RUN_REL=docs/superpowers/evidence/combination-policy-cycle2-train\n", f"RUN_REL={RUN_REL[kind]}\n"),
        ("GATE_REL=scripts/GPU/alphazero/combination_policy_cycle2_train_authorization.py\n",
         f"GATE_REL=scripts/GPU/alphazero/{gate_file}\n"),
        ("GATE=COMBINATION_POLICY_CYCLE2_TRAIN_AUTHORIZED\n", f"GATE={GATE[kind]}\n"),
        ("MODULE=scripts.GPU.alphazero.combination_policy_pilot\n", "MODULE=scripts.GPU.alphazero.htb_pilot\n"),
        ("TERM_S=2810\n", f"TERM_S={TERM_S[kind]}\n"),
        ("RECORD_KIND=combination-policy-pilot-record/1\n", f"RECORD_KIND=htb-{kind}-record/1\n"),
        ("""  0) outcome=COMPLETED ;;
  3) outcome=VOID ;;
  4) outcome=REFUSED ;;
  2) outcome=USAGE ;;
""", """  0) outcome=COMPLETED ;;
  2) outcome=REFUSED ;;
  3) outcome=STOP ;;
  4) outcome=FAILED ;;
  5) outcome=INTERRUPTED ;;
"""),
    ]
    if kind == "label":
        return common + [
            ('CKPT="$MAIN/checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"\n',
             'CKPT="$HOME/twixt-cache/htb-teacher/twixtbot-ui"            # the pinned clone (no checkpoint here)\n'
             'TPY="$HOME/twixt-cache/htb-teacher/teacher-venv/bin/python"\n'),
            ('[ -f "$CKPT" ] || refuse "$CKPT is not a file"\n',
             '[ -d "$CKPT/.git" ] || refuse "$CKPT is not the pinned clone"\n'
             '[ -x "$TPY" ] || refuse "$TPY is not executable"\n'),
            ('env MLX_ENABLE_TF32=0 PYTHONPATH=. "$PY" -B -m "$MODULE" run cycle2_train --checkpoint "$CKPT" \\\n',
             'env MLX_ENABLE_TF32=0 PYTHONPATH=. "$TPY" -B -m "$MODULE" run label --clone-root "$CKPT" \\\n'),
            # as the timing's: outer.json records the interpreter that runs the labelling (the teacher venv), with the
            # project interpreter, which runs only the idle and background-compute checks, beside it
            ('PRE=""; HEAD_C=""; PY=""; CKPT=""; start=0\n', 'PRE=""; HEAD_C=""; PY=""; TPY=""; CKPT=""; start=0\n'),
            ('"python": "%s", "checkpoint": "%s"}\\n\' \\\n',
             '"python": "%s", "checks_python": "%s", "clone_root": "%s"}\\n\' \\\n'),
            ('"$W" "$HEAD_C" "$PRE" "$PY" "$CKPT" > "$OUT_REL/outer.json"',
             '"$W" "$HEAD_C" "$PRE" "$TPY" "$PY" "$CKPT" > "$OUT_REL/outer.json"'),
        ]
    return common + [
        ('env MLX_ENABLE_TF32=0 PYTHONPATH=. "$PY" -B -m "$MODULE" run cycle2_train --checkpoint "$CKPT" \\\n',
         'env MLX_ENABLE_TF32=0 PYTHONPATH=. "$PY" -B -m "$MODULE" run fit \\\n'),
    ]


def derive(kind, source_text):
    body = source_text.split("\n", 1)[1]                       # the source minus its shebang
    for old, new in replacements(kind):
        if body.count(old) != 1:
            raise AssertionError(f"{kind}: anchor occurs {body.count(old)} times: {old[:60]!r}")
        body = body.replace(old, new)
    return header(kind) + body
