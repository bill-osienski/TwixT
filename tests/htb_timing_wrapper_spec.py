"""The H-TB timing wrappers as exact derivations of the cycle-2 training wrapper (the user's 2026-11-04 review:
'reused wrappers'). Used by the generator (once) and by tests/test_htb_timing.py (every run)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = "docs/superpowers/evidence/combination-policy-cycle2-train-run/01_wrapper.sh"
SOURCE_SHA256 = "249ead56a496092b3ba5432b6d3431cfb5b6d3d916335b50e177105b9a27db19"
#: plan §6: T-label attempt 2 runs from fresh label2 paths; T-fit's paths are unused and unchanged
RUN_DIRS = {"label": "label2", "fit": "fit"}
TARGETS = {"label": "docs/superpowers/evidence/htb-timing-label2-run/01_wrapper.sh",
           "fit": "docs/superpowers/evidence/htb-timing-fit-run/01_wrapper.sh"}
#: attempt 1's wrapper, spent (STOP, ledger #172): kept byte-for-byte, never regenerated
SPENT = {"docs/superpowers/evidence/htb-timing-label-run/01_wrapper.sh": "23e4085a8ee2eca8d6c54994e2a0f39c5e95355005b18d0e21b1539111e9a9a1"}


def header(kind):
    term = {"label": 1190, "fit": 1790}[kind]
    env = {"label": "the TEACHER venv ($HOME/twixt-cache/htb-teacher/teacher-venv/bin/python) with --clone-root "
                    "$HOME/twixt-cache/htb-teacher/twixtbot-ui",
           "fit": "the main checkout's project .venv, MLX on the CPU"}[kind]
    return (f"#!/bin/bash\n"
            f"# H-TB BOUNDED TIMING, T-{kind.upper()} -- OUTER WRAPPER (plan docs/superpowers/2026-11-04-htb-bounded-timing-plan.md\n"
            f"# §5; the user's exact ceiling: TERM at {term} s, KILL 10 s later, {term + 10} s in all; cleanup observation bounded\n"
            f"# through {term + 20} s). Idle waiting before launch (idle_gate, at most its MAX_WAIT_S) is NOT part of that ceiling.\n"
            + ({"label": "# ATTEMPT 2, from fresh paths (plan §6): attempt 1 (htb-timing-label-run/, htb-timing/label/) STOPPED on a\n"
                         "# black-first game and stays unchanged (ledger #172). Card §14: red-first games only.\n"}.get(kind, ""))
            + f"# Runs `htb_timing time {kind}` ONCE in {env}. BUILT ONLY (ledger, 2026-11-04): launching it is a separate approval.\n"
            f"# A copy of combination-policy-cycle2-train-run/01_wrapper.sh (whose header follows, from its line 2); the diff: the\n"
            f"# paths, the gate, the module and its `time {kind}` command, the record kind, the interpreter and inputs it\n"
            f"# checks, the inner-exit table (htb codes: 0 COMPLETED, 2 REFUSED, 3 STOP, 4 FAILED, 5 INTERRUPTED), TERM_S and this\n"
            f"# header. The idle gate, background-compute check, supervisor, survivors, records and gate restoration are unchanged.\n"
            f"#\n")


def replacements(kind):
    term = {"label": 1190, "fit": 1790}[kind]
    common = [
        ("OUT_REL=docs/superpowers/evidence/combination-policy-cycle2-train-run\n",
         f"OUT_REL=docs/superpowers/evidence/htb-timing-{RUN_DIRS[kind]}-run\n"),
        ("RUN_REL=docs/superpowers/evidence/combination-policy-cycle2-train\n",
         f"RUN_REL=docs/superpowers/evidence/htb-timing/{RUN_DIRS[kind]}\n"),
        ("GATE_REL=scripts/GPU/alphazero/combination_policy_cycle2_train_authorization.py\n",
         "GATE_REL=scripts/GPU/alphazero/htb_timing_authorization.py\n"),
        ("GATE=COMBINATION_POLICY_CYCLE2_TRAIN_AUTHORIZED\n", "GATE=HTB_TIMING_AUTHORIZED\n"),
        ("MODULE=scripts.GPU.alphazero.combination_policy_pilot\n", "MODULE=scripts.GPU.alphazero.htb_timing\n"),
        ("TERM_S=2810\n", f"TERM_S={term}\n"),
        ("RECORD_KIND=combination-policy-pilot-record/1\n", f"RECORD_KIND=htb-timing-{kind}-record/1\n"),
        ('  survivors=$(pgrep -f -- "-m ${MODULE} run" | tr \'\\n\' \' \')\n',
         '  survivors=$(pgrep -f -- "-m ${MODULE} time" | tr \'\\n\' \' \')\n'),
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
             'env MLX_ENABLE_TF32=0 PYTHONPATH=. "$TPY" -B -m "$MODULE" time label --clone-root "$CKPT" \\\n'),
            # the user's 2026-11-04 correction: outer.json records the interpreter that runs the timing (the teacher
            # venv); the project interpreter, which runs only the idle and background-compute checks, is kept beside it
            ('PRE=""; HEAD_C=""; PY=""; CKPT=""; start=0\n', 'PRE=""; HEAD_C=""; PY=""; TPY=""; CKPT=""; start=0\n'),
            ('"python": "%s", "checkpoint": "%s"}\\n\' \\\n',
             '"python": "%s", "checks_python": "%s", "clone_root": "%s"}\\n\' \\\n'),
            ('"$W" "$HEAD_C" "$PRE" "$PY" "$CKPT" > "$OUT_REL/outer.json"',
             '"$W" "$HEAD_C" "$PRE" "$TPY" "$PY" "$CKPT" > "$OUT_REL/outer.json"'),
        ]
    return common + [
        ('env MLX_ENABLE_TF32=0 PYTHONPATH=. "$PY" -B -m "$MODULE" run cycle2_train --checkpoint "$CKPT" \\\n',
         'env MLX_ENABLE_TF32=0 PYTHONPATH=. "$PY" -B -m "$MODULE" time fit \\\n'),
    ]


def derive(kind, source_text):
    body = source_text.split("\n", 1)[1]                       # the source minus its shebang
    for old, new in replacements(kind):
        if body.count(old) != 1:
            raise AssertionError(f"{kind}: anchor occurs {body.count(old)} times: {old[:60]!r}")
        body = body.replace(old, new)
    return header(kind) + body
