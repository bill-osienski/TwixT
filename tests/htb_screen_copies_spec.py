"""The H-TB screen's wrapper and seed-script copies as exact derivations of the cycle-2 screen's reviewed files (the
user's review, 2026-11-06: 'reuse the reviewed match loop, seed procedure and wrapper pattern'). Used by the generator
(once) and by tests/test_htb_screen.py (every run). Each copy is: a new header naming its source, then the source from
its second line on, with the replacements below and nothing else."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
EV = "docs/superpowers/evidence"
#: the user's review, 2026-11-06: TERM is the plan §8 proposal, PROPOSED for the launch review (not adopted)
TERM_S = 3410
SOURCES = {"wrapper": (f"{EV}/combination-policy-cycle2-screen-run/01_wrapper.sh",
                       "a21fb8ea4268642191e6567e0465ed331f4cad5c5420fb2124238b794a53ca7d"),
           "survey": (f"{EV}/combination-policy-cycle2-screen-seed-survey/01_seed_survey.py",
                      "2268da4bf164ebca0f409410c67c9f1ceb7f3b484e0110be5e2058ae6854c49e"),
           "closing": (f"{EV}/combination-policy-cycle2-screen-seed-closing/01_derive_usage.py",
                       "b1e6295478b76c950d2689a06e51f7307f63bbf7164189ebb6956ef745ef5717")}
TARGETS = {"wrapper": f"{EV}/htb-screen-run/01_wrapper.sh",
           "survey": f"{EV}/htb-screen-seed-survey/01_seed_survey.py",
           "closing": f"{EV}/htb-screen-seed-closing/01_derive_usage.py"}

_PLAN = "docs/superpowers/2026-11-06-htb-screen-plan.md"


def header(kind):
    if kind == "wrapper":
        return (
            "#!/bin/bash\n"
            "# THE H-TB SCREEN: THE WHOLE STUDENT IN THE PROVEN COMBINATION, 32 COLOUR-REVERSED PAIRS (64 GAMES) -- OUTER WRAPPER\n"
            f"# (plan {_PLAN} §§2-9; card docs/superpowers/2026-11-03-htb-twixtbot-distillation-pilot-card.md §9.1, §9.3;\n"
            "# the user's review 2026-11-06: the whole student, sha256 0e1cb621..., against the original proven combination, both\n"
            "# averaged, 200 simulations, halving from ply 20; STOP below a mean pair score of 0.50 or on either compute guard).\n"
            "# CEILING PROPOSED FOR THE LAUNCH REVIEW (plan §8): 0.5559 s per candidate search (the cycle-2 strength comparison's\n"
            "# candidate searches, Python 3.14.10) x 63.725 plies per game (its 1,018 games) x 64 games = 2,267.0 s; x 1.5, rounded up\n"
            f"# to the minute: 3,420 s -- TERM at {TERM_S} s, KILL 10 s later, {TERM_S + 10} s in all; cleanup observation bounded through\n"
            f"# {TERM_S + 20} s. An estimate: the student's game lengths and search cost are unmeasured. A run the ceiling stops is VOID,\n"
            "# with no automatic retry; its progress and cause go to the user. Idle waiting (idle_gate) is not part of the ceiling.\n"
            "# BUILT ONLY (ledger, 2026-11-06): launching it is a separate approval.\n"
            "# A copy of combination-policy-cycle2-screen-run/01_wrapper.sh (the cycle-2 screen's, whose header follows, from its\n"
            "# line 2); the diff: the paths, the gate (HTB_SCREEN_AUTHORIZED), the module (htb_screen) and its `run screen`\n"
            "# command, the record kind, TERM_S and this header. Everything else is byte-identical: the idle gate, the\n"
            "# background-compute check, the process-group supervisor and its bounded observation, the survivors, the inner-exit\n"
            "# table (0 COMPLETED, 3 VOID, 4 REFUSED, 2 USAGE), the records and the gate restoration.\n"
            "#\n")
    if kind == "survey":
        return (
            '"""THE H-TB SCREEN -- THE 64-SLOT GAME-SEED SURVEY AND ITS COLLISION PROOFS, IN THE 2027 BAND (read-only).\n'
            "\n"
            f"Plan {_PLAN} §7 (`htb_screen run screen`). BUILT ONLY (ledger, 2026-11-06): running any mode is a separate\n"
            "approval. A copy of evidence/combination-policy-cycle2-screen-seed-survey/01_seed_survey.py (the cycle-2 screen's, run\n"
            "once per mode, whose description follows from its first line); the diff: the paths, the plan's kind, the module name,\n"
            "the titles, the slot heading and verdict lines. The block size, method, rule, scope, controls and three modes are\n"
            "unchanged. A block the check finds NOT CLEAN is returned for review, never adjusted.\n"
            "\n")
    return (
        '"""THE H-TB SCREEN -- THE SEED-USAGE DERIVATION for closing (read-only, outcome-blind).\n'
        "\n"
        f"Plan {_PLAN} §7 step 5 (`htb_screen run screen`, 64 games). BUILT ONLY (ledger, 2026-11-06): running it is a\n"
        "separate approval. A copy of evidence/combination-policy-cycle2-screen-seed-closing/01_derive_usage.py (the cycle-2\n"
        "screen's, run once, whose description follows from its first line); the diff: the paths and the title. The block\n"
        "size and the rule are unchanged: the screen plays its games in order in one process and logs `game_start` before each.\n"
        "\n")


def replacements(kind):
    if kind == "wrapper":
        return [
            ("OUT_REL=docs/superpowers/evidence/combination-policy-cycle2-screen-run\n",
             "OUT_REL=docs/superpowers/evidence/htb-screen-run\n"),
            ("RUN_REL=docs/superpowers/evidence/combination-policy-cycle2-screen\n",
             "RUN_REL=docs/superpowers/evidence/htb-screen\n"),
            ("GATE_REL=scripts/GPU/alphazero/combination_policy_cycle2_screen_authorization.py\n",
             "GATE_REL=scripts/GPU/alphazero/htb_screen_authorization.py\n"),
            ("GATE=COMBINATION_POLICY_CYCLE2_SCREEN_AUTHORIZED\n", "GATE=HTB_SCREEN_AUTHORIZED\n"),
            ("MODULE=scripts.GPU.alphazero.combination_policy_pilot\n", "MODULE=scripts.GPU.alphazero.htb_screen\n"),
            ("TERM_S=3650\n", f"TERM_S={TERM_S}\n"),
            ("RECORD_KIND=combination-policy-pilot-record/1\n", "RECORD_KIND=htb-screen-record/1\n"),
            ('-m "$MODULE" run cycle2_screen --checkpoint "$CKPT"', '-m "$MODULE" run screen --checkpoint "$CKPT"'),
        ]
    if kind == "survey":
        return [
            ('HERE = "docs/superpowers/evidence/combination-policy-cycle2-screen-seed-survey"\n',
             'HERE = "docs/superpowers/evidence/htb-screen-seed-survey"\n'),
            ('PLAN = "docs/superpowers/evidence/combination-policy-cycle2-screen-run/plan.json"\n',
             'PLAN = "docs/superpowers/evidence/htb-screen-run/plan.json"\n'),
            ('VERDICT_OK = "THE COMBINATION-POLICY CYCLE-2 SCREEN BLOCK IS CLEAN"\n',
             'VERDICT_OK = "THE H-TB SCREEN BLOCK IS CLEAN"\n'),
            ('f"_combination_policy_cycle2_screen_{key}"', 'f"_htb_screen_{key}"'),
            ('"kind": "combination-policy-cycle2-screen-plan/1"', '"kind": "htb-screen-plan/1"'),
            ('faults.append("the cycle-2 screen plan is absent or not committed")',
             'faults.append("the H-TB screen plan is absent or not committed")'),
            ('faults.append("the cycle-2 screen plan is not build_plan of the surveyed block")',
             'faults.append("the H-TB screen plan is not build_plan of the surveyed block")'),
            ('say(f"COMBINATION POLICY PILOT -- CYCLE-2 SCREEN GAME-SEED {',
             'say(f"H-TB PILOT -- SCREEN GAME-SEED {'),
            ('say("SLOTS (combination_policy_pilot run cycle2_screen: game g plays seed BASE + g, in order, in one process):")',
             'say("SLOTS (htb_screen run screen: game g plays seed BASE + g, in order, in one process):")'),
            ("'THE COMBINATION-POLICY CYCLE-2 SCREEN BLOCK IS CLEAN AND THE STATE IS AS EXPECTED'",
             "'THE H-TB SCREEN BLOCK IS CLEAN AND THE STATE IS AS EXPECTED'"),
        ]
    return [
        ('HERE = "docs/superpowers/evidence/combination-policy-cycle2-screen-seed-closing"\n',
         'HERE = "docs/superpowers/evidence/htb-screen-seed-closing"\n'),
        ('RUN = "docs/superpowers/evidence/combination-policy-cycle2-screen"\n',
         'RUN = "docs/superpowers/evidence/htb-screen"\n'),
        ('RUN_DIR = "docs/superpowers/evidence/combination-policy-cycle2-screen-run"\n',
         'RUN_DIR = "docs/superpowers/evidence/htb-screen-run"\n'),
        ('"COMBINATION POLICY PILOT -- CYCLE-2 SCREEN SEED-USAGE DERIVATION (closing)"',
         '"H-TB PILOT -- SCREEN SEED-USAGE DERIVATION (closing)"'),
    ]


def derive(kind, source_text):
    """The header, then the source from its second line (a wrapper) or its first (a script: its own docstring's first
    line follows this header's, inside the same docstring), with every anchor replaced exactly once."""
    if kind == "wrapper":
        body = source_text.split("\n", 1)[1]
    else:
        if not source_text.startswith('"""'):
            raise AssertionError(f"{kind}: the source does not open with its docstring")
        body = source_text[3:]
    for old, new in replacements(kind):
        if body.count(old) != 1:
            raise AssertionError(f"{kind}: anchor occurs {body.count(old)} times: {old[:60]!r}")
        body = body.replace(old, new)
    return header(kind) + body
