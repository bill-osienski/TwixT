"""The H-TB confirmation test's wrapper and seed-script copies as exact derivations of the H-TB screen's reviewed files,
the ones that ran (ledger, Post-closure entry 53; the user's review, 2026-11-06: 'reuse the reviewed loader, loop and
controls; add separate paths, a closed gate and focused tests'). Used by the generator (once) and by
tests/test_htb_strength.py (every run). Each copy is: a new header naming its source, then the source from its second
line on (a wrapper) or from inside its docstring (a script), with the replacements below and nothing else."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
EV = "docs/superpowers/evidence"
#: plan §9: ADOPTED by the user's review (2026-11-06) -- the cycle-2 strength comparison's longer-game pace (35,978 s
#: over 1,018 games) x 220 x 1.5, up to the minute: 11,700 s; superseded the proposed 6,600 s (TERM 6,590)
TERM_S = 11690
N_SLOTS = 220
SOURCES = {"wrapper": (f"{EV}/htb-screen-run/01_wrapper.sh",
                       "2d221e31a16f29de0fff7b9be3a46f2f8fbadda8f1145172f97f229a02c4b189"),
           "survey": (f"{EV}/htb-screen-seed-survey/01_seed_survey.py",
                      "cecb6c98f057c5874611167dbc9f7b73ec9c3771bab086b42c95a023ae78b200"),
           "closing": (f"{EV}/htb-screen-seed-closing/01_derive_usage.py",
                       "64e42aa6df21fd1f30ab5e2001b946813a1ba66b176f0a5d10881c7b6d4bdcac")}
TARGETS = {"wrapper": f"{EV}/htb-strength-run/01_wrapper.sh",
           "survey": f"{EV}/htb-strength-seed-survey/01_seed_survey.py",
           "closing": f"{EV}/htb-strength-seed-closing/01_derive_usage.py"}

_PLAN = "docs/superpowers/2026-11-06-htb-strength-plan.md"


def header(kind):
    if kind == "wrapper":
        return (
            "#!/bin/bash\n"
            "# THE H-TB CONFIRMATION TEST: THE WHOLE STUDENT IN THE PROVEN COMBINATION, 110 FRESH COLOUR-REVERSED PAIRS (220 GAMES)\n"
            f"# -- OUTER WRAPPER (plan {_PLAN} §§2-10; card docs/superpowers/2026-11-03-htb-twixtbot-distillation-pilot-card.md\n"
            "# §9.2, §9.3; the user's review 2026-11-06: the same frozen whole student, sha256 0e1cb621..., against the original\n"
            "# proven combination, both averaged, 200 simulations, halving from ply 20; PASS only if mean - sqrt(ln 20 / 220) > 0.5\n"
            "# with evaluations ratio <= 1.05 and search-seconds ratio <= 1.10).\n"
            "# CEILING ADOPTED BY THE USER'S REVIEW (2026-11-06; plan §9), from the historical longer-game pace with the same\n"
            "# 1.5 allowance: the cycle-2 strength comparison's wall time 35,978 s over 1,018 games (63.7 plies, 35.3 s per\n"
            "# game) x 220 = 7,775.2 s; x 1.5 = 11,662.8 s, rounded up to the minute: 11,700 s -- TERM at\n"
            f"# {TERM_S} s, KILL 10 s later, {TERM_S + 10} s in all; cleanup observation bounded through {TERM_S + 20} s. The\n"
            "# expected runtime may still follow the H-TB screen's faster pace (about 4,397 s); the larger ceiling reduces the\n"
            "# risk of another incomplete test. A run the ceiling stops is VOID, with no automatic retry; its progress and cause\n"
            "# go to the user. Idle waiting (idle_gate) is not part of the ceiling.\n"
            "# BUILT ONLY (ledger, 2026-11-06): launching it is a separate approval.\n"
            "# A copy of htb-screen-run/01_wrapper.sh (the H-TB screen's, run once, whose header follows, from its line 2); the\n"
            "# diff: the paths, the gate (HTB_STRENGTH_AUTHORIZED), the module (htb_strength) and its `run strength` command,\n"
            "# the record kind, TERM_S and this header. Everything else is byte-identical: the idle gate, the background-compute\n"
            "# check, the process-group supervisor and its bounded observation, the survivors, the inner-exit table\n"
            "# (0 COMPLETED, 3 VOID, 4 REFUSED, 2 USAGE), the records and the gate restoration.\n"
            "#\n")
    if kind == "survey":
        return (
            '"""THE H-TB CONFIRMATION TEST -- THE 220-SLOT GAME-SEED SURVEY AND ITS COLLISION PROOFS, IN THE 2027 BAND (read-only).\n'
            "\n"
            f"Plan {_PLAN} §8 (`htb_strength run strength`: 110 colour-reversed pairs; game g plays seed BASE + g). BUILT\n"
            "ONLY (ledger, 2026-11-06): running any mode is a separate approval. A copy of\n"
            "evidence/htb-screen-seed-survey/01_seed_survey.py (the H-TB screen's, run once per mode, whose description follows\n"
            "from its first line); the diff: the block size (220), the paths, the plan's kind, the module name, the titles,\n"
            "the slot heading and verdict lines. The method, the rule (`seed_band_2027.candidate`; gap floor = the block's own\n"
            "size), the scope, the controls and the three modes are unchanged. Read the copied description's 64 as 220 and its\n"
            "screen games as this test's games. A block the check finds NOT CLEAN is returned for review, never adjusted.\n"
            "\n")
    return (
        '"""THE H-TB CONFIRMATION TEST -- THE SEED-USAGE DERIVATION for closing (read-only, outcome-blind).\n'
        "\n"
        f"Plan {_PLAN} §8 step 5 (`htb_strength run strength`, 220 games). BUILT ONLY (ledger, 2026-11-06): running it\n"
        "is a separate approval. A copy of evidence/htb-screen-seed-closing/01_derive_usage.py (the H-TB screen's, run once,\n"
        "whose description follows from its first line); the diff: the block size (220), the paths and the title. The rule\n"
        "is unchanged: the test plays its games in order in one process and logs `game_start` before each, so only the\n"
        "started prefix is exposed and the whole block is retired. Read the copied description's 64 as 220.\n"
        "\n")


def replacements(kind):
    if kind == "wrapper":
        return [
            ("OUT_REL=docs/superpowers/evidence/htb-screen-run\n", "OUT_REL=docs/superpowers/evidence/htb-strength-run\n"),
            ("RUN_REL=docs/superpowers/evidence/htb-screen\n", "RUN_REL=docs/superpowers/evidence/htb-strength\n"),
            ("GATE_REL=scripts/GPU/alphazero/htb_screen_authorization.py\n",
             "GATE_REL=scripts/GPU/alphazero/htb_strength_authorization.py\n"),
            ("GATE=HTB_SCREEN_AUTHORIZED\n", "GATE=HTB_STRENGTH_AUTHORIZED\n"),
            ("MODULE=scripts.GPU.alphazero.htb_screen\n", "MODULE=scripts.GPU.alphazero.htb_strength\n"),
            ("TERM_S=3410\n", f"TERM_S={TERM_S}\n"),
            ("RECORD_KIND=htb-screen-record/1\n", "RECORD_KIND=htb-strength-record/1\n"),
            ('-m "$MODULE" run screen --checkpoint "$CKPT"', '-m "$MODULE" run strength --checkpoint "$CKPT"'),
        ]
    if kind == "survey":
        return [
            ("N_SLOTS = 64\n", f"N_SLOTS = {N_SLOTS}\n"),
            ('HERE = "docs/superpowers/evidence/htb-screen-seed-survey"\n',
             'HERE = "docs/superpowers/evidence/htb-strength-seed-survey"\n'),
            ('PLAN = "docs/superpowers/evidence/htb-screen-run/plan.json"\n',
             'PLAN = "docs/superpowers/evidence/htb-strength-run/plan.json"\n'),
            ('VERDICT_OK = "THE H-TB SCREEN BLOCK IS CLEAN"\n', 'VERDICT_OK = "THE H-TB CONFIRMATION-TEST BLOCK IS CLEAN"\n'),
            ('f"_htb_screen_{key}"', 'f"_htb_strength_{key}"'),
            ('"kind": "htb-screen-plan/1"', '"kind": "htb-strength-plan/1"'),
            ('faults.append("the H-TB screen plan is absent or not committed")',
             'faults.append("the H-TB confirmation-test plan is absent or not committed")'),
            ('faults.append("the H-TB screen plan is not build_plan of the surveyed block")',
             'faults.append("the H-TB confirmation-test plan is not build_plan of the surveyed block")'),
            ('say(f"H-TB PILOT -- SCREEN GAME-SEED {', 'say(f"H-TB PILOT -- CONFIRMATION-TEST GAME-SEED {'),
            ('say("SLOTS (htb_screen run screen: game g plays seed BASE + g, in order, in one process):")',
             'say("SLOTS (htb_strength run strength: game g plays seed BASE + g, in order, in one process):")'),
            ("'THE H-TB SCREEN BLOCK IS CLEAN AND THE STATE IS AS EXPECTED'",
             "'THE H-TB CONFIRMATION-TEST BLOCK IS CLEAN AND THE STATE IS AS EXPECTED'"),
        ]
    return [
        ('HERE = "docs/superpowers/evidence/htb-screen-seed-closing"\n',
         'HERE = "docs/superpowers/evidence/htb-strength-seed-closing"\n'),
        ('RUN = "docs/superpowers/evidence/htb-screen"\n', 'RUN = "docs/superpowers/evidence/htb-strength"\n'),
        ('RUN_DIR = "docs/superpowers/evidence/htb-screen-run"\n',
         'RUN_DIR = "docs/superpowers/evidence/htb-strength-run"\n'),
        ("N_SLOTS = 64\n", f"N_SLOTS = {N_SLOTS}\n"),
        ('"H-TB PILOT -- SCREEN SEED-USAGE DERIVATION (closing)"',
         '"H-TB PILOT -- CONFIRMATION-TEST SEED-USAGE DERIVATION (closing)"'),
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
