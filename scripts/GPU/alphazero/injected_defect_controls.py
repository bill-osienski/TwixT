"""THE INJECTED-DEFECT CONTROL LIST -- definitions only, and NOTHING that runs.

Each control is `(label, file, anchor, replacement, test node)`: apply the defect
to the SOURCE, run the test that must catch it, and require that test to FAIL --
for the reason `EXPECTED_REASONS[label]` names. A control that never rejects has
not been shown to bind; a control that fails for some OTHER reason has not been
shown to bind what it CLAIMS to bind; and a control whose anchor no longer
matches has NOT RUN.

🔴 WHY THIS FILE HOLDS NO DRIVER (INCIDENT 2, 2026-09-12).
The previous harness put its clean baseline, its injection loop and its restores
at module level behind `if __name__ != "__main__": raise ImportError`. That guard
does not fire in a namespace where `__name__` is ALREADY `"__main__"` -- which is
exactly what `exec(path.read_text())` gives you. Reading the control list STARTED
the harness: for seven minutes it injected defects into `h1_viability_rules.py`
and `h1_viability_runner.py` while other tests ran against the tree it was
mutating, and its restore then wiped an unrelated repair.

So the guard is gone and the reason for it with it. This module is DATA. Import
it, `exec` it, read it however you like -- there is nothing here to start. The
driver lives in `run_injected_defect_controls.py`, is invoked explicitly, and
does its mutating inside a disposable `git worktree`, never in the shared tree.
"""
PROBE = "scripts/GPU/alphazero/d1_probe.py"
SEL = "scripts/GPU/alphazero/d1_selection.py"
G3 = "scripts/GPU/alphazero/twixtbot_g3_reference.py"
ADAPTER = "scripts/GPU/alphazero/t1j_adapter.py"
INTEG = "scripts/GPU/alphazero/e4_screen_integration.py"
D0 = "scripts/GPU/alphazero/d0_postmortem.py"
REF_SRC = "scripts/GPU/alphazero/e4_screen_reference.py"
LP = "scripts/GPU/alphazero/lowply_qualification.py"
H1R = "scripts/GPU/alphazero/h1_viability_rules.py"
H1RUN = "scripts/GPU/alphazero/h1_viability_runner.py"
VTR = "scripts/GPU/alphazero/void_trace.py"
H1P = "scripts/GPU/alphazero/h1_viability_plan.py"
L0R = "scripts/GPU/alphazero/l0_match_rules.py"
RQ = "scripts/GPU/alphazero/runtime_requalification.py"
CMD = "scripts/GPU/alphazero/h1_match_command.py"
DP = "scripts/GPU/alphazero/d1prime_analysis.py"
DS = "scripts/GPU/alphazero/d1second_analysis.py"
H2R = "scripts/GPU/alphazero/h2_match_rules.py"
H2P = "scripts/GPU/alphazero/h2_match_plan.py"
H2RUN = "scripts/GPU/alphazero/h2_match_runner.py"
H2CMD = "scripts/GPU/alphazero/h2_match_command.py"
G3SRC = "scripts/GPU/alphazero/twixtbot_g3_reference.py"
H3R = "scripts/GPU/alphazero/h3_pilot_rules.py"
H3A = "scripts/GPU/alphazero/h3_pilot_analysis.py"
H3RUN = "scripts/GPU/alphazero/h3_pilot_runner.py"
H3CMD = "scripts/GPU/alphazero/h3_pilot_command.py"

T_PROBE = "tests/test_d1_probe.py"
T_SEL = "tests/test_d1_selection.py"
T_INC = "tests/test_d1_incumbent.py"
T_ADP = "tests/test_t1j_adapter.py"
T_INTEG = "tests/test_e4_screen_integration.py"
T_D0 = "tests/test_d0_postmortem.py"
T_LP = "tests/test_lowply_qualification.py"
T_H1 = "tests/test_h1_viability.py"
T_H1R = "tests/test_h1_runner.py"
T_RQ = "tests/test_runtime_requalification.py"
T_CMD = "tests/test_h1_match_command.py"
T_DP = "tests/test_d1prime_analysis.py"
T_DS = "tests/test_d1second_analysis.py"
T_H2 = "tests/test_h2_match.py"
T_H1V = "tests/test_h1_viability.py"
T_H3O = "tests/test_h3_pilot_openings.py"
T_H3A = "tests/test_h3_pilot_analysis.py"
T_H3RUN = "tests/test_h3_pilot_runner.py"

# ── the H3 FULL STUDY
H3SR = "scripts/GPU/alphazero/h3_study_rules.py"
H3SA = "scripts/GPU/alphazero/h3_study_analysis.py"
H3SRUN = "scripts/GPU/alphazero/h3_study_runner.py"
H3SGEN = "scripts/GPU/alphazero/h3_study_generator.py"
T_H3SR = "tests/test_h3_study_rules.py"
T_H3SA = "tests/test_h3_study_analysis.py"
T_H3SRUN = "tests/test_h3_study_runner.py"
H3SCMD = "scripts/GPU/alphazero/h3_study_command.py"
H3SPF = "scripts/GPU/alphazero/h3_generation_preflight.py"
H3GCMD = "scripts/GPU/alphazero/h3_generation_command.py"
#: AMENDMENT 3 -- the uniform population writer, the retired path, the gates
H3OLD = "scripts/GPU/alphazero/h3_coproduced_generator_retired.py"
GINV = "scripts/GPU/alphazero/gate_inventory.py"
H3FCMD = "scripts/GPU/alphazero/h3_freeze_command.py"
H3PROTO = "scripts/GPU/alphazero/h3_generation_protocol.py"
SCREEN = "scripts/GPU/alphazero/e4_screen_reference.py"
T_H3POP = "tests/test_h3_uniform_population.py"
T_H3OLD = "tests/test_h3_coproduced_retired.py"
T_GATES = "tests/test_gate_inventory.py"
H3PRE = "scripts/GPU/alphazero/h3_study_prerun_verification.py"
DRV = "scripts/GPU/alphazero/run_injected_defect_controls.py"
T_DRV = "tests/test_injected_defect_controls.py"
H3FIN = "scripts/GPU/alphazero/h3_final_state.py"
H3CMB = "scripts/GPU/alphazero/h3_combine.py"
T_H3CMB = "tests/test_h3_combine.py"
H3CCMD = "scripts/GPU/alphazero/h3_combine_command.py"
T_H3FIN = "tests/test_h3_final_state.py"

# (label, file, anchor, replacement, test node)
DEFECTS = [
    #: 🔴 THE ALLOW-LIST REPAIR (2026-09-20). The combination refused
    #: ITSELF: verify_final_state rejected any open gate, including the one
    #: combine() requires open. The permission must stay as narrow as the act.
    ('the allow-list permits every open gate instead of the named ones', H3FIN,
     '    allowed = {tuple(g) for g in (allow_open or ())}',
     '    allowed = {tuple(g) for g in INVENTORY.open_gates()}',
     f"{T_H3CMB}::test_THE_PUBLIC_ENTRY_STILL_REFUSES_ANY_OTHER_OPEN_GATE"),
    ('the final-state check stops rejecting unexpected open gates', H3FIN,
     '    unexpected = [g for g in open_gates if tuple(g) not in allowed]',
     '    unexpected = []',
     f"{T_H3CMB}::test_AN_OPEN_GATE_IS_CAUGHT_FINAL"),
    ('the combiner allows ANY gate to be open, not just its own', H3CMB,
     'THIS_GATE = (("h3_combine", "H3_COMBINATION_AUTHORIZED"),)',
     'THIS_GATE = ()',
     f"{T_H3CMB}::test_THE_PUBLIC_ENTRY_ACCEPTS_ITS_OWN_OPEN_GATE"),
    ('the restoration regex goes back to the permissive form', H3CCMD,
     '_GATE_OPEN = re.compile(r"^H3_COMBINATION_AUTHORIZED = True$", re.M)',
     '_GATE_OPEN = re.compile(r"^H3_COMBINATION_AUTHORIZED\\s*=\\s*True\\s*$", re.M)',
     f"{T_H3CMB}::test_RESTORATION_RETURNS_THE_SOURCE_BYTE_FOR_BYTE"),
    ("attempt 2 is aimed at attempt 1's SPENT destination", H3CMB,
     'COMBINED_OUT_DIR = f"{RUN.OUT_ROOT}/2026-09-20-t1j-h3-study-combined-attempt2"',
     'COMBINED_OUT_DIR = f"{RUN.OUT_ROOT}/2026-09-20-t1j-h3-study-combined"',
     f"{T_H3CMB}::test_ATTEMPT_1S_DESTINATION_IS_SPENT_AND_ATTEMPT_2S_IS_FRESH"),
    #: 🔴 THE COMBINATION CLOSEOUT (2026-09-20). The study's answer now
    #: exists as a file; these prove it is still the file that was written,
    #: and that the verdict cannot be stored without its scope.
    ('the combined outputs are never re-hashed against their pins', H3FIN,
     '        got = hashlib.sha256(open(path, "rb").read()).hexdigest()\n        if got != pin:',
     '        got = pin\n        if got != pin:',
     f"{T_H3CMB}::test_THE_COMBINED_OUTPUTS_ARE_REHASHED_NOT_READ_BACK"),
    ('a pinned combined output may be MISSING', H3FIN,
     '        if not os.path.lexists(path):\n            out.append(f"the {label} is pinned but MISSING at {path}")',
     '        if False:\n            out.append(f"the {label} is pinned but MISSING at {path}")',
     f"{T_H3CMB}::test_THE_COMBINED_OUTPUTS_ARE_REHASHED_NOT_READ_BACK"),
    ("the combined report's digest pin is blanked", H3CMB,
     'COMBINED_REPORT_DIGEST = (\n    "252f25ca358f4a814b22f394f3a7efee0b3b276af93866071c03a6078551ed41")',
     'COMBINED_REPORT_DIGEST = None',
     f"{T_H3CMB}::test_THE_COMBINATION_HAS_RUN_AND_ITS_GATE_IS_SHUT_AGAIN"),
    ('the verdict is persisted without its population claim', H3CMB,
     '        "population": report["population"],',
     '        "population": {"stratum": report["population"]["stratum"]},',
     f"{T_H3CMB}::test_THE_COMBINER_PERSISTS_THE_CLAIM_IT_IS_GIVEN"),
    # ═════════ THE COMBINATION (2026-09-20) ════════════════════════════
    # 🔴 POOLING THE FOUR SEGMENTS IS THE STUDY'S ONE INTERPRETIVE ACT.
    # After it a number exists that people will quote, so every refusal on
    # the way there is proved to bind.
    ('the combination gate is not checked at all', H3CMB,
     '    if not H3_COMBINATION_AUTHORIZED:',
     '    if False:',
     f"{T_H3CMB}::test_combine_REFUSES_WHILE_THE_GATE_IS_SHUT"),
    ('the authorized entry point accepts a destination override', H3CMB,
     'def combine() -> Dict[str, Any]:',
     'def combine(out_dir: str = COMBINED_OUT_DIR) -> Dict[str, Any]:',
     f"{T_H3CMB}::test_combine_TAKES_NO_ARGUMENTS_AT_ALL"),
    ('the combination ignores the final-state disagreements', H3CMB,
     '    if problems:',
     '    if False:',
     f"{T_H3CMB}::test_A_FINAL_STATE_DISAGREEMENT_BLOCKS_AGGREGATION"),
    ('the combination stops counting the games', H3CMB,
     '    if len(task_ids) != RULES.N_GAMES or len(set(task_ids)) != RULES.N_GAMES:',
     '    if False:',
     f"{T_H3CMB}::test_A_MISSING_GAME_IS_REFUSED"),
    ('the combination stops requiring DISTINCT task ids', H3CMB,
     '    task_ids = [g["task_id"] for g in games]',
     '    task_ids = sorted(set(g["task_id"] for g in games))',
     f"{T_H3CMB}::test_A_DUPLICATED_GAME_IS_REFUSED"),
    ('the combination stops requiring the seeds to be the four blocks', H3CMB,
     '    if len(set(seeds)) != RULES.N_GAMES or set(seeds) != expected_seeds:',
     '    if False:',
     f"{T_H3CMB}::test_SEEDS_THAT_ARE_NOT_THE_FOUR_BLOCKS_ARE_REFUSED"),
    ('the combination accepts a pair that is not a pair', H3CMB,
     '    short = sorted(p for p, n in by_pair.items() if n != 2)',
     '    short = []',
     f"{T_H3CMB}::test_AN_INCOMPLETE_PAIR_IS_REFUSED"),
    ('the combination ignores the permission decision', H3CMB,
     '    if not permission.get("verdict_permitted"):',
     '    if False:',
     f"{T_H3CMB}::test_A_REFUSED_PERMISSION_STOPS_THE_COMBINATION"),
    ('the combination accepts a narrower analysis', H3CMB,
     '    missing = [s for s in REQUIRED_SENSITIVITIES if s not in report["sensitivities"]]',
     '    missing = []',
     f"{T_H3CMB}::test_AN_OMITTED_SENSITIVITY_IS_REFUSED"),
    ('the combination stops hashing its inputs', H3CMB,
     'def _sha256(path: str) -> str:\n    h = hashlib.sha256()',
     'def _sha256(path: str) -> str:\n    return "0" * 64\n    h = hashlib.sha256()',
     f"{T_H3CMB}::test_EVERY_INPUT_IS_RECORDED_BY_PATH_AND_HASH"),
    #: retargeted 2026-09-20: `_occupied` was extracted so the EARLY check and
    #: the ATOMIC install could be broken apart. While they shared one message,
    #: removing the early one changed nothing observable and no control bound it.
    ('the early occupied-destination check is removed', H3CMB,
     '    return os.path.lexists(path)',
     '    return False',
     f"{T_H3CMB}::test_THE_DESTINATION_IS_CREATE_ONLY"),
    #: 🔴 THE DURABILITY REPAIR AND THE SUPERVISED COMMAND (2026-09-20).
    #: The first writer opened the OFFICIAL path and serialised into it,
    #: cleaning up in an `except` that a SIGKILL never runs; and the gate had
    #: no wrapper to close it. These prove both replacements bind.
    ('the report is serialised straight into the official path', H3CMB,
     '    tmp = os.path.join(directory, f".{os.path.basename(path)}.{os.getpid()}.tmp")',
     '    tmp = path',
     f"{T_H3CMB}::test_THE_OFFICIAL_PATH_IS_NEVER_OPENED_FOR_WRITING"),
    ('the temporary file is left beside the report', H3CMB,
     '    finally:\n        try:\n            os.unlink(tmp)\n        except OSError:\n            pass',
     '    finally:\n        pass',
     f"{T_H3CMB}::test_THE_TEMPORARY_FILE_IS_IN_THE_SAME_DIRECTORY_AND_IS_REMOVED"),
    ('the install OVERWRITES instead of refusing an existing name', H3CMB,
     '            os.link(tmp, path)',
     '            os.replace(tmp, path)',
     f"{T_H3CMB}::test_THE_LINK_IS_A_BARRIER_IN_ITS_OWN_RIGHT"),
    ('the payload is not fsynced before it is installed', H3CMB,
     '            fh.flush()\n            os.fsync(fh.fileno())',
     '            fh.flush()',
     f"{T_H3CMB}::test_THE_PAYLOAD_AND_THE_DIRECTORY_ARE_BOTH_FSYNCED"),
    ('the directory is not fsynced after the link', H3CMB,
     '        dfd = os.open(directory, os.O_RDONLY)\n        try:\n            os.fsync(dfd)\n        finally:\n            os.close(dfd)',
     '        pass',
     f"{T_H3CMB}::test_THE_PAYLOAD_AND_THE_DIRECTORY_ARE_BOTH_FSYNCED"),
    ('the combination command never restores its gate', H3CCMD,
     '        restored = restore_gate(_combine_source)',
     '        restored = True',
     f"{T_H3CMB}::test_A_FAILING_COMBINATION_STILL_RESTORES_THE_GATE"),
    ('a failed restoration no longer supersedes the run outcome', H3CCMD,
     '    if not restored or readback != "False":',
     '    if False:',
     f"{T_H3CMB}::test_A_FAILED_RESTORATION_SUPERSEDES_A_SUCCESSFUL_COMBINATION"),
    ('the receipt records the run code instead of the superseding one', H3CCMD,
     '            "exit_code": status if restored else EXIT_GATE_NOT_RESTORED,',
     '            "exit_code": status,',
     f"{T_H3CMB}::test_A_FAILED_RESTORATION_SUPERSEDES_A_SUCCESSFUL_COMBINATION"),
    ('the combination receipt is not create-only', H3CCMD,
     '        if os.path.lexists(_receipt):\n            return False\n        os.makedirs(os.path.dirname(_receipt), exist_ok=True)\n        fd = os.open(_receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)',
     '        os.makedirs(os.path.dirname(_receipt), exist_ok=True)\n        fd = os.open(_receipt, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)',
     f"{T_H3CMB}::test_THE_RECEIPT_IS_CREATE_ONLY"),
    ('an unauthorized combination writes a receipt and occupies the destination', H3CCMD,
     '        print("the H3 COMBINATION is NOT AUTHORIZED "',
     '        write_receipt({}, _receipt)\n        print("the H3 COMBINATION is NOT AUTHORIZED "',
     f"{T_H3CMB}::test_A_SHUT_GATE_REFUSES_AND_WRITES_NO_RECEIPT"),
    ('the combination command accepts a destination override', H3CCMD,
     '    ap.add_argument("--run", action="store_true",',
     '    ap.add_argument("--out", default=None)\n    ap.add_argument("--run", action="store_true",',
     f"{T_H3CMB}::test_THE_COMMAND_ACCEPTS_NOTHING_BUT_run"),
    ('summarise is called once PER SEGMENT instead of once on the pool', H3CMB,
     '    report = summarise(games, total_elapsed_s=total_elapsed)',
     '    for _k in range(len(blocks)):\n        report = summarise([g for g in games if g["segment"] == _k],\n                           total_elapsed_s=total_elapsed)\n    report = summarise(games, total_elapsed_s=total_elapsed)',
     f"{T_H3CMB}::test_summarise_IS_CALLED_EXACTLY_ONCE"),
    # ═════════ THE FINAL-STATE CROSS-CHECK (2026-09-20) ════════════════
    # 🔴 EACH SEGMENT VERIFIED ITSELF; NOTHING VERIFIED THE SET. These prove
    # the cross-check binds each thing it claims to compare.
    ("the final-state check stops comparing the receipt fields", H3FIN,
     '        if got != value or type(got) is not type(value):\n'
     '            out.append(f"segment {k} receipt: {key} is {got!r}, expected {value!r}")',
     "        pass",
     f"{T_H3FIN}::test_A_RECEIPT_THAT_DID_NOT_COMPLETE_IS_CAUGHT"),
    ("the final-state check reads seeds from the PLAN, not the records", H3FIN,
     '    seeds = sorted(r["seed"] for r in rows if "seed" in r)',
     "    seeds = list(range(block[0], block[1]))",
     f"{T_H3FIN}::test_A_SEED_OUTSIDE_THE_BLOCK_IS_CAUGHT"),
    ("the final-state check stops requiring the artifacts to exist", H3FIN,
     '            if not os.path.lexists(path):',
     "            if False:",
     f"{T_H3FIN}::test_A_MISSING_ARTIFACT_IS_CAUGHT"),
    ("the final-state check ignores a fired degeneracy gate", H3FIN,
     '        if gate.get("status") != "CLEAR":',
     '        if False:',
     f"{T_H3FIN}::test_A_REPORT_WITH_A_FIRED_DEGENERACY_GATE_IS_CAUGHT"),
    ("the final-state check stops looking at the gate inventory", H3FIN,
     "    open_gates = INVENTORY.open_gates()",
     "    open_gates = []",
     f"{T_H3FIN}::test_AN_OPEN_GATE_IS_CAUGHT"),
    ("the final-state check stops requiring a spent block to be RETIRED", H3FIN,
     '        if not all(x["retired"] for x in st):\n'
     '            out.append(f"segment {k}: block is not wholly RETIRED")',
     "        pass",
     f"{T_H3FIN}::test_A_BLOCK_THAT_IS_NOT_RETIRED_IS_CAUGHT"),
    # ═════════ SEGMENT 1 CLOSEOUT (ran 2026-09-19) ═════════════════════
    # 🔴 FIVE TESTS NOW DERIVE THE SPENT SET instead of listing it, because a
    # listed one was wrong the day segment 1 finished. If the derivation stops
    # reading the registry, the tripwire that pins the concrete state must fire.
    ("the spent/unspent partition stops reading the registry", T_H3POP,
     '        bucket = spent if any(x["exposed"] or x["retired"] or x["test_only"]\n'
     '                              for x in st) else unspent',
     "        bucket = unspent",
     f"{T_H3POP}::test_WHERE_THE_STUDY_STANDS"),
    # ═════════ THE HARNESS'S OWN REASONS MUST NOT DRIFT WITH LOAD ══════════
    ("an ABSOLUTE python path is left in the reason", DRV,
     '    (re.compile(r"/[^\\s\'\\"]*/bin/python[0-9.]*"), "<python>"),',
     "    (re.compile(r'ZZZ_NEVER_MATCHES'), '<python>'),",
     f"{T_DRV}::test_stable_NORMALISES_ABSOLUTE_PATHS_SO_ANOTHER_CHECKOUT_MATCHES"),
    ("an ABSOLUTE home path is left in the reason", DRV,
     '    (re.compile(r"/(?:Users|home)/[^/\\s\'\\"]+/"), "<home>/"),',
     "    (re.compile(r'ZZZ_NEVER_MATCHES/'), '<home>/'),",
     f"{T_DRV}::test_stable_NORMALISES_ABSOLUTE_PATHS_SO_ANOTHER_CHECKOUT_MATCHES"),
    ("a PID is left in the temp-file name the reason is compared against", DRV,
     '    (re.compile(r"\\.\\d+\\.tmp\\b"), ".<pid>.tmp"),',
     '    (re.compile(r"ZZZ_NEVER_MATCHES"), ".<pid>.tmp"),',
     f"{T_DRV}::test_stable_NORMALISES_A_PID_IN_A_TEMPORARY_FILE_NAME"),
    ("a MEASURED duration is left in the reason it is compared against", DRV,
     '    (re.compile(r"(expired after )\\d+(?:\\.\\d+)?s"), r"\\1<elapsed>s"),',
     '    (re.compile(r"(expired after )ZZZ_NEVER_MATCHES"), r"\\1<elapsed>s"),',
     f"{T_DRV}::test_stable_NORMALISES_A_MEASURED_DURATION_BUT_NOT_A_CONFIGURED_ONE"),
    # ═════════ PROSE THAT WENT STALE (2026-09-19) ══════════════════════════
    # 🔴 A HEADER AND A --help LINE ARE CLAIMS. Both survived the repairs that
    # falsified them: the runner's barrier 2 named a deleted constant, and the
    # CLI told every reader the study had no seed block and no population.
    ("the runner's header names a barrier constant that is gone", H3SRUN,
     "  2. The seeds are checked PER SEGMENT.",
     "  2. `STUDY_SEED_BLOCK` is None, so nothing is registered.",
     f"{T_H3SRUN}::test_THE_RUNNERS_DOCSTRING_MAY_NOT_NAME_A_BARRIER_THAT_IS_GONE"),
    ("--help asserts the population was never generated", H3SCMD,
     '        description="THE H3 FULL STUDY, one segment. Runs only with the ',
     '        description="THE H3 FULL STUDY, one segment. It has no seed block, and its opening set has not been generated. Runs only with the ',
     f"{T_H3SRUN}::test_THE_CLI_HELP_MAY_NOT_ASSERT_STATE_THE_TREE_CONTRADICTS"),
    # ═════════ THE PREFLIGHT'S OWN EPILOGUE (2026-09-19) ═══════════════════
    # 🔴 ITS CLOSING LINE WAS PROSE. It kept printing "the population is not
    # frozen and pinned, and no seed block is reserved ... NOT READY TO RUN"
    # after the freeze, the pin, four registered blocks and a completed
    # segment 0 -- above 132 passing checks that each said otherwise.
    ("the epilogue is hardcoded instead of branching on readiness", H3PRE,
     "    if blocking:",
     "    if True:",
     f"{T_H3SRUN}::test_THE_EPILOGUE_TEXT_SAYS_WHAT_THE_READINESS_SAYS"),
    ("readiness stops reading the registry and calls nothing spent", H3PRE,
     '        spent = any(x["exposed"] or x["retired"] or x["test_only"] for x in st)',
     "        spent = False",
     f"{T_H3SRUN}::test_THE_PREFLIGHT_EPILOGUE_IS_DERIVED_FROM_LIVE_STATE"),
    ("readiness asserts the pin instead of reading it", H3PRE,
     '    return {"population_pinned": RULES.OPENING_SET_DIGEST is not None,',
     '    return {"population_pinned": True,',
     f"{T_H3SRUN}::test_THE_PREFLIGHT_EPILOGUE_IS_DERIVED_FROM_LIVE_STATE"),
    # ═════════ SEGMENT 0 CLOSEOUT (ran 2026-09-18) ═════════════════════════
    # 🔴 A SPENT SEGMENT MUST REFUSE RELAUNCH, and must not take the others with
    # it. Before segment 0 ran these could only be driven by monkeypatch; now a
    # real consumption exercises them.
    ("a SPENT segment may be relaunched",
     H3SRUN, '''    bad = [(x, st) for x, st in spent
           if st["exposed"] or st["retired"] or st["test_only"]]''',
     "    bad = []",
     f"{T_H3SRUN}::test_A_SPENT_SEGMENT_CANNOT_BE_RELAUNCHED"),
    ("the spent-seed check ignores EXPOSED and looks only at retired",
     H3SRUN, '''           if st["exposed"] or st["retired"] or st["test_only"]]''',
     '''           if st["test_only"]]''',
     f"{T_H3SRUN}::test_A_SPENT_SEGMENT_CANNOT_BE_RELAUNCHED"),
    #: 🔴 SEGMENT-INDEPENDENT. This used to point SEGMENT 1's block at segment
    #: 0's spent one. Segment 1 then ran, was spent anyway, and the injection
    #: stopped changing anything -- NOT CAUGHT while the test was fine. Pointing
    #: EVERY block at a spent one cannot go stale as segments are consumed.
    ("every segment's block is pointed at a SPENT one", H3SRUN,
     "SEGMENT_SEED_BLOCKS: tuple = (\n"
     "    (202_628_000, 202_628_148),\n"
     "    (202_626_148, 202_626_296),\n"
     "    (202_626_296, 202_626_444),\n"
     "    (202_626_444, 202_626_592),\n"
     ")",
     "SEGMENT_SEED_BLOCKS: tuple = (\n"
     "    (202_628_000, 202_628_148),\n"
     "    (202_628_000, 202_628_148),\n"
     "    (202_628_000, 202_628_148),\n"
     "    (202_628_000, 202_628_148),\n"
     ")",
     #: retargeted at the segment 3 closeout: with every segment spent,
     #: collapsing the blocks onto one SPENT block no longer moves the
     #: spent/unspent split at all. What it still destroys is DISJOINTNESS.
     f"{T_H3POP}::test_THE_BLOCKS_ARE_STILL_MUTUALLY_DISJOINT"),
    #: 🔴 SEGMENT-INDEPENDENT, for the same reason. It used to alias SEGMENT 1
    #: onto segment 0's occupied directory; once segment 1 had its own record the
    #: injection changed nothing. Aliasing the WHOLE fallback occupies every
    #: segment's destination however many have run.
    ("every segment writes into segment 0's occupied destination", H3SRUN,
     '    return f"{OUT_ROOT}/2026-09-15-t1j-h3-study-segment{segment}"',
     '    return f"{OUT_ROOT}/2026-09-18-t1j-h3-study-segment0-retry"',
     f"{T_H3POP}::test_UNSPENT_SEGMENTS_OUTPUT_PATHS_ARE_STILL_ABSENT"),
    ("a partial segment is reported as a strength verdict",
     H3SA, "    below_floor = len(scored) < R.REPORT_FLOOR_PAIRS",
     "    below_floor = False",
     f"{T_H3POP}::test_THE_REPORT_IS_RECOMPUTED_AND_STILL_WITHHOLDS"),

    # ═════════ THE SEGMENT-ISOLATION REPAIR (2026-09-18) ═══════════════════
    # 🔴 MISSING PARENT DIRECTORIES -- the defect that VOIDed segment 0 in one
    # second, spending an authorization and retiring a seed quarter for nothing.
    ("the runner stops creating its output directory",
     H3SRUN, "        if d:\n            os.makedirs(d, exist_ok=True)",
     "        if False:\n            os.makedirs(d, exist_ok=True)",
     f"{T_H3POP}::test_ensure_parent_dirs_ACTUALLY_CREATES_THEM"),
    ("the directory is created AFTER the create-only open",
     H3SRUN, "    ensure_parent_dirs(trace_path, results_path, report_path)",
     "    pass  # noqa",
     f"{T_H3POP}::test_THE_RUN_BODY_CALLS_ensure_parent_dirs_BEFORE_ITS_CREATE_ONLY_OPEN"),

    # 🔴 THE PARENT RECEIPT on an EARLY failure -- before any worker trace exists.
    ("the SEGMENT command writes no launch receipt at all",
     H3SCMD, "        write_receipt(a.segment, {", "        _skip = ({",
     f"{T_H3POP}::test_THE_COMMAND_WRITES_A_PARENT_OWNED_RECEIPT_ON_AN_EARLY_FAILURE"),
    ("a SEGMENT refusal before spawning leaves no receipt",
     H3SCMD, '            outcome, detail = "REFUSED_BEFORE_SPAWN", f"{type(e).__name__}: {e}"',
     '            return code',
     f"{T_H3POP}::test_THE_COMMAND_WRITES_A_PARENT_OWNED_RECEIPT_ON_AN_EARLY_FAILURE"),
    ("the receipt is not create-only",
     H3SCMD, "        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)",
     "        fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)",
     f"{T_H3POP}::test_THE_RECEIPT_IS_CREATE_ONLY"),
    ("an UNAUTHORIZED invocation writes a receipt and occupies the destination",
     H3SCMD,
     '        print("the H3 study is NOT AUTHORIZED (H3_STUDY_EXECUTION_AUTHORIZED is "',
     '        write_receipt(a.segment, {})\n'
     '        print("the H3 study is NOT AUTHORIZED (H3_STUDY_EXECUTION_AUTHORIZED is "',
     f"{T_H3POP}::test_AN_UNAUTHORIZED_INVOCATION_WRITES_NO_RECEIPT"),

    # 🔴 PRIOR-SEGMENT RETIREMENT must not block a later segment -- and the
    # retired block must still refuse to be relaunched.
    ("a retired earlier segment blocks every later one again",
     H3SRUN, "    todo = range(RULES.N_SEGMENTS) if segment is None else [segment]",
     "    todo = range(RULES.N_SEGMENTS)",
     f"{T_H3SRUN}::test_REGISTRATION_IS_ASKED_OF_ONE_SEGMENT_WHEN_ONE_IS_NAMED"),
    ("the RETIRED block may be relaunched",
     H3SRUN, "        if lo < retired_hi and retired_lo < hi:", "        if False:",
     f"{T_H3SRUN}::test_THE_RETIRED_OVERLAP_CHECK_IS_ITS_OWN_GUARD"),
    ("the launch check stops asking whether the seeds are spent",
     H3SRUN, '''    bad = [(x, st) for x, st in spent
           if st["exposed"] or st["retired"] or st["test_only"]]''',
     "    bad = []",
     f"{T_H3SRUN}::test_THE_RETIRED_OVERLAP_CHECK_IS_ITS_OWN_GUARD"),
    ("the launch check is not run before a segment starts",
     H3SRUN, "    check_segment_seeds(segment)\n    from . import d1_probe as D1",
     "    from . import d1_probe as D1",
     f"{T_H3SRUN}::test_run_segment_CHECKS_THE_SEGMENTS_SEEDS_BEFORE_ANYTHING_ELSE"),

    # 🔴 SEGMENT-LOCAL AVAILABILITY: construction must not consult seed status.
    ("build_tasks consults seed status again",
     H3SR, "        seeds = [x for lo, hi in blocks for x in range(lo, hi)]",
     '''        from . import e4_screen_reference as _REF
        for _lo, _hi in blocks:
            for _x in range(_lo, _hi):
                if any(_REF.seed_status(_x).values()):
                    raise H3StudyError("spent")
        seeds = [x for lo, hi in blocks for x in range(lo, hi)]''',
     f"{T_H3SR}::test_build_tasks_DOES_NOT_CONSULT_SEED_STATUS"),
    ("two segments may share one seed block",
     H3SR, "            if rng & seen:", "            if False:",
     f"{T_H3SR}::test_THE_SEED_BLOCKS_ARE_REFUSED_WHEN_MALFORMED"),
    ("a segment block may be the wrong size",
     H3SR, "            if hi - lo != GAMES_PER_SEGMENT:", "            if False:",
     f"{T_H3SR}::test_THE_SEED_BLOCKS_ARE_REFUSED_WHEN_MALFORMED"),
    ("the number of blocks need not match the segments",
     H3SR, "        if len(blocks) != N_SEGMENTS:", "        if False:",
     f"{T_H3SR}::test_THE_SEED_BLOCKS_ARE_REFUSED_WHEN_MALFORMED"),

    # 🔴 THE REPAIR MUST NOT REACH PAST SEGMENT 0.
    ("segment 1-3's pins are changed by segment 0's replacement",
     H3SRUN, '    "14aaefb3a01cc1d08bdfd7f0c4c21599414d0dc68fee8f51584e1e4a50660a41",  # 1, same',
     '    "cde6d04ce52e07088e437754e1ee60b750542de08c6b6e9fbc4744bc2c795d31",  # 1, same',
     f"{T_H3POP}::test_SEGMENTS_1_TO_3_PINS_ARE_UNCHANGED_BY_SEGMENT_0S_REPLACEMENT"),
    ("segment 0 reuses its RETIRED block",
     H3SRUN, "    (202_628_000, 202_628_148),\n    (202_626_148, 202_626_296),",
     "    (202_626_000, 202_626_148),\n    (202_626_148, 202_626_296),",
     f"{T_H3POP}::test_FOUR_SEGMENT_BLOCKS_EACH_148_SEEDS"),
    ("segment 0's retry writes into its SPENT destination",
     H3SRUN, '        return f"{OUT_ROOT}/2026-09-18-t1j-h3-study-segment0-retry"',
     '        return f"{OUT_ROOT}/2026-09-15-t1j-h3-study-segment0"',
     f"{T_H3POP}::test_SEGMENT_0S_DESTINATION_HOLDS_A_COMPLETED_RUN"),

    # ═════════ THE MATCH SEED BLOCK, registered 2026-09-18 ═════════════════
    ("the block is registered as EXPOSED as well as accounted",
     SCREEN, "EXPOSED_SEED_INTERVALS = (",
     "EXPOSED_SEED_INTERVALS = (\n    (202626000, 202626592),",
     f"{T_H3POP}::test_SEGMENT_0_QUARTER_IS_RETIRED_WHOLE_WITH_ZERO_EXPOSED"),
    #: 🔴 RETARGETED AT THE SEGMENT 3 CLOSEOUT. This retired [202626000,
    #: 202626592) -- every H3 block. All four are retired already, so the
    #: injection became a NO-OP and went NOT CAUGHT. The claim survives against
    #: the SYNTHETIC block, the only unspent one left, whose guard exists
    #: precisely to notice that it has stopped being unspent.
    ("the SYNTHETIC block is registered as RETIRED as well as accounted",
     SCREEN, "RETIRED_SEED_INTERVALS = (",
     "RETIRED_SEED_INTERVALS = (\n    (202608188, 202608336),",
     f"{T_H3POP}::test_UNSPENT_SEGMENTS_KEEP_THEIR_QUARTERS"),

    # 🔴 THE SEEDED SCHEDULE'S PINS. Without them the segment check compared a
    # digest to one computed from the same tasks: one source, two sides.
    ("the full-schedule pin is not checked at all",
     H3SRUN, "    check_schedule_digest(tasks)\n    deadline = D1.Deadline",
     "    deadline = D1.Deadline",
     f"{T_H3POP}::test_THE_SEGMENT_PIN_IS_NOT_COMPUTED_FROM_THE_TASKS_IT_CHECKS"),
    ("the segment pin is computed from the tasks it checks",
     H3SRUN, "        want_digest=SEGMENT_DIGESTS[segment],",
     "        want_digest=segment_digest(tasks, segment),",
     f"{T_H3POP}::test_THE_SEGMENT_PIN_IS_NOT_COMPUTED_FROM_THE_TASKS_IT_CHECKS"),

    # 🔴 POSITIONAL SEED BINDING across four contiguous quarters.

    # ═════════ THE SELF-INVALIDATING PIN (2026-09-17 review) ═══════════════
    # 🔴 An OUTPUT must never live inside a PINNED INPUT. Pinning the module that
    # holds OPENING_SET_DIGEST made the freeze sequence invalidate its own
    # artifact on the one edit it requires.
    ("the pinned sources include the module holding OPENING_SET_DIGEST",
     H3SGEN, '_IDENTITY_SOURCES = ("h3_generation_protocol.py", "game/twixt_state.py",\n                     "d1_selection.py")',
     '_IDENTITY_SOURCES = ("h3_study_rules.py", "game/twixt_state.py",\n                     "d1_selection.py")',
     f"{T_H3POP}::test_THE_PIN_EDIT_DID_NOT_INVALIDATE_THE_ARTIFACT"),
    ("the walk's own module is dropped from the pins",
     H3SGEN, '_IDENTITY_SOURCES = ("h3_generation_protocol.py", "game/twixt_state.py",\n                     "d1_selection.py")',
     '_IDENTITY_SOURCES = ("game/twixt_state.py", "d1_selection.py")',
     f"{T_H3POP}::test_THE_PINNED_SET_IS_NOT_VACUOUS"),
    ("the exclusion set is no longer pinned by value",
     H3SGEN, '        "excluded_digest_set": RULES.excluded_digest_set_pin(),',
     '        "excluded_digest_set": "",',
     f"{T_H3POP}::test_CHANGING_THE_EXCLUSION_SET_STILL_INVALIDATES"),
    ("numpy stops being enforced",
     H3SGEN, '                       "excluded_digest_set", "numpy")',
     '                       "excluded_digest_set")',
     f"{T_H3POP}::test_NUMPY_IS_ENFORCED_BECAUSE_GENERATOR_STREAMS_ARE_NOT_GUARANTEED"),
    ("an identity field is left unclassified",
     H3SGEN, 'IDENTITY_RECORDED_ONLY = ("python", "commit")',
     'IDENTITY_RECORDED_ONLY = ("commit",)',
     f"{T_H3POP}::test_EVERY_IDENTITY_FIELD_IS_CLASSIFIED"),
    ("python becomes enforced despite not determining the walk",
     H3SGEN, 'IDENTITY_RECORDED_ONLY = ("python", "commit")',
     'IDENTITY_RECORDED_ONLY = ("python", "commit")\nIDENTITY_MUST_MATCH = IDENTITY_MUST_MATCH + ("python",)',
     f"{T_H3POP}::test_PYTHON_AND_COMMIT_ARE_RECORDED_ONLY_AND_DO_NOT_REFUSE"),
    ("an output constant moves into the pinned protocol",
     H3PROTO, "GEN_SEED_UNIFORM = 20_261_600_000",
     'GEN_SEED_UNIFORM = 20_261_600_000\nOPENING_SET_DIGEST = None',
     f"{T_H3POP}::test_NO_OUTPUT_CONSTANT_LIVES_INSIDE_A_PINNED_SOURCE"),
    ("the seed allocation leaves the pinned protocol",
     H3SR, "    try:\n        return PROTO.seed_for(base, index, attempt)",
     "    try:\n        return base + index * MAX_ATTEMPTS + attempt  # noqa\n        return PROTO.seed_for(base, index, attempt)",
     f"{T_H3POP}::test_attempt_seed_DELEGATES_THE_ARITHMETIC_TO_THE_PINNED_PROTOCOL"),
    ("the rules re-exports drift from the pinned protocol",
     H3SR, "MAX_ATTEMPTS = PROTO.MAX_ATTEMPTS", "MAX_ATTEMPTS = 401",
     f"{T_H3SR}::test_THE_RULES_RE_EXPORTS_ARE_THE_PROTOCOL_S_OWN_VALUES"),
    ("the spent-range guard is no longer injected into the loop",
     H3SR, "                              guard=lambda b: attempt_seed(b, 0, 0),",
     "                              guard=None,",
     f"{T_H3POP}::test_GENERATION_ITSELF_REFUSES_A_RETIRED_BASE"),

    # ═════════ P1 REPAIRS, 2026-09-17 review ═══════════════════════════════
    # 🔴 P1-1: THE BARRIER MUST BE ONE-SHOT.
    ("freeze_population takes a caller-supplied destination again",
     H3SGEN, "def freeze_population() -> Dict[str, Any]:",
     "def freeze_population(out_path=DEFAULT_OUT, trace_path=DEFAULT_TRACE) -> Dict[str, Any]:",
     f"{T_H3POP}::test_freeze_population_ACCEPTS_NO_PATHS"),
    ("the freeze command does not restore the barrier",
     H3FCMD, "        restored = restore_barrier(GENERATOR_SOURCE)",
     "        restored = True",
     f"{T_H3POP}::test_THE_FREEZE_COMMAND_RESTORES_THE_BARRIER_ON_EVERY_OUTCOME"),
    ("restoration is trusted instead of verified from the file",
     H3FCMD, "        readback = barrier_readback(GENERATOR_SOURCE)",
     '        readback = "False"',
     f"{T_H3POP}::test_A_LYING_RESTORATION_IS_CAUGHT_BY_THE_READBACK"),
    ("a failed restoration no longer supersedes the freeze",
     H3FCMD, "        if not restored or readback != \"False\":",
     "        if False:",
     f"{T_H3POP}::test_A_FAILED_RESTORATION_SUPERSEDES_EVEN_A_SUCCESSFUL_FREEZE"),
    ("the freeze command runs without --run",
     H3FCMD, "    if not a.run:", "    if False:",
     f"{T_H3POP}::test_THE_COMMAND_NEVER_OPENS_ITS_OWN_BARRIER"),
    ("the freeze command ignores the barrier",
     H3FCMD, "    if not barrier_is_open():", "    if False:",
     f"{T_H3POP}::test_THE_COMMAND_NEVER_OPENS_ITS_OWN_BARRIER"),
    ("restore_barrier reports success on an unreadable source",
     H3FCMD, "    except OSError:\n        return False\n    if src.count",
     "    except OSError:\n        return True\n    if src.count",
     f"{T_H3POP}::test_THE_RESTORATION_IS_VERIFIED_FROM_THE_FILE_NOT_FROM_MEMORY"),

    # 🔴 P1-2: GENERATION MUST HAPPEN INSIDE THE DEADLINE AND THE TRACE.
    ("the deadline is not checked inside the candidate loop",
     H3PROTO, "            if check_deadline is not None:", "            if False:",
     f"{T_H3POP}::test_THE_DEADLINE_IS_CHECKED_INSIDE_THE_CANDIDATE_LOOP"),
    ("build_population swallows the deadline hook",
     H3SGEN, "        RULES.generate_uniform_openings(seed=base, n=n,\n                                        check_deadline=check_deadline))",
     "        RULES.generate_uniform_openings(seed=base, n=n))",
     f"{T_H3POP}::test_build_population_THREADS_THE_HOOK_WITH_NO_LAMBDA_IN_BETWEEN"),
    ("the runaway guard is reported as an ordinary VOID",
     H3SGEN, "            elif isinstance(e, H3GenerationDeadline):\n                verdict = \"TIMEOUT\"",
     "            elif False:\n                verdict = \"TIMEOUT\"",
     f"{T_H3POP}::test_THE_DEADLINE_IS_CHECKED_INSIDE_THE_CANDIDATE_LOOP"),
    ("the clock goes back to wall time",
     H3SGEN, "    started = time.monotonic()", "    started = time.time()",
     f"{T_H3POP}::test_THE_CLOCK_IS_MONOTONIC"),
    ("the artifact is not fsynced before OK is recorded",
     H3SGEN, "                os.fsync(fh.fileno())\n            verdict = \"OK\"",
     "            verdict = \"OK\"",
     f"{T_H3POP}::test_THE_ARTIFACT_IS_FSYNCED_BEFORE_OK_IS_RECORDED"),
    ("write_artifact accepts both a builder and a set",
     H3SGEN, "    if (build is None) == (openings is None):", "    if False:",
     f"{T_H3POP}::test_write_artifact_REFUSES_BOTH_OR_NEITHER_SOURCE"),

    # 🔴 P1-3: THE DIGEST MUST BIND THE MOVES.
    ("the validator trusts each row's declared digest",
     H3SGEN, "        if recomputed != o[\"digest\"]:", "        if False:",
     f"{T_H3POP}::test_ALTERED_MOVES_WITH_AN_UNTOUCHED_DIGEST_ARE_REFUSED"),
    ("the moves are never replayed through the engine",
     H3SGEN, "        moves = _typed_moves(o)\n        try:\n            st = RULES._replay(moves)",
     "        moves = _typed_moves(o)\n        try:\n            st = RULES._replay([])",
     f"{T_H3POP}::test_ALTERED_MOVES_WITH_AN_UNTOUCHED_DIGEST_ARE_REFUSED"),
    ("a coordinate may be a bool or a float",
     H3SGEN, "            if type(v) is not int or isinstance(v, bool):",
     "            if not isinstance(v, (int, float)):",
     f"{T_H3POP}::test_THE_BOUND_VALIDATOR_REFUSES"),
    ("the declared index need not be the study order",
     H3SGEN, "                or o[\"index\"] != pos:", "                or False:",
     f"{T_H3POP}::test_A_SWAPPED_PAIR_OF_OPENINGS_IS_REFUSED"),
    ("the segment need not match the frozen plan",
     H3SGEN, "        if o[\"segment\"] != RULES.segment_of(pos):", "        if False:",
     f"{T_H3POP}::test_THE_BOUND_VALIDATOR_REFUSES"),
    ("the seed need not match its own attempt count",
     H3SGEN, "            if o[\"seed\"] != want_seed:", "            if False:",
     f"{T_H3POP}::test_THE_BOUND_VALIDATOR_REFUSES"),
    ("the walk need not reproduce the recorded moves",
     H3SGEN, "            if [tuple(m) for m in derived_moves] != moves:",
     "            if False:",
     f"{T_H3POP}::test_A_SUBSTITUTED_OPENING_IS_CAUGHT_ONLY_BY_THE_WALK"),
    ("provenance binding is off by default",
     H3SGEN, "                      bind_provenance: bool = True) -> Dict[str, Any]:",
     "                      bind_provenance: bool = False) -> Dict[str, Any]:",
     f"{T_H3POP}::test_THE_BOUND_VALIDATOR_REFUSES"),
    ("the artifact schema admits unknown keys",
     H3SGEN, "    extra = [k for k in doc if k not in ARTIFACT_KEYS]", "    extra = []",
     f"{T_H3POP}::test_THE_BOUND_VALIDATOR_REFUSES"),
    ("an opening may carry unknown keys",
     H3SGEN, "        spare = [k for k in o if k not in OPENING_KEYS]", "        spare = []",
     f"{T_H3POP}::test_THE_BOUND_VALIDATOR_REFUSES"),
    ("the generator identity is not compared with the running code",
     H3SGEN, "    drift = [k for k in IDENTITY_MUST_MATCH if ident.get(k) != live[k]]",
     "    drift = []",
     f"{T_H3POP}::test_THE_BOUND_VALIDATOR_REFUSES"),
    ("the walk's source pins are dropped from the identity",
     H3SGEN, '                       "filters", "bit_generator", "source_pins",',
     '                       "filters", "bit_generator",',
     f"{T_H3POP}::test_THE_COMMIT_IS_RECORDED_BUT_NOT_ENFORCED"),
    ("the loader does not validate the artifact at all",
     H3SRUN, "        bound = GEN.validate_artifact(doc)",
     "        bound = {\"opening_set_digest\": doc[\"opening_set_digest\"]}",
     f"{T_H3POP}::test_THE_LOADER_VALIDATES_AND_DOES_NOT_REIMPLEMENT"),
    ("the loader accepts any self-consistent population",
     H3SRUN, "    if bound[\"opening_set_digest\"] != pinned:", "    if False:",
     f"{T_H3POP}::test_THE_LOADER_REFUSES_A_SELF_CONSISTENT_BUT_UNPINNED_POPULATION"),

    # ═════════ AMENDMENT 3: the uniform-only population (card §1, §1.7) ═══════
    # 🔴 THE OLD RANGES. Each retired range must stay refused. The first is the
    # 148-opening uniform range -- superseded, not spent -- and the other two were
    # drawn on by VOIDed attempts. A range that quietly becomes reusable is how a
    # study draws seeds an earlier run already touched.
    ("the 148-opening uniform range becomes reusable",
     H3SR, "    (20_261_000_000, 20_261_059_200,", "    (1, 2,",
     f"{T_H3POP}::test_EVERY_RETIRED_RANGE_IS_REFUSED"),
    ("attempt 1's spent co-produced range becomes reusable",
     H3SR, "    (20_261_200_000, 20_261_259_200,", "    (3, 4,",
     f"{T_H3POP}::test_EVERY_RETIRED_RANGE_IS_REFUSED"),
    ("attempt 2's spent co-produced range becomes reusable",
     H3SR, "    (20_261_400_000, 20_261_459_200,", "    (5, 6,",
     f"{T_H3POP}::test_EVERY_RETIRED_RANGE_IS_REFUSED"),
    ("the spent-range check is disabled entirely",
     H3SR, "    return any(not (hi <= s_lo or s_hi <= lo)",
     "    return False and any(not (hi <= s_lo or s_hi <= lo)",
     f"{T_H3POP}::test_EVERY_RETIRED_RANGE_IS_REFUSED"),

    # 🔴 THE RANGE SIZED FOR OPENINGS RATHER THAN CANDIDATES -- the exact defect
    # Amendment 3 repaired. 296 x 400 = 118,400; sizing by openings gives 296.
    ("the generation range covers openings, not candidates",
     H3SR, "    return (base, base + N_PAIRS * MAX_ATTEMPTS)",
     "    return (base, base + N_PAIRS)",
     f"{T_H3POP}::test_THE_RANGE_COVERS_EVERY_CANDIDATE_NOT_EVERY_OPENING"),
    ("the generation base slides back onto a retired range",
     H3PROTO, "GEN_SEED_UNIFORM = 20_261_600_000", "GEN_SEED_UNIFORM = 20_261_000_000",
     f"{T_H3POP}::test_THE_RANGE_COVERS_EVERY_CANDIDATE_NOT_EVERY_OPENING"),

    # 🔴 WRONG POPULATION SIZE. 296 is what the precision target fixed; 148 is the
    # two-stratum design's half -- exactly what a half-finished migration leaves.
    ("the population reverts to 148 openings",
     H3SR, "                              n: int = N_PAIRS,", "                              n: int = 148,",
     f"{T_H3POP}::test_THE_GENERATORS_DEFAULT_POPULATION_SIZE_IS_296"),
    ("the assembled set need not be the full population",
     H3SR, "    if len(uniform) != N_PAIRS:", "    if False:",
     f"{T_H3POP}::test_assemble_opening_set_REFUSES_A_SHORT_POPULATION"),
    ("the artifact may record a short population",
     H3SGEN, '    if type(doc["n"]) is not int or doc["n"] != RULES.N_PAIRS:', "    if False:",
     f"{T_H3POP}::test_THE_ARTIFACT_VALIDATOR_REFUSES"),

    # 🔴 STALE STRATUM AND ORDER FIELDS. One population means one stratum value and
    # no order at all; a record carrying either came from another design.
    ("the artifact may name the co-produced stratum",
     H3SGEN, '    if doc["stratum"] != RULES.STRATUM_UNIFORM:', "    if False:",
     f"{T_H3POP}::test_THE_ARTIFACT_VALIDATOR_REFUSES"),
    # 🔑 RE-ANCHORED. The `order`-specific branch was subsumed by the EXACT-KEY
    # rule and removed rather than left unreachable, so the control now injects
    # into the rule that actually refuses a stale `order`.
    ("an opening may carry a stale alternating order",
     H3SGEN, "        spare = [k for k in o if k not in OPENING_KEYS]",
     "        spare = [k for k in o if k not in OPENING_KEYS + ('order',)]",
     # 🔴 RETARGETED AGAIN, and the harness is why. The first target checked
     # `build_population()`'s OUTPUT and `OPENING_KEYS` -- neither of which the
     # validator's allow-list touches -- so widening that list changed nothing
     # the test could see. NOT CAUGHT. The test that actually feeds an artifact
     # carrying `order` to the validator is the one that binds it.
     f"{T_H3POP}::test_THE_ARTIFACT_VALIDATOR_REFUSES"),
    ("check_opening_set stops refusing a stale order field",
     H3SR, '    stale = [o["index"] for o in openings if o.get("order") is not None]',
     "    stale = []",
     f"{T_H3POP}::test_check_opening_set_REFUSES_A_STALE_ORDER_FIELD"),
    ("the analysis scores records from another population",
     H3SA, '        if g.get("stratum") != R.STRATUM_UNIFORM:', "        if False:",
     f"{T_H3POP}::test_THE_ANALYSIS_REFUSES_A_RECORD_FROM_ANOTHER_POPULATION"),
    ("the analysis accepts a record carrying an order",
     H3SA, '        if "order" in g:', "        if False:",
     f"{T_H3POP}::test_THE_ANALYSIS_REFUSES_A_RECORD_FROM_ANOTHER_POPULATION"),

    # 🔴 WEAKENED CLAIM WORDING. The study's whole exposure is a number quoted
    # without the population it is about, so the claim is a REQUIRED field and
    # softening it is a defect, not an edit.
    ("the narrowed claim is softened in the artifact",
     H3SGEN, '    "NOTHING about realistic play, about positions either engine would actually "',
     '    "little about realistic play, about positions either engine would actually "',
     f"{T_H3POP}::test_THE_CLAIM_NAMES_THE_POPULATION_AND_DISCLAIMS_PLAY"),
    ("the artifact stops requiring its claim",
     H3SGEN, '    if doc.get("claim") != CLAIM:', "    if False:",
     f"{T_H3POP}::test_THE_ARTIFACT_VALIDATOR_REFUSES"),
    ("a report may omit the narrowed claim",
     H3SA, "    if got is None:", "    if False:",
     f"{T_H3POP}::test_A_REPORT_WITHOUT_THE_VERBATIM_CLAIM_IS_REFUSED"),
    ("a report may alter the narrowed claim",
     H3SA, "    if got != CLAIM:", "    if False:",
     f"{T_H3POP}::test_A_REPORT_WITHOUT_THE_VERBATIM_CLAIM_IS_REFUSED"),
    ("the disclaimer that nobody plays these positions is dropped",
     H3SGEN, '    "position is not a position anyone plays."',
     '    "position is a reasonable proxy for play."',
     f"{T_H3POP}::test_THE_CLAIM_NAMES_THE_POPULATION_AND_DISCLAIMS_PLAY"),

    # 🔴 THE CO-PRODUCED PATHS. Retired code must stay unreachable; an import from
    # a live module makes "retired" a label rather than a fact.
    ("a live module imports the retired engine generator",
     H3SRUN, "from . import h3_study_generator as GEN\nfrom . import h3_study_rules as RULES",
     "from . import h3_study_generator as GEN\nfrom . import h3_coproduced_generator_retired as OLD\nfrom . import h3_study_rules as RULES",
     f"{T_H3POP}::test_NO_ACTIVE_MODULE_REACHES_THE_RETIRED_ENGINE_PATH"),
    ("the analysis imports the retired JVM supervisor",
     H3SA, "from . import h3_study_rules as R",
     "from . import h3_study_rules as R\nfrom . import h3_generation_command as GCMD",
     f"{T_H3POP}::test_NO_ACTIVE_MODULE_REACHES_THE_RETIRED_ENGINE_PATH"),

    # 🔴 THE FREEZE BARRIER. Generating is free; declaring one population THE
    # population is not. The barrier must be read BEFORE anything durable.
    ("the population freeze barrier is opened",
     H3SGEN, "H3_POPULATION_FREEZE_AUTHORIZED = False",
     "H3_POPULATION_FREEZE_AUTHORIZED = True",
     f"{T_H3POP}::test_GENERATING_IS_FREE_AND_FREEZING_IS_NOT"),
    ("the freeze barrier stops being read first",
     H3SGEN, "    check_freeze_barrier()\n    return write_artifact(",
     "    return write_artifact(",
     f"{T_H3POP}::test_THE_BARRIER_IS_READ_FIRST_BY_AST"),
    ("the barrier accepts any truthy value",
     H3SGEN, "    if H3_POPULATION_FREEZE_AUTHORIZED is not True:",
     "    if not H3_POPULATION_FREEZE_AUTHORIZED:",
     f"{T_H3POP}::test_THE_FREEZE_BARRIER_REFUSES_A_TRUTHY_NON_TRUE_VALUE"),

    # 🔴 CREATE-ONLY, the destination, and the provenance the artifact must carry.
    ("the artifact may be overwritten",
     H3SGEN, "    if os.path.lexists(path):", "    if False:",
     f"{T_H3POP}::test_A_DANGLING_SYMLINK_IS_REFUSED_NOT_WRITTEN_THROUGH"),
    ("attempt provenance becomes optional",
     H3SGEN, '                or o["attempts"] < 1:', "                or False:",
     f"{T_H3POP}::test_THE_ARTIFACT_VALIDATOR_REFUSES"),
    ("the official destination moves into a spent directory",
     H3SGEN, 'OUT_DIR = "docs/superpowers/evidence/2026-09-17-t1j-h3-study-openings-uniform"',
     'OUT_DIR = "docs/superpowers/evidence/2026-09-15-t1j-h3-study-openings"',
     f"{T_H3SRUN}::test_THE_FROZEN_DESTINATION_IS_SEPARATE_FROM_EVERY_SPENT_RUN"),
    ("the runner repeats the destination instead of reading it",
     H3SRUN, "OPENING_SET_PATH = GEN.DEFAULT_OUT",
     'OPENING_SET_PATH = "docs/superpowers/evidence/2026-09-15-t1j-h3-study-openings/01_opening_set.json"',
     f"{T_H3POP}::test_THE_RUNNER_READS_THE_DESTINATION_RATHER_THAN_REPEATING_IT"),

    # 🔴 THE GATE INVENTORY. Three hand-kept lists said seven, eight and ten for
    # weeks. The count must be DERIVED, and a retired gate still counts.
    ("the gate inventory stops seeing gates at all",
     GINV, '                if isinstance(target, ast.Name) and "AUTHORIZED" in target.id:',
     '                if isinstance(target, ast.Name) and target.id == "NOTHING":',
     f"{T_GATES}::test_THE_COUNT_IS_DERIVED_AND_ANY_REPORT_SAYING_OTHERWISE_IS_WRONG"),
    ("the inventory scans only the first statement of each module",
     GINV, "        for node in tree.body:", "        for node in tree.body[:1]:",
     f"{T_GATES}::test_A_RETIRED_GATE_STILL_COUNTS"),

    # ---------------------------------------------- 12.5 seed registration
    ("registration check disabled", PROBE,
     "    if missing:\n        raise D1Error(\n"
     '            f"the D1 diagnostic seed block',
     "    if False:\n        raise D1Error(\n"
     '            f"the D1 diagnostic seed block',
     f"{T_PROBE}::test_an_unregistered_block_is_refused"),
    ("registration check looks only at the first seed", PROBE,
     "    missing = [s for s in range(lo, hi) if not REF.seed_is_accounted(s)]",
     "    missing = [s for s in [lo] if not REF.seed_is_accounted(s)]",
     f"{T_PROBE}::test_a_PARTLY_registered_block_is_still_refused"),
    ("the D1 seed block un-registered from ACCOUNTED", REF_SRC,
     "    (202614000, 202614227),          # D1 SAME-POSITION INTERROGATION. 227 seeds,",
     "    # (202614000, 202614227),        # D1 SAME-POSITION INTERROGATION. 227 seeds,",
     f"{T_PROBE}::test_the_seed_interval_is_ACCOUNTED_and_now_RETIRED_after_the_VOID"),
    ("the D1 seed block marked EXPOSED as well as accounted", REF_SRC,
     "EXPOSED_SEED_INTERVALS = (\n",
     "EXPOSED_SEED_INTERVALS = (\n    (202614000, 202614227),\n",
     f"{T_PROBE}::test_the_seed_interval_is_ACCOUNTED_and_now_RETIRED_after_the_VOID"),

    # ------------------------ 2026-09-08: the §14 block's REGISTRATION, four ways
    # The four above bind the RETIRED 2026-08-28 block. These bind the block the
    # seed-preparation authorization just registered -- a different interval, a
    # different list membership, and a different pair of assertions.
    ("the §14 block un-registered from ACCOUNTED (the edit reverted)", REF_SRC,
     "    (202615000, 202615221),          # D1 SAME-POSITION INTERROGATION, §14 RETRY.",
     "    # (202615000, 202615221),        # D1 SAME-POSITION INTERROGATION, §14 RETRY.",
     f"{T_PROBE}::test_the_REAL_registry_now_SATISFIES_the_barrier_while_the_GATE_stays_shut"),
    ("the §14 block un-registered -- caught at the STATE assertion too", REF_SRC,
     "    (202615000, 202615221),          # D1 SAME-POSITION INTERROGATION, §14 RETRY.",
     "    # (202615000, 202615221),        # D1 SAME-POSITION INTERROGATION, §14 RETRY.",
     f"{T_PROBE}::test_the_NEW_block_is_now_EXPOSED_AND_RETIRED_after_the_COMPLETED_run"),
    # 🔴 RE-AIMED 2026-09-08 (post-run). These injected "the block is ALSO marked
    # EXPOSED / RETIRED", which the COMPLETED run has made the TRUE state -- so as
    # defects they no longer exist. The defect now is the opposite: a spent block
    # left un-exposed or un-retired, which would let its 221 seeds be scheduled a
    # second time.
    ("the spent §14 block is NOT exposed -- 221 draws unrecorded", REF_SRC,
     "    (202615000, 202615221),          # D1 SAME-POSITION INTERROGATION, §14 RETRY,",
     "    # (202615000, 202615221),        # D1 SAME-POSITION INTERROGATION, §14 RETRY,",
     f"{T_PROBE}::test_the_NEW_block_is_now_EXPOSED_AND_RETIRED_after_the_COMPLETED_run"),
    ("the spent §14 block is NOT retired -- a one-shot schedule left replayable", REF_SRC,
     "    (202615000, 202615221),          # D1 §14, retired WHOLE 2026-09-08: a",
     "    # (202615000, 202615221),        # D1 §14, retired WHOLE 2026-09-08: a",
     f"{T_PROBE}::test_the_NEW_block_is_now_EXPOSED_AND_RETIRED_after_the_COMPLETED_run"),
    # THE CONSEQUENCE, not just the tuples. A single anchor cannot un-spend two
    # lists (the 2026-09-07 lesson), so this removes the availability check
    # itself: if nothing refuses a spent seed, retirement is only a comment.
    ("a spent seed is schedulable because availability is never checked", REF_SRC,
     "def validate_task_executable(task: Dict[str, Any]) -> None:",
     "def validate_task_executable(task: Dict[str, Any]) -> None:\n    return None",
     f"{T_PROBE}::test_the_SPENT_block_can_no_longer_be_SCHEDULED"),
    # 🔑 THE ONE THAT MATTERS MOST THIS ROUND: registration is bookkeeping, not
    # permission. If the seed edit had also flipped the gate, the barrier test's
    # second half must say so -- otherwise "all gates stay False" is a claim no
    # test can contradict.
    ("registration ALSO opened the D1 execution gate", PROBE,
     "D1_EXECUTION_AUTHORIZED = False",
     "D1_EXECUTION_AUTHORIZED = True",
     f"{T_PROBE}::test_the_REAL_registry_now_SATISFIES_the_barrier_while_the_GATE_stays_shut"),

    # -------------------------------- the real, toolchain-backed compile step
    ("compile step skips toolchain verification", PROBE,
     "    tc = TC.verified_paths()",
     "    tc = {'root': 'x', 'source': 'x', 'verified': 0, 'jar': paths.jar,\n"
     "          'jdk_home': os.path.dirname(os.path.dirname(paths.java))}",
     f"{T_PROBE}::test_an_unverifiable_toolchain_stops_the_compile"),
    ("compile step accepts a jar that is not the verified one", PROBE,
     "    if os.path.realpath(paths.jar) != os.path.realpath(tc[\"jar\"]):",
     "    if False:",
     f"{T_PROBE}::test_a_jar_that_is_not_the_verified_one_is_refused"),
    ("compile step accepts a java outside the verified JDK", PROBE,
     "    if os.path.realpath(paths.java) != os.path.realpath(java):",
     "    if False:",
     f"{T_PROBE}::test_a_java_binary_outside_the_verified_jdk_is_refused"),
    ("compile step reuses an existing class directory", PROBE,
     "    if os.path.exists(paths.classes):",
     "    if False:",
     f"{T_PROBE}::test_an_existing_class_directory_is_refused"),
    ("compile step ignores a failing javac", PROBE,
     "    if result.returncode != 0:",
     "    if False:",
     f"{T_PROBE}::test_a_failing_javac_is_a_VOID"),
    ("compile step builds the E3b set, not the query path", PROBE,
     "paths.classes, sources=A.PREFLIGHT_SOURCES)",
     "paths.classes, sources=A.JAVA_SOURCES)",
     f"{T_PROBE}::test_the_compile_step_builds_the_PREFLIGHT_sources_d1_actually_queries"),
    ("compile step skips the E4 jar pin", PROBE,
     "    if jar_sha != SCREEN_CMD.JAR_SHA256:",
     "    if False:",
     f"{T_PROBE}::test_a_jar_that_disagrees_with_E4s_PIN_is_refused"),

    # ------------------------------------------- 5.3 the full T1j-side capture
    ("postcondition surface never read", PROBE,
     '            post = INT.check_postcond(out, expected_refl=INT.QUERY_REFL_N,\n'
     '                                      where=where, phase="query")',
     "            post = A.PostCond(True, 0, 0, True, True, True, INT.QUERY_REFL_N, 0)",
     f"{T_PROBE}::test_a_reply_with_no_postcondition_line_is_VOID"),
    ("reflection COUNT accepted as whatever came back", PROBE,
     "expected_refl=INT.QUERY_REFL_N,",
     "expected_refl=A.parse_postconds(out)[0].refl_n,",
     f"{T_PROBE}::test_a_wrong_reflection_COUNT_is_VOID"),
    ("searched position never re-bound", PROBE,
     "        searched = _searched_state(dumps[0], state, moves, where=where)",
     "        searched = {'ply': dumps[0].ply, 'next_player': dumps[0].next_player,\n"
     "                    'term_y': False, 'term_x': False, 'n_pegs': 0,\n"
     "                    'n_bridges': 0, 'n_legal': 0, 'history_len': 0}",
     f"{T_PROBE}::test_a_SEARCH_jvm_that_rebuilt_a_different_position_is_VOID"),
    ("more than one searched dump accepted", PROBE,
     "        if len(dumps) != 1:",
     "        if False:",
     f"{T_PROBE}::test_more_than_one_searched_dump_is_VOID"),
    ("a move illegal in OUR engine accepted", PROBE,
     "        if recs[0].move not in set(state.legal_moves()):",
     "        if False:",
     f"{T_PROBE}::test_a_move_illegal_in_OUR_engine_is_VOID"),
    ("depth agreement not recorded", PROBE,
     '                    "depths_agree": len(set(moves_by_depth.values())) == 1})',
     '                    "depths_agree": True})',
     f"{T_PROBE}::test_the_position_record_says_the_depths_DISAGREE_when_they_do"),

    # --------------- the diagnostic repair the 2026-08-28 VOID demanded
    ("the helper's stdout discarded again on a non-zero query exit", PROBE,
     '                f"T1j reported: {A.helper_failure_excerpt(out)}")',
     '                f"")',
     f"{T_PROBE}::test_a_nonzero_exit_carries_the_helpers_OWN_failure_lines"),
    ("the failure excerpt unbounded", ADAPTER,
     "FAILURE_EXCERPT_CHARS = 800", "FAILURE_EXCERPT_CHARS = 100000",
     f"{T_PROBE}::test_the_CHARACTER_cap_truncates_one_enormous_failure_line"),
    ("the excerpt line cap removed", ADAPTER,
     "FAILURE_EXCERPT_LINES = 12", "FAILURE_EXCERPT_LINES = 100000",
     f"{T_PROBE}::test_the_LINE_cap_drops_the_tail_of_a_long_verdict_list"),
    ("the excerpt falls back to silence", ADAPTER,
     "    if not lines:\n        lines = [l.strip() for l in out.splitlines()",
     "    if False:\n        lines = [l.strip() for l in out.splitlines()",
     f"{T_PROBE}::test_output_with_NO_verdict_line_falls_back_to_the_tail_not_silence"),
    ("the label drops the cohort", PROBE,
     '            f"[{pos.get(\'signature\') or \'?\'}/{pos.get(\'role\') or \'?\'}] "',
     '            f""',
     f"{T_PROBE}::test_the_label_names_all_four_fields"),
    ("the label drops the prefix digest", PROBE,
     '            f"digest={digest[:16] or \'?\'}")',
     '            f"")',
     f"{T_PROBE}::test_the_label_names_all_four_fields"),
    ("the run loop stops labelling its refusals", PROBE,
     "        where = position_label(pos)",
     '        where = f"{pos.get(\'task_id\')}"',
     f"{T_PROBE}::test_run_stage_refusals_identify_the_position_too"),
    ("the probe refusal loses the position again", PROBE,
     '        where = f"{label}: depth {depth} invocation {i}"',
     '        where = f"depth {depth} invocation {i}"',
     f"{T_PROBE}::test_every_probe_refusal_identifies_the_position"),

    ("the BINDER discards the transcript on a non-zero replay exit", INTEG,
     '                             f"{task[\'task_id\']} {where}: T1j replay exit {rc}. "\n'
     '                             f"T1j reported: {A.helper_failure_excerpt(out)}")',
     '                             f"{task[\'task_id\']} {where}: T1j replay exit {rc}.")',
     f"{T_INTEG}::test_a_nonzero_replay_exit_carries_the_helpers_OWN_failure_lines"),
    ("the AGENT discards the transcript on a non-zero query exit", INTEG,
     '            message = (f"{where}: exit {rc} with {len(recs)} record(s). "\n'
     '                       f"T1j reported: {A.helper_failure_excerpt(out)}")',
     '            message = (f"{where}: exit {rc} with {len(recs)} record(s).")',
     
     f"{T_INTEG}::test_a_nonzero_query_exit_carries_the_helpers_OWN_failure_lines"),
    ("the binder's excerpt carries the dump body", ADAPTER,
     '    lines = [l.strip() for l in out.splitlines() if l.startswith(_VERDICT_PREFIXES)]',
     '    lines = [l.strip() for l in out.splitlines() if l.strip()]',
     f"{T_INTEG}::test_the_replay_failure_excerpt_is_bounded_and_drops_the_dump_body"),

    # ------------------------------------------------ the run manifest (5.4)
    ("manifest drops the cohort label", SEL,
     '        entry = {field: row[field] for field in MANIFEST_FIELDS}',
     '        entry = {field: row[field] for field in MANIFEST_FIELDS\n'
     '                 if field not in ("signature", "role")}',
     f"{T_SEL}::test_the_run_manifest_carries_what_the_probe_and_5_4_both_need"),
    ("registration checked only after the clock starts", PROBE,
     "    _check_seed_registration()\n    deadline.start()",
     "    deadline.start()\n    _check_seed_registration()",
     f"{T_PROBE}::test_an_unregistered_block_arms_no_timer_at_all"),
    ("registration checked only after the supervisor is armed", PROBE,
     "    _check_seed_registration()\n    deadline.start()\n    with _supervisor(deadline):\n"
     "        return _run_stages(",
     "    deadline.start()\n    with _supervisor(deadline):\n"
     "        _check_seed_registration()\n        return _run_stages(",
     f"{T_PROBE}::test_an_unregistered_block_arms_no_timer_at_all"),
    ("registration no longer stops the run before compilation", PROBE,
     "    _check_seed_registration()\n    deadline.start()",
     "    deadline.start()",
     f"{T_PROBE}::test_an_unregistered_block_stops_the_run_before_anything_is_compiled"),

    # ------------------------- ONE deadline, ONE origin: the supervisor's arm
    ("supervisor armed from limit_s instead of the remaining time", PROBE,
     "    signal.setitimer(signal.ITIMER_REAL, left)",
     "    signal.setitimer(signal.ITIMER_REAL, deadline.limit_s)",
     f"{T_PROBE}::test_the_supervisor_arms_from_the_started_deadlines_REMAINING_time"),
    ("supervisor arms from a deadline that was never started", PROBE,
     "    if not deadline.started:",
     "    if False:",
     f"{T_PROBE}::test_the_supervisor_refuses_a_deadline_that_was_never_started"),
    ("supervisor arms a DISABLED timer when nothing remains", PROBE,
     "    left = deadline.remaining()\n    if left <= 0:",
     "    left = deadline.remaining()\n    if False:",
     f"{T_PROBE}::test_the_supervisor_refuses_when_no_time_remains"),
    ("remaining() ignores elapsed time, so the window restarts", PROBE,
     "        return self.limit_s - self.elapsed()",
     "        return self.limit_s",
     f"{T_PROBE}::test_the_supervisor_arms_from_the_started_deadlines_REMAINING_time"),

    # ------------------------------------------- 12.7 prefix identity / E3b
    ("digest re-check disabled", PROBE,
     '    got = SEL.canonical_digest(state)\n    if got != pos.get("digest"):',
     '    got = SEL.canonical_digest(state)\n    if False:',
     f"{T_PROBE}::test_a_prefix_that_does_not_replay_to_its_recorded_digest_voids"),
    ("prefix legality check disabled", PROBE,
     "        if move not in set(state.legal_moves()):",
     "        if False:",
     f"{T_PROBE}::test_an_illegal_move_in_a_retained_prefix_voids"),
    ("E3b binder never called", PROBE,
     "        _bind_prefix(binder, ctx, task_id=where, state=state, prefix=prefix)",
     "        pass",
     f"{T_PROBE}::test_every_retained_prefix_is_replayed_through_the_e3b_binder"),
    ("binder AbortError not translated to VOID", PROBE,
     '    except AbortError as e:\n        raise D1VoidError(\n'
     '            f"{task_id}: the E3b binder refused',
     '    except ZeroDivisionError as e:\n        raise D1VoidError(\n'
     '            f"{task_id}: the E3b binder refused',
     f"{T_PROBE}::test_a_binder_divergence_becomes_a_VOID_not_an_unexpected_error"),
    ("binder replay timeout not translated to VOID", PROBE,
     "    except subprocess.TimeoutExpired as e:\n        # The binder's replay",
     "    except ZeroDivisionError as e:\n        # The binder's replay",
     f"{T_PROBE}::test_a_forced_query_timeout_is_void_and_writes_no_report"),
    ("ply_cap not carried to the runtime", PROBE,
     "ply_cap=paths.ply_cap, timeout_s=PER_QUERY_TIMEOUT_S)",
     "ply_cap=999, timeout_s=PER_QUERY_TIMEOUT_S)",
     f"{T_PROBE}::test_the_binder_call_carries_the_frozen_timeout_and_the_explicit_ply_cap"),
    ("replay timeout widened at the runtime", PROBE,
     "ply_cap=paths.ply_cap, timeout_s=PER_QUERY_TIMEOUT_S)",
     "ply_cap=paths.ply_cap, timeout_s=99999)",
     f"{T_PROBE}::test_the_binder_call_carries_the_frozen_timeout_and_the_explicit_ply_cap"),

    # ------------------------------------------------- the adapter's new hole
    ("replay's unbounded-wait guard removed", ADAPTER,
     '    if timeout_s is None:\n        raise TypeError(\n'
     '            "timeout_s is required: subprocess.run(timeout=None) waits forever")',
     "    pass",
     f"{T_ADP}::test_replay_requires_a_timeout_and_refuses_an_unbounded_one"),
    ("replay timeout dropped at the last hop", ADAPTER,
     '        HELPER_MAIN, "replay", str(ply_cap),\n    ] + [f"{x},{y}" for (x, y) in xy]\n'
     "    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)",
     '        HELPER_MAIN, "replay", str(ply_cap),\n    ] + [f"{x},{y}" for (x, y) in xy]\n'
     "    p = subprocess.run(args, capture_output=True, text=True)",
     f"{T_ADP}::test_the_replay_timeout_reaches_subprocess_run"),
    ("T1jRuntime accepts an unbounded timeout", INTEG,
     '        if timeout_s is None:\n            raise TypeError("timeout_s is required',
     '        if False:\n            raise TypeError("timeout_s is required',
     f"{T_INTEG}::test_the_runtime_refuses_an_unbounded_timeout_before_any_replay"),

    # ------------------------------------------------ 12.6 incumbent capture
    ("capture draws from the readout stream", G3,
     "            self.last_capture = {",
     "            self.readout_rng.random()\n            self.last_capture = {",
     f"{T_INC}::test_capture_preserves_the_state_of_both_generators"),
    ("capture switched on by default", G3,
     "    def __init__(self, *, evaluator, colour: str, seed: int, config=None,\n"
     "                 capture: bool = False):",
     "    def __init__(self, *, evaluator, colour: str, seed: int, config=None,\n"
     "                 capture: bool = True):",
     f"{T_INC}::test_capture_is_off_by_default_and_exposes_nothing"),
    ("capture flag not forwarded by the qualified builder", G3,
     "        capture=capture\n    )",
     "        capture=False\n    )",
     f"{T_INC}::test_the_capture_flag_threads_through_the_qualified_builder"),
    ("root-noise check disabled", PROBE,
     "    if root.priors != root.priors_raw:",
     "    if False:",
     f"{T_INC}::test_a_root_policy_that_carries_noise_is_a_VOID"),
    ("evaluator reloaded for every position", PROBE,
     "        if self._evaluator is None:\n            self._evaluator = self._load(self.repo_root)",
     "        if True:\n            self._evaluator = self._load(self.repo_root)",
     f"{T_INC}::test_one_evaluator_is_loaded_for_the_entire_run"),
    ("incumbent readout spends no query", PROBE,
     "        budget.spend(1)                      # 12.4 funds ONE readout per position",
     "        pass                                 # 12.4 funds ONE readout per position",
     f"{T_INC}::test_the_incumbent_spends_exactly_one_query_per_position"),
    ("policy rank tie-break reversed", PROBE,
     "    ranked = sorted(policy.items(), key=lambda kv: (-kv[1], kv[0]))",
     "    ranked = sorted(policy.items(), key=lambda kv: (kv[1], kv[0]))",
     f"{T_INC}::test_the_raw_policy_covers_every_legal_move_and_ranks_the_chosen_one"),
    ("incumbent identity typed instead of read from the frozen plan", PROBE,
     '    reference, sha1 = pairs.pop()',
     '    reference, sha1 = "0379", "8ad62ac432c35c6ea9b0630b8a2b8c572a0b03a1"',
     f"{T_INC}::test_the_incumbent_identity_comes_from_the_frozen_l0_plan"),

    # ---------------------------------------------- 12.1-12.3 selection rule
    ("digest downgraded to the sha1 superset helper", SEL,
     "    return hashlib.sha256(payload.encode()).hexdigest()",
     "    return hashlib.sha1(payload.encode()).hexdigest()",
     f"{T_SEL}::test_the_digest_is_sha256_not_the_existing_sha1_helper"),
    ("digest payload widened beyond the frozen three fields", SEL,
     "    payload = json.dumps((to_move, pegs, bridges), sort_keys=True)",
     "    payload = json.dumps((to_move, pegs, bridges, _limit), sort_keys=True)",
     f"{T_SEL}::test_a_field_outside_the_frozen_three_does_not_enter_the_digest"),
    ("per-cell cap loosened", SEL,
     "PER_CELL_CAP = 3", "PER_CELL_CAP = 4",
     f"{T_SEL}::test_the_frozen_counts_are_reproduced"),
    ("deduplication disabled", SEL,
     '        if row["digest"] in seen:\n            continue',
     '        if False:\n            continue',
     f"{T_SEL}::test_the_frozen_counts_are_reproduced"),
    # 🔴 RE-AIMED 2026-09-12. It named `test_every_selected_position_has_our_
    # incumbent_to_move`, which takes the `selection` FIXTURE: dropping the
    # restriction changes the cohort sizes, `select_all` refuses them against
    # §12.1's frozen budget, and the test ERRORS in setup without ever running.
    # The requirement's observable is that budget, and the frozen-count test
    # reads §12's OWN output (`cohorts`), so it runs and fails on the numbers.
    ("incumbent-to-move restriction dropped", SEL,
     '        ours = [r for r in plies if r["system"] == "ours"]',
     '        ours = list(plies)',
     f"{T_SEL}::test_the_frozen_counts_are_reproduced"),
    # 🔴 RE-AIMED 2026-09-12, for the same reason as the control above. 12.3 says
    # controls are matched to the position cells BY CONSTRUCTION, so its
    # observable is the control count alone: this defect leaves both POSITION
    # counts untouched and moves only the two control counts.
    ("controls no longer matched to the position cells", SEL,
     '            [r for r in ours if bool(r[col]) is False and cell(r) in cells])',
     '            [r for r in ours if bool(r[col]) is False])',
     f"{T_SEL}::test_the_frozen_counts_are_reproduced"),
    ("seed written in place so a shared row is overwritten", SEL,
     '        group["rows"] = [dict(r, seed=seeds[i + n], signature=key[0], role=key[1])\n'
     '                         for n, r in enumerate(group["rows"])]',
     '        group["rows"] = [r.update({"seed": seeds[i + n], "signature": key[0],\n'
     '                                   "role": key[1]}) or r\n'
     '                         for n, r in enumerate(group["rows"])]',
     f"{T_SEL}::test_a_supplied_interval_is_assigned_injectively_and_exhausted"),
    ("frozen-count reconciliation removed", SEL,
     "            if got != sig[want]:",
     "            if False:",
     f"{T_SEL}::test_a_cohort_that_departs_from_the_frozen_table_is_refused"),


    # ═════════ the low-ply qualification runner ═════════
    ("lowply gate flipped open", LP,
     "LOWPLY_QUALIFICATION_AUTHORIZED = False",
     "LOWPLY_QUALIFICATION_AUTHORIZED = True",
     f"{T_LP}::test_the_gate_is_false_as_published"),
    ("lowply public runner ungated", LP,
     "    if not LOWPLY_QUALIFICATION_AUTHORIZED:\n        raise LowPlyError(",
     "    if False:\n        raise LowPlyError(",
     f"{T_LP}::test_the_public_runner_refuses_while_the_gate_is_shut"),

    # 🔴 THE CENTRAL REGRESSION: D1's exact defect, reintroduced. A non-zero exit
    # becomes an abort again, so the measurement is destroyed instead of recorded.
    ("lowply turns a non-zero exit back into a VOID (D1's defect)", LP,
     "    if len(recs) != 1:",
     "    if rc != 0 or len(recs) != 1:",
     f"{T_LP}::test_a_reply_that_does_not_complete_its_depth_is_a_recorded_FAIL"),
    ("lowply stops recording the incomplete-depth shortfall", LP,
     "    if not rec.completed or rec.completed_depth != depth:",
     "    if False:",
     f"{T_LP}::test_a_reply_that_does_not_complete_its_depth_is_a_recorded_FAIL"),
    ("lowply stops checking the move against OUR engine", LP,
     "    if rec.move is not None and rec.move not in set(state.legal_moves()):",
     "    if False:",
     f"{T_LP}::test_an_illegal_move_is_a_FAIL_not_an_abort"),
    ("lowply stops comparing the two invocations", LP,
     "            agree = len(set(keys)) == 1",
     "            agree = True",
     f"{T_LP}::test_two_invocations_that_disagree_are_a_FAIL"),
    ("lowply verdict always PASS", LP,
     '"verdict": "PASS" if n_failures == 0 else "FAIL",',
     '"verdict": "PASS",',
     f"{T_LP}::test_a_reply_that_does_not_complete_its_depth_is_a_recorded_FAIL"),
    ("lowply timeout no longer a VOID", LP,
     "    except subprocess.TimeoutExpired as e:\n        raise LowPlyVoidError(\n"
     '            f"{where}: T1j did not answer within',
     "    except ZeroDivisionError as e:\n        raise LowPlyVoidError(\n"
     '            f"{where}: T1j did not answer within',
     f"{T_LP}::test_a_QUERY_timeout_is_a_VOID_and_writes_nothing"),
    ("lowply D1 exception translation removed", LP,
     "    except D1VoidError as e:",
     "    except ZeroDivisionError as e:",
     f"{T_LP}::test_a_deadline_breach_is_a_VOID_and_writes_nothing"),
    ("lowply frozen-input hash check disabled", LP,
     "    if got != FROZEN_PREFIXES_SHA256:",
     "    if False:",
     f"{T_LP}::test_a_tampered_prefix_file_is_refused"),
    ("lowply per-call timeout dropped at the last hop", LP,
     "classes=paths.classes, repeats=1, timeout_s=PER_CALL_TIMEOUT_S)",
     "classes=paths.classes, repeats=1, timeout_s=None)",
     f"{T_LP}::test_every_call_carries_the_frozen_timeout_at_the_boundary"),
    ("lowply uses the same-JVM determinism mode", LP,
     "classes=paths.classes, repeats=1, timeout_s=PER_CALL_TIMEOUT_S)",
     "classes=paths.classes, repeats=2, timeout_s=PER_CALL_TIMEOUT_S)",
     f"{T_LP}::test_each_depth_issues_two_separate_repeats_1_invocations"),
    ("lowply query cap loosened", LP,
     "QUERY_CAP = N_PREFIXES * len(DEPTHS) * INVOCATIONS_PER_DEPTH      # 36",
     "QUERY_CAP = 99999      # 36",
     f"{T_LP}::test_the_frozen_constants_match_the_card"),
    ("lowply wall-clock cap widened", LP,
     "RUN_DEADLINE_S = 900", "RUN_DEADLINE_S = 90000",
     f"{T_LP}::test_the_frozen_constants_match_the_card"),
    ("lowply reads D1's gate", LP,
     "    if not LOWPLY_QUALIFICATION_AUTHORIZED:\n        raise LowPlyError(",
     "    from .d1_probe import D1_EXECUTION_AUTHORIZED\n"
     "    if not D1_EXECUTION_AUTHORIZED:\n        raise LowPlyError(",
     f"{T_LP}::test_this_module_reads_NO_OTHER_experiments_gate"),


    # ═════ malformed helper output must VOID, not escape as a traceback ═════
    ("adapter query parse failure re-raised bare", ADAPTER,
     '    except (ValueError, KeyError) as e:\n'
     '        raise HelperOutputError(\n'
     '            f"the helper\'s query output could not be parsed: {e}", p.stdout) from None',
     '    except (ValueError, KeyError):\n        raise',
     f"{T_LP}::test_a_malformed_QUERY_line_is_a_VOID_not_a_traceback"),
    ("adapter replay parse failure re-raised bare", ADAPTER,
     '    except (ValueError, KeyError) as e:\n'
     '        raise HelperOutputError(\n'
     '            f"the helper\'s replay output could not be parsed: {e}", p.stdout) from None',
     '    except (ValueError, KeyError):\n        raise',
     f"{T_LP}::test_a_malformed_REPLAY_dump_is_a_VOID_not_a_traceback"),
    ("lowply query parse failure not translated", LP,
     '    except A.HelperOutputError as e:\n'
     '        raise LowPlyVoidError(\n'
     '            f"{where}: the helper\'s QUERY output could not be parsed',
     '    except ZeroDivisionError as e:\n'
     '        raise LowPlyVoidError(\n'
     '            f"{where}: the helper\'s QUERY output could not be parsed',
     f"{T_LP}::test_a_malformed_QUERY_line_is_a_VOID_not_a_traceback"),
    ("lowply replay parse failure not translated", LP,
     '    except A.HelperOutputError as e:\n'
     '        raise LowPlyVoidError(\n'
     '            f"{label}: the replay output could not be parsed',
     '    except ZeroDivisionError as e:\n'
     '        raise LowPlyVoidError(\n'
     '            f"{label}: the replay output could not be parsed',
     f"{T_LP}::test_a_malformed_REPLAY_dump_is_a_VOID_not_a_traceback"),
    ("lowply POSTCOND parse failure not translated", LP,
     '        posts = A.parse_postconds(out)\n    except (ValueError, KeyError) as e:',
     '        posts = A.parse_postconds(out)\n    except ZeroDivisionError as e:',
     f"{T_LP}::test_a_malformed_POSTCOND_is_a_VOID_not_a_traceback"),
    ("lowply parse VOID carries the whole dump body", LP,
     '            f"{where}: the helper\'s QUERY output could not be parsed ({e}). "\n'
     '            f"T1j reported: {A.helper_failure_excerpt(e.stdout)}. VOID.") from None',
     '            f"{where}: the helper\'s QUERY output could not be parsed ({e}). "\n'
     '            f"T1j reported: {e.stdout}. VOID.") from None',
     f"{T_LP}::test_every_malformed_case_carries_the_bounded_transcript"),
    ("lowply main has no catch-all again", LP,
     '    except Exception as e:                                    # noqa: BLE001',
     '    except ZeroDivisionError as e:',
     f"{T_LP}::test_main_reports_rather_than_escaping_on_an_unexpected_error"),

    ("parse_postconds raises a bare ValueError again", ADAPTER,
     '            raise HelperOutputError(\n'
     '                f"the helper\'s POSTCOND output could not be parsed: line missing "',
     '            raise ValueError(\n'
     '                f"the helper\'s POSTCOND output could not be parsed: line missing "',
     f"{T_LP}::test_a_malformed_POSTCOND_in_the_REPLAY_output_is_a_VOID"),
    ("parse_postconds drops the transcript it attaches", ADAPTER,
     '                f"fields {sorted(missing)}: {line!r}", text)',
     '                f"fields {sorted(missing)}: {line!r}", "")',
     f"{T_LP}::test_parse_postconds_carries_the_transcript_when_it_refuses"),

    ("PostCond construction raises a bare ValueError again", ADAPTER,
     '        except (ValueError, KeyError) as e:\n'
     '            # A COMPLETE line whose numeric fields will not parse.',
     '        except ZeroDivisionError as e:\n'
     '            # A COMPLETE line whose numeric fields will not parse.',
     f"{T_LP}::test_a_non_numeric_POSTCOND_in_the_REPLAY_output_is_a_VOID"),
    ("PostCond parse failure drops its transcript", ADAPTER,
     '                f"the helper\'s POSTCOND output could not be parsed: {e} in "\n'
     '                f"{line!r}", text) from None',
     '                f"the helper\'s POSTCOND output could not be parsed: {e} in "\n'
     '                f"{line!r}", "") from None',
     f"{T_LP}::test_a_non_numeric_POSTCOND_field_carries_the_transcript"),

    # ═════════════ §13 exclusion, D1 parse translation, VOID trace ═════════
    # 🔴 RE-AIMED 2026-09-12. Every §13 test took the `selection` FIXTURE, so
    # §13.3's reconciliation refused the un-excluded counts during SETUP and each
    # one ERRORED without running. `test_the_enumerated_exclusion_IS_APPLIED` was
    # added to call `select_all` in its own body, where that refusal is the
    # test's own failure.
    ("§13 exclusion never applied", SEL,
     '        by_key[k]["rows"] = [r for r in frozen_rows[k] if r["digest"] not in excluded]',
     '        by_key[k]["rows"] = list(frozen_rows[k])',
     f"{T_SEL}::test_the_enumerated_exclusion_IS_APPLIED"),
    ("§12 count reconciliation removed", SEL,
     "            if got != sig[want]:",
     "            if False:",
     f"{T_SEL}::test_a_12_1_count_that_departs_from_the_record_is_refused"),
    ("§13 post-exclusion counts not reconciled", SEL,
     "        if got != want:",
     "        if False:",
     f"{T_SEL}::test_a_post_exclusion_count_that_departs_from_13_3_is_refused"),
    ("§13 accepts the retired seed block", SEL,
     "        if tuple(seed_interval) == RETIRED_SEED_INTERVAL or (",
     "        if False and (",
     f"{T_SEL}::test_the_retired_block_is_refused_as_a_seed_interval"),
    # 🔴 MUTATION REWRITTEN 2026-09-12. `if False:` sent the None case down the
    # `else` branch, which unpacked `lo, hi = None` and raised TypeError inside
    # the `selection` fixture -- the named test never ran. This one ASSIGNS
    # seeds, which is what the label claims, and the test reaches its assertion.
    ("§13 assigns seeds when none were reserved", SEL,
     "    if seed_interval is None:\n        seeds: List[Optional[int]] = [None] * total",
     "    if seed_interval is None:\n        seeds: List[Optional[int]] = "
     "list(range(300000000, 300000000 + total))",
     f"{T_SEL}::test_no_seed_is_assigned_because_no_interval_is_reserved"),
    ("D1 query cap raised back to the frozen 12.4 figure", PROBE,
     "N_POSITIONS = 221\nQUERY_CAP = 1105",
     "N_POSITIONS = 227\nQUERY_CAP = 1135",
     f"{T_PROBE}::test_the_query_cap_is_the_PROSPECTIVE_value_and_its_arithmetic_holds"),
    ("D1 query parse failure not translated", PROBE,
     "        except A.HelperOutputError as e:\n"
     "            # 12.7 treats unreadable output as an instrument failure.",
     "        except ZeroDivisionError as e:\n"
     "            # 12.7 treats unreadable output as an instrument failure.",
     f"{T_PROBE}::test_a_malformed_QUERY_line_is_a_VOID_naming_the_position"),
    ("D1 replay parse failure not translated", PROBE,
     '    except A.HelperOutputError as e:\n'
     '        raise D1VoidError(\n'
     '            f"{task_id}: the replay output could not be parsed',
     '    except ZeroDivisionError as e:\n'
     '        raise D1VoidError(\n'
     '            f"{task_id}: the replay output could not be parsed',
     f"{T_PROBE}::test_a_malformed_REPLAY_dump_is_a_VOID_and_writes_nothing"),
    ("D1 main has no catch-all again", PROBE,
     "    except Exception as e:                                    # noqa: BLE001",
     "    except ZeroDivisionError as e:",
     f"{T_PROBE}::test_d1_main_reports_rather_than_escaping_on_an_unexpected_error"),
    ("the trace stops validating VALUES (name-only again)", VTR,
     "    for name, value in fields.items():",
     "    for name, value in []:",
     f"{T_PROBE}::test_no_measurement_can_be_encoded_through_any_free_form_field"),
    ("the enum check is opened to free text", VTR,
     "            if value not in schema.enums[name]:",
     "            if False:",
     f"{T_PROBE}::test_no_measurement_can_be_encoded_through_any_free_form_field"),
    ("the event enum is opened to free text", VTR,
     "    if event not in schema.events:",
     "    if False:",
     f"{T_PROBE}::test_no_measurement_can_be_encoded_through_any_free_form_field"),
    ("counters accept any type or magnitude", VTR,
     "            if type(value) is not int or not 0 <= value <= cap:",
     "            if False:",
     f"{T_PROBE}::test_counters_must_be_bounded_plain_integers"),
    ("counters accept bools as integers", VTR,
     "            if type(value) is not int or not 0 <= value <= cap:",
     "            if not isinstance(value, int) or not 0 <= value <= cap:",
     f"{T_PROBE}::test_counters_must_be_bounded_plain_integers"),
    ("counter ceilings widened past the run's real limits", PROBE,
     '    "queries_spent": QUERY_CAP,',
     '    "queries_spent": 10 ** 12,',
     f"{T_PROBE}::test_a_counter_beyond_its_own_ceiling_is_refused"),
    ("the schema string is no longer pinned", VTR,
     "            if value != schema.schema:",
     "            if False:",
     f"{T_PROBE}::test_the_schema_string_must_be_exactly_the_declared_one"),
    ("an event may carry extra fields", VTR,
     "    if extra:",
     "    if False:",
     f"{T_PROBE}::test_each_event_carries_EXACTLY_its_declared_fields"),
    ("an event may omit declared fields", VTR,
     '    if missing:\n        raise schema.error(f"a {event!r} line is missing {missing}")',
     '    if False:\n        raise schema.error(f"a {event!r} line is missing {missing}")',
     f"{T_PROBE}::test_each_event_carries_EXACTLY_its_declared_fields"),
    ("the identity strings are readmitted", PROBE,
     '    "position_start": frozenset({"index"}),',
     '    "position_start": frozenset({"index", "task_id", "digest", "ply"}),',
     f"{T_PROBE}::test_the_free_form_identity_fields_are_gone"),
    ("the trace validates AFTER writing", VTR,
     "    check(schema, event, fields)\n    fh.write(",
     "    fh.write(",
     f"{T_PROBE}::test_no_measurement_can_be_encoded_through_any_free_form_field"),
    ("the trace is not written on a VOID", PROBE,
     "    finally:\n"
     "        # ALWAYS, including when the supervisor terminates a hung stage.",
     "    except ZeroDivisionError:\n"
     "        # ALWAYS, including when the supervisor terminates a hung stage.",
     f"{T_PROBE}::test_the_trace_SURVIVES_a_VOID_and_says_how_far_it_got"),
    ("the trace stops counting seeds drawn", PROBE,
     "            drawn += 1",
     "            drawn += 0",
     f"{T_PROBE}::test_the_trace_SURVIVES_a_VOID_and_says_how_far_it_got"),
    ("the trace is not fsynced per line", VTR,
     "    fh.flush()\n    os.fsync(fh.fileno())",
     "    fh.flush()",
     f"{T_PROBE}::test_every_trace_line_is_fsynced"),

    # ═════ the public boundary must require the EXACT cohort, not fit the cap ═
    ("run_d1 accepts any cohort (the ceiling-only defect)", PROBE,
     "    _check_cohort(positions)\n    return _run_d1_unguarded(",
     "    return _run_d1_unguarded(",
     f"{T_PROBE}::test_run_d1_checks_the_cohort_and_the_private_seam_does_not"),
    ("the cohort size check is dropped", PROBE,
     "    if len(positions) != N_POSITIONS:",
     "    if False:",
     f"{T_PROBE}::test_a_short_cohort_is_refused_at_the_public_boundary"),
    ("the cohort ORDER check is dropped", PROBE,
     "    for i, (got_row, want) in enumerate(zip(positions, expected)):",
     "    for i, (got_row, want) in enumerate([]):",
     f"{T_PROBE}::test_a_reordered_cohort_is_refused"),
    ("an excluded row is readmitted to the cohort", PROBE,
     "    if present:",
     "    if False:",
     f"{T_PROBE}::test_a_cohort_containing_an_excluded_row_is_refused"),
    ("the cohort source is no longer hash-pinned", PROBE,
     "    if got != COHORT_SOURCE_SHA256:",
     "    if False:",
     f"{T_PROBE}::test_a_tampered_cohort_source_is_refused"),
    ("the exact-spend check is dropped", PROBE,
     "        if budget.spent != want:",
     "        if False:",
     f"{T_PROBE}::test_an_under_spend_cannot_produce_an_OK_report"),
    ("the expected spend becomes a maximum again", PROBE,
     "        want = len(positions) * QUERIES_PER_POSITION",
     "        want = budget.spent",
     f"{T_PROBE}::test_an_under_spend_cannot_produce_an_OK_report"),
    ("the private seam starts enforcing the cohort too", PROBE,
     "def _run_d1_unguarded(*, positions, paths, out_path, repo_root=\".\", deadline=None,",
     "def _run_d1_unguarded(*, positions, paths, out_path, repo_root=\".\", deadline=None,  # _check_cohort(",
     f"{T_PROBE}::test_run_d1_checks_the_cohort_and_the_private_seam_does_not"),

    ("cohort binds digests only, not the labels", PROBE,
     "        for field in COHORT_IDENTITY_FIELDS:\n"
     "            if not _same(got[field], want[field]):",
     '        for field in ("digest",):\n'
     "            if not _same(got[field], want[field]):",
     f"{T_PROBE}::test_a_row_that_keeps_its_digest_but_alters_its_LABEL_is_refused"),
    ("cohort drops the grouping fields from the identity set", PROBE,
     '                          "role", "opening", "colour_arm", "phase",\n'
     '                          "mover_more_fragmented", "created_threat")',
     '                          "role")',
     f"{T_PROBE}::test_the_expected_cohort_carries_every_identity_and_grouping_field"),
    ("the RAW mover_more_fragmented column is left unbound", PROBE,
     '                          "mover_more_fragmented", "created_threat")',
     '                          "created_threat")',
     f"{T_PROBE}::test_a_row_that_flips_a_RAW_SIGNATURE_COLUMN_is_refused"),
    ("the RAW created_threat column is left unbound", PROBE,
     '                          "mover_more_fragmented", "created_threat")',
     '                          "mover_more_fragmented")',
     f"{T_PROBE}::test_a_row_that_flips_a_RAW_SIGNATURE_COLUMN_is_refused"),
    ("a source field silently drops out of the binding", PROBE,
     '                          "role", "opening", "colour_arm", "phase",',
     '                          "role", "opening", "phase",',
     f"{T_PROBE}::test_SEED_is_the_only_source_field_left_unbound"),
    ("cohort tolerates a row missing an identity field", PROBE,
     "        if field not in row:",
     "        if False:",
     f"{T_PROBE}::test_a_row_missing_an_identity_field_is_refused"),
    ("cohort pins the SEED as an IDENTITY field (the frozen source carries none)", PROBE,
     '                          "mover_more_fragmented", "created_threat")',
     '                          "mover_more_fragmented", "created_threat", "seed")',
     f"{T_PROBE}::test_the_exact_cohort_is_accepted"),

    ("cohort comparison drops type-strictness (== again)", PROBE,
     "            if not _same(got[field], want[field]):",
     "            if got[field] != want[field]:",
     f"{T_PROBE}::test_a_NUMERIC_boolean_is_refused_for_a_frozen_bool"),
    ("the comparison helper stops checking type", PROBE,
     "    return type(a) is type(b) and a == b",
     "    return a == b",
     f"{T_PROBE}::test_the_comparison_helper_binds_type_as_well_as_value"),
    ("bools slip through as ints (isinstance instead of type)", PROBE,
     "    return type(a) is type(b) and a == b",
     "    return isinstance(a, type(b)) and a == b",
     f"{T_PROBE}::test_a_BOOL_forged_for_a_frozen_ZERO_OR_ONE_coordinate_is_refused"),
    ("prefix normalisation coerces through int() again", PROBE,
     "    if isinstance(value, (list, tuple)):\n"
     "        return [_structural(v) for v in value]\n"
     "    return value",
     "    if isinstance(value, (list, tuple)):\n"
     "        return [_structural(v) for v in value]\n"
     "    return int(value) if not isinstance(value, (str, bool)) else value",
     f"{T_PROBE}::test_a_COERCED_prefix_coordinate_is_refused"),
    ("structural normalisation dropped, so a tuple prefix is refused", PROBE,
     "        out[field] = _structural(row[field])",
     "        out[field] = row[field]",
     f"{T_PROBE}::test_a_tuple_prefix_is_still_accepted"),
    ("the corrected 5.4 justification is reinstated as the wrong one", PROBE,
     "#: ⚠ The justification stated here was wrong and is corrected: 5.4 requires the",
     "#: 5.4 records both. The justification stated here was wrong and is: 5.4 requires the",
     f"{T_PROBE}::test_the_report_does_not_claim_5_4_requires_the_raw_columns"),


    # ═════════ the §14 seed-interval handoff: ONE canonical source ═════════
    ("the interval is RETYPED as a literal in d1_probe", PROBE,
     "SEED_INTERVAL = SEL.SEED_INTERVAL",
     "SEED_INTERVAL = (202615000, 202615221)",
     f"{T_PROBE}::test_the_interval_has_ONE_canonical_source"),
    ("the canonical interval points back at the RETIRED block", SEL,
     "SEED_INTERVAL = (202615000, 202615221)",
     "SEED_INTERVAL = (202614000, 202614227)",
     f"{T_PROBE}::test_the_canonical_interval_is_14s_reservation_sized_for_13"),
    ("the canonical interval is sized 227 again, not 221", SEL,
     "SEED_INTERVAL = (202615000, 202615221)",
     "SEED_INTERVAL = (202615000, 202615227)",
     f"{T_PROBE}::test_the_canonical_interval_is_14s_reservation_sized_for_13"),

    # 🔴 THE BLANKET-REPLACE DEFECT: the retirement constant moved with the
    # reservation, so the guard compares the new block against itself and the
    # OLD retired block is silently readmitted.
    ("RETIRED_SEED_INTERVAL moved to the new block", SEL,
     "RETIRED_SEED_INTERVAL = (202614000, 202614227)",
     "RETIRED_SEED_INTERVAL = (202615000, 202615221)",
     f"{T_PROBE}::test_the_RETIRED_interval_is_a_SEPARATE_constant_THAT_DID_NOT_MOVE"),
    ("RETIRED_SEED_INTERVAL moved -- caught by the REFUSAL, not the constant", SEL,
     "RETIRED_SEED_INTERVAL = (202614000, 202614227)",
     "RETIRED_SEED_INTERVAL = (202615000, 202615221)",
     f"{T_SEL}::test_selection_still_REFUSES_the_retired_block"),
    ("the retirement guard is removed from select_all", SEL,
     "        if tuple(seed_interval) == RETIRED_SEED_INTERVAL or (\n"
     "                lo < RETIRED_SEED_INTERVAL[1] and hi > RETIRED_SEED_INTERVAL[0]):",
     "        if False:",
     f"{T_SEL}::test_selection_still_REFUSES_the_retired_block"),
    ("the runtime seed check admits the retired block too", PROBE,
     "    if not isinstance(seed, int) or isinstance(seed, bool) or not lo <= seed < hi:",
     "    if not isinstance(seed, int) or isinstance(seed, bool) or not 202614000 <= seed < hi:",
     f"{T_PROBE}::test_every_seed_of_the_RETIRED_block_is_refused_by_the_runtime_check"),
    # 🔴 RE-TARGETED 2026-09-08. This named `test_the_REAL_registry_still_refuses_
    # the_new_block`, which the registration INVERTED and renamed -- so the control
    # was orphaned and its node id no longer existed. The clean-baseline check
    # caught it (pytest exits nonzero for a missing node id, which would otherwise
    # have scored REJECTED for free). It also could not bind at the real registry
    # any more: with the block registered, `missing = []` changes nothing there.
    # A test that STRIPS the block is the only place this defect is observable.
    ("the registration barrier computes nothing and so is always satisfied", PROBE,
     "    missing = [s for s in range(lo, hi) if not REF.seed_is_accounted(s)]",
     "    missing = []",
     f"{T_PROBE}::test_a_PARTLY_registered_block_is_still_refused"),


    # ═══════════════ H1 head-to-head viability screen ═══════════════
    ("bind_results design becomes defaultable", L0R,
     "                 *, design: Design):",
     "                 *, design: Design = L0_DESIGN):",
     f"{T_H1}::test_bind_results_REQUIRES_a_design_and_has_no_default"),
    ("H1 silently reverts to L0's 4 repetitions", H1R,
     "N_REPS = 14", "N_REPS = 4",
     f"{T_H1}::test_the_design_is_8_openings_2_arms_14_reps"),
    ("the viability threshold moves to parity", H1R,
     "VIABILITY_THRESHOLD = 0.75", "VIABILITY_THRESHOLD = 0.5",
     f"{T_H1}::test_the_threshold_is_the_predeclared_0_75"),

    # 🔴 THE BOUNDARY: an upper bound exactly AT the threshold does not exclude it.
    ("the verdict admits an upper bound AT the threshold", H1R,
     "    if hi < VIABILITY_THRESHOLD:", "    if hi <= VIABILITY_THRESHOLD:",
     f"{T_H1}::test_the_verdict_is_where_0_75_falls_relative_to_the_interval"),
    ("the verdict reads the bounds in the wrong order", H1R,
     "    if lo >= VIABILITY_THRESHOLD:", "    if hi >= VIABILITY_THRESHOLD:",
     f"{T_H1}::test_the_verdict_follows_the_predeclared_bands"),
    ("the decisive bands are hardcoded instead of derived", H1R,
     "    half_width = L0.hoeffding_interval(n / 2, n)[1] - 0.5",
     "    half_width = 0.0907",
     f"{T_H1}::test_the_decisive_bands_are_DERIVED_not_typed"),

    # 🔴 THE ANTI-FABRICATION CONTROL: the hand-typed digest this module carried
    # while it was being written. Plausible, 64 hex characters, matched nothing.
    ("the task digest is a hand-typed constant again", H1R,
     'H1_TASK_DIGEST = "23e3fa129b85c2669ef6d974e99c3fd0106735377298bc80c272346376db5c74"',
     'H1_TASK_DIGEST = "6ff6b6c69bbcbba49b4b1f6afc4b6e7b39ff3ac8f4a7f6bbb43e1e0a41b1a7dc"',
     f"{T_H1}::test_the_task_digest_constant_IS_the_digest_of_the_built_tasks"),
    ("H1 reuses L0's SPENT and retired seed block", H1R,
     "H1_SEED_BLOCK = (202617000, 202617224)",
     "H1_SEED_BLOCK = (202613000, 202613224)",
     f"{T_H1}::test_the_ATTEMPT2_block_is_ACCOUNTED_EXPOSED_224_and_RETIRED_WHOLE"),

    # 🔴 THE BLOCK THE REGISTRIES CANNOT SEE: overlapping D1's paper reservation
    # passes every registry check, because §14's block is in no registry.
    ("H1's block overlaps D1's PAPER reservation", H1R,
     "H1_SEED_BLOCK = (202617000, 202617224)",
     "H1_SEED_BLOCK = (202615100, 202615324)",
     f"{T_H1}::test_the_H1_block_is_disjoint_from_the_D1_paper_reservation"),
    ("a shared vocabulary is RETYPED instead of re-exported", H1R,
     "WINNERS = L0.WINNERS", 'WINNERS = ("red", "black", None)',
     f"{T_H1}::test_H1_REDECLARES_ONLY_WHAT_DIFFERS_FROM_L0"),
    ("H1 acquires an early stop", H1R,
     '    """Always False. H1 plays all 224 games."""\n    return False',
     '    """Always False. H1 plays all 224 games."""\n    return True',
     f"{T_H1}::test_there_is_no_early_stop"),
    ("cap saturation stops refusing a rate", H1R,
     "    if caps > CAP_NO_RATE_THRESHOLD:", "    if False:",
     f"{T_H1}::test_cap_saturation_yields_NO_RATE_AND_NO_VERDICT"),
    ("the reporter binds H1 results to the 64-game design", H1R,
     "    pairs, why = L0.bind_results(results, tasks, design=H1_DESIGN)",
     "    pairs, why = L0.bind_results(results, tasks, design=L0.L0_DESIGN)",
     f"{T_H1}::test_a_complete_result_set_reports_and_carries_its_verdict"),
    ("the verdict is read off the WILSON interval", H1R,
     "    verdict = viability_verdict(hl, hh)", "    verdict = viability_verdict(wl, wh)",
     f"{T_H1}::test_the_verdict_reads_the_HOEFFDING_interval_and_nothing_else"),
    ("the no-pooling prohibition is deleted", H1R,
     '    "any pooling of H1 with L0, or any combined rate across the two matches; they "',
     '    "POOLING IS FINE ACTUALLY; they "',
     f"{T_H1}::test_H1_inherits_every_COUNT_FREE_claim_recounts_two_and_adds_two"),
    ("the repetition LABELS stop being checked", H1P,
     "        if reps != list(range(RULES.N_REPS)):", "        if False:",
     f"{T_H1}::test_a_cell_with_the_wrong_repetition_LABELS_is_refused"),
    ("an incomplete match is reported instead of refused", H1R,
     "    if pairs is None:", "    if pairs is None and False:",
     f"{T_H1}::test_an_incomplete_match_is_REFUSED_not_reported"),




    # ═════════ H1 review corrections: composed rules, own denominators ═════════

    # 🔴 THE ORIGINAL DEFECT: append instead of compose. Both seed rules stay
    # active and every valid H1 seed is outside the L0 block.
    ("H1 APPENDS its seed rule to L0's instead of composing", H1R,
     'H1_ABORT_RULES = L0.abort_rules("H1") + (\n'
     '    "the whole-run wall-clock cap being exceeded",\n)',
     'H1_ABORT_RULES = L0.L0_ABORT_RULES + (\n'
     '    "the whole-run wall-clock cap being exceeded",\n'
     '    "any seed outside the reserved H1 block",\n)',
     f"{T_H1}::test_H1s_abort_rules_name_H1s_BLOCK_AND_ONLY_H1s"),
    ("the block name stops reaching the seed rule", L0R,
     '        f"any seed outside the reserved {block_name} block, or any seed used twice",',
     '        "any seed outside the reserved L0 block, or any seed used twice",',
     f"{T_H1}::test_H1s_abort_rules_name_H1s_BLOCK_AND_ONLY_H1s"),
    ("L0's own abort rules change under the factoring", L0R,
     '        "any failure to write or fsync a durable record",',
     '        "any failure to write a durable record",',
     f"{T_H1}::test_L0s_ABORT_RULES_are_unchanged_by_the_factoring"),

    # the seed rule as a PREDICATE -- prose cannot be run
    ("the H1 seed predicate admits the L0 block too", H1R,
     "    return not lo <= int(seed) < hi", "    return not 202613000 <= int(seed) < hi",
     f"{T_H1}::test_an_H1_SEED_is_accepted_and_an_L0_BLOCK_seed_is_refused"),
    ("the H1 seed predicate refuses H1's own seeds", H1R,
     "    return not lo <= int(seed) < hi", "    return lo <= int(seed) < hi",
     f"{T_H1}::test_an_H1_SEED_is_accepted_and_an_L0_BLOCK_seed_is_refused"),

    # 🔴 THE OTHER ORIGINAL DEFECT: inherit the count-bearing prohibitions whole
    ("H1 INHERITS L0's count-bearing prohibitions", H1R,
     "FORBIDDEN_CLAIMS = L0.forbidden_claims(N_GAMES, N_OPENINGS, N_ARMS) + (",
     "FORBIDDEN_CLAIMS = L0.FORBIDDEN_CLAIMS + (",
     f"{T_H1}::test_NO_H1_forbidden_claim_carries_an_L0_DENOMINATOR"),
    ("the independence prohibition hardcodes 64 again", L0R,
     'return (f"any statement that the {n_games} games ARE independent, or that "',
     'return (f"any statement that the 64 games ARE independent, or that "',
     f"{T_H1}::test_NO_H1_forbidden_claim_carries_an_L0_DENOMINATOR"),
    ("the per-cell denominators are swapped", L0R,
     "            per_cell_prohibition(n_games // n_openings, n_games // n_arms),",
     "            per_cell_prohibition(n_games // n_arms, n_games // n_openings),",
     f"{T_H1}::test_NO_H1_forbidden_claim_carries_an_L0_DENOMINATOR"),
    ("L0's own forbidden claims change under the factoring", L0R,
     '    "any Elo figure, or any conversion of this rate into one",',
     '    "any Elo number, or any conversion of this rate into one",',
     f"{T_H1}::test_L0s_FORBIDDEN_CLAIMS_are_unchanged_by_the_factoring"),

    # the frozen artifact must not drift back to the superseded one
    ("the plan pin reverts to the SUPERSEDED v1 artifact", H1P,
     '               "06_h1_plan_v4.json")', '               "06_h1_plan_v2.json")',
     f"{T_H1}::test_the_frozen_plan_STATES_the_cap_saturation_branch"),


    # ═════ [P2] H1's NON-ABORT rules must describe H1's protocol, not L0's ═════
    ("H1 INHERITS L0's non-abort rules whole", H1R,
     "NOT_ABORT_RULES = (\n"
     '    f"cap-termination saturation: caps never stop an H1 match',
     "NOT_ABORT_RULES = L0.NOT_ABORT_RULES\nUNUSED = (\n"
     '    f"cap-termination saturation: caps never stop an H1 match',
     f"{T_H1}::test_H1s_NON_ABORT_RULES_CONTAIN_NO_L0_WORDING"),
    ("H1 claims it has NO band, as L0 does", H1R,
     'f"score saturation: H1 HAS a band -- the {VIABILITY_THRESHOLD} viability "',
     'f"score saturation: H1 measures a rate and has no band to saturate "',
     f"{T_H1}::test_H1s_non_abort_rules_state_H1s_ACTUAL_protocol"),
    ("the early-stop prohibition is softened", H1R,
     "    L0.EARLY_STOP_NOT_ABORT_RULE,\n)",
     '    "any early stop after the verdict is clear",\n)',
     f"{T_H1}::test_the_substantive_early_stop_PROHIBITION_survives_verbatim"),
    ("L0's own non-abort rules change under the extraction", L0R,
     '    "score saturation: L0 measures a rate and has no band to saturate",',
     '    "score saturation: L0 measures a rate and has no band",',
     f"{T_H1}::test_L0s_NOT_ABORT_RULES_are_unchanged"),
    ("the artifact keeps L0-only wording", H1P,
     '               "06_h1_plan_v4.json")', '               "06_h1_plan_v2.json")',
     f"{T_H1}::test_NO_L0_ONLY_WORDING_REACHES_THE_FROZEN_H1_ARTIFACT"),


    # ═════ the corrected barrier claim must NOTICE when a barrier appears ═════
    ("an H1 gate appears without the claim being revisited", H1R,
     "VIABILITY_THRESHOLD = 0.75",
     "H1_EXECUTION_AUTHORIZED = False\nVIABILITY_THRESHOLD = 0.75",
     f"{T_H1}::test_the_DESIGN_layer_still_declares_no_gate_and_no_barrier"),
    ("an H1 registration precondition appears silently", H1P,
     "class H1PlanError(Exception):",
     "def _check_seed_registration():\n    pass\n\n\nclass H1PlanError(Exception):",
     f"{T_H1}::test_the_DESIGN_layer_still_declares_no_gate_and_no_barrier"),


    # ═══════════════════ the H1 runner: two independent barriers ═══════════════
    ("the H1 gate is flipped open", H1RUN,
     "H1_EXECUTION_AUTHORIZED = False", "H1_EXECUTION_AUTHORIZED = True",
     f"{T_H1R}::test_the_gate_is_false_as_published"),
    ("the H1 gate no longer binds the match path", H1RUN,
     "    if not H1_EXECUTION_AUTHORIZED:", "    if False:",
     f"{T_H1R}::test_THE_GATE_fires_before_plan_load_setup_recorder_or_play"),
    ("the registration barrier no longer binds", H1RUN,
     "    missing = [s for s in range(lo, hi) if not REF.seed_is_accounted(s)]",
     "    missing = []",
     f"{T_H1R}::test_THE_REGISTRATION_BARRIER_fires_before_plan_load_setup_recorder_or_play"),
    ("the registration barrier checks only the endpoints", H1RUN,
     "    missing = [s for s in range(lo, hi) if not REF.seed_is_accounted(s)]",
     "    missing = [s for s in (lo, hi - 1) if not REF.seed_is_accounted(s)]",
     f"{T_H1R}::test_a_PARTLY_registered_block_is_still_refused"),

    # 🔴 ORDER IS THE GUARANTEE: a refusal that has already written has not refused.
    ("the barriers move BELOW the plan load and the recorder", H1RUN,
     "    if match:\n        check_gate()\n        check_seed_registration()",
     "    if match:\n        pass",
     f"{T_H1R}::test_THE_GATE_fires_before_plan_load_setup_recorder_or_play"),
    ("the gate opens the registration barrier too", H1RUN,
     "        check_gate()\n        check_seed_registration()",
     "        check_gate()",
     f"{T_H1R}::test_THE_REGISTRATION_BARRIER_fires_before_plan_load_setup_recorder_or_play"),
    ("registering the block opens the gate too", H1RUN,
     "        check_gate()\n        check_seed_registration()",
     "        check_seed_registration()",
     f"{T_H1R}::test_REGISTERING_THE_BLOCK_DOES_NOT_OPEN_THE_GATE"),

    # the frozen design
    ("match mode accepts a supplied plan", H1RUN,
     "    if match and _plan_path is not None:", "    if False:",
     f"{T_H1R}::test_match_mode_REFUSES_a_supplied_plan_path"),
    ("the content binding narrows to the digest dimensions (dead again)", H1RUN,
     "        if c is None or any(t.get(k) != c.get(k) for k in set(t) | set(c)):",
     "        if c is None or any(t.get(k) != c.get(k)\n                            for k in RULES.L0.L0_TASK_DIMENSIONS):",
     f"{T_H1R}::test_a_task_renamed_onto_synthetic_CONTENT_is_refused"),
    ("the seed-block check is dropped from the PLAN validator", H1P,
     "        if outside:\n            raise H1PlanError(f\"{t['task_id']} seed {t['seed']} outside [{lo}, {hi})\")",
     "        if False:\n            raise H1PlanError(f\"{t['task_id']} seed {t['seed']} outside [{lo}, {hi})\")",
     f"{T_H1R}::test_a_seed_outside_the_reserved_block_is_refused"),
    ("the whole-run deadline is widened", H1RUN,
     "RUN_DEADLINE_S = 180 * 60", "RUN_DEADLINE_S = 180 * 6000",
     f"{T_H1R}::test_the_frozen_limits_are_the_cards"),
    ("the per-call timeout is dropped", H1RUN,
     "PER_CALL_TIMEOUT_S = D1.PER_QUERY_TIMEOUT_S", "PER_CALL_TIMEOUT_S = None",
     f"{T_H1R}::test_the_frozen_limits_are_the_cards"),
    ("the cooperative deadline check is removed", H1RUN,
     "    deadline.check(where)", "    pass",
     f"{T_H1R}::test_a_deadline_breach_is_a_VOID_with_no_report"),
    ("the deadline is never checked between games", H1RUN,
     '                _check_deadline(deadline, f"before game {index}")',
     "                pass",
     f"{T_H1R}::test_a_deadline_breach_is_a_VOID_with_no_report"),

    # create-only outputs and the non-analytic trace
    ("the trace file stops being create-only", H1RUN,
     "    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)",
     "    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)",
     f"{T_H1R}::test_the_trace_file_is_CREATE_ONLY"),
    ("the trace admits an identity string", H1RUN,
     '    "task_start": frozenset({"index"}),',
     '    "task_start": frozenset({"index", "task_id"}),',
     f"{T_H1R}::test_the_trace_carries_ONLY_counters_and_closed_enums"),
    ("the trace validates AFTER the write", VTR,
     "    check(schema, event, fields)\n    fh.write(",
     "    fh.write(",
     f"{T_H1R}::test_a_refusal_writes_NOTHING_to_the_trace"),
    ("the trace stops recording how far a VOID run got", H1RUN,
     '            _trace(tfh, event="run_end",\n'
     '                   verdict="INTERRUPTED" if interrupted else "VOID",\n'
     '                   games_completed=completed)',
     "            pass",
     f"{T_H1R}::test_the_trace_records_HOW_FAR_a_void_run_got_and_nothing_else"),
    ("cap saturation is reported as a verdict after all", H1RUN,
     '        if report.get("outcome") == CAP_SATURATED_NO_RATE:',
     "        if False:",
     f"{T_H1R}::test_cap_saturation_exits_OK_and_publishes_NO_VERDICT"),

    # 🔴 the ban that named functions which do not exist
    ("the early-stop ban names a function that does not exist", H1R,
     '    "e4_screen_rules.classify_joint",',
     '    "e4_screen_rules.should_continue",',
     f"{T_H1R}::test_the_screen_decision_functions_still_exist_and_are_not_imported"),


    # ═════════ H1 runner review: public path, setup, alarm, VOID contract ═════
    ("the match is hidden behind a mode list again", H1RUN,
     'MODES = ("qualify", MATCH_MODE)', 'MODES = ("qualify",)',
     f"{T_H1R}::test_the_PUBLIC_entry_point_reaches_the_GATE_not_a_mode_list"),
    ("the public match path supplies no production setup", H1RUN,
     "_setup_factory=_production_setup if mode == MATCH_MODE else None)",
     "_setup_factory=None)",
     f"{T_H1R}::test_the_public_match_path_supplies_the_PRODUCTION_setup"),
    ("qualification builds production collaborators too", H1RUN,
     "_setup_factory=_production_setup if mode == MATCH_MODE else None)",
     "_setup_factory=_production_setup)",
     f"{T_H1R}::test_the_public_match_path_supplies_the_PRODUCTION_setup"),
    ("T1j loses the frozen per-call timeout in the setup", H1RUN,
     "                                 timeout_s=PER_CALL_TIMEOUT_S)",
     "                                 timeout_s=None)",
     f"{T_H1R}::test_the_production_setup_wires_every_qualified_collaborator"),
    ("the setup keeps the REFUSING binder", H1RUN,
     '            "binder": INT.make_binder(runtime, ctx),            # the E3b binder',
     '            "binder": H._refuse_binder,',
     f"{T_H1R}::test_the_production_setup_wires_every_qualified_collaborator"),
    ("the openings come from H1's plan again (KeyError on the real path)", H1RUN,
     '        openings = PLAN.load_source_plan()["openings"]',
     '        openings = PLAN.load_h1_plan()["openings"]',
     f"{T_H1R}::test_the_production_setup_wires_every_qualified_collaborator"),
    ("the incumbent is loaded per game instead of once", H1RUN,
     '        evaluator = SCREEN_CMD._default_load_evaluator(".")     # the incumbent, ONCE',
     '        evaluator = [SCREEN_CMD._default_load_evaluator("."),\n'
     '                     SCREEN_CMD._default_load_evaluator(".")][0]',
     f"{T_H1R}::test_the_production_setup_loads_the_incumbent_EXACTLY_ONCE"),

    # 🔴 the asynchronous alarm must not escape as D1's exception
    ("the SIGALRM void escapes as D1's exception", H1RUN,
     "            if isinstance(e, D1.D1VoidError) and not isinstance(e, H1VoidError):",
     "            if False:",
     f"{T_H1R}::test_a_QUALIFY_mode_deadline_breach_is_STILL_translated"),
    ("the supervisor is never armed", H1RUN,
     "    supervisor = D1._supervisor(deadline) if _supervise else contextlib.nullcontext()",
     "    supervisor = contextlib.nullcontext()",
     f"{T_H1R}::test_the_SIGALRM_supervisor_raises_H1s_void_not_D1s"),

    # the VOID record contract
    ("the VOID diagnostic stops naming the position", H1RUN,
     '                rec.emit_terminal({"record_type": ("interrupt_diagnostic" if interrupted\n'
     '                                                   else "void_diagnostic"),',
     '                _ = ({"record_type": ("interrupt_diagnostic" if interrupted\n'
     '                                                   else "void_diagnostic"),',
     f"{T_H1R}::test_the_VOID_diagnostic_NAMES_THE_POSITION_structurally"),
    ("the VOID diagnostic drops the helper transcript", H1RUN,
     '    d["helper_excerpt"] = _bounded_excerpt(error)',
     '    d["helper_excerpt"] = None',
     f"{T_H1R}::test_the_VOID_diagnostic_NAMES_THE_POSITION_structurally"),
    ("the VOID diagnostic carries the WHOLE transcript", H1RUN,
     '    d["helper_excerpt"] = _bounded_excerpt(error)',
     '    d["helper_excerpt"] = getattr(error, "stdout", None) or str(error)',
     f"{T_H1R}::test_the_VOID_diagnostic_excerpt_is_BOUNDED"),
    ("the ply counter stops observing durable ply records", H1RUN,
     '        if record.get("record_type") in ("ply", "opening_bound"):',
     "        if False:",
     f"{T_H1R}::test_the_VOID_diagnostic_records_the_PLY_it_died_on"),
    ("a partial vector can produce a report after all", H1R,
     "    pairs, why = L0.bind_results(results, tasks, design=H1_DESIGN)",
     "    pairs, why = (list(zip(results, tasks)), None)",
     f"{T_H1R}::test_the_retained_rows_CANNOT_become_a_result"),


    # ══════════ H1 runner review 2: the five execution-path findings ══════════
    ("the agent's own query timeout is dropped again", H1RUN,
     "                t1j_timeout_s=PER_CALL_TIMEOUT_S,\n", "",
     f"{T_H1R}::test_the_production_setup_wires_every_qualified_collaborator"),
    ("compilation gets a SECOND deadline (two clocks again)", H1RUN,
     "        artifacts = D1._default_compile(deadline, paths=paths)",
     "        artifacts = D1._default_compile(D1.Deadline(RUN_DEADLINE_S).start(),\n"
     "                                        paths=paths)",
     f"{T_H1R}::test_the_production_setup_wires_every_qualified_collaborator"),
    ("the setup factory is never invoked", H1RUN,
     "    if _setup_factory is not None:", "    if False:",
     f"{T_H1R}::test_the_setup_FACTORY_is_invoked_with_the_running_clock"),

    # 🔴 the excerpt must survive the AbortError wrapper real failures pass through
    ("the excerpt reads only error.stdout again", H1RUN,
     "    seen, err = set(), error\n"
     "    while err is not None and id(err) not in seen:",
     "    seen, err = set(), error\n"
     "    while False and err is not None and id(err) not in seen:",
     f"{T_H1R}::test_the_excerpt_walks_the_SUPPRESSED_context_chain"),
    ("the excerpt drops the abort-message fallback", H1RUN,
     "    return message[:A.FAILURE_EXCERPT_CHARS]", "    return None",
     f"{T_H1R}::test_the_excerpt_falls_back_to_the_bounded_ABORT_MESSAGE"),
    ("the abort-message fallback is unbounded", H1RUN,
     "    return message[:A.FAILURE_EXCERPT_CHARS]", "    return message",
     f"{T_H1R}::test_the_excerpt_falls_back_to_the_bounded_ABORT_MESSAGE"),

    # 🔴 the ply must be the one being ATTEMPTED
    ("the ply tracker ignores opening_bound again", H1RUN,
     '        if record.get("record_type") in ("ply", "opening_bound"):',
     '        if record.get("record_type") == "ply":',
     f"{T_H1R}::test_a_failure_on_the_FIRST_searched_move_still_names_a_PLY"),
    ("the binder wrapper stops noting the attempted ply", H1RUN,
     "                rec.note_bind(ply)\n                return _b(task, state, ply, move)",
     "                return _b(task, state, ply, move)",
     f"{T_H1R}::test_a_BINDER_failure_after_a_move_names_the_ply_it_was_ATTEMPTING"),

    # 🔴 recorder construction must be inside the protected boundary
    ("the recorder is built OUTSIDE the protected try again", H1RUN,
     "        rec = None\n        try:\n            try:\n"
     "                rec = _PlyCountingRecorder(results_path)",
     "        rec = _PlyCountingRecorder(results_path)\n        try:\n            try:\n"
     "                pass",
     f"{T_H1R}::test_a_MIDRUN_recorder_failure_is_a_VOID_and_the_TRACE_AGREES"),


    # ═══════ the output preflight, and the verdict the exception must match ═══
    ("the output preflight is removed", H1RUN,
     "    check_output_paths(results_path, trace_path, require_trace=match)", "    pass",
     f"{T_H1R}::test_a_PREEXISTING_output_path_is_a_PRECONDITION_refusal"),
    ("the preflight ignores the trace path", H1RUN,
     '    for label, path in (("results", results_path), ("trace", trace_path)):',
     '    for label, path in (("results", results_path),):',
     f"{T_H1R}::test_a_PREEXISTING_TRACE_path_is_refused_too"),
    ("a precondition refusal is reported as a VOID", H1RUN,
     "            raise H1Error(\n"
     '                f"the {label} path already exists: {path}"',
     "            raise H1VoidError(\n"
     '                f"the {label} path already exists: {path}"',
     f"{T_H1R}::test_a_PREEXISTING_output_path_is_a_PRECONDITION_refusal"),
    ("a mid-run recorder failure escapes unclassified again", H1RUN,
     "            except H.HarnessError as e:", "            except ZeroDivisionError as e:",
     f"{T_H1R}::test_a_MIDRUN_recorder_failure_is_a_VOID_and_the_TRACE_AGREES"),


    # ═════════ a match needs a trace, and two DISTINCT output files ══════════
    ("match mode accepts a missing trace path again", H1RUN,
     "    if require_trace and not trace_path:", "    if False:",
     f"{T_H1R}::test_MATCH_MODE_REQUIRES_a_trace_path"),
    ("the trace requirement is not applied to the match", H1RUN,
     "    check_output_paths(results_path, trace_path, require_trace=match)",
     "    check_output_paths(results_path, trace_path, require_trace=False)",
     f"{T_H1R}::test_MATCH_MODE_REQUIRES_a_trace_path"),
    ("results and trace may be the same file", H1RUN,
     "    if trace_path and _canonical(results_path) == _canonical(trace_path):",
     "    if False:",
     f"{T_H1R}::test_THE_TWO_OUTPUT_PATHS_MUST_BE_DIFFERENT_FILES"),
    ("the paths are compared WITHOUT canonicalisation", H1RUN,
     "    return os.path.realpath(os.path.abspath(path))", "    return path",
     f"{T_H1R}::test_TWO_NAMES_FOR_ONE_FILE_are_refused_after_canonicalisation"),
    ("canonicalisation stops resolving symlinks", H1RUN,
     "    return os.path.realpath(os.path.abspath(path))",
     "    return os.path.abspath(path)",
     f"{T_H1R}::test_a_SYMLINKED_trace_path_is_refused_too"),


    # 🔴 a dangling link is a present directory entry: `exists` follows, `lexists` does not
    ("the output precheck follows symlinks again (exists, not lexists)", H1RUN,
     "        if path and os.path.lexists(path):", "        if path and os.path.exists(path):",
     f"{T_H1R}::test_a_DANGLING_RESULTS_link_is_refused_by_the_precondition"),
    ("the precheck stops agreeing with create-exclusive open", H1RUN,
     "        if path and os.path.lexists(path):", "        if path and os.path.exists(path):",
     f"{T_H1R}::test_the_precondition_agrees_with_what_CREATE_EXCLUSIVE_open_would_do"),
    ("only the results name is checked for existence", H1RUN,
     '    for label, path in (("results", results_path), ("trace", trace_path)):\n'
     "        if path and os.path.lexists(path):",
     '    for label, path in (("results", results_path),):\n'
     "        if path and os.path.lexists(path):",
     f"{T_H1R}::test_a_DANGLING_TRACE_link_is_refused_by_the_precondition"),


    ("the public docstring denies that match mode is selectable", H1RUN,
     "    ⚠ CORRECTED: this said \"MATCH MODE IS NOT SELECTABLE HERE\", which stopped",
     "    MATCH MODE IS NOT SELECTABLE HERE. This said, which stopped",
     f"{T_H1R}::test_the_public_docstring_DESCRIBES_THE_TESTED_BEHAVIOUR"),


    # ═════════ H1 seed preparation: registered ACCOUNTED and nothing else ═════
    ("the H1 block is un-registered from ACCOUNTED", REF_SRC,
     "    (202616000, 202616224),          # H1 HEAD-TO-HEAD VIABILITY SCREEN. 224",
     "    # removed (202616000, 202616224),  # H1 HEAD-TO-HEAD VIABILITY SCREEN. 224",
     f"{T_H1}::test_the_ATTEMPT1_block_is_ACCOUNTED_60_EXPOSED_and_RETIRED_WHOLE"),
    ("the H1 exposure overstates the draws (all 224, not the 60 taken)", REF_SRC,
     "    (202616000, 202616060),          # THE H1 MATCH, run once 2026-09-05 and",
     "    (202616000, 202616224),          # THE H1 MATCH, run once 2026-09-05 and",
     f"{T_H1}::test_the_ATTEMPT1_block_is_ACCOUNTED_60_EXPOSED_and_RETIRED_WHOLE"),
    ("the H1 retirement covers only the seeds actually drawn", REF_SRC,
     "    (202616000, 202616224),          # THE H1 BLOCK, retired WHOLE 2026-09-05",
     "    (202616000, 202616060),          # THE H1 BLOCK, retired WHOLE 2026-09-05",
     f"{T_H1}::test_the_ATTEMPT1_block_is_ACCOUNTED_60_EXPOSED_and_RETIRED_WHOLE"),
    # 🔴 REPLACED 2026-09-08. This injected "D1's paper reservation is registered
    # too" -- the exact edit the seed-preparation authorization has now made
    # legitimately, so as a DEFECT it no longer exists and the test it named has
    # been inverted with it. The H1-side property that survives is scope: an H1
    # round must not reach outside its own block, and D1's must stay accounted
    # only. Injecting a draw D1 has not made still violates it.
    ("D1's spent block loses its ACCOUNTED entry, from the H1 side", REF_SRC,
     "    (202615000, 202615221),          # D1 SAME-POSITION INTERROGATION, §14 RETRY.",
     "    # (202615000, 202615221),        # D1 SAME-POSITION INTERROGATION, §14 RETRY.",
     f"{T_H1}::test_D1s_reservation_is_SPENT_and_DISJOINT_from_both_H1_blocks"),
    ("an H1 block is made to overlap D1's spent block", H1R,
     "H1_SEED_BLOCK = (202617000, 202617224)",
     "H1_SEED_BLOCK = (202615100, 202615324)",
     f"{T_H1}::test_D1s_reservation_is_SPENT_and_DISJOINT_from_both_H1_blocks"),
    ("registering the block also opens the gate", H1RUN,
     "H1_EXECUTION_AUTHORIZED = False", "H1_EXECUTION_AUTHORIZED = True",
     f"{T_H1}::test_the_DESIGN_layer_still_declares_no_gate_and_no_barrier"),
    # (removed: BYTE-IDENTICAL to "an H1 gate appears without the claim being
    #  revisited" -- same file, anchor, replacement and target test. Two labels
    #  over one injection counted one defect twice.)


    # ═════ 2026-09-05: prefs attribution, and outcome/verdict agreement ══════
    ("a match abort escapes without becoming a VOID", H1RUN,
     "            elif match and isinstance(e, Exception) and not isinstance(e, H1VoidError):",
     "            elif False:",
     f"{T_H1R}::test_an_ABORT_on_the_match_path_leaves_as_a_VOID_so_the_TYPE_AGREES"),
    ("qualification aborts are relabelled as voids too", H1RUN,
     "            elif match and isinstance(e, Exception) and not isinstance(e, H1VoidError):",
     "            elif isinstance(e, Exception) and not isinstance(e, H1VoidError):",
     f"{T_H1R}::test_QUALIFICATION_aborts_are_NOT_relabelled_as_voids"),
    ("a KeyboardInterrupt is relabelled as an instrument failure", H1RUN,
     "            elif match and isinstance(e, Exception) and not isinstance(e, H1VoidError):",
     "            elif match and not isinstance(e, H1VoidError):",
     f"{T_H1R}::test_a_KEYBOARD_INTERRUPT_is_NOT_relabelled_as_an_instrument_failure"),
    ("the diagnostic records the TRANSLATION instead of the cause", H1RUN,
     "                                   **_void_diagnostic(tsk, idx, rec.last_ply,\n"
     "                                                      original),",
     "                                   **_void_diagnostic(tsk, idx, rec.last_ply, e),",
     f"{T_H1R}::test_the_diagnostic_records_the_ORIGINAL_failure_not_the_translation"),
    ("a prefs failure is reported without the attribution note", H1RUN,
     '        d["prefs_attribution"] = PREFS_ATTRIBUTION_NOTE',
     "        pass",
     f"{T_H1R}::test_a_PREFS_failure_records_the_surface_and_REFUSES_to_attribute_it"),
    ("the attribution note claims T1j did it", H1RUN,
     '    "prefs_ok=false means the helper\'s before/after comparison of "',
     '    "prefs_ok=false means T1j mutated the preference store. "',
     f"{T_H1R}::test_a_PREFS_failure_records_the_surface_and_REFUSES_to_attribute_it"),
    ("the note is attached to every failure, not just prefs ones", H1RUN,
     '    if ((observed is not None and observed["prefs_ok"] is False)\n'
     '            or "prefs_ok=false" in (d["helper_excerpt"] or "")):',
     "    if True:",
     f"{T_H1R}::test_a_NON_PREFS_failure_carries_no_attribution_note"),
    ("the surface probe collapses ABSENT and UNREADABLE, as Java does", H1RUN,
     '        out["plist_sha256"], out["plist_state"] = None, "ABSENT"\n'
     "    except OSError as e:\n"
     '        out["plist_sha256"], out["plist_state"] = None, f"ERROR:{type(e).__name__}"',
     '        out["plist_sha256"], out["plist_state"] = None, "ABSENT"\n'
     "    except OSError:\n"
     '        out["plist_sha256"], out["plist_state"] = None, "ABSENT"',
     f"{T_H1R}::test_the_surface_probe_DISTINGUISHES_absent_from_UNREADABLE"),
    ("the directory probe hides a read failure", H1RUN,
     '        out["entry_count"], out["dir_state"] = -1, f"ERROR:{type(e).__name__}"',
     '        out["entry_count"], out["dir_state"] = -1, "PRESENT"',
     f"{T_H1R}::test_the_surface_probe_reports_an_UNREADABLE_DIRECTORY_as_such"),
    ("no baseline surface is recorded at run start", H1RUN,
     '                      "preference_surface_at_start": preference_surface(),',
     "",
     f"{T_H1R}::test_a_PREFS_failure_records_the_surface_and_REFUSES_to_attribute_it"),

    # -------------------------------------------- the extracted D0 mover cut
    ("mover cut inverted", D0,
     '    return "t1j" if mover == colour_arm.split("_")[1] else "ours"',
     '    return "ours" if mover == colour_arm.split("_")[1] else "t1j"',
     f"{T_D0}::test_by_system_classification_is_unchanged_by_the_extraction"),

    # ------------- 2026-09-05 reporting-contract gap 1: helper-observed prefs
    ("the observation fields are never read from the POSTCOND line", ADAPTER,
     '    return {"prefs_before": kv["prefs_before"], "prefs_after": kv["prefs_after"],',
     '    return {"prefs_before": None, "prefs_after": kv["prefs_after"],',
     f"{T_ADP}::test_parse_postconds_reads_the_helpers_OWN_prefs_observations"),
    ("the observation is parsed but not attached to PostCond", ADAPTER,
     '                refl_n=int(kv["refl_n"]), failures=int(kv["failures"]), **obs)',
     '                refl_n=int(kv["refl_n"]), failures=int(kv["failures"]))',
     f"{T_ADP}::test_parse_postconds_reads_the_helpers_OWN_prefs_observations"),
    ("the POSTCOND self-agreement check disabled", ADAPTER,
     '        if all(v is not None for v in obs.values()):\n            unchanged',
     '        if False:\n            unchanged',
     f"{T_ADP}::test_a_POSTCOND_line_that_DISAGREES_WITH_ITSELF_is_refused"),
    ("the self-agreement check compares the hash only, not the count", ADAPTER,
     '                         and obs["count_before"] == obs["count_after"])',
     '                         and True)',
     f"{T_ADP}::test_a_POSTCOND_line_that_DISAGREES_WITH_ITSELF_is_refused"),
    ("the excerpt reader returns None when the earlier source said nothing", ADAPTER,
     '    if not present:\n        return {k: None for k in _PREFS_OBS_FIELDS}',
     '    if not present:\n        raise ValueError("no observation")',
     f"{T_ADP}::test_the_prefs_observation_is_readable_from_a_BOUNDED_EXCERPT"),
    ("the excerpt reader requires POSTCOND to start a line", ADAPTER,
     '    at = text.rfind("POSTCOND ")',
     '    at = 0 if text.startswith("POSTCOND ") else -1',
     f"{T_ADP}::test_the_prefs_observation_is_readable_from_a_BOUNDED_EXCERPT"),
    ("the VOID diagnostic drops the helper's own observation", H1RUN,
     '        d["helper_prefs_observed"] = observed',
     '        pass',
     f"{T_H1R}::test_the_VOID_diagnostic_carries_the_FAILING_JVMS_OWN_prefs_observation"),
    ("the diagnostic reads only chained stdout, never the AbortError message", H1RUN,
     '    return (_chained_stdout(error) or getattr(error, "message", None)\n'
     '            or str(error) or None)',
     '    return _chained_stdout(error)',
     f"{T_H1R}::test_the_VOID_diagnostic_carries_the_FAILING_JVMS_OWN_prefs_observation"),

    # --------------- 2026-09-05 reporting-contract gap 2: INTERRUPT vs VOID
    ("an operator interrupt is traced as VOID again", H1RUN,
     '                   verdict="INTERRUPTED" if interrupted else "VOID",',
     '                   verdict="VOID",',
     f"{T_H1R}::test_a_KEYBOARD_INTERRUPT_is_NOT_relabelled_as_an_instrument_failure"),
    ("every failure is classified as an interrupt", H1RUN,
     '            interrupted = not isinstance(e, Exception)',
     '            interrupted = True',
     f"{T_H1R}::test_a_VOID_still_writes_VOID_and_a_void_diagnostic_beside_the_interrupt_path"),
    ("the interrupt record is still called a void_diagnostic", H1RUN,
     '                rec.emit_terminal({"record_type": ("interrupt_diagnostic" if interrupted\n'
     '                                                   else "void_diagnostic"),',
     '                rec.emit_terminal({"record_type": "void_diagnostic",',
     f"{T_H1R}::test_a_KEYBOARD_INTERRUPT_is_NOT_relabelled_as_an_instrument_failure"),
    ("the accounting rule no longer travels with the interrupt record", H1RUN,
     '                                   **({"seed_accounting": INTERRUPT_ACCOUNTING_RULE}\n'
     '                                      if interrupted else {})})',
     '                                   })',
     f"{T_H1R}::test_a_KEYBOARD_INTERRUPT_is_NOT_relabelled_as_an_instrument_failure"),
    ("INTERRUPTED removed from the closed verdict enum", H1RUN,
     'TRACE_VERDICTS = ("OK", "VOID", "INTERRUPTED")',
     'TRACE_VERDICTS = ("OK", "VOID")',
     f"{T_H1R}::test_a_KEYBOARD_INTERRUPT_is_NOT_relabelled_as_an_instrument_failure"),
    ("the accounting rule softened: drawn seeds become reusable", H1RUN,
     '    "a VOID. It does NOT make any drawn seed reusable: every seed drawn before the "',
     '    "a VOID. It makes every drawn seed reusable: every seed drawn before the "',
     f"{T_H1R}::test_the_interrupt_accounting_rule_says_drawn_seeds_are_NOT_reusable"),

    # -------- 2026-09-06 review correction: partial observations + truncation
    ("the partial-observation refusal removed", ADAPTER,
     '    if present and len(present) != len(_PREFS_OBS_FIELDS):\n        raise ValueError(',
     '    if False:\n        raise ValueError(',
     f"{T_ADP}::test_a_PARTIAL_observation_is_REFUSED_by_the_parser"),
    ("the partial-observation refusal removed, seen from the REAL query path", ADAPTER,
     '    if present and len(present) != len(_PREFS_OBS_FIELDS):\n        raise ValueError(',
     '    if False:\n        raise ValueError(',
     f"{T_H1R}::test_a_PARTIAL_observation_is_a_VOID_on_the_REAL_query_path"),
    ("a truncated segment ending in the marker reads as the earlier source", ADAPTER,
     '    if segment.endswith("..."):\n        return None',
     '    if False:\n        return None',
     f"{T_ADP}::test_an_INCOMPLETE_segment_reads_as_UNKNOWN_not_as_the_earlier_source"),
    ("a segment missing a BASE field reads as the earlier source", ADAPTER,
     '    if _POSTCOND_BASE_FIELDS - set(kv):\n        return None',
     '    if False:\n        return None',
     f"{T_ADP}::test_an_INCOMPLETE_segment_reads_as_UNKNOWN_not_as_the_earlier_source"),
    ("the query path stops chaining the full transcript", INTEG,
     '            raise AbortError(PHASE_MOVE, message) from A.HelperOutputError(message, out)',
     '            raise AbortError(PHASE_MOVE, message) from None',
     f"{T_H1R}::test_the_observation_SURVIVES_excerpt_truncation_on_the_REAL_query_path"),
    ("the diagnostic parses the BOUNDED excerpt instead of the full text", H1RUN,
     '    observed = A.postcond_prefs_observation(_helper_text(error) or "")',
     '    observed = A.postcond_prefs_observation(d["helper_excerpt"] or "")',
     f"{T_H1R}::test_the_observation_SURVIVES_excerpt_truncation_on_the_REAL_query_path"),
    ("the attribution trigger reads only the bounded excerpt", H1RUN,
     '    if ((observed is not None and observed["prefs_ok"] is False)\n'
     '            or "prefs_ok=false" in (d["helper_excerpt"] or "")):',
     '    if "prefs_ok=false" in (d["helper_excerpt"] or ""):',
     f"{T_H1R}::test_the_observation_SURVIVES_excerpt_truncation_on_the_REAL_query_path"),

    # ------------- 2026-09-06 runtime requalification: gated runner (NOT RUN)
    ("requal gate removed at the public runner", RQ,
     '    if not RUNTIME_REQUAL_AUTHORIZED:\n        raise RequalError(',
     '    if False:\n        raise RequalError(',
     f"{T_RQ}::test_the_public_runner_refuses_while_the_gate_is_shut"),
    ("requal gate removed at the worker entry", RQ,
     '    if not RUNTIME_REQUAL_AUTHORIZED:\n        print("the runtime requalification is UNAUTHORIZED. No JVM was started, no "',
     '    if False:\n        print("the runtime requalification is UNAUTHORIZED. No JVM was started, no "',
     f"{T_RQ}::test_the_WORKER_entry_refuses_in_a_fresh_subprocess_too"),
    ("requal gate removed at main (the worker still refuses, so only the no-spawn test sees it)", RQ,
     '    if not RUNTIME_REQUAL_AUTHORIZED:\n        print("the runtime requalification is UNAUTHORIZED. No worker was spawned, "',
     '    if False:\n        print("the runtime requalification is UNAUTHORIZED. No worker was spawned, "',
     f"{T_RQ}::test_main_refuses_without_spawning_while_the_gate_is_shut"),
    ("the frozen prefix file's hash pin removed", RQ,
     '    if got != FROZEN_PREFIXES_SHA256:\n        raise RequalError(',
     '    if False:\n        raise RequalError(',
     f"{T_RQ}::test_a_tampered_prefix_file_is_refused"),
    ("the prefix-file vs plan cross-check removed", RQ,
     '    if ours != plans:\n        raise RequalError(',
     '    if False:\n        raise RequalError(',
     f"{T_RQ}::test_a_prefix_file_that_disagrees_with_the_PLAN_is_refused"),
    ("a reply without the observation is read as legacy instead of VOID", RQ,
     '    if all(v is None for v in observed.values()):\n        raise RequalVoidError(',
     '    if False:\n        raise RequalVoidError(',
     f"{T_RQ}::test_a_reply_WITHOUT_the_observation_is_a_VOID_not_a_legacy_reading"),
    ("the parser/reader agreement check removed", RQ,
     '    if prefs_observation != {"prefs_ok": post.prefs_ok, **observed}:\n        raise RequalVoidError(',
     '    if False:\n        raise RequalVoidError(',
     f"{T_RQ}::test_the_parser_and_the_diagnostic_reader_must_AGREE"),
    ("a dirty preference surface no longer recorded as a failure", RQ,
     '    if not post.clean:\n        failures.append(f"postcondition surface not clean (preference or other): {post}")',
     '    if False:\n        failures.append(f"postcondition surface not clean (preference or other): {post}")',
     f"{T_RQ}::test_a_prefs_failure_is_a_recorded_FAIL_whose_observation_SURVIVES_truncation"),
    ("the diagnostic observation read from the bounded excerpt, not the chain", RQ,
     '                   helper_prefs_observed=A.postcond_prefs_observation(\n'
     '                       DIAG._helper_text(abort) or ""))',
     '                   helper_prefs_observed=A.postcond_prefs_observation(\n'
     '                       DIAG._bounded_excerpt(abort) or ""))',
     f"{T_RQ}::test_a_prefs_failure_is_a_recorded_FAIL_whose_observation_SURVIVES_truncation"),
    ("the exact-eight requirement removed from the public runner", RQ,
     '    if len(prefixes) != len(frozen):\n        raise RequalError(',
     '    if False:\n        raise RequalError(',
     f"{T_RQ}::test_the_public_runner_requires_EXACTLY_the_frozen_eight"),
    ("the create-only precheck removed (O_EXCL still fails, but after 16 launches)", RQ,
     '    if os.path.lexists(out_path):\n        raise FileExistsError(',
     '    if False:\n        raise FileExistsError(',
     f"{T_RQ}::test_the_record_is_create_only_and_refuses_BEFORE_any_launch"),
    ("the supervisor no longer starts the worker in its own session", RQ,
     '    p = subprocess.Popen(list(cmd), start_new_session=True, pass_fds=tuple(pass_fds))',
     '    p = subprocess.Popen(list(cmd), pass_fds=tuple(pass_fds))',
     f"{T_RQ}::test_the_supervisor_starts_the_worker_in_a_NEW_SESSION"),
    ("the supervisor stops at SIGTERM and never SIGKILLs the group", RQ,
     '        rc = EXIT_TIMEOUT\n        _killpg(pgid, signal.SIGTERM)\n        try:\n            p.wait(timeout=kill_grace_s)\n        except subprocess.TimeoutExpired:\n            _killpg(pgid, signal.SIGKILL)\n            p.wait()',
     '        rc = EXIT_TIMEOUT\n        _killpg(pgid, signal.SIGTERM)\n        try:\n            p.wait(timeout=kill_grace_s)\n        except subprocess.TimeoutExpired:\n            p.wait()',
     f"{T_RQ}::test_the_supervisor_kills_the_WHOLE_process_group_on_timeout"),
    ("a supervisor kill is reported as VOID instead of TIMEOUT", RQ,
     '        rc = EXIT_TIMEOUT',
     '        rc = EXIT_VOID',
     f"{T_RQ}::test_the_supervisor_kills_the_WHOLE_process_group_on_timeout"),
    ("a FAIL result exits 0 like a pass", RQ,
     '    return EXIT_PASS if report["verdict"] == "PASS" else EXIT_FAIL',
     '    return EXIT_PASS',
     f"{T_RQ}::test_the_worker_maps_each_outcome_to_its_own_exit_code[FAIL-2]"),
    ("the outer cap shrinks to the inner deadline (no grace for the inner VOID)", RQ,
     '                  timeout_s=RUN_DEADLINE_S + SUPERVISOR_GRACE_S, kill_grace_s=5.0)',
     '                  timeout_s=RUN_DEADLINE_S, kill_grace_s=5.0)',
     f"{T_RQ}::test_main_supervises_a_WORKER_subprocess_under_the_outer_cap"),
    ("main spawns itself without --worker", RQ,
     '    r = supervise([sys.executable, "-m", MODULE, "--worker", *argv],',
     '    r = supervise([sys.executable, "-m", MODULE, *argv],',
     f"{T_RQ}::test_main_supervises_a_WORKER_subprocess_under_the_outer_cap"),
    ("every call no longer carries the frozen per-call timeout", RQ,
     '    agent = INT.T1jAgent(runtime=runtime, ctx=ctx, depth=DEPTH, colour=state.to_move,\n'
     '                         timeout_s=PER_CALL_TIMEOUT_S, _query=keep)',
     '    agent = INT.T1jAgent(runtime=runtime, ctx=ctx, depth=DEPTH, colour=state.to_move,\n'
     '                         timeout_s=None, _query=keep)',
     f"{T_RQ}::test_one_replay_and_one_query_per_prefix_each_with_the_frozen_timeout"),

    # -------- 2026-09-06 review: cleanup after ANY exit; complete row binding
    ("no cleanup after a worker that exits before the timeout", RQ,
     '    cleared = _group_cleared(pgid, wait_s=0.5)\n    if not cleared:\n        _killpg(pgid, signal.SIGTERM)',
     '    cleared = _group_cleared(pgid, wait_s=0.5)\n    if False:\n        _killpg(pgid, signal.SIGTERM)',
     f"{T_RQ}::test_a_child_that_OUTLIVES_a_finished_worker_is_killed_and_the_group_cleared"),
    ("main reports the worker's code although cleanup failed", RQ,
     '    if not r["group_cleared"]:\n        # 🔴 NEVER SUCCESS BESIDE A SURVIVOR.',
     '    if False:\n        # 🔴 NEVER SUCCESS BESIDE A SURVIVOR.',
     f"{T_RQ}::test_main_NEVER_reports_the_workers_code_when_cleanup_failed[0]"),
    ("EPERM from the group probe read as CLEARED", RQ,
     '        except PermissionError:\n            pass\n        if time.monotonic() >= end:',
     '        except PermissionError:\n            return True\n        if time.monotonic() >= end:',
     f"{T_RQ}::test_a_group_member_we_CANNOT_SIGNAL_counts_as_occupied_not_cleared"),
    ("row binding compares digests only again", RQ,
     '        for key in want:\n            if key not in got or not _same(got[key], want[key]):',
     '        for key in ("digest",):\n            if key not in got or not _same(got[key], want[key]):',
     f"{T_RQ}::test_a_DIGEST_PRESERVING_metadata_change_is_refused_by_the_public_entry[_relabel_opening]"),
    ("row binding drops the TYPE check (False == 0)", RQ,
     '    if type(a) is not type(b):\n        return False',
     '    if False:\n        return False',
     f"{T_RQ}::test_a_DIGEST_PRESERVING_metadata_change_is_refused_by_the_public_entry[_bool_as_int]"),
    ("row binding accepts extra keys", RQ,
     '        extra = sorted(set(got) - set(want))\n        if extra:',
     '        extra = sorted(set(got) - set(want))\n        if False:',
     f"{T_RQ}::test_a_DIGEST_PRESERVING_metadata_change_is_refused_by_the_public_entry[_extra_key]"),
    ("row binding normalises VALUES, not only shape", RQ,
     '    if isinstance(v, (list, tuple)):\n        return [_normalise_shape(x) for x in v]\n    return v',
     '    if isinstance(v, (list, tuple)):\n        return [_normalise_shape(x) for x in v]\n    return int(v) if isinstance(v, (bool, str)) and str(v).lstrip("-").isdigit() else v',
     f"{T_RQ}::test_a_DIGEST_PRESERVING_metadata_change_is_refused_by_the_public_entry[_ply_as_str]"),

    # ---------------- 2026-09-07 H1 retry preparation: fresh block, v4 plan, wrapper
    ("the attempt-2 block moved back onto the SPENT attempt-1 block", H1R,
     'H1_SEED_BLOCK = (202617000, 202617224)',
     'H1_SEED_BLOCK = (202616000, 202616224)',
     f"{T_H1V}::test_seeds_are_the_ATTEMPT2_block_one_per_task_in_order"),
    ("the plan loader stops verifying the v4 file hash", H1P,
     '    if got != sha256:\n        raise H1PlanError(f"H1 plan sha256 {got} != pinned {sha256}")',
     '    if False:\n        raise H1PlanError(f"H1 plan sha256 {got} != pinned {sha256}")',
     f"{T_H1V}::test_a_plan_with_the_SAME_TASKS_but_a_REWRITTEN_THRESHOLD_is_refused"),
    ("the plan loader stops verifying the task digest", H1P,
     '    if digest != task_digest:\n        raise H1PlanError(',
     '    if False:\n        raise H1PlanError(',
     f"{T_H1V}::test_a_plan_whose_TASKS_were_swapped_is_refused_even_if_the_FILE_hashes"),
    # (removed: BYTE-IDENTICAL to "the registration barrier no longer binds".)
    ("the wrapper's main stops reading the runner's gate", CMD,
     '    if not gate_is_open():\n        print("the H1 match is NOT AUTHORIZED (H1_EXECUTION_AUTHORIZED is False). No "\n              "worker was spawned',
     '    if False:\n        print("the H1 match is NOT AUTHORIZED (H1_EXECUTION_AUTHORIZED is False). No "\n              "worker was spawned',
     f"{T_CMD}::test_main_refuses_without_spawning_while_the_gate_is_shut"),
    ("the wrapper's worker stops reading the runner's gate", CMD,
     '    if not gate_is_open():\n        print("the H1 match is NOT AUTHORIZED (H1_EXECUTION_AUTHORIZED is False). No "\n              "JVM was started',
     '    if False:\n        print("the H1 match is NOT AUTHORIZED (H1_EXECUTION_AUTHORIZED is False). No "\n              "JVM was started',
     f"{T_CMD}::test_the_worker_refuses_without_running_while_the_gate_is_shut"),
    ("the wrapper skips the output-path precheck", CMD,
     '        try:\n            RUN.check_output_paths(a.results, a.trace)\n            refused = False',
     '        try:\n            refused = False',
     f"{T_CMD}::test_main_refuses_an_EXISTING_output_path_before_spawning"),
    ("a failed gate restoration no longer supersedes", CMD,
     '            code = EXIT_GATE_NOT_RESTORED\n    return code',
     '            pass\n    return code',
     f"{T_CMD}::test_main_restores_the_gate_after_EVERY_outcome_and_a_failed_restore_supersedes[0]"),
    ("the gate is restored only on the happy path (else, not finally)", CMD,
     '        code = EXIT_UNEXPECTED\n    finally:\n        # 🔴 THE GATE IS RESTORED WHATEVER HAPPENED',
     '        code = EXIT_UNEXPECTED\n    else:\n        # 🔴 THE GATE IS RESTORED WHATEVER HAPPENED',
     f"{T_CMD}::test_main_restores_the_gate_even_when_the_supervisor_RAISES"),
    ("restore_gate reports success without verifying the rewrite", CMD,
     '    return back.count(_GATE_CLOSED + "\\n") == 1 and not _GATE_OPEN.search(back)',
     '    return True',
     f"{T_CMD}::test_restore_gate_reports_FAILURE_when_the_rewrite_did_not_take"),
    ("the wrapper reports the worker's code although cleanup failed", CMD,
     '                code = EXIT_CLEANUP_FAILED\n            elif r["timed_out"]:',
     '                code = r["exit_code"]\n            elif r["timed_out"]:',
     f"{T_CMD}::test_main_NEVER_reports_success_when_cleanup_failed"),
    ("an operator interrupt in the worker exits 0", CMD,
     '        return EXIT_INTERRUPTED\n    except Exception as e:                                    # noqa: BLE001\n        print(f"UNEXPECTED {type(e).__name__}: {e}", file=sys.stderr)\n        return EXIT_UNEXPECTED\n    print(f"COMPLETED',
     '        return EXIT_COMPLETED\n    except Exception as e:                                    # noqa: BLE001\n        print(f"UNEXPECTED {type(e).__name__}: {e}", file=sys.stderr)\n        return EXIT_UNEXPECTED\n    print(f"COMPLETED',
     f"{T_CMD}::test_the_worker_maps_each_outcome_to_its_own_exit_code[interrupt-9]"),
    ("the supervisor no longer forwards an operator interrupt", RQ,
     '    except KeyboardInterrupt:\n        interrupted = True\n        _killpg(pgid, signal.SIGINT)               # forward the operator\'s stop',
     '    except KeyboardInterrupt:\n        interrupted = True\n        pass',
     f"{T_CMD}::test_the_supervisor_FORWARDS_an_operator_interrupt_to_the_worker_group"),
    ("the wrapper's outer cap shrinks to the runner deadline", CMD,
     '                      timeout_s=RUN.RUN_DEADLINE_S + SUPERVISOR_GRACE_S,',
     '                      timeout_s=RUN.RUN_DEADLINE_S,',
     f"{T_CMD}::test_main_supervises_a_WORKER_under_the_H1_deadline_plus_grace_and_passes_0_through"),

    # -------- 2026-09-07 review: restoration target bound; refusal inside the boundary
    # 🔴 A first version returned from INSIDE the try, where the finally still
    # runs -- NOT CAUGHT, because it was not the defect it named. The defect is
    # the precheck sitting IN FRONT of the boundary, which this reproduces exactly.
    ("the output precheck moves back IN FRONT of the restoration boundary", CMD,
     '    code = EXIT_UNEXPECTED\n    try:\n        try:\n            RUN.check_output_paths(a.results, a.trace)\n            refused = False\n        except RUN.H1Error as e:\n            print(f"refused before spawning: {e}", file=sys.stderr)\n            code, refused = EXIT_REFUSED, True   # no return: the finally must run',
     '    try:\n        RUN.check_output_paths(a.results, a.trace)\n    except RUN.H1Error as e:\n        print(f"refused before spawning: {e}", file=sys.stderr)\n        return EXIT_REFUSED\n    code = EXIT_UNEXPECTED\n    try:\n        refused = False',
     f"{T_CMD}::test_main_refuses_an_EXISTING_output_path_before_spawning"),
    ("a failed restoration no longer supersedes a refusal", CMD,
     '            code = EXIT_GATE_NOT_RESTORED\n    return code',
     '            code = EXIT_GATE_NOT_RESTORED if code != EXIT_REFUSED else code\n    return code',
     f"{T_CMD}::test_output_refusal_with_a_FAILED_restoration_is_exit_10_not_7"),
    ("the --runner-source override returns to the production CLI", CMD,
     '    ap.add_argument("--worker", action="store_true",\n                    help="internal: run the match in this process (spawned by main)")\n    return ap',
     '    ap.add_argument("--runner-source", default=RUNNER_SOURCE)\n    ap.add_argument("--worker", action="store_true",\n                    help="internal: run the match in this process (spawned by main)")\n    return ap',
     f"{T_CMD}::test_the_CLI_has_NO_runner_source_override_and_a_decoy_is_rejected"),
    # 🔴 A first version substituted an EQUIVALENT expression -- a vacuous
    # control. This one retargets the default to a path that is not the runner.
    ("the default restoration target is not the imported runner's source", CMD,
     '    target = RUNNER_SOURCE if _runner_source is None else _runner_source',
     '    target = RUNNER_SOURCE + ".decoy" if _runner_source is None else _runner_source',
     f"{T_CMD}::test_restoration_targets_the_IMPORTED_runners_source_by_default"),

    # -------------------- 2026-09-07 attempt-2 seed registration (ACCOUNTED only)
    ("the attempt-2 block un-registered from ACCOUNTED", REF_SRC,
     "    (202617000, 202617224),          # H1 ATTEMPT 2, registered 2026-09-07 as",
     "    # (202617000, 202617224),        # H1 ATTEMPT 2, registered 2026-09-07 as",
     f"{T_H1V}::test_the_ATTEMPT2_block_is_ACCOUNTED_EXPOSED_224_and_RETIRED_WHOLE"),
    ("the attempt-2 block un-RETIRED after the completed match", REF_SRC,
     "    (202617000, 202617224),          # H1 ATTEMPT 2, retired WHOLE 2026-09-07:",
     "    # (202617000, 202617224),        # H1 ATTEMPT 2, retired WHOLE 2026-09-07:",
     f"{T_H1V}::test_the_ATTEMPT2_block_is_ACCOUNTED_EXPOSED_224_and_RETIRED_WHOLE"),
    ("the attempt-2 block un-EXPOSED after the completed match", REF_SRC,
     "    (202617000, 202617224),          # H1 ATTEMPT 2, run once 2026-09-07 and",
     "    # (202617000, 202617224),        # H1 ATTEMPT 2, run once 2026-09-07 and",
     f"{T_H1V}::test_the_ATTEMPT2_block_is_ACCOUNTED_EXPOSED_224_and_RETIRED_WHOLE"),
    # 🔴 The first version un-EXPOSED the block only; it stayed RETIRED, so the
    # schedule was still refused and the control was NOT CAUGHT. The defect a
    # single anchor CAN reproduce is the runner no longer asking the executable
    # question at all.
    ("the runner stops asking whether the schedule may be RUN (spent block accepted)", H1RUN,
     '            REF.validate_schedule_executable(tasks)',
     '            pass  # REF.validate_schedule_executable(tasks)',
     f"{T_H1R}::test_the_SPENT_attempt2_schedule_is_refused_in_the_REAL_state"),

    # ------------------- 2026-09-07 D1' analysis: the frozen plan's Sec. 9 controls
    ("rank ties broken in REVERSE (row, col)", DP,
     "    order = sorted(masses, key=lambda m: (-masses[m], m[0], m[1]))",
     "    order = sorted(masses, key=lambda m: (-masses[m], -m[0], -m[1]))",
     f"{T_DP}::test_rank_raw_orders_by_mass_then_breaks_ties_by_row_col_only"),
    ("agree coupled to lprd (the withdrawn implication)", DP,
     '        "agree": ours == t1j,',
     '        "agree": ours == t1j and ranks[t1j] <= K_LPRD,',
     f"{T_DP}::test_agree_AND_lprd_can_both_be_true_neither_implies_the_other"),
    ("k drifts from 5 to 6", DP,
     '        "lprd": ranks[t1j] > K_LPRD,',
     '        "lprd": ranks[t1j] > K_LPRD + 1,',
     f"{T_DP}::test_lprd_is_true_when_t1j_move_ranks_below_fifth_and_false_otherwise"),
    ("the depth-3 move read instead of mdPly 6", DP,
     '    depth6 = [d for d in pos["depths"] if int(d["depth"]) == 6]',
     '    depth6 = [d for d in pos["depths"] if int(d["depth"]) == 3]',
     f"{T_DP}::test_position_row_reads_the_mdPly_6_move_not_the_depth_3_move"),
    ("MH weights replaced by equal weights", DP,
     "            w = len(pos) * len(ctl) / (len(pos) + len(ctl))",
     "            w = 1.0",
     f"{T_DP}::test_MH_weighted_statistic_known_answer_two_cells"),
    ("common support pooled: cells with ONE role enter the statistic", DP,
     "        if pos and ctl:",
     "        if pos or ctl:",
     f"{T_DP}::test_common_support_BINDS_an_excess_that_lives_only_in_control_less_cells_is_nothing"),
    ("the direction flipped (controls minus positions)", DP,
     "            diff = sum(pos) / len(pos) - sum(ctl) / len(ctl)",
     "            diff = sum(ctl) / len(ctl) - sum(pos) / len(pos)",
     f"{T_DP}::test_the_same_excess_at_CONTROLS_is_NO_GO_direction_binds"),
    ("the decision threshold lowered", DP,
     '    return "GO" if (T >= T_THRESHOLD and lo > 0) else "NO_GO"',
     '    return "GO" if (T >= 0.05 and lo > 0) else "NO_GO"',
     
     f"{T_DP}::test_an_excess_AT_OR_ABOVE_0_15_is_GO_and_one_below_is_NO_GO_despite_a_positive_lower_bound"),
    ("the lower-bound condition dropped (T alone decides)", DP,
     '    return "GO" if (T >= T_THRESHOLD and lo > 0) else "NO_GO"',
     '    return "GO" if T >= T_THRESHOLD else "NO_GO"',
     f"{T_DP}::test_an_effect_carried_by_ONE_game_has_a_lower_bound_at_zero_and_is_NO_GO"),
    ("undefined replicates silently DISCARDED", DP,
     '        values.append(st["T"])',
     '        if st["T"] is not None:\n            values.append(st["T"])',
     f"{T_DP}::test_a_replicate_that_loses_a_role_in_a_cell_drops_that_cell_and_is_counted"),
    ("a game drawn twice contributes its rows ONCE (re-deduplicated)", DP,
     "    for g in games:\n        rep_rows += by_game.get(g, [])",
     "    for g in dict.fromkeys(games):\n        rep_rows += by_game.get(g, [])",
     f"{T_DP}::test_resampling_draws_whole_games_within_strata_with_the_pinned_PRNG_and_order"),
    ("the second draw copies the first (no replacement)", DP,
     '        games += [s["games"][i], s["games"][j]]',
     '        games += [s["games"][i], s["games"][i]]',
     f"{T_DP}::test_resampling_draws_whole_games_within_strata_with_the_pinned_PRNG_and_order"),
    ("the PRNG seed drifts", DP,
     "    rng = np.random.default_rng(seed)",
     "    rng = np.random.default_rng(seed + 1)",
     f"{T_DP}::test_resampling_draws_whole_games_within_strata_with_the_pinned_PRNG_and_order"),
    ("the quantile convention changes from type 7", DP,
     '    lo, hi = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975], method="linear")',
     '    lo, hi = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975], method="nearest")',
     f"{T_DP}::test_the_stability_interval_is_the_type_7_central_95_percent_range"),
    ("the eligibility floor becomes strict (refuses AT the floor)", DP,
     "    return all(detail[k] >= FLOOR[k] for k in FLOOR), detail",
     "    return all(detail[k] > FLOOR[k] for k in FLOOR), detail",
     f"{T_DP}::test_the_eligibility_floor_accepts_AT_the_floor_and_refuses_one_below"),
    ("contributing games counted from UNMATCHED rows too", DP,
     "            games_cs |= games_by_cell[cell]\n        else:",
     "            games_cs |= games_by_cell[cell]\n        else:\n            games_cs |= games_by_cell[cell]",
     f"{T_DP}::test_contributing_games_count_rows_of_EITHER_role_in_common_support_only"),
    ("an undefined arm reported as plain NO_GO (treated as zero)", DP,
     '        out["outcome"] = "NO_GO — arm undefined"',
     '        out["outcome"] = "NO_GO"',
     f"{T_DP}::test_an_UNDEFINED_arm_is_NO_GO_arm_undefined_never_zero_never_skipped"),
    ("the both-arm rule dropped", DP,
     '    if not all(arms[a] > 0 for a in ARMS):\n        out["outcome"] = "NO_GO"',
     '    pass',
     f"{T_DP}::test_a_ONE_ARM_effect_cannot_confirm"),
    ("cross-half duplicates no longer removed", DP,
     '        if r["digest"] in seen:\n            removed["seen_in_development"] += 1',
     '        if False:\n            removed["seen_in_development"] += 1',
     f"{T_DP}::test_cross_half_duplicates_are_REMOVED_BEFORE_the_cap_so_they_consume_no_slot"),
    ("the ply < 5 eligibility filter dropped", DP,
     "        if int(r[\"ply\"]) < CONFIRMATION_MIN_PLY:",
     "        if False:",
     f"{T_DP}::test_rows_below_ply_5_are_ineligible_for_BOTH_roles_before_anything_else"),
    ("the ceiling drops silently instead of refusing", DP,
     "    if len(selected) > ceiling:\n        raise D1PrimeRefused(",
     "    if False:\n        raise D1PrimeRefused(",
     f"{T_DP}::test_the_ceiling_refuses_241_rows_and_accepts_240_before_any_seed"),
    ("within-half dedup keeps the LATEST row", DP,
     "    for r in sorted(candidates, key=_order):",
     "    for r in sorted(candidates, key=_order, reverse=True):",
     f"{T_DP}::test_within_half_dedup_keeps_the_earliest_by_task_id_then_ply"),
    ("cohort binding loses type strictness", DP,
     '            if k not in got or not _same(got[k], want[k]):',
     '            if k not in got or got[k] != want[k]:',
     f"{T_DP}::test_the_analysis_refuses_a_cohort_that_is_not_the_frozen_one_field_by_field"),
    ("cohort binding stops checking the count", DP,
     "    if len(rows) != len(frozen):",
     "    if False:",
     f"{T_DP}::test_the_analysis_refuses_a_cohort_that_is_not_the_frozen_one_field_by_field"),

    # -------- 2026-09-08 D1' integration repair: binding, schema, field name
    ("the checked entry stops binding the cohort", DP,
     "    check_cohort(rows, frozen_cohort)\n    strata = design_strata(tasks, reps=reps)",
     "    strata = design_strata(tasks, reps=reps)",
     f"{T_DP}::test_the_binding_layer_binds_the_cohort_BEFORE_any_calculation"),
    ("the checked entry takes a cohort argument again (secondary promotable)", DP,
     '    out = dict(kernel(rows, strata, cohort=PRIMARY_COHORT, B=B, seed=seed))',
     '    out = dict(kernel(rows, strata, cohort=SECONDARY_COHORT, B=B, seed=seed))',
     f"{T_DP}::test_the_binding_layer_FIXES_the_primary_hypothesis"),
    ("the design/cohort game check removed", DP,
     "    missing = sorted({r[\"task_id\"] for r in frozen_cohort} - in_design)\n    if missing:",
     "    missing = []\n    if missing:",
     f"{T_DP}::test_a_cohort_whose_GAMES_are_not_in_the_designs_strata_is_REFUSED"),
    ("strata inferred from the ROWS instead of the design", DP,
     "    strata = design_strata(tasks, reps=reps)\n    # 🔴 THE DESIGN MUST CONTAIN THE COHORT'S GAMES.",
     "    strata = design_strata([t for t in tasks if any(r[\"task_id\"] == t[\"task_id\"] for r in frozen_cohort)], reps=reps)\n    # 🔴 THE DESIGN MUST CONTAIN THE COHORT'S GAMES.",
     f"{T_DP}::test_the_binding_layer_builds_strata_from_the_DESIGN_not_from_the_rows"),
    ("eligibility defaults a missing 'system' to acceptance", DP,
     '        if system not in ("ours", "t1j"):',
     '        if False:',
     f"{T_DP}::test_the_selector_uses_the_PRODUCTION_system_field_and_refuses_a_row_without_it"),
    ("eligibility reads an invented flag instead of 'system'", DP,
     '        system = r.get("system")',
     '        system = "ours" if r.get("incumbent_to_move", True) else "t1j"',
     f"{T_DP}::test_the_selector_uses_the_PRODUCTION_system_field_and_refuses_a_row_without_it"),
    ("the opponent-to-move filter inverted", DP,
     '        if system != "ours":',
     '        if system == "ours":',
     f"{T_DP}::test_the_selector_matches_D0s_ONE_DEFINITION_of_which_engine_moved"),
    ("the override flag read from the invented field name", DP,
     '        "overrode_leader": bool(inc["readout_overrode_leader"]),',
     '        "overrode_leader": inc.get("overrode_leader"),',
     f"{T_DP}::test_the_override_flag_is_read_from_the_REAL_writers_field_name"),
    ("a record without the override field is silently None", DP,
     '    if "readout_overrode_leader" not in inc:\n        raise D1PrimeError(',
     '    if False:\n        raise D1PrimeError(',
     f"{T_DP}::test_a_record_without_the_override_field_is_REFUSED_not_silently_None"),

    # --- 2026-09-08 analysis-boundary repair: canonical resolution, design metadata, arm order
    ("the production entry takes the cohort from the caller again", DP,
     "    canon = resolve_canonical_cohort()\n    acquisition = check_report_contract",
     "    canon = dict(resolve_canonical_cohort(), rows=rows_from_d1_report(d1_report))\n    acquisition = check_report_contract",
     f"{T_DP}::test_the_production_entry_REFUSES_a_report_that_is_not_the_canonical_cohort"),
    ("the production entry uses a smaller B than the frozen one", DP,
     "                   B=B_REPLICATES, seed=BOOTSTRAP_SEED)\n    out[\"resolved\"] = {k: canon[k] for k in",
     "                   B=100, seed=BOOTSTRAP_SEED)\n    out[\"resolved\"] = {k: canon[k] for k in",
     f"{T_DP}::test_the_production_entry_uses_the_FROZEN_B_and_seed"),
    # 🔴 MUTATION AND TARGET BOTH REWRITTEN 2026-09-12. The old replacement changed
    # the DIRECTORY and kept the L0 filename, so the path did not exist at all;
    # and the old target asserted `out["resolved"]["record"] == DP.L0_RECORD_REL`,
    # which READS THE CONSTANT IT IS CHECKING and would have moved with the
    # defect. It never got that far regardless: `D0.bind_record` pins the
    # record's header plan digest, so the fixture raised during SETUP. This names
    # a record that EXISTS, and a target that pins both paths as LITERALS.
    ("the canonical cohort is resolved from a different record", DP,
     'L0_RECORD_REL = ("docs/superpowers/evidence/2026-08-27-t1j-l0-canonical-match/"\n'
     '                 "06_l0_match_results.jsonl")',
     'L0_RECORD_REL = ("docs/superpowers/evidence/2026-09-07-t1j-h1-match-attempt2/"\n'
     '                 "01_h1_results.jsonl")',
     f"{T_DP}::test_the_canonical_RECORD_AND_PLAN_are_the_L0_ONES_written_as_LITERALS"),
    ("the design metadata binding removed", DP,
     "            if not _same(r[field], t.get(field)):",
     "            if False:",
     f"{T_DP}::test_a_design_whose_task_METADATA_disagrees_with_the_rows_is_REFUSED"),
    ("the design binding loses type strictness on rep", DP,
     "            if not _same(r[field], t.get(field)):",
     "            if r[field] != t.get(field):",
     f"{T_DP}::test_the_design_binding_is_TYPE_STRICT_on_rep[0.0]"),
    ("the design binding skips a field the row omits", DP,
     "            if field not in r:\n                raise D1PrimeError(",
     "            if False:\n                raise D1PrimeError(",
     f"{T_DP}::test_a_row_that_does_not_CARRY_a_bound_design_field_is_REFUSED"),
    ("the undefined-arm check runs AFTER the generic NO_GO again", DP,
     '    if any(arms[a] is None for a in ARMS):\n        out["outcome"] = "NO_GO — arm undefined"\n        return out\n    if dev["outcome"].startswith("NO_GO"):\n        return out',
     '    if dev["outcome"].startswith("NO_GO"):\n        return out\n    if any(arms[a] is None for a in ARMS):\n        out["outcome"] = "NO_GO — arm undefined"\n        return out',
     f"{T_DP}::test_an_undefined_arm_is_named_EVEN_WHEN_support_is_insufficient"),

    # ---- 2026-09-08 pre-execution binding: seeds to positions, report to a completed run
    ("D1 stops binding the seed assignment to positions", PROBE,
     "    _check_seed_assignment(positions)",
     "    pass",
     f"{T_PROBE}::test_the_SAME_SEED_on_every_row_is_refused"),
    ("D1's seed rule stops being POSITIONAL (interval membership only)", PROBE,
     "        if type(seed) is not int or seed != want:",
     "        if type(seed) is not int or not lo <= seed < hi:",
     f"{T_PROBE}::test_row_i_must_carry_exactly_the_ith_seed_of_the_interval"),
    ("D1's seed rule loses type strictness", PROBE,
     "        if type(seed) is not int or seed != want:",
     "        if seed != want:",
     f"{T_PROBE}::test_the_seed_assignment_is_TYPE_STRICT_and_inside_the_interval"),
    ("D1' analyses a record with no acquisition metadata", DP,
     "    acquisition = check_report_contract(d1_report, canon[\"rows\"])",
     "    acquisition = {}",
     f"{T_DP}::test_the_report_contract_runs_BEFORE_the_statistic"),
    ("D1' stops requiring the acquisition fields to be present", DP,
     "    if missing:\n        raise D1PrimeError(\n            f\"the input is not a completed D1 record",
     "    if False:\n        raise D1PrimeError(\n            f\"the input is not a completed D1 record",
     f"{T_DP}::test_every_acquisition_FIELD_is_required[queries_spent]"),
    ("D1' stops requiring elapsed_s and positions to be present", DP,
     '    missing = [k for k in (*contract, *ACQUISITION_IDENTITY_FIELDS,\n                           "elapsed_s", "positions") if k not in d1_report]',
     '    missing = [k for k in (*contract, *ACQUISITION_IDENTITY_FIELDS) if k not in d1_report]',
     f"{T_DP}::test_a_report_that_OMITS_elapsed_s_is_REFUSED"),
    ("D1' accepts an acquisition value that disagrees with the frozen run", DP,
     "        if not _same(d1_report[key], want):",
     "        if False:",
     f"{T_DP}::test_an_acquisition_field_that_DISAGREES_with_the_frozen_run_is_REFUSED[queries_spent-1100]"),
    ("D1' stops binding the per-position seed assignment", DP,
     "        if type(seed) is not int or seed != want_seed:",
     "        if False:",
     f"{T_DP}::test_positions_must_carry_the_FROZEN_SEED_ASSIGNMENT_row_by_row"),
    ("D1' stops binding the per-position prefix", DP,
     "        if not _same(pos[\"prefix\"], want_row[\"prefix\"]):",
     "        if False:",
     f"{T_DP}::test_positions_must_carry_a_PREFIX_matching_the_canonical_row"),
    ("D1' runs the contract AFTER the statistic", DP,
     "    acquisition = check_report_contract(d1_report, canon[\"rows\"])\n    rows = rows_from_d1_report(d1_report)",
     "    rows = rows_from_d1_report(d1_report)\n    acquisition = check_report_contract(d1_report, canon[\"rows\"])",
     f"{T_DP}::test_the_report_contract_runs_BEFORE_the_statistic"),

    # ---- 2026-09-08 identity contract: the record must be OUR model and OUR toolchain
    ("D1' accepts any non-empty incumbent identity again", DP,
     "    if not _same(got_inc, want_inc):",
     "    if not got_inc:",
     f"{T_DP}::test_the_incumbent_identity_must_EQUAL_the_frozen_one"),
    ("D1' accepts any non-empty toolchain identity again", DP,
     "    for field, want in _expected_toolchain_identity().items():",
     "    for field, want in []:",
     f"{T_DP}::test_a_toolchain_identity_that_FAILS_a_qualification_pin_is_REFUSED[jar_sha256-0000000000000000000000000000000000000000000000000000000000000000]"),
    ("D1' stops checking the compiled-class identity", DP,
     '            "classes": dict(QUALIFIED_CLASSES)}',
     '            }',
     f"{T_DP}::test_a_toolchain_identity_MISSING_a_pinned_component_is_REFUSED[classes]"),
    ("D1' stops checking the helper SOURCE identity", DP,
     '            "sources": {p_.name: INT._sha256(str(p_)) for p_ in A.PREFLIGHT_SOURCES},',
     '',
     f"{T_DP}::test_a_toolchain_identity_MISSING_a_pinned_component_is_REFUSED[sources]"),
    ("D1' stops checking the MAIN CLASS", DP,
     '            "main_class": A.PREFLIGHT_MAIN,',
     '',
     f"{T_DP}::test_a_toolchain_identity_MISSING_a_pinned_component_is_REFUSED[main_class]"),
    ("the compiled-class pin drifts from the qualified builds", DP,
     '        "1f2425fd7b528ad4c1efbdc47331bd0d10261e4a0c7e8a149ecdabfbdcfc88a8",',
     '        "0000000000000000000000000000000000000000000000000000000000000000",',
     f"{T_DP}::test_the_QUALIFIED_CLASS_PIN_matches_both_qualified_builds"),
    ("a malformed toolchain identity raises a TypeError instead of a refusal", DP,
     "    if not isinstance(got_tc, dict):",
     "    if False:",
     f"{T_DP}::test_a_MALFORMED_toolchain_identity_is_a_D1PrimeError_not_a_TypeError[42]"),
    ("elapsed_s is no longer bounded by the deadline", DP,
     "            or not 0 <= value <= deadline:",
     "            or not 0 <= value <= 1e12:",
     f"{T_DP}::test_elapsed_s_must_be_a_real_finite_number_within_the_deadline"),
    ("malformed positions raise a TypeError instead of a refusal", DP,
     "    if not isinstance(positions, list) or not all(isinstance(p, dict) for p in positions):",
     "    if False:",
     f"{T_DP}::test_positions_is_REQUIRED_and_malformed_input_is_a_D1PrimeError"),
    ("the analysis reaches an EXECUTING adapter function", DP,
     '            "main_class": A.PREFLIGHT_MAIN,',
     '            "main_class": A.query.__name__,',
     f"{T_DP}::test_the_analysis_reads_only_PINS_from_the_adapter_and_calls_nothing_that_executes"),
    # ══════════════ 2026-09-08: D1″ SEARCH SUPPRESSION (code-and-test only) ═════
    # Nothing below executes anything: these controls injure arithmetic over a
    # synthetic report and require the named test to catch it.
    ("rank_visit breaks ties by DESCENDING move order", DS,
     "    order = sorted(counts, key=lambda m: (-counts[m], m[0], m[1]))",
     "    order = sorted(counts, key=lambda m: (-counts[m], -m[0], -m[1]))",
     f"{T_DS}::test_rank_visit_orders_by_visits_then_breaks_ties_by_row_col_only"),
    ("rank_visit orders by ASCENDING visits", DS,
     "    order = sorted(counts, key=lambda m: (-counts[m], m[0], m[1]))",
     "    order = sorted(counts, key=lambda m: (counts[m], m[0], m[1]))",
     f"{T_DS}::test_rank_visit_orders_by_visits_then_breaks_ties_by_row_col_only"),
    ("rank_visit DROPS zero-visit moves", DS,
     "    order = sorted(counts, key=lambda m: (-counts[m], m[0], m[1]))",
     "    counts = {m: v for m, v in counts.items() if v}\n"
     "    order = sorted(counts, key=lambda m: (-counts[m], m[0], m[1]))",
     f"{T_DS}::test_rank_visit_ranks_zero_visit_moves_and_does_not_drop_them"),
    ("rank_visit accepts an empty root instead of refusing", DS,
     "    if not counts:",
     "    if False:",
     f"{T_DS}::test_rank_visit_refuses_an_empty_root"),
    ("ss is EXCLUSIVE at five on the policy side", DS,
     '        "ss": rank_raw_t1j <= K_SS and rank_visit_t1j > K_SS,',
     '        "ss": rank_raw_t1j < K_SS and rank_visit_t1j > K_SS,',
     f"{T_DS}::test_ss_is_inclusive_at_five_on_the_policy_side_and_exclusive_on_the_visit_side"),
    ("ss is INCLUSIVE at five on the visit side", DS,
     '        "ss": rank_raw_t1j <= K_SS and rank_visit_t1j > K_SS,',
     '        "ss": rank_raw_t1j <= K_SS and rank_visit_t1j >= K_SS,',
     f"{T_DS}::test_ss_is_inclusive_at_five_on_the_policy_side_and_exclusive_on_the_visit_side"),
    ("ss ignores the VISITS entirely -- the whole question", DS,
     '        "ss": rank_raw_t1j <= K_SS and rank_visit_t1j > K_SS,',
     '        "ss": rank_raw_t1j <= K_SS,',
     f"{T_DS}::test_ss_is_true_only_when_the_policy_ranks_it_top_five_and_the_visits_do_not"),
    ("ss drops the policy condition, so it OVERLAPS lprd", DS,
     '        "ss": rank_raw_t1j <= K_SS and rank_visit_t1j > K_SS,',
     '        "ss": rank_visit_t1j > K_SS,',
     f"{T_DS}::test_ss_and_lprd_are_DISJOINT_by_construction_over_every_rank_combination"),
    ("the strict variant fires on ANY visit count", DS,
     '        "ss0": rank_raw_t1j <= K_SS and visits_t1j == 0.0,',
     '        "ss0": rank_raw_t1j <= K_SS and visits_t1j >= 0.0,',
     f"{T_DS}::test_the_strict_variant_needs_no_tiebreak_and_fires_only_on_zero_visits"),
    ("the policy and the visits may cover DIFFERENT legal sets", DS,
     "    if set(ranks_visit) != set(ranks_raw):",
     "    if False:",
     f"{T_DS}::test_a_position_whose_visits_do_not_cover_the_policy_is_REFUSED"),
    # ⚠ THE CONTROL FOR THE EMPTY-ROOT GUARD IS GONE WITH THE GUARD. It was NOT
    # CAUGHT because `rank_visit` refused the empty root one line later anyway;
    # the duplicate branch was deleted rather than the miss waived, and the
    # refusal below is the one that owns it.
    ("rank_visit accepts an empty root, so an empty record is scored", DS,
     "    if not counts:",
     "    if False:",
     f"{T_DS}::test_a_position_with_no_root_visits_at_all_is_REFUSED_not_scored_as_False"),
    ("the disjointness precondition checks nothing", DS,
     '    bad = [r.get("task_id") for r in rows if r.get("ss") and r.get("lprd")]',
     "    bad = []",
     f"{T_DS}::test_the_disjointness_precondition_REFUSES_a_row_scoring_BOTH"),
    ("the public entry never RUNS the precondition", DS,
     "    check_disjoint(rows)                       # precondition, on the REAL rows",
     "    pass",
     f"{T_DS}::test_the_public_entry_RUNS_the_disjointness_precondition"),
    ("the entry skips the completed-D1 contract", DS,
     "    acquisition = DP.check_report_contract(d1_report, canon[\"rows\"])",
     "    acquisition = {}",
     f"{T_DS}::test_the_entry_refuses_a_report_that_is_not_a_completed_D1_run"),
    ("the entry exposes B and the seed as knobs again", DS,
     "def analyse_search_suppression(d1_report: Mapping[str, Any]) -> Dict[str, Any]:",
     "def analyse_search_suppression(d1_report: Mapping[str, Any], *, B: int = B_REPLICATES,\n"
     "                               seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:",
     f"{T_DS}::test_the_public_entry_takes_the_REPORT_AND_NOTHING_ELSE"),
    ("D1″ reuses D1′'s bootstrap seed, so the draw repeats it", DS,
     "BOOTSTRAP_SEED = 20260908",
     "BOOTSTRAP_SEED = DP.BOOTSTRAP_SEED",
     f"{T_DS}::test_the_frozen_constants_match_the_plan_and_are_BOUND_not_retyped"),
    ("the threshold is RETYPED and drifts below D1′'s", DS,
     "T_THRESHOLD = DP.T_THRESHOLD                  # 0.15, DELIBERATELY unchanged",
     "T_THRESHOLD = 0.05",
     f"{T_DS}::test_the_frozen_constants_match_the_plan_and_are_BOUND_not_retyped"),
    ("the eligibility floor is never enforced", DS,
     "    if not met:",
     "    if False:",
     f"{T_DS}::test_insufficient_support_is_its_own_NO_GO_and_computes_no_interval"),
    ("the SECONDARY strict variant decides the outcome", DS,
     '    return {"outcome": DP._decide(stab), "T": stab["T"], "interval": stab["interval"],',
     '    return {"outcome": "GO" if (secondary["ss0"]["T"] or 0) > 0.5 else DP._decide(stab),\n'
     '            "T": stab["T"], "interval": stab["interval"],',
     f"{T_DS}::test_the_secondary_reports_are_present_and_CANNOT_produce_GO"),
    ("the readout summary tolerates an absent measurement", DS,
     "            if any(field not in r for r in sel):",
     "            if False:",
     f"{T_DS}::test_the_readout_summary_REFUSES_a_row_that_does_not_carry_the_field"),
    # the shared statistic, now indicator-parameterised
    ("the matched statistic IGNORES the requested indicator", DP,
     '        per_cell[_cell(r)][r["role"]].append(bool(r[indicator]))',
     '        per_cell[_cell(r)][r["role"]].append(bool(r["lprd"]))',
     f"{T_DS}::test_the_matched_statistic_reads_the_REQUESTED_indicator"),
    ("a missing indicator field is read as False", DP,
     "        if indicator not in r:",
     "        if False:",
     f"{T_DS}::test_a_row_missing_the_indicator_field_is_REFUSED_not_read_as_False"),
    # ⚠ the one-line form of this anchor matched BOTH matched_statistic and
    # stability_interval, so it named no single defect; the signature's first
    # line disambiguates it.
    ("the indicator changes D1′'s own default result", DP,
     'def matched_statistic(rows: Iterable[Mapping[str, Any]], *, cohort: str,\n'
     '                      arm: Optional[str] = None, indicator: str = "lprd") -> Dict[str, Any]:',
     'def matched_statistic(rows: Iterable[Mapping[str, Any]], *, cohort: str,\n'
     '                      arm: Optional[str] = None, indicator: str = "ss") -> Dict[str, Any]:',
     f"{T_DS}::test_the_indicator_DEFAULTS_to_lprd_so_D1primes_result_is_untouched"),
    # ───────── 2026-09-09 pre-execution repair: validation, complement, summary
    ("the validation pass is never RUN by the entry", DS,
     "    validated = validate_record(d1_report)          # EVERY position, BEFORE any row",
     "    validated = {}",
     f"{T_DS}::test_validation_runs_over_the_WHOLE_record_BEFORE_a_single_row_is_scored"),
    ("validation runs PER ROW, interleaved with the scoring", DS,
     "    for i, pos in enumerate(positions):\n"
     "        if not isinstance(pos, dict):",
     "    for i, pos in enumerate(positions[:1]):\n"
     "        if not isinstance(pos, dict):",
     f"{T_DS}::test_validation_runs_over_the_WHOLE_record_BEFORE_a_single_row_is_scored"),
    ("a required observable is optional again", DS,
     "    missing = [f for f in REQUIRED_OBSERVABLES if f not in inc]",
     "    missing = []",
     f"{T_DS}::test_every_required_observable_is_REQUIRED_by_name[selected_policy_rank]"),
    ("a visit count is coerced with int() instead of type-checked", DS,
     "        if type(v) is not int or v < 0:",
     "        if False:",
     f"{T_DS}::test_a_BOOLEAN_visit_count_is_refused_because_True_is_not_one_visit"),
    ("a negative or fractional visit count is accepted", DS,
     "        if type(v) is not int or v < 0:",
     "        if not isinstance(v, (int, float)):",
     f"{T_DS}::test_a_visit_count_that_is_not_a_NON_NEGATIVE_INT_is_refused[-1]"),
    ("a NaN or infinite policy mass reaches the ranking", DS,
     "        if not _real(v) or v < 0:",
     "        if False:",
     f"{T_DS}::test_a_policy_mass_that_is_not_a_FINITE_NON_NEGATIVE_REAL_is_refused[nan]"),
    ("an all-zero policy is ranked by tie-break alone", DS,
     "    if sum(float(v) for v in policy.values()) <= 0:",
     "    if False:",
     f"{T_DS}::test_an_all_zero_policy_is_refused_because_its_ranking_would_be_arbitrary"),
    ("an observable may CONTRADICT the map it describes", DS,
     "        if type(got) is not int or got != want:",
     "        if False:",
     f"{T_DS}::test_an_observable_that_CONTRADICTS_the_maps_it_describes_is_refused[n_legal-99]"),
    # (the LOOSE-comparison control that stood here lost its anchor when the
    #  redundant bool clause was deleted, and this round's "a rank comparison
    #  admits a bool equal to the right number" is the same defect against the
    #  same test -- kept once, below, rather than twice with one of them stale.)
    ("a truthy int passes as the override flag", DS,
     '    if type(inc["readout_overrode_leader"]) is not bool:',
     "    if False:",
     f"{T_DS}::test_a_non_boolean_override_flag_is_refused"),
    ("validation lets the policy and the visits differ", DS,
     "    if {DP._move(k) for k in policy} != {DP._move(k) for k in visits}:",
     "    if False:",
     f"{T_DS}::test_the_policy_and_the_visits_must_cover_the_SAME_legal_set"),
    ("a depth-6 move of strings is accepted", DS,
     "    if (not isinstance(mv, (list, tuple)) or len(mv) != 2\n"
     "            or any(type(x) is not int for x in mv)):",
     "    if False:",
     f"{T_DS}::test_a_depth6_move_that_is_not_a_PAIR_OF_INTS_is_refused"),
    # the complement: DESCRIPTIVE, and it must not become a floor
    ("a small complement is reported as INSUFFICIENT SUPPORT", DS,
     "    if not met:",
     '    if not met or (secondary["readout"]["position"]["complement_rank_raw_le_k"][0]\n'
     '                   < FLOOR["positions"]):',
     f"{T_DS}::test_a_TINY_complement_yields_an_ORDINARY_NO_GO_not_insufficient_support"),
    ("the complement count is never reported", DS,
     '            "complement_rank_raw_le_k": [n_elig, n_elig / len(sel)],',
     '            "complement_rank_raw_le_k": [0, 0.0],',
     f"{T_DS}::test_the_complement_is_REPORTED_per_role_and_decides_nothing"),
    # the frozen descriptive representation
    ("the summary averages the ranks instead of counting them", DS,
     "    return [[v, counts[v]] for v in sorted(counts)]",
     "    return [[sum(counts) / len(counts), len(rows)]]",
     f"{T_DS}::test_the_histogram_counts_ranks_and_not_something_averaged"),
    ("the histogram is ordered by COUNT instead of by rank", DS,
     "    return [[v, counts[v]] for v in sorted(counts)]",
     "    return [[v, counts[v]] for v in sorted(counts, key=lambda k: -counts[k])]",
     f"{T_DS}::test_the_histogram_is_ordered_by_RANK_even_when_the_counts_disagree"),
    ("the histogram emits zero-count ranks", DS,
     "    return [[v, counts[v]] for v in sorted(counts)]",
     "    return [[v, counts.get(v, 0)] for v in range(1, max(counts) + 1)]",
     f"{T_DS}::test_the_histogram_is_ordered_by_RANK_even_when_the_counts_disagree"),
    ("the row drops the rank fields the summary histograms", DS,
     '        "selected_visit_rank": inc["selected_visit_rank"],',
     '        "selected_visit_rank": 1,',
     f"{T_DS}::test_the_row_carries_BOTH_rank_fields_the_summary_needs"),
    ("the row hardcodes the POLICY rank the summary histograms", DS,
     '        "selected_policy_rank": inc["selected_policy_rank"],',
     '        "selected_policy_rank": 1,',
     f"{T_DS}::test_the_row_carries_BOTH_rank_fields_the_summary_needs"),
    # ───── 2026-09-09 second pre-execution repair: four validation gaps ──────
    ("the depth is coerced, so '6' and 6.0 pass a type-strict contract", DS,
     '        if type(d["depth"]) is not int:',
     "        if False:",
     f"{T_DS}::test_a_malformed_depths_container_or_entry_is_REFUSED_BY_NAME[depths2-a depth is an int]"),
    ("a non-mapping depth entry escapes as an AttributeError", DS,
     "        if not isinstance(d, dict):",
     "        if False:",
     f"{T_DS}::test_a_malformed_depths_container_or_entry_is_REFUSED_BY_NAME[depths1-depth record]"),
    ("a rank comparison admits a bool equal to the right number", DS,
     "        if type(got) is not int or got != want:",
     "        if got != want:",
     f"{T_DS}::test_a_BOOLEAN_rank_is_refused_even_when_it_EQUALS_the_right_number"),
    ("the depths container itself is never validated", DS,
     "    if not isinstance(depths, list) or not depths:",
     "    if False:",
     f"{T_DS}::test_a_malformed_depths_container_or_entry_is_REFUSED_BY_NAME[not-a-list-depths]"),
    ("the search budget is not bound at all", DS,
     "    if total != budget:",
     "    if total <= 0:",
     f"{T_DS}::test_a_SELF_CONSISTENT_but_under_searched_root_is_REFUSED"),
    ("the budget is retyped in this module instead of read", DS,
     '    sims = D1P.frozen_incumbent_identity()["eval_config"]["mcts_sims"]',
     "    sims = 400",
     f"{T_DS}::test_the_simulation_budget_is_READ_from_the_frozen_identity_not_retyped"),
    ("derived values are compared APPROXIMATELY again", DS,
     "        if not _real(got) or got != want:",
     "        if not _real(got) or not __import__('math').isclose(got, want, rel_tol=1e-9,\n"
     "                                                            abs_tol=1e-12):",
     f"{T_DS}::test_a_derived_value_perturbed_BELOW_any_tolerance_is_still_refused[root_top1_share]"),
    # 🔴 RE-AIMED. This deleted a REDUNDANT `isinstance(got, bool)` clause -- `_real`
    # already rejects bools -- so nothing observable changed and it was NOT
    # CAUGHT. The duplicate clauses are gone; the defect now weakens the single
    # owner, which is observable wherever a bool could equal the expected value.
    ("_real accepts bools, so True passes as a number", DS,
     "    return type(x) in (int, float) and math.isfinite(x)",
     "    return isinstance(x, (int, float)) and math.isfinite(x)",
     f"{T_DS}::test_a_policy_mass_that_is_not_a_FINITE_NON_NEGATIVE_REAL_is_refused[True]"),
    ("top2 is silently omitted rather than declared unused", DS,
     '    UNUSED_OBSERVABLES = ("top2",)' if False else 'UNUSED_OBSERVABLES = ("top2",)',
     'UNUSED_OBSERVABLES = ()',
     f"{T_DS}::test_top2_is_declared_UNUSED_rather_than_silently_omitted"),
    # ═══════════════ 2026-09-09: H2 deterministic-readout, code+tests ═════════
    # Nothing below executes a game: every injection injures a rule, a plan, a
    # barrier or the wrapper, and the named test must catch it.
    ("the parity rule reads 0.75 instead of parity", H2R,
     "PARITY = 0.50",
     "PARITY = 0.75",
     f"{T_H2}::test_the_card_numbers_are_the_module_numbers"),
    ("the verdict is decided at the wrong side of the interval", H2R,
     "    if lo > PARITY:\n        return \"T1J_STRONGER\"",
     "    if hi > PARITY:\n        return \"T1J_STRONGER\"",
     f"{T_H2}::test_the_parity_rule_at_and_around_the_threshold[0.45-0.55-INCONCLUSIVE]"),
    ("AT parity counts as above it", H2R,
     "    if lo > PARITY:",
     "    if lo >= PARITY:",
     f"{T_H2}::test_the_parity_rule_at_and_around_the_threshold[0.5-0.6-INCONCLUSIVE]"),
    ("the sample size drifts from the card", H2R,
     "N_REPS = 46\nN_GAMES = N_OPENINGS * N_ARMS * N_REPS",
     "N_REPS = 14\nN_GAMES = N_OPENINGS * N_ARMS * N_REPS",
     f"{T_H2}::test_the_card_numbers_are_the_module_numbers"),
    ("the per-cell threshold is relaxed by one", H2R,
     "MIN_DISTINCT_PER_CELL = 42",
     "MIN_DISTINCT_PER_CELL = 41",
     f"{T_H2}::test_the_per_cell_threshold_accepts_AT_42_and_refuses_one_below[41-False]"),
    # 🔑 THE VACUITY THE TRANSCRIPT DEFINITION EXISTS TO PREVENT
    ("the transcript carries the SEED, so identical play looks distinct", H2R,
     '        out.append((mover, _int(move[0], "row"), _int(move[1], "col")))',
     '        out.append((mover, _int(move[0], "row"), _int(move[1], "col"),\n'
     '                    p.get("seed"), p.get("task_id")))',
     f"{T_H2}::test_two_games_with_DIFFERENT_seeds_and_ids_but_IDENTICAL_PLAY_are_ONE_transcript"),
    ("the transcript drops the moves, so every game looks the same", H2R,
     '        out.append((mover, _int(move[0], "row"), _int(move[1], "col")))',
     "        pass",
     f"{T_H2}::test_changing_ONE_played_move_creates_a_DISTINCT_transcript"),
    ("the ply sequence is checked for CONTIGUITY, not for its exact span", H2R,
     "    if got != want:",
     "    if got != list(range(min(got), max(got) + 1)) if got else False:",
     f"{T_H2}::test_REMOVING_THE_FIRST_ply_REFUSES_though_the_rest_stays_contiguous"),
    ("a truncated tail passes because only the head is anchored", H2R,
     "    if got != want:",
     "    if got and got[0] != want[0]:",
     f"{T_H2}::test_REMOVING_THE_FINAL_ply_REFUSES_though_the_rest_stays_contiguous"),
    ("movers must merely ALTERNATE, not match the arm", H2R,
     "        if mover != want_mover:",
     "        if False:",
     f"{T_H2}::test_FLIPPING_EVERY_MOVER_refuses_even_though_alternation_is_preserved"),
    ("the terminal reason is not checked against the two the protocol has", H2R,
     "    if reason not in TERMINAL_REASONS:",
     "    if False:",
     f"{T_H2}::test_a_terminal_reason_outside_the_two_is_REFUSED"),
    ("coordinates are coerced instead of type-checked", H2R,
     "    if type(value) is not int:",
     "    if not isinstance(value, (int, float, str)) and False:",
     f"{T_H2}::test_a_move_whose_coordinates_are_STRINGS_is_refused"),
    # the per-cell screen
    ("the degeneracy screen counts GLOBALLY instead of per cell", H2R,
     '        key = (str(g["opening"]), str(g["colour_arm"]))',
     '        key = ("all", "cells")',
     f"{T_H2}::test_ONE_WHOLLY_COLLAPSED_CELL_FAILS_though_the_GLOBAL_rate_is_above_90_percent"),
    # ⚠ THE 16-CELL CLAUSE IS GONE, and its control with it -- not re-aimed. With
    # the vector bound to 736 rows and every cell to exactly 46 games, sixteen
    # cells follow arithmetically, so deleting the clause changed nothing
    # observable. My replacement control was worse than none: it injected the SAME
    # defect as "a cell with fewer than 46 GAMES is screened anyway" below, at a
    # test that refuses on the ROW COUNT first, so it could never be caught. One
    # injection, one owner, one control -- the entry below.
    # THE ORDER: the screen must gate the interval
    ("the interval is computed even when the screen fails", H2R,
     '    if not screen["passes"]:',
     "    if False:",
     f"{T_H2}::test_a_FAILING_SCREEN_PREVENTS_THE_INTERVAL_FROM_BEING_COMPUTED"),
    # ⚠ RE-ANCHORED: the first version removed the copy inside `overall`, while the
    # test reads the TOP-LEVEL one, so nothing observable changed.
    ("the report drops the interval's standing", H2R,
     '        "interval_standing": INTERVAL_STANDING,\n        "outcome": verdict,',
     '        "outcome": verdict,',
     f"{T_H2}::test_the_report_carries_the_interval_STANDING_and_all_sixteen_counts"),
    # the plan
    ("the plan assigns seeds by membership, not POSITION", H2P,
     '        if type(t["seed"]) is not int or t["seed"] != lo + i:',
     "        if False:",
     f"{T_H2}::test_the_schedule_validator_refuses_a_broken_design[<lambda>-bound POSITIONALLY]"),
    ("the plan omits the readout mode from its tasks", H2P,
     '                    "selection_mode": RULES.SELECTION_MODE,',
     "",
     f"{T_H2}::test_every_row_carries_its_POSITIONAL_seed_and_the_readout_mode"),
    ("the schedule accepts a task without the readout mode", H2P,
     '        if t.get("selection_mode") != RULES.SELECTION_MODE:',
     "        if False:",
     f"{T_H2}::test_the_schedule_validator_refuses_a_broken_design[<lambda>-selection_mode]"),
    ("a short schedule is accepted", H2P,
     '    if len(tasks) != RULES.N_GAMES:\n        raise H2PlanError(f"{len(tasks)} tasks, expected exactly',
     '    if False:\n        raise H2PlanError(f"{len(tasks)} tasks, expected exactly',
     f"{T_H2}::test_the_schedule_validator_refuses_a_broken_design[<lambda>-expected exactly]"),
    # the barriers
    ("the gate defaults OPEN", H2RUN,
     "H2_EXECUTION_AUTHORIZED = False",
     "H2_EXECUTION_AUTHORIZED = True",
     f"{T_H2}::test_THE_GATE_IS_SHUT_IN_THE_REAL_REPOSITORY"),
    # 🔴 REMOVED 2026-09-12, AND NOT RE-AIMED. "The public entry never reads the
    #  gate" deleted `check_gate()` from `run_h2`. With the production seam wired,
    #  that control COMPILED THE HELPER, LOADED THE MODEL AND PLAYED 383 REAL GAMES,
    #  spending a registered seed block. A control whose content is "remove the only
    #  safeguard" must not exist in a harness that can reach production: the property
    #  is now held by `test_the_public_entry_READS_THE_GATE_before_anything_else`
    #  (behaviour) and by an AST test that the call is present and FIRST, neither of
    #  which can execute anything. The seam also checks the gate itself and is inert
    #  under a test runner -- see the containment tests.
    ("the registration barrier checks only the first seed", H2RUN,
     "    missing = [s for s in range(lo, hi) if not REF.seed_is_accounted(s)]",
     "    missing = [s for s in [lo] if not REF.seed_is_accounted(s)]",
     f"{T_H2}::test_the_registration_barrier_checks_EVERY_seed_not_the_endpoints"),
    # ⚠ RE-AIMED: the inverted test now asserts the barrier is SATISFIED, which a
    # disabled barrier also satisfies. Only a test that STRIPS the block can see it.
    ("the registration barrier is disabled", H2RUN,
     "    if missing:",
     "    if False:",
     f"{T_H2}::test_an_UNREGISTERED_block_is_still_refused"),
    ("outputs are overwritten instead of refused", H2RUN,
     "        if os.path.lexists(path):",
     "        if False:",
     f"{T_H2}::test_ALL_THREE_OUTPUTS_are_preflighted_before_any_game"),
    ("one path may serve as both results and trace", H2RUN,
     "            if _canonical(pa) == _canonical(pb):",
     "            if False:",
     f"{T_H2}::test_NO_TWO_OUTPUTS_MAY_BE_THE_SAME_FILE[pair0]"),
    # the identity binding -- H2 IS the readout change
    ("the identity keeps the OLD readout", H2RUN,
     '    cfg["selection_mode"] = RULES.SELECTION_MODE',
     "    pass",
     f"{T_H2}::test_the_frozen_identity_carries_ARGMAX_and_records_the_inert_settings"),
    ("the inert settings are left looking ACTIVE", H2RUN,
     "    inert = {k: cfg.pop(k) for k in RULES.INERT_UNDER_ARGMAX if k in cfg}",
     "    inert = {k: cfg[k] for k in RULES.INERT_UNDER_ARGMAX if k in cfg}",
     f"{T_H2}::test_the_frozen_identity_carries_ARGMAX_and_records_the_inert_settings"),
    ("a run recording the OLD readout is reported as H2", H2RUN,
     "    if got_mode != RULES.SELECTION_MODE:",
     "    if False:",
     f"{T_H2}::test_an_identity_that_still_says_opening_temperature_VOIDS"),
    ("the identity is compared SHALLOWLY again", H2RUN,
     '    _same(dict(identity), want, "incumbent_identity")',
     "    pass",
     f"{T_H2}::test_an_identity_naming_another_model_VOIDS[reference]"),
    ("the schedule digest is not compared with the pin", H2RUN,
     '    if summary["task_digest"] != RULES.H2_TASK_DIGEST:',
     "    if False:",
     f"{T_H2}::test_the_schedule_must_BE_the_frozen_736_by_digest"),
    ("the deadline never fires", H2RUN,
     "            if deadline.elapsed() > deadline_s:",
     "            if False:",
     f"{T_H2}::test_the_deadline_VOIDS_mid_run_and_reports_no_partial_rate"),
    # ⚠ RE-WRITTEN: the first replacement still CALLED `transcript` behind an
    # `if False`, so it injured nothing. This one skips it outright.
    ("a malformed ply record is skipped instead of VOIDing", H2RUN,
     '            t = RULES.transcript(out["plies"], row,\n'
     '                                 opening_bound=out["opening_bound"])',
     '            t = ("fabricated", i)',
     f"{T_H2}::test_a_malformed_ply_record_VOIDS_the_run_rather_than_being_counted"),
    # the wrapper
    ("the wrapper reports success without verifying the gate", H2CMD,
     '    finally:\n'
     '        # 🔴 RESTORED WHATEVER HAPPENED -- refusal, timeout, interrupt, crash or\n'
     '        # completion -- and a failed restoration SUPERSEDES every other code,\n'
     '        # including a refusal: an open gate is the larger fact.\n'
     '        if not restore_gate(_runner_source):',
     '    finally:\n'
     '        if False:',
     f"{T_H2}::test_THE_FINALLY_PATH_also_restores_and_reports_its_own_failure"),
    ("restoration claims success without reading the file back", H2CMD,
     "    return back.count(_GATE_CLOSED + \"\\n\") == 1 and not _GATE_OPEN.search(back)",
     "    return True",
     f"{T_H2}::test_restore_gate_is_FALSE_when_the_READBACK_disagrees"),
    ("the wrapper gains a --runner-source override", H2CMD,
     '    ap.add_argument("--trace", default=DEFAULT_TRACE)',
     '    ap.add_argument("--trace", default=DEFAULT_TRACE)\n'
     '    ap.add_argument("--runner-source", default=RUNNER_SOURCE)',
     f"{T_H2}::test_the_wrapper_has_NO_runner_source_flag"),
    # ══════════ 2026-09-09 implementation repair: seven production defects ═════
    # 1. THE MOVER RULE -- the one that would have voided half the schedule.
    ("the mover is derived from the ARM again", H2R,
     '    return STARTING_COLOUR if _int(ply, "ply") % 2 == 1 else (',
     '    return "black" if _int(ply, "ply") % 2 == 1 else (',
     f"{T_H2}::test_THE_MOVER_FOLLOWS_PLY_PARITY_AND_NOT_THE_ARM_checked_against_the_engine"),
    ("the starting colour drifts from the engine's", H2R,
     'STARTING_COLOUR = "red"',
     'STARTING_COLOUR = "black"',
     f"{T_H2}::test_THE_MOVER_FOLLOWS_PLY_PARITY_AND_NOT_THE_ARM_checked_against_the_engine"),
    # 2. A PUBLIC ENTRY THAT CAN BE FED GAMEPLAY
    ("the public entry accepts caller-supplied play again", H2RUN,
     "def run_h2(*, results_path: str, trace_path: str, report_path: str) -> Dict[str, Any]:",
     "def run_h2(*, results_path: str, trace_path: str, report_path: str,\n"
     "           play=None, tasks=None, identity=None) -> Dict[str, Any]:",
     f"{T_H2}::test_the_public_entry_TAKES_ONLY_THE_OUTPUT_PATHS"),
    ("the production seam is a refusing stub again", H2RUN,
     "        cap = _CapturingRecorder()",
     "        raise H2Error('not wired')\n        cap = _CapturingRecorder()",
     f"{T_H2}::test_the_production_play_seam_EXISTS_and_is_built_lazily"),
    ("the seam no longer calls the harness game loop", H2RUN,
     '        result = state["harness"].play_task(',
     # ⚠ REWRITTEN TWICE. v1 still CONTAINED the call behind an `if False`, so the
     # AST test found it and the control injured nothing. v2 removed the call but
     # left its keyword arguments dangling after `_unused = (` -- a SYNTAX ERROR,
     # so `h2_match_runner` would not import and pytest collected nothing (rc=4).
     # The call is all-keyword, so swapping the callee for `dict` is valid Python,
     # executes, and leaves no `play_task` anywhere for the AST test to find.
     '        result = dict(',
     f"{T_H2}::test_the_seam_WIRES_THE_HARNESS_GAME_LOOP"),
    ("the production seam loads the model AT IMPORT", H2RUN,
     "from . import h2_match_plan as PLAN",
     "from . import h2_match_plan as PLAN\nfrom . import e4_screen_runner as _EAGER",
     f"{T_H2}::test_NOTHING_EFFECTFUL_IS_IMPORTED_AT_MODULE_LEVEL"),
    ("the incumbent keeps the temperature config in production", H2RUN,
     '            argmax_cfg = cfg.__class__(**{**cfg.__dict__,\n'
     '                                          "selection_mode": RULES.SELECTION_MODE})',
     "            argmax_cfg = cfg",
     f"{T_H2}::test_the_seam_BUILDS_AN_ARGMAX_CONFIG_rather_than_reusing_the_frozen_one"),
    # 3. SCHEDULE AND IDENTITY BINDING
    # (removed with the branch it injured: the field-by-field rebuild comparison
    #  was UNREACHABLE behind the full-field pin, so this went NOT CAUGHT. The pin
    #  below is the single owner.)
    # 5. THE SYMLINK AND THE PROTECTED BLOCK
    ("a DANGLING SYMLINK reads as absent", H2RUN,
     "        if os.path.lexists(path):",
     "        if os.path.exists(path):",
     f"{T_H2}::test_a_DANGLING_SYMLINK_at_an_output_path_is_REFUSED"),
    # 6. THE SCREEN BINDS ITS INPUT
    ("the screen accepts a SHORT transcript vector", H2R,
     "    elif len(per_game) != N_GAMES:",
     "    elif False:",
     f"{T_H2}::test_a_SHORT_VECTOR_of_42_per_cell_is_REFUSED_not_passed"),
    ("the screen ignores the canonical task ids", H2R,
     "        if len(got) != len(want) or sorted(map(str, got)) != sorted(map(str, want)):",
     "        if False:",
     f"{T_H2}::test_a_vector_whose_TASK_IDS_are_not_the_canonical_ones_is_REFUSED"),
    ("a cell with fewer than 46 GAMES is screened anyway", H2R,
     '    short = [c["cell"] for c in cells if c["n_games"] != N_REPS]',
     "    short = []",
     f"{T_H2}::test_A_FULL_LENGTH_vector_with_UNEQUAL_CELLS_is_REFUSED"),
    ("invalid results are masked as a DEGENERATE DESIGN", H2R,
     "    pairs, why = L0.bind_results(list(results), list(tasks), design=h2_design(task_digest))",
     "    pairs, why = (['dummy'], None)",
     f"{T_H2}::test_INVALID_RESULTS_are_REFUSED_not_reported_as_degenerate"),
    # 7. THE TRANSCRIPT EVIDENCE AND THE VOID RECORD
    ("the transcript evidence is not persisted", H2RUN,
     '            emit(rec, {"record_type": "transcript", "task_id": task["task_id"],',
     '            _unused = ({"record_type": "transcript", "task_id": task["task_id"],',
     f"{T_H2}::test_THE_PLIES_AND_DIGESTS_ARE_PERSISTED_so_the_screen_can_be_recomputed"),
    ("a mid-run failure leaves no run_end VOID", H2RUN,
     '        if trace is not None:',
     "        if False:",
     f"{T_H2}::test_ANY_mid_run_failure_still_writes_run_end_VOID"),
    # 4. THE SUPERVISOR
    ("the wrapper runs the match IN PROCESS, unsupervised", H2CMD,
     "                r = supervise([sys.executable, \"-m\", MODULE, \"--worker\",",
     "                r = {\"exit_code\": 0, \"timed_out\": False, \"interrupted\": False,\n"
     "                     \"group_cleared\": True} if True else supervise(\n"
     "                    [sys.executable, \"-m\", MODULE, \"--worker\",",
     f"{T_H2}::test_the_worker_is_SUPERVISED_in_its_own_group_under_an_OUTER_cap"),
    ("the outer cap is unbounded", H2CMD,
     "                          timeout_s=RUN.RUN_DEADLINE_S + SUPERVISOR_GRACE_S,",
     "                          timeout_s=None,",
     f"{T_H2}::test_the_worker_is_SUPERVISED_in_its_own_group_under_an_OUTER_cap"),
    ("a surviving descendant accompanies a success", H2CMD,
     '            if not r["group_cleared"]:',
     "            if False:",
     f"{T_H2}::test_a_timeout_an_interrupt_and_a_SURVIVING_DESCENDANT_each_get_their_own_code[r2-EXIT_CLEANUP_FAILED]"),
    ("a timeout is reported as the worker's own exit code", H2CMD,
     '            elif r["timed_out"]:\n                code = EXIT_TIMEOUT',
     "            elif False:\n                code = EXIT_TIMEOUT",
     f"{T_H2}::test_a_timeout_an_interrupt_and_a_SURVIVING_DESCENDANT_each_get_their_own_code[r0-EXIT_TIMEOUT]"),
    # ═════════ 2026-09-10: four execution blockers + two hardening gaps ═══════
    # 1. THE CLOCK -- the first production setup was guaranteed to refuse.
    ("compile is handed a fresh, unstarted clock again", H2RUN,
     "            D1._default_compile(deadline, paths=paths)",
     "            D1._default_compile(D1.Deadline(RUN_DEADLINE_S), paths=paths)",
     f"{T_H2}::test_the_seam_HANDS_COMPILE_THE_RUNS_OWN_STARTED_CLOCK"),
    ("the seam accepts an unstarted clock", H2RUN,
     "            if deadline is None or not deadline.started:",
     "            if False:",
     f"{T_H2}::test_the_seam_REFUSES_an_unstarted_or_absent_clock"),
    ("the run never starts its deadline", H2RUN,
     "    if not deadline.started:\n        deadline.start()",
     "    pass",
     f"{T_H2}::test_the_run_ARMS_the_deadline_so_a_BLOCKED_game_can_be_cut_off"),
    ("the deadline is never ARMED, so a blocked game runs on", H2RUN,
     "        stack.enter_context(sup(deadline))",
     "        pass",
     f"{T_H2}::test_the_run_ARMS_the_deadline_so_a_BLOCKED_game_can_be_cut_off"),
    # 2. THE REPORT
    ("the outcome is never classified", H2CMD,
     "    return _persist_and_classify(report, a.report)",
     '    print(f"COMPLETED: outcome {report.get(\'outcome\')!r}")\n'
     "    return EXIT_COMPLETED",
     f"{T_H2}::test_the_WORKER_persists_the_report_and_returns_its_mapped_code"),
    ("a REFUSED report still exits COMPLETED", H2CMD,
     '    if not report.get("reported"):',
     "    if False:",
     f"{T_H2}::test_every_outcome_gets_ITS_OWN_exit_code[report4-EXIT_REFUSED]"),
    ("a degenerate design is reported as a completed match", H2CMD,
     '        if outcome == "INCONCLUSIVE — DEGENERATE DESIGN":',
     "        if False:",
     f"{T_H2}::test_every_outcome_gets_ITS_OWN_exit_code[report2-EXIT_DEGENERATE]"),
    # ⚠ the report write MOVED from the wrapper to the runner, so the file did too:
    # it is persisted there now, before the terminal OK is committed.
    ("the report overwrites an existing one", H2RUN,
     "        fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)\n"
     "        with os.fdopen(fd, \"w\") as fh:",
     "        fd = os.open(report_path, os.O_WRONLY | os.O_CREAT, 0o644)\n"
     "        with os.fdopen(fd, \"w\") as fh:",
     f"{T_H2}::test_A_REPORT_THAT_CANNOT_BE_WRITTEN_LEAVES_NO_OK_TRACE"),
    ("the preflight lets an existing report through to the games", H2RUN,
     "    for label, path in named:",
     "    for label, path in ():",
     f"{T_H2}::test_an_EXISTING_report_is_refused_by_the_PREFLIGHT_before_any_game"),
    # 3. THE INTERRUPT CONTRACT
    ("an operator interrupt is recorded as VOID", H2RUN,
     '        verdict = "INTERRUPTED" if isinstance(e, KeyboardInterrupt) else "VOID"',
     '        verdict = "VOID"',
     f"{T_H2}::test_an_OPERATOR_INTERRUPT_records_INTERRUPTED_not_VOID"),
    # (removed with the duplicate clause: a KeyboardInterrupt is not an Exception,
    #  so the bare `raise` already re-raised it and naming it changed nothing.)
    # 4. THE BETWEEN-GAMES CLEANUP
    ("no cleanup runs between games", H2RUN,
     "            state[\"cleanup\"]()\n            play.cleanups += 1",
     "            pass",
     f"{T_H2}::test_THE_CLEANUP_RUNS_AFTER_EVERY_GAME_including_a_failed_one"),
    ("cleanup runs only after a SUCCESSFUL game", H2RUN,
     "        finally:\n            # AFTER EVERY GAME, completed or failed. A cleanup that runs only on\n"
     "            # the happy path is the one that matters least.\n"
     "            state[\"cleanup\"]()",
     "        finally:\n            pass\n        if True:\n            state[\"cleanup\"]()",
     f"{T_H2}::test_THE_CLEANUP_RUNS_AFTER_EVERY_GAME_including_a_failed_one"),
    ("the evaluator is reloaded for every game", H2RUN,
     "        state = play._state\n        if state is None:",
     "        state = None\n        if state is None:",
     f"{T_H2}::test_the_evaluator_is_loaded_ONCE_for_the_whole_run"),
    # THE ARGMAX CONFIG, reached rather than read
    ("the incumbent is built with the frozen TEMPERATURE config", H2RUN,
     "                        colour=REF.reference_colour(t), config=argmax_cfg,",
     "                        colour=REF.reference_colour(t), config=cfg,",
     f"{T_H2}::test_the_incumbent_IS_ACTUALLY_BUILT_WITH_AN_ARGMAX_CONFIG"),
    # HARDENING: the full-field pin and the worker bypass
    ("the schedule is pinned only by its DESIGN digest", H2RUN,
     "    if full != RULES.H2_FULL_TASK_DIGEST:",
     "    if False:",
     f"{T_H2}::test_a_forged_reference_sha256_is_REFUSED_though_the_DIGEST_still_matches"),
    ("the full-field digest projects fields away", H2R,
     "    return hashlib.sha256(json.dumps([dict(t) for t in tasks], sort_keys=True,",
     '    return hashlib.sha256(json.dumps([{"task_id": t["task_id"]} for t in tasks],\n'
     "                                     sort_keys=True,",
     f"{T_H2}::test_THE_FULL_FIELD_DIGEST_ITSELF_distinguishes_a_forged_field"),
    ("--worker is a public bypass again", H2CMD,
     "    if not _consume_capability(a.capability_fd):",
     "    if False:",
     f"{T_H2}::test_WORKER_REFUSES_without_the_supervisors_CAPABILITY"),
    # ═══════ 2026-09-10: preflight, terminal ordering, capability, wording ════
    ("the report is left out of the preflight", H2RUN,
     "    if not report_path:",
     "    if False:",
     f"{T_H2}::test_ALL_THREE_OUTPUTS_are_preflighted_before_any_game"),
    ("only two outputs are compared for aliasing", H2RUN,
     "    named = ((\"results\", results_path), (\"trace\", trace_path), (\"report\", report_path))",
     "    named = ((\"results\", results_path), (\"trace\", trace_path))",
     f"{T_H2}::test_NO_TWO_OUTPUTS_MAY_BE_THE_SAME_FILE[pair1]"),
    ("the terminal OK is committed BEFORE the report is durable", H2RUN,
     "        report = RULES.h2_report(results, list(tasks), per_game,",
     '        emit(trace, {"event": "run_end", "verdict": "OK",\n'
     '                     "games_completed": len(results)})\n'
     "        report = RULES.h2_report(results, list(tasks), per_game,",
     f"{T_H2}::test_A_REPORT_THAT_CANNOT_BE_WRITTEN_LEAVES_NO_OK_TRACE"),
    # (removed: an fsync is NOT OBSERVABLE in-process, so this control could never
    #  fail. The ordering control above covers the risk that matters -- a terminal
    #  OK committed before the report is durable.)
    ("the trace does not name the verdict it committed", H2RUN,
     '                     "outcome": report.get("outcome")})',
     "                     })",
     f"{T_H2}::test_THE_RUNNER_WRITES_THE_REPORT_BEFORE_THE_TERMINAL_OK"),
    # (removed: whether the first descriptor is registered with the ExitStack is
    #  NOT OBSERVABLE, because CPython refcounting closes the file object as soon as
    #  it goes out of scope. The stack makes the close deterministic and
    #  exception-safe, which is why the code does it; no control can tell.)
    # the capability
    ("the capability is a fixed, caller-settable value", H2CMD,
     "        fh.write(secrets.token_hex(CAPABILITY_BYTES))",
     '        fh.write("1" * 64)',
     f"{T_H2}::test_the_capability_is_RANDOM_and_carries_no_path"),
    # ⚠ the first version of this injected a bare `import tempfile`, which changed
    # nothing and went NOT CAUGHT. This one actually stops the channel being a pipe.
    ("the capability is not an anonymous pipe", H2CMD,
     "    r_fd, w_fd = os.pipe()",
     '    r_fd = os.open("/dev/null", os.O_RDONLY)\n'
     '    w_fd = os.open("/dev/null", os.O_WRONLY)',
     f"{T_H2}::test_THE_CAPABILITY_IS_A_PIPE_AND_IS_SINGLE_USE"),
    # (removed: duplicating the descriptor instead of closing it changes NOTHING
    #  observable -- a second read finds EOF either way, because single-use comes
    #  from the pipe being DRAINED. The test now says so explicitly.)

    ("any token of the RIGHT LENGTH is accepted", H2CMD,
     '    return len(token) == CAPABILITY_BYTES * 2 and all(\n'
     '        c in "0123456789abcdef" for c in token)',
     "    return len(token) == CAPABILITY_BYTES * 2",
     f"{T_H2}::test_a_FORGED_capability_is_refused_INCLUDING_one_of_the_RIGHT_LENGTH[gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg]"),
    ("any capability at all is accepted", H2CMD,
     '    return len(token) == CAPABILITY_BYTES * 2 and all(\n'
     '        c in "0123456789abcdef" for c in token)',
     "    return True",
     f"{T_H2}::test_a_FORGED_capability_is_refused_INCLUDING_one_of_the_RIGHT_LENGTH[short]"),
    # 🔴 THE DESTRUCTIVE FAULT: nothing may be deleted, ever.
    ("the capability check DELETES the thing it was given", H2CMD,
     '    if fd is None:\n        return False',
     '    if fd is None:\n        return False\n    import os as _o\n'
     '    _o.unlink(str(fd)) if isinstance(fd, str) else None',
     f"{T_H2}::test_the_capability_check_TAKES_A_DESCRIPTOR_not_a_path"),
    ("an environment variable reaches the worker path", H2CMD,
     "    if not _consume_capability(a.capability_fd):",
     '    if os.environ.get("H2_SUPERVISED_WORKER") != "1":',
     f"{T_H2}::test_NO_ENVIRONMENT_VARIABLE_reaches_the_worker_path"),
    ("a capability outlives the run", H2CMD,
     "                    os.close(cap_fd)         # nothing on disk, nothing to unlink",
     "                    pass",
     f"{T_H2}::test_the_supervisor_HANDS_the_child_a_capability_and_leaves_none_behind"),
    ("the descriptor is not PASSED to the child at all", H2CMD,
     "                              pass_fds=[cap_fd])",
     "                              pass_fds=[])",
     f"{T_H2}::test_the_supervisor_HANDS_the_child_a_capability_and_leaves_none_behind"),
    # ⚠ RE-AIMED at a test that spawns a REAL child: the H2 wrapper test uses a fake
    # supervise, so the Popen line was never exercised and this went NOT CAUGHT.
    ("the shared supervisor drops pass_fds", RQ,
     "    p = subprocess.Popen(list(cmd), start_new_session=True, pass_fds=tuple(pass_fds))",
     "    p = subprocess.Popen(list(cmd), start_new_session=True)",
     f"{T_RQ}::test_supervise_PASSES_AN_INHERITED_DESCRIPTOR_to_the_child"),
    # the classifier verifies its own artifact
    ("the classifier assumes the report exists", H2CMD,
     "    if not os.path.lexists(report_path):",
     "    if False:",
     f"{T_H2}::test_a_MISSING_report_file_is_VOID_not_a_completed_run"),
    # P2: the wording
    ("the degeneracy refusal claims the games were non-independent", H2R,
     '                       f"is NOT computed: the preregistered DIVERSITY screen failed. "',
     '                       f"is NOT computed; those repetitions are not independent plays. "',
     f"{T_H2}::test_the_degeneracy_refusal_CLAIMS_NO_DEPENDENCE"),
    # ═════════════ 2026-09-11: the H2 SEED REGISTRATION, four ways ═══════════
    ("H2's block un-registered from ACCOUNTED (the edit reverted)", REF_SRC,
     "    (202618000, 202618736),          # H2 DETERMINISTIC-READOUT HEAD-TO-HEAD. 736",
     "    # (202618000, 202618736),        # H2 DETERMINISTIC-READOUT HEAD-TO-HEAD. 736",
     f"{T_H2}::test_ATTEMPT_ONES_BLOCK_IS_STILL_SPENT_and_the_retirement_STANDS"),
    ("H2's block ALSO marked EXPOSED -- a reservation claimed as a draw", REF_SRC,
     "EXPOSED_SEED_INTERVALS = (\n",
     "EXPOSED_SEED_INTERVALS = (\n    (202618000, 202618736),\n",
     f"{T_H2}::test_ATTEMPT_ONES_BLOCK_IS_STILL_SPENT_and_the_retirement_STANDS"),
    # ⚠ RE-AIMED after the VOID: marking the block RETIRED is now the TRUE state, so
    # the defect is the opposite one -- a spent block left un-retired and therefore
    # schedulable again.
    ("the SPENT block is not retired, so it can be scheduled again", REF_SRC,
     "    (202618000, 202618736),          # H2, retired WHOLE 2026-09-11 after the single",
     "    # (202618000, 202618736),        # H2, retired WHOLE 2026-09-11 after the single",
     f"{T_H2}::test_ATTEMPT_ONES_BLOCK_IS_STILL_SPENT_and_the_retirement_STANDS"),
    # 🔴 RELABELLED 2026-09-12: this carried the SAME label as the D1 control over
    # the same injection, and reasons are now keyed BY LABEL. The two are distinct
    # controls -- same removal, different test -- so both stay, under names that
    # say which one they are.
    ("a spent seed is schedulable because availability is never checked (H2)", REF_SRC,
     "def validate_task_executable(task: Dict[str, Any]) -> None:",
     "def validate_task_executable(task: Dict[str, Any]) -> None:\n    return None",
     f"{T_H2}::test_THE_SPENT_BLOCK_CANNOT_BE_SCHEDULED_AGAIN"),
    # (the two blocker controls that stood here are gone with the tests they named:
    #  they pinned the PRE-REPAIR refusal, which the repair deliberately removed.)
    # 🔑 REGISTRATION IS BOOKKEEPING, NOT PERMISSION -- the claim the same test makes
    # in its second half, so "the gate stayed shut" is something a test can deny.
    ("registration ALSO opened the H2 gate", H2RUN,
     "H2_EXECUTION_AUTHORIZED = False",
     "H2_EXECUTION_AUTHORIZED = True",
     f"{T_H2}::test_ATTEMPT_ONES_BLOCK_IS_STILL_SPENT_and_the_retirement_STANDS"),
    # ⚠ RE-ANCHORED AGAIN 2026-09-12, onto ATTEMPT 3's line. `_registry_without_h2`
    # strips whatever `H2_SEED_BLOCK` currently is, so once that became attempt 3
    # this control was removing an entry the strip no longer touches and went NOT
    # CAUGHT. Stripping a block that is not registered removes nothing, and
    # `_registry_without_h2` asserts the strip bit.
    ("the negative control stops stripping anything", REF_SRC,
     "    (202622000, 202622736),          # H2 ATTEMPT 3 -- the match H2 has still not",
     "    # (202622000, 202622736),        # H2 ATTEMPT 3 -- the match H2 has still not",
     f"{T_H2}::test_an_UNREGISTERED_block_is_still_refused"),
    # ══════════ 2026-09-11: the qualified-builder repair, both halves ════════
    # HALF ONE -- H2's readout must be ADMITTED.
    ("the builder refuses H2's readout again", G3SRC,
     "            if got not in ADMISSIBLE_SELECTION_MODES:",
     "            if True:",
     f"{T_H2}::test_THE_BUILDER_CONSTRUCTS_AN_AGENT_WITH_H2s_ARGMAX_CONFIG"),
    ("the admitted set loses argmax", G3SRC,
     'ADMISSIBLE_SELECTION_MODES = ("opening_temperature", "argmax")',
     'ADMISSIBLE_SELECTION_MODES = ("opening_temperature",)',
     f"{T_H2}::test_the_ADMITTED_MODES_are_DECLARED_and_are_exactly_two"),
    # ⚠ REWRITTEN: the first version rebound a LOCAL name inside the checker, which
    # the builder never reads again -- it injured nothing. The defect that matters is
    # the agent being built with a config the check did not see.
    ("the readout is named argmax but the AGENT is built from the frozen config", G3SRC,
     '        evaluator=evaluator, colour=colour, seed=task["seed"], config=cfg,',
     '        evaluator=evaluator, colour=colour, seed=task["seed"], config=eval_config(),',
     f"{T_H2}::test_THE_BUILDER_CONSTRUCTS_AN_AGENT_WITH_H2s_ARGMAX_CONFIG"),
    # HALF TWO -- EVERYTHING ELSE must still be refused.
    ("admitting the readout admits every other field too", G3SRC,
     "        if type(got) is not type(want) or got != want:",
     "        if False:",
     f"{T_H2}::test_EVERY_UNRELATED_FIELD_is_still_REFUSED[mcts_sims-800]"),
    ("the simulation budget may drift alongside the readout", G3SRC,
     "        if type(got) is not type(want) or got != want:",
     "        if f.name != 'mcts_sims' and (type(got) is not type(want) or got != want):",
     f"{T_H2}::test_a_field_that_differs_ALONGSIDE_the_readout_is_still_refused"),
    ("fields are compared loosely, so 400 and 400.0 agree", G3SRC,
     "        if type(got) is not type(want) or got != want:",
     "        if got != want:",
     f"{T_H2}::test_a_field_whose_VALUE_MATCHES_but_whose_TYPE_DIFFERS_is_refused[mcts_sims-400.0]"),
    ("any selection_mode string is admitted", G3SRC,
     "            if got not in ADMISSIBLE_SELECTION_MODES:",
     "            if not isinstance(got, str):",
     f"{T_H2}::test_an_UNADMITTED_readout_mode_is_REFUSED[hoeffding_lcb]"),
    ("a config of the wrong TYPE is accepted", G3SRC,
     "    if type(cfg) is not type(frozen):",
     "    if False:",
     f"{T_H2}::test_a_config_of_the_WRONG_TYPE_is_refused"),
    ("the check is skipped entirely", G3SRC,
     "    _check_config(cfg)",
     "    pass",
     f"{T_H2}::test_EVERY_UNRELATED_FIELD_is_still_REFUSED[board_size-19]"),
    ("omitting the config no longer falls back to the frozen one", G3SRC,
     "    cfg = config if config is not None else eval_config()",
     "    cfg = config",
     f"{T_H2}::test_omitting_the_config_still_uses_the_frozen_one"),
    # ═══════════ 2026-09-11: the H2 RETRY -- fresh block, both pins ══════════
    ("the retry block un-registered from ACCOUNTED", REF_SRC,
     "    (202620000, 202620736),          # H2 ATTEMPT 2 -- THE RETRY against the",
     "    # (202620000, 202620736),        # H2 ATTEMPT 2 -- THE RETRY against the",
     f"{T_H2}::test_THE_RETRY_BLOCK_IS_EXPOSED_383_AND_RETIRED_WHOLE"),
    # 🔴 BOTH OF THESE INVERTED 2026-09-12, BY THE INCIDENT. "ALSO marked RETIRED /
    # EXPOSED before it is drawn" were the defects while the block was unspent. It is
    # now EXPOSED 383 and RETIRED WHOLE, so what must fail is DROPPING either record
    # -- and claiming a draw the record does not support.
    ("the WHOLE-BLOCK retirement is dropped", REF_SRC,
     "    (202620000, 202620736),          # H2 ATTEMPT 2, RETIRED WHOLE 2026-09-12.",
     "    # (202620000, 202620736),        # H2 ATTEMPT 2, RETIRED WHOLE 2026-09-12.",
     f"{T_H2}::test_THE_RETRY_BLOCK_IS_EXPOSED_383_AND_RETIRED_WHOLE"),
    ("the exposure is widened to the WHOLE block, claiming task 383's seed as drawn",
     REF_SRC,
     "    (202620000, 202620383),          # H2 ATTEMPT 2's block, DRAWN BY AN",
     "    (202620000, 202620736),          # H2 ATTEMPT 2's block, DRAWN BY AN",
     f"{T_H2}::test_THE_RETRY_BLOCK_IS_EXPOSED_383_AND_RETIRED_WHOLE"),
    ("the 383 confirmed draws are not recorded as EXPOSED at all", REF_SRC,
     "    (202620000, 202620383),          # H2 ATTEMPT 2's block, DRAWN BY AN",
     "    # (202620000, 202620383),        # H2 ATTEMPT 2's block, DRAWN BY AN",
     f"{T_H2}::test_THE_RETRY_BLOCK_IS_EXPOSED_383_AND_RETIRED_WHOLE"),
    # 🔴 THE FRESH BLOCK MUST NOT BE A SPENT ONE -- the whole reason for a third.
    # RE-ANCHORED 2026-09-12: `H2_SEED_BLOCK` is attempt 3's now, and attempt 2 is
    # the most recently spent block, so it is the sharpest thing to try to reuse.
    ("attempt 3 reuses attempt 2's SPENT block", H2R,
     "H2_SEED_BLOCK = (202622000, 202622000 + N_GAMES)        # [202622000, 202622736)",
     "H2_SEED_BLOCK = (202620000, 202620000 + N_GAMES)        # the SPENT block",
     f"{T_H2}::test_the_card_numbers_are_the_module_numbers"),
    # (removed: byte-identical to "the SPENT block is not retired, so it can be
    #  scheduled again" -- one injection, one control.)
    ("attempt 1's digest is overwritten, so its record stops verifying", H2R,
     'H2_ATTEMPT1_TASK_DIGEST = "3c0a0ae12c61dae69b134b30d0f4adb7dc97b9e46478cebe679784996d7172d2"',
     'H2_ATTEMPT1_TASK_DIGEST = "0" * 64',
     f"{T_H2}::test_ATTEMPT_ONES_SCHEDULE_IS_REFUSED_FOR_EXECUTION_though_it_still_PARSES"),
    ("attempt 3's own digest is not re-pinned", H2R,
     'H2_TASK_DIGEST = "4cec38c75a4297d1b494df1c5754215962b138767d8f4c38fc4996078b94c332"',
     'H2_TASK_DIGEST = H2_ATTEMPT2_TASK_DIGEST',
     f"{T_H2}::test_THE_THIRD_SCHEDULE_STILL_MATCHES_BOTH_PINS_BUT_IS_NOW_SPENT"),
    ("attempt 3's FULL-FIELD digest is not re-pinned", H2R,
     'H2_FULL_TASK_DIGEST = ("2d330ede735b578ea83ab68ebbb22611a7"\n'
     '                       "b1e41c1af64ca9f98c36ee84a3e2da")',
     'H2_FULL_TASK_DIGEST = H2_ATTEMPT2_FULL_TASK_DIGEST',
     f"{T_H2}::test_THE_THIRD_SCHEDULE_STILL_MATCHES_BOTH_PINS_BUT_IS_NOW_SPENT"),
    ("attempt 2's digest is overwritten, so the INCIDENT's record stops verifying", H2R,
     'H2_ATTEMPT2_TASK_DIGEST = "ef68cb9962d1a8d42092ddf329dbdcb27a51f5594567e3f21c14cbe35c435c69"',
     'H2_ATTEMPT2_TASK_DIGEST = "0" * 64',
     f"{T_H2}::test_THE_RETRY_SCHEDULE_STILL_MATCHES_BOTH_PINS_BUT_IS_NOW_SPENT"),

    # 🔴 THE PRE-RUN VERIFICATION FOUND THIS: the outputs pointed at attempt 1's
    # directory, where two of the three already exist. Create-only would have
    # refused the launch AFTER the gate was opened -- an authorization spent on a
    # run that could not start. Nothing bound the destination until now.
    ("the outputs point back into a SPENT attempt's directory", H2CMD,
     'OUT_DIR = "docs/superpowers/evidence/2026-09-12-t1j-h2-match-attempt3"',
     'OUT_DIR = "docs/superpowers/evidence/2026-09-09-t1j-h2-deterministic-readout"',
     f"{T_H2}::test_THE_OUTPUT_DESTINATION_IS_NOT_A_SPENT_ATTEMPTS_DIRECTORY"),
    # ⚠ The first version of this control inserted a COMMENT inside the tuple and
    # emptied nothing, so the test passed and it injured nothing. The tuple itself
    # has to go.
    ("the SPENT output directories are no longer named, so nothing is excluded", H2CMD,
     'SPENT_OUT_DIRS = (\n'
     '    "docs/superpowers/evidence/2026-09-09-t1j-h2-deterministic-readout",   # attempt 1\n'
     '    "docs/superpowers/evidence/2026-09-12-t1j-INCIDENT-control-harness-ran-a-match",\n'
     ')',
     'SPENT_OUT_DIRS = ()',
     f"{T_H2}::test_THE_OUTPUT_DESTINATION_IS_NOT_A_SPENT_ATTEMPTS_DIRECTORY"),

    # ═════════ 2026-09-12: ATTEMPT 3's REGISTRATION -- bookkeeping, not permission
    ("the third block un-registered from ACCOUNTED", REF_SRC,
     "    (202622000, 202622736),          # H2 ATTEMPT 3 -- the match H2 has still not",
     "    # (202622000, 202622736),        # H2 ATTEMPT 3 -- the match H2 has still not",
     f"{T_H2}::test_THE_THIRD_BLOCK_IS_EXPOSED_693_AND_RETIRED_WHOLE"),
    # 🔴 BOTH INVERTED 2026-09-13 BY THE RUN. "ALSO marked EXPOSED / RETIRED before
    # anything is drawn" were the defects while the block was unspent. 693 seeds are
    # now drawn and the block is retired whole, so what must fail is DROPPING either
    # record -- and claiming draws the record does not support.
    ("the third block's exposure is dropped -- 693 draws unrecorded", REF_SRC,
     "    (202622000, 202622693),          # H2 ATTEMPT 3, DRAWN BY THE ONE AUTHORIZED",
     "    # (202622000, 202622693),        # H2 ATTEMPT 3, DRAWN BY THE ONE AUTHORIZED",
     f"{T_H2}::test_THE_THIRD_BLOCK_IS_EXPOSED_693_AND_RETIRED_WHOLE"),
    ("the third block's exposure is widened to the WHOLE block", REF_SRC,
     "    (202622000, 202622693),          # H2 ATTEMPT 3, DRAWN BY THE ONE AUTHORIZED",
     "    (202622000, 202622736),          # H2 ATTEMPT 3, DRAWN BY THE ONE AUTHORIZED",
     f"{T_H2}::test_THE_THIRD_BLOCK_IS_EXPOSED_693_AND_RETIRED_WHOLE"),
    ("the third block's WHOLE-BLOCK retirement is dropped", REF_SRC,
     "    (202622000, 202622736),          # H2 ATTEMPT 3, RETIRED WHOLE 2026-09-13.",
     "    # (202622000, 202622736),        # H2 ATTEMPT 3, RETIRED WHOLE 2026-09-13.",
     f"{T_H2}::test_THE_THIRD_BLOCK_IS_EXPOSED_693_AND_RETIRED_WHOLE"),
    # 🔑 THE ONE THAT MATTERS MOST: registration is bookkeeping, not permission. If
    # the seed edit had also flipped the gate, the barrier test's second half must
    # say so -- otherwise "the gate stays False" is a claim no test can contradict.
    ("the H2 execution gate was left OPEN after the run", H2RUN,
     "H2_EXECUTION_AUTHORIZED = False",
     "H2_EXECUTION_AUTHORIZED = True",
     f"{T_H2}::test_THE_THIRD_BLOCK_IS_EXPOSED_693_AND_RETIRED_WHOLE"),
    # 🔴 INVERTED 2026-09-13. It used to be "the registry refuses a schedule it
    # should admit". The block is SPENT now, so the defect is the opposite: a
    # retirement that does not refuse the next schedule is a comment, not a state.
    ("the spent third schedule is schedulable because availability is never checked",
     REF_SRC,
     "def validate_task_executable(task: Dict[str, Any]) -> None:",
     "def validate_task_executable(task: Dict[str, Any]) -> None:\n    return None",
     f"{T_H2}::test_THE_THIRD_SCHEDULE_STILL_MATCHES_BOTH_PINS_BUT_IS_NOW_SPENT"),
    # the retry's EXACT configuration, through the REAL builder
    ("the seam's readout mode is not the rules' one", H2R,
     'SELECTION_MODE = "argmax"',
     'SELECTION_MODE = "opening_temperature"',
     f"{T_H2}::test_THE_EXACT_RETRY_TASK_CONSTRUCTS_THROUGH_THE_REAL_BUILDER"),
    # (removed: disabling the VALIDATOR cannot change what `build_tasks` PRODUCES,
    #  so it went NOT CAUGHT -- and "the plan assigns seeds by membership, not
    #  POSITION" above already injects that defect where a test can see it.)
    ("our side is built with the WRONG colour for the arm", G3SRC,
     '    expected_colour = "black" if task["anchor_colour"] == "red" else "red"',
     '    expected_colour = colour',
     f"{T_H2}::test_THE_BUILDER_STILL_REFUSES_THE_WRONG_COLOUR_FOR_THE_ARM"),
    # ── 2026-09-14: THE IDENTITY FIELDS THE QUALIFIED CONSTRUCTION READS.
    # Found by the pre-run verification, not by the suite: 597 controls and a
    # green suite said nothing, because every seam test supplied its OWN agent
    # factory and so never subscripted the fields the real one does.
    ("the pilot's tasks carry no reference identity", H3R,
     '                "reference": ref["name"],\n'
     '                "reference_sha1": ref["sha1"],\n',
     "",
     f"{T_H3O}::test_the_EXECUTABLE_PATH_still_admits_an_UNSPENT_block"),
    ("the task names no reference_colour, so the factory cannot route", H3R,
     '            t["reference_colour"] = REF.reference_colour(t)',
     "            pass",
     f"{T_H3RUN}::test_THE_REAL_AGENT_FACTORY_ROUTES_BY_THE_TASKS_OWN_reference_colour"),
    ("the reference colour is the ANCHOR's, so the sides are swapped", H3R,
     '            t["reference_colour"] = REF.reference_colour(t)',
     '            t["reference_colour"] = t["anchor_colour"]',
     f"{T_H3O}::test_every_task_NAMES_the_colour_OUR_SIDE_plays"),
    # ═════════════ 2026-09-15: the H3 FULL STUDY -- design-only, gate-shut ═══
    # 🔑 THE DESIGN'S OWN ARITHMETIC IS CONTROLLED, not just its plumbing. Every
    # number in the card was wrong once before it was right, so each is injected.
    ("the sample size is rounded DOWN and misses its own target", H3PROTO,
     "N_PAIRS = 296",
     "N_PAIRS = 288",
     f"{T_H3SR}::test_THE_PRECISION_TARGET_IS_ACTUALLY_MET"),
    # 🔴 RE-AIMED: it named the per-segment composition test, but the injection
    # breaks `assemble_opening_set`, so the module-scope openings FIXTURE raised in
    # SETUP and the named test NEVER RAN -- the "control that demonstrates nothing"
    # class, which the INDETERMINATE outcome exists to surface. The constants test
    # reads them directly, in its own body, with no fixture in the way.
    # 🔴 RE-ANCHORED TWICE by one change: widening the separation moved the
    # constant this injects into AND renamed the test it targets. The base is now
    # 20_261_200_000, so the overlap control puts it back INSIDE the uniform
    # range, while a separate control puts it merely inside the GAP FLOOR -- two
    # distinct claims, one target, different injections.
    ("exhausting the attempts yields a SHORT set instead of aborting",
     H3PROTO,
     '        else:\n            raise H3ProtocolError(',
     '        else:\n            break\n        if False:\n            raise H3ProtocolError(',
     f"{T_H3SR}::test_exhausting_MAX_ATTEMPTS_ABORTS_rather_than_yielding_fewer"),
    ("the study population no longer excludes the PILOT's openings", H3SR,
     '    out |= {o["digest"] for o in PILOT.generate_openings()}  # the pilot\'s twenty',
     "    pass",
     f"{T_H3SR}::test_the_exclusion_set_actually_contains_the_pilot_AND_H1H2"),
    # ── the ANALYSIS: the identity rule and the verdict rule
    ("the opening is dropped from the pair identity", H3SA,
     '    return (games[0]["opening_digest"],\n'
     '            by["red"]["transcript_digest"], by["black"]["transcript_digest"])',
     '    return ("", by["red"]["transcript_digest"],\n'
     '            by["black"]["transcript_digest"])',
     f"{T_H3SA}::test_two_pairs_from_DIFFERENT_openings_are_NEVER_duplicates"),
    ("the opening is dropped from the GAME identity", H3SA,
     '    return (g["opening_digest"], g["incumbent_colour"], g["transcript_digest"])',
     '    return ("", g["incumbent_colour"], g["transcript_digest"])',
     f"{T_H3SA}::test_GAME_AND_PAIR_IDENTITY_INCLUDE_THE_OPENING"),
    ("a duplicate pair is tolerated instead of being a harness fault", H3SR,
     "MAX_DUPLICATE_PAIRS = 0",
     "MAX_DUPLICATE_PAIRS = 7",
     f"{T_H3SA}::test_a_DUPLICATE_PAIR_is_a_HARNESS_FAULT_at_threshold_zero"),
    ("caps are EXCLUDED from the primary instead of scoring 0.5", H3SA,
     '    if g.get("terminal_reason") == "cap":\n        return 0.5',
     '    if g.get("terminal_reason") == "cap":\n        return 0.0',
     f"{T_H3SA}::test_a_CAP_scores_one_half_FOR_THAT_GAME_and_stays_in"),
    ("a record without its SEED is admitted", H3SA,
     "        s = g.get(\"seed\")\n        if type(s) is not int or isinstance(s, bool):",
     "        s = g.get(\"seed\")\n        if False:",
     f"{T_H3SA}::test_a_record_WITHOUT_A_SEED_is_REFUSED"),
    ("an UNCOMPUTABLE sensitivity is treated as a passed check", H3SA,
     '        if not s.get("computable"):',
     "        if False:",
     f"{T_H3SA}::test_an_UNCOMPUTABLE_sensitivity_suppresses_the_verdict"),
    ("a sensitivity sitting EXACTLY on parity is allowed through", H3SA,
     "        if mean == PARITY:",
     "        if False:",
     f"{T_H3SA}::test_a_sensitivity_estimate_of_EXACTLY_ONE_HALF_suppresses_the_verdict"),
    ("a sensitivity DECISIVE the other way is allowed through", H3SA,
     '        if s.get("decisive") and s.get("favours") not in (None, side):',
     "        if False:",
     f"{T_H3SA}::test_a_sensitivity_DECISIVE_THE_OTHER_WAY_suppresses_the_verdict"),
    ("the interval is described as PROVEN rather than nominal", H3SA,
     '    "NOMINAL UNDER THE DECLARED PAIR-INDEPENDENCE MODEL. Rejection sampling "',
     '    "The pairs are independent by construction. Rejection sampling "',
     f"{T_H3SA}::test_the_interval_is_declared_NOMINAL_under_a_MODEL"),
    # ── the RUNNER and the GENERATOR
    ("the study seam trusts the entry's gate check", H3SRUN,
     "            check_gate()                      # the gate, before anything effectful",
     "            pass",
     f"{T_H3SRUN}::test_THE_SEAM_CHECKS_THE_GATE_ITSELF_not_only_the_entry"),
    ("a seam declaring no config is given one anyway", H3SRUN,
     '    config = getattr(play, "config", None)\n    if config is None:',
     '    config = getattr(play, "config", None) or frozen_argmax_config()\n    if False:',
     f"{T_H3SRUN}::test_a_seam_that_declares_NO_config_is_REFUSED"),
    ("a segment schedule is not compared with its own digest", H3SRUN,
     "    if got != want_digest:",
     "    if False:",
     f"{T_H3SRUN}::test_check_segment_schedule_REFUSES_a_tampered_schedule"),
    ("a segment never asks the registry whether its seeds may RUN", H3SRUN,
     "        REF.validate_schedule_executable(list(tasks))",
     "        pass",
     f"{T_H3SRUN}::test_check_segment_schedule_ALSO_ASKS_THE_REGISTRY"),
    ("segments share one output directory", H3SRUN,
     'return f"{OUT_ROOT}/2026-09-15-t1j-h3-study-segment{segment}"',
     'return f"{OUT_ROOT}/2026-09-15-t1j-h3-study-segment"',
     f"{T_H3SRUN}::test_EVERY_SEGMENT_WRITES_TO_ITS_OWN_DIRECTORY"),
    ("the create-only precheck follows symlinks", H3SRUN,
     "        if os.path.lexists(p):",
     "        if os.path.exists(p):",
     f"{T_H3SRUN}::test_a_DANGLING_SYMLINK_at_an_output_path_is_REFUSED"),
    # 🔴 OPTIONAL STOPPING -- the opening that separate operational authorizations
    # create, and the one no amount of arithmetic would reveal.
    ("a segment withheld after inspecting outcomes still permits a verdict", H3SRUN,
     "    if withheld and inspected:",
     "    if False:",
     f"{T_H3SRUN}::test_A_WITHHELD_SEGMENT_AFTER_INSPECTION_PREVENTS_A_VERDICT"),
    ("a VOIDed segment is treated as a clean exclusion", H3SRUN,
     "    if voided:",
     "    if False:",
     f"{T_H3SRUN}::test_a_VOIDED_segment_FLAGS_the_primary_and_blocks_an_automatic_verdict"),
    # ── 2026-09-16: the CONTEXT RESET (the defect that VOIDed attempt 1) and
    # the spent range / destination it left behind.
    # ── 2026-09-16: the GENERATION DEADLINE, preregistered separately, and the
    # PARENT-OWNED LAUNCH RECEIPT. The worker's own `finally` cannot record an
    # outcome decided after the parent has killed it.
    ("the generation wrapper borrows the match segment's deadline", H3GCMD,
     "                              timeout_s=(RUN.RULES.GENERATION_DEADLINE_S\n"
     "                                         + SUPERVISOR_GRACE_S),",
     "                              timeout_s=(RUN.RULES.SEGMENT_DEADLINE_S\n"
     "                                         + SUPERVISOR_GRACE_S),",
     f"{T_H3OLD}::test_GENERATION_HAS_ITS_OWN_PREREGISTERED_DEADLINE"),
    ("no launch receipt is written at all", H3GCMD,
     "        wrote = _write_receipt(RECEIPT, {",
     "        wrote = True or _write_receipt(RECEIPT, {",
     f"{T_H3OLD}::test_THE_RECEIPT_EXISTS_EVEN_WHEN_THE_WORKER_WROTE_NO_TRACE"),
    ("a refusal before spawning leaves no receipt", H3GCMD,
     "        wrote = _write_receipt(RECEIPT, {",
     "        wrote = (not sup) or _write_receipt(RECEIPT, {",
     f"{T_H3OLD}::test_a_REFUSAL_BEFORE_SPAWNING_still_leaves_a_receipt"),
    ("the launch receipt is not create-only", H3GCMD,
     "        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)",
     "        fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)",
     f"{T_H3OLD}::test_O_EXCL_refuses_the_receipt_EVEN_WITH_THE_PRECHECK_DISABLED"),
    ("the receipt is not part of the destination check", H3GCMD,
     "    for p in (out, trace, RECEIPT):",
     "    for p in (out, trace):",
     f"{T_H3OLD}::test_THE_RECEIPT_IS_CREATE_ONLY_and_a_SECOND_LAUNCH_is_refused"),
    # 🔴 A TEST WROTE INTO THE RUN'S OWN DESTINATION. This control puts that back.
    # ── 2026-09-15: the SUPERVISED GENERATION LAUNCH PATH. Generation had none:
    # the public entry drove the production collaborators directly.
    ("a failed gate restoration does not become the wrapper's exit code", H3GCMD,
     "        restored = restore_gate(_runner_source)\n        if not restored:",
     "        restored = restore_gate(_runner_source)\n        if False:",
     f"{T_H3OLD}::test_THE_FINALLY_PATH_also_restores_the_generation_gate"),
    ("a surviving descendant of the generator is reported as success", H3GCMD,
     '            if not r["group_cleared"]:',
     "            if False:",
     f"{T_H3OLD}::test_every_supervisor_outcome_gets_ITS_OWN_exit_code"),
    ("the generation wrapper spawns before checking the destination", H3GCMD,
     "            _check_destination(out_path, trace_path)",
     "            pass",
     f"{T_H3OLD}::test_the_wrapper_REFUSES_BEFORE_SPAWNING_when_the_destination_exists"),
    ("the generation worker runs without the supervisor's capability", H3GCMD,
     "    if not _consume_capability(a.capability_fd):",
     "    if False:",
     f"{T_H3OLD}::test_THE_WORKER_REFUSES_without_the_supervisors_CAPABILITY"),
    ("the generation destination may sit in a SPENT directory", H3GCMD,
     "            if p.startswith(spent.rstrip(\"/\") + \"/\"):",
     "            if False:",
     f"{T_H3OLD}::test_the_destination_INSIDE_A_SPENT_DIRECTORY_is_refused"),
    # ── the generator's terminal semantics
    # ── 2026-09-15: PREPARATION for the opening-generation run.
    # 🔑 RE-AIMED. This control injected a pin while OPENING_SET_DIGEST was
    # None -- "a pin invented before the artifact exists". The population is now
    # frozen AND pinned, so that defect can no longer be expressed. The one that
    # replaces it is the live version of the same danger: A PIN THAT IS NOT THE
    # ARTIFACT'S OWN DIGEST. The study would then play a population nobody froze.
    ("the opening-set pin is not the frozen artifact's own digest",
     H3SR, '    "35932b3fabd9c6463d615b0b3af380134dadd700e2ca1e882a0c863faf772e46")',
     '    "0000000000000000000000000000000000000000000000000000000000000000")',
     f"{T_H3POP}::test_THE_POPULATION_IS_FROZEN_AND_PINNED"),
    ("the H3 pilot gate is opened", H3RUN,
     "H3_PILOT_EXECUTION_AUTHORIZED = False",
     "H3_PILOT_EXECUTION_AUTHORIZED = True",
     f"{T_H3RUN}::test_THE_GATE_IS_SHUT_IN_THE_REAL_REPOSITORY"),
    # 🔴 RE-AIMED 2026-09-14 by the registration. It injected "a block that was
    # never reserved" while none existed; one is reserved and ACCOUNTED now, so the
    # defect is naming a DIFFERENT block than the registered one -- which would
    # draw from seeds no registry accounts for.
    ("the H3 pilot names a block other than the registered one", H3RUN,
     "PILOT_SEED_BLOCK: Optional[tuple] = (202624000, 202624040)",
     "PILOT_SEED_BLOCK: Optional[tuple] = (777000000, 777000040)",
     f"{T_H3RUN}::test_THE_SEED_BLOCK_IS_RESERVED_AND_REGISTERED_but_the_GATE_IS_STILL_SHUT"),
    ("the NO-BLOCK refusal is removed, so an unnamed block runs", H3RUN,
     '    if PILOT_SEED_BLOCK is None:\n        raise H3PilotRunError(\n'
     '            "NO SEED BLOCK IS RESERVED',
     '    if False:\n        raise H3PilotRunError(\n'
     '            "NO SEED BLOCK IS RESERVED',
     f"{T_H3RUN}::test_the_NO_BLOCK_refusal_is_still_REACHED_ALONE"),
    # ───────── the REGISTRATION itself: bookkeeping, not permission ─────────
    ("the pilot block un-registered from ACCOUNTED", REF_SRC,
     "    (202624000, 202624040),          # H3 PILOT. 40 seeds, one per game: 20",
     "    # (202624000, 202624040),        # H3 PILOT. 40 seeds, one per game: 20",
     f"{T_H3RUN}::test_an_UNREGISTERED_block_is_still_refused"),
    # 🔴 BOTH INVERTED 2026-09-15 BY THE RUN. They said "a reservation is not a
    # draw" -- true until 00:47Z that day. They injected "ALSO marked EXPOSED /
    # RETIRED before anything is drawn" -- which is now simply TRUE and correct, so
    # the injection would add a duplicate line rather than a defect. The live
    # defects are the opposite ones: draws left unrecorded, and a spent one-shot
    # schedule left replayable. H2's blocks took exactly this turn after their runs.
    ("the 40 confirmed draws are not recorded as EXPOSED at all", REF_SRC,
     "    (202624000, 202624040),          # H3 PILOT, DRAWN BY THE ONE AUTHORIZED RUN on\n",
     "",
     f"{T_H3O}::test_THE_BLOCK_IS_EXPOSED_40_AND_RETIRED_WHOLE"),
    ("the PILOT block's whole-block retirement is dropped", REF_SRC,
     "    (202624000, 202624040),          # H3 PILOT, RETIRED WHOLE 2026-09-15. All 40\n",
     "",
     f"{T_H3O}::test_THE_BLOCK_IS_EXPOSED_40_AND_RETIRED_WHOLE"),
    # 🔑 THE ONE THAT MATTERS MOST: registration is bookkeeping, not permission.
    # 🔴 RE-AIMED AND RE-LABELLED 2026-09-15. Registration is long past and the RUN
    # has happened, so the live claim is that the wrapper left the gate shut behind it.
    ("the pilot gate is left OPEN after the run", H3RUN,
     "H3_PILOT_EXECUTION_AUTHORIZED = False",
     "H3_PILOT_EXECUTION_AUTHORIZED = True",
     f"{T_H3O}::test_THE_BLOCK_IS_EXPOSED_40_AND_RETIRED_WHOLE"),
    ("the seeded pin no longer matches the digest the run recorded", H3R,
     'SEEDED_TASK_DIGEST: Optional[str] = (\n'
     '    "aa527cc9a1a7b1e657911171c63f19fc006909dd64518bd96de3ce4ddfabfba9")',
     "SEEDED_TASK_DIGEST: Optional[str] = TASK_DIGEST",
     f"{T_H3O}::test_THE_SEEDED_PIN_IS_WITNESSED_BY_THE_RUNS_OWN_DURABLE_HEADER"),
    ("the registered block overlaps a spent one", H3RUN,
     "PILOT_SEED_BLOCK: Optional[tuple] = (202624000, 202624040)",
     "PILOT_SEED_BLOCK: Optional[tuple] = (202622000, 202622040)",
     f"{T_H3O}::test_the_registered_block_is_DISJOINT_from_every_spent_one"),
    ("the pilot entry no longer reads the gate first", H3RUN,
     "    check_gate()\n    from . import d1_probe as D1",
     "    from . import d1_probe as D1\n    check_gate()",
     f"{T_H3RUN}::test_run_pilot_CALLS_check_gate_FIRST_by_AST"),
    ("the pilot seam trusts the entry's gate check", H3RUN,
     "            check_gate()\n            # 🔑 CALLED DIRECTLY IN THE SEAM",
     "            # 🔑 CALLED DIRECTLY IN THE SEAM",
     f"{T_H3RUN}::test_THE_SEAM_CHECKS_THE_GATE_ITSELF_not_only_the_entry"),
    # 🔴 RE-AIMED BEFORE IT EVER RAN, by `test_NO_CONTROL_DELETES_AN_AUTHORIZATION_
    # CHECK`. It named the test that FORCES THE GATE OPEN, so with the boundary
    # deleted the seam would have run on to `TC.verified_paths()` -- a real
    # production act, and the exact shape of the 2026-09-12 incident. Its target is
    # now the AST test, which detects the boundary's absence WITHOUT EXECUTING THE
    # SEAM. The behavioural half is covered by the forced-gate test itself, which
    # needs no control: it fails if the boundary stops refusing.
    ("the pilot's containment boundary is removed", H3RUN,
     "                SCREEN_CMD.assert_production_acts_are_inert(\n",
     "                _noop(\n",
     f"{T_H3RUN}::test_the_seam_checks_BEFORE_anything_effectful"),
    ("the pilot checks the gate once instead of every game", H3RUN,
     "        check_gate()                      # EVERY game, not only the first",
     "        pass                              # EVERY game, not only the first",
     f"{T_H3RUN}::test_the_gate_is_checked_for_EVERY_GAME_not_only_the_first"),
    ("the pilot's create-only precheck follows symlinks", H3RUN,
     "        if os.path.lexists(p):",
     "        if os.path.exists(p):",
     f"{T_H3RUN}::test_a_DANGLING_SYMLINK_at_an_output_path_is_REFUSED"),
    ("the pilot admits two outputs that are the same file", H3RUN,
     "    if len({os.path.abspath(p) for p in paths}) != 3:",
     "    if False:",
     f"{T_H3RUN}::test_the_output_paths_are_CREATE_ONLY_and_must_be_THREE_files"),
    # ── 2026-09-14: THE INCUMBENT'S IDENTITY -- provenance the record carries.
    # ── 2026-09-14: THE OBJECT, not an equal one. The identity was read off one
    # config and the seam built another later, after the outputs were already
    # open -- same function, same values, two objects.
    ("the identity is read off a FRESH config, not the seam's", H3RUN,
     '    config = getattr(play, "config", None)',
     "    config = frozen_argmax_config()",
     f"{T_H3RUN}::test_a_SEAM_WHOSE_CONFIG_DIFFERS_IS_REFUSED_BEFORE_ANY_PLAY"),
    ("a seam that declares no config is given one anyway", H3RUN,
     '    config = getattr(play, "config", None)\n    if config is None:',
     '    config = getattr(play, "config", None) or frozen_argmax_config()\n'
     "    if False:",
     f"{T_H3RUN}::test_a_seam_that_DECLARES_NO_CONFIG_is_REFUSED"),
    ("the seam builds a config instead of using the one it was given", H3RUN,
     "            argmax_cfg = play.config",
     "            argmax_cfg = frozen_argmax_config()",
     f"{T_H3RUN}::test_THE_RECORDED_IDENTITY_IS_THE_CONFIG_PASSED_TO_THE_REAL_BUILDER"),
    ("the pilot entry builds the config twice", H3RUN,
     "        play=_production_play(results_path, deadline, openings, argmax_cfg),",
     "        play=_production_play(results_path, deadline, openings,\n"
     "                              frozen_argmax_config()),",
     f"{T_H3RUN}::test_run_pilot_CONSTRUCTS_THE_CONFIG_ONCE_and_hands_it_to_the_seam"),
    ("the pilot never checks the recorded identity", H3RUN,
     "    check_incumbent_identity(ident, config)",
     "    pass",
     f"{T_H3RUN}::test_the_identity_is_VERIFIED_BEFORE_ANY_OUTPUT_IS_OPENED"),
    # 🔑 THE RECURRING DEFECT CLASS, EXACTLY: a default that switches the check
    # off. Only the default-path test can see this one -- a supplied identity is
    # still checked, so every other identity test passes.
    ("the identity is checked only when one is SUPPLIED", H3RUN,
     "    check_incumbent_identity(ident, config)",
     "    if identity is not None:\n        check_incumbent_identity(ident, config)",
     f"{T_H3RUN}::test_THE_DEFAULT_IDENTITY_IS_CHECKED_TOO_not_only_a_supplied_one"),
    ("the durable header records no identity", H3RUN,
     '                   "identity": ident})',
     "                   })",
     f"{T_H3RUN}::test_THE_DURABLE_HEADER_CARRIES_THE_FULL_IDENTITY_AND_THE_READOUT"),
    ("the config object is never compared with the qualified path", H3RUN,
     '        H2RUN._same(want["argmax_config"], merged, "argmax_config")',
     "        pass",
     f"{T_H3RUN}::test_the_ARGMAX_CONFIG_CROSS_CHECK_refuses_a_DRIFTED_CONFIG_OBJECT"),
    ("the pilot never asks the registry whether the seeds may RUN", H3RUN,
     "        REF.validate_schedule_executable(list(tasks))",
     "        pass",
     f"{T_H3RUN}::test_check_schedule_ALSO_ASKS_THE_REGISTRY_whether_the_seeds_MAY_RUN"),
    ("a seedless schedule is admitted, so it bypasses the registry", H3RUN,
     "    if seedless:",
     "    if False:",
     f"{T_H3RUN}::test_a_SEEDLESS_schedule_is_REFUSED_because_it_cannot_be_RUN"),
    ("the pilot schedule digest is not compared with the pin", H3RUN,
     "    if got != want:",
     "    if False:",
     f"{T_H3RUN}::test_a_DIFFERENT_schedule_is_REFUSED"),
    # THE REAL BUILDER -- the check attempt 1's VOID made necessary.
    ("the pilot's tasks carry the wrong anchor for the incumbent's colour", H3R,
     '"anchor_colour": "black" if colour == "red" else "red",',
     '"anchor_colour": colour,',
     f"{T_H3RUN}::test_EVERY_PAIRS_BOTH_TASKS_construct_through_the_REAL_builder"),
    # GENERATION.
    ("the pilot's openings are no longer deduplicated", H3R,
     "        if digest in seen or digest in prior:",
     "        if False:",
     f"{T_H3O}::test_the_DEDUPLICATION_branch_is_reached_alone"),
    ("the pilot reuses H1/H2's openings", H3R,
     "        if digest in seen or digest in prior:",
     "        if digest in seen:",
     f"{T_H3O}::test_the_PRIOR_OPENING_exclusion_is_reached_alone"),
    ("terminal positions are admitted as pilot openings", H3R,
     "        if len(moves) != OPENING_PLIES or st.is_terminal():",
     "        if len(moves) != OPENING_PLIES:",
     f"{T_H3O}::test_the_TERMINAL_rejection_is_reached_alone"),
    ("the opening generator ignores its seed, so it is not reproducible", H3R,
     "    rng = np.random.Generator(np.random.PCG64(seed))",
     "    rng = np.random.Generator(np.random.PCG64())",
     f"{T_H3O}::test_GENERATION_IS_DETERMINISTIC"),
    # SEEDS.
    ("the pilot coerces its seed endpoints again", H3R,
     "            if type(v) is not int:",
     "            if not isinstance(v, (int, float, str)):",
     f"{T_H3O}::test_a_TYPE_DIFFERENT_seed_endpoint_is_REFUSED[iv0-a string endpoint]"),
    ("a spent block is schedulable because availability is never checked", H3R,
     '            if st["exposed"] or st["retired"] or st["test_only"]:',
     "            if False:",
     f"{T_H3O}::test_a_RETIRED_or_EXPOSED_interval_is_REFUSED[202622000-H2 attempt 3: exposed AND retired]"),
    # THE ANALYSIS.
    ("a duplicated task is counted twice instead of refused", H3A,
     "        if tid in seen:",
     "        if False:",
     f"{T_H3A}::test_a_DUPLICATED_task_id_is_REFUSED"),
    ("an unexpected pair_id is admitted", H3A,
     "        if not 0 <= pid < R.N_OPENINGS:",
     "        if False:",
     f"{T_H3A}::test_an_UNEXPECTED_pair_id_is_REFUSED"),
    ("duplicate pairs are counted as fresh evidence", H3A,
     "        if key in seen:\n            duplicates += 1\n            continue",
     "        if False:\n            duplicates += 1\n            continue",
     f"{T_H3A}::test_DUPLICATE_PAIRS_collapse_to_one_and_keep_BOTH_their_games"),
    ("a NaN duration reaches the quantiles", H3A,
     "    if not math.isfinite(v):",
     "    if False:",
     f"{T_H3A}::test_a_NON_FINITE_per_game_duration_is_REFUSED[nan]"),
    ("a partial run declares its stop rules CLEAR", H3A,
     '        return {"status": CLEAR if complete else UNDETERMINED,',
     '        return {"status": CLEAR,',
     f"{T_H3A}::test_on_a_partial_run_NOTHING_is_ever_declared_CLEAR"),
    # ───────── the RUN BODY and the WRAPPER (2026-09-14) ─────────
    # 🔑 A DEADLINE IS NOT A VOID for this design: the pilot's outputs are
    # diagnostics and a partial run is informative. H2 raises here; the pilot
    # must not.
    ("the pilot VOIDs on its deadline instead of reporting PARTIAL", H3RUN,
     "                timed_out = True",
     "                raise H3PilotVoidError('deadline')",
     f"{T_H3RUN}::test_A_TIMEOUT_STOPS_CLEANLY_and_still_reports_PARTIAL"),
    ("the pilot keeps playing past its deadline", H3RUN,
     "            if deadline.elapsed() > deadline_s:",
     "            if False:",
     f"{T_H3RUN}::test_A_TIMEOUT_STOPS_CLEANLY_and_still_reports_PARTIAL"),
    ("an exception is reported as a partial run rather than a VOID", H3RUN,
     '        verdict = "INTERRUPTED" if isinstance(e, KeyboardInterrupt) else "VOID"',
     '        verdict = "PARTIAL"',
     f"{T_H3RUN}::test_AN_EXCEPTION_IS_A_VOID_and_the_trace_says_so"),
    ("the durable transcript evidence is never written", H3RUN,
     '            emit(rec, {"record_type": "transcript", "task_id": task["task_id"],',
     '            _skip = ({"record_type": "transcript", "task_id": task["task_id"],',
     f"{T_H3RUN}::test_the_results_file_carries_the_transcripts_and_the_durations"),
    ("the per-game duration is dropped from the record", H3RUN,
     '                    "elapsed_s": out["elapsed_s"]}',
     '                    "plies_again": row.get("plies")}',
     f"{T_H3RUN}::test_the_results_file_carries_the_transcripts_and_the_durations"),
    ("the outputs are no longer create-only inside the run", H3RUN,
     "            os.open(results_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), \"w\"))",
     "            os.open(results_path, os.O_WRONLY | os.O_CREAT, 0o644), \"w\"))",
     f"{T_H3RUN}::test_O_EXCL_refuses_even_with_the_precheck_disabled"),
    ("the generated openings are not checked against the frozen pin", H3RUN,
     "    if RULES.opening_set_digest(openings) != RULES.OPENING_SET_DIGEST:",
     "    if False:",
     f"{T_H3RUN}::test_run_pilot_REFUSES_a_set_that_does_not_match_the_pin"),
    # THE WRAPPER.
    ("the PILOT wrapper reports success without verifying the gate", H3CMD,
     "        if not restore_gate(_runner_source):\n"
     "            print(f\"GATE NOT RESTORED:",
     "        if False:\n            print(f\"GATE NOT RESTORED:",
     f"{T_H3RUN}::test_THE_FINALLY_PATH_also_restores_and_reports_its_own_failure"),
    ("a surviving descendant is reported as a success", H3CMD,
     '            if not r["group_cleared"]:',
     "            if False:",
     f"{T_H3RUN}::test_a_SURVIVING_DESCENDANT_is_a_CLEANUP_FAILURE_not_a_success"),
    ("the worker runs without the supervisor's capability", H3CMD,
     "    if not _consume_capability(a.capability_fd):",
     "    if False:",
     f"{T_H3RUN}::test_WORKER_REFUSES_without_the_supervisors_CAPABILITY"),
    ("a forged capability of the right length is accepted", H3CMD,
     '    return len(token) == CAPABILITY_BYTES * 2 and all(\n'
     '        c in "0123456789abcdef" for c in token)',
     "    return len(token) == CAPABILITY_BYTES * 2",
     f"{T_H3RUN}::test_a_FORGED_capability_is_refused_INCLUDING_one_of_the_RIGHT_LENGTH"),
    ("the wrapper spawns before checking the outputs", H3CMD,
     "            RUN.check_output_paths(a.results, a.trace, a.report)",
     "            pass",
     f"{T_H3RUN}::test_the_wrapper_REFUSES_BEFORE_SPAWNING_when_an_output_exists"),
    # 🔑 A CROSSED MONOTONE COUNT IS CONCLUSIVE; A PARTIAL RUN IS NOT. The order
    # of these two branches is the whole claim.
    ("a fired stop rule is masked by the partial-run code", H3CMD,
     '    if report.get("any_fired"):\n        return EXIT_STOP_RULE_FIRED\n'
     '    if report.get("timed_out") or not report.get("complete"):\n'
     "        return EXIT_PARTIAL",
     '    if report.get("timed_out") or not report.get("complete"):\n'
     "        return EXIT_PARTIAL\n"
     '    if report.get("any_fired"):\n        return EXIT_STOP_RULE_FIRED',
     f"{T_H3RUN}::test_the_classifier_separates_RESULTS_from_failures[report3-14]"),
    ("the pilot's outputs point into a SPENT run's directory", H3CMD,
     'OUT_DIR = "docs/superpowers/evidence/2026-09-14-t1j-h3-pilot"',
     'OUT_DIR = "docs/superpowers/evidence/2026-09-12-t1j-h2-match-attempt3"',
     f"{T_H3RUN}::test_THE_OUTPUT_DESTINATION_IS_NOT_A_SPENT_RUNS_DIRECTORY"),
    # ───────── the INTERFACE the fixtures replaced (2026-09-14) ─────────
    # 🔴 THE TWO DEFECTS THAT WOULD HAVE FAILED ON GAME ONE and that no fixture
    # could see, because every fixture replaced `play_task` itself.
    ("the harness is handed no recorder", H3RUN,
     "            rec=cap, ply_cap=H2R.PLY_CAP)",
     "            rec=None, ply_cap=H2R.PLY_CAP)",
     f"{T_H3RUN}::test_THE_RECORDER_IS_REAL_and_receives_the_openings_bound_first"),
    ("the seam returns the harness's flat result instead of the contract", H3RUN,
     '    return {"result": {"task_id": task["task_id"], "seed": task.get("seed"),\n'
     "                       **result},",
     '    return {"flat": dict(result),',
     f"{T_H3RUN}::test_THE_SEAM_DRIVES_THE_REAL_HARNESS_AND_RETURNS_THE_RUN_BODYS_CONTRACT"),
    ("two opening_bound records are accepted, so the ply span is unanchored", H3RUN,
     "    if len(bounds) != 1:",
     "    if False:",
     f"{T_H3RUN}::test_MORE_THAN_ONE_opening_bound_is_a_VOID"),
    # THE SEEDED PIN.
    ("a seeded schedule is checked against the UNSEEDED pin", H3R,
     "    if SEEDED_TASK_DIGEST is None:",
     "    if False:",
     f"{T_H3O}::test_a_SEEDED_schedule_is_REFUSED_while_the_seeded_pin_is_unset"),
    ("a half-seeded schedule is admitted", H3R,
     "    if len(seeded) != len(tasks):",
     "    if False:",
     f"{T_H3O}::test_a_HALF_SEEDED_schedule_matches_NEITHER_pin"),
    ("the runner ignores which pin applies", H3RUN,
     "        want = RULES.expected_task_digest(tasks)",
     "        want = RULES.TASK_DIGEST",
     f"{T_H3RUN}::test_check_schedule_REFUSES_a_seeded_schedule_while_the_pin_is_unset"),
]

# ═══════════════════════════ THE EXPECTED REASONS ════════════════════════════
# 🔴 `rejected = r.returncode != 0` counted a collection error, a crash, an
# interruption and "no tests ran" as successful controls. A control is REJECTED
# only when the named node FAILS -- not errors -- and its output carries the text
# below. Anything else is INDETERMINATE, reported as such, and exits nonzero.
#
# SCOPE OF THESE STRINGS. They were RECORDED from an observed run and frozen, not
# authored from the tests. They therefore detect DRIFT -- a control that starts
# failing at a different assertion than the one it was shown to reach -- and they
# do not independently prove that the assertion reached is the right one.
# A label with no entry here is INDETERMINATE; there is no permissive default.
#
# 🔴 SIX OF THESE WERE REPAIRED ON 2026-09-12 AND THEIR REASONS ARE NOT HARVESTED.
# Under their defect the named test NEVER RAN: five raised during FIXTURE SETUP and
# one left a SyntaxError so nothing collected. All six exited nonzero, which is why
# `rejected = r.returncode != 0` scored all six as successful controls in the run
# packaged at 144a141 -- "539/539 rejected" overstated by six. Each kept its claim;
# what changed is the mutation, the target, or both, so that setup and collection
# complete and the named test reaches the behaviour. Their reasons below were
# DERIVED from the requirement and its assertion and then verified against a run,
# which is why each names a count or a sentence rather than a first line.
EXPECTED_REASONS = {
    'the combination gate is not checked at all':
        'scripts.GPU.alphazero.h3_combine.H3CombineError: docs/superpowers/evidence/2026-09-20-t1j-h3-study-combined-attempt2/09_combined_report.json already exists. The combined report is written ONCE; a second combination needs a new reviewed destination.',
    # ── the combination closeout (2026-09-20). OBSERVED.
    'the combined outputs are never re-hashed against their pins':
        'AssertionError: []',
    'a pinned combined output may be MISSING':
        "FileNotFoundError: [Errno 2] No such file or directory: '/private/<tmp>/pytest-of-bill/pytest-<n>/test_THE_COMBINED_OUTPUTS_ARE_0/gone.json'",
    "the combined report's digest pin is blanked":
        'AssertionError: the combined output must still be byte-identical to what was written',
    'the verdict is persisted without its population claim':
        "KeyError: 'claim'",
    'the combination receipt is not create-only':
        'AssertionError: a second run must not overwrite the first receipt',
    # ── the allow-list repair (2026-09-20). OBSERVED, never guessed.
    'the allow-list permits every open gate instead of the named ones':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_combine.H3CombineError'>",
    'the final-state check stops rejecting unexpected open gates':
        'AssertionError: an unexpected open gate must still be reported',
    'the combiner allows ANY gate to be open, not just its own':
        'scripts.GPU.alphazero.h3_combine.H3CombineError: the final state reports 1 disagreement(s); the segments may not be pooled until they describe one study:',
    'the restoration regex goes back to the permissive form':
        'AssertionError: the restored file must differ from the original in NOTHING',
    "attempt 2 is aimed at attempt 1's SPENT destination":
        "AssertionError: assert 'docs/superpowers/evidence/2026-09-20-t1j-h3-study-combined' != 'docs/superpowers/evidence/2026-09-20-t1j-h3-study-combined'",
    'the gate inventory stops seeing gates at all':
        'AssertionError: the derived inventory and EXPECTED_GATES must agree',
    'the combination stops hashing its inputs':
        'AssertionError: every input must be recorded with the hash of what was read',
    'a PID is left in the temp-file name the reason is compared against':
        'AssertionError: a PID in a temp-file name must not change the reason',
    # ── PID-free re-harvest (2026-09-20). OBSERVED.
    'the authorized entry point accepts a destination override':
        "AssertionError: assert ['out_dir'] == []",
    'the combination ignores the final-state disagreements':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_combine.H3CombineError'>",
    'the combination stops counting the games':
        'scripts.GPU.alphazero.h3_combine.H3CombineError: the 591 records carry 591 distinct seeds, which are not exactly the union of the four blocks',
    'the combination stops requiring DISTINCT task ids':
        'scripts.GPU.alphazero.h3_combine.H3CombineError: 1 pair(s) do not hold exactly two games (first: [0]); a half pair is not a pair',
    'the combination stops requiring the seeds to be the four blocks':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_combine.H3CombineError'>",
    'the combination accepts a pair that is not a pair':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_combine.H3CombineError'>",
    'the combination ignores the permission decision':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_combine.H3CombineError'>",
    'the combination accepts a narrower analysis':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_combine.H3CombineError'>",
    'the early occupied-destination check is removed':
        "FileExistsError: [Errno 17] File exists: '/private/<tmp>/pytest-of-bill/pytest-<n>/test_THE_DESTINATION_IS_CREATE0/combined/.09_combined_report.json.<pid>.tmp' -> '/private/<tmp>/pytest-of-bill/pytest-<n>/test_THE_DESTINATION_IS_CREATE0/combined/09_combined_report.json'",
    'the report is serialised straight into the official path':
        "FileExistsError: [Errno 17] File exists: '/private/<tmp>/pytest-of-bill/pytest-<n>/test_THE_OFFICIAL_PATH_IS_NEVE0/combined/09_combined_report.json' -> '/private/<tmp>/pytest-of-bill/pytest-<n>/test_THE_OFFICIAL_PATH_IS_NEVE0/combined/09_combined_report.json'",
    'the temporary file is left beside the report':
        'AssertionError: a temporary file was left beside the report',
    'the install OVERWRITES instead of refusing an existing name':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_combine.H3CombineError'>",
    'the payload is not fsynced before it is installed':
        'AssertionError: the payload file was never fsynced',
    'the directory is not fsynced after the link':
        'AssertionError: the destination directory was never fsynced',
    'the combination command never restores its gate':
        'assert 9 == 8',
    'a failed restoration no longer supersedes the run outcome':
        'assert 0 == 9',
    'the receipt records the run code instead of the superseding one':
        "AssertionError: the receipt must record the superseding code, not the run's own",
    'an unauthorized combination writes a receipt and occupies the destination':
        'AssertionError: assert not True',
    'the combination command accepts a destination override':
        "AssertionError: ['--help', '--out', '--run', '-h']",
    'summarise is called once PER SEGMENT instead of once on the pool':
        'AssertionError: [148, 148, 148, 148, 592]',
    # ── durability + the supervised command (2026-09-20). OBSERVED.
    # ── the combination (2026-09-20). OBSERVED, never guessed.
    # ── segment 3 closeout, RE-ARMED controls (2026-09-20). OBSERVED.
    'the epilogue is hardcoded instead of branching on readiness':
        "AssertionError: assert 'CANNOT RUN' not in '⚠ SCOPE. Th...ANNOT RUN: .'",
    "every segment's block is pointed at a SPENT one":
        "AssertionError: each segment's block must be its own -- two segments sharing seeds would play the same games twice and count them as independent",
    "every segment writes into segment 0's occupied destination":
        "AssertionError: each segment's destination must be its own -- collapsing them would let one run overwrite another's record",
    'an UNAUTHORIZED invocation writes a receipt and occupies the destination':
        'AssertionError: an unauthorized call must not write a receipt',
    'the SYNTHETIC block is registered as RETIRED as well as accounted':
        'RuntimeError: the synthetic block must be UNSPENT or it isolates nothing',
    # ── the final-state cross-check (2026-09-20). OBSERVED, never guessed.
    'the final-state check stops comparing the receipt fields':
        'AssertionError: a receipt that did not COMPLETE must be reported',
    'the final-state check reads seeds from the PLAN, not the records':
        'AssertionError: a seed outside the block must be reported -- exposure is read off the records',
    'the final-state check stops requiring the artifacts to exist':
        "FileNotFoundError: [Errno 2] No such file or directory: '/private/<tmp>/pytest-of-bill/pytest-<n>/test_A_MISSING_ARTIFACT_IS_CAU0/segment2/04_trace.jsonl'",
    'the final-state check ignores a fired degeneracy gate':
        'AssertionError: a FIRED degeneracy gate must be reported',
    'the final-state check stops looking at the gate inventory':
        'AssertionError: an open authorization gate must be reported',
    'the final-state check stops requiring a spent block to be RETIRED':
        'AssertionError: a block that ran but is not RETIRED must be reported',
    # ── segment 1 closeout, REPAIRED controls (2026-09-19). OBSERVED.
    'readiness stops reading the registry and calls nothing spent':
        "AssertionError: readiness must report the registry's spent segments",
    # ── segment 1 closeout (2026-09-19). OBSERVED, never guessed.
    'the spent/unspent partition stops reading the registry':
        "AssertionError: the registry's spent set is not the one recorded here",
    # ── the harness's own normaliser (2026-09-19). OBSERVED, never guessed.
    'a MEASURED duration is left in the reason it is compared against':
        'AssertionError: load must not change the reason',
    # ── the stale prose (2026-09-19). OBSERVED, never guessed.
    "the runner's header names a barrier constant that is gone":
        "AssertionError: the header names constants the module does not have: ['STUDY_SEED_BLOCK']",
    '--help asserts the population was never generated':
        "AssertionError: --help still asserts 'has no seed block', which this tree contradicts",
    # ── the preflight epilogue (2026-09-19). OBSERVED, never guessed.
    'readiness asserts the pin instead of reading it':
        'assert True is False',
    # ── segment 0 closeout (2026-09-18/19). OBSERVED, never guessed.
    'a SPENT segment may be relaunched':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_runner.H3StudyRunError'>",
    'the spent-seed check ignores EXPOSED and looks only at retired':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_runner.H3StudyRunError'>",
    'a partial segment is reported as a strength verdict':
        'AssertionError: 74 pairs must be below the 148-pair floor',

    'the SEGMENT command writes no launch receipt at all':
        'AssertionError: no receipt on a pre-spawn refusal',
    'a SEGMENT refusal before spawning leaves no receipt':
        "AssertionError: assert 'UNEXPECTED' == 'REFUSED_BEFORE_SPAWN'",
    # ── the segment-isolation repair (2026-09-18). OBSERVED, never guessed.
    'the runner stops creating its output directory':
        'AssertionError: the parent directory was not created',
    'the directory is created AFTER the create-only open':
        "AssertionError: assert 'ensure_parent_dirs(trace_path, results_path, report_path)' in 'def _run_segment_unguarded(*, segment: int, tasks, openings, results_path,\\n                           trace_path, re...T directory\\n#: while every test still passed, because nothing compared the two.\\nOPENING_SET_PATH = GEN.DEFAULT_OUT\\n'",
    'the receipt is not create-only':
        'AssertionError: a second write must refuse',
    'a retired earlier segment blocks every later one again':
        "scripts.GPU.alphazero.h3_study_runner.H3StudyRunError: segment 0's block [909090000, 909090148) is not registered: 148 of 148 seeds are absent from ACCOUNTED_SEED_INTERVALS (first 909090000).",
    'the RETIRED block may be relaunched':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_runner.H3StudyRunError'>",
    'the launch check stops asking whether the seeds are spent':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_runner.H3StudyRunError'>",
    'the launch check is not run before a segment starts':
        "AssertionError: assert 'check_segment_seeds(segment)' in 'def run_segment(*, segment: int, results_path: str, trace_path: str,\\n                report_path: str) -> Dict[str, ...cfg),\\n        want_digest=SEGMENT_DIGESTS[segment],\\n        deadline_s=RULES.SEGMENT_DEADLINE_S, _deadline=deadline)'",
    'build_tasks consults seed status again':
        'scripts.GPU.alphazero.h3_study_rules.H3StudyError: spent',
    'two segments may share one seed block':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_rules.H3StudyError'>",
    'a segment block may be the wrong size':
        'scripts.GPU.alphazero.h3_study_rules.H3StudyError: 454 seeds for 592 games',
    'the number of blocks need not match the segments':
        'scripts.GPU.alphazero.h3_study_rules.H3StudyError: 444 seeds for 592 games',
    "segment 1-3's pins are changed by segment 0's replacement":
        "AssertionError: assert ('cde6d04ce52...8d0a4b8753f0') == ('14aaefb3a01...8d0a4b8753f0')",
    'segment 0 reuses its RETIRED block':
        'AssertionError: segment 0 is the fresh block',
    "segment 0's retry writes into its SPENT destination":
        'AssertionError: assert False',

    # ── the 592-seed match block (2026-09-18). OBSERVED, never guessed.
    'the block is registered as EXPOSED as well as accounted':
        'assert not True',
    'the full-schedule pin is not checked at all':
        "AssertionError: assert 'check_schedule_digest(tasks)' in 'def run_segment(*, segment: int, results_path: str, trace_path: str,\\n                report_path: str) -> Dict[str, ...cfg),\\n        want_digest=SEGMENT_DIGESTS[segment],\\n        deadline_s=RULES.SEGMENT_DEADLINE_S, _deadline=deadline)'",
    'the segment pin is computed from the tasks it checks':
        "AssertionError: assert 'want_digest=SEGMENT_DIGESTS[segment]' in 'def run_segment(*, segment: int, results_path: str, trace_path: str,\\n                report_path: str) -> Dict[str, ...n        want_digest=segment_digest(tasks, segment),\\n        deadline_s=RULES.SEGMENT_DEADLINE_S, _deadline=deadline)'",

    "the opening-set pin is not the frozen artifact's own digest":
        "AssertionError: assert '000000000000...0000000000000' == '35932b3fabd9...c863faf772e46'",
    'the spent-range guard is no longer injected into the loop':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_rules.H3StudyError'>",
    'the rules re-exports drift from the pinned protocol':
        'AssertionError: MAX_ATTEMPTS',
    'the seed allocation leaves the pinned protocol':
        'AssertionError: attempt_seed computes a seed itself; the allocation must be pinned',
    'an output constant moves into the pinned protocol':
        "AssertionError: h3_generation_protocol.py holds output constant(s) {'OPENING_SET_DIGEST'}",
    'python becomes enforced despite not determining the walk':
        "scripts.GPU.alphazero.h3_study_generator.H3GenerationError: the generator identity disagrees with the running code on ['python']. The same seed gives a DIFFERENT opening under a different walk, bit generator or engine, so this population cannot be reproduced here and may not be played.",
    'an identity field is left unclassified':
        "AssertionError: unclassified: {'python'}",
    'numpy stops being enforced':
        "AssertionError: assert 'numpy' in ('kind', 'engine_free', 'gen_seed_base', 'seed_range', 'max_attempts', 'opening_plies', ...)",
    'the exclusion set is no longer pinned by value':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_generator.H3GenerationError'>",
    "the walk's own module is dropped from the pins":
        "AssertionError: assert ({'d1_selectio...ixt_state.py'} == {'d1_selectio..._protocol.py'}",
    'the pinned sources include the module holding OPENING_SET_DIGEST':
        "scripts.GPU.alphazero.h3_study_generator.H3GenerationError: the generator identity disagrees with the running code on ['source_pins']. The same seed gives a DIFFERENT opening under a different walk, bit generator or engine, so this population cannot be reproduced here and may not be played.",
    'restoration is trusted instead of verified from the file':
        'AssertionError: assert 0 == 9',
    'the walk need not reproduce the recorded moves':
        'scripts.GPU.alphazero.h3_study_generator.H3GenerationError: opening 4: the re-derived digest is b2595f7c7ce921d3dcdbb3d20e7a5264ceac57ba1bbd246a7cce03b5777a8e2b, not 733ea8a5827888dfe53b52c44810ca0fdafcf8bc4254405a921a89b844a78c44',
    'an opening may carry unknown keys':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_generator.H3GenerationError'>",
    # ── the three P1 repairs (2026-09-17 review). Reasons OBSERVED by the
    # ── harness, never guessed.
    'freeze_population takes a caller-supplied destination again':
        "AssertionError: <Signature (out_path='docs/superpowers/evidence/2026-09-17-t1j-h3-study-openings-uniform/01_opening_set.json', trace_path='docs/superpowers/evidence/2026-09-17-t1j-h3-study-openings-uniform/02_generation_trace.jsonl') -> 'Dict[str, Any]'>",
    'the freeze command does not restore the barrier':
        'AssertionError: None',
    'a failed restoration no longer supersedes the freeze':
        'AssertionError: assert 0 == 9',
    'the freeze command runs without --run':
        'assert 5 == 2',
    'the freeze command ignores the barrier':
        'AssertionError: assert 8 == 5',
    'restore_barrier reports success on an unreadable source':
        'AssertionError: assert True is False',
    'the deadline is not checked inside the candidate loop':
        'scripts.GPU.alphazero.h3_study_generator.H3GenerationDeadline: the 0s runaway guard expired after 0.7s at opening 296, attempt 0 (296 accepted) of an ENGINE-FREE generation. This is a DEFECT REPORT, not a capacity result: the whole population builds in under a second.',
    'build_population swallows the deadline hook':
        'assert 0 == 296',
    'the runaway guard is reported as an ordinary VOID':
        'AssertionError: a timeout must not be reported as VOID',
    'the clock goes back to wall time':
        "AssertionError: assert 'time.time' not in {'fh.fileno', 'fh.flush', 'json.dump', 'json.dumps', 'o.get', 'os.fdopen', ...}",
    'the artifact is not fsynced before OK is recorded':
        'ValueError: substring not found',
    'write_artifact accepts both a builder and a set':
        "TypeError: 'NoneType' object is not callable",
    "the validator trusts each row's declared digest":
        'scripts.GPU.alphazero.h3_study_generator.H3GenerationError: opening 5: re-running the PRNG walk from seed 20261602000 does NOT produce the recorded moves. The population cannot be re-derived from its own provenance.',
    'the moves are never replayed through the engine':
        'scripts.GPU.alphazero.h3_study_generator.H3GenerationError: opening 0: replaying its moves gives canonical digest c1ea6b8891e06135c7905d64d8f56d761355c85971706f392ca86e35e4e3c5ee but the record claims fbb90a54ca869ae357cef316cb535256baae665c6d8b4c6cdc9471a7859a027f. The moves have been changed and the digest left behind -- which is invisible to any check that only re-hashes the declared digests.',
    'a coordinate may be a bool or a float':
        'scripts.GPU.alphazero.h3_study_generator.H3GenerationError: opening 4: its moves are NOT LEGAL on a real board (ValueError: Illegal move (True, 1) for active_size=24, to_move=black)',
    'the declared index need not be the study order':
        'scripts.GPU.alphazero.h3_study_generator.H3GenerationError: opening 0 records seed 20261600400 but attempt_seed(20261600000, 0, 0) is 20261600000; the seed and the attempt count disagree',
    'the segment need not match the frozen plan':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_generator.H3GenerationError'>",
    'the seed need not match its own attempt count':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_generator.H3GenerationError'>",
    'provenance binding is off by default':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_generator.H3GenerationError'>",
    'the artifact schema admits unknown keys':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_generator.H3GenerationError'>",
    'the generator identity is not compared with the running code':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_generator.H3GenerationError'>",
    "the walk's source pins are dropped from the identity":
        "AssertionError: assert 'source_pins' in ('kind', 'engine_free', 'gen_seed_base', 'seed_range', 'max_attempts', 'opening_plies', ...)",
    'the loader does not validate the artifact at all':
        'assert \'GEN.validate_artifact(doc)\' in \'def load_opening_set(path: str) -> List[Dict[str, Any]]:\\n    """The FROZEN population, VALIDATED AND BOUND, from the...tudy is pinned to {pinned}. A different population is a different "\\n            f"study.")\\n    return openings\\n\\n\\n\'',
    'the loader accepts any self-consistent population':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_runner.H3StudyRunError'>",

    # ── AMENDMENT 3: the uniform-only population. Reasons OBSERVED by the
    # ── harness on 2026-09-17, never guessed.
    'the 148-opening uniform range becomes reusable':
        'assert False is True',
    "attempt 1's spent co-produced range becomes reusable":
        'assert False is True',
    "attempt 2's spent co-produced range becomes reusable":
        'assert False is True',
    'the spent-range check is disabled entirely':
        'assert False is True',
    'the generation range covers openings, not candidates':
        'assert (20261600296 - 20261600000) == (296 * 400)',
    'the generation base slides back onto a retired range':
        'assert (20261000000, 20261118400) == (20261600000, 20261718400)',
    'the population reverts to 148 openings':
        'assert 148 == 296',
    'the assembled set need not be the full population':
        'scripts.GPU.alphazero.h3_study_rules.H3StudyError: segment 2 assembled 0 pairs',
    'the artifact may record a short population':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_generator.H3GenerationError'>",
    'the artifact may name the co-produced stratum':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_generator.H3GenerationError'>",
    'an opening may carry a stale alternating order':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_generator.H3GenerationError'>",
    'check_opening_set stops refusing a stale order field':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_rules.H3StudyError'>",
    'the analysis scores records from another population':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_analysis.H3StudyAnalysisError'>",
    'the analysis accepts a record carrying an order':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_analysis.H3StudyAnalysisError'>",
    'the narrowed claim is softened in the artifact':
        "AssertionError: assert ('NOTHING' in 'Over legal six-ply TwixT positions drawn UNIFORMLY AT RANDOM and filtered only by the structural admissibility rules ...ly reach, or about any engine-produced population. A uniformly random six-ply position is not a position anyone plays.')",
    'the artifact stops requiring its claim':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_generator.H3GenerationError'>",
    'a report may omit the narrowed claim':
        "scripts.GPU.alphazero.h3_study_analysis.H3StudyAnalysisError: the report's claim has been altered. It must reproduce the narrowed uniform-position claim VERBATIM -- softening it is how a result about uniformly random positions becomes a result about play.",
    'a report may alter the narrowed claim':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_analysis.H3StudyAnalysisError'>",
    'the disclaimer that nobody plays these positions is dropped':
        "AssertionError: assert 'not a position anyone plays' in 'Over legal six-ply TwixT positions drawn UNIFORMLY AT RANDOM and filtered only by the structural admissibility rules ...ly reach, or about any engine-produced population. A uniformly random six-ply position is a reasonable proxy for play.'",
    'a live module imports the retired engine generator':
        "AssertionError: h3_study_runner imports {'h3_coproduced_generator_retired'}",
    'the analysis imports the retired JVM supervisor':
        "AssertionError: h3_study_analysis imports {'h3_generation_command'}",
    'the population freeze barrier is opened':
        'assert True is False',
    'the freeze barrier stops being read first':
        'AssertionError: assert (False)',
    'the barrier accepts any truthy value':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_generator.H3GenerationError'>",
    'the artifact may be overwritten':
        "FileExistsError: [Errno 17] File exists: '/<tmp>/tmp<x>/a.json'",
    'attempt provenance becomes optional':
        'scripts.GPU.alphazero.h3_study_rules.H3StudyError: attempt must be a non-negative int, got -1',
    'the official destination moves into a spent directory':
        'AssertionError: assert (False)',
    'the runner repeats the destination instead of reading it':
        "AssertionError: assert 'docs/superpo...ning_set.json' == 'docs/superpo...ning_set.json'",
    'the inventory scans only the first statement of each module':
        "AssertionError: assert 'LOWPLY_QUALIFICATION_AUTHORIZED' in set()",

    'registration check disabled':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'registration check looks only at the first seed':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'the D1 seed block un-registered from ACCOUNTED':
        "AssertionError: (202614000, {'accounted': False, 'exposed': False, 'retired': True, 'test_only': False})",
    'the D1 seed block marked EXPOSED as well as accounted':
        "AssertionError: (202614000, {'accounted': True, 'exposed': True, 'retired': True, 'test_only': False})",
    'the §14 block un-registered from ACCOUNTED (the edit reverted)':
        'scripts.GPU.alphazero.d1_probe.D1Error: the D1 diagnostic seed block [202615000, 202615221) is not registered: 221 of 221 seeds are absent from ACCOUNTED_SEED_INTERVALS (first 202615000). Registering it is a reviewed edit to that registry, part of the D1 EXECUTION authorization; nothing here writes a registry at runtime.',
    'the §14 block un-registered -- caught at the STATE assertion too':
        "AssertionError: (202615000, {'accounted': False, 'exposed': True, 'retired': True, 'test_only': False})",
    'the spent §14 block is NOT exposed -- 221 draws unrecorded':
        "AssertionError: (202615000, {'accounted': True, 'exposed': False, 'retired': True, 'test_only': False})",
    'the spent §14 block is NOT retired -- a one-shot schedule left replayable':
        "AssertionError: (202615000, {'accounted': True, 'exposed': True, 'retired': False, 'test_only': False})",
    'a spent seed is schedulable because availability is never checked':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.e4_screen_reference.E4ReferenceError'>",
    'registration ALSO opened the D1 execution gate':
        'AssertionError: registration must not open the gate; only an execution authorization does',
    'compile step skips toolchain verification':
        "FileNotFoundError: [Errno 2] No such file or directory: 'j'",
    'compile step accepts a jar that is not the verified one':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'compile step accepts a java outside the verified JDK':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'compile step reuses an existing class directory':
        "FileExistsError: [Errno 17] File exists: '/private/var/folders/vm/g32c3nts67bfrpr06cmdzz4h0000gn/T/pytest-of-bill/pytest-<n>/test_an_existing_class_directo0/classes'",
    'compile step ignores a failing javac':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1VoidError'>",
    'compile step builds the E3b set, not the query path':
        "AssertionError: assert (PosixPath('/...3bDump.java')) == (PosixPath('/...flight.java'))",
    'compile step skips the E4 jar pin':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'postcondition surface never read':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1VoidError'>",
    'reflection COUNT accepted as whatever came back':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1VoidError'>",
    'searched position never re-bound':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1VoidError'>",
    'more than one searched dump accepted':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1VoidError'>",
    'a move illegal in OUR engine accepted':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1VoidError'>",
    'depth agreement not recorded':
        "AssertionError: [{'completed': True, 'completed_depth': 3, 'current_max_ply': 3, 'depth': 3, ...}, {'completed': True, 'completed_depth': 6, 'current_max_ply': 6, 'depth': 6, ...}]",
    "the helper's stdout discarded again on a non-zero query exit":
        'AssertionError: l0match-000-strong6-o1_center-t1j_red-r0@ply6 [mover_fragmentation/position] digest=aaaaaaaaaaaaaaaa: depth 6 invocation 0: exit 3 with 1 query records.',
    'the failure excerpt unbounded':
        'AssertionError: assert False',
    'the excerpt line cap removed':
        'AssertionError: assert 40 == 100000',
    'the excerpt falls back to silence':
        'AssertionError: (the helper produced no readable output)',
    'the label drops the cohort':
        "AssertionError: assert ('ply6' in 'l0match-000-strong6-o1_center-t1j_red-r0@ply6 digest=aaaaaaaaaaaaaaaa' and 'mover_fragmentation/position' in 'l0match-000-strong6-o1_center-t1j_red-r0@ply6 digest=aaaaaaaaaaaaaaaa')",
    'the label drops the prefix digest':
        "AssertionError: assert ('digest=' + ('a' * 16)) in 'l0match-000-strong6-o1_center-t1j_red-r0@ply6 [mover_fragmentation/position] '",
    'the run loop stops labelling its refusals':
        "AssertionError: t: the retained prefix replays to digest 0ae621381af163f0a180f590ebd7333d6c6070f5d9a364047ac25647df2c3e49, not the recorded '0000000000000000000000000000000000000000000000000000000000000000'. It is a different position: VOID.",
    'the probe refusal loses the position again':
        "AssertionError: assert 'l0match-000-strong6-o1_center-t1j_red-r0' in 'depth 6 invocation 0: exit 3 with 1 query records. T1j reported: POSTCOND no_throw=true windows=0 frames=0 headless=true prefs_ok=true refl_ok=true refl_n=3 failures=0'",
    'the BINDER discards the transcript on a non-zero replay exit':
        'AssertionError: t opening: T1j replay exit 3.',
    'the AGENT discards the transcript on a non-zero query exit':
        'AssertionError: t query at ply 6: exit 3 with 0 record(s).',
    "the binder's excerpt carries the dump body":
        'AssertionError: the legal-cell map leaked into the abort',
    'manifest drops the cohort label':
        "AssertionError: ['role', 'signature']",
    'registration checked only after the clock starts':
        'AssertionError: the reported clock started before registration passed',
    'registration checked only after the supervisor is armed':
        'AssertionError: a timer was armed before the registration check refused',
    'registration no longer stops the run before compilation':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'supervisor armed from limit_s instead of the remaining time':
        "AssertionError: armed with 100, expected the started deadline's remaining 95.0 (limit 100 minus 5 elapsed). 100.0 means it armed from limit_s and restarted the window.",
    'supervisor arms from a deadline that was never started':
        'scripts.GPU.alphazero.d1_probe.D1Error: deadline was never started',
    'supervisor arms a DISABLED timer when nothing remains':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1VoidError'>",
    'remaining() ignores elapsed time, so the window restarts':
        "AssertionError: armed with 100, expected the started deadline's remaining 95.0 (limit 100 minus 5 elapsed). 100.0 means it armed from limit_s and restarted the window.",
    'digest re-check disabled':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1VoidError'>",
    'prefix legality check disabled':
        'ValueError: Illegal move (11, 11) for active_size=24, to_move=black',
    'E3b binder never called':
        "AssertionError: [['13,10', '10,12', '14,14'], ['13,10', '10,12', '14,14'], ['13,10', '10,12', '14,14'], ['13,10', '10,12', '14,14']]",
    'binder AbortError not translated to VOID':
        "scripts.GPU.alphazero.e4_screen_runner.AbortError: [per_ply_binding] t@ply6 [?/?] digest=0ae621381af163f0 opening: pegs (ours-only ['14,14,X'], t1j-only ['20,20,X']); bridges (ours-only ['13,12|14,14|X'], t1j-only []); legal set |ours|=522 |t1j|=522",
    'binder replay timeout not translated to VOID':
        "subprocess.TimeoutExpired: Command '['/nonexistent/java', '-Djava.util.prefs.PreferencesFactory=e2probe.ScratchPrefsFactory', '-Djava.awt.headless=true', '-cp', '/nonexistent/t1j.jar:/nonexistent/classes', 'net.schwagereit.t1j.E3bDump', 'replay', '280', '11,11', '13,12', '12,13', '13,10', '10,12', '14,14']' timed out after 120 seconds",
    'ply_cap not carried to the runtime':
        "AssertionError: assert '999' == '280'",
    'replay timeout widened at the runtime':
        'AssertionError: assert 99999 == 120',
    "replay's unbounded-wait guard removed":
        "FileNotFoundError: [Errno 2] No such file or directory: 'j'",
    'replay timeout dropped at the last hop':
        'assert [None] == [42]',
    'T1jRuntime accepts an unbounded timeout':
        "Failed: DID NOT RAISE <class 'TypeError'>",
    'capture draws from the readout stream':
        'AssertionError: assert (3, (16201903...1, ...), None) == (3, (16201903...1, ...), None)',
    'capture switched on by default':
        'assert True is False',
    'capture flag not forwarded by the qualified builder':
        'assert False is True',
    'root-noise check disabled':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1VoidError'>",
    'evaluator reloaded for every position':
        "AssertionError: ['.', '.', '.']",
    'incumbent readout spends no query':
        'AssertionError: 12.4 funds ONE incumbent readout per position',
    'policy rank tie-break reversed':
        'assert 152 == (298 + 1)',
    'incumbent identity typed instead of read from the frozen plan':
        "AssertionError: assert {('calib020_0...54867b9473e')} == {('0379', '8a...c572a0b03a1')}",
    'digest downgraded to the sha1 superset helper':
        'AssertionError: assert (40 == 64)',
    'digest payload widened beyond the frozen three fields':
        "AssertionError: assert 'e64bef4b7176...dcafdce401860' == 'c6c8f50152b9...8926c44efcc07'",
    'per-cell cap loosened':
        "AssertionError: assert {('created_th...sition'): 129} == {('created_th...sition'): 101}",
    'deduplication disabled':
        "AssertionError: assert {('created_th...sition'): 104} == {('created_th...sition'): 101}",
    # §12.1 froze 101 mover_fragmentation positions BECAUSE the rule keeps only
    # plies where OUR incumbent moved. Without the restriction it keeps 147.
    'incumbent-to-move restriction dropped':
        "{('mover_fragmentation', 'position'): 147} != "
        "{('mover_fragmentation', 'position'): 101}",
    # 12.3 matches controls to the position cells BY CONSTRUCTION, so its
    # observable is the CONTROL count alone -- both position counts are
    # untouched by this defect, which is what distinguishes it from the one above.
    'controls no longer matched to the position cells':
        "{('mover_fragmentation', 'control'): 137} != "
        "{('mover_fragmentation', 'control'): 60}",
    'seed written in place so a shared row is overwritten':
        'assert [300000000, 3...00000005, ...] == [300000000, 3...00000005, ...]',
    'frozen-count reconciliation removed':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_selection.D1SelectionError'>",
    'lowply gate flipped open':
        'assert True is False',
    "lowply turns a non-zero exit back into a VOID (D1's defect)":
        'scripts.GPU.alphazero.lowply_qualification.LowPlyVoidError: t0@ply3 [mover_fragmentation/position] digest=470721202fb36f18: depth 3 invocation 0: exit 3 and no usable query record (1 parsed). T1j reported: FAIL q1: requested depth 3 completed | POSTCOND no_throw=true windows=0 frames=0 headless=true prefs_ok=true refl_ok=true refl_n=3 failures=1. VOID.',
    'lowply stops recording the incomplete-depth shortfall':
        "AssertionError: assert 'did not complete' in 'postcondition surface not clean: PostCond(no_throw=True, windows=0, frames=0, headless=True, prefs_ok=True, refl_ok=True, refl_n=3, failures=1, prefs_before=None, prefs_after=None, count_before=None, count_after=None)'",
    'lowply stops checking the move against OUR engine':
        "AssertionError: assert 'PASS' == 'FAIL'",
    'lowply stops comparing the two invocations':
        "AssertionError: assert 'PASS' == 'FAIL'",
    'lowply verdict always PASS':
        "AssertionError: assert 'PASS' == 'FAIL'",
    'lowply timeout no longer a VOID':
        "subprocess.TimeoutExpired: Command '['/nonexistent/java', '-Djava.util.prefs.PreferencesFactory=e2probe.ScratchPrefsFactory', '-Djava.awt.headless=true', '-cp', '/nonexistent/t1j.jar:/nonexistent/classes', 'net.schwagereit.t1j.E4Preflight', 'query', '3', '11,11', '13,12', '12,13']' timed out after 120 seconds",
    'lowply D1 exception translation removed':
        'scripts.GPU.alphazero.lowply_qualification.LowPlyError: before t0@ply3 [mover_fragmentation/position] digest=470721202fb36f18: whole-run deadline exceeded (99999.0s > 900s). The run is VOID: no partial-cohort analysis is produced.',
    'lowply frozen-input hash check disabled':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.lowply_qualification.LowPlyError'>",
    'lowply per-call timeout dropped at the last hop':
        "AssertionError: {'capture_output': True, 'text': True, 'timeout': None}",
    'lowply uses the same-JVM determinism mode':
        'ValueError: list.index(x): x not in list',
    'lowply query cap loosened':
        'assert 99999 == 36',
    'lowply wall-clock cap widened':
        'assert 90000 == 900',
    "lowply reads D1's gate":
        'AssertionError: D1_EXECUTION_AUTHORIZED',
    'adapter query parse failure re-raised bare':
        "ValueError: QUERY line missing fields ['completed', 'completed_depth', 'currentMaxPly', 'elapsed_us', 'eval_regime', 'legal', 'moveNr', 'move_y', 'null_sentinel', 'to_move', 'usealphabeta']: 'QUERY q=1 requested_depth=3 move_x=11'",
    'adapter replay parse failure re-raised bare':
        'ValueError: legal map is 4 bits, expected exactly 576',
    'lowply query parse failure not translated':
        "scripts.GPU.alphazero.t1j_adapter.HelperOutputError: the helper's query output could not be parsed: QUERY line missing fields ['completed', 'completed_depth', 'currentMaxPly', 'elapsed_us', 'eval_regime', 'legal', 'moveNr', 'move_y', 'null_sentinel', 'to_move', 'usealphabeta']: 'QUERY q=1 requested_depth=3 move_x=11'",
    'lowply replay parse failure not translated':
        "scripts.GPU.alphazero.t1j_adapter.HelperOutputError: the helper's replay output could not be parsed: legal map is 4 bits, expected exactly 576",
    'lowply POSTCOND parse failure not translated':
        "scripts.GPU.alphazero.t1j_adapter.HelperOutputError: the helper's POSTCOND output could not be parsed: line missing fields ['failures', 'frames', 'headless', 'prefs_ok', 'refl_n', 'refl_ok']: 'POSTCOND no_throw=true windows=0'",
    'lowply parse VOID carries the whole dump body':
        'AssertionError: the legal-cell map leaked into the refusal',
    'lowply main has no catch-all again':
        'RuntimeError: boom',
    'parse_postconds raises a bare ValueError again':
        'ValueError: ("the helper\'s POSTCOND output could not be parsed: line missing fields [\'failures\', \'frames\', \'headless\', \'prefs_ok\', \'refl_n\', \'refl_ok\']: \'POSTCOND no_throw=true windows=0\'", \'PLY 0 moveNr=0 next=Y termY=false termX=false\\n  PEGS \\n  BRIDGES \\n  HIST \\n  LEGAL 000000000000000000000000111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111000000000000000000000000\\nPLY 1 moveNr=1 next=X termY=false termX=false\\n  PEGS 11,11,Y\\n  BRIDGES \\n  HIST 11,11\\n  LEGAL 011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111110111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110\\nPLY 2 moveNr=2 next=Y termY=false termX=false\\n  PEGS 11,11,Y 13,12,X\\n  BRIDGES \\n  HIST 11,11 13,12\\n  LEGAL 000000000000000000000000111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111110111111111111111111111111111111111111111111111111011111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111000000000000000000000000\\nPLY 3 moveNr=3 next=X termY=false termX=false\\n  PEGS 11,11,Y 12,13,Y 13,12,X\\n  BRIDGES 11,11|12,13|Y\\n  HIST 11,11 13,12 12,13\\n  LEGAL 011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111110111111111110011111111111101111111110011111111111011111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110011111111111111111111110\\nPOSTCOND no_throw=true windows=0\\n\')',
    'parse_postconds drops the transcript it attaches':
        "AssertionError: assert '' == 'POSTCOND no_...e windows=0\\n'",
    'PostCond construction raises a bare ValueError again':
        "ValueError: invalid literal for int() with base 10: 'bad'",
    'PostCond parse failure drops its transcript':
        "AssertionError: assert '' == 'POSTCOND no_... failures=0\\n'",
    # §13.3's own reconciliation, raised where the test can see it fail.
    '§13 exclusion never applied':
        'scripts.GPU.alphazero.d1_selection.D1SelectionError: §13 mover_fragmentation controls: 60 after exclusion, but §13.3 froze 54',
    '§12 count reconciliation removed':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_selection.D1SelectionError'>",
    '§13 post-exclusion counts not reconciled':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_selection.D1SelectionError'>",
    '§13 accepts the retired seed block':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_selection.D1SelectionError'>",
    '§13 assigns seeds when none were reserved':
        'AssertionError: a position carries a seed although §13 reserved no interval',
    'D1 query cap raised back to the frozen 12.4 figure':
        'assert (227 == 221)',
    'D1 query parse failure not translated':
        "scripts.GPU.alphazero.t1j_adapter.HelperOutputError: the helper's query output could not be parsed: QUERY line missing fields ['completed', 'completed_depth', 'currentMaxPly', 'elapsed_us', 'eval_regime', 'legal', 'moveNr', 'move_y', 'null_sentinel', 'to_move', 'usealphabeta']: 'QUERY q=1 requested_depth=3 move_x=11'",
    'D1 replay parse failure not translated':
        "scripts.GPU.alphazero.t1j_adapter.HelperOutputError: the helper's replay output could not be parsed: legal map is 4 bits, expected exactly 576",
    'D1 main has no catch-all again':
        "FileNotFoundError: [Errno 2] No such file or directory: '/nonexistent'",
    'the trace stops validating VALUES (name-only again)':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'the enum check is opened to free text':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'the event enum is opened to free text':
        "KeyError: 'move=(11,11)'",
    'counters accept any type or magnitude':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'counters accept bools as integers':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    "counter ceilings widened past the run's real limits":
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'the schema string is no longer pinned':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'an event may carry extra fields':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'an event may omit declared fields':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'the identity strings are readmitted':
        "AssertionError: assert 'task_id' not in frozenset({'digest', 'event', 'index', 'n_positions', 'ply', 'positions_completed', ...})",
    'the trace validates AFTER writing':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'the trace is not written on a VOID':
        "AssertionError: assert ('position_start' in ['run_start', 'position_start', 'position_stage', 'position_stage', 'position_stage'] and 'run_end' in ['run_start', 'position_start', 'position_stage', 'position_stage', 'position_stage'])",
    'the trace stops counting seeds drawn':
        'AssertionError: one seed was drawn; the accounting must say so',
    'the trace is not fsynced per line':
        'AssertionError: []',
    'run_d1 accepts any cohort (the ceiling-only defect)':
        'AssertionError: the public entry does not require it',
    'the cohort size check is dropped':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'the cohort ORDER check is dropped':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'an excluded row is readmitted to the cohort':
        "scripts.GPU.alphazero.d1_probe.D1Error: the cohort is not the §13 selection in frozen order: row 220 has task_id='l0match-000-strong6-o1_center-t1j_red-r0' (str), expected 'l0match-060-strong6-o8_contact-t1j_black-r0' (str). The digest fixes the board; it does not fix the cohort label.",
    'the cohort source is no longer hash-pinned':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'the exact-spend check is dropped':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1VoidError'>",
    'the expected spend becomes a maximum again':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1VoidError'>",
    'the private seam starts enforcing the cohort too':
        'AssertionError: the private seam must stay usable for small fake cohorts',
    'cohort binds digests only, not the labels':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'cohort drops the grouping fields from the identity set':
        "AssertionError: assert {'digest', 'p...e', 'task_id'} == {'colour_arm'... 'phase', ...}",
    'the RAW mover_more_fragmented column is left unbound':
        "KeyError: 'mover_more_fragmented'",
    'the RAW created_threat column is left unbound':
        "KeyError: 'created_threat'",
    'a source field silently drops out of the binding':
        "AssertionError: ['colour_arm', 'seed'] are unbound as an IDENTITY field. `seed` is the only one that may be, because the frozen source carries none: its assignment is bound POSITIONALLY instead, by `_check_seed_assignment` (row i must carry SEED_INTERVAL[0] + i), not by comparison with the source.",
    'cohort tolerates a row missing an identity field':
        "KeyError: 'role'",
    'cohort pins the SEED as an IDENTITY field (the frozen source carries none)':
        'scripts.GPU.alphazero.d1_probe.D1Error: the cohort is not the §13 selection in frozen order: row 0 has seed=202615000 (int), expected 202614000 (int). The digest fixes the board; it does not fix the cohort label.',
    'cohort comparison drops type-strictness (== again)':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'the comparison helper stops checking type':
        'AssertionError: False == 0 in Python; the types differ',
    'bools slip through as ints (isinstance instead of type)':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'prefix normalisation coerces through int() again':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'structural normalisation dropped, so a tuple prefix is refused':
        'scripts.GPU.alphazero.d1_probe.D1Error: the cohort is not the §13 selection in frozen order: row 0 has prefix=((11, 11), (12, 13), (13, 12), (10, 13), (12, 10)) (tuple), expected [[11, 11], [12, 13], [13, 12], [10, 13], [12, 10]] (list). The digest fixes the board; it does not fix the cohort label.',
    'the corrected 5.4 justification is reinstated as the wrong one':
        'AssertionError: the withdrawn justification is back',
    'the interval is RETYPED as a literal in d1_probe':
        'assert (202615000, 202615221) is (202615000, 202615221)',
    'the canonical interval points back at the RETIRED block':
        'assert (202614000, 202614227) == (202615000, 202615221)',
    'the canonical interval is sized 227 again, not 221':
        'assert (202615000, 202615227) == (202615000, 202615221)',
    'RETIRED_SEED_INTERVAL moved to the new block':
        'assert (202615000, 202615221) == (202614000, 202614227)',
    'RETIRED_SEED_INTERVAL moved -- caught by the REFUSAL, not the constant':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_selection.D1SelectionError'>",
    'the retirement guard is removed from select_all':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_selection.D1SelectionError'>",
    'the runtime seed check admits the retired block too':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1VoidError'>",
    'the registration barrier computes nothing and so is always satisfied':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    'bind_results design becomes defaultable':
        "Failed: DID NOT RAISE <class 'TypeError'>",
    "H1 silently reverts to L0's 4 repetitions":
        'assert (8, 2, 4) == (8, 2, 14)',
    'the viability threshold moves to parity':
        'assert 0.5 == 0.75',
    'the verdict admits an upper bound AT the threshold':
        "AssertionError: assert 'VIABLE' == 'INCONCLUSIVE'",
    'the verdict reads the bounds in the wrong order':
        'AssertionError: (149, 0.6651785714285714, [0.5744365969935754, 0.7559205458635674])',
    'the decisive bands are hardcoded instead of derived':
        'assert 4.1974434995983856e-05 < 1e-12',
    'the task digest is a hand-typed constant again':
        "AssertionError: assert '6ff6b6c69bbc...e1e0a41b1a7dc' == '23e3fa129b85...2346376db5c74'",
    "H1 reuses L0's SPENT and retired seed block":
        'assert False',
    "H1's block overlaps D1's PAPER reservation":
        'AssertionError: ((202615100, 202615324), (202615000, 202615221))',
    'a shared vocabulary is RETYPED instead of re-exported':
        'AssertionError: WINNERS',
    'H1 acquires an early stop':
        'assert True is False',
    'cap saturation stops refusing a rate':
        'assert True is False',
    'the reporter binds H1 results to the 64-game design':
        'assert False is True',
    'the verdict is read off the WILSON interval':
        "AssertionError: assert 'VIABLE' == 'INCONCLUSIVE'",
    'the no-pooling prohibition is deleted':
        'assert False',
    'the repetition LABELS stop being checked':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h1_viability_plan.H1PlanError'>",
    'an incomplete match is reported instead of refused':
        "TypeError: 'NoneType' object is not iterable",
    "H1 APPENDS its seed rule to L0's instead of composing":
        "AssertionError: assert 'L0 block' not in 'any per-ply...ved H1 block'",
    'the block name stops reaching the seed rule':
        "AssertionError: assert 'reserved H1 block' in 'any per-ply state divergence between the two engines any T1j query that does not complete its requested depth any ill...ock, or any seed used twice any failure to write or fsync a durable record the whole-run wall-clock cap being exceeded'",
    "L0's own abort rules change under the factoring":
        "AssertionError: assert ('any per-ply...d twice', ...) == ('any per-ply...d twice', ...)",
    'the H1 seed predicate admits the L0 block too':
        'assert False is True',
    "the H1 seed predicate refuses H1's own seeds":
        'assert True is False',
    "H1 INHERITS L0's count-bearing prohibitions":
        "AssertionError: assert 'any per-cell comparison presented as a finding: 8 games per opening and 32 per colour arm carry no interval here and were not preregistered as tests' not in ('any Elo figure, or any conversion of this rate into one', 'any absolute strength placement -- T1j is uncalibrated, s...f the Wilson interval as conservative, exact or guaranteed; it is nominal, and hoeffding_interval is the primary', ...)",
    'the independence prohibition hardcodes 64 again':
        "AssertionError: assert 'any statement that the 64 games ARE independent, or that independence was verified; distinct streams rule out reuse, not dependence' not in ('any Elo figure, or any conversion of this rate into one', 'any absolute strength placement -- T1j is uncalibrated, s...f the Wilson interval as conservative, exact or guaranteed; it is nominal, and hoeffding_interval is the primary', ...)",
    'the per-cell denominators are swapped':
        "AssertionError: assert 'any per-cell comparison presented as a finding: 28 games per opening and 112 per colour arm carry no interval here and were not preregistered as tests' in ('any Elo figure, or any conversion of this rate into one', 'any absolute strength placement -- T1j is uncalibrated, s...f the Wilson interval as conservative, exact or guaranteed; it is nominal, and hoeffding_interval is the primary', ...)",
    "L0's own forbidden claims change under the factoring":
        "AssertionError: assert 'ca91ab858948...687f4661f0071' == '809f0ed8fa63...c8de54e0b2d7a'",
    'the plan pin reverts to the SUPERSEDED v1 artifact':
        "scripts.GPU.alphazero.h1_viability_plan.H1PlanError: cannot read the H1 plan: [Errno 2] No such file or directory: 'docs/superpowers/evidence/2026-09-07-t1j-h1-retry-prep/06_h1_plan_v2.json'",
    "H1 INHERITS L0's non-abort rules whole":
        'AssertionError: cap-termination saturation: caps never stop an L0 match; see CAP_NO_RATE_THRESHOLD',
    'H1 claims it has NO band, as L0 does':
        "AssertionError: assert '0.75' in 'cap-termination saturation: caps never stop an H1 match -- all 224 games are played. Past 112 of them the report is C...stop criterion. A verdict reachable early is a verdict biased by when someone chose to look any early stop of any kind'",
    'the early-stop prohibition is softened':
        "AssertionError: assert 'any early stop of any kind' in ('cap-termination saturation: caps never stop an H1 match -- all 224 games are played. Past 112 of them the report is ...erdict reachable early is a verdict biased by when someone chose to look', 'any early stop after the verdict is clear')",
    "L0's own non-abort rules change under the extraction":
        "AssertionError: assert ('cap-termina... of any kind') == ('cap-termina... of any kind')",
    'the artifact keeps L0-only wording':
        "scripts.GPU.alphazero.h1_viability_plan.H1PlanError: cannot read the H1 plan: [Errno 2] No such file or directory: 'docs/superpowers/evidence/2026-09-07-t1j-h1-retry-prep/06_h1_plan_v2.json'",
    'an H1 gate appears without the claim being revisited':
        'assert not True',
    'an H1 registration precondition appears silently':
        'AssertionError: a registration barrier appeared in scripts.GPU.alphazero.h1_viability_plan',
    'the H1 gate is flipped open':
        'assert True is False',
    'the H1 gate no longer binds the match path':
        'scripts.GPU.alphazero.h1_viability_runner.H1Error: the H1 schedule may not be executed: seed 202617000 was EXPOSED -- it has been drawn from -- and cannot be scheduled',
    'the registration barrier no longer binds':
        'scripts.GPU.alphazero.h1_viability_runner.H1Error: the H1 schedule may not be executed: seed 202617000 was EXPOSED -- it has been drawn from -- and cannot be scheduled',
    'the registration barrier checks only the endpoints':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h1_viability_runner.H1Error'>",
    'the barriers move BELOW the plan load and the recorder':
        'scripts.GPU.alphazero.h1_viability_runner.H1Error: the H1 schedule may not be executed: seed 202617000 was EXPOSED -- it has been drawn from -- and cannot be scheduled',
    'the gate opens the registration barrier too':
        'scripts.GPU.alphazero.h1_viability_runner.H1Error: the H1 schedule may not be executed: seed 202617000 was EXPOSED -- it has been drawn from -- and cannot be scheduled',
    'registering the block opens the gate too':
        'scripts.GPU.alphazero.h1_viability_runner.H1Error: match mode requires a trace path: the card freezes a create-only, non-analytic VOID trace, and a match that cannot say how far it got is not the design that was preregistered. Nothing has been written.',
    'match mode accepts a supplied plan':
        "scripts.GPU.alphazero.h1_viability_plan.H1PlanError: cannot read the H1 plan: [Errno 2] No such file or directory: '/private/var/folders/vm/g32c3nts67bfrpr06cmdzz4h0000gn/T/pytest-of-bill/pytest-<n>/test_match_mode_REFUSES_a_supp0/other.json'",
    'the content binding narrows to the digest dimensions (dead again)':
        'scripts.GPU.alphazero.h1_viability_runner.H1VoidError: [precondition] no state factory: the public runner opens no games The H1 match is VOID: no viability report is produced and the seed block retires whole.',
    'the seed-block check is dropped from the PLAN validator':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h1_viability_plan.H1PlanError'>",
    'the whole-run deadline is widened':
        'assert 1080000 == (180 * 60)',
    'the per-call timeout is dropped':
        'assert None == 120',
    'the cooperative deadline check is removed':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h1_viability_runner.H1VoidError'>",
    'the deadline is never checked between games':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h1_viability_runner.H1VoidError'>",
    'the trace file stops being create-only':
        "Failed: DID NOT RAISE <class 'FileExistsError'>",
    'the trace admits an identity string':
        "AssertionError: assert {'event', 'ga...task_id', ...} == {'event', 'ga...a', 'ts', ...}",
    'the trace validates AFTER the write':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h1_viability_runner.H1Error'>",
    'the trace stops recording how far a VOID run got':
        "AssertionError: assert 'task_start' == 'run_end'",
    'cap saturation is reported as a verdict after all':
        'scripts.GPU.alphazero.h1_viability_runner.H1VoidError: [classification] the match produced no report and no preregistered outcome: 113 of 224 games terminated at the ply cap, more than the preregistered threshold of 112; the positions did not resolve and there is no rate to report, and so no viability verdict The H1 match is VOID: no viability report is produced and the seed block retires whole.',
    'the early-stop ban names a function that does not exist':
        'AssertionError: e4_screen_rules.should_continue no longer exists; the ban is vacuous',
    'the match is hidden behind a mode list again':
        "AssertionError: assert 'match' in ('qualify',)",
    'the public match path supplies no production setup':
        'assert None is <function _production_setup at 0x<addr>>',
    'qualification builds production collaborators too':
        'AssertionError: qualification must not build production collaborators',
    'T1j loses the frozen per-call timeout in the setup':
        'assert None == 120',
    'the setup keeps the REFUSING binder':
        "AssertionError: assert <function _refuse_binder at 0x<addr>> == 'BINDER'",
    "the openings come from H1's plan again (KeyError on the real path)":
        "KeyError: 'openings'",
    'the incumbent is loaded per game instead of once':
        'AssertionError: the incumbent must be loaded once for the whole match',
    "the SIGALRM void escapes as D1's exception":
        'scripts.GPU.alphazero.d1_probe.D1VoidError: before game 1: whole-run deadline exceeded (100.0s > 1.0s). The run is VOID: no partial-cohort analysis is produced.',
    'the supervisor is never armed':
        "AssertionError: assert 'deadline' in 'the supervisor did not fire the h1 match is void: no viability report is produced and the seed block retires whole.'",
    'the VOID diagnostic stops naming the position':
        'assert 0 == 1',
    'the VOID diagnostic drops the helper transcript':
        "TypeError: argument of type 'NoneType' is not a container or iterable",
    'the VOID diagnostic carries the WHOLE transcript':
        'AssertionError: assert 5000050 <= (800 + 200)',
    'the ply counter stops observing durable ply records':
        'assert None == 17',
    'a partial vector can produce a report after all':
        'assert (True is False)',
    "the agent's own query timeout is dropped again":
        "KeyError: 't1j_timeout_s'",
    'compilation gets a SECOND deadline (two clocks again)':
        "AssertionError: compilation must use the RUN's clock, not a second one",
    'the setup factory is never invoked':
        "KeyError: 'path'",
    'the excerpt reads only error.stdout again':
        "AssertionError: assert 'FAIL: postcond' in 'binder raised HelperOutputError'",
    'the excerpt drops the abort-message fallback':
        'assert (None)',
    'the abort-message fallback is unbounded':
        "AssertionError: assert ('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX...XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' and 100000 <= 800)",
    'the ply tracker ignores opening_bound again':
        'AssertionError: a first-move failure must still name the position',
    'the binder wrapper stops noting the attempted ply':
        'AssertionError: the diagnostic named a ply the binder was not on',
    'the recorder is built OUTSIDE the protected try again':
        'scripts.GPU.alphazero.e4_screen_runner.HarnessError: cannot create the results file: [Errno 28] no space',
    'the output preflight is removed':
        'AssertionError: a precondition refusal is not a VOID',
    'the preflight ignores the trace path':
        "FileExistsError: [Errno 17] File exists: '/private/var/folders/vm/g32c3nts67bfrpr06cmdzz4h0000gn/T/pytest-of-bill/pytest-<n>/test_a_PREEXISTING_TRACE_path_0/t.jsonl'",
    'a precondition refusal is reported as a VOID':
        'AssertionError: a precondition refusal is not a VOID',
    'a mid-run recorder failure escapes unclassified again':
        "AssertionError: assert 'results file could not be created' in 'cannot create the results file: [Errno 28] no space The H1 match is VOID: no viability report is produced and the seed block retires whole.'",
    'match mode accepts a missing trace path again':
        'scripts.GPU.alphazero.h1_viability_runner.H1VoidError: [precondition] no state factory: the public runner opens no games The H1 match is VOID: no viability report is produced and the seed block retires whole.',
    'the trace requirement is not applied to the match':
        'scripts.GPU.alphazero.h1_viability_runner.H1VoidError: [precondition] no state factory: the public runner opens no games The H1 match is VOID: no viability report is produced and the seed block retires whole.',
    'results and trace may be the same file':
        'scripts.GPU.alphazero.h1_viability_runner.H1VoidError: the results file could not be created: results path already exists: /private/var/folders/vm/g32c3nts67bfrpr06cmdzz4h0000gn/T/pytest-of-bill/pytest-<n>/test_THE_TWO_OUTPUT_PATHS_MUST0/both.jsonl. A run writes a NEW file; appending would merge two runs. The H1 match is VOID: no viability report is produced and the seed block retires whole.',
    'the paths are compared WITHOUT canonicalisation':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h1_viability_runner.H1Error'>",
    'canonicalisation stops resolving symlinks':
        'scripts.GPU.alphazero.h1_viability_runner.H1Error: the trace path already exists: /private/var/folders/vm/g32c3nts67bfrpr06cmdzz4h0000gn/T/pytest-of-bill/pytest-<n>/test_a_SYMLINKED_trace_path_is0/link.jsonl (a dangling symlink -- the directory entry is present). A run writes NEW files; appending would merge two runs, and overwriting would destroy the record of one. Nothing has been written and no trace was opened.',
    'the output precheck follows symlinks again (exists, not lexists)':
        'AssertionError: a knowable path condition became a VOID',
    'the precheck stops agreeing with create-exclusive open':
        'AssertionError: dangling',
    'only the results name is checked for existence':
        "FileExistsError: [Errno 17] File exists: '/private/var/folders/vm/g32c3nts67bfrpr06cmdzz4h0000gn/T/pytest-of-bill/pytest-<n>/test_a_DANGLING_TRACE_link_is_0/t.jsonl'",
    'the public docstring denies that match mode is selectable':
        'AssertionError: the correction should record what it corrects',
    'the H1 block is un-registered from ACCOUNTED':
        'assert False',
    'the H1 exposure overstates the draws (all 224, not the 60 taken)':
        'assert 224 == 60',
    'the H1 retirement covers only the seeds actually drawn':
        'AssertionError: the block retires WHOLE',
    "D1's spent block loses its ACCOUNTED entry, from the H1 side":
        "AssertionError: (202615000, {'accounted': False, 'exposed': True, 'retired': True, 'test_only': False})",
    "an H1 block is made to overlap D1's spent block":
        'AssertionError: (202615100, 202615324)',
    'registering the block also opens the gate':
        'AssertionError: assert True is False',
    'a match abort escapes without becoming a VOID':
        'scripts.GPU.alphazero.e4_screen_runner.AbortError: [move] postcondition failure',
    'qualification aborts are relabelled as voids too':
        'scripts.GPU.alphazero.h1_viability_runner.H1VoidError: [move] synthetic The H1 match is VOID: no viability report is produced and the seed block retires whole.',
    'a KeyboardInterrupt is relabelled as an instrument failure':
        'scripts.GPU.alphazero.h1_viability_runner.H1VoidError:  The H1 match is VOID: no viability report is produced and the seed block retires whole.',
    'the diagnostic records the TRANSLATION instead of the cause':
        'AssertionError: the translation hid the cause',
    'a prefs failure is reported without the attribution note':
        "KeyError: 'prefs_attribution'",
    'the attribution note claims T1j did it':
        'assert False',
    'the note is attached to every failure, not just prefs ones':
        "AssertionError: assert 'prefs_attribution' not in {'colour_arm': 't1j_red', 'error_type': 'AbortError', 'games_completed': 0, 'helper_excerpt': 'per-ply divergence at ply 12', ...}",
    'the surface probe collapses ABSENT and UNREADABLE, as Java does':
        "AssertionError: assert 'ABSENT' == 'ERROR:PermissionError'",
    'the directory probe hides a read failure':
        "AssertionError: assert (-1 == -1 and 'PRESENT' == 'ERROR:PermissionError'",
    'no baseline surface is recorded at run start':
        'AssertionError: no baseline to compare against',
    'mover cut inverted':
        "AssertionError: assert {'created_thr...7710843373}}}} == {'created_thr...6541353385}}}}",
    'the observation fields are never read from the POSTCOND line':
        "AssertionError: assert (None, 'ERROR') == ('6cb3a052650...28d', 'ERROR')",
    'the observation is parsed but not attached to PostCond':
        "AssertionError: assert (None, None) == ('6cb3a052650...28d', 'ERROR')",
    'the POSTCOND self-agreement check disabled':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.t1j_adapter.HelperOutputError'>",
    'the self-agreement check compares the hash only, not the count':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.t1j_adapter.HelperOutputError'>",
    'the excerpt reader returns None when the earlier source said nothing':
        "AssertionError: assert None == {'count_after': None, 'count_before': None, 'prefs_after': None, 'prefs_before': None, ...}",
    'the excerpt reader requires POSTCOND to start a line':
        "AssertionError: assert None == {'count_after': -1, 'count_before': 527, 'prefs_after': 'ERROR', 'prefs_before': '6cb3a052650f90de53f34a8eb25455c470c6254c5f0fcac3f80c3ca9e8d0128d', ...}",
    "the VOID diagnostic drops the helper's own observation":
        "KeyError: 'helper_prefs_observed'",
    'the diagnostic reads only chained stdout, never the AbortError message':
        "AssertionError: assert None == {'count_after': -1, 'count_before': 527, 'prefs_after': 'ERROR', 'prefs_before': '6cb3a052650f90de53f34a8eb25455c470c6254c5f0fcac3f80c3ca9e8d0128d', ...}",
    'an operator interrupt is traced as VOID again':
        'AssertionError: the trace must not call a stop a VOID',
    'every failure is classified as an interrupt':
        "AssertionError: assert 'INTERRUPTED' == 'VOID'",
    'the interrupt record is still called a void_diagnostic':
        "AssertionError: assert ('void_diagnostic' not in ['run_header', 'task_start', 'task_result', 'task_start', 'task_result', 'task_start', ...])",
    'the accounting rule no longer travels with the interrupt record':
        "KeyError: 'seed_accounting'",
    'INTERRUPTED removed from the closed verdict enum':
        'KeyboardInterrupt',
    'the accounting rule softened: drawn seeds become reusable':
        'AssertionError: does NOT make any drawn seed reusable',
    'the partial-observation refusal removed':
        "scripts.GPU.alphazero.t1j_adapter.HelperOutputError: the helper's POSTCOND output could not be parsed: 'prefs_after' in 'POSTCOND no_throw=true windows=0 frames=0 headless=true prefs_ok=true refl_ok=true refl_n=3 failures=0 prefs_before=ABSENT'",
    'the partial-observation refusal removed, seen from the REAL query path':
        "scripts.GPU.alphazero.e4_screen_runner.AbortError: [move] h1match-000-strong6-o1_center-t1j_red-r0 ply 0: red raised the helper's POSTCOND output could not be parsed: 'count_after' in 'POSTCOND no_throw=true windows=0 frames=0 headless=true prefs_ok=true refl_ok=true refl_n=3 failures=0 prefs_before=ABSENT prefs_after=ERROR count_before=527'",
    'a truncated segment ending in the marker reads as the earlier source':
        "AssertionError: assert {'count_after': None, 'count_before': None, 'prefs_after': None, 'prefs_before': None, ...} is None",
    'a segment missing a BASE field reads as the earlier source':
        "AssertionError: assert {'count_after': None, 'count_before': None, 'prefs_after': None, 'prefs_before': None, ...} is None",
    'the query path stops chaining the full transcript':
        'AssertionError: assert False',
    'the diagnostic parses the BOUNDED excerpt instead of the full text':
        "KeyError: 'helper_prefs_observed'",
    'the attribution trigger reads only the bounded excerpt':
        "KeyError: 'helper_prefs_observed'",
    'requal gate removed at the worker entry':
        "AssertionError: (7, 'refused: the runtime requalification is UNAUTHORIZED. Gating only the CLI would protect nothing: a direct Python caller reaches this runner without passing it. Nothing has been compiled, queried or written.",
    'requal gate removed at main (the worker still refuses, so only the no-spawn test sees it)':
        'AssertionError: spawned',
    "the frozen prefix file's hash pin removed":
        'scripts.GPU.alphazero.runtime_requalification.RequalError: the frozen prefix file and the pinned screen plan disagree on the openings; two frozen sources must say one thing.',
    'the prefix-file vs plan cross-check removed':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.runtime_requalification.RequalError'>",
    'a reply without the observation is read as legacy instead of VOID':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.runtime_requalification.RequalVoidError'>",
    'the parser/reader agreement check removed':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.runtime_requalification.RequalVoidError'>",
    'a dirty preference surface no longer recorded as a failure':
        "AssertionError: ['the agent refused the reply: [move] o3_low@ply6 [H1-FAILED] digest=0dc546b435a20e17 query at ply 6: exit 3 with 1 re...xxxxxxxxxxxxxxxxxx | FAIL q1: check number 7 did not hold for reason xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...']",
    'the diagnostic observation read from the bounded excerpt, not the chain':
        "AssertionError: assert None == {'count_after': -1, 'count_before': 527, 'prefs_after': 'ERROR', 'prefs_before': '6cb3a052650f90de53f34a8eb25455c470c6254c5f0fcac3f80c3ca9e8d0128d', ...}",
    'the exact-eight requirement removed from the public runner':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.runtime_requalification.RequalError'>",
    'the create-only precheck removed (O_EXCL still fails, but after 16 launches)':
        'AssertionError: it ran the whole sample and then failed to write',
    'the supervisor no longer starts the worker in its own session':
        'AssertionError: the worker is not its own process-group leader',
    'the supervisor stops at SIGTERM and never SIGKILLs the group':
        'AssertionError: the supervisor did not KILL',
    'a supervisor kill is reported as VOID instead of TIMEOUT':
        'assert (True is True and 3 == 6)',
    'a FAIL result exits 0 like a pass':
        'assert 0 == 2',
    'the outer cap shrinks to the inner deadline (no grace for the inner VOID)':
        'assert 900 == (900 + 60)',
    'every call no longer carries the frozen per-call timeout':
        'assert False',
    'no cleanup after a worker that exits before the timeout':
        'assert False is True',
    "main reports the worker's code although cleanup failed":
        'AssertionError: assert 0 == 8',
    'EPERM from the group probe read as CLEARED':
        'assert True is False',
    'row binding compares digests only again':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.runtime_requalification.RequalError'>",
    'row binding drops the TYPE check (False == 0)':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.runtime_requalification.RequalError'>",
    'row binding accepts extra keys':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.runtime_requalification.RequalError'>",
    'row binding normalises VALUES, not only shape':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.runtime_requalification.RequalError'>",
    'the attempt-2 block moved back onto the SPENT attempt-1 block':
        'assert (202616000, 202616224) == (202617000, 202617224)',
    'the plan loader stops verifying the v4 file hash':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h1_viability_plan.H1PlanError'>",
    'the plan loader stops verifying the task digest':
        'scripts.GPU.alphazero.h1_viability_plan.H1PlanError: 17 opening/colour cells, expected 16',
    "the wrapper's main stops reading the runner's gate":
        'AssertionError: restored',
    "the wrapper's worker stops reading the runner's gate":
        'assert 4 == 5',
    'the wrapper skips the output-path precheck':
        'assert 4 == 7',
    'a failed gate restoration no longer supersedes':
        'assert 0 == 10',
    'the gate is restored only on the happy path (else, not finally)':
        'AssertionError: an exception in the supervisor skipped the restore',
    'restore_gate reports success without verifying the rewrite':
        'AssertionError: assert True is False',
    "the wrapper reports the worker's code although cleanup failed":
        'assert (0 == 8)',
    'an operator interrupt in the worker exits 0':
        'assert 0 == 9',
    'the supervisor no longer forwards an operator interrupt':
        'AssertionError: the worker did not exit on the forwarded SIGINT',
    "the wrapper's outer cap shrinks to the runner deadline":
        'assert 10800 == (10800 + 60)',
    'the output precheck moves back IN FRONT of the restoration boundary':
        'AssertionError: output refusal skipped gate restoration',
    'a failed restoration no longer supersedes a refusal':
        'assert 7 == 10',
    'the --runner-source override returns to the production CLI':
        "Failed: DID NOT RAISE <class 'SystemExit'>",
    'the attempt-2 block un-registered from ACCOUNTED':
        'assert False',
    'the attempt-2 block un-RETIRED after the completed match':
        'AssertionError: the block retires WHOLE',
    'the attempt-2 block un-EXPOSED after the completed match':
        'assert 0 == 224',
    'the runner stops asking whether the schedule may be RUN (spent block accepted)':
        'scripts.GPU.alphazero.h1_viability_runner.H1VoidError: [precondition] no state factory: the public runner opens no games The H1 match is VOID: no viability report is produced and the seed block retires whole.',
    'rank ties broken in REVERSE (row, col)':
        'assert (4 == 2)',
    'agree coupled to lprd (the withdrawn implication)':
        'assert (True is True and False is True)',
    'k drifts from 5 to 6':
        'assert False is True',
    'the depth-3 move read instead of mdPly 6':
        'assert (0, 1) == (0, 8)',
    'MH weights replaced by equal weights':
        'assert 0.5 == 0.42857142857142855 ± 4.3e-07',
    'common support pooled: cells with ONE role enter the statistic':
        'ZeroDivisionError: division by zero',
    'the direction flipped (controls minus positions)':
        'assert (0.3333333333333333 == -0.3333333333333333 ± 3.3e-07',
    'the decision threshold lowered':
        'AssertionError: assert (0.08333333333333333 == 0.08333333333333333 ± 8.3e-08',
    'the lower-bound condition dropped (T alone decides)':
        "AssertionError: assert 'GO' == 'NO_GO'",
    'undefined replicates silently DISCARDED':
        'assert 0 == 109',
    'a game drawn twice contributes its rows ONCE (re-deduplicated)':
        'AssertionError: assert 72 == 96',
    'the second draw copies the first (no replacement)':
        'AssertionError: 0',
    'the PRNG seed drifts':
        'AssertionError: 0',
    'the quantile convention changes from type 7':
        'assert (1.0, 4.0) == approx((1.075...25 ± 3.9e-06))',
    'the eligibility floor becomes strict (refuses AT the floor)':
        "AssertionError: {'cells': 8, 'controls': 30, 'games': 12, 'positions': 40}",
    'contributing games counted from UNMATCHED rows too':
        'assert 3 == 2',
    'an undefined arm reported as plain NO_GO (treated as zero)':
        "AssertionError: assert 'NO_GO' == 'NO_GO — arm undefined'",
    'the both-arm rule dropped':
        "AssertionError: assert 'GO' == 'NO_GO'",
    'cross-half duplicates no longer removed':
        "AssertionError: assert {'x0', 'x1', 'x2'} == {'x1', 'x2', 'x3'}",
    'the ply < 5 eligibility filter dropped':
        "AssertionError: assert {'a', 'b', 'c', 'd'} == {'c', 'd'}",
    'the ceiling drops silently instead of refusing':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1prime_analysis.D1PrimeRefused'>",
    'within-half dedup keeps the LATEST row':
        "AssertionError: assert [('g2', 10)] == [('g1', 12)]",
    'cohort binding loses type strictness':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1prime_analysis.D1PrimeError'>",
    'cohort binding stops checking the count':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1prime_analysis.D1PrimeError'>",
    'the checked entry stops binding the cohort':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1prime_analysis.D1PrimeError'>",
    'the checked entry takes a cohort argument again (secondary promotable)':
        "AssertionError: assert 'GO' == 'NO_GO — insufficient support'",
    'the design/cohort game check removed':
        "KeyError: 'g-o1_center-t1j_red-r0'",
    'strata inferred from the ROWS instead of the design':
        "scripts.GPU.alphazero.d1prime_analysis.D1PrimeError: stratum ('o1_center', 't1j_red') holds 1 games; the design has exactly 2 per stratum in a half",
    "eligibility defaults a missing 'system' to acceptance":
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1prime_analysis.D1PrimeError'>",
    "eligibility reads an invented flag instead of 'system'":
        "AssertionError: assert {'a', 'b'} == {'a'}",
    'the opponent-to-move filter inverted':
        "AssertionError: assert {'d0'} == {'d1'}",
    'the override flag read from the invented field name':
        'assert None is True',
    'a record without the override field is silently None':
        "KeyError: 'readout_overrode_leader'",
    'the production entry takes the cohort from the caller again':
        "KeyError: 'prefix'",
    'the production entry uses a smaller B than the frozen one':
        'assert (100 == 10000)',
    # The record D1' actually resolved from. It can only appear if the constant moved.
    'the canonical cohort is resolved from a different record':
        'docs/superpowers/evidence/2026-09-07-t1j-h1-match-attempt2/01_h1_results.jsonl',
    'the design metadata binding removed':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1prime_analysis.D1PrimeError'>",
    'the design binding loses type strictness on rep':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1prime_analysis.D1PrimeError'>",
    'the design binding skips a field the row omits':
        "KeyError: 'rep'",
    'the undefined-arm check runs AFTER the generic NO_GO again':
        'AssertionError: NO_GO — insufficient support',
    'D1 stops binding the seed assignment to positions':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    "D1's seed rule stops being POSITIONAL (interval membership only)":
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    "D1's seed rule loses type strictness":
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1_probe.D1Error'>",
    "D1' analyses a record with no acquisition metadata":
        'Failed: a row was built from an unchecked record',
    "D1' stops requiring the acquisition fields to be present":
        "KeyError: 'queries_spent'",
    "D1' stops requiring elapsed_s and positions to be present":
        "KeyError: 'elapsed_s'",
    "D1' accepts an acquisition value that disagrees with the frozen run":
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1prime_analysis.D1PrimeError'>",
    "D1' stops binding the per-position seed assignment":
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1prime_analysis.D1PrimeError'>",
    "D1' stops binding the per-position prefix":
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1prime_analysis.D1PrimeError'>",
    "D1' runs the contract AFTER the statistic":
        'Failed: a row was built from an unchecked record',
    "D1' accepts any non-empty incumbent identity again":
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1prime_analysis.D1PrimeError'>",
    "D1' accepts any non-empty toolchain identity again":
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1prime_analysis.D1PrimeError'>",
    "D1' stops checking the compiled-class identity":
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1prime_analysis.D1PrimeError'>",
    "D1' stops checking the helper SOURCE identity":
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1prime_analysis.D1PrimeError'>",
    "D1' stops checking the MAIN CLASS":
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1prime_analysis.D1PrimeError'>",
    'the compiled-class pin drifts from the qualified builds':
        "AssertionError: assert {'e2probe/Scr...000000000000'} == {'e2probe/Scr...abfbdcfc88a8'}",
    'a malformed toolchain identity raises a TypeError instead of a refusal':
        "TypeError: argument of type 'int' is not a container or iterable",
    'elapsed_s is no longer bounded by the deadline':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1prime_analysis.D1PrimeError'>",
    'malformed positions raise a TypeError instead of a refusal':
        "TypeError: object of type 'int' has no len()",
    'the analysis reaches an EXECUTING adapter function':
        "AssertionError: {'PREFLIGHT_SOURCES', 'query'}",
    'rank_visit breaks ties by DESCENDING move order':
        'assert (3 == 2)',
    'rank_visit orders by ASCENDING visits':
        'assert 3 == 1',
    'rank_visit DROPS zero-visit moves':
        'KeyError: (1, 1)',
    'rank_visit accepts an empty root instead of refusing':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1second_analysis.D1SecondError'>",
    'ss is EXCLUSIVE at five on the policy side':
        'assert False is True',
    'ss is INCLUSIVE at five on the visit side':
        'assert True is False',
    'ss ignores the VISITS entirely -- the whole question':
        'assert True is False',
    'ss drops the policy condition, so it OVERLAPS lprd':
        "AssertionError: (6, 6, {'agree': False, 'colour_arm': 't1j_red', 'digest': 'dX', 'lprd': True, ...})",
    'the strict variant fires on ANY visit count':
        'assert True is False',
    'the policy and the visits may cover DIFFERENT legal sets':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1second_analysis.D1SecondError'>",
    'rank_visit accepts an empty root, so an empty record is scored':
        'scripts.GPU.alphazero.d1second_analysis.D1SecondError: g-o1_center-t1j_red-r0: the raw policy covers 10 moves and the root visits 0; they must be the same legal set',
    'the disjointness precondition checks nothing':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1second_analysis.D1SecondError'>",
    'the public entry never RUNS the precondition':
        "KeyError: 'n'",
    'the entry skips the completed-D1 contract':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1prime_analysis.D1PrimeError'>",
    'the entry exposes B and the seed as knobs again':
        "AssertionError: ['d1_report', 'B', 'seed']",
    "D1″ reuses D1′'s bootstrap seed, so the draw repeats it":
        'assert 20260907 == 20260908',
    "the threshold is RETYPED and drifts below D1′'s":
        'assert 0.05 is 0.15',
    'the eligibility floor is never enforced':
        "AssertionError: assert 'GO' == 'NO_GO — insufficient support'",
    'the SECONDARY strict variant decides the outcome':
        "AssertionError: assert 'GO' == 'NO_GO'",
    'the readout summary tolerates an absent measurement':
        "KeyError: 'overrode_leader'",
    'the matched statistic IGNORES the requested indicator':
        'assert 1.0 == -1.0',
    'a missing indicator field is read as False':
        "KeyError: 'ss'",
    "the indicator changes D1′'s own default result":
        "scripts.GPU.alphazero.d1prime_analysis.D1PrimeError: a row of 'mover_fragmentation' carries no 'ss' field, so the statistic would score an absent measurement; every row must carry it.",
    'the validation pass is never RUN by the entry':
        'Failed: a row was scored before validation finished',
    'validation runs PER ROW, interleaved with the scoring':
        'Failed: a row was scored before validation finished',
    'a required observable is optional again':
        "KeyError: 'selected_policy_rank'",
    'a visit count is coerced with int() instead of type-checked':
        "scripts.GPU.alphazero.d1second_analysis.D1SecondError: position 3 ('l0match-000-strong6-o1_center-t1j_red-r0'): root_total_visits is 10, but the frozen configuration spends 400 simulations. A self-consistent map that totals something else describes a search this hypothesis is not about.",
    'a negative or fractional visit count is accepted':
        "scripts.GPU.alphazero.d1second_analysis.D1SecondError: position 3 ('l0match-000-strong6-o1_center-t1j_red-r0'): root_total_visits is 8, but the frozen configuration spends 400 simulations. A self-consistent map that totals something else describes a search this hypothesis is not about.",
    'a NaN or infinite policy mass reaches the ranking':
        "scripts.GPU.alphazero.d1second_analysis.D1SecondError: position 3 ('l0match-000-strong6-o1_center-t1j_red-r0'): selected_policy_mass is 0.1 but the record's own maps give nan; they are computed from the same maps, so they must agree exactly.",
    'an all-zero policy is ranked by tie-break alone':
        "scripts.GPU.alphazero.d1second_analysis.D1SecondError: position 3 ('l0match-000-strong6-o1_center-t1j_red-r0'): selected_policy_mass is 0.1 but the record's own maps give 0.0; they are computed from the same maps, so they must agree exactly.",
    'an observable may CONTRADICT the map it describes':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1second_analysis.D1SecondError'>",
    'a truthy int passes as the override flag':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1second_analysis.D1SecondError'>",
    'validation lets the policy and the visits differ':
        "scripts.GPU.alphazero.d1second_analysis.D1SecondError: position 3 ('l0match-000-strong6-o1_center-t1j_red-r0'): n_legal is 10 but the record's own maps give 11; an observable that contradicts what it describes is not evidence.",
    'a depth-6 move of strings is accepted':
        "scripts.GPU.alphazero.d1second_analysis.D1SecondError: position 3 ('l0match-000-strong6-o1_center-t1j_red-r0'): the depth-6 move ('0', '0') is not a legal move of this root",
    'a small complement is reported as INSUFFICIENT SUPPORT':
        "AssertionError: assert 'NO_GO — insufficient support' == 'NO_GO'",
    'the complement count is never reported':
        'assert 0 == 64',
    'the summary averages the ranks instead of counting them':
        'assert [[2.5, 3]] == [[1, 2], [4, 1]]',
    'the histogram is ordered by COUNT instead of by rank':
        'assert [[5, 2], [1, 1], [9, 1]] == [[1, 1], [5, 2], [9, 1]]',
    'the histogram emits zero-count ranks':
        'assert [[1, 1], [2, ..., [6, 0], ...] == [[1, 1], [5, 2], [9, 1]]',
    'the row drops the rank fields the summary histograms':
        'assert 1 == 2',
    'the row hardcodes the POLICY rank the summary histograms':
        'assert 1 == 4',
    "the depth is coerced, so '6' and 6.0 pass a type-strict contract":
        "scripts.GPU.alphazero.d1second_analysis.D1SecondError: position 3 ('l0match-000-strong6-o1_center-t1j_red-r0'): expected exactly one depth-6 record, got 0",
    'a non-mapping depth entry escapes as an AttributeError':
        'TypeError: list indices must be integers or slices, not str',
    'a rank comparison admits a bool equal to the right number':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1second_analysis.D1SecondError'>",
    'the depths container itself is never validated':
        "scripts.GPU.alphazero.d1second_analysis.D1SecondError: position 3 ('l0match-000-strong6-o1_center-t1j_red-r0'): depth record 0 is 'n', not a mapping",
    'the search budget is not bound at all':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1second_analysis.D1SecondError'>",
    'the budget is retyped in this module instead of read':
        'AssertionError: the budget is retyped as a literal on line 114',
    'derived values are compared APPROXIMATELY again':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.d1second_analysis.D1SecondError'>",
    '_real accepts bools, so True passes as a number':
        "scripts.GPU.alphazero.d1second_analysis.D1SecondError: position 3 ('l0match-000-strong6-o1_center-t1j_red-r0'): selected_policy_mass is 0.1 but the record's own maps give 1.0; they are computed from the same maps, so they must agree exactly.",
    'top2 is silently omitted rather than declared unused':
        "AssertionError: assert 'top2' in ()",
    'the parity rule reads 0.75 instead of parity':
        'assert 0.75 == 0.5',
    'the verdict is decided at the wrong side of the interval':
        "AssertionError: assert 'T1J_STRONGER' == 'INCONCLUSIVE'",
    'AT parity counts as above it':
        "AssertionError: assert 'T1J_STRONGER' == 'INCONCLUSIVE'",
    'the sample size drifts from the card':
        'assert (14, 8, 2, 224) == (46, 8, 2, 736)',
    'the per-cell threshold is relaxed by one':
        'assert True is False',
    'the transcript carries the SEED, so identical play looks distinct':
        "AssertionError: assert '5833cb96c1d7...677aac4920fcc' == 'db3014c9d890...53ce3c412f74e'",
    'the transcript drops the moves, so every game looks the same':
        "AssertionError: assert 'b5d7c9900cf4e353104d8a4eaf49467f2f8f03b2afc5a7b3835e81e217c89de1' != 'b5d7c9900cf4e353104d8a4eaf49467f2f8f03b2afc5a7b3835e81e217c89de1'",
    'the ply sequence is checked for CONTIGUITY, not for its exact span':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_rules.H2RulesError'>",
    'a truncated tail passes because only the head is anchored':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_rules.H2RulesError'>",
    'movers must merely ALTERNATE, not match the arm':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_rules.H2RulesError'>",
    'the terminal reason is not checked against the two the protocol has':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_rules.H2RulesError'>",
    'coordinates are coerced instead of type-checked':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_rules.H2RulesError'>",
    'the degeneracy screen counts GLOBALLY instead of per cell':
        "scripts.GPU.alphazero.h2_match_rules.H2RulesError: 1 cell(s) do not hold exactly 46 games: [['all', 'cells']]",
    'the interval is computed even when the screen fails':
        'assert True is False',
    "the report drops the interval's standing":
        "KeyError: 'interval_standing'",
    'the plan assigns seeds by membership, not POSITION':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_plan.H2PlanError'>",
    'the plan omits the readout mode from its tasks':
        "KeyError: 'selection_mode'",
    'the schedule accepts a task without the readout mode':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_plan.H2PlanError'>",
    'a short schedule is accepted':
        "scripts.GPU.alphazero.h2_match_plan.H2PlanError: cells without exactly 46 repetitions: [(('o8_contact', 't1j_black'), 45)]",
    'the gate defaults OPEN':
        'assert True is False',
    'the registration barrier checks only the first seed':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_runner.H2Error'>",
    'the registration barrier is disabled':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_runner.H2Error'>",
    'outputs are overwritten instead of refused':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_runner.H2Error'>",
    'one path may serve as both results and trace':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_runner.H2Error'>",
    'the identity keeps the OLD readout':
        "AssertionError: assert 'opening_temperature' == 'argmax'",
    'the inert settings are left looking ACTIVE':
        'AssertionError: an inert setting must not look active',
    'a run recording the OLD readout is reported as H2':
        "scripts.GPU.alphazero.h2_match_runner.H2VoidError: incumbent_identity.eval_config.selection_mode: recorded 'opening_temperature' (str) but the frozen configuration gives 'argmax' (str)",
    'the identity is compared SHALLOWLY again':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_runner.H2VoidError'>",
    # 🔴 CORRECTED THREE TIMES, AND THE PATTERN IS THE POINT. With the digest
    # comparison removed, `check_schedule` runs on to the AVAILABILITY check, so
    # WHAT THIS CONTROL OBSERVES DEPENDS ON WHETHER THE BLOCK HAS BEEN DRAWN FROM:
    #   attempt 2 spent -> E4ReferenceError, seed 202620000 EXPOSED
    #   attempt 3 fresh -> DID NOT RAISE (the digest check was the only guard left)
    #   attempt 3 spent -> E4ReferenceError, seed 202622000 EXPOSED   <- now
    # Only the middle state actually exercised the digest comparison. That is a
    # property of the TARGET TEST, not of this control, and it is recorded here
    # rather than papered over.
    'the schedule digest is not compared with the pin':
        'scripts.GPU.alphazero.e4_screen_reference.E4ReferenceError: seed 202622000 '
        'was EXPOSED -- it has been drawn from -- and cannot be scheduled',
    'the deadline never fires':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_runner.H2VoidError'>",
    'a malformed ply record is skipped instead of VOIDing':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_runner.H2VoidError'>",
    # 🔴 CORRECTED TWICE-OVER 2026-09-12. The frozen reason was `assert 7 == 10`:
    # with the gate forced open the wrapper was REFUSED at the output precheck,
    # because the defaults pointed at attempt 1's occupied directory -- so the
    # control was observing a refusal, not the finally. The target test is hermetic
    # now (tmp outputs, stubbed supervisor) and the wrapper returns the supervised
    # code 0 with the finally's check disabled, instead of EXIT_GATE_NOT_RESTORED.
    'the wrapper reports success without verifying the gate':
        'AssertionError: assert 0 == 10',
    'restoration claims success without reading the file back':
        'AssertionError: assert True is False',
    'the wrapper gains a --runner-source override':
        "Failed: DID NOT RAISE <class 'SystemExit'>",
    'the mover is derived from the ARM again':
        "AssertionError: (1, 'red')",
    "the starting colour drifts from the engine's":
        "AssertionError: assert 'red' == 'black'",
    'the public entry accepts caller-supplied play again':
        "AssertionError: assert ['results_pat...', 'identity'] == ['results_pat...'report_path']",
    'the production seam is a refusing stub again':
        'AssertionError: the seam refuses before playing: that is a stub, not a path',
    'the seam no longer calls the harness game loop':
        'AssertionError: the seam must call e4_screen_runner.play_task',
    'the production seam loads the model AT IMPORT':
        "AssertionError: ('e4_screen_runner', ['', 'Any', 'Callable', 'Dict', 'List', 'Mapping', ...])",
    'the incumbent keeps the temperature config in production':
        'AssertionError: the config must be REBUILT, not aliased',
    'a DANGLING SYMLINK reads as absent':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_runner.H2Error'>",
    'the screen accepts a SHORT transcript vector':
        "scripts.GPU.alphazero.h2_match_rules.H2RulesError: 16 cell(s) do not hold exactly 46 games: [['o1', 't1j_black'], ['o1', 't1j_red'], ['o2', 't1j_black']]",
    'the screen ignores the canonical task ids':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_rules.H2RulesError'>",
    'a cell with fewer than 46 GAMES is screened anyway':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_rules.H2RulesError'>",
    'invalid results are masked as a DEGENERATE DESIGN':
        'AssertionError: INCONCLUSIVE — DEGENERATE DESIGN',
    'the transcript evidence is not persisted':
        "AssertionError: {'header', 'ply', 'task_result'}",
    'a mid-run failure leaves no run_end VOID':
        "AssertionError: assert {'event': 'ta...', 'index': 0} == {'error': 'Ru...dict': 'VOID'}",
    'the wrapper runs the match IN PROCESS, unsupervised':
        "KeyError: 'cmd'",
    'the outer cap is unbounded':
        'assert None == (28800 + 60)',
    'a surviving descendant accompanies a success':
        'AssertionError: assert 0 == 8',
    "a timeout is reported as the worker's own exit code":
        'AssertionError: assert 0 == 6',
    'compile is handed a fresh, unstarted clock again':
        'assert [<scripts.GPU... 0x<addr>>] == [<scripts.GPU... 0x<addr>>]',
    'the seam accepts an unstarted clock':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_runner.H2Error'>",
    'the run never starts its deadline':
        'scripts.GPU.alphazero.d1_probe.D1Error: deadline was never started',
    'the deadline is never ARMED, so a blocked game runs on':
        "KeyError: 'entered'",
    'the outcome is never classified':
        'assert 0 == 7',
    'a REFUSED report still exits COMPLETED':
        'AssertionError: assert 0 == 7',
    'a degenerate design is reported as a completed match':
        'AssertionError: assert 7 == 11',
    'the report overwrites an existing one':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_runner.H2VoidError'>",
    'the preflight lets an existing report through to the games':
        'Failed: a game was played',
    'an operator interrupt is recorded as VOID':
        "AssertionError: {'error': 'KeyboardInterrupt', 'event': 'run_end', 'games_completed': 0, 'verdict': 'VOID'}",
    'no cleanup runs between games':
        'assert 0 == 3',
    'cleanup runs only after a SUCCESSFUL game':
        'AssertionError: a failed game must still clean up',
    'the evaluator is reloaded for every game':
        'assert 5 == 1',
    'the incumbent is built with the frozen TEMPERATURE config':
        "AssertionError: assert 'opening_temperature' == 'argmax'",
    'the schedule is pinned only by its DESIGN digest':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_runner.H2Error'>",
    'the full-field digest projects fields away':
        "AssertionError: assert '14328238f88046b995fb65bab9dc7be2f60cb026122440440899bcdb2d2a1a71' != '14328238f88046b995fb65bab9dc7be2f60cb026122440440899bcdb2d2a1a71'",
    '--worker is a public bypass again':
        'Failed: the match RAN',
    'the report is left out of the preflight':
        'TypeError: expected str, bytes or os.PathLike object, not NoneType',
    'only two outputs are compared for aliasing':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h2_match_runner.H2Error'>",
    'the terminal OK is committed BEFORE the report is durable':
        "AssertionError: [{'event': 'run_end', 'games_completed': 736, 'verdict': 'OK'}, {'error': 'FileExistsError', 'event': 'run_end', 'games_completed': 736, 'verdict': 'VOID'}]",
    'the trace does not name the verdict it committed':
        "KeyError: 'outcome'",
    'the capability is a fixed, caller-settable value':
        "AssertionError: assert ('1111111111111111111111111111111111111111111111111111111111111111' != '1111111111111111111111111111111111111111111111111111111111111111')",
    'the capability is not an anonymous pipe':
        'AssertionError: it must be an anonymous pipe',
    'any token of the RIGHT LENGTH is accepted':
        "AssertionError: 'gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg' was accepted",
    'any capability at all is accepted':
        "AssertionError: 'short' was accepted",
    'the capability check DELETES the thing it was given':
        "AssertionError: assert not [Call(func=Attribute(value=Name(id='_o', ctx=Load(...)), attr='unlink', ctx=Load()), args=[Call(func=Name(id='str', ctx=Load(...)), args=[Name(id='fd', ctx=Load(...))], keywords=[])], keywords=[])]",
    'an environment variable reaches the worker path':
        'AssertionError: an environment variable is being read',
    'a capability outlives the run':
        "Failed: DID NOT RAISE <class 'OSError'>",
    'the descriptor is not PASSED to the child at all':
        'AssertionError: the descriptor must be INHERITED',
    'the shared supervisor drops pass_fds':
        "AssertionError: {'exit_code': 1, 'group_cleared': True, 'interrupted': False, 'timed_out': False}",
    'the classifier assumes the report exists':
        'assert 0 == 3',
    'the degeneracy refusal claims the games were non-independent':
        'assert \'DIVERSITY screen failed\' in "1 of 16 cell(s) hold fewer than 42 distinct transcripts of 46: [[\'o1\', \'t1j_red\']]. The primary interval is NOT compu...nd §3.1 says plainly that this screen cannot test independence. What failed is the design\'s own diversity requirement."',
    "H2's block un-registered from ACCOUNTED (the edit reverted)":
        "AssertionError: (202618000, {'accounted': False, 'exposed': False, 'retired': True, 'test_only': False})",
    "H2's block ALSO marked EXPOSED -- a reservation claimed as a draw":
        "AssertionError: (202618000, {'accounted': True, 'exposed': True, 'retired': True, 'test_only': False})",
    'the SPENT block is not retired, so it can be scheduled again':
        "AssertionError: (202618000, {'accounted': True, 'exposed': False, 'retired': False, 'test_only': False})",
    'a spent seed is schedulable because availability is never checked (H2)':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.e4_screen_reference.E4ReferenceError'>",
    'registration ALSO opened the H2 gate':
        'assert True is False',
    'the negative control stops stripping anything':
        "AssertionError: nothing was stripped: H2's block is NOT registered, so this controls nothing",
    "the builder refuses H2's readout again":
        "scripts.GPU.alphazero.twixtbot_g3_reference.ReferenceError: selection_mode 'argmax' is not one of the admitted readouts ('opening_temperature', 'argmax'). Admitting the FIELD is not admitting every value of it: a study that needs another mode names it in a preregistration and adds it here under review.",
    'the admitted set loses argmax':
        "AssertionError: assert ('opening_temperature',) == ('opening_tem...re', 'argmax')",
    'the readout is named argmax but the AGENT is built from the frozen config':
        "AssertionError: assert 'opening_temperature' == 'argmax'",
    'admitting the readout admits every other field too':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.twixtbot_g3_reference.ReferenceError'>",
    'the simulation budget may drift alongside the readout':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.twixtbot_g3_reference.ReferenceError'>",
    'fields are compared loosely, so 400 and 400.0 agree':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.twixtbot_g3_reference.ReferenceError'>",
    'any selection_mode string is admitted':
        "ValueError: unknown selection_mode 'hoeffding_lcb'",
    'a config of the wrong TYPE is accepted':
        "AttributeError: 'dict' object has no attribute 'board_size'",
    'the check is skipped entirely':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.twixtbot_g3_reference.ReferenceError'>",
    'omitting the config no longer falls back to the frozen one':
        'scripts.GPU.alphazero.twixtbot_g3_reference.ReferenceError: config is a NoneType, not the EvalConfig the frozen research configuration is expressed in',
    'the retry block un-registered from ACCOUNTED':
        "AssertionError: (202620000, {'accounted': False, 'exposed': True, 'retired': True, 'test_only': False})",
    'the WHOLE-BLOCK retirement is dropped':
        "AssertionError: (202620000, {'accounted': True, 'exposed': True, 'retired': False, 'test_only': False})",
    "the exposure is widened to the WHOLE block, claiming task 383's seed as drawn":
        "AssertionError: (202620383, {'accounted': True, 'exposed': True, 'retired': True, 'test_only': False})",
    'the 383 confirmed draws are not recorded as EXPOSED at all':
        "AssertionError: (202620000, {'accounted': True, 'exposed': False, 'retired': True, 'test_only': False})",
    "attempt 1's digest is overwritten, so its record stops verifying":
        "AssertionError: attempt 1's digest must still verify, or its record is unverifiable",
    "the seam's readout mode is not the rules' one":
        "AssertionError: assert 'opening_temperature' == 'argmax'",
    'our side is built with the WRONG colour for the arm':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.twixtbot_g3_reference.ReferenceError'>",
    # ── 2026-09-12: attempt 3's reservation and registration. Derived from each
    #    requirement and its assertion, then verified against a run.
    "attempt 3 reuses attempt 2's SPENT block":
        'assert (202620000, 202620736) == (202622000, 202622736)',
    "attempt 3's own digest is not re-pinned":
        "AssertionError: assert '4cec38c75a42...996078b94c332' == 'ef68cb9962d1...4cbe35c435c69'",
    "attempt 3's FULL-FIELD digest is not re-pinned":
        "AssertionError: assert '2d330ede735b...c36ee84a3e2da' == 'e513b479824c...54bb7bf830a65'",
    "attempt 2's digest is overwritten, so the INCIDENT's record stops verifying":
        "AssertionError: attempt 2's digest must still verify, or the incident's "
        "record is unverifiable",
    # a reservation is not a draw: ACCOUNTED yes, EXPOSED and RETIRED no
    # 🔴 CORRECTED 2026-09-13: the target inverted with the block, so the state
    # printed alongside `accounted: False` is now the SPENT one.
    'the third block un-registered from ACCOUNTED':
        "AssertionError: (202622000, {'accounted': False, 'exposed': True, "
        "'retired': True, 'test_only': False})",
    # ── the output destination, found by attempt 3's pre-run verification
    "the outputs point back into a SPENT attempt's directory":
        "AssertionError: ('docs/superpowers/evidence/2026-09-09-t1j-h2-deterministic-"
        "readout/03_h2_results.jsonl', 'docs/superpowers/evidence/2026-09-09-t1j-h2-"
        "deterministic-readout')",
    'the SPENT output directories are no longer named, so nothing is excluded':
        'AssertionError: vacuous: no spent directory is named',
    # ── 2026-09-13: attempt 3's block after the VOID -- 693 drawn, retired whole
    "the third block's exposure is dropped -- 693 draws unrecorded":
        "AssertionError: (202622000, {'accounted': True, 'exposed': False, "
        "'retired': True, 'test_only': False})",
    # the boundary seed: 202622693 is the FIRST UNDRAWN one, so widening the
    # exposure to the whole block claims a draw the record does not support
    "the third block's exposure is widened to the WHOLE block":
        "AssertionError: (202622693, {'accounted': True, 'exposed': True, "
        "'retired': True, 'test_only': False})",
    "the third block's WHOLE-BLOCK retirement is dropped":
        "AssertionError: (202622000, {'accounted': True, 'exposed': True, "
        "'retired': False, 'test_only': False})",
    'the H2 execution gate was left OPEN after the run':
        'assert True is False',
    'the spent third schedule is schedulable because availability is never checked':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.e4_screen_reference."
        "E4ReferenceError'>",
    # ── 2026-09-14: the H3 pilot -- gate, seedless barrier, containment,
    #    create-only outputs, the pinned schedule, generation, seeds, analysis
    'the H3 pilot gate is opened':
        'assert True is False',
    'the pilot entry no longer reads the gate first':
        'AssertionError: THE FIRST STATEMENT IN run_pilot IS A ImportFrom, NOT A CALL',
    "the pilot seam trusts the entry's gate check":
        "scripts.GPU.alphazero.h3_pilot_runner.H3PilotContainmentError: refusing to run the H3 pilot's production seam from a test process: scripts.GPU.alphazero.t1j_toolchain.verified_paths, scripts.GPU.alphazero.d1_probe._default_compile, scripts.GPU.alphazero.e4_screen_command._default_load_evaluator, scripts.GPU.alphazero.e4_screen_runner.play_task are still the REAL production function. Nothing was resolved, compiled, loaded, drawn or played. A test that reaches this point has removed an authorization check: on 2026-09-12 exactly that happened and 383 real games were played before it was killed.",
    "the pilot's containment boundary is removed":
        'AssertionError: assert_production_acts_are_inert',
    'the pilot checks the gate once instead of every game':
        'AssertionError: 1',
    "the pilot's create-only precheck follows symlinks":
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_pilot_runner.H3PilotRunError'>",
    'the pilot admits two outputs that are the same file':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_pilot_runner.H3PilotRunError'>",
    # 🔑 RE-OBSERVED after the target was repaired: it tampers a SEEDED schedule
    # in a field the registry ignores, so nothing but the digest can reject it and
    # removing the comparison leaves NOTHING raising at all.
    'the pilot schedule digest is not compared with the pin':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_pilot_runner.H3PilotRunError'>",
    "the pilot's tasks carry the wrong anchor for the incumbent's colour":
        "AssertionError: h3pilot-000-p00-inc_red: the anchor and the incumbent's colour disagree",
    "the pilot's openings are no longer deduplicated":
        "Failed: DID NOT RAISE <class 'RuntimeError'>",
    "the pilot reuses H1/H2's openings":
        "Failed: DID NOT RAISE <class 'RuntimeError'>",
    'terminal positions are admitted as pilot openings':
        "Failed: DID NOT RAISE <class 'RuntimeError'>",
    # 🔑 THE DECLARED REASON MUST NOT BE RANDOM. This control removes the
    # generator's seed, so the set it produces DIFFERS EVERY RUN -- the frozen
    # reason carried the digests of one particular random set and could never
    # match a second time. The target now asserts with a MESSAGE, which is the
    # stable part of the failure.
    'the opening generator ignores its seed, so it is not reproducible':
        'AssertionError: GENERATION IS NOT REPRODUCIBLE: the same constant gave a '
        'different set',
    'the pilot coerces its seed endpoints again':
        "TypeError: unsupported operand type(s) for -: 'int' and 'str'",
    'a spent block is schedulable because availability is never checked':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_pilot_rules.H3PilotError'>",
    'a duplicated task is counted twice instead of refused':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_pilot_analysis.H3AnalysisError'>",
    'an unexpected pair_id is admitted':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_pilot_analysis.H3AnalysisError'>",
    'duplicate pairs are counted as fresh evidence':
        'assert 0 == 1',
    'a NaN duration reaches the quantiles':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_pilot_analysis.H3AnalysisError'>",
    'a partial run declares its stop rules CLEAR':
        "AssertionError: assert 'CLEAR' == 'NOT FIRED -- UNDETERMINED'",
    # ── 2026-09-14: the H3 pilot's RUN BODY and WRAPPER
    'the pilot VOIDs on its deadline instead of reporting PARTIAL':
        'scripts.GPU.alphazero.h3_pilot_runner.H3PilotVoidError: deadline',
    'the pilot keeps playing past its deadline':
        'AssertionError: it stopped at the deadline, not after 40',
    'an exception is reported as a partial run rather than a VOID':
        "AssertionError: assert 'PARTIAL' == 'VOID'",
    'the durable transcript evidence is never written':
        "AssertionError: assert {'header', 'p... 'transcript'} <= {'header', 'p...'task_result'}",
    'the per-game duration is dropped from the record':
        'scripts.GPU.alphazero.h3_pilot_analysis.H3AnalysisError: h3pilot-000-p00-inc_red: no elapsed_s. Per-game timing is REQUIRED; without it the pilot cannot answer its own runtime question.',
    'the outputs are no longer create-only inside the run':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_pilot_runner.H3PilotVoidError'>",
    # 🔴 THE OLD REASON WAS NOT EVIDENCE ABOUT THE OPENINGS PIN AT ALL: it was the
    # NO-SEED-BLOCK refusal, which fired whether or not the check existed.
    # Registering a block moved the symptom to the containment boundary and the
    # harness reported it INDETERMINATE. The target now asserts WHICH refusal.
    # 🔑 RE-OBSERVED A THIRD TIME, 2026-09-15. This control's target keeps being
    # reached by a DIFFERENT refusal as the surrounding truth moves: first the
    # NO-SEED-BLOCK message (which was never evidence about the openings pin at
    # all), then the containment boundary once a block was registered, and now the
    # spent-block refusal once the pilot had run. The target asks WHICH refusal, so
    # each move changes the reason and none of them weakens the claim.
    'the generated openings are not checked against the frozen pin':
        'AssertionError: it ran PAST the openings pin and was stopped later, by H3PilotError',
    'the PILOT wrapper reports success without verifying the gate':
        'AssertionError: assert 0 == 10',
    'a surviving descendant is reported as a success':
        'AssertionError: assert 0 == 8',
    "the worker runs without the supervisor's capability":
        'AssertionError: assert 5 == 7',
    'a forged capability of the right length is accepted':
        'assert True is False',
    'the wrapper spawns before checking the outputs':
        'AssertionError: assert 0 == 7',
    'a fired stop rule is masked by the partial-run code':
        'AssertionError: assert 13 == 14',
    "the pilot's outputs point into a SPENT run's directory":
        "AssertionError: ('docs/superpowers/evidence/2026-09-12-t1j-h2-match-attempt3/03_pilot_results.jsonl', 'docs/superpowers/evidence/2026-09-12-t1j-h2-match-attempt3')",
    # ── 2026-09-14: the seam INTERFACE and the seeded pin
    'the harness is handed no recorder':
        "AttributeError: 'NoneType' object has no attribute 'emit'",
    "the seam returns the harness's flat result instead of the contract":
        "AssertionError: assert {'elapsed_s',...s', 'records'} >= {'elapsed_s',...ds', 'result'}",
    'two opening_bound records are accepted, so the ply span is unanchored':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_pilot_runner.H3PilotVoidError'>",
    'a seeded schedule is checked against the UNSEEDED pin':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_pilot_rules.H3PilotError'>",
    # 🔑 MORE CORRECT THAN THE REASON IT REPLACES. The old one was the
    # "SEEDED_TASK_DIGEST is None" message, recorded while no seeded pin existed --
    # so the control was demonstrating the missing pin, not the missing guard. With
    # the pin set, removing the guard hands a half-seeded schedule the seeded pin
    # in silence, and the target sees exactly that: nothing raised.
    'a half-seeded schedule is admitted':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_pilot_rules.H3PilotError'>",
    'the runner ignores which pin applies':
        "scripts.GPU.alphazero.h3_pilot_runner.H3PilotRunError: the schedule digest is 0ed4514ad6b324d18ce7a2dc8959b24bcf1a45515de274c8ef5840cc03b87a32 but the frozen pilot schedule is b972ce46beb637d45a56fe3b002eb88932652fc5134b5db3dbbf6702de192337. A different schedule is a different experiment wearing this one's name.",
    # ── 2026-09-14: the pilot's SEED REGISTRATION -- bookkeeping, not permission
    'the H3 pilot names a block other than the registered one':
        'assert (777000000, 777000040) == (202624000, 202624040)',
    'the NO-BLOCK refusal is removed, so an unnamed block runs':
        'TypeError: cannot unpack non-iterable NoneType object',
    'the pilot block un-registered from ACCOUNTED':
        'AssertionError: nothing was stripped: the block is NOT registered, so this controls nothing',
    'the registered block overlaps a spent one':
        'AssertionError: H2 a3',
    # ── 2026-09-14: the IDENTITY FIELDS the qualified construction READS, and
    # the two checks H3's `check_schedule` was missing beside H2's.
    "the pilot's tasks carry no reference identity":
        "scripts.GPU.alphazero.e4_screen_reference.E4ReferenceError: task is missing ['reference', 'reference_sha1']; build_reference_agent would refuse it",
    'the task names no reference_colour, so the factory cannot route':
        "KeyError: 'reference_colour'",
    "the reference colour is the ANCHOR's, so the sides are swapped":
        'AssertionError: h3pilot-000-p00-inc_red: reference_colour is not OUR side',
    'the pilot never asks the registry whether the seeds may RUN':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_pilot_runner.H3PilotRunError'>",
    # 🔑 THE OBSERVED FAILURE IS THE CRASH THE GUARD PREVENTS, not a quiet
    # admission: with the guard off, the seedless schedule reaches
    # `validate_schedule_executable`, which does `int(task["seed"])` on None. The
    # test names an `H3PilotRunError` refusal and gets a TypeError instead, which
    # is precisely the difference between a refusal and an accident.
    "a failed gate restoration does not become the wrapper's exit code":
        "AssertionError: a failed restoration SUPERSEDES the worker's own exit 0",
    'the generation wrapper spawns before checking the destination':
        'assert 0 == 7',
    # ── 2026-09-16: the CONTEXT RESET that attempt 1's VOID demanded, and
    # the spent range and destination it left behind.
    # ── 2026-09-16: the preregistered GENERATION deadline and the
    # PARENT-OWNED LAUNCH RECEIPT. Every reason observed, never predicted.
    'a refusal before spawning leaves no receipt':
        "TypeError: 'NoneType' object is not subscriptable",
    'no launch receipt is written at all':
        'assert None is not None',
    "the generation wrapper borrows the match segment's deadline":
        "AssertionError: the generation wrapper must not borrow the match segment's cap",
    'the launch receipt is not create-only':
        "AssertionError: the earlier launch's receipt was OVERWRITTEN",
    'the receipt is not part of the destination check':
        'assert 4 == 7',
    # ── 2026-09-15: the SUPERVISED GENERATION LAUNCH PATH and the
    # generator's terminal semantics. Every reason observed, never predicted.
    'a surviving descendant of the generator is reported as success':
        'assert 0 == 8',
    'the generation destination may sit in a SPENT directory':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_coproduced_generator_retired.H3GenerationError'>",
    "the generation worker runs without the supervisor's capability":
        'AssertionError: assert 5 == 7',
    # ── 2026-09-15: PREPARATION for the opening-generation run.
    # Every reason observed in a throwaway worktree, never predicted.
    # ── 2026-09-15: the H3 FULL STUDY. Every reason observed in a throwaway
    # worktree through the driver's own classify path, never predicted.
    'a VOIDed segment is treated as a clean exclusion':
        "assert True is False",
    'a duplicate pair is tolerated instead of being a harness fault':
        "assert (7 == 0)",
    'a record without its SEED is admitted':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_analysis.H3StudyAnalysisError'>",
    'a seam declaring no config is given one anyway':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_runner.H3StudyRunError'>",
    'a segment never asks the registry whether its seeds may RUN':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_runner.H3StudyRunError'>",
    'a segment schedule is not compared with its own digest':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_runner.H3StudyRunError'>",
    'a segment withheld after inspecting outcomes still permits a verdict':
        "assert True is False",
    'a sensitivity DECISIVE the other way is allowed through':
        "assert 'opposite' in \"the cap_free sensitivity's estimate lies on the other side of parity from the primary\"",
    'a sensitivity sitting EXACTLY on parity is allowed through':
        "AssertionError: the cap_free sensitivity's estimate lies on the other side of parity from the primary",
    'an UNCOMPUTABLE sensitivity is treated as a passed check':
        "AssertionError: assert 'comput' in 'the cap_free sensitivity has no mean'",
    'caps are EXCLUDED from the primary instead of scoring 0.5':
        "AssertionError: 1 + 0.5, over two",
    'exhausting the attempts yields a SHORT set instead of aborting':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_rules.H3StudyError'>",
    'segments share one output directory':
        "AssertionError: assert 2 == 4",
    'the create-only precheck follows symlinks':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_study_runner.H3StudyRunError'>",
    'the interval is described as PROVEN rather than nominal':
        "AssertionError: assert ('nominal' in 'the pairs are independent by construction. rejection sampling conditions the population jointly, the strata are fixed...stic. what the pairing buys is narrower and real: the two games of a pair are not counted as",
    'the opening is dropped from the GAME identity':
        "AssertionError: same continuation, different openings -- different games",
    'the opening is dropped from the pair identity':
        "assert 1 == 0",
    'the sample size is rounded DOWN and misses its own target':
        "AssertionError: h=0.08002689927666005 exceeds the declared target",
    "the study population no longer excludes the PILOT's openings":
        "AssertionError: 8",
    "the study seam trusts the entry's gate check":
        "scripts.GPU.alphazero.h3_study_runner.H3StudyContainmentError: refusing to run H3 study's production seam from a test process: scripts.GPU.alphazero.t1j_toolchain.verified_paths, scripts.GPU.alphazero.d1_probe._default_compile, scripts.GPU.alphazero.",
    # ── 2026-09-15: RE-DECLARED BY THE RUN. Four labels changed when the run
    # inverted what they claim, so their old reasons were removed in the same edit
    # rather than left to become orphans.
    'the 40 confirmed draws are not recorded as EXPOSED at all':
        "AssertionError: (202624000, {'accounted': True, 'exposed': False, 'retired': True, 'test_only': False})",
    "the PILOT block's whole-block retirement is dropped":
        "AssertionError: (202624000, {'accounted': True, 'exposed': True, 'retired': False, 'test_only': False})",
    'the pilot gate is left OPEN after the run':
        'AssertionError: the wrapper must restore the gate on every exit path',
    'the seeded pin no longer matches the digest the run recorded':
        "AssertionError: assert 'aa527cc9a1a7...3ce4ddfabfba9' == 'b972ce46beb6...f6702de192337'",
    # ── 2026-09-14: THE OBJECT, not an equal one.
    # 🔑 The third reason is the whole point in one line: with the seam building
    # its own, the captured config COMPARED EQUAL and only an `is` caught it.
    "the identity is read off a FRESH config, not the seam's":
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_pilot_runner.H3PilotRunError'>",
    'a seam that declares no config is given one anyway':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_pilot_runner.H3PilotRunError'>",
    'the seam builds a config instead of using the one it was given':
        'AssertionError: the builder must receive THE object the identity was read off, not an equal one',
    'the pilot entry builds the config twice':
        'AssertionError: the config was constructed 2 times, not once',
    # ── 2026-09-14: the INCUMBENT'S IDENTITY. Three share a DID NOT RAISE, and
    # that is not a weakness: `classify` binds a reason to the NAMED node, so each
    # is the absence of a refusal in a different test.
    'the pilot never checks the recorded identity':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_pilot_runner.H3PilotRunError'>",
    'the identity is checked only when one is SUPPLIED':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_pilot_runner.H3PilotRunError'>",
    'the durable header records no identity':
        "KeyError: 'identity'",
    'the config object is never compared with the qualified path':
        "Failed: DID NOT RAISE <class 'scripts.GPU.alphazero.h3_pilot_runner.H3PilotRunError'>",
    'a seedless schedule is admitted, so it bypasses the registry':
        "TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'",
}
