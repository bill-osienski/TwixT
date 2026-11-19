"""THE GATE INVENTORY, DERIVED FROM SOURCE — never a hand-kept list.

🔴 WHY THIS FILE EXISTS. Every report in this programme for weeks said "all EIGHT
gates are False". There were TEN. The two missing ones --
`LOWPLY_QUALIFICATION_AUTHORIZED` and `RUNTIME_REQUAL_AUTHORIZED` -- belong to
qualifications that have already run, so nobody thought about them; but a retired
gate is still a closed authorization constant in the source, and a count that
does not include it is wrong.

The defect is not the number. It is that the count came from a LIST TYPED BY HAND
in each checker. A hand-kept list cannot see a gate nobody remembered to add to
it, which is the one case the count exists to catch. So the inventory is
ENUMERATED FROM THE SOURCE by AST, and the expected total is asserted -- adding
or removing a gate must be a deliberate edit to `EXPECTED_GATES`, not a silent
drift.
"""
import ast
import pathlib
import re

import pytest

from scripts.GPU.alphazero import gate_inventory as INVENTORY

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

#: 🔑 THE INVENTORY IS THE ASSERTION. Editing this set is how a gate is added or
#: removed; nothing else may change the count silently.
EXPECTED_GATES = {
    "d1_probe.py": "D1_EXECUTION_AUTHORIZED",
    "e4_screen_command.py": "SCREEN_AUTHORIZED",
    "h1_viability_runner.py": "H1_EXECUTION_AUTHORIZED",
    "h2_match_runner.py": "H2_EXECUTION_AUTHORIZED",
    "h3_pilot_runner.py": "H3_PILOT_EXECUTION_AUTHORIZED",
    #: 🔴 AMENDMENT 3 SWAPPED THIS ONE, IT DID NOT DELETE IT. The generation
    #: EXECUTION gate went because engine-free generation runs nothing; the
    #: POPULATION-FREEZE barrier took its place, guarding the one act still
    #: irreversible -- declaring a population THE population. Ten either way,
    #: and the swap is a deliberate edit here rather than a silent drift.
    "h3_study_generator.py": "H3_POPULATION_FREEZE_AUTHORIZED",
    "h3_study_runner.py": "H3_STUDY_EXECUTION_AUTHORIZED",
    #: 🔴 THE ELEVENTH, ADDED 2026-09-20. Combining the four segments is the
    #: study's one interpretive act -- after it a number exists that people will
    #: quote -- so it takes its own reviewed edit rather than riding on the
    #: execution gate that authorized playing the games.
    "h3_combine.py": "H3_COMBINATION_AUTHORIZED",
    #: 🔴 THE TWELFTH, ADDED 2026-09-21. H4 §4A raw capability characterization.
    #: It runs no game and computes no score, but it LAUNCHES JVMS against the
    #: real T1j build, and anything that spawns the external engine takes its own
    #: reviewed authorization rather than riding on another stage's.
    "h4_4a_characterization.py": "H4_4A_CHARACTERIZATION_AUTHORIZED",
    #: 🔴 THE THIRTEENTH, ADDED 2026-09-21. The H4 repair qualification. Like
    #: §4A it runs no game, but it launches JVMS against the real T1j build
    #: THROUGH A BEHAVIOURALLY MODIFIED HELPER -- the opt-in MatchData
    #: injection -- so it takes its own reviewed authorization rather than
    #: riding on §4A's, which qualified a different helper.
    "h4_repair_qualification.py": "H4_REPAIR_QUALIFICATION_AUTHORIZED",
    "h4_4b_acceptance_qualification.py": "H4_4B_ACCEPTANCE_QUALIFICATION_AUTHORIZED",
    #: 🔴 MOVED 2026-09-23 (runner card §12.6.2), not added: it lived INSIDE the
    #: hashed `h4_runner.py`, so the one-line commit opening it would have changed
    #: the code hash every manifest binds. Its own single-purpose module now.
    "h4_pilot_authorization.py": "H4_PILOT_EXECUTION_AUTHORIZED",
    #: 🔴 THE SEVENTEENTH, ADDED 2026-09-23. The 3P-b production-path qualification:
    #: no game, but it compiles with the verified JDK and loads the checkpoint, and
    #: like the pilot gate it lives OUTSIDE the hashed module it guards.
    "h4_production_qualification_authorization.py": "H4_PRODUCTION_QUALIFICATION_AUTHORIZED",
    #: 🔴 THE SIXTEENTH, ADDED 2026-09-23. H4 confirmatory aggregation: the one act
    #: after which an H4 strength number exists. It runs no engine, but it READS
    #: OUTCOMES, so it takes its own reviewed edit -- not the pilot's execution gate
    #: and not the proceed/stop artifact, which it requires in addition.
    "h4_confirmatory_analysis.py": "H4_STUDY_AGGREGATION_AUTHORIZED",
    #: 🔴 THE EIGHTEENTH TO TWENTIETH, ADDED 2026-09-27 (H5 implementation card §1).
    #: H5 edits no H4 file, so it gets its OWN three doors -- segment execution,
    #: the EARLY/BINDING qualification, and confirmatory aggregation -- each in a
    #: single-declaration module outside the hashed code, and none of H4's opens
    #: any of them.
    "h5_study_authorization.py": "H5_STUDY_EXECUTION_AUTHORIZED",
    "h5_production_qualification_authorization.py": "H5_PRODUCTION_QUALIFICATION_AUTHORIZED",
    "h5_aggregation_authorization.py": "H5_STUDY_AGGREGATION_AUTHORIZED",
    #: 🔴 THE TWENTY-FIRST, ADDED 2026-10-02 (cap-completion diagnostic card §6 A1.5).
    #: EXPLORATORY: the freeze, Stage 1 (both rules engines) and Stage 2 (the unchanged
    #: agents) each run only under their own authorization with this door opened.
    "h5_cap_completion_authorization.py": "H5_CAP_COMPLETION_AUTHORIZED",
    #: 🔴 THE TWENTY-SECOND, ADDED 2026-10-02 (incumbent defence mechanism card §9 A1.7).
    #: EXPLORATORY: one final-tree observation run of the unchanged incumbent at five
    #: frozen positions, only under its own authorization with this door opened.
    "h5_incumbent_defence_authorization.py": "H5_INCUMBENT_DEFENCE_AUTHORIZED",
    #: 🔴 THE TWENTY-THIRD, ADDED 2026-10-03 (controlled one-factor study card §9 A1.6).
    #: EXPLORATORY: one run of the B-repro / B-seed / F / S arms at five frozen fixtures,
    #: only under its own authorization with this door opened.
    "h5_incumbent_defence_factors_authorization.py": "H5_INCUMBENT_DEFENCE_FACTORS_AUTHORIZED",
    #: 🔴 THE TWENTY-FOURTH, ADDED 2026-10-03 (incumbent tactics card §8, §10). EXPLORATORY:
    #: the corpus freeze and the single run, each only under its own authorization.
    "h5_incumbent_tactics_authorization.py": "H5_INCUMBENT_TACTICS_AUTHORIZED",
    "tactical_candidate_authorization.py": "TACTICAL_CANDIDATE_QUALIFICATION_AUTHORIZED",
    "candidate_match_authorization.py": "CANDIDATE_MATCH_AUTHORIZED",
    #: 🔴 THE TWENTY-SEVENTH, ADDED 2026-10-07 (tactical candidate card §19 A9.1). The pilot
    #: tooling -- post-segment verification, outcome-blind report, sizing -- writes the records
    #: that admit study segments and commit n, so each run takes its own reviewed authorization.
    "tactical_pilot_authorization.py": "TACTICAL_PILOT_ANALYSIS_AUTHORIZED",
    #: 🔴 THE TWENTY-EIGHTH, ADDED 2026-10-08 (external-engine learning card, design baseline ff1422fb).
    #: Stage 1 acquires and QUERIES the external twixtbot engine and plays twelve games against the
    #: incumbent, so it takes its own reviewed door, in a single-declaration module.
    "external_learning_authorization.py": "EXTERNAL_LEARNING_STAGE1_AUTHORIZED",
    #: 🔴 THE TWENTY-NINTH, ADDED 2026-10-09 (external-engine learning card, stage 2). Stage 2 TRAINS a
    #: copy of the incumbent's policy head and writes a candidate checkpoint, so it takes its own
    #: reviewed door, separate from stage 1's, in a single-declaration module.
    "external_learning_stage2_authorization.py": "EXTERNAL_LEARNING_STAGE2_AUTHORIZED",
    #: 🔴 THE THIRTIETH, ADDED 2026-10-09 (external-engine learning card, stage 3). Stage 3 runs the
    #: incumbent and the stage-2 candidate on the sealed held-out rows and then PLAYS 32 games between
    #: them, so it takes its own reviewed door, separate from stages 1 and 2, in a single-declaration module.
    "external_learning_stage3_authorization.py": "EXTERNAL_LEARNING_STAGE3_AUTHORIZED",
    #: 🔴 THE THIRTY-FIRST, ADDED 2026-10-21 (policy-head capacity planted-qualification card, rev 2, 28abdbfe).
    #: The qualification LOADS calib020_0001, computes encoder features and fits two arms to planted targets,
    #: so it takes its own reviewed door, in a single-declaration module.
    "policy_head_capacity_planted_authorization.py": "POLICY_HEAD_CAPACITY_PLANTED_AUTHORIZED",
    #: 🔴 THE THIRTY-SECOND, ADDED 2026-10-22 (search distillation D2, Stage 1 card). The measurement LOADS
    #: calib020_0001 and PLAYS two self-play games at 1,600 simulations through the serial capture driver, so it
    #: takes its own reviewed door, in a single-declaration module.
    "search_distillation_d2_measure_authorization.py": "SEARCH_DISTILLATION_D2_MEASURE_AUTHORIZED",
    #: 🔴 THE THIRTY-THIRD, ADDED 2026-10-22 (search distillation D2, Stage 2 card). Generation LOADS
    #: calib020_0001 and PLAYS 120 self-play games at 1,600 simulations, writing the training corpus, so it takes
    #: its own reviewed door, separate from Stage 1's, in a single-declaration module.
    "search_distillation_d2_generate_authorization.py": "SEARCH_DISTILLATION_D2_GENERATE_AUTHORIZED",
    #: 🔴 THE THIRTY-FOURTH, ADDED 2026-10-22 (search distillation D2, Stage 3 card). Training LOADS calib020_0001
    #: twice and TRAINS a student copy on the Stage 2 corpus, writing one candidate checkpoint, so it takes its own
    #: reviewed door, separate from Stages 1 and 2, in a single-declaration module.
    "search_distillation_d2_train_authorization.py": "SEARCH_DISTILLATION_D2_TRAIN_AUTHORIZED",
    #: 🔴 THE THIRTY-FIFTH, ADDED 2026-10-22 (search distillation D2, gradient-diagnostic card). The diagnostic
    #: LOADS calib020_0001 twice and REPLAYS Stage 3's 401 optimizer updates on the training games, writing no
    #: checkpoint, so it takes its own reviewed door, separate from Stages 1-3, in a single-declaration module.
    "search_distillation_d2_gradtrace_authorization.py": "SEARCH_DISTILLATION_D2_GRADTRACE_AUTHORIZED",
    #: 🔴 THE THIRTY-SIXTH, ADDED 2026-10-22 (search distillation D2, reproducibility card). The diagnostic LOADS
    #: calib020_0001 and REPLAYS Stage 3's updates 0-21 three times, writing no checkpoint, so it takes its own
    #: reviewed door, separate from Stages 1-3 and the gradient diagnostic, in a single-declaration module.
    "search_distillation_d2_gradrepeat_authorization.py": "SEARCH_DISTILLATION_D2_GRADREPEAT_AUTHORIZED",
    #: 🔴 THE THIRTY-SEVENTH, ADDED 2026-10-22 (search distillation D2, gradient-check card). The check LOADS
    #: calib020_0001, runs one optimizer prefix to reconstruct a state and compares gradients across devices and
    #: against a NumPy reference, writing no checkpoint, so it takes its own reviewed door, in a single-declaration
    #: module.
    "search_distillation_d2_gradcheck_authorization.py": "SEARCH_DISTILLATION_D2_GRADCHECK_AUTHORIZED",
    #: 🔴 THE THIRTY-EIGHTH, ADDED 2026-10-23 (search distillation D2, linear-reproducer card). The reproducer
    #: loads no network but runs GPU computations on the gradient check's saved fixtures under a frozen case list,
    #: so it takes its own reviewed door, in a single-declaration module.
    "search_distillation_d2_linrepro_authorization.py": "SEARCH_DISTILLATION_D2_LINREPRO_AUTHORIZED",
    #: 🔴 THE THIRTY-NINTH, ADDED 2026-10-23 (search distillation D2, synthetic row-count probe card). The probe
    #: loads no network and no fixture but runs GPU computations on synthetic operands under a frozen case list, so
    #: it takes its own reviewed door, in a single-declaration module.
    "search_distillation_d2_mlxrows_authorization.py": "SEARCH_DISTILLATION_D2_MLXROWS_AUTHORIZED",
    #: 🔴 THE FORTIETH, ADDED 2026-10-23 (search distillation D2, full-float32 setting validation card). A fresh
    #: process launched with MLX_ENABLE_TF32=0 runs the synthetic probe and the four saved linear-operation cases on
    #: the GPU under a frozen case list, so it takes its own reviewed door, in a single-declaration module.
    "search_distillation_d2_tf32off_authorization.py": "SEARCH_DISTILLATION_D2_TF32OFF_AUTHORIZED",
    #: 🔴 THE FORTY-FIRST, ADDED 2026-10-23 (search distillation D2, full-float32 setting path validation card). A
    #: fresh process launched with MLX_ENABLE_TF32=0 checks the policy head against NumPy and the full model's
    #: gradients at the incumbent and after 21 in-memory optimizer updates on the GPU, so it takes its own reviewed
    #: door, in a single-declaration module.
    "search_distillation_d2_tf32path_authorization.py": "SEARCH_DISTILLATION_D2_TF32PATH_AUTHORIZED",
    #: 🔴 THE FORTY-SECOND, ADDED 2026-10-23 (search distillation D2, convolution-reference diagnostic card). A fresh
    #: process launched with MLX_ENABLE_TF32=0 reproduces entry 32's two failing encoder gradients, captures their
    #: convolutions' operands on the CPU and GPU (fixtures outside the repository) and compares both backends against a
    #: NumPy float64 reference, so it takes its own reviewed door, in a single-declaration module.
    "search_distillation_d2_convref_authorization.py": "SEARCH_DISTILLATION_D2_CONVREF_AUTHORIZED",
    #: 🔴 THE FORTY-THIRD, ADDED 2026-10-23 (search distillation D2, encoder branch-trace card). A fresh process launched
    #: with MLX_ENABLE_TF32=0 traces the encoder's BatchNorms, residual additions and ReLUs on the CPU and GPU (fixtures
    #: outside the repository) and checks each local step in NumPy float64, so it takes its own reviewed door, in a
    #: single-declaration module.
    "search_distillation_d2_branchtrace_authorization.py": "SEARCH_DISTILLATION_D2_BRANCHTRACE_AUTHORIZED",
    #: 🔴 THE FORTY-FOURTH, ADDED 2026-10-23 (search distillation D2, 21-update diagnostic prefix card). A fresh
    #: process launched with MLX_ENABLE_TF32=0 runs two prefixes of 21 in-memory optimizer updates on the GPU and
    #: checks the updated state against a prospective criterion, so it takes its own reviewed door, in a
    #: single-declaration module.
    "search_distillation_d2_prefix_authorization.py": "SEARCH_DISTILLATION_D2_PREFIX_AUTHORIZED",
    #: 🔴 THE FORTY-FIFTH, ADDED 2026-10-23 (search distillation D2, corrected-configuration trial card §4). A fresh
    #: process launched with MLX_ENABLE_TF32=0 plays the trial's 12 fresh held-out self-play games on the GPU, consuming
    #: registered seeds, so it takes its own reviewed door, in a single-declaration module.
    "search_distillation_d2_heldout_authorization.py": "SEARCH_DISTILLATION_D2_HELDOUT_AUTHORIZED",
    #: 🔴 THE FORTY-SIXTH, ADDED 2026-10-23 (search distillation D2, corrected-configuration trial card §5). A fresh
    #: process launched with MLX_ENABLE_TF32=0 trains and saves ONE candidate at Stage 3's frozen recipe and judges it on
    #: the fresh held-out set, so it takes its own reviewed door, in a single-declaration module.
    "search_distillation_d2_trial_authorization.py": "SEARCH_DISTILLATION_D2_TRIAL_AUTHORIZED",
    #: 🔴 THE FORTY-SEVENTH, ADDED 2026-10-26 (mirror averaging + root halving combination card §5 C1). A fresh
    #: process launched with MLX_ENABLE_TF32=0 loads calib020_0001 and runs real GPU searches on a fixed panel of
    #: committed positions (a correctness check, two identity controls, the cost totals), so it takes its own reviewed
    #: door, in a single-declaration module.
    "mirror_halving_c1_authorization.py": "MIRROR_HALVING_C1_AUTHORIZED",
    #: 🔴 THE FORTY-EIGHTH, ADDED 2026-10-26 (the same card, §5 S). A fresh process launched with MLX_ENABLE_TF32=0
    #: plays the 32-pair gross-regression screen, consuming registered seeds, so it takes its own reviewed door, in a
    #: single-declaration module.
    "mirror_halving_screen_authorization.py": "MIRROR_HALVING_SCREEN_AUTHORIZED",
    #: 🔴 THE FORTY-NINTH, ADDED 2026-10-26 (the same card, §5 T). A fresh process launched with MLX_ENABLE_TF32=0
    #: plays the 509-pair strength comparison, consuming registered seeds, so it takes its own reviewed door, separate
    #: from the screen's, in a single-declaration module.
    "mirror_halving_strength_authorization.py": "MIRROR_HALVING_STRENGTH_AUTHORIZED",
    #: 🔴 THE FIFTIETH, ADDED 2026-10-27 (attempt-2 amendment to the same card's §5 T). The ONE fresh attempt at the
    #: strength comparison after attempt 1's BOINC-contended timeout: fresh seeds, the same run, BOINC kept out. It takes
    #: its own reviewed door, so opening it can never relaunch attempt 1's paths, in a single-declaration module.
    "mirror_halving_strength2_authorization.py": "MIRROR_HALVING_STRENGTH2_AUTHORIZED",
    #: 🔴 THE FIFTY-FIRST TO FIFTY-THIRD, ADDED 2026-10-28 (the combination policy pilot card, §5 steps 2-4). The
    #: throughput check and the corpus generation each load calib020_0001 on the GPU and consume a registered block;
    #: the training loads it weights-only and fits the residual policy branch on the CPU. Each takes its own reviewed
    #: door, in a single-declaration module.
    "combination_policy_throughput_authorization.py": "COMBINATION_POLICY_THROUGHPUT_AUTHORIZED",
    "combination_policy_generate_authorization.py": "COMBINATION_POLICY_GENERATE_AUTHORIZED",
    "combination_policy_train_authorization.py": "COMBINATION_POLICY_TRAIN_AUTHORIZED",
    #: 🔴 THE FIFTY-FOURTH, ADDED 2026-10-29 (the throughput attempt-2 amendment to the same card's §5 step 2). The ONE
    #: fresh throughput check after attempt 1's timeout: fresh seeds, the same 144 games, a 30-minute ceiling. It takes
    #: its own reviewed door, so opening it can never relaunch attempt 1's paths, in a single-declaration module.
    "combination_policy_throughput2_authorization.py": "COMBINATION_POLICY_THROUGHPUT2_AUTHORIZED",
    #: 🔴 THE FIFTY-FIFTH, ADDED 2026-10-31 (the same card's §5 step 5, on the user's review). The screen loads
    #: calib020_0001 twice on the GPU, installs the trained policy head on one copy and plays 64 games on a registered
    #: block. It takes its own reviewed door, in a single-declaration module.
    "combination_policy_screen_authorization.py": "COMBINATION_POLICY_SCREEN_AUTHORIZED",
    #: 🔴 THE FIFTY-SIXTH, ADDED 2026-11-01 (the same card's §5 step 6, on the user's review). The strength comparison
    #: loads calib020_0001 twice on the GPU, installs the trained policy head on one copy and plays 1,018 games on a
    #: registered block. It takes its own reviewed door, in a single-declaration module.
    "combination_policy_strength_authorization.py": "COMBINATION_POLICY_STRENGTH_AUTHORIZED",
    #: 🔴 THE FIFTY-SEVENTH AND FIFTY-EIGHTH, ADDED 2026-11-02 (the second policy-improvement cycle's frozen amendment,
    #: Build A, on the user's review). Generation loads calib020_0001 on the GPU with the first head as teacher and plays
    #: 1,000 games on a registered block; the fit trains from that head on the CPU. Each takes its own reviewed door, in
    #: a single-declaration module. Build B's two doors do not exist yet.
    "combination_policy_cycle2_generate_authorization.py": "COMBINATION_POLICY_CYCLE2_GENERATE_AUTHORIZED",
    "combination_policy_cycle2_train_authorization.py": "COMBINATION_POLICY_CYCLE2_TRAIN_AUTHORIZED",
    #: 🔴 THE FIFTY-NINTH, ADDED 2026-11-02 (the new generation attempt's amendment, the user's decision). The first attempt's
    #: door is spent (VOID at the version check, ledger entry 47); this attempt takes its own.
    "combination_policy_cycle2_generate2_authorization.py": "COMBINATION_POLICY_CYCLE2_GENERATE2_AUTHORIZED",
    #: 🔴 THE SIXTIETH AND SIXTY-FIRST, ADDED 2026-11-02 (Build B, on the user's review). Each loads calib020_0001 twice on the
    #: GPU, installs the head the ONE fit saved on one copy and plays the original proven combination on a registered block.
    "combination_policy_cycle2_screen_authorization.py": "COMBINATION_POLICY_CYCLE2_SCREEN_AUTHORIZED",
    "combination_policy_cycle2_strength_authorization.py": "COMBINATION_POLICY_CYCLE2_STRENGTH_AUTHORIZED",
    #: 🔴 THE SIXTY-SECOND AND SIXTY-THIRD, ADDED 2026-11-03 (the H-TB pilot card and build, on the user's decision).
    #: Labelling runs twixtbot's raw network on the CPU over C0 positions; the fit trains calib020_0001 whole on the
    #: CPU and scores F1-F3. Each takes its own reviewed door, in a single-declaration module.
    "htb_label_authorization.py": "HTB_LABEL_AUTHORIZED",
    "htb_fit_authorization.py": "HTB_FIT_AUTHORIZED",
    #: 🔴 THE SIXTY-FOURTH, ADDED 2026-11-04 (the H-TB timing build, on the user's review). One door for the two bounded
    #: timing runs (T-label in the teacher environment, T-fit on the CPU), opened for one wrapper run at a time.
    "htb_timing_authorization.py": "HTB_TIMING_AUTHORIZED",
    #: 🔴 THE SIXTY-FIFTH, ADDED 2026-11-06 (the H-TB screen build, on the user's review). The screen loads the WHOLE
    #: student and calib020_0001 on the GPU and plays the original proven combination on a registered block.
    "htb_screen_authorization.py": "HTB_SCREEN_AUTHORIZED",
    #: 🔴 THE SIXTY-SIXTH, ADDED 2026-11-06 (the H-TB confirmation-test build, on the user's review). The test loads
    #: the WHOLE student and calib020_0001 on the GPU and plays 110 fresh pairs against the original proven combination.
    "htb_strength_authorization.py": "HTB_STRENGTH_AUTHORIZED",
    #: 🔴 THE SIXTY-SEVENTH TO SEVENTIETH, ADDED 2026-11-14 (the H-OP model-free build, on the user's approval). One door
    #: per runtime stage: the 40-game generation smoke and the generation run (the student on the GPU research path),
    #: labelling (the teacher on the CPU, teacher venv) and the fit with its imitation gate (MLX on the CPU).
    "htb_onpolicy_gen_smoke_authorization.py": "HTB_ONPOLICY_GEN_SMOKE_AUTHORIZED",
    "htb_onpolicy_generate_authorization.py": "HTB_ONPOLICY_GENERATE_AUTHORIZED",
    "htb_onpolicy_label_authorization.py": "HTB_ONPOLICY_LABEL_AUTHORIZED",
    "htb_onpolicy_fit_authorization.py": "HTB_ONPOLICY_FIT_AUTHORIZED",
    #: 🔴 THE SEVENTY-FIRST, ADDED 2026-11-14 (the H-OP build): one export of an ELIGIBLE challenger, then parity.
    "htb_onpolicy_export_authorization.py": "HTB_ONPOLICY_EXPORT_AUTHORIZED",
    #: 🔴 THE SEVENTY-SECOND AND SEVENTY-THIRD, ADDED 2026-11-18 (the H-MX model-free build, on the user's approval).
    #: One door per Python runtime stage: the mixed fit with its one-time screens (MLX on the CPU), and one export of an
    #: ELIGIBLE challenger with parity. The two match doors are JavaScript (`mix_authorization.mjs`).
    "htb_mix_fit_authorization.py": "HTB_MIX_FIT_AUTHORIZED",
    "htb_mix_export_authorization.py": "HTB_MIX_EXPORT_AUTHORIZED",
    "l0_match_command.py": "L0_EXECUTION_AUTHORIZED",
    "lowply_qualification.py": "LOWPLY_QUALIFICATION_AUTHORIZED",
    "runtime_requalification.py": "RUNTIME_REQUAL_AUTHORIZED",
}


#: 🔴 IMPORTED, NEVER REIMPLEMENTED. This file used to carry its OWN copy of the
#: discovery walk. Two injected-defect controls -- blanking the AUTHORIZED match,
#: and scanning only the first statement of each module -- were NOT CAUGHT,
#: because breaking `gate_inventory` could not affect a test that never called
#: it. A test that reimplements its subject tests the copy.
discover_gates = INVENTORY.discover_gates


def test_THE_GATE_INVENTORY_IS_EXACTLY_WHAT_THE_SOURCE_CONTAINS():
    found = {f"{stem}.py": name for stem, name in discover_gates()}
    assert found == EXPECTED_GATES, (
        "the source and the declared inventory disagree; a gate was added or "
        "removed without updating EXPECTED_GATES\n"
        f"  only in source   : {set(found.items()) - set(EXPECTED_GATES.items())}\n"
        f"  only in inventory: {set(EXPECTED_GATES.items()) - set(found.items())}")


def test_THE_COUNT_IS_DERIVED_AND_ANY_REPORT_SAYING_OTHERWISE_IS_WRONG():
    #: 🔴 ONE TRIPWIRE, AND `EXPECTED_GATES` IS IT. The literal `10` sat here
    #: beside `len(EXPECTED_GATES)`, so adding the eleventh gate failed this on
    #: the number rather than on the inventory -- two pins for one fact, which is
    #: how seven, eight and ten were all claimed at once. Adding a gate is now a
    #: single deliberate edit, to the dict above.
    assert INVENTORY.gate_count() == len(discover_gates()) == len(EXPECTED_GATES), (
        "the derived inventory and EXPECTED_GATES must agree")
    assert len(EXPECTED_GATES) == 73, (
        "the total moved: add or remove the entry in EXPECTED_GATES above "
        "deliberately, then update this number in the same edit")


def test_EVERY_GATE_IN_THE_INVENTORY_IS_CLOSED():
    assert INVENTORY.open_gates() == [], (
        f"OPEN GATES IN THE SOURCE: {INVENTORY.open_gates()}")


def test_A_RETIRED_GATE_STILL_COUNTS():
    """The two that were missed, named so their absence cannot recur silently."""
    found = {name for _stem, name in discover_gates()}
    assert "LOWPLY_QUALIFICATION_AUTHORIZED" in found
    assert "RUNTIME_REQUAL_AUTHORIZED" in found


@pytest.mark.parametrize("nonsense", ["EXIT_UNAUTHORIZED = 5",
                                      "UNAUTHORIZED_EXIT = 'x'"])
def test_NON_BOOL_AUTHORIZED_NAMES_ARE_NOT_GATES(nonsense, tmp_path):
    """NEGATIVE CONTROL: exit codes named *UNAUTHORIZED* must not inflate the count."""
    tree = ast.parse(nonsense)
    node = tree.body[0]
    assert not (isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, bool)), nonsense


def _stale_count_claims(text):
    """Claim forms only -- `the EIGHT gates`, `all eight gates`, `EIGHT gates are`.

    🔴 NARROWED DELIBERATELY, AND THE CONTROL BELOW PROVES IT STILL BITES. The
    first version forbade any number adjacent to the word "gates" and flagged the
    sentence "Two gates were missed" in this very file's own explanation. A guard
    that cannot tell a COUNT CLAIM from prose about counts is too broad, and
    broadening is how a guard gets deleted rather than fixed. So it matches the
    three shapes a checker actually uses to ASSERT a total.
    """
    words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
             "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
             "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
             "twenty-one": 21, "twenty-two": 22, "twenty-three": 23,
             "twenty-four": 24, "twenty-five": 25, "twenty-six": 26, "twenty-seven": 27,
             "twenty-eight": 28, "twenty-nine": 29, "thirty": 30, "thirty-one": 31,
             "thirty-two": 32, "thirty-three": 33, "thirty-four": 34,
             "thirty-five": 35, "thirty-six": 36, "thirty-seven": 37,
             "thirty-eight": 38, "thirty-nine": 39, "forty": 40, "forty-one": 41,
             "forty-two": 42, "forty-three": 43, "forty-four": 44,
             "forty-five": 45, "forty-six": 46, "forty-seven": 47, "forty-eight": 48,
             "forty-nine": 49, "fifty": 50, "fifty-one": 51, "fifty-two": 52, "fifty-three": 53,
             "fifty-four": 54, "fifty-five": 55, "fifty-six": 56, "fifty-seven": 57, "fifty-eight": 58, "fifty-nine": 59, "sixty": 60, "sixty-one": 61,
             "sixty-two": 62, "sixty-three": 63, "sixty-four": 64, "sixty-five": 65,
             "sixty-six": 66, "sixty-seven": 67, "sixty-eight": 68, "sixty-nine": 69, "seventy": 70,
             "seventy-one": 71, "seventy-two": 72, "seventy-three": 73}
    n = len(EXPECTED_GATES)
    bad = []
    for word, value in words.items():
        if value == n:
            continue
        for claim in (f"the {word} gates", f"all {word} gates", f"{word} gates are"):
            for variant in (claim, claim.upper(),
                            claim.replace(word, word.upper()),
                            claim.replace(word, word.capitalize())):
                #: not preceded by a letter or hyphen: "two gates are" must not match
                #: inside "twenty-two gates are" (first possible at 22 gates)
                if re.search(r"(?<![A-Za-z-])" + re.escape(variant), text):
                    bad.append(f"'{variant}' but there are {n}")
    return bad


def test_NO_CHECKER_HARDCODES_A_GATE_COUNT_THAT_DISAGREES():
    """🔴 THE ORIGINAL DEFECT: three checkers, three hand-typed lists, three
    different totals -- seven, eight and ten -- and reports quoting whichever was
    nearest. A checker must DERIVE the count, never write it."""
    stale = []
    for path in sorted(SCRIPTS.rglob("*.py")):
        for claim in _stale_count_claims(path.read_text(encoding="utf-8")):
            stale.append(f"{path}: {claim}")
    assert not stale, "\n".join(stale)


def test_THE_STALE_COUNT_GUARD_IS_NOT_VACUOUS():
    """NEGATIVE CONTROL: the guard must reject the exact strings that were there.

    Without this, narrowing the guard until it matched nothing would look like a
    fix. These are the two real stale claims, verbatim from the source before the
    repair, plus prose that must NOT trip it.
    """
    assert _stale_count_claims("== the SEVEN gates ==")      # H2's, real
    assert _stale_count_claims("== the EIGHT gates ==")      # the pilot's, real
    assert _stale_count_claims("all eight gates stay False")  # the reports', real
    assert not _stale_count_claims("Two gates were missed")   # prose, not a claim
    assert _stale_count_claims("all ten gates are False")      # was right, now stale
    assert not _stale_count_claims("== the 11 gates ==")       # derived, correct
    assert _stale_count_claims("all eleven gates are False")   # was right, now stale
    assert _stale_count_claims("all twelve gates are False")   # was right, now stale
    assert _stale_count_claims("all thirteen gates are False")  # was right, now stale
    assert _stale_count_claims("all fourteen gates are False")  # was right, now stale
    assert _stale_count_claims("all fifteen gates are False")  # was right, now stale
    assert _stale_count_claims("all sixteen gates are False")  # was right, now stale
    assert _stale_count_claims("all seventeen gates are False")  # was right, now stale
    assert _stale_count_claims("all twenty gates are False")  # was right, now stale
    assert _stale_count_claims("all twenty-one gates are False")  # was right, now stale
    assert _stale_count_claims("all twenty-two gates are False")  # was right, now stale
    assert _stale_count_claims("all twenty-three gates are False")  # was right, now stale
    assert _stale_count_claims("all twenty-four gates are False")  # was right, now stale
    assert _stale_count_claims("all twenty-five gates are False")  # was right, now stale
    assert _stale_count_claims("all twenty-six gates are False")  # was right, now stale
    assert _stale_count_claims("all twenty-seven gates are False")  # was right, now stale
    assert _stale_count_claims("all twenty-eight gates are False")  # was right, now stale
    assert _stale_count_claims("all twenty-nine gates are False")  # was right, now stale
    assert _stale_count_claims("all thirty gates are False")  # was right, now stale
    assert _stale_count_claims("all thirty-one gates are False")  # was right, now stale
    assert _stale_count_claims("all thirty-two gates are False")  # was right, now stale
    assert _stale_count_claims("all thirty-three gates are False")  # was right, now stale
    assert _stale_count_claims("all thirty-four gates are False")  # was right, now stale
    assert _stale_count_claims("all thirty-five gates are False")  # was right, now stale
    assert _stale_count_claims("all thirty-six gates are False")  # was right, now stale
    assert _stale_count_claims("all thirty-seven gates are False")  # was right, now stale
    assert _stale_count_claims("all thirty-eight gates are False")  # was right, now stale
    assert _stale_count_claims("all thirty-nine gates are False")  # was right, now stale
    assert _stale_count_claims("all forty gates are False")  # was right, now stale
    assert _stale_count_claims("all forty-one gates are False")  # was right, now stale
    assert _stale_count_claims("all forty-two gates are False")  # was right, now stale
    assert _stale_count_claims("all forty-three gates are False")  # was right, now stale
    assert _stale_count_claims("all forty-four gates are False")  # was right, now stale
    assert _stale_count_claims("all forty-five gates are False")  # never the total; stale
    assert _stale_count_claims("all forty-six gates are False")  # was right, now stale
    assert _stale_count_claims("all forty-seven gates are False")  # never the total; stale
    assert _stale_count_claims("all forty-eight gates are False")  # never the total; stale
    assert _stale_count_claims("all forty-nine gates are False")  # was right, now stale
    assert _stale_count_claims("all fifty gates are False")  # was right, now stale
    assert _stale_count_claims("all fifty-one gates are False")  # never the total; stale
    assert _stale_count_claims("all fifty-two gates are False")  # never the total; stale
    assert _stale_count_claims("all fifty-three gates are False")  # was right, now stale
    assert _stale_count_claims("all fifty-four gates are False")  # was right, now stale
    assert _stale_count_claims("all fifty-five gates are False")  # was right, now stale
    assert _stale_count_claims("all fifty-six gates are False")  # was right, now stale
    assert _stale_count_claims("all fifty-seven gates are False")  # never the total; stale
    assert _stale_count_claims("all fifty-eight gates are False")  # was right, now stale
    assert _stale_count_claims("all fifty-nine gates are False")  # was right, now stale
    assert _stale_count_claims("all sixty-one gates are False")  # was right, now stale
    assert _stale_count_claims("all sixty-two gates are False")  # never the total; stale
    assert _stale_count_claims("all sixty-three gates are False")  # was right, now stale
    assert _stale_count_claims("all sixty-four gates are False")  # was right, now stale
    assert _stale_count_claims("all sixty-five gates are False")  # was right, now stale
    assert _stale_count_claims("all sixty-six gates are False")  # was right, now stale
    assert _stale_count_claims("all seventy gates are False")  # was right, now stale
    assert _stale_count_claims("all seventy-one gates are False")  # was right, now stale
    assert _stale_count_claims("all seventy-two gates are False")  # never the total; stale
    assert not _stale_count_claims("all seventy-three gates are False")  # correct total
