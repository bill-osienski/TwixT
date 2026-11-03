#!/bin/bash
# THE SECOND POLICY-IMPROVEMENT CYCLE: THE STRENGTH COMPARISON, 509 COLOUR-REVERSED PAIRS (1,018 GAMES) -- OUTER WRAPPER (amendment
# docs/superpowers/2026-11-02-combination-policy-second-cycle-amendment.md §6, §7; Python 3.14.10 by docs/superpowers/2026-11-02-combination-policy-second-cycle-generation-attempt2-amendment.md; the user's review 2026-11-02: head sha256 c7e96802... against the original proven
# combination, the search settings unchanged).
# THE CEILING HERE IS PROVISIONAL (TERM_S 51770): it is the first comparison's. The amendment prices this run from the NEW screen's
# completed wall time x 1,018 / 64 x 1.5, rounded up to the minute, so TERM_S must be re-derived by a reviewed edit once that screen
# exists, before any launch approval; tests/test_combination_policy_cycle2.py fails until the wrapper matches. The frozen rule (PASS
# iff the mean pair score - sqrt(ln 40 / 1018) > 0.5 and the evaluations ratio <= 1.05 and the search-seconds ratio <= 1.10) is
# unchanged. A run the ceiling stops is incomplete and receives no score, with no automatic retry.
# BUILT ONLY (ledger, 2026-11-02): launching it is a separate approval.
# A copy of combination-policy-strength-run/01_wrapper.sh (the first cycle's, whose header follows); the diff: the paths, the gate,
# `run cycle2_strength` and this header. The value of TERM_S and everything else are byte-identical.
#
# THE COMBINATION POLICY PILOT: THE STRENGTH COMPARISON, 509 COLOUR-REVERSED PAIRS (1,018 GAMES, CARD STEP 6) -- OUTER
# WRAPPER (card docs/superpowers/2026-10-28-combination-policy-pilot-card.md §5 step 6; the user's review 2026-11-01:
# the same trained head against the same frozen proven combination, the search settings unchanged). CEILING SET BY THE
# USER BEFORE LAUNCH (the ledger's approval addendum, 2026-11-01), from the screen's completed wall time: 2,169 s x
# 1,018 / 64 x 1.5 = 51,751.0 s, rounded up to the minute: 51,780 s (14 h 23 min) -- TERM at 51770 s, KILL 10 s later,
# 51780 s in all; cleanup observation bounded through 51790 s. An operational estimate, not a completion guarantee. A
# run the ceiling stops is VOID, with no partial score and no automatic retry; its progress and cause go to the user
# (card §5 step 6: no interim look, extension, top-up or retry). The frozen rule (PASS iff the mean pair score -
# sqrt(ln 40 / 1018) > 0.5 and the evaluations ratio <= 1.05 and the search-seconds ratio <= 1.10), the checkpoint, the
# trained head and admission are the pilot's.
# BUILT ONLY (ledger, 2026-11-01): launching it is a separate approval.
# A copy of combination-policy-screen-run/01_wrapper.sh (the screen's, whose header follows); the diff: the paths, the
# gate, `run strength`, the ceiling (TERM_S 51770) and this header. Everything else is byte-identical: the idle gate,
# the background-compute check, the process-group supervisor and its bounded observation, the survivors, the records
# and the gate restoration. Read the copied headers' 64 games as this comparison's 1,018, and the screen's ceiling and
# rule as the screen's own.
#
# THE COMBINATION POLICY PILOT: THE SCREEN, 32 COLOUR-REVERSED PAIRS (64 GAMES, CARD STEP 5) -- OUTER WRAPPER (card
# docs/superpowers/2026-10-28-combination-policy-pilot-card.md §5 step 5; the user's review 2026-10-31: the trained head
# against the frozen proven combination, both averaged, 200 simulations, halving from ply 20). CEILING PROPOSED FOR
# REVIEW (the ledger's build addendum, 2026-10-31), priced from completed match timings because the card's throughput
# check never completed: 0.5508 s per combination search (strength2's 29,453 candidate searches: the same loop, loader,
# machine and TF32=0) x 64.75 plies per game (generation's 1,000 games of this agent against itself) x 64 games =
# 2,282.3 s; x 1.5, rounded up to the minute: 3,480 s -- TERM at 3470 s, KILL 10 s later, 3480 s in all; cleanup
# observation bounded through 3490 s. Long capped games: a 280-ply game at the slowest per-search rate seen in any
# capped match game (0.6285 s) takes about 176 s; with the rest at generation's uncapped mean (55.1 plies), TERM leaves
# room for 10 capped games of the 64, against 2.75 expected at generation's 4.3% rate. A run the ceiling stops is VOID,
# with no automatic retry; its progress and cause go to the user. The frozen rule (STOP below a mean pair score of
# 0.40, otherwise CONTINUE; no strength claim), the checkpoint, the trained head and admission are the pilot's.
# BUILT ONLY (ledger, 2026-10-31): launching it is a separate approval.
# A copy of combination-policy-train-run/01_wrapper.sh (the training's, whose header follows); the diff: the paths,
# the gate, `run screen`, the ceiling (TERM_S 3470) and this header. Everything else is byte-identical: the idle gate,
# the background-compute check, the process-group supervisor and its bounded observation, the survivors, the records
# and the gate restoration. Read the copied headers' runs as this screen's 64 games, and their checkpoint as the one
# both sides load (the candidate adds the trained head inside the script).
#
# THE COMBINATION POLICY PILOT: CPU TRAINING, THEN THE CHECK (CARD STEP 4) -- OUTER WRAPPER (card
# docs/superpowers/2026-10-28-combination-policy-pilot-card.md §5 step 4; ceiling stated before approval in the
# ledger's decision note after entry 43 (2026-10-31): the timing record's formula on the real corpus counts (44,747
# rows, 4,158 held out) gives 1,865.3 s; x 1.5, rounded up to the minute: 2,820 s -- TERM at 2810 s, KILL 10 s later,
# 2820 s in all; cleanup observation bounded through 2830 s). The timing record's own 2,700 s, priced from the card's
# estimated counts, stays as written. A run the ceiling stops is VOID and closes the pilot (card §5). No retry,
# extension or top-up. The recipe, the starting checkpoint (calib020_0001), admission and the STOP rules are the
# pilot's, unchanged.
# BUILT ONLY (ledger, 2026-10-31): launching it is a separate approval.
# A copy of combination-policy-generate-run/01_wrapper.sh (corpus generation's, whose header follows); the diff: the
# paths, the gate, `run train`, the ceiling (TERM_S 2810) and this header. Everything else is byte-identical: the idle
# gate, the background-compute check, the process-group supervisor and its bounded observation, the survivors, the
# records and the gate restoration. Read the copied header's games as this run's training and its checkpoint as the
# parent the training starts from.
#
# THE COMBINATION POLICY PILOT: CORPUS GENERATION (1,000 SLOTS, CARD STEP 3) -- OUTER WRAPPER (amendment
# docs/superpowers/2026-10-30-combination-policy-generation-amendment.md to the card's §5 steps 2-3: no completed
# throughput check is required; ceiling six hours: TERM at 21590 s, KILL 10 s later, 21600 s in all; cleanup
# observation bounded through 21610 s). A run the ceiling stops is VOID and its games are not used; the user then
# assesses progress and the cost of continuing. No automatic retry, extension or top-up.
# BUILT ONLY (ledger, 2026-10-30): launching it is a separate approval.
# A copy of combination-policy-throughput2-run/01_wrapper.sh (the throughput check's attempt 2, whose header follows);
# the diff: the paths, the gate, `run generate`, the ceiling (TERM_S 21590) and this header. Everything else is
# byte-identical: the idle gate, the background-compute check, the process-group supervisor and its bounded
# observation, the survivors, the records and the gate restoration. Read the copied header's 144 games as this run's
# 1,000 and its "closes the pilot" as the amendment's assessment.
#
# THE COMBINATION POLICY PILOT: THE THROUGHPUT CHECK, ATTEMPT 2 -- THE ONE FRESH ATTEMPT (144 SLOTS) -- OUTER WRAPPER
# (amendment docs/superpowers/2026-10-29-combination-policy-throughput-attempt2-amendment.md to the card's §5 step 2;
# ceiling 30 minutes: TERM at 1790 s, KILL 10 s later, 1800 s in all; cleanup observation bounded through 1810 s).
# ALL 144 GAMES MUST FINISH INSIDE IT: a run the ceiling stops is VOID and closes the pilot again.
# BUILT ONLY (ledger, 2026-10-29): launching it is a separate approval.
# A copy of combination-policy-throughput-run/01_wrapper.sh (attempt 1's, whose header follows); the diff: the paths,
# the gate, `run throughput2`, the ceiling (TERM_S 1790) and this header. Everything else is byte-identical: the idle
# gate, the background-compute check, the process-group supervisor and its bounded observation, the survivors, the
# records and the gate restoration.
#
# THE COMBINATION POLICY PILOT: THE THROUGHPUT CHECK (144 SLOTS, CARD STEP 2) -- OUTER WRAPPER (card
# docs/superpowers/2026-10-28-combination-policy-pilot-card.md §5 step 2; ceiling 15 minutes: TERM at 890 s, KILL
# 10 s later, 900 s in all). ALL 144 GAMES MUST FINISH INSIDE IT: a run the ceiling stops is VOID and closes the pilot.
# BUILT ONLY (ledger, decision note after entry 40): launching it is a separate approval.
# A copy of mirror-halving-strength2-run/01_wrapper.sh (attempt 2's, whose header follows); the diff: the paths, the
# gate, the module and its `run throughput` argument, the record kind, the ceiling, this header, and THE RUN'S PROCESS
# GROUP: this run spawns 18 worker processes whose command lines do not name the module, so the supervisor writes the
# run's process group id, create-only, to pgid.txt here, and the survivors are the processes that name the module
# PLUS every process still in that group. A started run whose pgid.txt cannot be read records survivors as unknown
# (closing then refuses). If the supervisor cannot write pgid.txt it KILLs the group and exits 126 (CRASHED).
# AND THE ESCALATION OUTLIVES THE MAIN PROCESS (the user's review, 2026-10-29): once it has reaped the main process,
# the supervisor keeps its alarm armed until no process of the group survives, so a worker that ignores TERM is still
# KILLed at the ceiling; a group that outlives its main process past TERM_S is a TIMEOUT (124, or 137 if KILL was needed).
# AND THAT OBSERVATION IS BOUNDED (the user's review, 2026-10-29): the main process and the group are watched for at most
# GRACE_S seconds after KILL. If the group can still be signalled then, the supervisor stops waiting and exits 125:
# TIMEOUT with cleanup unconfirmed. The wrapper never waits past TERM_S + 2 x GRACE_S (910 s). Survivors read
# "unconfirmed cleanup: process group <pgid> <pids>" when the supervisor gave up (125) or the group can still be
# signalled once the supervisor has gone (a lost supervisor, or the wrapper interrupted); closing then refuses.
#
# MIRROR AVERAGING + ROOT SEQUENTIAL HALVING: THE STRENGTH COMPARISON, ATTEMPT 2 -- THE ONE FRESH ATTEMPT (509 PAIRS,
# 1,018 GAMES) -- OUTER WRAPPER (amendment docs/superpowers/2026-10-27-mirror-halving-strength-attempt2-amendment.md
# to the card's §5 T; the same ceiling, 55,601 s).
# A copy of mirror-halving-strength-run/01_wrapper.sh (attempt 1's, whose header follows); the diff: the paths, the
# gate, the `run strength2` argument, this header, and THE BACKGROUND-COMPUTE CHECK: after the idle gate and before
# anything is launched, `python -B -m scripts.GPU.alphazero.background_compute check` must exit 0 (no BOINC process;
# BOINC's podman VM exempt). BOINC detected, or a failed check, REFUSES (exit 1): the run never starts, nothing is
# written here, and the gate is restored. Its observation is kept, create-only, in bg_check.txt once the run starts.
# During the run the script itself repeats the check after every game (VOID on detection or failure).
#
# MIRROR AVERAGING + ROOT SEQUENTIAL HALVING: THE 509-PAIR STRENGTH COMPARISON (1,018 GAMES) -- OUTER WRAPPER (card
# docs/superpowers/2026-10-26-mirror-averaging-plus-root-halving-combination-card.md §5 T, §6; ceiling priced in the
# ledger's addendum to Post-closure entry 39 from C1's recorded projection x the frozen 1.5, rounded up: 55,601 s (15 h 26 min 41 s)).
# A copy of mirror-halving-c1-run/01_wrapper.sh; the diff: the paths, the gate, the module and its `run strength`
# argument, the record kind, the ceiling (TERM at 55591 s, KILL 10 s later), this header, and THE IDLE GATE:
# after every other precondition and before anything is launched, `python -B -m scripts.GPU.alphazero.idle_gate wait`
# must exit 0 (1-minute load below 3 and no process above 50% CPU, held 60 s in a row, within 30 minutes). Otherwise
# the wrapper REFUSES (exit 1): the run never starts, nothing is written here, and the gate is restored. Its samples
# are kept, create-only, in idle_check.txt once the run starts (ledger #93: a printed check is not a check).
# Runs `MLX_ENABLE_TF32=0 python -B -m scripts.GPU.alphazero.mirror_halving_match run strength --checkpoint
# <calib020_0001>` ONCE, under ONE ceiling: TERM to the run's whole process group at 55591 s, KILL 10 s later.
# The perl supervisor keeps its exit codes (124: the ceiling fired and TERM sufficed; 137: KILL was needed).
# INT/TERM/HUP to the wrapper are forwarded. MLX_ENABLE_TF32=0 is set for that process only; nothing is persisted.
# The gate is restored on EVERY exit (EXIT trap; INT/TERM/HUP -> exit 130) and read back against the pre-open
# commit HEAD~1. Refuses unless the tree is clean and HEAD is a one-line commit opening ONLY this gate.
# Run from anywhere; it works in the tree that holds it. Python and the checkpoint come from the main checkout
# (the parent of the shared git directory), by absolute path; nothing is installed or copied.
# Writes, create-only, in THIS directory: idle_check.txt, run.stdout.txt, run.stderr.txt, outer.json. The script
# owns docs/superpowers/evidence/mirror-halving-strength/ (create-only: it refuses if that exists).
# The script writes record.json itself on every exit it survives, SIGTERM included (VOID at step "terminated").
# If the run TIMED OUT, CRASHED or was INTERRUPTED and no record.json exists once no run process survives, THE
# WRAPPER writes record.json there, create-only, marked "written_by": "wrapper", status VOID, step timeout |
# crashed | interrupted. outer.json's "record_by" says who wrote it: script | wrapper | none.
#
#   inner exit -> outcome: 0 COMPLETED | 3 VOID | 4 REFUSED | 2 USAGE | 124, 125, 137 TIMEOUT | anything else CRASHED;
#                          a signal to the wrapper -> INTERRUPTED
#   outer exit: the inner exit; 1 if the wrapper refused before the run (the idle gate included); 130 if
#               interrupted; 5 if the gate was NOT restored, or a needed wrapper record could not be written
#               (overrides all).
set -u
OUT_REL=docs/superpowers/evidence/combination-policy-cycle2-strength-run
RUN_REL=docs/superpowers/evidence/combination-policy-cycle2-strength
GATE_REL=scripts/GPU/alphazero/combination_policy_cycle2_strength_authorization.py
GATE=COMBINATION_POLICY_CYCLE2_STRENGTH_AUTHORIZED
MODULE=scripts.GPU.alphazero.combination_policy_pilot
TERM_S=51770
GRACE_S=10

W=$(cd "$(dirname "$0")/../../../.." && pwd -P) || exit 1
cd "$W" || exit 1
status=1; inner=""; outcome="NOT_STARTED"; started=false; PRE=""; HEAD_C=""; PY=""; CKPT=""; start=0
RECORD_KIND=combination-policy-pilot-record/1

restore() {
  /usr/bin/sed -i '' "s/^${GATE} = True\$/${GATE} = False/" "$GATE_REL"
}

finish() {
  trap - EXIT INT TERM HUP
  restore
  restored=false
  if grep -qx "${GATE} = False" "$GATE_REL" && { [ -z "$PRE" ] || git diff --quiet "$PRE" -- "$GATE_REL"; }; then
    restored=true
  fi
  survivors=$(pgrep -f -- "-m ${MODULE} run" | tr '\n' ' ')
  if [ "$started" = true ]; then
    if PGID=$(cat "$OUT_REL/pgid.txt" 2>/dev/null) && [[ "$PGID" =~ ^[0-9]+$ ]]; then
      survivors=$( { printf '%s\n' $survivors; pgrep -g "$PGID"; } | grep -E '^[0-9]+$' | sort -un | tr '\n' ' ')
      if [ "$inner" = 125 ] || kill -0 -- "-$PGID" 2>/dev/null; then    # the supervisor gave up, or the group remains
        survivors="unconfirmed cleanup: process group $PGID ${survivors}"
      fi
    else
      survivors="unknown: no readable pgid.txt ${survivors}"
    fi
  fi
  final=$status
  record_by=none
  if [ "$started" = true ]; then
    if [ -f "$RUN_REL/record.json" ]; then
      record_by=script
    elif [ -z "$survivors" ]; then
      case "$outcome" in
        TIMEOUT) step=timeout ;;
        CRASHED) step=crashed ;;
        INTERRUPTED) step=interrupted ;;
        *) step="" ;;
      esac
      if [ -n "$step" ]; then
        mkdir -p "$RUN_REL" &&
          printf '{"kind": "%s", "written_by": "wrapper", "status": "VOID", "void": {"step": "%s", "detail": "inner exit %s; the script left no record.json (ceiling: TERM at %d s, KILL %d s later)"}, "inner_exit": %s, "term_s": %d, "grace_s": %d}\n' \
            "$RECORD_KIND" "$step" "${inner:-none}" "$TERM_S" "$GRACE_S" "${inner:-null}" "$TERM_S" "$GRACE_S" \
            > "$RUN_REL/record.json" && record_by=wrapper
        [ "$record_by" = wrapper ] || { echo "WRAPPER RECORD NOT WRITTEN: $RUN_REL/record.json" >&2; final=5; }
      fi
    fi
  fi
  if [ "$restored" != true ]; then
    echo "GATE NOT RESTORED: $GATE_REL" >&2
    final=5
  fi
  if [ "$started" = true ]; then
    printf '{"inner_exit": %s, "outcome": "%s", "outer_exit": %d, "gate_restored": %s, "survivors": "%s", "record_by": "%s", "wall_seconds": %d, "term_s": %d, "grace_s": %d, "tree": "%s", "head": "%s", "pre_open": "%s", "python": "%s", "checkpoint": "%s"}\n' \
      "${inner:-null}" "$outcome" "$final" "$restored" "$survivors" "$record_by" "$(( $(date +%s) - start ))" "$TERM_S" "$GRACE_S" \
      "$W" "$HEAD_C" "$PRE" "$PY" "$CKPT" > "$OUT_REL/outer.json" || final=5
  fi
  exit "$final"
}

refuse() {
  echo "REFUSED: $1" >&2
  status=1
  exit 1
}

trap finish EXIT
trap 'status=130; outcome=INTERRUPTED; exit 130' INT TERM HUP

PRE=$(git rev-parse --verify -q "HEAD~1^{commit}") || refuse "no pre-open commit HEAD~1"
HEAD_C=$(git rev-parse HEAD) || refuse "no HEAD"
MAIN=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd -P) || refuse "no main checkout"
PY="$MAIN/.venv/bin/python"
CKPT="$MAIN/checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"
[ -z "$(git status --porcelain)" ] || refuse "the tree is not clean"
[ "$(git diff --name-only "$PRE" HEAD)" = "$GATE_REL" ] || refuse "HEAD does not change only $GATE_REL"
[ "$(git diff -U0 "$PRE" HEAD | grep -E '^[-+]' | grep -vE '^(---|\+\+\+) ')" = \
  "$(printf -- '-%s = False\n+%s = True' "$GATE" "$GATE")" ] || refuse "HEAD is not the one-line opening of $GATE"
grep -qx "${GATE} = True" "$GATE_REL" || refuse "$GATE is not open"
for f in idle_check.txt bg_check.txt pgid.txt run.stdout.txt run.stderr.txt outer.json; do
  [ -e "$OUT_REL/$f" ] && refuse "$OUT_REL/$f exists (one-shot)"
done
[ -e "$RUN_REL" ] && refuse "$RUN_REL exists (one-shot)"
[ -x "$PY" ] || refuse "$PY is not executable"
[ -f "$CKPT" ] || refuse "$CKPT is not a file"
# THE IDLE GATE (last precondition): the launch below is reached ONLY if it exits 0.
IDLE_LOG=$(env PYTHONPATH=. "$PY" -B -m scripts.GPU.alphazero.idle_gate wait 2>&1) || {
  idle_rc=$?
  printf '%s\n' "$IDLE_LOG" >&2
  refuse "the machine did not become idle (idle_gate exit $idle_rc): nothing launched"
}
# THE BACKGROUND-COMPUTE CHECK (after the idle gate): the launch below is reached ONLY if it exits 0.
BG_LOG=$(env PYTHONPATH=. "$PY" -B -m scripts.GPU.alphazero.background_compute check 2>&1) || {
  bg_rc=$?
  printf '%s\n' "$BG_LOG" >&2
  refuse "BOINC detected or the check failed (background_compute exit $bg_rc): nothing launched"
}

SUPERVISOR='
use strict; use warnings; use Fcntl; use POSIX ":sys_wait_h";
my ($pgid_file, $term, $grace, @cmd) = @ARGV;
my $pid = fork();
die "fork: $!" unless defined $pid;
if ($pid == 0) { setpgrp(0, 0); exec { $cmd[0] } @cmd; exit 127; }
setpgrp($pid, $pid);
my $pf;
sysopen($pf, $pgid_file, O_WRONLY | O_CREAT | O_EXCL) && print($pf "$pid\n") && close($pf)
  or do { kill "KILL", -$pid; waitpid($pid, 0); exit 126; };
my ($termed, $killed, $gave_up, $timed_out) = (0, 0, 0, 0);
my $stop = sub { if (!$termed) { $termed = 1; kill "TERM", -$pid; alarm $grace; }
                 elsif (!$killed) { $killed = 1; kill "KILL", -$pid; alarm $grace; } else { $gave_up = 1; } };
$SIG{ALRM} = sub { $timed_out = 1 unless $termed; $stop->(); };
$SIG{$_} = $stop for qw(INT TERM HUP);
alarm $term;
my $st;
until ($gave_up) {                     # the ceiling stays armed while the run or any of its group survives, and at
  $st = $? if !defined $st && waitpid($pid, WNOHANG) == $pid;    # most GRACE_S s after KILL
  last if defined $st && !kill(0, -$pid);
  select(undef, undef, undef, 0.1);
}
alarm 0;
exit 125 if $gave_up;
exit($killed ? 137 : 124) if $timed_out;
exit(($st & 127) ? 128 + ($st & 127) : ($st >> 8));
'

set -C                                   # create-only redirections from here on
started=true
printf '%s\n' "$IDLE_LOG" > "$OUT_REL/idle_check.txt"
printf '%s\n' "$BG_LOG" > "$OUT_REL/bg_check.txt"
start=$(date +%s)
/usr/bin/perl -e "$SUPERVISOR" "$OUT_REL/pgid.txt" "$TERM_S" "$GRACE_S" env MLX_ENABLE_TF32=0 PYTHONPATH=. "$PY" -B -m "$MODULE" run cycle2_strength --checkpoint "$CKPT" \
  > "$OUT_REL/run.stdout.txt" 2> "$OUT_REL/run.stderr.txt"
inner=$?
case "$inner" in
  0) outcome=COMPLETED ;;
  3) outcome=VOID ;;
  4) outcome=REFUSED ;;
  2) outcome=USAGE ;;
  124|125|137) outcome=TIMEOUT ;;
  *) outcome=CRASHED ;;
esac
status=$inner
exit "$status"
