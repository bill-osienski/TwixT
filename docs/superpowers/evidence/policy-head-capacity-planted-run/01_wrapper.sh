#!/bin/bash
# POLICY-HEAD CAPACITY PLANTED QUALIFICATION -- OUTER WRAPPER (card §9 step 2; rev 2, 28abdbfe).
# Runs `python -B -m scripts.GPU.alphazero.policy_head_capacity_planted run --checkpoint <calib020_0001>` ONCE,
# under ONE ceiling: TERM to the run's whole process group at 890 s, KILL 10 s later -- 900 s in all, the card's
# 15 minutes. This machine has no `timeout`; the perl supervisor below does its job and keeps its exit codes
# (124: the ceiling fired and TERM sufficed; 137: KILL was needed). INT/TERM/HUP to the wrapper are forwarded.
# The gate is restored on EVERY exit (EXIT trap; INT/TERM/HUP -> exit 130) and read back against the pre-open
# commit HEAD~1. Refuses unless the tree is clean and HEAD is a one-line commit opening ONLY this gate.
# Run from anywhere; it works in the tree that holds it. Python and the checkpoint come from the main checkout
# (the parent of the shared git directory), by absolute path; nothing is installed or copied.
# Writes, create-only, in THIS directory: run.stdout.txt, run.stderr.txt, outer.json. The script owns
# docs/superpowers/evidence/policy-head-capacity-planted/ (create-only: it refuses if that exists).
#
#   inner exit -> outcome: 0 COMPLETED (verdict in the script's record.json) | 3 VOID | 4 REFUSED | 2 USAGE |
#                          124, 137 TIMEOUT | anything else CRASHED; a signal to the wrapper -> INTERRUPTED
#   outer exit: the inner exit; 1 if the wrapper refused before the run; 130 if interrupted;
#               5 if the gate was NOT restored (overrides every other status).
set -u
OUT_REL=docs/superpowers/evidence/policy-head-capacity-planted-run
RUN_REL=docs/superpowers/evidence/policy-head-capacity-planted
GATE_REL=scripts/GPU/alphazero/policy_head_capacity_planted_authorization.py
GATE=POLICY_HEAD_CAPACITY_PLANTED_AUTHORIZED
MODULE=scripts.GPU.alphazero.policy_head_capacity_planted
TERM_S=890
GRACE_S=10

W=$(cd "$(dirname "$0")/../../../.." && pwd -P) || exit 1
cd "$W" || exit 1
status=1; inner=""; outcome="NOT_STARTED"; started=false; PRE=""; HEAD_C=""; PY=""; CKPT=""; start=0

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
  final=$status
  if [ "$restored" != true ]; then
    echo "GATE NOT RESTORED: $GATE_REL" >&2
    final=5
  fi
  if [ "$started" = true ]; then
    printf '{"inner_exit": %s, "outcome": "%s", "outer_exit": %d, "gate_restored": %s, "survivors": "%s", "wall_seconds": %d, "term_s": %d, "grace_s": %d, "tree": "%s", "head": "%s", "pre_open": "%s", "python": "%s", "checkpoint": "%s"}\n' \
      "${inner:-null}" "$outcome" "$final" "$restored" "$survivors" "$(( $(date +%s) - start ))" "$TERM_S" "$GRACE_S" \
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
for f in run.stdout.txt run.stderr.txt outer.json; do
  [ -e "$OUT_REL/$f" ] && refuse "$OUT_REL/$f exists (one-shot)"
done
[ -e "$RUN_REL" ] && refuse "$RUN_REL exists (one-shot)"
[ -x "$PY" ] || refuse "$PY is not executable"
[ -f "$CKPT" ] || refuse "$CKPT is not a file"

SUPERVISOR='
use strict; use warnings;
my ($term, $grace, @cmd) = @ARGV;
my $pid = fork();
die "fork: $!" unless defined $pid;
if ($pid == 0) { setpgrp(0, 0); exec { $cmd[0] } @cmd; exit 127; }
setpgrp($pid, $pid);
my ($termed, $killed, $timed_out) = (0, 0, 0);
my $stop = sub { if (!$termed) { $termed = 1; kill "TERM", -$pid; alarm $grace; } else { $killed = 1; kill "KILL", -$pid; } };
$SIG{ALRM} = sub { $timed_out = 1 unless $termed; $stop->(); };
$SIG{$_} = $stop for qw(INT TERM HUP);
alarm $term;
my $w;
do { $w = waitpid($pid, 0); } while ($w == -1 && $!{EINTR});
my $st = $?;
alarm 0;
exit($killed ? 137 : 124) if $timed_out;
exit(($st & 127) ? 128 + ($st & 127) : ($st >> 8));
'

set -C                                   # create-only redirections from here on
started=true
start=$(date +%s)
/usr/bin/perl -e "$SUPERVISOR" "$TERM_S" "$GRACE_S" env PYTHONPATH=. "$PY" -B -m "$MODULE" run --checkpoint "$CKPT" \
  > "$OUT_REL/run.stdout.txt" 2> "$OUT_REL/run.stderr.txt"
inner=$?
case "$inner" in
  0) outcome=COMPLETED ;;
  3) outcome=VOID ;;
  4) outcome=REFUSED ;;
  2) outcome=USAGE ;;
  124|137) outcome=TIMEOUT ;;
  *) outcome=CRASHED ;;
esac
status=$inner
exit "$status"
