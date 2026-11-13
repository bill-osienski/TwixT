#!/bin/bash
# H-TB DIAGNOSTIC O (OPENING ATTRIBUTION) -- OUTER WRAPPER (plan docs/superpowers/2026-11-12-htb-diagnostic-o-plan.md §8).
# The user's approval (2026-11-12): one run on the 1,100 saved Black opening positions of the G3 match, within the
# proposed 3.2 h ceiling: TERM at 11510 s, KILL 10 s later, 11520 s in all; cleanup observation bounded through
# 11530 s. Idle waiting before launch (idle_gate) is NOT part of the ceiling. No games, no labelling, no training.
# No automatic retry: any refusal, stop, timeout or invalid evidence returns for assessment.
# A copy of docs/superpowers/evidence/htb-g3-match-run/01_wrapper.sh (the G3 match's); the diff: the paths, the gate
# (DIAG_O_AUTHORIZED), the driver and its `run` command, the record kind, a fixed ceiling instead of the smoke-derived
# one, the teacher interpreter and clone it checks and passes, the frozen analyser's sha256 check against the plan, an
# Unreal Editor check (refuse, never stop it), the teacher module counted among survivors, and this header. The idle
# gate, background-compute check, supervisor, survivors by process group, records and gate restoration are unchanged.
set -u
OUT_REL=docs/superpowers/evidence/htb-diag-o-run
RUN_REL=docs/superpowers/evidence/htb-diag-o
PLAN_REL=docs/superpowers/2026-11-12-htb-diagnostic-o-plan.md
ANALYSER=tests/htb_integration/diag_o_analyse.mjs
TEACHER_MOD=htb_diag_o_teacher
GATE_REL=tests/htb_integration/diag_o_authorization.mjs
GATE=DIAG_O_AUTHORIZED
DRIVER=tests/htb_integration/diag_o.mjs
STAGE=run
TERM_S=11510
KILL_S=11520
GRACE_S=10

W=$(cd "$(dirname "$0")/../../../.." && pwd -P) || exit 1
cd "$W" || exit 1
status=1; inner=""; outcome="NOT_STARTED"; started=false; PRE=""; HEAD_C=""; PY=""; NODE=""; start=0
RECORD_KIND=htb-diag-o-record/1

restore() {
  /usr/bin/sed -i '' "s/^export const ${GATE} = true;\$/export const ${GATE} = false;/" "$GATE_REL"
}

finish() {
  trap - EXIT INT TERM HUP
  restore
  restored=false
  if grep -qx "export const ${GATE} = false;" "$GATE_REL" && { [ -z "$PRE" ] || git diff --quiet "$PRE" -- "$GATE_REL"; }; then
    restored=true
  fi
  survivors=$( { pgrep -f -- "${DRIVER} ${STAGE}"; pgrep -f -- "$TEACHER_MOD"; } | sort -un | tr '\n' ' ')
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
          printf '{"schema": "%s", "written_by": "wrapper", "status": "VOID", "void": {"step": "%s", "detail": "inner exit %s; the script left no record.json (ceiling: TERM at %d s, KILL %d s later)"}, "inner_exit": %s, "term_s": %d, "grace_s": %d}\n' \
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
    printf '{"inner_exit": %s, "outcome": "%s", "outer_exit": %d, "gate_restored": %s, "survivors": "%s", "record_by": "%s", "wall_seconds": %d, "term_s": %d, "grace_s": %d, "tree": "%s", "head": "%s", "pre_open": "%s", "python": "%s", "node": "%s"}\n' \
      "${inner:-null}" "$outcome" "$final" "$restored" "$survivors" "$record_by" "$(( $(date +%s) - start ))" "$TERM_S" "$GRACE_S" \
      "$W" "$HEAD_C" "$PRE" "$PY" "$NODE" > "$OUT_REL/outer.json" || final=5
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
NODE=$(command -v node) || refuse "no node on PATH"
[ -z "$(git status --porcelain)" ] || refuse "the tree is not clean"
[ "$(git diff --name-only "$PRE" HEAD)" = "$GATE_REL" ] || refuse "HEAD does not change only $GATE_REL"
[ "$(git diff -U0 "$PRE" HEAD | grep -E '^[-+]' | grep -vE '^(---|\+\+\+) ')" = \
  "$(printf -- '-export const %s = false;\n+export const %s = true;' "$GATE" "$GATE")" ] || refuse "HEAD is not the one-line opening of $GATE"
grep -qx "export const ${GATE} = true;" "$GATE_REL" || refuse "$GATE is not open"
for f in idle_check.txt bg_check.txt pgid.txt run.stdout.txt run.stderr.txt outer.json; do
  [ -e "$OUT_REL/$f" ] && refuse "$OUT_REL/$f exists (one-shot)"
done
[ -e "$RUN_REL" ] && refuse "$RUN_REL exists (one-shot)"
[ -x "$PY" ] || refuse "$PY is not executable"
[ -d node_modules ] && [ ! -L node_modules ] || refuse "node_modules is not a real directory in this tree"
# THE CEILING, from the committed smoke timing, re-derived from the smoke's evidence by the analyser.
TPY="$HOME/twixt-cache/htb-teacher/teacher-venv/bin/python"
CLONE="$HOME/twixt-cache/htb-teacher/twixtbot-ui"
[ -x "$TPY" ] || refuse "$TPY (the teacher venv) is not executable"
[ -d "$CLONE" ] || refuse "$CLONE (the pinned teacher clone) is missing"
git ls-files --error-unmatch "$PLAN_REL" > /dev/null 2>&1 || refuse "$PLAN_REL is not committed"
WANT=$(grep -oE 'diag_o_analyse.mjs sha256 `[0-9a-f]{64}`' "$PLAN_REL" | grep -oE '[0-9a-f]{64}') || refuse "the plan records no analyser sha256"
[ "$(shasum -a 256 "$ANALYSER" | cut -d' ' -f1)" = "$WANT" ] || refuse "$ANALYSER is not the analysis frozen in the plan"
pgrep -x UnrealEditor > /dev/null && refuse "Unreal Editor is running (the user keeps it closed during runs; not stopped here)"
[ $(( TERM_S + GRACE_S )) -eq "$KILL_S" ] || refuse "TERM_S + GRACE_S is not KILL_S"
# THE IDLE GATE (last precondition but one): the launch below is reached ONLY if it exits 0.
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
/usr/bin/perl -e "$SUPERVISOR" "$OUT_REL/pgid.txt" "$TERM_S" "$GRACE_S" env -u MODEL_MANIFEST -u HARD_MODEL_MANIFEST -u HARD_MODE "$NODE" "$DRIVER" "$STAGE" --teacher-python "$TPY" --clone-root "$CLONE" --kill-s "$KILL_S" \
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
