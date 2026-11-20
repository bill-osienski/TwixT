#!/bin/bash
# H-MX `htb_mix_export run export-2` -- OUTER WRAPPER, EXPORT ATTEMPT 2 (export amendment 1,
# docs/superpowers/2026-11-20-htb-mix-export-attempt-2-amendment.md §2-§3). Fixed ceiling: TERM at 3590 s, KILL 10 s
# later, 3600 s in all; cleanup observation bounded through 3610 s. Idle waiting before launch (idle_gate)
# is NOT part of the ceiling. Runs the card's §8 export and parity of the verified challenger (entry 74) into fresh,
# create-only evidence; attempt 1 (entry 75, VOID) stays as written. MLX on the CPU.
# BUILT ONLY (the user's amendment-and-build approval, 2026-11-20): launching it is a separate approval. No
# automatic retry; a run the ceiling stops is VOID and nothing from it is used.
# DERIVED by tests/htb_mix_export_2_wrapper_spec.py from docs/superpowers/evidence/htb-mix-export-run/01_wrapper.sh (its text from
# `set -u` on), with exact-once substitutions only (the two evidence paths and the subcommand, run export-2) and
# one inserted block, the Node dependency preflight before the idle gate. The ceiling, the gate, the module, the
# record kind, the input checked (the current student file), the idle gate, the background-compute check, the
# perl process-group supervisor, pgid.txt, survivors, the records, outer.json and the gate restoration are the
# parent's, unchanged.
set -u
OUT_REL=docs/superpowers/evidence/htb-mix-export-2-run
RUN_REL=docs/superpowers/evidence/htb-mix-export-2
GATE_REL=scripts/GPU/alphazero/htb_mix_export_authorization.py
GATE=HTB_MIX_EXPORT_AUTHORIZED
MODULE=scripts.GPU.alphazero.htb_mix_export
TERM_S=3590
GRACE_S=10

W=$(cd "$(dirname "$0")/../../../.." && pwd -P) || exit 1
cd "$W" || exit 1
status=1; inner=""; outcome="NOT_STARTED"; started=false; PRE=""; HEAD_C=""; PY=""; CKPT=""; start=0
RECORD_KIND=htb-mix-export-record/1

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
CKPT="$HOME/twixt-cache/htb-pilot/student.safetensors"     # the current student (card §3); no checkpoint
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
# THE NODE DEPENDENCY (export attempt-2 amendment §3), after the one-shot checks and before the idle gate: a real
# node_modules in THIS tree, from which Node (resolving as server/inference.js does) finds onnxruntime-node at
# exactly $W/node_modules and at version 1.23.2. A miss refuses: nothing is written or launched.
NODE=$(command -v node) || refuse "no node on PATH"
[ -d node_modules ] && [ ! -L node_modules ] || refuse "node_modules is not a real directory in this tree (clone it before the gate opens: amendment §3)"
ORT_P=$("$NODE" -e 'console.log(require("module").createRequire(process.argv[1]).resolve("onnxruntime-node/package.json"))' "$W/server/inference.js" 2>/dev/null) ||
  refuse "Node cannot resolve onnxruntime-node from this tree"
[ "$ORT_P" = "$W/node_modules/onnxruntime-node/package.json" ] || refuse "onnxruntime-node resolves to $ORT_P, not this tree's node_modules"
ORT_V=$("$NODE" -e 'console.log(require(process.argv[1]).version)' "$ORT_P" 2>/dev/null) || refuse "cannot read the onnxruntime-node version"
[ "$ORT_V" = "1.23.2" ] || refuse "onnxruntime-node $ORT_V, the amendment pins 1.23.2"
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
/usr/bin/perl -e "$SUPERVISOR" "$OUT_REL/pgid.txt" "$TERM_S" "$GRACE_S" env MLX_ENABLE_TF32=0 PYTHONPATH=. "$PY" -B -m "$MODULE" run export-2 \
  > "$OUT_REL/run.stdout.txt" 2> "$OUT_REL/run.stderr.txt"
inner=$?
case "$inner" in
  0) outcome=COMPLETED ;;
  2) outcome=REFUSED ;;
  3) outcome=STOP ;;
  4) outcome=FAILED ;;
  5) outcome=INTERRUPTED ;;
  124|125|137) outcome=TIMEOUT ;;
  *) outcome=CRASHED ;;
esac
status=$inner
exit "$status"
