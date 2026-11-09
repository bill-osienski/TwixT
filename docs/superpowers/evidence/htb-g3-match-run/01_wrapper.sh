#!/bin/bash
# H-TB INTEGRATION G3, V8b CONFIRMATION MATCH: 110 FRESH COLOUR-REVERSED PAIRS (220 GAMES), NEW HARD AGAINST TODAY'S
# HARD, ON THE PRODUCT STACK -- OUTER WRAPPER (card docs/superpowers/2026-11-09-htb-g3-match-card.md §3-§8; the user's
# G2 review, 2026-11-09: PASS only if the mean pair score - sqrt(ln 20 / 220) > 0.5, under the product's game-end rule).
# CEILING DERIVED MECHANICALLY FROM THE SMOKE (card §5): KILL_S is the committed
# docs/superpowers/evidence/htb-g3-smoke/timing.json's kill_s (1.5 x the projection, rounded up to the minute), which
# this wrapper re-derives with g3_analyse.mjs and requires byte-equal and WITHIN_24H (kill_s <= 86400 s). TERM at
# KILL_S - 10 s; cleanup observation bounded through KILL_S + 10 s. A run the ceiling stops is VOID, with no automatic
# retry; its progress and cause go to the user. Idle waiting (idle_gate) is not part of the ceiling.
# BUILT ONLY (ledger, 2026-11-09): launching it is a separate approval (the G3 launch review), and only after the
# smoke has completed and its timing.json is committed.
# A copy of docs/superpowers/evidence/htb-g3-smoke-run/01_wrapper.sh (the smoke's); the diff: the paths, the gate
# (G3_MATCH_AUTHORIZED), the stage (`run match`, with `--kill-s`), and the ceiling, read from the committed smoke
# timing after re-deriving it, instead of a fixed TERM_S. Everything else is the same.
#
#   inner exit -> outcome: 0 COMPLETED | 3 VOID | 4 REFUSED | 2 USAGE | 124, 125, 137 TIMEOUT | anything else CRASHED;
#                          a signal to the wrapper -> INTERRUPTED
#   outer exit: the inner exit; 1 if the wrapper refused before the run (the idle gate included); 130 if
#               interrupted; 5 if the gate was NOT restored, or a needed wrapper record could not be written
#               (overrides all).
set -u
OUT_REL=docs/superpowers/evidence/htb-g3-match-run
RUN_REL=docs/superpowers/evidence/htb-g3-match
SMOKE_REL=docs/superpowers/evidence/htb-g3-smoke
TIMING_REL=$SMOKE_REL/timing.json
ANALYSER=tests/htb_integration/g3_analyse.mjs
GATE_REL=tests/htb_integration/g3_authorization.mjs
GATE=G3_MATCH_AUTHORIZED
DRIVER=tests/htb_integration/g3_match.mjs
STAGE=match
TERM_S=0
KILL_S=0
GRACE_S=10

W=$(cd "$(dirname "$0")/../../../.." && pwd -P) || exit 1
cd "$W" || exit 1
status=1; inner=""; outcome="NOT_STARTED"; started=false; PRE=""; HEAD_C=""; PY=""; NODE=""; start=0
RECORD_KIND=htb-g3-record/1

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
  survivors=$(pgrep -f -- "${DRIVER} run ${STAGE}" | tr '\n' ' ')
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
git ls-files --error-unmatch "$TIMING_REL" > /dev/null 2>&1 || refuse "$TIMING_REL is not committed"
RECOMPUTED=$("$NODE" "$ANALYSER" timing "$SMOKE_REL") || refuse "the smoke timing does not re-derive as WITHIN_24H"
[ "$RECOMPUTED" = "$(cat "$TIMING_REL")" ] || refuse "$TIMING_REL is not what the analyser derives from the smoke"
KILL_S=$("$NODE" -e 'const t = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"));
if (t.verdict !== "WITHIN_24H" || !Number.isInteger(t.kill_s) || t.kill_s <= 0 || t.kill_s > 86400 || t.term_s !== t.kill_s - 10) process.exit(1);
console.log(t.kill_s);' "$TIMING_REL") || refuse "$TIMING_REL does not carry a usable ceiling within 24 h"
TERM_S=$(( KILL_S - GRACE_S ))
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
/usr/bin/perl -e "$SUPERVISOR" "$OUT_REL/pgid.txt" "$TERM_S" "$GRACE_S" env -u MODEL_MANIFEST -u HARD_MODEL_MANIFEST -u HARD_MODE "$NODE" "$DRIVER" run "$STAGE" --python "$PY" --kill-s "$KILL_S" \
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
