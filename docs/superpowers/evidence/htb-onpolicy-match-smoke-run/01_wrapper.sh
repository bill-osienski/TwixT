#!/bin/bash
# H-OP MATCH, TIMING SMOKE: TWO COLOUR-REVERSED PAIRS (4 GAMES, UNSCORED) -- OUTER WRAPPER
# (card docs/superpowers/2026-11-14-htb-onpolicy-distillation-card.md, revision 3, §10, §12). The challenger (the export
# qualified by the committed export record) against the current hard 78c16a31c3dba37f, both through protocolMove at 200
# simulations, under the product's game-end rule. CEILING (card §12): TERM at 1790 s, KILL 10 s later, 1800 s in all.
# Idle waiting (idle_gate) is not part of the ceiling. A run the ceiling stops is VOID, with no automatic retry.
# BUILT ONLY: launching it needs the user's launch review.
# A copy of docs/superpowers/evidence/htb-teacher-bench-match-run/01_wrapper.sh (the benchmark, which ran once); the diff:
# the paths, the gate (a line of tests/htb_integration/onpolicy_authorization.mjs), the driver and its stage, the record
# kind, the ceiling, the frozen analyser's sha256 pinned HERE (the card pins no analyser), the committed export record
# and the challenger directory it names (card §9) in place of the teacher venv, clone and teacher runtime check, the
# teacher module dropped from the survivors, and this header. Everything else is unchanged: the idle gate, the BOINC
# check, the one-line gate-open commit, the ONNX Runtime shutdown test (ledger entry 64), the onnxruntime-node pin, the
# perl process-group supervisor, outer.json, and gate restoration with the fallback to HEAD when the pre-open commit
# predates the gate file.
set -u
OUT_REL=docs/superpowers/evidence/htb-onpolicy-match-smoke-run
RUN_REL=docs/superpowers/evidence/htb-onpolicy-match-smoke
PLAN_REL=docs/superpowers/2026-11-14-htb-onpolicy-distillation-card.md
ANALYSER=tests/htb_integration/onpolicy_match_analyse.mjs
ANALYSER_SHA256=1876706130de56fe81ea44c8ef513fe2802440c0fa3d2c77eafce8ca07e6f7b7
EXPORT_REL=docs/superpowers/evidence/htb-onpolicy-export/record.json
GATE_REL=tests/htb_integration/onpolicy_authorization.mjs
GATE=ONPOLICY_MATCH_SMOKE_AUTHORIZED
DRIVER=tests/htb_integration/onpolicy_match.mjs
STAGE=smoke
TERM_S=1790
KILL_S=1800
GRACE_S=10

W=$(cd "$(dirname "$0")/../../../.." && pwd -P) || exit 1
cd "$W" || exit 1
status=1; inner=""; outcome="NOT_STARTED"; started=false; PRE=""; HEAD_C=""; PY=""; NODE=""; start=0
RECORD_KIND=htb-onpolicy-match-record/1

restore() {
  /usr/bin/sed -i '' "s/^export const ${GATE} = true;\$/export const ${GATE} = false;/" "$GATE_REL"
}

finish() {
  trap - EXIT INT TERM HUP
  restore
  restored=false
  REF=$PRE    # the pre-open commit; if it predates the gate file (no launch tree: the run was refused), HEAD instead
  [ -n "$PRE" ] && ! git cat-file -e "$PRE:$GATE_REL" 2>/dev/null && REF=HEAD
  if grep -qx "export const ${GATE} = false;" "$GATE_REL" && { [ -z "$REF" ] || git diff --quiet "$REF" -- "$GATE_REL"; }; then
    restored=true
  fi
  survivors=$( { pgrep -f -- "${DRIVER} run ${STAGE}"; } | sort -un | tr '\n' ' ')
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
git ls-files --error-unmatch "$PLAN_REL" > /dev/null 2>&1 || refuse "$PLAN_REL is not committed"
[ "$(shasum -a 256 "$ANALYSER" | cut -d' ' -f1)" = "$ANALYSER_SHA256" ] || refuse "$ANALYSER is not the frozen analyser (sha256 $ANALYSER_SHA256)"
# THE EXPORT RECORD (card §9) must be committed; the challenger directory is the one it names.
git ls-files --error-unmatch "$EXPORT_REL" > /dev/null 2>&1 || refuse "$EXPORT_REL is not committed"
CHALLENGER_DIR=$("$NODE" -e 'const r = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8")); if (r.status !== "COMPLETED" || r.verdict !== "PARITY_PASS" || !r.challenger?.dir) process.exit(1); console.log(r.challenger.dir);' "$EXPORT_REL") || refuse "$EXPORT_REL is not a COMPLETED PARITY_PASS naming a challenger directory"
[ -d "$CHALLENGER_DIR" ] || refuse "$CHALLENGER_DIR (the exported challenger) is missing"
"$NODE" --test server/test_ort_shutdown.js > /dev/null 2>&1 || refuse "the ONNX Runtime shutdown test fails (ledger entry 64)"
# THE PINNED RUNTIMES (card §4), checked from package metadata before anything loads a model.
ORT_V=$("$NODE" -e 'console.log(require("module").createRequire(process.argv[1])("onnxruntime-node/package.json").version)' "$W/server/inference.js") || refuse "cannot read the onnxruntime-node version"
[ "$ORT_V" = "1.23.2" ] || refuse "onnxruntime-node $ORT_V, the card pins 1.23.2"
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
/usr/bin/perl -e "$SUPERVISOR" "$OUT_REL/pgid.txt" "$TERM_S" "$GRACE_S" env -u MODEL_MANIFEST -u HARD_MODEL_MANIFEST -u HARD_MODE "$NODE" "$DRIVER" run "$STAGE" --challenger-dir "$CHALLENGER_DIR" --kill-s "$KILL_S" \
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
