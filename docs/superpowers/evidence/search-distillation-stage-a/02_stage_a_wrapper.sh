#!/bin/sh
# STAGE A OUTER WRAPPER -- the whole run under ONE 24 h ceiling: TERM at 86,390 s, KILL 10 s later (GNU
# timeout signals the run's whole process group, caffeinate included). `caffeinate -i` keeps the machine
# from idle-sleeping for the run's lifetime. Card: docs/superpowers/2026-10-11-search-distillation-stage-a-card.md.
# A copy of evidence/search-distillation-a1/02_a1_wrapper.sh; the diff: the ceiling, caffeinate, the paths,
# and this header.
#
# THE FROZEN DETACHED LAUNCH (run once, from the worktree root, in a POSIX shell). Both record files are
# RESERVED first, create-only (set -C; set -e stops at the first refusal), so a second launch fails before
# anything starts; then nohup detaches the wrapper from the session, its output on fd 3, and its PID goes to
# the already-open fd 4:
#
#   sh -c 'set -euC; D=docs/superpowers/evidence/search-distillation-stage-a;
#          exec 3>"$D/launch.out.txt";
#          exec 4>"$D/launch.pid.txt";
#          nohup sh "$D/02_stage_a_wrapper.sh" >&3 2>&1 < /dev/null 3>&- 4>&- &
#          printf "%s\n" "$!" >&4'
#
# Run from anywhere; it works in the tree that holds it. The environment and the checkpoint come from the
# main checkout (the parent of the shared git directory), by absolute path; nothing is installed or copied.
# Writes, create-only: run.stdout.txt, run.stderr.txt, outer.json. Exits with the outer status.
set -u
OUT_REL=docs/superpowers/evidence/search-distillation-stage-a
W=$(cd "$(dirname "$0")/../../../.." && pwd -P) || exit 4
cd "$W" || exit 4
MAIN=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd -P) || exit 4
PY="$MAIN/.venv/bin/python"
CKPT="$MAIN/checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"
for f in progress.jsonl record.json run.stdout.txt run.stderr.txt outer.json; do
  [ -e "$OUT_REL/$f" ] && { echo "REFUSED: $OUT_REL/$f exists (one-shot)" >&2; exit 1; }
done
set -C                                   # create-only redirections from here on
start=$(date +%s)
timeout -k 10 86390 caffeinate -i env PYTHONPATH=. "$PY" -B "$OUT_REL/01_stage_a_run.py" --checkpoint "$CKPT" \
  > "$OUT_REL/run.stdout.txt" 2> "$OUT_REL/run.stderr.txt"
status=$?
end=$(date +%s)
survivors=$(pgrep -f "$OUT_REL/01_stage_a_run.py" | tr '\n' ' ')
printf '{"outer_exit": %d, "wall_seconds": %d, "survivors": "%s", "tree": "%s", "python": "%s", "checkpoint": "%s"}\n' \
  "$status" "$((end - start))" "$survivors" "$W" "$PY" "$CKPT" > "$OUT_REL/outer.json"
exit "$status"
