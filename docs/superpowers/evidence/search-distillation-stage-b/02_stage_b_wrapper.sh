#!/bin/sh
# STAGE B OUTER WRAPPER -- ONE phase under its card cap: TERM at the cap less 10 s, KILL 10 s later (GNU timeout
# signals the whole process group, caffeinate included). Card: docs/superpowers/2026-10-12-search-distillation-
# stage-b-card.md §10. A copy of evidence/search-distillation-stage-a/02_stage_a_wrapper.sh; the diff: the phase
# argument, the cap table, the phase directory, the candidates directory, and this header.
#
# THE DETACHED LAUNCH (G, L1, L2; run once each, from the worktree root, in a POSIX shell). mkdir refuses an
# existing phase directory (set -e), so a second launch fails before anything starts:
#
#   sh -c 'set -euC; P=<PHASE>; D=docs/superpowers/evidence/search-distillation-stage-b/$P; mkdir "$D";
#          exec 3>"$D/launch.out.txt";
#          exec 4>"$D/launch.pid.txt";
#          nohup sh docs/superpowers/evidence/search-distillation-stage-b/02_stage_b_wrapper.sh "$P" >&3 2>&1 \
#            < /dev/null 3>&- 4>&- &
#          printf "%s\n" "$!" >&4'
#
# P, Q, T and K run in the foreground: mkdir "$D" && sh <this file> <PHASE>; echo "EXIT=$?".
# The environment, the checkpoint and the candidates directory come from the main checkout by absolute path.
# Writes, create-only: <PHASE>/run.stdout.txt, run.stderr.txt, outer.json. Exits with the outer status.
set -u
PHASE=${1:-}
case "$PHASE" in
  G) CAP=43190 ;; P) CAP=290 ;; L1) CAP=5090 ;; L2) CAP=12590 ;; T) CAP=1790 ;;   # card caps less the KILL grace
  Q|K) CAP=50 ;;                                    # operational: no model, no search
  *) echo "REFUSED: the phase must be one of G P L1 Q L2 T K" >&2; exit 4 ;;
esac
OUT_REL=docs/superpowers/evidence/search-distillation-stage-b
D="$OUT_REL/$PHASE"
W=$(cd "$(dirname "$0")/../../../.." && pwd -P) || exit 4
cd "$W" || exit 4
MAIN=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd -P) || exit 4
PY="$MAIN/.venv/bin/python"
CKPT="$MAIN/checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"
CAND="$MAIN/runs/search-distillation-stage-b"
[ -d "$D" ] || { echo "REFUSED: $D must be created by the launch" >&2; exit 4; }
for f in progress.jsonl record.json run.stdout.txt run.stderr.txt outer.json; do
  [ -e "$D/$f" ] && { echo "REFUSED: $D/$f exists (one-shot)" >&2; exit 1; }
done
set -C                                   # create-only redirections from here on
start=$(date +%s)
timeout -k 10 "$CAP" caffeinate -i env PYTHONPATH=. "$PY" -B "$OUT_REL/01_stage_b_run.py" --phase "$PHASE" \
  --checkpoint "$CKPT" --candidates "$CAND" > "$D/run.stdout.txt" 2> "$D/run.stderr.txt"
status=$?
end=$(date +%s)
survivors=$(pgrep -f "$OUT_REL/01_stage_b_run.py --phase $PHASE " | tr '\n' ' ')
printf '{"outer_exit": %d, "wall_seconds": %d, "cap_seconds": %d, "survivors": "%s", "tree": "%s", "python": "%s", "checkpoint": "%s"}\n' \
  "$status" "$((end - start))" "$CAP" "$survivors" "$W" "$PY" "$CKPT" > "$D/outer.json"
exit "$status"
