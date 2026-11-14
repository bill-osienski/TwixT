/**
 * Student-teacher benchmark launch gates (card docs/superpowers/2026-11-14-htb-teacher-benchmark-card.md §9, §11).
 *
 * Both are CLOSED in every committed state except a launch commit. A launch commit changes exactly one line of this
 * file, `false` to `true`, and nothing else; the stage's wrapper refuses any other HEAD and restores the gate on every
 * exit. teacher_bench.mjs and htb_bench_teacher.py refuse to run a stage whose gate is closed.
 */
export const BENCH_SMOKE_AUTHORIZED = false;
export const BENCH_MATCH_AUTHORIZED = true;
