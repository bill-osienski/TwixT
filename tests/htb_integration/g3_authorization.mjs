/**
 * G3 launch gates (match card docs/superpowers/2026-11-09-htb-g3-match-card.md §8).
 *
 * Both are CLOSED in every committed state except a launch commit. A launch commit changes exactly one line of this
 * file, `false` to `true`, and nothing else; the stage's wrapper refuses any other HEAD and restores the gate on every
 * exit. g3_match.mjs refuses to play a stage whose gate is closed. This file is excluded from the execution binding
 * because opening a gate is the one change a launch makes.
 */
export const G3_SMOKE_AUTHORIZED = true;
export const G3_MATCH_AUTHORIZED = false;
