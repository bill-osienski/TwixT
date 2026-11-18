/**
 * H-OP match launch gates (card docs/superpowers/2026-11-14-htb-onpolicy-distillation-card.md, revision 3, §10, §12, §15).
 *
 * Both are CLOSED in every committed state except a launch commit. A launch commit changes exactly one line of this
 * file, `false` to `true`, and nothing else; the stage's wrapper refuses any other HEAD and restores the gate on every
 * exit. onpolicy_match.mjs refuses to run a stage whose gate is closed.
 */
export const ONPOLICY_MATCH_SMOKE_AUTHORIZED = true;
export const ONPOLICY_MATCH_AUTHORIZED = false;
