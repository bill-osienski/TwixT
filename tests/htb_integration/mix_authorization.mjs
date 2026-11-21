/**
 * H-MX match launch gates (card docs/superpowers/2026-11-18-htb-mix-distillation-card.md, revision 1, §9, §11, §14).
 *
 * Both are CLOSED in every committed state except a launch commit. A launch commit changes exactly one line of this
 * file, `false` to `true`, and nothing else; the stage's wrapper refuses any other HEAD and restores the gate on every
 * exit. mix_match.mjs refuses to run a stage whose gate is closed.
 */
export const MIX_MATCH_SMOKE_AUTHORIZED = false;
export const MIX_MATCH_AUTHORIZED = true;
