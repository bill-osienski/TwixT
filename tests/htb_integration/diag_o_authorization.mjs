/**
 * Diagnostic O launch gate (plan docs/superpowers/2026-11-12-htb-diagnostic-o-plan.md §8).
 *
 * CLOSED in every committed state except the launch commit, which changes exactly this one line, `false` to `true`.
 * The wrapper refuses any other HEAD and restores the gate on every exit. diag_o.mjs and htb_diag_o_teacher.py
 * refuse to run while it is closed.
 */
export const DIAG_O_AUTHORIZED = true;
