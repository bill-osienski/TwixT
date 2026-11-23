/**
 * H-MX integration check launch gate (proposal docs/superpowers/2026-11-22-htb-mix-integration-proposal.md, revision
 * 1, §5). CLOSED in every committed state except a launch commit. A launch commit changes exactly one line of this
 * file, `false` to `true`, and nothing else; the wrapper refuses any other HEAD and restores the gate on every exit.
 * hmx_integration_check.mjs refuses to run while it is closed.
 */
export const HMX_INTEGRATION_CHECK_AUTHORIZED = true;
