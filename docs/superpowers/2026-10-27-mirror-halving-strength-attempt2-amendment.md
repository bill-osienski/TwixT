# Mirror averaging + root sequential halving — strength comparison, attempt 2 (amendment, 2026-10-27)

An amendment to `docs/superpowers/2026-10-26-mirror-averaging-plus-root-halving-combination-card.md` §5 T. The card
and attempt 1's evidence are unchanged; everything the card does not mention below still applies.

## Why

Attempt 1 (ledger entry 39, `28737707`) timed out at 55,591 s after 799 of 1,018 games and is VOID with no score.
Games 0–349 ran at the expected 0.61 s per move. A BOINC client started about 3.8 h in, took the machine while it was
user-idle, and slowed play to 2.2 s per move. The idle gate checks the launch only. The block
[202693600, 202694618) is closed (`21a2c6f0`), and the 799 games remain unscored for good.

## The exception

The user approved **one** fresh attempt because of this documented machine contention. There is no further attempt,
whatever happens to this one.

## Unchanged

- The candidate, the incumbent, simulations (200 vs 400), 24×24, the existing rules, TF32=0 and eval mode.
- 509 colour-reversed pairs (1,018 games), with no game of the screen or attempt 1 pooled.
- The rule: PASS iff the mean pair score − √(ln(40)/(2×509)) > 0.5, with candidate/incumbent ratios of total network
  evaluations ≤ 1.05 and of total search seconds ≤ 1.10. Clearing the bound while failing a guard is NOT_COMPUTE_MATCHED.
- The ceiling: TERM at 55,591 s, KILL 10 s later (55,601 s).
- The idle gate before launch, the wrapper's gate restoration, and no interim scoring, early decision, extension,
  top-up, retry or parameter change. An incomplete run receives no score.

## Changed

1. **Fresh seeds.** A new 1,018-seed block, from copies of the strength survey and closing procedures under
   `mirror-halving-strength2-seed-{survey,closing}`. The block is chosen by the survey's rule when it runs.
2. **Separate paths.**
   - Plan: `mirror-halving-strength2-run/plan.json`.
   - Wrapper outputs: `mirror-halving-strength2-run/`.
   - Run evidence: `mirror-halving-strength2/`.
   - Run name: `run strength2`.
3. **Gate 50**, `MIRROR_HALVING_STRENGTH2_AUTHORIZED` in its own module, initially closed.
4. **BOINC kept out for the whole run.**
   - The user quits BOINC before the run and keeps it quit until the run finishes. Nothing suspends or restores it
     automatically.
   - `scripts/GPU/alphazero/background_compute.py` checks this read-only, from one `ps` listing. A process counts as
     BOINC when either:
     - its executable path contains "boinc" (any case). The command line is never matched: a shell or `pgrep`
       whose arguments mention BOINC is not BOINC; or
     - it runs as `boinc_master` or `boinc_project` from outside macOS's system directories.
   - That covers the client, the Manager and project workers. The per-user macOS daemons those accounts leave behind
     are excluded.
   - **Exempt, by the user's decision:** BOINC's podman VM (`/opt/podman/bin/` gvproxy and krunkit). It does none of
     the work that slowed attempt 1. Every observation lists it as exempt; it is never counted as BOINC.
   - A `ps` failure or an empty listing is a failed check.
   - **Before launch:** the wrapper runs the check after the idle gate. Detection or a failed check refuses the launch.
     The observation is kept in `bg_check.txt`.
   - **During the run:** the script checks before loading, after every game, and once more before the verdict. Each
     observation is logged as a `bg_check` event. Detection or a failed check ends the attempt as VOID at step
     `background_compute`.
   - These are periodic checks, not continuous ones. A BOINC process that starts and exits within one game would be
     missed.

## Not authorized by this amendment

Seed procedures, registration, checkpoint loading, GPU work, gameplay, opening gate 50, seed closing, deployment,
training and pushing. Each follows its own approval.
