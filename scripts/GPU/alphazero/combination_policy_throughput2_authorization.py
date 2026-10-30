"""THE COMBINATION POLICY PILOT'S THROUGHPUT-CHECK ATTEMPT-2 GATE -- and nothing else.

`docs/superpowers/2026-10-28-combination-policy-pilot-card.md` §5 step 2, as amended by
`docs/superpowers/2026-10-29-combination-policy-throughput-attempt2-amendment.md`: a fresh process launched with
MLX_ENABLE_TF32=0 loads calib020_0001 on the GPU and plays the ONE fresh 144-slot throughput check on its own
registered block, under a 30-minute ceiling. It takes its own reviewed door, so opening it can never relaunch
attempt 1's paths. Opened for ONE approved run and restored on every exit. Created closed. This file may
hold ONLY this declaration.
"""
COMBINATION_POLICY_THROUGHPUT2_AUTHORIZED = True
