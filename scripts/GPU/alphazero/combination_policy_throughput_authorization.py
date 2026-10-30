"""THE COMBINATION POLICY PILOT'S THROUGHPUT-CHECK GATE -- and nothing else.

`docs/superpowers/2026-10-28-combination-policy-pilot-card.md` §5 step 2: a fresh process launched with
MLX_ENABLE_TF32=0 loads calib020_0001 on the GPU and plays the 144-slot throughput check on a registered block, so it
takes its own reviewed door. Opened for ONE approved run and restored on every exit. Created closed. This file may
hold ONLY this declaration.
"""
COMBINATION_POLICY_THROUGHPUT_AUTHORIZED = True
