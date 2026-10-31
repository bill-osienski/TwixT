"""THE COMBINATION POLICY PILOT'S TRAINING GATE -- and nothing else.

`docs/superpowers/2026-10-28-combination-policy-pilot-card.md` §5 step 4: a fresh process launched with
MLX_ENABLE_TF32=0 loads calib020_0001 weights-only and trains the residual policy branch on the CPU from the committed
corpus, then applies the check's STOP rules, so it takes its own reviewed door. Opened for ONE approved run and
restored on every exit. Created closed. This file may hold ONLY this declaration.
"""
COMBINATION_POLICY_TRAIN_AUTHORIZED = True
