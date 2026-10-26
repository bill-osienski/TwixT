"""THE SEARCH DISTILLATION D2 CORRECTED-CONFIGURATION TRAINING TRIAL GATE -- and nothing else.

`docs/superpowers/2026-10-23-search-distillation-d2-corrected-trial-card.md` §5: a fresh process launched with
MLX_ENABLE_TF32=0 trains ONE candidate from calib020_0001 at Stage 3's frozen recipe on the 108 original training games
and judges it on the 12 fresh held-out games. It loads the incumbent, runs on the GPU, makes optimizer updates and saves
a candidate, so it takes its own reviewed door. Opened for ONE approved run and restored on every exit. Created closed.
This file may hold ONLY this declaration.
"""
SEARCH_DISTILLATION_D2_TRIAL_AUTHORIZED = True
