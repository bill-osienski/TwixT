"""THE SEARCH DISTILLATION D2 FRESH HELD-OUT SET GATE -- and nothing else.

`docs/superpowers/2026-10-23-search-distillation-d2-corrected-trial-card.md` §4: a fresh process launched with
MLX_ENABLE_TF32=0 plays 12 fresh self-play games of calib020_0001 at 1,600 simulations on the GPU, the trial's new
held-out set, stored outside the repository apart from the original corpus. It loads the incumbent, runs on the GPU and
consumes registered seeds, so it takes its own reviewed door. Opened for ONE approved run and restored on every exit.
Created closed. This file may hold ONLY this declaration.
"""
SEARCH_DISTILLATION_D2_HELDOUT_AUTHORIZED = False
