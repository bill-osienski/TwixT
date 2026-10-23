"""THE SEARCH DISTILLATION D2 ENCODER BRANCH-TRACE GATE -- and nothing else.

`docs/superpowers/2026-10-23-search-distillation-d2-branch-trace-card.md`: a fresh process launched with
MLX_ENABLE_TF32=0 traces the encoder's BatchNorms, residual additions and ReLUs at the incumbent on the update-0 batch
on the CPU and GPU (forward values, branch masks, backward gradients), checks each local step in NumPy float64 and
accounts for any branch flips. No optimizer update and no model state saved, but it loads the incumbent, runs on the
GPU and writes fixtures outside the repository, so it takes its own reviewed door. Opened for ONE approved run and
restored on every exit. Created closed. This file may hold ONLY this declaration.
"""
SEARCH_DISTILLATION_D2_BRANCHTRACE_AUTHORIZED = False
