"""THE SEARCH DISTILLATION D2 GRADIENT-CHECK GATE -- and nothing else.

`docs/superpowers/2026-10-22-search-distillation-d2-gradient-check-card.md`: a CPU/GPU comparison of the training
loss's gradients at two fixed states, a NumPy reference for the policy head and stage-by-stage localization; one
GPU prefix of Stage 3's updates 0-20 reconstructs the second state; no checkpoint or full model state is written.
LOADS the network and runs one optimizer prefix, so it takes its own reviewed door, separate from Stages 1-3 and the
earlier diagnostics. Opened for ONE approved run and restored on every exit. Created closed. This file may hold ONLY
this declaration.
"""
SEARCH_DISTILLATION_D2_GRADCHECK_AUTHORIZED = False
