"""THE SEARCH DISTILLATION D2 FULL-FLOAT32 SETTING VALIDATION GATE -- and nothing else.

`docs/superpowers/2026-10-23-search-distillation-d2-tf32off-validation-card.md`: a fresh process launched with
MLX_ENABLE_TF32=0 runs the unchanged synthetic probe and entry 27's four original saved linear-operation cases, on the
GPU against NumPy. Loads no network, trains nothing, but runs GPU computations under a frozen case list, so it takes
its own reviewed door. Opened for ONE approved run and restored on every exit. Created closed. This file may hold
ONLY this declaration.
"""
SEARCH_DISTILLATION_D2_TF32OFF_AUTHORIZED = True
