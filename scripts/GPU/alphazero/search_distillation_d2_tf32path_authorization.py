"""THE SEARCH DISTILLATION D2 FULL-FLOAT32 SETTING PATH VALIDATION GATE -- and nothing else.

`docs/superpowers/2026-10-23-search-distillation-d2-tf32off-path-validation-card.md`: a fresh process launched with
MLX_ENABLE_TF32=0 checks the policy head against the NumPy reference, then the full model's forward outputs and
gradients on the CPU and GPU at the incumbent, then (only if both pass) at the state after 21 in-memory diagnostic
optimizer updates. Saves no model, but loads the incumbent and runs GPU optimizer updates, so it takes its own
reviewed door. Opened for ONE approved run and restored on every exit. Created closed. This file may hold ONLY this
declaration.
"""
SEARCH_DISTILLATION_D2_TF32PATH_AUTHORIZED = True
