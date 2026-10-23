"""THE SEARCH DISTILLATION D2 CONVOLUTION-REFERENCE DIAGNOSTIC GATE -- and nothing else.

`docs/superpowers/2026-10-23-search-distillation-d2-conv-reference-card.md`: a fresh process launched with
MLX_ENABLE_TF32=0 reproduces entry 32's two failing encoder gradient comparisons at the incumbent on the update-0 batch,
captures each convolution's input and upstream gradient on the CPU and GPU, and compares both backends against a NumPy
float64 convolution weight-gradient reference. No optimizer update and no model state saved, but it loads the incumbent,
runs on the GPU and writes operand fixtures outside the repository, so it takes its own reviewed door. Opened for ONE
approved run and restored on every exit. Created closed. This file may hold ONLY this declaration.
"""
SEARCH_DISTILLATION_D2_CONVREF_AUTHORIZED = False
