"""THE EXTERNAL-ENGINE LEARNING STAGE-2 GATE -- and nothing else.

`docs/superpowers/2026-10-08-external-engine-learning-viability.md` (design baseline ff1422fb), stage 2:
the training smoke in `external_learning_stage2` -- the scalar-temperature control on the cached training
logits, then ONE 128-step, batch-32 policy-head-only update of a verified incumbent copy, saved once and
reloaded. Opened for ONE approved run and restored on every exit. Created closed. This file may hold ONLY
this declaration.
"""
EXTERNAL_LEARNING_STAGE2_AUTHORIZED = False
