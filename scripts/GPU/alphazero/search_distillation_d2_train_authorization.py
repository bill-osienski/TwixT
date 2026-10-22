"""THE SEARCH DISTILLATION D2 STAGE 3 TRAINING GATE -- and nothing else.

`docs/superpowers/2026-10-22-search-distillation-d2-stage3-card.md`: one training run of a student copy of
calib020_0001 on the Stage 2 corpus (401 updates of the encoder and policy head), writing one candidate checkpoint.
LOADS the network and TRAINS, so it takes its own reviewed door, separate from Stages 1 and 2. Opened for ONE approved
run and restored on every exit. Created closed. This file may hold ONLY this declaration.
"""
SEARCH_DISTILLATION_D2_TRAIN_AUTHORIZED = False
