"""THE SEARCH DISTILLATION D2 REPRODUCIBILITY-DIAGNOSTIC GATE -- and nothing else.

`docs/superpowers/2026-10-22-search-distillation-d2-reproducibility-card.md`: three replays of Stage 3's updates 0-21
in one process, on the training games only, from fresh loads of the unchanged incumbent; no checkpoint is written.
LOADS the network and REPLAYS optimizer updates, so it takes its own reviewed door, separate from Stages 1-3 and the
gradient diagnostic. Opened for ONE approved run and restored on every exit. Created closed. This file may hold ONLY
this declaration.
"""
SEARCH_DISTILLATION_D2_GRADREPEAT_AUTHORIZED = False
