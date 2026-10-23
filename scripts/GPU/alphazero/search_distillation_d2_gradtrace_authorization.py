"""THE SEARCH DISTILLATION D2 GRADIENT-DIAGNOSTIC GATE -- and nothing else.

`docs/superpowers/2026-10-22-search-distillation-d2-gradient-diagnostic-card.md`: one instrumented replay of Stage 3's
401 updates on the training games only, from the unchanged incumbent; no checkpoint is written. LOADS the network and
REPLAYS optimizer updates, so it takes its own reviewed door, separate from Stages 1-3. Opened for ONE approved run and
restored on every exit. Created closed. This file may hold ONLY this declaration.
"""
SEARCH_DISTILLATION_D2_GRADTRACE_AUTHORIZED = False
