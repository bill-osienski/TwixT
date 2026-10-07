"""THE TACTICAL CANDIDATE QUALIFICATION GATE -- and nothing else.

`docs/superpowers/2026-10-05-tactical-candidate-evaluation-card.md` §3, §14 A4, §17 A7.
It guards the qualification steps: the corpus freeze (`tactical_corpus`, run once at f84c8f90)
and the real-model qualification run (`tactical_qualification`). Each is opened for ONE step
under its own authorization and restored on every exit. This file may hold ONLY this declaration.
"""
TACTICAL_CANDIDATE_QUALIFICATION_AUTHORIZED = True
