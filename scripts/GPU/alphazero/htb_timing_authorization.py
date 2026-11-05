"""THE H-TB PILOT'S BOUNDED TIMING GATE -- and nothing else.

`docs/superpowers/2026-11-04-htb-bounded-timing-plan.md`: `htb_timing time label` (the teacher environment: corpus
authentication, train-arm keying, teacher calls) and `htb_timing time fit` (the project environment, CPU: examples,
two checkpoint loads, 60 updates, scoring) each run once under their own wrapper. Opened for ONE approved run at a time
and restored on every exit. Created closed. This file may hold ONLY this declaration.
"""
HTB_TIMING_AUTHORIZED = False
