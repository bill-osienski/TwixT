"""THE H4 PILOT/STUDY EXECUTION GATE -- and nothing else (runner card §12.6.2).

It lives OUTSIDE `h4_runner.py` because the runner is hashed into every manifest:
a gate inside it would change the bound hash on the one-line commit that opens it.
This file is NOT in `code`, so it may hold ONLY this declaration -- a test refuses
any import, function, class or other statement here. Opening it is a reviewed
one-line change; the runner reads it at call time at both public entries.
"""
H4_PILOT_EXECUTION_AUTHORIZED = True
