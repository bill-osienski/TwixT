"""THE H5 STUDY-SEGMENT EXECUTION GATE -- and nothing else.

H5 implementation card §1. It lives OUTSIDE `h5_runner.py` because the runner is
hashed into every H5 manifest: a gate inside it would change the bound hash on
the one-line commit that opens it. This file may hold ONLY this declaration; a
test refuses any import, function, class or other statement here.
"""
H5_STUDY_EXECUTION_AUTHORIZED = False
