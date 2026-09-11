# H2 — the capability, corrected: an inherited pipe, and nothing is deleted

Code and tests only. Nothing ran: **all seven gates False**, **0 of 736** H2 seeds accounted.

## The two faults, both real

1. **Any 64-character file was accepted.** The token was never compared with anything — only its
   length was checked — so a file of the right size was a valid capability.
2. 🔴 **It DELETED whatever path it was given, valid or not.** `--worker --capability
   <ordinary-file>` destroyed that file **with the gate still closed**. A bypass guard that damages
   data is worse than no guard, and my "forged capability" test only tried empty and short values, so
   it missed the same-length forgery **and endorsed the deletion**.

## The fix: a parent-bound channel with no path at all

The supervisor creates an **anonymous pipe**, writes a random token into it, closes the write end and
passes the **read end** to the child through `pass_fds`. Everything above descriptor 2 is closed on
exec by default, so the descriptor exists in the child only because the parent passed it.

**There is no path, so there is nothing to delete**, and the `--capability` flag that took one is
gone — the path form cannot even be expressed on the command line. The token must now be exactly
64 **hex** characters, so a same-length forgery of other characters is refused. Single use follows
from the pipe being **drained**: the write end is closed at creation, so a second read finds EOF.

`runtime_requalification.supervise` gained an optional `pass_fds`, defaulting to empty, with its own
tests spawning a **real child** that reads the inherited descriptor — and a negative test proving the
child **cannot** read it when the descriptor is not passed.

⚠ **What this is not, stated in the source and here.** It is not an authentication boundary. A caller
who deliberately constructs a pipe, writes 64 hex characters and passes the descriptor number can
still reach the worker. No local mechanism proves parentage without a secret shared out of band. What
it does is remove the **accidental** bypass and make the mechanism incapable of damaging anything.
**The gate is what authorizes a run.**

## Verification

| | |
|---|---|
| H2 tests | **136 passed** |
| controls | **515 / 515 rejected**, 0 not caught, 0 stale, 0 duplicates |
| clean baseline | **PASS** over 426 target tests |
| full suite | **4,385 passed, 4 skipped, 0 failed** |

New tests: a same-length forgery in every wrong alphabet is refused; nothing is deleted and the
path-taking flag does not exist; the check takes a descriptor and contains no deletion at all, proven
by AST; a missing descriptor is refused; the supervisor passes the descriptor and closes it after.

## Three controls were NOT CAUGHT, and one of my claims was wrong

One injected a bare `import tempfile`, which changed nothing. One duplicated the descriptor instead
of closing it, which also changed nothing — **and that one taught me my own claim was wrong**: single
use comes from the pipe being drained, not from closing the descriptor, so the test now says that.
The third was aimed at a wrapper test that **mocks the supervisor**, so the real spawn line was never
exercised; it now targets a test that spawns a real child. A fourth attempt at re-aiming it silently
failed to apply, which the harness caught again on the next run.

## Still not done

Nothing has run. The next phase is the collision re-proof and registration of
`[202618000, 202618736)`, with all seven gates remaining False.
