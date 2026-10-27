"""The background-compute check (`scripts/GPU/alphazero/background_compute.py`): is BOINC on this machine? CPU only,
read-only; nothing here stops, suspends or starts a process. Listings are scripted, except the one real `ps` read."""
import pytest

from scripts.GPU.alphazero import background_compute as BG

#: what `ps -o pid=,user=,comm=` showed on this machine with BOINC quit (2026-10-27): macOS's per-user daemons under
#: the BOINC accounts, BOINC's idle podman VM, and ordinary processes
QUIT = [(1070, "boinc_master", "/usr/sbin/distnoted"), (26605, "boinc_master", "/usr/libexec/lsd"),
        (26607, "boinc_master", "/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/"
                                "Metadata.framework/Versions/A/Support/mdbulkimport"),
        (53099, "boinc_project", "/usr/sbin/cfprefsd"),
        (97690, "boinc_project", "/opt/podman/bin/gvproxy"), (97693, "boinc_project", "/opt/podman/bin/krunkit"),
        (501, "bill", "/bin/zsh"), (502, "bill", "/usr/bin/pgrep")]
CLIENT = (5000, "boinc_master", "/Applications/BOINCManager.app/Contents/Resources/boinc")
MANAGER = (5001, "bill", "/Applications/BOINCManager.app/Contents/MacOS/BOINCManager")
WORKER = (5635, "boinc_project", "hsgamma_FGRP5_1.14_arm64-apple-darwin")       # attempt 1's Einstein@Home task


def _obs(procs):
    return BG.observe(lambda: list(procs))


def test_BOINC_QUIT_IS_CLEAR_and_the_podman_VM_is_listed_as_exempt():
    obs = _obs(QUIT)
    assert obs["ok"] is True and obs["hits"] == [] and obs["n_hits"] == 0 and obs["error"] is None
    assert obs["exempt"] == ["97690 boinc_project /opt/podman/bin/gvproxy", "97693 boinc_project /opt/podman/bin/krunkit"]
    assert BG.describe(obs).startswith("CLEAR: no BOINC process among 8 (exempt: 97690")


@pytest.mark.parametrize("proc", [CLIENT, MANAGER, WORKER], ids=["client", "manager", "worker"])
def test_THE_CLIENT_THE_MANAGER_OR_A_WORKER_IS_DETECTED(proc):
    obs = _obs(QUIT + [proc])
    assert obs["ok"] is False and obs["error"] is None and obs["n_hits"] == 1
    assert obs["hits"] == [f"{proc[0]} {proc[1]} {proc[2]}"] and BG.describe(obs).startswith("BOINC DETECTED: 1 of 9")


def test_ATTEMPT_1_S_MACHINE_IS_DETECTED():
    """At TERM: the client and eight workers at ~99% CPU each."""
    obs = _obs(QUIT + [CLIENT] + [(6000 + i, "boinc_project", WORKER[2]) for i in range(8)])
    assert obs["ok"] is False and obs["n_hits"] == 9 and len(obs["hits"]) == BG.SHOWN


def test_ONLY_THE_EXECUTABLE_COUNTS():
    """A shell or pgrep whose ARGUMENTS mention BOINC is not BOINC (the listing carries the executable only), and an
    executable path naming BOINC is BOINC whoever runs it."""
    assert not BG.is_boinc("bill", "/bin/zsh") and not BG.is_boinc("bill", "/usr/bin/pgrep")
    assert BG.is_boinc("bill", "/Library/Application Support/BOINC Data/boinc")
    assert BG.is_boinc("boinc_project", "../../projects/einstein.phys.uwm.edu/hsgamma_FGRP5")
    assert not BG.is_boinc("boinc_project", "/usr/libexec/trustd")


def test_THE_EXEMPTION_IS_PODMAN_S_DIRECTORY_ONLY():
    assert BG.EXEMPT_DIRS == ("/opt/podman/bin/",)
    assert not BG.is_boinc("boinc_project", "/opt/podman/bin/krunkit")
    assert BG.is_boinc("boinc_project", "/opt/podmanx/bin/krunkit") and BG.is_boinc("boinc_project", "/opt/krunkit")


@pytest.mark.parametrize("listing, error", [(None, "OSError: ps failed"), ([], "the process listing is empty")],
                         ids=["ps-fails", "empty"])
def test_A_FAILED_CHECK_IS_NEVER_A_PASS(listing, error):
    def fn():
        if listing is None:
            raise OSError("ps failed")
        return listing
    obs = BG.observe(fn)
    assert obs["ok"] is False and obs["error"] == error and BG.describe(obs) == f"CHECK FAILED: {error}"


@pytest.mark.parametrize("procs, code", [(QUIT, BG.EXIT_CLEAR), (QUIT + [CLIENT], BG.EXIT_DETECTED),
                                         ([], BG.EXIT_FAILED)], ids=["clear", "detected", "failed"])
def test_THE_COMMAND_S_EXIT_CODES(monkeypatch, capsys, procs, code):
    monkeypatch.setattr(BG, "listing", lambda: list(procs))
    monkeypatch.setattr(BG.observe, "__defaults__", (BG.listing,))
    assert BG.main(["check"]) == code and capsys.readouterr().out.strip() == BG.describe(_obs(procs))


def test_ANYTHING_BUT_check_IS_USAGE():
    assert BG.main([]) == BG.main(["wait"]) == BG.EXIT_USAGE


def test_THE_REAL_LISTING_READS_EXECUTABLES_AND_EXCLUDES_ITSELF():
    import os
    procs = BG.listing()
    assert procs and all(type(p) is int and u and e for p, u, e in procs) and os.getpid() not in {p for p, _, _ in procs}
