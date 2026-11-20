"""H-MX recovery amendment 1: model-free tests (docs/superpowers/2026-11-19-htb-mix-recovery-amendment.md).

No gate is opened and the student is never loaded. The real-path test drives the REAL `save_student` and the REAL loader
(`probe_eval.load_network_for_scoring`) on a full-size, randomly initialized network into a cache whose parent
directory does not exist. The replay runs end to end on tiny networks and fake label sets: its "failed record" is
produced here by the real `htb_mix.run_fit` with a save that fails exactly as the run did.
"""
import dataclasses
import hashlib
import json
import pathlib

import mlx.core as mx
import pytest

from scripts.GPU.alphazero import htb_mix as M
from scripts.GPU.alphazero import htb_mix_export as XM
from scripts.GPU.alphazero import htb_mix_recovery as R
from scripts.GPU.alphazero import htb_pilot as P
from scripts.GPU.alphazero import search_distillation_d2 as D
from scripts.GPU.alphazero.network import AlphaZeroNetwork, create_network
from tests import test_htb_mix as TM
from tests import test_htb_pilot as TP

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAVE_ERROR = "[save_safetensors] Failed to open file {}"


@pytest.fixture(autouse=True)
def _cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


# ─────────────────────────────── pins, gate, code (amendment §1, §3, §5) ───────────────────────────────

def test_THE_AMENDMENT_IS_PINNED_AND_THE_CARD_AND_THE_FIT_CODE_ARE_UNCHANGED():
    assert _sha(ROOT / R.AMENDMENT) == R.AMENDMENT_SHA256
    assert _sha(ROOT / M.CARD) == M.CARD_SHA256
    assert _sha(ROOT / "scripts/GPU/alphazero/htb_mix.py") == R.FAILED["module_sha256"]     # the code that trained it
    assert R.TARGET == R.FAILED["student_digest"] == "adba50fb2e241ec0388fa99998b392c3e0b431b55c10676800a546899aee34b2"
    assert R.KILL_S == M.KILL_S["fit"] == 11_700 and R.EV != M.EV["fit"] and R.OUTER != M.OUTER["fit"]


def test_THE_REAL_FAILED_RECORD_IS_THE_PINNED_COMMITTED_ONE_AND_BINDS():
    """Entry 73's evidence, preserved: the pinned bytes, FAILED at the save after a full fit, every screen held."""
    for k in ("record", "outer"):
        assert _sha(ROOT / R.FAILED[k]) == R.FAILED[f"{k}_sha256"]
    rec = R.bind_failed(R.Config())
    assert rec["status"] == "FAILED" and rec["fit"]["student_digest"] == R.TARGET and rec["fit"]["updates"] == 7_500
    assert rec["gate"]["held"] == ["M1", "M2", "M3", "M4", "M5", "M6"] and "challenger" not in rec
    code = R.code_binding(R.Config(), R.FAILED)
    assert code["fit_module_sha256"] == R.FAILED["module_sha256"] and set(code["changed"]) <= R.MAY_DIFFER


def test_THE_RUNTIME_ON_THIS_MACHINE_IS_THE_PINNED_ONE():
    assert R.runtime_binding(R.Config())["mlx"] == "0.30.3"


def test_THE_GATE_IS_CLOSED_AND_THE_ENTRY_REFUSES_WRITING_NOTHING(capsys):
    text = (ROOT / "scripts/GPU/alphazero/htb_mix_recovery_authorization.py").read_text()
    assert [ln for ln in text.split('"""')[-1].splitlines() if ln.strip()] == ["HTB_MIX_RECOVERY_AUTHORIZED = False"]
    assert R.gate_open("recovery") is False
    # compared before and after, not asserted absent: the recovery's evidence and the cache exist since entry 74
    snap = lambda: [sorted(p.name for p in d.iterdir()) if d.exists() else None  # noqa: E731
                    for d in (ROOT / R.EV, ROOT / R.OUTER, M.CACHE)]
    before = snap()
    assert R.main(["run", "replay"]) == P.EXIT["REFUSED"]
    assert "HTB_MIX_RECOVERY_AUTHORIZED is closed" in capsys.readouterr().err
    assert snap() == before


def test_THE_WRAPPER_IS_ITS_DERIVATION_WITH_THE_FITS_CEILING():
    from tests import htb_mix_recovery_wrapper_spec as W

    text = W.wrapper_path().read_text()
    assert text == W.derive()
    body = text.split("set -u\n", 1)[1]
    assert "TERM_S=11690\n" in body and "GRACE_S=10\n" in body and 'run replay \\\n' in body and "run fit" not in body
    assert f"RUN_REL={R.EV}\n" in body and f"OUT_REL={R.OUTER}\n" in body and "GATE=HTB_MIX_RECOVERY_AUTHORIZED\n" in body
    assert "MODULE=scripts.GPU.alphazero.htb_mix_recovery\n" in body and f"RECORD_KIND={R.RECORD_KIND}\n" in body
    parent = (ROOT / W.PARENT).read_text().split("set -u\n", 1)[1]
    for old, new in W.SUBS:
        parent = parent.replace(old, new)
    assert body == parent


# ─────────────────────────────── the repair: the real save and reload (amendment §2) ───────────────────────────────

def _full_net(seed=3):
    mx.random.seed(seed)
    net = create_network()                       # the production architecture the real loader builds
    mx.eval(net.parameters())
    net.eval()
    return net


def test_THE_REAL_SAVE_INTO_A_MISSING_DIRECTORY_FAILS_AS_THE_RUN_DID(tmp_path):
    with pytest.raises(RuntimeError, match=r"\[save_safetensors\] Failed to open file"):
        P._save_student_real(_full_net(), tmp_path / "missing" / "htb-mix" / "challenger.safetensors")
    assert not (tmp_path / "missing").exists()


def test_THE_PREPARED_CACHE_TAKES_THE_REAL_SAVE_AND_THE_REAL_RELOAD_VERIFIES_THE_DIGEST(tmp_path):
    net = _full_net()
    target = P._digest(net)
    cfg = R.Config(cache=tmp_path / "missing-parent" / "htb-mix")
    assert not cfg.cache.parent.exists()
    prep = R.prepare_cache(cfg)
    assert prep == {"path": str(cfg.cache), "existed_before": False, "writable": True}
    assert cfg.cache.is_dir() and list(cfg.cache.iterdir()) == []                 # the probe is removed
    out = {}
    R.verified_save(cfg, net, P._save_student_real, R._reload_real, out, target=target)
    assert out["verified"] is True and out["reload_state_digest"] == out["state_digest"] == target
    assert out["path"] == str(cfg.challenger) and out["sha256"] == _sha(cfg.challenger)
    assert sorted(p.name for p in cfg.cache.iterdir()) == ["challenger.safetensors"]   # no temporary file left


def test_A_RELOAD_AT_ANY_OTHER_DIGEST_PUBLISHES_NOTHING(tmp_path):
    net = _full_net()
    cfg = R.Config(cache=tmp_path / "c")
    R.prepare_cache(cfg)
    out = {}
    with pytest.raises(P.Stop, match="reloaded file's state digest"):
        R.verified_save(cfg, net, P._save_student_real, lambda p: _full_net(seed=4), out, target=P._digest(net))
    assert out["verified"] is False and cfg.challenger.is_file()


def test_THE_CACHE_IS_REFUSED_IF_IT_CANNOT_BE_PREPARED_OR_ALREADY_HOLDS_A_CHALLENGER(tmp_path):
    (tmp_path / "file").write_text("x")
    with pytest.raises(P.Refused, match="cannot be prepared"):
        R.prepare_cache(R.Config(cache=tmp_path / "file" / "htb-mix"))
    cfg = R.Config(cache=tmp_path / "c")
    cfg.cache.mkdir()
    cfg.challenger.write_bytes(b"old")
    with pytest.raises(P.Refused, match="already exists"):
        R.prepare_cache(cfg)


# ─────────────────────────────── the replay, end to end on fakes (amendment §3, §4) ───────────────────────────────

def _eligible(monkeypatch):
    """The fakes' tiny fit misses the screens; the recovery's machinery is tested on a record whose screens held."""
    real = M.gate_verdict

    def gate(*a):
        g = real(*a)
        return {**g, **{k: True for k in ("M1", "M2", "M3", "M4", "M5", "M6")},
                "held": ["M1", "M2", "M3", "M4", "M5", "M6"], "failed": [], "verdict": "ELIGIBLE"}
    monkeypatch.setattr(M, "gate_verdict", gate)


def _failed_run(tmp_path, monkeypatch, eligible=True):
    """A FAILED fit record made by the real run_fit with a save that fails exactly as the run did, its outer.json, the
    amendment, and a recovery config bound to them."""
    if eligible:
        _eligible(monkeypatch)
    cfg, base, games = TM._fit_setup(tmp_path)

    def bad_save(net, path):
        raise RuntimeError(SAVE_ERROR.format(path))
    assert M.run_fit(cfg, load_student=lambda: TP.twin(base), games=lambda c: games, save=bad_save) == P.EXIT["FAILED"]
    root = cfg.root
    rec = json.loads((root / M.EV["fit"] / "record.json").read_text())
    out = root / M.OUTER["fit"]
    out.mkdir(parents=True)
    (out / "outer.json").write_text(json.dumps({"inner_exit": 4, "outcome": "FAILED", "outer_exit": 4,
                                                "gate_restored": True, "survivors": "", "record_by": "script",
                                                "head": rec["head"]}))
    (root / R.AMENDMENT).write_bytes((ROOT / R.AMENDMENT).read_bytes())
    failed = {"record": f"{M.EV['fit']}/record.json", "record_sha256": _sha(root / M.EV["fit"] / "record.json"),
              "outer": f"{M.OUTER['fit']}/outer.json", "outer_sha256": _sha(out / "outer.json"),
              "head": rec["head"], "module_sha256": rec["module_sha256"], "reference_digest": rec["fit"]["reference_digest"],
              "student_digest": rec["fit"]["student_digest"]}
    fields = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(M.Config) if f.name != "gate_open"}
    fields["cache"] = tmp_path / "no-such-parent" / "htb-mix"
    rcfg = R.Config(**fields, gate_open=lambda s: s == "recovery", failed=failed, code_changes=lambda r, s: [],
                    runtime=lambda: dict(R.RUNTIME))
    return rcfg, base, games, rec


def _tiny_reload(path):
    return D.reload(path, lambda: AlphaZeroNetwork(hidden=16, n_blocks=1))


def test_THE_REPLAY_REPRODUCES_THE_FAILED_FIT_SAVES_RELOADS_AND_ONLY_THEN_IS_ELIGIBLE(tmp_path, monkeypatch):
    rcfg, base, games, failed = _failed_run(tmp_path, monkeypatch)
    failed_bytes = (rcfg.root / M.EV["fit"] / "record.json").read_bytes()
    scored = []
    real_score = P.score
    monkeypatch.setattr(P, "score", lambda net, rows: scored.append(len(rows)) or real_score(net, rows))
    out = R.run_replay(rcfg, load_student=lambda: TP.twin(base), games=lambda c: games, save=P._save_student_real,
                       reload=_tiny_reload)
    rec = json.loads((rcfg.ev_recovery / "record.json").read_text())
    assert out == 0 and rec["status"] == "COMPLETED" and rec["verdict"] == "ELIGIBLE" and rec["kind"] == R.RECORD_KIND
    assert scored == [rcfg.start_rows]                         # the start check only: no validation scoring
    assert "skipped" in rec["validation_scoring"]
    assert rec["reproduction"]["fit_record_equal"] is True and rec["reproduction"]["differs"] == []
    assert rec["fit"] == failed["fit"] and rec["selection"] == failed["selection"] and rec["start"] == failed["start"]
    ch = rec["challenger"]
    assert ch["verified"] is True and ch["reload_state_digest"] == ch["state_digest"] == failed["fit"]["student_digest"]
    assert ch["sha256"] == _sha(rcfg.challenger) and rec["cache"]["existed_before"] is False
    assert rec["screens"]["gate"] == failed["gate"] and rec["screens"]["challenger_val"] == failed["challenger_val"]
    assert "entry 73" in rec["links"]["ledger"] and rec["links"]["failed_record_sha256"] == rcfg.failed["record_sha256"]
    assert rec["amendment_sha256"] == R.AMENDMENT_SHA256 and rec["kill_s"] == 11_700
    assert (rcfg.root / M.EV["fit"] / "record.json").read_bytes() == failed_bytes          # preserved
    with pytest.raises(P.Refused, match="destination already exists"):
        R.run_replay(rcfg, load_student=lambda: TP.twin(base), games=lambda c: games)


def test_A_FIT_THAT_DOES_NOT_REPRODUCE_STOPS_AND_SAVES_NOTHING(tmp_path, monkeypatch):
    rcfg, base, games, failed = _failed_run(tmp_path, monkeypatch)
    rcfg.recipe = dataclasses.replace(rcfg.recipe, learning_rate=rcfg.recipe.learning_rate * 2)   # same start, other fit
    saved = []
    assert R.run_replay(rcfg, load_student=lambda: TP.twin(base), games=lambda c: games,
                        save=lambda n, p: saved.append(p), reload=_tiny_reload) == P.EXIT["STOP"]
    rec = json.loads((rcfg.ev_recovery / "record.json").read_text())
    assert rec["status"] == "STOP" and rec["stop"]["step"] == "replay" and R.NOT_REPRODUCED in rec["stop"]["detail"]
    assert rec["reproduction"]["student_digest"]["equal"] is False and "student_digest" in rec["reproduction"]["differs"]
    assert saved == [] and "verdict" not in rec and "screens" not in rec and not rcfg.challenger.exists()


def test_A_DIFFERENT_START_OR_SELECTION_STOPS_BEFORE_ANY_UPDATE(tmp_path, monkeypatch):
    rcfg, base, games, failed = _failed_run(tmp_path / "a", monkeypatch)
    rcfg2, base2, games2, _ = _failed_run(tmp_path / "b", monkeypatch)
    fits = []
    monkeypatch.setattr(M, "fit_mixed", lambda *a, **k: fits.append(1))
    other = TP.tiny_net(8)
    assert R.run_replay(rcfg, load_student=lambda: TP.twin(other), games=lambda c: games) == P.EXIT["STOP"]
    rec = json.loads((rcfg.ev_recovery / "record.json").read_text())
    assert rec["stop"]["step"] == "replay" and "start check differs" in rec["stop"]["detail"] and fits == []

    real = M.mix_selection
    monkeypatch.setattr(M, "mix_selection", lambda *a: (lambda s: {**s, "record": {**s["record"], "tb_dropped": 99}})(real(*a)))
    boom = lambda *a: (_ for _ in ()).throw(AssertionError("loaded"))           # noqa: E731
    assert R.run_replay(rcfg2, load_student=boom, games=boom) == P.EXIT["STOP"]
    rec = json.loads((rcfg2.ev_recovery / "record.json").read_text())
    assert "selection differs" in rec["stop"]["detail"] and fits == []


def test_A_RELOAD_MISMATCH_OR_A_FAILED_SAVE_PUBLISHES_NO_VERDICT(tmp_path, monkeypatch):
    rcfg, base, games, failed = _failed_run(tmp_path, monkeypatch)
    assert R.run_replay(rcfg, load_student=lambda: TP.twin(base), games=lambda c: games, save=P._save_student_real,
                        reload=lambda p: TP.tiny_net(99)) == P.EXIT["STOP"]
    rec = json.loads((rcfg.ev_recovery / "record.json").read_text())
    assert rec["stop"]["step"] == "reload" and rec["challenger"]["verified"] is False and "verdict" not in rec

    rcfg2, base2, games2, failed2 = _failed_run(tmp_path / "b", monkeypatch)

    def bad_save(net, path):
        raise RuntimeError(SAVE_ERROR.format(path))
    assert R.run_replay(rcfg2, load_student=lambda: TP.twin(base2), games=lambda c: games2, save=bad_save,
                        reload=_tiny_reload) == P.EXIT["FAILED"]
    rec = json.loads((rcfg2.ev_recovery / "record.json").read_text())
    assert rec["status"] == "FAILED" and rec["challenger"]["verified"] is False and "verdict" not in rec


@pytest.mark.parametrize("fault", ["bytes", "status", "screens", "digest", "outer"])
def test_THE_REPLAY_REFUSES_A_FAILED_RECORD_IT_CANNOT_RELY_ON_BEFORE_LOADING_ANYTHING(tmp_path, monkeypatch, fault):
    rcfg, base, games, failed = _failed_run(tmp_path, monkeypatch, eligible=fault != "screens")
    rel = M.EV["fit"] + "/record.json"
    if fault == "bytes":
        (rcfg.root / rel).write_text((rcfg.root / rel).read_text() + " ")
    elif fault in ("status", "digest"):
        r = dict(failed)
        if fault == "status":
            r["status"] = "COMPLETED"
        else:
            r["fit"] = {**r["fit"], "student_digest": "0" * 64}
        (rcfg.root / rel).write_text(json.dumps(r))
        rcfg.failed = {**rcfg.failed, "record_sha256": _sha(rcfg.root / rel)}
    elif fault == "outer":
        o = rcfg.root / M.OUTER["fit"] / "outer.json"
        o.write_text(json.dumps({**json.loads(o.read_text()), "survivors": "unconfirmed cleanup"}))
        rcfg.failed = {**rcfg.failed, "outer_sha256": _sha(o)}
    want = {"bytes": "not the pinned failed record", "status": r"\['status'\]", "screens": r"\['screens'\]",
            "digest": r"\['fit'\]", "outer": r"\['outer'\]"}[fault]
    boom = lambda *a: (_ for _ in ()).throw(AssertionError("loaded"))           # noqa: E731
    with pytest.raises(P.Refused, match=want):
        R.run_replay(rcfg, load_student=boom, games=boom, save=boom, reload=boom)
    assert not rcfg.ev_recovery.exists() and not rcfg.cache.exists()


@pytest.mark.parametrize("fault", ["module", "code", "runtime", "amendment", "gate", "memory"])
def test_THE_REPLAY_REFUSES_OTHER_CODE_RUNTIME_OR_PINS_BEFORE_PREPARING_ANYTHING(tmp_path, monkeypatch, fault):
    rcfg, base, games, failed = _failed_run(tmp_path, monkeypatch)
    if fault == "module":
        rcfg.fit_module_sha256 = lambda: "1" * 64
    elif fault == "code":
        rcfg.code_changes = lambda r, s: ["scripts/GPU/alphazero/htb_pilot.py", "scripts/GPU/alphazero/htb_mix_export.py"]
    elif fault == "runtime":
        rcfg.runtime = lambda: {**R.RUNTIME, "mlx": "0.31.0"}
    elif fault == "amendment":
        rcfg.amendment_sha256 = "2" * 64
    elif fault == "gate":
        rcfg.gate_open = lambda s: False
    else:
        rcfg.available_memory = lambda: {"available_bytes": 1}
    want = {"module": "htb_mix.py is", "code": "htb_pilot.py", "runtime": "not the failed run's", "amendment":
            "amendment is not", "gate": "HTB_MIX_RECOVERY_AUTHORIZED is closed", "memory": "below the recovery"}[fault]
    boom = lambda *a: (_ for _ in ()).throw(AssertionError("loaded"))           # noqa: E731
    with pytest.raises(P.Refused, match=want):
        R.run_replay(rcfg, load_student=boom, games=boom, save=boom, reload=boom)
    assert not rcfg.ev_recovery.exists() and not rcfg.cache.exists()


# ─────────────────────────────── the export admits only a verified recovery (amendment §6) ───────────────────────────────

def test_THE_EXPORT_ADMITS_A_CLEAN_VERIFIED_RECOVERY(tmp_path):
    cfg = TM._cfg(TM._tree(tmp_path))
    ch = TM._recovery_evidence(cfg, tmp_path)
    adm = XM.recovery_admission(cfg)
    assert adm["challenger"]["path"] == str(ch) and adm["challenger"]["state_digest"] == R.TARGET
    assert adm["challenger"]["sha1"] == hashlib.sha1(ch.read_bytes()).hexdigest()


@pytest.mark.parametrize("fault", ["absent", "only_failed_fit", "stop", "outer", "pgid", "ceiling", "unverified",
                                   "reload_digest", "tampered", "unbound", "amendment", "verdict"])
def test_THE_EXPORT_REQUIRES_A_CLEAN_RECOVERY_AND_THE_VERIFIED_ARTIFACT(tmp_path, fault):
    root = TM._tree(tmp_path)
    TM._current(root)
    cfg = TM._cfg(root)
    if fault in ("absent", "only_failed_fit"):
        if fault == "only_failed_fit":                                   # entry 73's situation: a FAILED fit only
            TM._stage_evidence(cfg, "fit", rec_extra={"status": "FAILED", "verdict": "ELIGIBLE"})
        want = "absent, uncommitted"
    else:
        rec_over, ch_over, outer_over = {}, {}, {}
        if fault == "stop":
            rec_over, want = {"status": "STOP"}, "not a COMPLETED recovery"
        elif fault == "outer":
            outer_over, want = {"survivors": "unconfirmed cleanup"}, "outer.json is not clean"
        elif fault == "ceiling":
            outer_over, want = {"term_s": 3_590}, "ceiling"
        elif fault == "unverified":
            ch_over, want = {"verified": False}, "not verified"
        elif fault == "reload_digest":
            ch_over, want = {"reload_state_digest": "3" * 64}, "not verified"
        elif fault == "unbound":
            rec_over, want = {"links": {"failed_record_sha256": "4" * 64}}, "not bound to the failed fit record"
        elif fault == "amendment":
            rec_over, want = {"amendment_sha256": "5" * 64}, "not a COMPLETED recovery"
        elif fault == "verdict":
            rec_over, want = {"verdict": M.GATE_MISSED}, "only an ELIGIBLE, verified challenger"
        else:
            want = {"pgid": "pgid.txt", "tampered": "not the one the recovery saved"}[fault]
        ch = TM._recovery_evidence(cfg, tmp_path, rec_over=rec_over, ch_over=ch_over, **outer_over)
        if fault == "pgid":
            (root / R.OUTER / "pgid.txt").write_text("1\n")
        elif fault == "tampered":
            ch.write_bytes(b"other weights")
    run, calls = TM._fake_runner()
    with pytest.raises(P.Refused, match=want):
        XM.run_export(cfg, runner=run, **TM._xargs(tmp_path))
    assert calls == [] and not (root / M.EV["export"]).exists()
