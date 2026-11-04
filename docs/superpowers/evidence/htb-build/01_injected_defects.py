import hashlib, pathlib, subprocess, sys
ROOT = pathlib.Path("/Users/bill/projects/TwixT_Game")
MOD = ROOT / "scripts/GPU/alphazero/htb_pilot.py"
orig = MOD.read_bytes(); osha = hashlib.sha256(orig).hexdigest()
MUT = [
 ("raw teacher target", "        target = 0.5 * (p + rp_back)\n", "        target = p\n"),
 ("raw teacher value", '"value": 0.5 * (v + rv)', '"value": v'),
 ("no val leakage filter", "if k not in train_keys}", "if True}"),
 ("key without reflection", "return _sha(min(state_serial(state), state_serial(state, reflect=True)))", "return _sha(state_serial(state))"),
 ("policy index colour fixed", "sel = z[[A.policy_index(colour, r, c)", "sel = z[[A.policy_index(\"red\", r, c)"),
 ("CHW boards", "            board = board_hwc(s)\n", "            board = s.to_tensor()\n"),
 ("eval-mode check off", "    training = [n or \"<root>\" for n, m in network.named_modules() if m.training]\n", "    training = []\n"),
 ("F2 not strict", 'student["top1"] > reference["top1"]', 'student["top1"] >= reference["top1"]'),
 ("F1 factor ignored", 'student["kl"] <= F1_FACTOR * reference["kl"]', 'student["kl"] <= reference["kl"]'),
 ("epoch ignored", 'f"{FIT_TAG}|{epoch}|{keys[i]}"', 'f"{FIT_TAG}|{keys[i]}"'),
 ("mirror orientation dropped", 'recs += [examples[i]["record"], examples[i]["mirror"]]', 'recs += [examples[i]["record"], examples[i]["record"]]'),
 ("value term dropped", "policy + recipe.value_weight * value", "policy + 0.0 * value"),
 ("reflection check off", "        if not reflect_state_matches(s, rs):\n", "        if False:\n"),
 ("MA uses one orientation", "probs.append(0.5 * (np.exp(lp1[j, :n]) + np.exp(lp2[j, :n])))", "probs.append(np.exp(lp1[j, :n]))"),
 ("shard sha not checked", "        if _sha(data) != sh[\"sha256\"]:\n", "        if False:\n"),
 ("gate check off", '    if getattr(gate_module, gate_name) is not True:\n        raise Refused(f"{gate_name} is closed")\n', '    pass\n'),
]
res = []
try:
    for name, old, new in MUT:
        src = orig.decode()
        if src.count(old) != 1:
            res.append((name, "ANCHOR MISSING")); continue
        MOD.write_text(src.replace(old, new))
        r = subprocess.run([str(ROOT/".venv/bin/python"), "-B", "-m", "pytest", "tests/test_htb_pilot.py", "-q", "-x",
                            "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True,
                           env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin", "HOME": str(pathlib.Path.home())})
        last = [l for l in r.stdout.splitlines() if l.strip()][-1]
        res.append((name, "CAUGHT" if r.returncode != 0 else "NOT CAUGHT", last[:110]))
finally:
    MOD.write_bytes(orig)
assert hashlib.sha256(MOD.read_bytes()).hexdigest() == osha
for x in res: print(" | ".join(x))
print("restored", osha[:12])
