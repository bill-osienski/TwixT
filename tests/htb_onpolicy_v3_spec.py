"""V3's mirror-averaged parity tools for the H-OP challenger, DERIVED from tests/parity/htb_mirror/ (card §9).

The reviewed V3 tools hard-code the current student (python side) and resolve the pinned hard model (node side). The
H-OP copies take the challenger from the command line instead; everything else is the originals' text, with
exact-once substitutions. The comparator (tests/parity/htb_mirror/compare.mjs) is reused unmodified.
tests/test_htb_onpolicy.py re-derives both copies and compares bytes.

    .venv/bin/python -m tests.htb_onpolicy_v3_spec
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC, DST = "tests/parity/htb_mirror", "tests/parity/htb_onpolicy_mirror"
NOTE_PY = ("# H-OP COPY (tests/htb_onpolicy_v3_spec.py): the challenger file, its sha256 and its state digest come from\n"
           "# the command line (argv 2-4, from the H-OP fit record); otherwise tests/parity/htb_mirror/python_side.py.\n")
NOTE_JS = ("// H-OP COPY (tests/htb_onpolicy_v3_spec.py): the challenger's exported pair is resolved from its directory\n"
           "// (argv 3) through resolveModel; otherwise tests/parity/htb_mirror/node_side.mjs.\n")
SUBS = {
    "python_side.py": [
        ('STUDENT = pathlib.Path.home() / "twixt-cache/htb-pilot/student.safetensors"\n',
         NOTE_PY + "STUDENT = pathlib.Path(sys.argv[2])\n"),
        ('STUDENT_SHA256 = "0e1cb6211aa0b3e2dec6ab6f39bec74ceb8c472c7dcf603e7d331b1d1c1c34e5"\n',
         "STUDENT_SHA256 = sys.argv[3]\n"),
        ('STUDENT_DIGEST = "586dca4fcdb87c2611449099b4b7acfda460b13b6717c573a55011b8620a630b"\n',
         "STUDENT_DIGEST = sys.argv[4]\n"),
        ('"side": "research MirrorAveragingEvaluator(student)"', '"side": "research MirrorAveragingEvaluator(H-OP challenger)"'),
    ],
    "node_side.mjs": [
        ("import { assertSessionContract, resolveHardModel } from '../../../server/model_manifest.js';\n",
         "import { assertSessionContract, resolveModel } from '../../../server/model_manifest.js';\n"),
        ("const hard = await resolveHardModel();\n",
         NOTE_JS + "const hard = await resolveModel({ MODEL_MANIFEST: join(process.argv[3], 'manifest.json') });\n"),
        ("side: 'product MirrorAveragingInference(hard model)'", "side: 'product MirrorAveragingInference(H-OP challenger)'"),
    ],
}


def derive(name: str) -> str:
    text = (ROOT / SRC / name).read_text()
    for old, new in SUBS[name]:
        if text.count(old) != 1:
            raise ValueError(f"{name}: {old!r} occurs {text.count(old)} times")
        text = text.replace(old, new)
    return text


def main() -> None:
    (ROOT / DST).mkdir(parents=True, exist_ok=True)
    for name in SUBS:
        with open(ROOT / DST / name, "x") as fh:
            fh.write(derive(name))
        print(f"wrote {DST}/{name}")


if __name__ == "__main__":
    main()
