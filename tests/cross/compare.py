"""Compare ppocr_replay output with expected_cpu_texts.json.
usage: compare.py CASE_DIR REPLAY_OUTPUT [--raw-ref OTHER_REPLAY_OUTPUT]"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
case_dir = Path(sys.argv[1])
manifest = json.load(open(case_dir / "manifest.json"))
expected = json.load(open(ROOT / "python/tests/data/expected_cpu_texts.json", encoding="utf-8"))


def parse(path):
    out, cur = {}, None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("== "):
            cur = Path(line.split()[1]).name
            out[cur] = []
        elif cur is not None and "\t" in line:
            score, text = line.split("\t", 1)
            out[cur].append((float(score), text))
    return out


got = parse(sys.argv[2])
bad = 0
for case in manifest["cases"]:
    lines = got.get(case["file"], [])
    kept = [t for s, t in lines if t.strip() and s >= manifest["text_score"]]
    want = expected[case["name"]]
    diffs = sum(a != b for a, b in zip(kept, want)) + abs(len(kept) - len(want))
    print(f"{case['name']}: {len(kept)} lines, {diffs} diffs")
    bad += diffs
if len(sys.argv) > 4 and sys.argv[3] == "--raw-ref":
    ref = parse(sys.argv[4])
    raw = sum([t for _, t in ref.get(k, [])] != [t for _, t in v] for k, v in got.items())
    dmax = max((abs(a[0] - b[0]) for k, v in got.items() for a, b in zip(ref.get(k, []), v)), default=0.0)
    print(f"raw text cases differing from reference run: {raw}; max |score diff| {dmax:.2e}")
    bad += raw
sys.exit(1 if bad else 0)
