"""Lite-model regression on the README sample images (docs/lite_examples).

Runs model_type="lite" on the six sample images and compares every line with
docs/lite_examples/results/<name>_lite.json: text and box corners (0.1 px) exactly,
score (saved with 4 decimals) within 1e-4. Prints the
number of identical images and the per-image time; exits 1 on any difference.

PPOCR_FORCE_ISA (scalar / avx2 / avx512) selects the float kernels as usual.
"""
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from rapidocr import RapidOCR

ROOT = Path(__file__).resolve().parents[2]
EX = ROOT / "docs" / "lite_examples"
NAMES = ["ch_en_num.jpg", "check_return_word_len.jpeg", "return_word_debug.jpg",
         "synthetic_dense_mixed.png", "synthetic_scan_report.jpg", "text_vertical_words.jpg"]


def lines_of(r):
    boxes = [] if r.boxes is None else r.boxes.tolist()
    return [{"box": [[round(x, 1), round(y, 1)] for x, y in b], "text": t, "score": round(float(s), 4)}
            for b, t, s in zip(boxes, list(r.txts or []), list(r.scores or []))]


def run(threads):
    eng = RapidOCR(params={"Det.model_type": "lite", "Rec.model_type": "lite", "Global.log_level": "error",
                           "EngineConfig.onnxruntime.intra_op_num_threads": threads})
    same, bad = 0, []
    for fn in NAMES:
        want = json.loads((EX / "results" / (Path(fn).stem + "_lite.json")).read_text(encoding="utf-8"))["lines"]
        img = np.asarray(Image.open(EX / "images" / fn).convert("RGB"))
        r = eng(img, use_cls=False)
        ts = []
        for _ in range(int(os.environ.get("LITE_REPS", "2"))):
            t = time.perf_counter(); eng(img, use_cls=False); ts.append((time.perf_counter() - t) * 1000)
        got = lines_of(r)
        exp = [{"box": l["box"], "text": l["text"], "score": l["score"]} for l in want]
        exact = got == exp
        # text and boxes must match exactly; a score may differ by one unit of its 4th decimal
        # (MSVC AVX-512 build: one line of check_return_word_len.jpeg reads 0.8533 vs 0.8534)
        ok = exact or (len(got) == len(exp) and all(
            g["text"] == e["text"] and g["box"] == e["box"] and abs(g["score"] - e["score"]) <= 1e-4 + 1e-9
            for g, e in zip(got, exp)))
        same += ok
        med = statistics.median(ts) if ts else float("nan")
        tag = "identical" if exact else ("score_within_1e-4" if ok else "DIFFERENT")
        print(f"{fn}: lines={len(got)} {tag} median_ms={med:.1f}", flush=True)
        if not exact:
            for i in range(max(len(got), len(exp))):
                g = got[i] if i < len(got) else None
                e = exp[i] if i < len(exp) else None
                if g != e:
                    print(f"  line {i + 1}: got={g} want={e}")
        if not ok:
            bad.append(fn)
    print(f"lite_examples identical={same}/{len(NAMES)} isa={os.environ.get('PPOCR_FORCE_ISA', 'auto')} threads={threads}",
          flush=True)
    return not bad


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    # --matrix (or LITE_MATRIX=1): auto ISA at 4 and 1 threads, then AVX2 forced (x86); otherwise one run as configured.
    if "--matrix" in sys.argv or os.environ.get("LITE_MATRIX") == "1":
        runs = [("", 4), ("", 1)]
        if platform.machine().lower() in {"amd64", "x86_64", "x86", "i686", "i386"}:
            runs.append(("avx2", 4))
    else:
        runs = [(None, int(os.environ.get("LITE_THREADS", "4")))]
    ok = True
    for isa, threads in runs:
        if isa is not None:
            if isa:
                os.environ["PPOCR_FORCE_ISA"] = isa
            else:
                os.environ.pop("PPOCR_FORCE_ISA", None)
        ok = run(threads) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
