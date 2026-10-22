"""Capture the exact native inputs the Python wrapper produces for the
reference images (options struct, det thresholds, prepared RGB image) so a
cross-compiled librapidocrlite can replay them under an emulator (no Python needed
on the target). Usage: capture_cases.py OUT_DIR  (PPOCR_TEST_DET/REC/DICT set,
rapidocr importable with a native librapidocrlite for the host)."""
import ctypes
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

from rapidocr import RapidOCR, _native

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "python" / "tests" / "data"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

captured = {}
orig_init = _native.NativeOCR.__init__
orig_thr = _native.NativeOCR.set_det_thresholds
orig_run = _native.NativeOCR.run
orig_run_ex = _native.NativeOCR.run_ex


def init(self, det, rec, dictionary, options):
    captured["options"] = bytes(ctypes.string_at(ctypes.addressof(options), ctypes.sizeof(options)))
    orig_init(self, det, rec, dictionary, options)


def thr(self, a, b, c):
    captured["thr"] = (float(a), float(b), float(c))
    orig_thr(self, a, b, c)


def run(self, rgb, tasks):
    captured["image"] = np.ascontiguousarray(rgb, dtype=np.uint8)
    captured["tasks"] = int(tasks)
    return orig_run(self, rgb, tasks)


_native.NativeOCR.__init__ = init
_native.NativeOCR.set_det_thresholds = thr
def run_ex(self, rgb, tasks, *args, **kwargs):
    captured["image"] = np.ascontiguousarray(rgb, dtype=np.uint8)
    # use_cls is off here, so the bits equal rpocr_run's (the replay calls rpocr_run).
    captured["tasks"] = int(tasks) & ~_native.RPOCR_TASK_CLS
    return orig_run_ex(self, rgb, tasks, *args, **kwargs)


_native.NativeOCR.run = run
_native.NativeOCR.run_ex = run_ex

engine = RapidOCR(params={
    "Det.model_path": os.environ["PPOCR_TEST_DET"],
    "Rec.model_path": os.environ["PPOCR_TEST_REC"],
    "Rec.rec_keys_path": os.environ["PPOCR_TEST_DICT"],
    "Det.model_type": "tiny",
    "Rec.model_type": "tiny",
    "Global.use_cls": False,
})
expected = json.load(open(DATA / "expected_cpu_texts.json", encoding="utf-8"))
manifest = {"text_score": float(engine.cfg.Global.text_score), "cases": []}
for name in expected:
    engine(str(DATA / name))
    img = captured["image"]
    h, w = img.shape[:2]
    stem = name.rsplit(".", 1)[0]
    with open(out / f"{stem}.case", "wb") as f:
        f.write(struct.pack("<iiifff", w, h, captured["tasks"], *captured["thr"]))
        f.write(img.tobytes())
    manifest["cases"].append({"name": name, "file": f"{stem}.case"})
(out / "options.bin").write_bytes(captured["options"])
json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
print("captured", [c["name"] for c in manifest["cases"]])
