"""Metal with the default PP-OCRv6 small models (transformer recognizer).

Regression tests for two M4 bugs:

* every Metal run failed with "GPU-only recognizer batch/CTC graph failed"
  because the small recognizer's FusedScaleShift had no GPU kernel;
* CPU attention rows became NaN on wide lines (T >= ~290) under -ffast-math,
  giving empty text that changed from run to run.

Models come from PPOCR_TEST_SMALL_DET / _REC / _DICT, or are downloaded like
``RapidOCR()`` does. Metal tests skip without a Metal device; the CPU wide
line test runs everywhere. PPOCR_LIBRARY may point at a local librapidocrlite.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"

# Runs in a fresh process: the GPU driver (Metal or Vulkan) is chosen once per
# process, so every backend gets its own interpreter.
_SCRIPT = r"""
import json, os, sys
from rapidocr import RapidOCR

backend, images = sys.argv[1], sys.argv[2:]
params = {"Global.use_cls": False, "Global.text_score": 0.0,
          "EngineConfig.ppocr_cpp.backend": backend}
for key, env in (("Det.model_path", "PPOCR_TEST_SMALL_DET"),
                 ("Rec.model_path", "PPOCR_TEST_SMALL_REC"),
                 ("Rec.rec_keys_path", "PPOCR_TEST_SMALL_DICT")):
    if os.environ.get(env):
        params[key] = os.environ[env]
engine = RapidOCR(params=params)

def full(path):
    r = engine(path, use_det=True, use_rec=True)
    boxes = [] if r.boxes is None else [[float(v) for v in b.reshape(-1)] for b in r.boxes]
    return {"txts": list(r.txts or ()), "boxes": boxes}

out = {"pages": {}, "repeat": {}}
for path in images:
    out["pages"][path] = full(path)
    out["repeat"][path] = full(path)
out["active"] = engine._engine.active_backend()
wide = images[-1]
out["rec_only"] = list(engine(wide, use_det=False, use_rec=True).txts)
out["lines"] = [list(engine(p, use_det=False, use_rec=True).txts)
                for p in os.environ.get("PPOCR_TEST_LINES", "").split(os.pathsep) if p]
det = engine(images[0], use_det=True, use_rec=False)
out["det_only"] = [] if det.boxes is None else [[float(v) for v in b.reshape(-1)] for b in det.boxes]
print("JSON" + json.dumps(out))
"""


def _line_image(width: int):
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # Pillow < 10.1 has no scalable default font
        pytest.skip("needs Pillow >= 10.1 for a scalable default font")
    text = ("ipsum dolor eiusmod tempor OCR amet ipsum tempor consectetur Metal "
            "tempor tempor lorem elit 2026 adipiscing sed do")
    image = Image.new("RGB", (width, 21), "white")
    ImageDraw.Draw(image).text((2, 0), text, fill="black", font=font)
    return image


@pytest.fixture(scope="module")
def wide_lines(tmp_path_factory):
    """~1000-1100 x 21 text lines: 2300-2500 px at rec height 48 (T 285-315).

    Which widths hit the old NaN softmax rows depended on the data, so a
    sweep of widths is used; before the fix about half of them read empty.
    """
    folder = tmp_path_factory.mktemp("metal")
    paths = []
    for width in range(1000, 1104, 8):
        path = folder / f"line_{width}.png"
        _line_image(width).save(path)
        paths.append(str(path))
    return paths


@pytest.fixture(scope="module")
def wide_line(wide_lines):
    return wide_lines[-1]


def _run(backend: str, images, lines=()) -> dict:
    env = os.environ.copy()
    env["PPOCR_TEST_LINES"] = os.pathsep.join(lines)
    completed = subprocess.run(
        [sys.executable, "-c", _SCRIPT, backend, *images],
        env=env, check=False, capture_output=True, text=True,
    )
    if completed.returncode != 0 and "download" in completed.stderr.lower():
        pytest.skip("small models are not available offline: " + completed.stderr[-300:])
    assert completed.returncode == 0, completed.stderr
    line = [l for l in completed.stdout.splitlines() if l.startswith("JSON")][-1]
    result = json.loads(line[4:])
    result["stderr"] = completed.stderr
    return result


def _iou(a, b) -> float:
    ax, ay, bx, by = a[0::2], a[1::2], b[0::2], b[1::2]
    a0, a1, a2, a3 = min(ax), min(ay), max(ax), max(ay)
    b0, b1, b2, b3 = min(bx), min(by), max(bx), max(by)
    iw = max(0.0, min(a2, b2) - max(a0, b0))
    ih = max(0.0, min(a3, b3) - max(a1, b1))
    inter = iw * ih
    union = (a2 - a0) * (a3 - a1) + (b2 - b0) * (b3 - b1) - inter
    return inter / union if union > 0 else 1.0


def _images(wide_line):
    return [str(DATA / "hello.png"), str(DATA / "page_zh.png"), wide_line]


def test_cpu_wide_lines_are_read_and_deterministic(wide_line, wide_lines):
    cpu = _run("cpu", _images(wide_line), wide_lines)
    for path, page in cpu["pages"].items():
        assert page == cpu["repeat"][path], path
    empty = [p for p, t in zip(wide_lines, cpu["lines"]) if not "".join(t).strip()]
    assert not empty, f"CPU read wide lines as empty text: {empty}"


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal runs on macOS only")
def test_metal_small_matches_cpu(wide_line, wide_lines):
    from rapidocr import _native

    if not _native.metal_available():
        pytest.skip("no Metal device")
    images = _images(wide_line)
    metal = _run("metal", images, wide_lines)
    cpu = _run("cpu", images, wide_lines)
    if metal["active"] is not None:
        assert metal["active"] == _native.PPOCR_BACKEND_METAL, metal["stderr"]
    for path in images:
        m, c = metal["pages"][path], cpu["pages"][path]
        assert m == metal["repeat"][path], f"Metal not deterministic on {path}"
        assert m["txts"] == c["txts"], path
        assert len(m["boxes"]) == len(c["boxes"]), path
        for left, right in zip(c["boxes"], m["boxes"]):
            assert _iou(left, right) >= 0.99, (path, left, right)
    # Partial modes run on the CPU graph with Metal instead of raising.
    assert metal["rec_only"] == cpu["rec_only"]
    assert metal["lines"] == cpu["lines"]
    assert len(metal["det_only"]) == len(cpu["det_only"]) > 0
