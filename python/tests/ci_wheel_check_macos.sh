#!/bin/bash
# macOS wheel check used by cibuildwheel: the shared wheel check, then the
# deployment target, the architecture, Metal compiled in, and hello OCR.
#
# Environment knobs (set by pyproject.toml [tool.cibuildwheel.overrides] for the
# Intel wheels):
#   MACOS_CHECK_ARCH        expected lipo arch      (default arm64)
#   MACOS_CHECK_MINOS       expected minos          (default 14.0)
#   MACOS_CHECK_METAL_OCR   run the OCR pass with   (default 1)
#                           use_metal=True
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
bash "$ROOT/python/tests/ci_wheel_check.sh"

MACOS_CHECK_ARCH="${MACOS_CHECK_ARCH:-arm64}" \
MACOS_CHECK_MINOS="${MACOS_CHECK_MINOS:-14.0}" \
MACOS_CHECK_METAL_OCR="${MACOS_CHECK_METAL_OCR:-1}" \
python - << 'PY'
import os
import subprocess
from pathlib import Path

import rapidocr
from rapidocr import _native

arch = os.environ["MACOS_CHECK_ARCH"]
minos = os.environ["MACOS_CHECK_MINOS"]
metal_ocr = os.environ["MACOS_CHECK_METAL_OCR"] == "1"
print("checking arch=%s minos=%s metal_ocr=%s" % (arch, minos, metal_ocr))

root = Path(rapidocr.__file__).resolve().parent
libs = sorted(p for p in root.glob("librapidocrlite*.dylib") if p.is_file() and not p.is_symlink())
assert libs, list(root.iterdir())
for lib in libs:
    out = subprocess.run(["vtool", "-show-build", str(lib)], check=True,
                         capture_output=True, text=True).stdout
    print(lib.name, out)
    assert "minos %s" % minos in out, out
    got = subprocess.run(["lipo", "-archs", str(lib)], check=True,
                         capture_output=True, text=True).stdout.split()
    assert got == [arch], got

# Metal must be compiled in: linked against Metal.framework and carrying the
# embedded MSL kernels. GitHub's arm64 macOS VMs expose no Metal device, so a
# missing device only downgrades the runtime check (backend=metal must then
# fall back to CPU and still read the text). The Intel runners are
# GPU-virtualised VMs where a Metal command buffer hangs, so the Intel override
# sets MACOS_CHECK_METAL_OCR=0 and verifies via the CPU path instead.
for lib in libs:
    links = subprocess.run(["otool", "-L", str(lib)], check=True, capture_output=True, text=True).stdout
    assert "Metal.framework" in links, links
    syms = subprocess.run(["nm", "-g", str(lib)], check=True, capture_output=True, text=True).stdout
    assert "kMetalBinaryMsl" in syms and "_rpocr_metal_device_available" in syms, lib
print("metal_built_ok")

from rapidocr import RapidOCR

models = Path(os.environ.get("PPOCR_CI_MODELS", Path.cwd() / "ci-models"))
engine = RapidOCR(params={
    "Det.model_path": str(models / "PP-OCRv6_det_tiny.onnx"),
    "Rec.model_path": str(models / "PP-OCRv6_rec_tiny.onnx"),
    "Rec.rec_keys_path": str(models / "ppocrv6_tiny_dict.txt"),
    "Det.model_type": "tiny",
    "Rec.model_type": "tiny",
    "Global.use_cls": False,
    "EngineConfig.ppocr_cpp.use_metal": metal_ocr,
})
result = engine("python/tests/data/hello.png")
print("hello", result.txts, result.scores)
assert result.txts == ("Hello RapidOCR 123",), result.txts
if metal_ocr:
    has_device = _native.metal_available()
    print("metal_device_available", has_device)
    print("metal_hello_ocr_ok" if has_device else "metal_fallback_hello_ocr_ok")
else:
    print("cpu_hello_ocr_ok")
PY
echo "ci_wheel_check_macos_ok"
