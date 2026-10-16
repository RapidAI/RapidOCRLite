"""ppocr_options.model_type through the Python binding (Det/Rec model_type 'lite')."""
import ctypes
from pathlib import Path

import pytest

from rapidocr import _native

ROOT = Path(__file__).resolve().parents[2]
LITE = ROOT / "python/src/rapidocr/models/lite"
DET, REC, DICT = (ROOT / "ci-models" / n for n in ("PP-OCRv6_det_tiny.onnx", "PP-OCRv6_rec_tiny.onnx", "ppocrv6_tiny_dict.txt"))


def opts(mt=None):
    o = _native.default_options()
    o.backend = _native.PPOCR_BACKEND_CPU
    if mt is not None:
        o.model_type = mt
    return o


def test_defaults():
    o = _native.default_options()
    assert o.struct_size == ctypes.sizeof(_native.Options) and o.model_type == _native.PPOCR_MODEL_AUTO


@pytest.mark.parametrize("mt", [None, _native.PPOCR_MODEL_AUTO, _native.PPOCR_MODEL_LITE])
def test_lite_ok(mt):
    assert "lite" in _native.NativeOCR(str(LITE), "", "", opts(mt)).backend_note().lower()


def test_old_struct_size_is_auto():
    o = opts(99)  # garbage beyond the old layout must be ignored
    o.struct_size = _native.Options.model_type.offset
    _native.NativeOCR(str(LITE), "", "", o)
    _native.NativeOCR(str(DET), str(REC), str(DICT), o)


def test_ppocr_explicit():
    _native.NativeOCR(str(DET), str(REC), str(DICT), opts(_native.PPOCR_MODEL_PPOCR))


@pytest.mark.parametrize("det,mt,msg", [
    (DET, _native.PPOCR_MODEL_LITE, "meta.json"),
    (LITE, _native.PPOCR_MODEL_PPOCR, "lite model folder"),
    (LITE, 7, "model_type"),
])
def test_mismatch(det, mt, msg):
    with pytest.raises(RuntimeError, match=msg):
        _native.NativeOCR(str(det), str(REC), str(DICT), opts(mt))


