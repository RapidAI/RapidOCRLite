"""Text-line orientation classifier (Global.use_cls), mirroring rapidocr main
python/tests/test_cls.py (head 8ac45f4, including the #748 long-line case).

Runs against the native library (PPOCR_LIBRARY) with the tiny CI models and
the cached v2.0 classifier (PPOCR_CLS_MODEL). The PP-OCRv5 PP-LCNet cases
download their model on first use and are skipped when that fails.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from rapidocr import ModelType, OCRVersion, RapidOCR

DATA = Path(__file__).resolve().parent / "data"


def _paths():
    det = os.environ.get("PPOCR_TEST_DET")
    rec = os.environ.get("PPOCR_TEST_REC")
    dictionary = os.environ.get("PPOCR_TEST_DICT")
    cls = os.environ.get("PPOCR_CLS_MODEL")
    if not (os.environ.get("PPOCR_LIBRARY") and det and rec and dictionary and cls):
        pytest.skip("set PPOCR_LIBRARY, PPOCR_TEST_DET/REC/DICT and PPOCR_CLS_MODEL")
    if not all(Path(p).is_file() for p in (det, rec, dictionary, cls)):
        pytest.skip("CI models are missing")
    return det, rec, dictionary, cls


def _engine(**extra):
    det, rec, dictionary, cls = _paths()
    params = {
        "Det.model_path": det,
        "Rec.model_path": rec,
        "Rec.rec_keys_path": dictionary,
        "Det.model_type": "tiny",
        "Rec.model_type": "tiny",
        "Cls.model_path": cls,
        "Global.log_level": "error",
    }
    params.update(extra)
    return RapidOCR(params=params)


def _bgr(name: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(DATA / name).convert("RGB"))[:, :, ::-1].copy()


def _rot180(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[::-1, ::-1])


def test_use_cls_is_on_by_default():
    engine = _engine()
    assert engine.use_cls is True


def test_upside_down_page_is_read_with_cls():
    engine = _engine()
    flipped = _rot180(_bgr("hello.png"))
    with_cls = engine(flipped, use_det=True, use_cls=True, use_rec=True)
    joined = " ".join(with_cls.txts or ())
    assert "Hello" in joined and "RapidOCR" in joined and "123" in joined, with_cls.txts
    without = engine(flipped, use_det=True, use_cls=False, use_rec=True)
    assert "RapidOCR" not in " ".join(without.txts or ()), without.txts


def test_upright_page_unchanged_by_cls():
    engine = _engine()
    img = _bgr("hello.png")
    assert engine(img, use_det=True, use_cls=True, use_rec=True).txts == engine(
        img, use_det=True, use_cls=False, use_rec=True
    ).txts


def test_cls_only_returns_cls_output():
    engine = _engine()
    result = engine(DATA / "text_cls.jpg", use_det=False, use_cls=True, use_rec=False)
    assert type(result).__name__ == "TextClsOutput"
    assert result.cls_res is not None and len(result.cls_res) == 1
    assert result.cls_res[0][0] == "180"
    assert len(result.img_list) == 1


def test_ppocrv4_cls_long_line():
    # main #748: a long line is classified on its centre at the model aspect
    # ratio instead of being squeezed into 192x48.
    engine = _engine()
    img = _bgr("test_letterbox_like.jpg")
    thresh = engine.cfg.Cls.cls_thresh

    label, score = engine(img, use_det=False, use_cls=True, use_rec=False).cls_res[0]
    assert not ("180" in label and score > thresh)

    label, score = engine(_rot180(img), use_det=False, use_cls=True, use_rec=False).cls_res[0]
    assert "180" in label and score > thresh


def test_env_override_disables_cls(monkeypatch):
    monkeypatch.setenv("PPOCR_USE_CLS", "0")
    engine = _engine()
    assert engine.use_cls is False


@pytest.mark.parametrize("model_type", [ModelType.MOBILE, ModelType.SERVER])
def test_ppocrv5_cls(model_type, tmp_path):
    det, rec, dictionary, _ = _paths()
    try:
        engine = RapidOCR(
            params={
                "Det.model_path": det,
                "Rec.model_path": rec,
                "Rec.rec_keys_path": dictionary,
                "Det.model_type": "tiny",
                "Rec.model_type": "tiny",
                "Cls.ocr_version": OCRVersion.PPOCRV5,
                "Cls.model_type": model_type,
                "Global.model_root_dir": str(tmp_path),
                "Global.log_level": "error",
            }
        )
        result = engine(DATA / "text_cls.jpg", use_det=False, use_cls=True, use_rec=False)
    except Exception as exc:  # network
        if "download" in str(exc).lower() or "url" in str(exc).lower():
            pytest.skip(f"PP-OCRv5 cls model unavailable: {exc}")
        raise
    assert result.cls_res is not None
    assert result.cls_res[0][0] == "180"
