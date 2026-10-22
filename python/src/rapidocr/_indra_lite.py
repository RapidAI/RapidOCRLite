"""model_type="lite": bundled Indra OCR weights (python/src/rapidocr/models/lite).

A thin wrapper over librapidocrlite's C API, like the default models: rpocr_create(<lite folder>, ...) selects the
C++ Indra pipeline (src/indra/lite_pipeline.cpp, ported from indra-ocr 0.3.2, Apache-2.0): deskew, line
detection, script routing, CTC recognition with both vocabularies, line splitting, reading order. No Python
pipeline code, onnxruntime, OpenCV or indra-ocr at runtime.
PPOCR_INDRA_BACKEND=ort runs the upstream indra_ocr.OrtOCR instead: a test-only reference that needs
indra-ocr, onnxruntime and opencv installed separately.
The weights are licensed under a modified OpenRAIL-M license (models/lite/LICENSE, NOTICE).
"""
from __future__ import annotations
import os

import hashlib
from pathlib import Path

from . import _native

_PACKAGE_DIR = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bundled_model_dir(cfg) -> Path:
    """Folder with the Indra files: Det.model_dir if set, else the bundled route from
    default_models.yaml (onnxruntime -> PP-OCRv6 -> det -> lite_indra_det). Checks SHA256.
    Never downloads."""
    override = cfg.Det.get("model_dir")
    if override:
        return Path(override).expanduser()
    from .typings import ModelType, OCRVersion, TaskType
    from .utils.model_resolver import _MODEL_CONFIG, route_to_model_key

    key = route_to_model_key(TaskType.DET, OCRVersion.PPOCRV6, cfg.Det.get("lang_type", "ch"), ModelType.LITE)
    entry = _MODEL_CONFIG["onnxruntime"]["PP-OCRv6"]["det"][key]
    folder = _PACKAGE_DIR / entry["bundled_dir"]
    for name, sha in entry["files"].items():
        p = folder / name
        if not p.is_file():
            raise FileNotFoundError(f"bundled lite model file missing: {p}")
        if _sha256(p) != sha:
            raise ValueError(f"SHA256 mismatch for bundled lite model file {p}")
    return folder


class IndraLiteEngine:
    def __init__(self, model_dir: Path, threads: int = 4, turn: bool = False):
        self.backend = os.environ.get("PPOCR_INDRA_BACKEND", "native").lower()
        self.ocr = None
        self.native = None
        if self.backend == "ort":  # test-only reference: upstream indra-ocr on onnxruntime + OpenCV
            try:
                from indra_ocr.pipeline import OrtOCR
            except ImportError as e:
                raise ImportError("PPOCR_INDRA_BACKEND=ort is a test reference: pip install indra-ocr==0.3.2") from e
            self.ocr = OrtOCR(str(model_dir), threads=threads)  # local folder only: no network
            self._ort_round = os.environ.get("PPOCR_INDRA_BOX_ROUND", "1") != "0"
            if self._ort_round:
                import numpy as np
                ro_fn = self.ocr.reading_order

                def _order_rounded(boxes, *a, _f=ro_fn, **k):
                    if len(boxes):
                        boxes = boxes.copy()
                        boxes[:, :4] = np.round(boxes[:, :4])
                    return _f(boxes, *a, **k)
                self.ocr.reading_order = _order_rounded
        else:
            # The C++ engine: det_model = the lite folder (meta.json); rec_model / dictionary unused.
            options = _native.default_options()
            options.backend = _native.RPOCR_BACKEND_CPU
            options.rec_parallelism = int(threads)
            options.model_type = _native.RPOCR_MODEL_LITE  # Det/Rec model_type "lite"
            self.native = _native.NativeOCR(str(model_dir), "", "", options)
        self.turn = turn

    def set_det_thresholds(self, det_threshold, box_threshold, unclip_ratio):
        """Accepted for contract compatibility; Indra's line decoder has fixed thresholds."""
        self._thr = (det_threshold, box_threshold, unclip_ratio)

    def run(self, rgb, tasks):
        if self.native is None:
            return self._run_ort(rgb, tasks)
        boxes, txts, scores = self.native.run(rgb, tasks)
        if tasks == _native.RPOCR_TASK_DET:
            return boxes, [], scores
        if tasks == _native.RPOCR_TASK_REC:
            return [], txts, scores
        # full page: the engine reports boxes at 0.1 px and confidences at 1e-4 (float32 in the C ABI)
        boxes = [[[round(x, 1), round(y, 1)] for x, y in b] for b in boxes]
        return boxes, txts, [round(s, 4) for s in scores]

    def _run_ort(self, rgb, tasks):
        import numpy as np
        import cv2
        g = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
        quad = lambda b: [[float(b[0]), float(b[1])], [float(b[2]), float(b[1])], [float(b[2]), float(b[3])], [float(b[0]), float(b[3])]]
        if tasks == _native.RPOCR_TASK_DET:
            b = self.ocr.detect(g)
            return [quad(x) for x in b], [], [float(x[4]) for x in b]
        if tasks == _native.RPOCR_TASK_REC:
            h, w = g.shape
            p = max(16, h // 2)
            gp = np.pad(g, p, mode="constant", constant_values=255)
            t, c, _ = self.ocr.recognize(gp, np.array([[p, p, p + w, p + h]], np.float32))[0]
            return [], [t], [float(c)]
        page = self.ocr(g, turn=self.turn)
        lines = [l for l in page.lines if l.text]
        return [quad(l.box) for l in lines], [l.text for l in lines], [float(l.conf) for l in lines]

    def recognize(self, rgb):
        return self.run(rgb, _native.RPOCR_TASK_DET | _native.RPOCR_TASK_REC)

    def has_chars(self) -> bool:
        return self.native is not None and self.native.has_chars()

    def run_ex(self, rgb, tasks, chars=None, c_words=None):
        """run() plus the per-character data of return_word_box (native engine only)."""
        boxes, txts, scores, cls = self.native.run_ex(rgb, tasks, chars, c_words)
        if tasks == _native.RPOCR_TASK_REC:
            return [], txts, scores, cls
        boxes = [[[round(x, 1), round(y, 1)] for x, y in b] for b in boxes]
        return boxes, txts, [round(s, 4) for s in scores], cls

    def active_backend(self):
        return _native.RPOCR_BACKEND_CPU

    def backend_note(self):
        if self.native is not None:
            return self.native.backend_note()
        return "bundled Indra OCR models on onnxruntime (model_type=lite, PPOCR_INDRA_BACKEND=ort test reference)"
