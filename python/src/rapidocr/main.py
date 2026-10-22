"""RapidOCR entry point backed by the native PP-OCRv6 engine.

The constructor and ``__call__`` follow ``rapidocr.main.RapidOCR`` on the
``main`` branch. This build always runs the in-process C++ ONNX interpreter
(detector, text-line orientation classifier and recognizer). ``use_cls``
(default on, as on ``main``) runs the 0/180 classifier between detection and
recognition with ``main``'s preprocessing. ``return_word_box`` /
``return_single_char_box`` give ``main``'s word and character boxes: the
engine records each decoded character's CTC time step and probability, and
``cal_rec_boxes`` (a port of ``main``'s CalRecBoxes) turns them into boxes.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from . import _native
from .inference_engine.base import FileInfo, InferSession
from .output import RapidOCROutput, TextClsOutput, TextDetOutput, TextRecOutput, filter_by_score
from .typings import EngineType, ModelType, OCRVersion, TaskType
from .utils.load_image import LoadImage, LoadImageError
from .utils.model_resolver import list_supported_langs as _list_supported_langs
from .utils.parse_parameters import ConfigNode, ParseParams
from .utils.process_img import apply_vertical_padding, map_boxes_to_original, resize_image_within_bounds

logger = logging.getLogger("rapidocr")
_PACKAGE_DIR = Path(__file__).resolve().parent

_NATIVE_ENGINES = {EngineType.ONNXRUNTIME.value, EngineType.PPOCR_CPP.value, "onnxruntime", "ppocr_cpp"}


class RapidOCRError(Exception):
    pass


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


class RapidOCR:
    @staticmethod
    def list_supported_langs(task_type, ocr_version, model_type=None):
        """Languages routed by ``model_routes`` in default_models.yaml, as on ``main``."""

        if isinstance(task_type, str):
            task_type = TaskType(task_type.lower())
        if isinstance(ocr_version, str):
            ocr_version = OCRVersion(ocr_version)
        if isinstance(model_type, str):
            model_type = ModelType(model_type.lower())
        return _list_supported_langs(task_type, ocr_version, model_type)

    def __init__(self, config_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        self.cfg = _load_config(config_path, params)
        level = str(self.cfg.Global.get("log_level", "info")).upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
        self.text_score = float(self.cfg.Global.text_score)
        self.use_det = bool(self.cfg.Global.use_det)
        self.use_cls = _env_bool("PPOCR_USE_CLS", bool(self.cfg.Global.use_cls))
        self.use_rec = bool(self.cfg.Global.use_rec)
        self.return_word_box = bool(self.cfg.Global.get("return_word_box", False))
        self.return_single_char_box = bool(self.cfg.Global.get("return_single_char_box", False))
        self.min_side_len = self.cfg.Global.get("min_side_len", 30)
        self.max_side_len = self.cfg.Global.get("max_side_len", 2000)
        self.min_height = self.cfg.Global.get("min_height", 30)
        self.width_height_ratio = self.cfg.Global.get("width_height_ratio", 8)
        self.text_det = None
        self.text_cls = None
        self.text_rec = None
        self.load_img = LoadImage()
        self._engine: Optional[_native.NativeOCR] = None
        self._warned_cls = False
        self._warned_word_backend = False
        self._engine_has_cls = False
        self._check_engine_choice()

    def update_params(
        self,
        use_det: Optional[bool] = None,
        use_cls: Optional[bool] = None,
        use_rec: Optional[bool] = None,
        return_word_box: Optional[bool] = None,
        return_single_char_box: Optional[bool] = None,
        text_score: Optional[float] = None,
        box_thresh: Optional[float] = None,
        unclip_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        values = {
            "use_det": use_det,
            "use_cls": use_cls,
            "use_rec": use_rec,
            "return_word_box": return_word_box,
            "return_single_char_box": return_single_char_box,
            "text_score": text_score,
            "box_thresh": box_thresh,
            "unclip_ratio": unclip_ratio,
        }
        values.update(kwargs)
        for key, value in values.items():
            if value is None:
                continue
            if key in {"box_thresh", "unclip_ratio"}:
                setattr(self.cfg.Det, key, value)
                if self.text_det is not None:
                    setattr(self.text_det.postprocess_op, key, value)
                continue
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter: {key}")
            setattr(self, key, value)
            if key in {"use_det", "use_cls", "use_rec", "return_word_box", "return_single_char_box", "text_score"}:
                setattr(self.cfg.Global, key if key != "text_score" else "text_score", value)

    def __call__(
        self,
        img_content: Any,
        use_det: Optional[bool] = None,
        use_cls: Optional[bool] = None,
        use_rec: Optional[bool] = None,
        return_word_box: Optional[bool] = None,
        return_single_char_box: Optional[bool] = None,
        text_score: Optional[float] = None,
        box_thresh: Optional[float] = None,
        unclip_ratio: Optional[float] = None,
    ) -> RapidOCROutput:
        self.update_params(
            use_det=use_det,
            use_cls=use_cls,
            use_rec=use_rec,
            return_word_box=return_word_box,
            return_single_char_box=return_single_char_box,
            text_score=text_score,
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
        )
        lite = self._is_lite()
        use_cls = bool(self.use_cls)
        if use_cls and lite:
            if not self._warned_cls:
                logger.warning("use_cls is ignored by the lite (Indra) model: it has no orientation classifier")
                self._warned_cls = True
            use_cls = False
        if not self.use_det and not self.use_rec and not use_cls:
            raise RapidOCRError("use_det, use_cls and use_rec cannot all be False")

        started = time.perf_counter()
        original = self.load_img(img_content)
        # main pads short pages only in detect_and_crop, so use_det=False
        # gets the bounded resize alone.
        if lite:  # Indra resizes internally (long side 1024, adaptive)
            prepared, ratio_w, ratio_h, pad_top = original, 1.0, 1.0, 0
        else:
            prepared, ratio_w, ratio_h, pad_top = _prepare_image(self, original, pad=self.use_det)
        engine = self._ensure_engine(need_cls=use_cls)
        if lite:
            return self._call_lite(engine, original, prepared, ratio_w, ratio_h, pad_top, started)
        cls_bit = _native.RPOCR_TASK_CLS if use_cls else 0
        labels = [str(x) for x in (self.cfg.Cls.get("label_list") or ["0", "180"])]
        cls_thresh = float(self.cfg.Cls.get("cls_thresh", 0.9))

        def label_of(index: int) -> str:
            return labels[index] if 0 <= index < len(labels) else str(index)

        def corrected(img: np.ndarray, item) -> np.ndarray:
            # main: cv2.rotate(img, ROTATE_180) when "180" in label and score > thresh
            index, score = item
            if index >= 0 and "180" in label_of(index) and score > cls_thresh:
                return np.ascontiguousarray(img[::-1, ::-1])
            return img

        det_threshold = float(self.cfg.Det.get("thresh", 0.3))
        box_threshold = float(self.cfg.Det.get("box_thresh", 0.5))
        unclip = float(self.cfg.Det.get("unclip_ratio", 1.6))
        engine.set_det_thresholds(det_threshold, box_threshold, unclip)
        if not self.use_det and not self.use_rec:
            # main returns TextClsOutput when only the classifier ran.
            _, _, _, cls = engine.run_ex(prepared, _native.RPOCR_TASK_CLS)
            elapsed = time.perf_counter() - started
            return TextClsOutput(
                img_list=[corrected(prepared, cls[0])],
                cls_res=[(label_of(cls[0][0]), cls[0][1])],
                elapse=elapsed,
            )
        if not self.use_rec:
            # main classifies here too but returns only the detector output.
            boxes, _, scores, _ = engine.run_ex(prepared, _native.RPOCR_TASK_DET)
            elapsed = time.perf_counter() - started
            if not boxes:
                return TextDetOutput()
            ori_h, ori_w = original.shape[:2]
            mapped = map_boxes_to_original(np.asarray(boxes, dtype=np.float32), ratio_w, ratio_h, pad_top, ori_w, ori_h)
            return TextDetOutput(img=original, boxes=mapped, scores=list(scores), elapse=elapsed)
        want_words = bool(self.return_word_box)
        chars = [] if want_words else None
        if not self.use_det:
            rec_tasks = _native.RPOCR_TASK_REC | cls_bit
            if want_words:
                _, txts, scores, cls = engine.run_ex(prepared, rec_tasks, chars)
            else:
                _, txts, scores, cls = engine.run_ex(prepared, rec_tasks)
            elapsed = time.perf_counter() - started
            img = corrected(original, cls[0]) if (use_cls and cls) else original
            word_results = tuple((t, s, None) for t, s in zip(txts, scores))
            if want_words:
                # main's TextRecognizer returns the raw WordInfo per line here.
                infos = self._word_infos(txts, chars)
                if infos is not None:
                    word_results = tuple(infos)
            return TextRecOutput(
                imgs=[img],
                txts=tuple(txts),
                scores=tuple(scores),
                word_results=word_results,
                elapse=elapsed,
            )
        full_tasks = _native.RPOCR_TASK_DET | _native.RPOCR_TASK_REC | cls_bit
        if want_words:
            boxes, txts, scores, _ = engine.run_ex(prepared, full_tasks, chars)
        else:
            boxes, txts, scores, _ = engine.run_ex(prepared, full_tasks)
        if want_words:
            return self._word_box_output(
                original, boxes, txts, scores, chars, ratio_w, ratio_h, pad_top, started
            )
        kept_boxes, kept_txts, kept_scores = filter_by_score(boxes, txts, scores, self.text_score)
        elapsed = time.perf_counter() - started
        if len(kept_txts) == 0:
            return RapidOCROutput()
        ori_h, ori_w = original.shape[:2]
        kept_boxes = map_boxes_to_original(kept_boxes, ratio_w, ratio_h, pad_top, ori_w, ori_h)
        return RapidOCROutput(
            img=original,
            boxes=kept_boxes,
            txts=kept_txts,
            scores=kept_scores,
            word_results=(),
            elapse_list=[None, None, elapsed],
        )

    def _word_infos(self, txts, chars):
        """main's CTCLabelDecode word info per line, or None (and a warning)
        when the backend that ran did not record per-character data."""

        from .cal_rec_boxes import build_word_info

        if chars is None or len(chars) != len(txts):
            if not self._warned_word_backend:
                logger.warning(
                    "return_word_box: the engine returned no per-character data for "
                    "some lines (old library?), so word_results stays empty"
                )
                self._warned_word_backend = True
            return None
        infos = []
        for txt, (steps, crop_w, crop_h, max_wh, cols, probs) in zip(txts, chars):
            if steps <= 0 and txt:
                # No per-character record for this line (a lite-model line
                # whose tokens do not map 1:1 onto its text): spread the
                # characters evenly over the line, 4 steps each.
                steps, crop_w, crop_h = 4 * len(txt), max(crop_w, 1), max(crop_h, 1)
                cols = [4 * k + 1 for k in range(len(txt))]
                probs = [1.0] * len(txt)
                max_wh = crop_w * 1.0 / crop_h
            elif steps <= 0:
                steps, crop_w, crop_h, max_wh = 1, max(crop_w, 1), max(crop_h, 1), 1.0
            infos.append(build_word_info(txt, cols, probs, steps, crop_w * 1.0 / crop_h, max_wh))
        return infos

    def _word_box_output(self, original, boxes, txts, scores, chars, ratio_w, ratio_h, pad_top, started):
        """main's build_final_output with return_word_box=True (det + rec)."""

        from .cal_rec_boxes import CalRecBoxes

        infos = self._word_infos(txts, chars)
        if infos is None:
            kept_boxes, kept_txts, kept_scores = filter_by_score(boxes, txts, scores, self.text_score)
            if len(kept_txts) == 0:
                return RapidOCROutput()
            ori_h, ori_w = original.shape[:2]
            kept_boxes = map_boxes_to_original(kept_boxes, ratio_w, ratio_h, pad_top, ori_w, ori_h)
            return RapidOCROutput(img=original, boxes=kept_boxes, txts=kept_txts, scores=kept_scores,
                                  word_results=(), elapse_list=[None, None, time.perf_counter() - started])
        if not boxes:
            return RapidOCROutput()
        ori_h, ori_w = original.shape[:2]
        mapped = map_boxes_to_original(np.asarray(boxes, dtype=np.float32), ratio_w, ratio_h, pad_top, ori_w, ori_h)
        # main: crops resized back with map_img_to_original, then lines with
        # empty text dropped before the word boxes are computed.
        valid = [i for i, t in enumerate(txts) if t.strip()]
        crop_sizes = [(round(chars[i][2] * ratio_h), round(chars[i][1] * ratio_w)) for i in valid]
        v_boxes = mapped[valid] if valid else np.zeros((0, 4, 2), dtype=np.float32)
        v_txts = tuple(txts[i] for i in valid)
        v_scores = tuple(float(scores[i]) for i in valid)
        v_infos = [infos[i] for i in valid]
        word_results = v_infos
        if all(word_results):
            lines = CalRecBoxes()(crop_sizes, v_boxes, v_txts, v_infos, bool(self.return_single_char_box))
            origin_words = []
            for word_line in lines:
                item = [(txt, score, bbox) for txt, score, bbox in word_line if bbox is not None]
                if item:
                    origin_words.append(tuple(item))
            word_results = tuple(origin_words)
        # main's filter_by_text_score (word_results indexed by line).
        f_boxes, f_txts, f_scores, f_words = [], [], [], []
        for i, (box, txt, score) in enumerate(zip(v_boxes, v_txts, v_scores)):
            if score < self.text_score:
                continue
            if i < len(word_results) and word_results[i]:
                f_words.append(word_results[i])
            f_boxes.append(box)
            f_txts.append(txt)
            f_scores.append(score)
        elapsed = time.perf_counter() - started
        if not f_txts:
            return RapidOCROutput()
        return RapidOCROutput(
            img=original,
            boxes=np.asarray(f_boxes, dtype=np.float32),
            txts=tuple(f_txts),
            scores=tuple(f_scores),
            word_results=tuple(f_words),
            elapse_list=[None, None, elapsed],
        )

    def _call_lite(self, engine, original, prepared, ratio_w, ratio_h, pad_top, started):
        if not self.use_det and not self.use_rec:
            raise RapidOCRError("the lite model has no orientation classifier; use_det or use_rec must be True")
        det_threshold = float(self.cfg.Det.get("thresh", 0.3))
        box_threshold = float(self.cfg.Det.get("box_thresh", 0.5))
        unclip = float(self.cfg.Det.get("unclip_ratio", 1.6))
        engine.set_det_thresholds(det_threshold, box_threshold, unclip)
        if not self.use_rec:
            boxes, _, scores = engine.run(prepared, _native.RPOCR_TASK_DET)
            elapsed = time.perf_counter() - started
            if not boxes:
                return TextDetOutput()
            ori_h, ori_w = original.shape[:2]
            mapped = map_boxes_to_original(np.asarray(boxes, dtype=np.float32), ratio_w, ratio_h, pad_top, ori_w, ori_h)
            return TextDetOutput(img=original, boxes=mapped, scores=list(scores), elapse=elapsed)
        want_words = bool(self.return_word_box) and engine.has_chars()
        chars = [] if want_words else None
        if not self.use_det:
            if want_words:
                _, txts, scores, _ = engine.run_ex(prepared, _native.RPOCR_TASK_REC, chars)
            else:
                _, txts, scores = engine.run(prepared, _native.RPOCR_TASK_REC)
            elapsed = time.perf_counter() - started
            word_results = tuple((t, s, None) for t, s in zip(txts, scores))
            if want_words:
                infos = self._word_infos(txts, chars)
                if infos is not None:
                    word_results = tuple(infos)
            return TextRecOutput(
                imgs=[original],
                txts=tuple(txts),
                scores=tuple(scores),
                word_results=word_results,
                elapse=elapsed,
            )
        if want_words:
            boxes, txts, scores, _ = engine.run_ex(
                prepared, _native.RPOCR_TASK_DET | _native.RPOCR_TASK_REC, chars
            )
            return self._word_box_output(
                original, boxes, txts, scores, chars, ratio_w, ratio_h, pad_top, started
            )
        boxes, txts, scores = engine.recognize(prepared)
        kept_boxes, kept_txts, kept_scores = filter_by_score(boxes, txts, scores, self.text_score)
        elapsed = time.perf_counter() - started
        if len(kept_txts) == 0:
            return RapidOCROutput()
        ori_h, ori_w = original.shape[:2]
        kept_boxes = map_boxes_to_original(kept_boxes, ratio_w, ratio_h, pad_top, ori_w, ori_h)
        return RapidOCROutput(
            img=original,
            boxes=kept_boxes,
            txts=kept_txts,
            scores=kept_scores,
            word_results=(),
            elapse_list=[None, None, elapsed],
        )

    def _cls_settings(self):
        """main's TextClassifier geometry for Cls.ocr_version (CLS_SHAPE_BY_OCR_VERSION)."""

        version = _enum_value(self.cfg.Cls.get("ocr_version", "PP-OCRv4"))
        if version == OCRVersion.PPOCRV4.value:
            return 48, 192, True
        if version == OCRVersion.PPOCRV5.value:
            return 80, 160, False
        raise NotImplementedError(f"Cls.ocr_version={version!r}: use PP-OCRv4 or PP-OCRv5")

    def _check_engine_choice(self) -> None:
        for section in ("Det", "Rec", "Cls"):
            choice = _enum_value(self.cfg[section].get("engine_type", "onnxruntime"))
            if choice not in _NATIVE_ENGINES:
                raise NotImplementedError(
                    f"{section}.engine_type={choice!r} is not available in the native "
                    "PP-OCRv6 build. Use EngineType.ONNXRUNTIME or EngineType.PPOCR_CPP."
                )
            version = _enum_value(self.cfg[section].get("ocr_version", "PP-OCRv6"))
            if section != "Cls" and version not in (OCRVersion.PPOCRV6.value, "PP-OCRv6", None):
                raise NotImplementedError(
                    f"The native engine runs PP-OCRv6 ONNX models, not {version}."
                )

    def _is_lite(self) -> bool:
        # Lite needs both sections (unchanged rule); a half-lite config keeps its previous behavior.
        return all(str(_enum_value(self.cfg[s].get("model_type", ""))).lower() == "lite" for s in ("Det", "Rec"))

    def _ensure_engine(self, need_cls: bool = False) -> _native.NativeOCR:
        if self._engine is not None and (not need_cls or self._engine_has_cls or self._is_lite()):
            return self._engine
        if self._engine is not None:
            # use_cls was turned on after construction without a classifier:
            # main loads its model lazily; reload the handle with it.
            self._engine.close()
            self._engine = None
        if self._is_lite():
            # Det/Rec model_type="lite": bundled Indra OCR weights (models/lite), run by the
            # native C++ engine with rpocr_options.model_type = RPOCR_MODEL_LITE.
            from ._indra_lite import IndraLiteEngine, bundled_model_dir
            th = int(self.cfg.EngineConfig.onnxruntime.get("intra_op_num_threads", -1) or -1)
            if th <= 0:
                import os
                th = int(os.environ.get("PPOCR_BENCH_THREADS", 0)) or min(4, os.cpu_count() or 1)
            self._engine = IndraLiteEngine(bundled_model_dir(self.cfg), threads=th)
            return self._engine
        det, rec, dictionary = _resolve_models(self.cfg)
        want_cls = bool(need_cls or self.use_cls)
        cls_model = _resolve_cls_model(self.cfg) if want_cls else None
        options = _native.default_options()
        options.backend = _native.RPOCR_BACKEND_CPU
        options.model_type = _native.RPOCR_MODEL_PPOCR
        _apply_vulkan_options(self.cfg, options)
        options.det_threshold = float(self.cfg["Det"].get("thresh", options.det_threshold))
        options.det_box_threshold = float(self.cfg["Det"].get("box_thresh", options.det_box_threshold))
        options.det_unclip_ratio = float(self.cfg["Det"].get("unclip_ratio", options.det_unclip_ratio))
        limit = self.cfg["Det"].get("limit_side_len")
        if limit:
            options.det_limit_side_len = int(limit)
        limit_type = str(self.cfg["Det"].get("limit_type", "max")).lower()
        options.det_limit_type = 1 if limit_type == "min" else 0
        _copy_norm(options.det_mean, options.det_std, self.cfg["Det"].get("mean"), self.cfg["Det"].get("std"))
        _copy_norm(
            options.rec_mean,
            options.rec_std,
            self.cfg["Rec"].get("mean"),
            self.cfg["Rec"].get("std"),
            default=(0.5, 0.5, 0.5),
        )
        options.det_use_dilation = 1 if bool(self.cfg["Det"].get("use_dilation", False)) else 0
        candidates = self.cfg["Det"].get("max_candidates")
        if candidates:
            options.det_max_candidates = int(candidates)
        shape = self.cfg["Rec"].get("rec_img_shape")
        if shape and len(shape) >= 2:
            options.rec_height = int(shape[1])
        batch = self.cfg["Rec"].get("rec_batch_num")
        if batch:
            options.rec_batch_size = int(batch)
        # Lite extension: Rec.fast_mode (False = exact rapidocr 3.10 output).
        options.rec_fast_mode = 1 if _as_bool(self.cfg["Rec"].get("fast_mode", False)) else 0
        # Orientation classifier: main's Cls section (use_cls itself is a
        # per-call task bit, so the handle default does not matter here).
        options.use_cls = 1 if want_cls else 0
        if cls_model:
            height, width, crop = self._cls_settings()
            options.cls_height = height
            options.cls_width = width
            options.cls_crop_long_lines = 1 if crop else 0
            options.cls_thresh = float(self.cfg.Cls.get("cls_thresh", 0.9))
            options.cls_batch_size = int(self.cfg.Cls.get("cls_batch_num", 6) or 6)
            self._cls_model_bytes = str(cls_model).encode()
            options.cls_model = self._cls_model_bytes
        self._engine = _native.NativeOCR(det, rec, dictionary, options)
        self._engine_has_cls = bool(cls_model) and self._engine.cls_loaded()
        _log_active_backend(self._engine, int(options.backend))
        return self._engine


_BACKEND_NAMES = {
    _native.RPOCR_BACKEND_CPU: "cpu",
    _native.RPOCR_BACKEND_GPU: "vulkan",
    _native.RPOCR_BACKEND_HYBRID: "hybrid",
    _native.RPOCR_BACKEND_METAL: "metal",
}


def _log_active_backend(engine: _native.NativeOCR, requested: int) -> None:
    """Log the backend the engine really runs, never just the requested one."""
    active = engine.active_backend()
    if active is None:
        return
    name = _BACKEND_NAMES.get(active, str(active))
    if requested == _native.RPOCR_BACKEND_METAL and active != _native.RPOCR_BACKEND_METAL:
        note = engine.backend_note()
        logger.warning(
            "Metal was requested but the engine runs on the %s backend%s",
            name,
            f": {note}" if note else "",
        )
    else:
        logger.info("Native engine backend: %s", name)


def _env_bool(name: str, default: bool) -> bool:
    """PPOCR_USE_CLS-style override: unset or empty keeps ``default``."""

    import os

    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _apply_vulkan_options(cfg: ConfigNode, options: _native.Options) -> None:
    """Honor EngineConfig.ppocr_cpp the way 3.x honors use_cuda.

    The default stays on CPU. ``use_vulkan`` or ``backend: vulkan`` uses the
    GPU graph when a compute device is already available, and otherwise logs
    a warning and keeps CPU. ``backend: hybrid`` is the C++ best-effort path.
    ``gpu_only`` in C++ still fails closed; this wrapper does not.
    ``use_metal`` or ``backend: metal`` runs the GPU graph on Metal (macOS,
    Apple Silicon) and keeps CPU when Metal is not available.
    """

    engine = cfg.get("EngineConfig")
    section = engine.get("ppocr_cpp") if isinstance(engine, ConfigNode) else None
    if not isinstance(section, ConfigNode):
        return
    use_vulkan = _as_bool(section.get("use_vulkan", False))
    use_metal = _as_bool(section.get("use_metal", False))
    backend_name = str(section.get("backend", "cpu")).strip().lower()
    if use_metal or backend_name == "metal":
        # Probe without initializing the GPU runtime: a Vulkan probe here
        # would fix the process on Vulkan before the engine asks for Metal.
        if _native.metal_available():
            # The engine reports the backend it really uses after creation.
            options.backend = _native.RPOCR_BACKEND_METAL
        else:
            logger.warning("Metal was requested but is not available; using the CPU backend")
            options.backend = _native.RPOCR_BACKEND_CPU
        return
    try:
        device_index = int(section.get("device_index", -1))
    except (TypeError, ValueError):
        device_index = -1
    options.vulkan_device_index = device_index
    wants_vulkan = use_vulkan or backend_name in {"vulkan", "gpu", "gpu_only"}
    wants_hybrid = backend_name == "hybrid"
    if not wants_vulkan and not wants_hybrid and device_index < 0:
        return
    if device_index >= 0:
        _native.request_vulkan_device(device_index)
    info = _native.backend_info()
    if wants_hybrid:
        options.backend = _native.RPOCR_BACKEND_HYBRID
        if not info["vulkan_compute_available"]:
            logger.warning("Vulkan hybrid was requested but no compute device is available; using CPU")
        return
    if wants_vulkan:
        if info["vulkan_compute_available"]:
            options.backend = _native.RPOCR_BACKEND_GPU
            logger.info("Vulkan device: %s", info["device_name"] or "(unnamed)")
        else:
            logger.warning(
                "Vulkan was requested but no compute device is available; using the CPU backend"
            )
            options.backend = _native.RPOCR_BACKEND_CPU


def _copy_norm(mean_out, std_out, mean, std, default=None) -> None:
    """Copy a length-3 mean/std into the C options. Missing values stay at the ABI default."""

    def take(slot, value):
        if value is None:
            if default is None:
                return
            value = default
        values = list(value)
        if len(values) != 3:
            return
        for index, item in enumerate(values):
            slot[index] = float(item)

    take(mean_out, mean)
    take(std_out, std)


def _load_config(config_path: Optional[str], params: Optional[Dict[str, Any]]) -> ConfigNode:
    cfg = ParseParams.load(config_path)
    if params:
        ParseParams.update_batch(cfg, params)
    if cfg.Global.get("model_root_dir") is None:
        cfg.Global.model_root_dir = _PACKAGE_DIR / "models"
    return cfg


def _section_info(section: ConfigNode) -> FileInfo:
    return FileInfo(
        section.engine_type,
        section.ocr_version,
        section.task_type,
        section.lang_type,
        section.model_type,
    )


def _prepare_image(engine: RapidOCR, original: np.ndarray, pad: bool = True):
    image = original
    ratio_w = ratio_h = 1.0
    pad_top = 0
    if bool(engine.cfg.Global.get("use_preprocess_img", True)):
        image, ratio_h, ratio_w = resize_image_within_bounds(
            image, float(engine.min_side_len), float(engine.max_side_len)
        )
    if pad and bool(engine.cfg.Global.get("use_vertical_padding", True)):
        image, pad_top = apply_vertical_padding(
            image, float(engine.width_height_ratio), float(engine.min_height)
        )
    return image, ratio_w, ratio_h, pad_top


def _resolve_models(cfg: ConfigNode) -> tuple:
    """Download missing det/rec/dict the way 3.x InferSession does.

    Files land in ``Global.model_root_dir`` under the URL filename. The cls
    model is resolved separately (``_resolve_cls_model``) and only when
    ``use_cls`` is on.
    """

    from .utils.download_file import DownloadFile, DownloadFileInput
    from .utils.download_models import download_task
    from .utils.log import logger as download_logger

    root = Path(cfg.Global.model_root_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    det = cfg.Det.get("model_path")
    rec = cfg.Rec.get("model_path")
    dictionary = cfg.Rec.get("rec_keys_path")

    if not det:
        info = _section_info(cfg.Det)
        download_task(root, info)
        det = root / Path(InferSession.get_model_url(info)["model_dir"]).name

    if not rec or not dictionary:
        info = _section_info(cfg.Rec)
        url_info = InferSession.get_model_url(info)
        if not rec:
            download_task(root, info)
            rec = root / Path(url_info["model_dir"]).name
        if not dictionary:
            dict_url = url_info.get("dict_url")
            if not dict_url:
                raise ValueError("recognizer model has no dict_url in default_models.yaml")
            dictionary = root / Path(dict_url).name
            if not dictionary.is_file():
                DownloadFile.run(
                    DownloadFileInput(
                        file_url=dict_url,
                        save_path=dictionary,
                        logger=download_logger,
                        sha256=None,
                    )
                )

    return str(det), str(rec), str(dictionary)


def _resolve_cls_model(cfg: ConfigNode) -> str:
    """Cls.model_path, else download main's Cls model (PP-OCRv4 mobile =
    ch_ppocr_mobile_v2.0_cls_mobile.onnx; PP-OCRv5 mobile/server = PP-LCNet
    textline_ori) into Global.model_root_dir, as det/rec are."""

    from .utils.download_models import download_task

    import os

    path = cfg.Cls.get("model_path") or os.environ.get("PPOCR_CLS_MODEL")
    if path:
        return str(path)
    root = Path(cfg.Global.model_root_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    info = _section_info(cfg.Cls)
    download_task(root, info)
    return str(root / Path(InferSession.get_model_url(info)["model_dir"]).name)


def parse_args(arg_list: Optional[list] = None):
    import argparse

    parser = argparse.ArgumentParser(description="RapidOCR native PP-OCRv6 demo")
    parser.add_argument("image", nargs="?", default=None)
    parser.add_argument("-img", "--img_path", dest="img_path", default=None)
    parser.add_argument("--det", default=None)
    parser.add_argument("--rec", default=None)
    parser.add_argument("--dict", dest="dictionary", default=None)
    parser.add_argument("--cls", dest="cls_model", default=None, help="orientation classifier ONNX")
    parser.add_argument("--text_score", type=float, default=None)
    parser.add_argument(
        "--use_cls",
        "--use-cls",
        dest="use_cls",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="text-line orientation classifier (default: Global.use_cls, on)",
    )
    parser.add_argument("-word", "--return_word_box", dest="return_word_box", action="store_true",
                        default=False, help="print word / character boxes (main's return_word_box)")
    parser.add_argument("--return_single_char_box", dest="return_single_char_box", action="store_true",
                        default=False, help="with -word: one box per character for English lines too")
    parser.add_argument("--lang_type", default=None)
    parser.add_argument(
        "--model-type",
        "--model_type",
        dest="model_type",
        default=None,
        choices=[x.value for x in ModelType],
    )
    parser.add_argument("--list-langs", action="store_true", dest="list_langs")
    parser.add_argument("--task", choices=[x.value for x in TaskType], default="rec")
    parser.add_argument(
        "--ocr_version", choices=[x.value for x in OCRVersion], default="PP-OCRv6"
    )
    return parser.parse_args(arg_list)


def main(arg_list: Optional[list] = None) -> None:
    args = parse_args(arg_list)

    if args.list_langs:
        print("\n".join(RapidOCR.list_supported_langs(args.task, args.ocr_version, args.model_type)))
        return

    image = args.img_path or args.image
    if image is None:
        raise SystemExit("rapidocr: an image path is required (positional or -img/--img_path)")
    # No --model-type: keep config.yaml defaults (small), same as RapidOCR().
    params: Dict[str, Any] = {}
    if args.use_cls is not None:
        params["Global.use_cls"] = bool(args.use_cls)
    if args.model_type:
        if args.model_type not in {"tiny", "small", "medium"}:
            raise SystemExit(f"rapidocr: PP-OCRv6 has no {args.model_type!r} model")
        params["Det.model_type"] = args.model_type
        params["Rec.model_type"] = args.model_type
    if args.text_score is not None:
        params["Global.text_score"] = args.text_score
    if args.lang_type:
        params["Det.lang_type"] = args.lang_type
        params["Rec.lang_type"] = args.lang_type
    if args.det:
        params["Det.model_path"] = args.det
    if args.rec:
        params["Rec.model_path"] = args.rec
    if args.dictionary:
        params["Rec.rec_keys_path"] = args.dictionary
    if args.cls_model:
        params["Cls.model_path"] = args.cls_model
    engine = RapidOCR(params=params)
    if args.return_word_box:
        result = engine(image, return_word_box=True,
                        return_single_char_box=bool(args.return_single_char_box))
    else:
        result = engine(image)
    print(result)
    if result.txts:
        for text, score in zip(result.txts, result.scores or ()):
            print(f"{text}\t{score:.4f}")
    if args.return_word_box:
        for line in result.word_results or ():
            if not line or not isinstance(line[0], tuple):
                continue
            for word, score, box in line:
                print(f"{word}\t{score:.4f}\t{box}")


if __name__ == "__main__":
    main()
