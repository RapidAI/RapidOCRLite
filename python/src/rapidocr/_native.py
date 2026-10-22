"""ctypes binding over the stable C ABI in ``include/rapidocrlite/rapidocrlite.h``."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import Optional

import numpy as np

RPOCR_OK = 0
RPOCR_BACKEND_CPU = 0
RPOCR_BACKEND_GPU = 1
RPOCR_BACKEND_HYBRID = 2
RPOCR_BACKEND_METAL = 3


RPOCR_TASK_DET = 1
RPOCR_TASK_REC = 2
RPOCR_TASK_CLS = 4


class Options(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_int),
        ("backend", ctypes.c_int),
        ("det_threshold", ctypes.c_float),
        ("det_box_threshold", ctypes.c_float),
        ("det_unclip_ratio", ctypes.c_float),
        ("det_limit_side_len", ctypes.c_int),
        ("rec_height", ctypes.c_int),
        ("rec_max_width", ctypes.c_int),
        ("rec_batch_size", ctypes.c_int),
        ("rec_batch_width_bucket", ctypes.c_int),
        ("rec_parallelism", ctypes.c_int),
        ("hybrid_graph_parallelism", ctypes.c_int),
        ("det_batch_size", ctypes.c_int),
        ("batch_preprocess_parallelism", ctypes.c_int),
        ("image_batch_parallelism", ctypes.c_int),
        ("det_limit_type", ctypes.c_int),
        ("det_mean", ctypes.c_float * 3),
        ("det_std", ctypes.c_float * 3),
        ("rec_mean", ctypes.c_float * 3),
        ("rec_std", ctypes.c_float * 3),
        ("det_use_dilation", ctypes.c_int),
        ("det_max_candidates", ctypes.c_int),
        ("vulkan_device_index", ctypes.c_int),
        ("rec_fast_mode", ctypes.c_int),
        ("model_type", ctypes.c_int),  # RPOCR_MODEL_*
        # Text-line orientation classifier (appended; see rapidocrlite.h).
        ("use_cls", ctypes.c_int),
        ("cls_thresh", ctypes.c_float),
        ("cls_batch_size", ctypes.c_int),
        ("cls_height", ctypes.c_int),
        ("cls_width", ctypes.c_int),
        ("cls_crop_long_lines", ctypes.c_int),
        ("cls_model", ctypes.c_char_p),
    ]


RPOCR_MODEL_AUTO = 0
RPOCR_MODEL_PPOCR = 1
RPOCR_MODEL_LITE = 2


class Image(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("rgb", ctypes.POINTER(ctypes.c_uint8)),
    ]


class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class TextBox(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("confidence", ctypes.c_float),
        ("bbox_x", ctypes.c_int),
        ("bbox_y", ctypes.c_int),
        ("bbox_w", ctypes.c_int),
        ("bbox_h", ctypes.c_int),
        ("box", Point * 4),
    ]


class Result(ctypes.Structure):
    _fields_ = [("items", ctypes.POINTER(TextBox)), ("count", ctypes.c_size_t)]


class LineChars(ctypes.Structure):
    """rpocr_line_chars: per-character CTC data for word boxes."""

    _fields_ = [
        ("rec_steps", ctypes.c_int),
        ("crop_width", ctypes.c_int),
        ("crop_height", ctypes.c_int),
        ("rec_max_wh_ratio", ctypes.c_double),
        ("count", ctypes.c_size_t),
        ("cols", ctypes.POINTER(ctypes.c_int)),
        ("probs", ctypes.POINTER(ctypes.c_float)),
    ]


class Word(ctypes.Structure):
    _fields_ = [("text", ctypes.c_char_p), ("confidence", ctypes.c_double), ("box", Point * 4)]


class Words(ctypes.Structure):
    _fields_ = [("items", ctypes.POINTER(Word)), ("count", ctypes.c_size_t)]


REC_ORDER_FN = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p
)


class BackendInfo(ctypes.Structure):
    _fields_ = [
        ("vulkan_loader_available", ctypes.c_int),
        ("vulkan_compute_available", ctypes.c_int),
        ("full_graph_gpu_available", ctypes.c_int),
        ("device_name", ctypes.c_char * 256),
        ("vulkan_runtime_generation", ctypes.c_uint64),
    ]


class CpuInfo(ctypes.Structure):
    _fields_ = [
        ("avx2_compiled", ctypes.c_int),
        ("avx512_compiled", ctypes.c_int),
        ("neon_compiled", ctypes.c_int),
        ("avx2", ctypes.c_int),
        ("avx512", ctypes.c_int),
        ("neon", ctypes.c_int),
        ("threads", ctypes.c_int),
        ("active_isa", ctypes.c_char * 16),
    ]


def _candidate_libraries():
    env = os.environ.get("PPOCR_LIBRARY")
    if env:
        yield Path(env)
    bundled = Path(__file__).resolve().parent
    names = ("librapidocrlite.so", "librapidocrlite.dylib", "rapidocrlite.dll")
    for name in names:
        yield bundled / name
    yield from bundled.glob("librapidocrlite.so.*")
    yield from bundled.glob("librapidocrlite.*.dylib")
    found = ctypes.util.find_library("rapidocrlite")
    if found:
        yield Path(found)


def load_library():
    errors = []
    for path in _candidate_libraries():
        if not path or not Path(path).exists():
            continue
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue
        _bind(lib)
        return lib
    hint = "; ".join(errors) if errors else "library not found"
    raise ImportError(
        "Could not load librapidocrlite. Build it with CMake, set PPOCR_LIBRARY, "
        f"or install the wheel. ({hint})"
    )


def _bind(lib: ctypes.CDLL) -> None:
    lib.rpocr_options_init.argtypes = [ctypes.POINTER(Options)]
    lib.rpocr_options_init.restype = None
    lib.rpocr_options_init_ex.argtypes = [ctypes.POINTER(Options), ctypes.c_size_t]
    lib.rpocr_options_init_ex.restype = None
    lib.rpocr_create.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(Options),
    ]
    lib.rpocr_create.restype = ctypes.c_void_p
    lib.rpocr_destroy.argtypes = [ctypes.c_void_p]
    lib.rpocr_destroy.restype = None
    lib.rpocr_set_det_thresholds.argtypes = [
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.rpocr_set_det_thresholds.restype = ctypes.c_int
    lib.rpocr_recognize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Image),
        ctypes.POINTER(Result),
    ]
    lib.rpocr_recognize.restype = ctypes.c_int
    run = getattr(lib, "rpocr_run", None)
    if run is not None:
        run.argtypes = [ctypes.c_void_p, ctypes.POINTER(Image), ctypes.c_int, ctypes.POINTER(Result)]
        run.restype = ctypes.c_int
    run_ex = getattr(lib, "rpocr_run_ex", None)
    if run_ex is not None:
        run_ex.argtypes = [ctypes.c_void_p, ctypes.POINTER(Image), ctypes.c_int, ctypes.POINTER(Result)]
        run_ex.restype = ctypes.c_int
    result_cls = getattr(lib, "rpocr_result_cls", None)
    if result_cls is not None:
        result_cls.argtypes = [ctypes.POINTER(Result), ctypes.c_size_t,
                               ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)]
        result_cls.restype = ctypes.c_int
    result_chars = getattr(lib, "rpocr_result_chars", None)
    if result_chars is not None:
        result_chars.argtypes = [ctypes.POINTER(Result), ctypes.c_size_t, ctypes.POINTER(LineChars)]
        result_chars.restype = ctypes.c_int
    result_words = getattr(lib, "rpocr_result_words", None)
    if result_words is not None:
        result_words.argtypes = [ctypes.POINTER(Result), ctypes.c_size_t, ctypes.c_int, ctypes.POINTER(Words)]
        result_words.restype = ctypes.c_int
        lib.rpocr_words_free.argtypes = [ctypes.POINTER(Words)]
        lib.rpocr_words_free.restype = None
    set_rec_order = getattr(lib, "rpocr_set_rec_order", None)
    if set_rec_order is not None:
        set_rec_order.argtypes = [ctypes.c_void_p, REC_ORDER_FN, ctypes.c_void_p]
        set_rec_order.restype = ctypes.c_int
    argsort = getattr(lib, "rpocr_argsort", None)
    if argsort is not None:
        argsort.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        argsort.restype = None
    argsort_isa = getattr(lib, "rpocr_argsort_isa", None)
    if argsort_isa is not None:
        argsort_isa.argtypes = []
        argsort_isa.restype = ctypes.c_int
    cls_loaded = getattr(lib, "rpocr_cls_loaded", None)
    if cls_loaded is not None:
        cls_loaded.argtypes = [ctypes.c_void_p]
        cls_loaded.restype = ctypes.c_int
    lib.rpocr_result_free.argtypes = [ctypes.POINTER(Result)]
    lib.rpocr_result_free.restype = None
    lib.rpocr_last_error.argtypes = []
    lib.rpocr_last_error.restype = ctypes.c_char_p
    lib.rpocr_query_cpu_info.argtypes = [ctypes.POINTER(CpuInfo)]
    lib.rpocr_query_cpu_info.restype = None
    query_backend = getattr(lib, "rpocr_query_backend_info", None)
    if query_backend is not None:
        query_backend.argtypes = [ctypes.POINTER(BackendInfo)]
        query_backend.restype = None
    active = getattr(lib, "rpocr_active_backend", None)
    if active is not None:
        active.argtypes = [ctypes.c_void_p]
        active.restype = ctypes.c_int
    note = getattr(lib, "rpocr_backend_note", None)
    if note is not None:
        note.argtypes = [ctypes.c_void_p]
        note.restype = ctypes.c_char_p
    request_device = getattr(lib, "rpocr_request_vulkan_device", None)
    if request_device is not None:
        request_device.argtypes = [ctypes.c_int]
        request_device.restype = None
    resize = getattr(lib, "rpocr_resize_linear_rgb", None)
    if resize is not None:
        resize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                           ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        resize.restype = ctypes.c_int


class NativeOCR:
    def __init__(self, det: str, rec: str, dictionary: str, options: Options):
        self.lib = load_library()
        handle = self.lib.rpocr_create(
            det.encode(), rec.encode(), dictionary.encode(), ctypes.byref(options)
        )
        if not handle:
            raise RuntimeError(self._error() or "rpocr_create failed")
        self.handle = handle

    def active_backend(self) -> Optional[int]:
        """RPOCR_BACKEND_* that runs the full pipeline, or None on an old build."""
        query = getattr(self.lib, "rpocr_active_backend", None)
        return None if query is None else int(query(self.handle))

    def backend_note(self) -> str:
        """Why a requested Metal backend fell back to CPU ("" otherwise)."""
        query = getattr(self.lib, "rpocr_backend_note", None)
        raw = query(self.handle) if query is not None else None
        return raw.decode("utf-8", errors="replace") if raw else ""

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.rpocr_destroy(self.handle)
            self.handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def set_det_thresholds(self, det_threshold: float, box_threshold: float, unclip_ratio: float) -> None:
        status = self.lib.rpocr_set_det_thresholds(
            self.handle, float(det_threshold), float(box_threshold), float(unclip_ratio)
        )
        if status != RPOCR_OK:
            raise RuntimeError(self._error() or "rpocr_set_det_thresholds failed")

    def recognize(self, rgb: np.ndarray):
        return self.run(rgb, RPOCR_TASK_DET | RPOCR_TASK_REC)

    def cls_loaded(self) -> bool:
        query = getattr(self.lib, "rpocr_cls_loaded", None)
        return bool(query(self.handle)) if query is not None else False

    def has_chars(self) -> bool:
        return getattr(self.lib, "rpocr_result_chars", None) is not None

    def run_ex(self, rgb: np.ndarray, tasks: int, chars: Optional[list] = None,
               c_words: Optional[list] = None):
        """rpocr_run_ex: use_cls is exactly ``tasks & RPOCR_TASK_CLS``.

        Returns (boxes, txts, scores, cls) where cls holds one
        (label_index, score) per line; label_index is -1 when the line was
        not classified. When ``chars`` is a list it receives one
        (rec_steps, crop_width, crop_height, rec_max_wh_ratio, cols, probs)
        tuple per line from rpocr_result_chars (rec_steps 0: not recorded).
        ``c_words`` (a list) receives, per line, the engine's own
        rpocr_result_words as a (words, single_chars) pair of
        [(text, conf, [[x, y] * 4])] lists (parity checks).
        """
        if getattr(self.lib, "rpocr_run_ex", None) is None:
            raise RuntimeError("this librapidocrlite has no rpocr_run_ex; use_cls needs a newer build")
        array = np.ascontiguousarray(rgb, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("native OCR expects an HWC RGB uint8 image")
        height, width = array.shape[:2]
        image = Image(width, height, array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        result = Result()
        status = self.lib.rpocr_run_ex(self.handle, ctypes.byref(image), int(tasks), ctypes.byref(result))
        try:
            if status != RPOCR_OK:
                raise RuntimeError(self._error() or "rpocr_run_ex failed")
            boxes, txts, scores, cls = [], [], [], []
            label = ctypes.c_int(-1)
            score = ctypes.c_float(0.0)
            for index in range(int(result.count)):
                item = result.items[index]
                txts.append(item.text.decode("utf-8", errors="replace") if item.text else "")
                scores.append(float(item.confidence))
                boxes.append([[item.box[k].x, item.box[k].y] for k in range(4)])
                self.lib.rpocr_result_cls(ctypes.byref(result), index, ctypes.byref(label), ctypes.byref(score))
                cls.append((int(label.value), float(score.value)))
                if chars is not None:
                    info = LineChars()
                    if self.has_chars() and self.lib.rpocr_result_chars(
                        ctypes.byref(result), index, ctypes.byref(info)
                    ) == RPOCR_OK:
                        n = int(info.count)
                        chars.append((
                            int(info.rec_steps), int(info.crop_width), int(info.crop_height),
                            float(info.rec_max_wh_ratio),
                            [int(info.cols[k]) for k in range(n)],
                            [float(info.probs[k]) for k in range(n)],
                        ))
                    else:
                        chars.append((0, 0, 0, 0.0, [], []))
                if c_words is not None:
                    pair = []
                    for single in (0, 1):
                        words = Words()
                        got = []
                        if self.lib.rpocr_result_words(ctypes.byref(result), index, single,
                                                       ctypes.byref(words)) == RPOCR_OK:
                            for k in range(int(words.count)):
                                w = words.items[k]
                                got.append((w.text.decode("utf-8", errors="replace"), float(w.confidence),
                                            [[int(w.box[j].x), int(w.box[j].y)] for j in range(4)]))
                            self.lib.rpocr_words_free(ctypes.byref(words))
                        pair.append(got)
                    c_words.append(tuple(pair))
            return boxes, txts, scores, cls
        finally:
            self.lib.rpocr_result_free(ctypes.byref(result))

    def run(self, rgb: np.ndarray, tasks: int):
        array = np.ascontiguousarray(rgb, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("native OCR expects an HWC RGB uint8 image")
        height, width = array.shape[:2]
        image = Image(width, height, array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        result = Result()
        if tasks == RPOCR_TASK_DET | RPOCR_TASK_REC:
            status = self.lib.rpocr_recognize(self.handle, ctypes.byref(image), ctypes.byref(result))
        else:
            if getattr(self.lib, "rpocr_run", None) is None:
                raise RuntimeError("this librapidocrlite has no rpocr_run; use_det/use_rec need a newer build")
            status = self.lib.rpocr_run(self.handle, ctypes.byref(image), int(tasks), ctypes.byref(result))
        try:
            if status != RPOCR_OK:
                raise RuntimeError(self._error() or "rpocr_run failed")
            boxes = []
            txts = []
            scores = []
            for index in range(int(result.count)):
                item = result.items[index]
                txts.append(item.text.decode("utf-8", errors="replace") if item.text else "")
                scores.append(float(item.confidence))
                boxes.append([[item.box[k].x, item.box[k].y] for k in range(4)])
            return boxes, txts, scores
        finally:
            self.lib.rpocr_result_free(ctypes.byref(result))

    def _error(self) -> str:
        raw = self.lib.rpocr_last_error()
        return raw.decode() if raw else ""


def cpu_info() -> dict:
    lib = load_library()
    info = CpuInfo()
    lib.rpocr_query_cpu_info(ctypes.byref(info))
    return {
        "active_isa": info.active_isa.decode(),
        "avx2": bool(info.avx2),
        "avx512": bool(info.avx512),
        "neon": bool(info.neon),
        "threads": int(info.threads),
        "avx2_compiled": bool(info.avx2_compiled),
        "avx512_compiled": bool(info.avx512_compiled),
        "neon_compiled": bool(info.neon_compiled),
    }


def backend_info() -> dict:
    lib = load_library()
    query = getattr(lib, "rpocr_query_backend_info", None)
    if query is None:
        return {
            "vulkan_loader_available": False,
            "vulkan_compute_available": False,
            "full_graph_gpu_available": False,
            "device_name": "",
            "vulkan_runtime_generation": 0,
        }
    info = BackendInfo()
    query(ctypes.byref(info))
    return {
        "vulkan_loader_available": bool(info.vulkan_loader_available),
        "vulkan_compute_available": bool(info.vulkan_compute_available),
        "full_graph_gpu_available": bool(info.full_graph_gpu_available),
        "device_name": info.device_name.split(b"\x00", 1)[0].decode(errors="replace"),
        "vulkan_runtime_generation": int(info.vulkan_runtime_generation),
    }


def resize_linear_rgb(img: np.ndarray, width: int, height: int) -> Optional[np.ndarray]:
    """cv2.resize(img, (width, height)) for an HxWx3 uint8 image, bit-exact
    with OpenCV 5.0. None when the loaded library predates the export."""

    lib = load_library()
    resize = getattr(lib, "rpocr_resize_linear_rgb", None)
    if resize is None or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        return None
    src = np.ascontiguousarray(img)
    out = np.empty((height, width, 3), dtype=np.uint8)
    if resize(src.ctypes.data, src.shape[1], src.shape[0], out.ctypes.data, width, height) != 0:
        return None
    return out


def metal_available() -> bool:
    """True when this library has the Metal backend and a Metal device exists.

    It does not initialize the GPU runtime, so a later engine can still pick
    Metal for the process.
    """

    lib = load_library()
    probe = getattr(lib, "rpocr_metal_available", None)
    if probe is None:
        return False
    probe.argtypes = []
    probe.restype = ctypes.c_int
    return bool(probe())


def request_vulkan_device(index: int) -> None:
    """Pin a Vulkan physical device before the first engine or backend probe."""

    if index < 0:
        return
    lib = load_library()
    request = getattr(lib, "rpocr_request_vulkan_device", None)
    if request is None:
        return
    request(int(index))


def default_options() -> Options:
    lib = load_library()
    options = Options()
    lib.rpocr_options_init_ex(ctypes.byref(options), ctypes.sizeof(Options))
    return options


def native_argsort(values) -> np.ndarray:
    """rpocr_argsort: the engine's native np.argsort (recognition batch order)."""

    lib = load_library()
    if getattr(lib, "rpocr_argsort", None) is None:
        raise RuntimeError("this librapidocrlite has no rpocr_argsort")
    arr = np.ascontiguousarray(values, dtype=np.float64)
    out = np.empty(arr.shape[0], dtype=np.uintp)
    lib.rpocr_argsort(arr.ctypes.data, arr.shape[0], out.ctypes.data)
    return out.astype(np.intp)


def native_argsort_isa() -> str:
    lib = load_library()
    query = getattr(lib, "rpocr_argsort_isa", None)
    return {0: "scalar", 1: "avx2", 2: "avx512"}.get(int(query()), "?") if query else "none"
