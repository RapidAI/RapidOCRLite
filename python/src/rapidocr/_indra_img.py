"""OpenCV-free image ops for the lite pipeline (librapidocrlite ppocr_img_* C API, numpy fallbacks for the trivial ones).

The C functions reproduce the OpenCV 5.0 arithmetic of cv2.resize (INTER_AREA / INTER_LINEAR, uint8),
cv2.warpAffine (INTER_LINEAR) and cv2.connectedComponentsWithStats (4-connectivity) used by the Indra
pipeline, so results match the onnxruntime + OpenCV reference without cv2 installed.
"""
from __future__ import annotations

import ctypes

import numpy as np

INTER_LINEAR = 1
INTER_AREA = 3
BORDER_CONSTANT = 0
BORDER_REPLICATE = 1

_lib = None


def bind(lib_path: str):
    global _lib
    lib = ctypes.CDLL(lib_path)
    lib.rpocr_img_resize_u8.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.rpocr_img_warp_affine.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double,
                                          ctypes.c_int, ctypes.c_int]
    lib.rpocr_img_rgb2gray.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.rpocr_img_set_threads.argtypes = [ctypes.c_int]
    lib.rpocr_img_rotation_matrix.argtypes = [ctypes.c_double] * 4 + [ctypes.c_int, ctypes.c_void_p]
    lib.rpocr_img_cc4.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    _lib = lib
    return lib


def set_threads(n: int):
    _lib.rpocr_img_set_threads(int(n))


def resize(img: np.ndarray, size, interpolation=INTER_LINEAR, flags: int = -1) -> np.ndarray:
    """cv2.resize(img, (w, h), interpolation) for 2-D uint8 images."""
    src = np.ascontiguousarray(img, np.uint8)
    w, h = int(size[0]), int(size[1])
    out = np.empty((h, w), np.uint8)
    r = _lib.rpocr_img_resize_u8(src.ctypes.data, src.shape[1], src.shape[0], out.ctypes.data, w, h, interpolation, flags)
    if r:
        raise ValueError(f"rpocr_img_resize_u8 failed ({r})")
    return out


def rotation_matrix(center, angle: float, scale: float = 1.0, flags: int = -1) -> np.ndarray:
    """cv2.getRotationMatrix2D (same rounding as the OpenCV build of the platform)."""
    M = np.empty((2, 3), np.float64)
    _lib.rpocr_img_rotation_matrix(float(center[0]), float(center[1]), float(angle), float(scale), flags, M.ctypes.data)
    return M


def warp_affine(img: np.ndarray, M: np.ndarray, size, border=BORDER_CONSTANT, value: float = 0.0,
                uf: int = 0, flags: int = -1) -> np.ndarray:
    """cv2.warpAffine(img, M, size, flags=INTER_LINEAR, borderMode=border) for 2-D uint8 / float32 images."""
    is_f = img.dtype == np.float32
    src = np.ascontiguousarray(img, np.float32 if is_f else np.uint8)
    w, h = int(size[0]), int(size[1])
    out = np.empty((h, w), src.dtype)
    m = np.ascontiguousarray(M, np.float64)
    _lib.rpocr_img_warp_affine(src.ctypes.data, src.shape[1], src.shape[0], out.ctypes.data, w, h, m.ctypes.data,
                               int(is_f), border, float(value), uf, flags)
    return out


def connected_components_with_stats(mask: np.ndarray):
    """cv2.connectedComponentsWithStats(mask, connectivity=4)[:3]: (n, labels int32, stats [n,5])."""
    m = np.ascontiguousarray(mask, np.uint8)
    h, w = m.shape
    lab = np.empty((h, w), np.int32)
    cap = 1024
    while True:
        st = np.empty((cap, 5), np.int32)
        n = _lib.rpocr_img_cc4(m.ctypes.data, w, h, lab.ctypes.data, st.ctypes.data, cap)
        if n <= cap:
            return n, lab, st[:n]
        cap = n  # rerun with room for all stats


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """cv2.cvtColor(rgb, COLOR_RGB2GRAY) for uint8 (15-bit fixed point RY15/GY15/BY15, round half up)."""
    if rgb.ndim == 2:
        return np.ascontiguousarray(rgb, np.uint8)
    a = np.ascontiguousarray(rgb[..., :3]) if rgb.shape[2] != 3 else rgb
    if a.dtype != np.uint8 or a.strides[1:] != (3, 1):
        a = np.ascontiguousarray(a, np.uint8)
    h, w = a.shape[:2]
    out = np.empty((h, w), np.uint8)
    _lib.rpocr_img_rgb2gray(a.ctypes.data, w, h, a.strides[0], out.ctypes.data)
    return out


def pad_constant(img: np.ndarray, p: int, value: int = 255) -> np.ndarray:
    """cv2.copyMakeBorder(img, p, p, p, p, BORDER_CONSTANT, value)."""
    return np.pad(img, p, mode="constant", constant_values=value)
