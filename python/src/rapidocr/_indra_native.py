"""ctypes wrapper: drop-in replacement for an onnxruntime session of Indra rec/rec_cjk."""
import ctypes, numpy as np
class NativeRec:
    def __init__(self, lib_path, onnx_path, threads):
        self.lib = lib = ctypes.CDLL(lib_path)
        lib.rpocr_indra_rec_create.restype = ctypes.c_void_p
        lib.rpocr_indra_rec_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.rpocr_indra_rec_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.rpocr_indra_rec_steps.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.rpocr_indra_isa.argtypes = [ctypes.c_void_p]; lib.rpocr_indra_isa.restype = ctypes.c_char_p
        lib.rpocr_indra_last_error.restype = ctypes.c_char_p
        lib.rpocr_indra_rec_destroy.argtypes = [ctypes.c_void_p]
        self.h = lib.rpocr_indra_rec_create(onnx_path.encode(), threads)
        if not self.h: raise RuntimeError(lib.rpocr_indra_last_error().decode())
        self.isa = lib.rpocr_indra_isa(self.h).decode()
    def run(self, _names, feeds):
        x = np.ascontiguousarray(feeds['x'], np.float32); w = np.ascontiguousarray(feeds['widths'], np.int64)
        B, _, H, W = x.shape; T = self.lib.rpocr_indra_rec_steps(self.h, W)
        ids = np.zeros((B, T), np.int64); conf = np.zeros((B, T), np.float32); ln = np.zeros(B, np.int64)
        if self.lib.rpocr_indra_rec_run(self.h, x.ctypes.data, B, H, W, w.ctypes.data, ids.ctypes.data, conf.ctypes.data, ln.ctypes.data):
            raise RuntimeError(self.lib.rpocr_indra_last_error().decode())
        return [ids, conf, ln]
    def __del__(self):
        try: self.lib.rpocr_indra_rec_destroy(self.h)
        except Exception: pass


class NativeDet:
    """Drop-in for the det.onnx onnxruntime session: run(None, {"x": [1,1,H,W]}) -> [out[1,C,h,w]]."""
    def __init__(self, lib_path, onnx_path, threads):
        self.lib = lib = ctypes.CDLL(lib_path)
        lib.rpocr_indra_det_create.restype = ctypes.c_void_p
        lib.rpocr_indra_det_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.rpocr_indra_det_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.rpocr_indra_det_fetch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.rpocr_indra_det_destroy.argtypes = [ctypes.c_void_p]
        lib.rpocr_indra_last_error.restype = ctypes.c_char_p
        self.h = lib.rpocr_indra_det_create(onnx_path.encode(), threads)
        if not self.h: raise RuntimeError(lib.rpocr_indra_last_error().decode())
    def run(self, _names, feeds):
        x = np.ascontiguousarray(feeds['x'], np.float32)
        H, W = x.shape[-2:]
        dims = (ctypes.c_int * 3)()
        if self.lib.rpocr_indra_det_run(self.h, x.ctypes.data, H, W, dims):
            raise RuntimeError(self.lib.rpocr_indra_last_error().decode())
        out = np.empty((1, dims[0], dims[1], dims[2]), np.float32)
        self.lib.rpocr_indra_det_fetch(self.h, out.ctypes.data)
        return [out]
    def __del__(self):
        try: self.lib.rpocr_indra_det_destroy(self.h)
        except Exception: pass


class NativeScript:
    """Drop-in for the script.onnx session: run(None, {"x": [B,1,32,W], "widths": [B]}) -> [p[B]]."""
    def __init__(self, lib_path, onnx_path, threads):
        self.lib = lib = ctypes.CDLL(lib_path)
        lib.rpocr_indra_script_create.restype = ctypes.c_void_p
        lib.rpocr_indra_script_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.rpocr_indra_script_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                               ctypes.c_void_p, ctypes.c_void_p]
        lib.rpocr_indra_script_destroy.argtypes = [ctypes.c_void_p]
        lib.rpocr_indra_last_error.restype = ctypes.c_char_p
        self.h = lib.rpocr_indra_script_create(onnx_path.encode(), threads)
        if not self.h: raise RuntimeError(lib.rpocr_indra_last_error().decode())
    def run(self, _names, feeds):
        x = np.ascontiguousarray(feeds['x'], np.float32); w = np.ascontiguousarray(feeds['widths'], np.int64)
        B, _, H, W = x.shape
        p = np.zeros(B, np.float32)
        if self.lib.rpocr_indra_script_run(self.h, x.ctypes.data, B, H, W, w.ctypes.data, p.ctypes.data):
            raise RuntimeError(self.lib.rpocr_indra_last_error().decode())
        return [p]
    def __del__(self):
        try: self.lib.rpocr_indra_script_destroy(self.h)
        except Exception: pass


class NativeOrder:
    """Drop-in for the order.onnx session: run(None, {"feats": [1,N,F], "mask": [1,N]}) -> [a, cls, same, tab]."""
    need_tab = True  # set False when the caller ignores the table heads (skips three N x N products)

    def __init__(self, lib_path, onnx_path, threads):
        self.lib = lib = ctypes.CDLL(lib_path)
        lib.rpocr_indra_order_create.restype = ctypes.c_void_p
        lib.rpocr_indra_order_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.rpocr_indra_order_ncls.argtypes = [ctypes.c_void_p]
        lib.rpocr_indra_order_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 4
        lib.rpocr_indra_order_destroy.argtypes = [ctypes.c_void_p]
        lib.rpocr_indra_last_error.restype = ctypes.c_char_p
        self.h = lib.rpocr_indra_order_create(onnx_path.encode(), threads)
        if not self.h: raise RuntimeError(lib.rpocr_indra_last_error().decode())
        self.ncls = lib.rpocr_indra_order_ncls(self.h)
    def run(self, _names, feeds):
        f = np.ascontiguousarray(feeds['feats'][0], np.float32); N = f.shape[0]
        m = np.ascontiguousarray(feeds['mask'][0], np.uint8)
        a = np.empty((1, N, N), np.float32); s = np.empty((1, N, N), np.float32)
        c = np.empty((1, N, self.ncls), np.float32)
        t = np.empty((1, 3, N, N), np.float32) if self.need_tab else np.zeros((1, 3, 0, 0), np.float32)
        if self.lib.rpocr_indra_order_run(self.h, f.ctypes.data, m.ctypes.data, N, a.ctypes.data, s.ctypes.data, c.ctypes.data,
                                          t.ctypes.data if self.need_tab else None):
            raise RuntimeError(self.lib.rpocr_indra_last_error().decode())
        return [a, c, s, t]
    def __del__(self):
        try: self.lib.rpocr_indra_order_destroy(self.h)
        except Exception: pass
