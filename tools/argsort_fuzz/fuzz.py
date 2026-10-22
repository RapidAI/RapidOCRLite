#!/usr/bin/env python3
"""Fuzz the engine's native np.argsort (rpocr_argsort) against NumPy.

  fuzz.py [--cases N] [--seed S] [--variant native|avx2|scalar]

`native` compares with NumPy as imported; `avx2` / `scalar` re-run this
script in a subprocess with NPY_DISABLE_CPU_FEATURES hiding NumPy's X86_V4
(and X86_V3) argsort targets and PPOCR_ARGSORT_ISA forcing the same native
variant, so all three code paths are checked on an AVX-512 runner. Arrays:
lengths 1..2000, quantized width/height ratios with heavy ties, plus
constant, sorted, reversed, +-0.0, inf and NaN cases. Exit 1 on any
mismatch.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cases", type=int, default=20000)
ap.add_argument("--seed", type=int, default=20261022)
ap.add_argument("--variant", default="native", choices=["native", "avx2", "scalar", "all"])
ap.add_argument("--child", action="store_true")
a = ap.parse_args()

if a.variant == "all" or (a.variant in ("avx2", "scalar") and not a.child):
    variants = ["native", "avx2", "scalar"] if a.variant == "all" else [a.variant]
    rc = 0
    for v in variants:
        env = dict(os.environ)
        if v == "avx2":
            env["NPY_DISABLE_CPU_FEATURES"] = "X86_V4 AVX512_ICL AVX512_SPR"
            env["PPOCR_ARGSORT_ISA"] = "avx2"
        elif v == "scalar":
            env["NPY_DISABLE_CPU_FEATURES"] = "X86_V3 X86_V4 AVX512_ICL AVX512_SPR"
            env["PPOCR_ARGSORT_ISA"] = "scalar"
        cmd = [sys.executable, __file__, "--cases", str(a.cases), "--seed", str(a.seed),
               "--variant", v, "--child"]
        rc |= subprocess.call(cmd, env=env)
    sys.exit(rc)

from numpy._core import _multiarray_umath as mu  # noqa: E402

from rapidocr import _native  # noqa: E402

feats = mu.__cpu_features__
numpy_path = "x86-simd-sort AVX-512" if feats.get("X86_V4") else (
    "x86-simd-sort AVX2" if feats.get("X86_V3") else "scalar introsort")
rng = np.random.default_rng(a.seed)


def make(i):
    n = int(rng.integers(1, 2001))
    kind = i % 10
    if kind == 0:  # crop ratios w/h with small integer sizes: many exact ties
        w = rng.integers(1, 60, n)
        h = rng.integers(1, 12, n)
        return w / h.astype(np.float64)
    if kind == 1:  # few distinct values
        return rng.choice(rng.random(int(rng.integers(1, 6))), n)
    if kind == 2:  # quantized
        return np.round(rng.random(n) * int(rng.integers(2, 50))) / 7.0
    if kind == 3:
        return np.full(n, 6.666666666666667)
    if kind == 4:
        return np.sort(rng.integers(0, 20, n)).astype(np.float64)
    if kind == 5:
        return np.sort(rng.integers(0, 20, n)).astype(np.float64)[::-1].copy()
    if kind == 6:
        x = rng.choice(np.array([0.0, -0.0, 1.0, np.inf, -np.inf]), n)
        return x
    if kind == 7:  # NaN placement
        x = np.round(rng.random(n) * 9)
        x[rng.random(n) < 0.1] = np.nan
        return x
    if kind == 8:  # realistic page: line ratios from crop sizes
        h = rng.integers(14, 60, n)
        w = (h * rng.choice(np.array([1, 2, 3, 4, 6.5, 8, 10, 20]), n) + rng.integers(0, 3, n)).astype(np.int64)
        return w / h.astype(np.float64)
    return rng.random(n)


bad = 0
lengths = 0
for i in range(a.cases):
    x = make(i)
    lengths += len(x)
    ref = np.argsort(x)
    got = _native.native_argsort(x)
    if not np.array_equal(ref, got):
        bad += 1
        if bad <= 5:
            first = int(np.argmax(ref != got))
            print(f"MISMATCH case {i} n={len(x)} kind={i % 10} first diff at {first}: "
                  f"numpy {ref[first:first + 6].tolist()} native {got[first:first + 6].tolist()}")
print(f"[{a.variant}] numpy {np.__version__} path={numpy_path}, native={_native.native_argsort_isa()}: "
      f"{a.cases - bad}/{a.cases} identical (total length {lengths})")
expect = {"x86-simd-sort AVX-512": "avx512", "x86-simd-sort AVX2": "avx2"}.get(numpy_path, "scalar")
if _native.native_argsort_isa() != expect:
    print(f"ISA MISMATCH: numpy uses {numpy_path}, engine uses {_native.native_argsort_isa()}")
    bad += 1
sys.exit(1 if bad else 0)
