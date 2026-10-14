"""Portable wheel check (used on Windows): ISA dispatch, OCR on the sample
images against the Linux CPU reference texts, CLI, GPU fallback, pytest."""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
DATA = ROOT / "python" / "tests" / "data"
MODELS = ROOT / "ci-models"

import rapidocr  # noqa: E402
from rapidocr import _native  # noqa: E402

info = rapidocr.cpu_info()
print(rapidocr.__version__, platform.platform(), info, flush=True)
isa = info["active_isa"]
if platform.machine().lower() in {"x86_64", "amd64"}:
    # 32-bit (win32) builds ship AVX2 + scalar only.
    assert info["avx2_compiled"], info
    assert info["avx512_compiled"] or sys.maxsize < 2**32, info
    assert isa in {"avx2", "avx512"}, info
print("isa_dispatch_ok", isa)

if not (MODELS / "PP-OCRv6_det_tiny.onnx").is_file():
    subprocess.run([sys.executable, str(ROOT / "python/tests/fetch_ci_models.py")], check=True)

root = Path(rapidocr.__file__).resolve().parent
lib = next(p for p in (root / "ppocr.dll", root / "libppocr.so", root / "libppocr.dylib") if p.is_file())
os.environ.update(
    PPOCR_LIBRARY=str(lib),
    PPOCR_TEST_IMAGE=str(DATA / "hello.png"),
    PPOCR_TEST_DET=str(MODELS / "PP-OCRv6_det_tiny.onnx"),
    PPOCR_TEST_REC=str(MODELS / "PP-OCRv6_rec_tiny.onnx"),
    PPOCR_TEST_DICT=str(MODELS / "ppocrv6_tiny_dict.txt"),
)

try:
    print("backend_info", _native.backend_info())
except Exception as exc:  # GPU probing must never break the CPU path
    raise SystemExit(f"backend probe raised: {exc!r}")

from rapidocr import RapidOCR  # noqa: E402

engine = RapidOCR(params={
    "Det.model_path": os.environ["PPOCR_TEST_DET"],
    "Rec.model_path": os.environ["PPOCR_TEST_REC"],
    "Rec.rec_keys_path": os.environ["PPOCR_TEST_DICT"],
    "Det.model_type": "tiny",
    "Rec.model_type": "tiny",
    "Global.use_cls": False,
})
def check_reference(isa_name):
    """Run the sample images in a child process under PPOCR_FORCE_ISA."""
    code = (
        "import json,os,sys,time\n"
        "from rapidocr import RapidOCR, cpu_info\n"
        "e=RapidOCR(params={'Det.model_path':os.environ['PPOCR_TEST_DET'],"
        "'Rec.model_path':os.environ['PPOCR_TEST_REC'],'Rec.rec_keys_path':os.environ['PPOCR_TEST_DICT'],"
        "'Det.model_type':'tiny','Rec.model_type':'tiny','Global.use_cls':False})\n"
        "exp=json.load(open(sys.argv[1],encoding='utf-8'));bad=0\n"
        "isa=cpu_info()['active_isa']\n"
        "for n,t in exp.items():\n"
        "    p=os.path.join(sys.argv[2],n);g=list(e(p).txts or ());ts=[]\n"
        "    for _ in range(3):\n"
        "        t0=time.perf_counter();e(p);ts.append((time.perf_counter()-t0)*1000)\n"
        "    ts.sort();print(f'timing {n} isa={isa} median_ms={ts[1]:.1f} lines={len(g)}',flush=True)\n"
        "    if g!=t:\n"
        "        bad+=1;print('MISMATCH',n,len(g),len(t))\n"
        "        [print('  diff',i,repr(x),'!=',repr(y)) for i,(x,y) in enumerate(zip(g,t)) if x!=y]\n"
        "    else: print('match',n,len(g))\n"
        "sys.exit(1 if bad else 0)\n"
    )
    env = dict(os.environ, PPOCR_FORCE_ISA=isa_name, PYTHONIOENCODING="utf-8")
    print(f"== reference check, PPOCR_FORCE_ISA={isa_name}", flush=True)
    return subprocess.call([sys.executable, "-c", code, str(DATA / "expected_cpu_texts.json"), str(DATA)], env=env)


isas = ["avx512", "avx2"] if info["avx512"] and info["avx512_compiled"] else ["avx2"]
if os.environ.get("PPOCR_CHECK_SCALAR"):
    isas.append("scalar")
if os.environ.get("PPOCR_CHECK_ISAS"):  # e.g. "avx512" under Intel SDE
    isas = os.environ["PPOCR_CHECK_ISAS"].split(",")
failed = [name for name in isas if check_reference(name) != 0]
if failed:
    raise SystemExit(f"text differs from the Linux CPU reference under: {failed}")
print("reference_texts_ok", isas)
if os.environ.get("PPOCR_CHECK_TEXTS_ONLY"):
    sys.exit(0)

env = dict(os.environ, PYTHONIOENCODING="utf-8")
args = [str(DATA / "hello.png"), "--model-type", "tiny", "--det", os.environ["PPOCR_TEST_DET"],
        "--rec", os.environ["PPOCR_TEST_REC"], "--dict", os.environ["PPOCR_TEST_DICT"]]
out = subprocess.run([sys.executable, "-m", "rapidocr", *args], check=True, capture_output=True,
                     text=True, encoding="utf-8", env=env).stdout
print(out)
assert "Hello RapidOCR 123" in out
print("cli_ok")

sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", str(ROOT / "python/tests")]))
