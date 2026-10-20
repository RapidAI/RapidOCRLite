#!/bin/bash
# Cross-build the linux loongarch64 wheels on an x86_64 host.
#
# PyPI does not accept any loongarch64 platform tag yet (warehouse only knows
# x86_64/i686/aarch64/armv7l/ppc64le/s390x/riscv64), so these wheels are
# attached to the GitHub release only and never enter the PyPI dist dir.
#
# The native library is ctypes-loaded and the package has no CPython
# extension, so one cross build is retagged for every CPython (cp39..cp313),
# exactly like the manylinux2014 wheels. The wheel library links libstdc++ /
# libgcc statically; the remaining glibc symbols (GLIBC_2.38 __isoc23_*)
# make auditwheel pick manylinux_2_38_loongarch64, which is checked here.
#
# Needs: loongarch64-linux-gnu-gcc/g++ (GCC >= 14) on PATH, cmake, ninja,
# python3 with pip. Usage: tools/build_loongarch64_wheels.sh [outdir] [testdir]
#   outdir  release wheels (default wheelhouse-loongarch64)
#   testdir optional: also write a cp<host> retag of the same payload, used to
#           install-test the wheel on the emulated distro's Python.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mkdir -p "${1:-wheelhouse-loongarch64}" && cd "${1:-wheelhouse-loongarch64}" && pwd)
TESTDIR=${2:-}
TEST_PY=${LA_TEST_PYTAG:-cp314}
PLAT=manylinux_2_38_loongarch64
PYTAGS=(cp39 cp310 cp311 cp312 cp313)

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
python3 -m pip install -q "auditwheel>=6.1" patchelf "wheel>=0.42" scikit-build-core

# _PYTHON_HOST_PLATFORM makes scikit-build-core tag the wheel linux_loongarch64.
_PYTHON_HOST_PLATFORM=linux-loongarch64 \
CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=$ROOT/cmake/toolchains/loongarch64-linux-gnu.cmake \
-DCMAKE_SHARED_LINKER_FLAGS=-static-libstdc++\ -static-libgcc \
-DPPOCR_BUILD_EXAMPLES=OFF -DPPOCR_BUILD_TESTS=OFF -DPPOCR_BUILD_STATIC=OFF \
-DPPOCR_BUILD_SHARED=ON -DPPOCR_ENABLE_VULKAN=OFF" \
CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)} \
  python3 -m pip wheel "$ROOT" --no-deps --no-build-isolation -w "$work/raw"

raw=$(ls "$work"/raw/*-linux_loongarch64.whl)
python3 -m auditwheel show "$raw"
python3 -m auditwheel repair -w "$work/repaired" "$raw"
base=$(ls "$work"/repaired/*.whl)
case "$(basename "$base")" in
  *-"$PLAT".whl) ;;
  *) echo "auditwheel produced $(basename "$base"), expected $PLAT" >&2; exit 1 ;;
esac
unzip -l "$base" | grep -q 'rapidocr/librapidocrlite.so$'
if unzip -l "$base" | grep -q '\.libs/'; then
  echo "auditwheel grafted extra libraries into the wheel" >&2; exit 1
fi

retag() {  # retag <wheel> <pytag> <destdir>
  local d; d=$(mktemp -d -p "$work")
  cp "$1" "$d/"
  python3 -m wheel tags --remove --python-tag "$2" --abi-tag "$2" "$d"/*.whl >/dev/null
  mv "$d"/*.whl "$3/"
}
for t in "${PYTAGS[@]}"; do retag "$base" "$t" "$OUT"; done
if [[ -n "$TESTDIR" ]]; then
  mkdir -p "$TESTDIR"
  retag "$base" "$TEST_PY" "$TESTDIR"
fi

# Every retag must carry the same payload (everything except dist-info).
python3 - "$OUT" ${TESTDIR:+"$TESTDIR"} << 'PY'
import hashlib, sys, zipfile
from pathlib import Path
wheels = sorted(w for d in sys.argv[1:] for w in Path(d).glob("*.whl"))
def payload(w):
    with zipfile.ZipFile(w) as z:
        return {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if ".dist-info/" not in n and not n.endswith("/")}
ref = payload(wheels[0])
for w in wheels:
    assert payload(w) == ref, f"payload differs: {w.name}"
    print("payload_ok", w.name)
PY
python3 -m auditwheel show "$OUT"/*cp313*.whl
test "$(ls "$OUT"/*-"$PLAT".whl | wc -l)" -eq "${#PYTAGS[@]}"
ls -la "$OUT"
