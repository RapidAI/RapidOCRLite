#!/bin/bash
# Install-test the loongarch64 wheels inside a Debian loong64 userland
# (QEMU user-mode emulation via binfmt; emulator-verified only).
#
# Usage (as root, inside the loong64 container/chroot, repo at $ROOT):
#   tools/test_loongarch64_wheels.sh <release-wheel-dir> <test-wheel-dir>
# Debian loong64 ships numpy & co. only for its default python3 (3.14 in
# forky), so the OCR runs on the test retag of the very same payload
# (build_loongarch64_wheels.sh asserts the payloads are identical); the
# cp313 release wheel is install-checked with python3.13 (pip tag check).
# ci-models/ must already hold the tiny models (fetch_ci_models.py on the host).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
REL=$1
TEST=$2
cd "$ROOT"
uname -m
test "$(uname -m)" = loongarch64

export DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
if ! command -v python3.13 >/dev/null; then
  for i in 1 2 3 4 5; do apt-get update -qq && break; sleep 20; done
  apt-get install -y -qq --no-install-recommends python3 python3.13 python3-pip python3-numpy \
    python3-pil python3-yaml python3-requests python3-tqdm >/dev/null
fi
python3 --version

# Release wheel: pip accepts the cp313 manylinux_2_38_loongarch64 tag here.
python3.13 -m pip install -q --no-deps --target /tmp/rapidocr-cp313 "$REL"/*-cp313-cp313-*.whl
ls /tmp/rapidocr-cp313/rapidocr/librapidocrlite.so
echo "cp313_install_ok"

python3 -m pip install -q --no-deps --force-reinstall "$TEST"/*.whl
cd /tmp
python3 - << 'PY'
import rapidocr
info = rapidocr.cpu_info()
print(rapidocr.__version__, info)
assert info["active_isa"] == "lasx", info
print("isa_dispatch_ok lasx")
PY
cd "$ROOT"

# Texts against the x86 Linux CPU reference: lasx, lsx and scalar kernels.
PPOCR_CHECK_ISAS=lasx,lsx,scalar PPOCR_CHECK_REPS=0 PPOCR_CHECK_TEXTS_ONLY=1 \
  python3 python/tests/ci_wheel_check.py 2>&1 | grep -v '^timing'

# model_type="lite" on the README samples: texts + boxes against the baselines.
PPOCR_FORCE_ISA=lasx LITE_REPS=0 python3 python/tests/ci_lite_examples_check.py

python3 -m rapidocr python/tests/data/hello.png --model-type tiny \
  --det ci-models/PP-OCRv6_det_tiny.onnx --rec ci-models/PP-OCRv6_rec_tiny.onnx \
  --dict ci-models/ppocrv6_tiny_dict.txt | tee /tmp/rapidocr-cli.txt
grep -q "Hello RapidOCR 123" /tmp/rapidocr-cli.txt
echo "loongarch64_wheel_check_ok"
