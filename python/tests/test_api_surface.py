"""Public Python API must match the original RapidOCRLite (only the 'lite' model_type value is new)."""
import json, os, subprocess, sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
NEW = HERE.parent / "src"
ORIG = Path(os.environ.get("RAPIDOCR_ORIG_SRC", HERE.parents[2] / "RapidOCRLite_orig/python/src"))
ALLOWED = {("rapidocr.typings.ModelType", "members", "LITE")}


def dump(src):
    out = subprocess.run([sys.executable, str(HERE / "api_surface.py"), str(src)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def diff(a, b, path=()):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                yield path + (k,), a.get(k, "<missing>"), b.get(k, "<missing>")
            else:
                yield from diff(a[k], b[k], path + (k,))
    elif a != b:
        yield path, a, b


@pytest.mark.skipif(not ORIG.exists(), reason="original RapidOCRLite source not available")
def test_public_api_unchanged():
    d = [x for x in diff(dump(ORIG), dump(NEW)) if x[0] not in ALLOWED]
    assert not d, "\n".join(f"{'.'.join(p)}: orig={o!r} new={n!r}" for p, o, n in d)
