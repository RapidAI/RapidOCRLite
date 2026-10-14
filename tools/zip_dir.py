#!/usr/bin/env python3
"""Zip a directory portably: forward-slash names, real directory entries,
Unix permission bits. Used for SDK archives instead of Compress-Archive,
which writes backslash separators that break unzip / zipfile on Linux.

usage: zip_dir.py SRC_DIR OUT.zip   (archive root is SRC_DIR's basename)
"""
import os
import stat
import sys
import zipfile


def main() -> int:
    src, out = os.path.abspath(sys.argv[1]), sys.argv[2]
    base = os.path.dirname(src)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for root, dirs, files in os.walk(src):
            dirs.sort()
            rel_root = os.path.relpath(root, base).replace(os.sep, "/")
            info = zipfile.ZipInfo(rel_root + "/")
            info.external_attr = (stat.S_IFDIR | 0o755) << 16 | 0x10
            zf.writestr(info, b"")
            for name in sorted(files):
                path = os.path.join(root, name)
                info = zipfile.ZipInfo.from_file(path, f"{rel_root}/{name}")
                exe = name.endswith((".dll", ".exe", ".so", ".dylib")) or ".so." in name
                info.external_attr = (stat.S_IFREG | (0o755 if exe else 0o644)) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                with open(path, "rb") as fh:
                    zf.writestr(info, fh.read())
    with zipfile.ZipFile(out) as zf:
        bad = [n for n in zf.namelist() if "\\" in n]
        if bad:
            raise SystemExit(f"backslash in zip names: {bad[:3]}")
        print(f"{out}: {len(zf.namelist())} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
