#!/usr/bin/env python3
"""Minimal Mach-O load-command dumper for verifying cross-built dylibs."""
import struct
import sys
import os

MH_MAGIC_64 = 0xFEEDFACF
LC_LOAD_DYLIB = 0x0C
LC_ID_DYLIB = 0x0D
LC_RPATH = 0x1C | 0x80000000
LC_REEXPORT_DYLIB = 0x1F | 0x80000000
LC_LOAD_WEAK_DYLIB = 0x18 | 0x80000000
LC_DYLD_INFO = 0x22 | 0x80000000
LC_DYLD_INFO_ONLY = 0x22 | 0x80000000
LC_DYLD_EXPORTS_TRIE = 0x33 | 0x80000000

CPU_NAMES = {0x0100000C: "arm64", 0x0200000C: "arm64_32", 0x00000007: "x86_64"}


def read_uleb(data, off):
    result = 0
    shift = 0
    while True:
        b = data[off]
        off += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, off
        shift += 7


def collect_exports(data, start, end, prefix=""):
    out = []
    stack = [(start, prefix)]
    while stack:
        node_off, node_prefix = stack.pop()
        if node_off >= end:
            continue
        off = node_off
        b = data[off]
        off += 1
        if b != 0:
            flags, off = read_uleb(data, off)
            if flags & 0x08:
                ordinal, off = read_uleb(data, off)
                name_end = data.index(b"\0", off)
                out.append((node_prefix, "reexport:" +
                            data[off:name_end].decode()))
                off = name_end + 1
            else:
                addr, off = read_uleb(data, off)
                if flags & 0x01:
                    _, off = read_uleb(data, off)
                out.append((node_prefix, hex(addr)))
            # NOTE: a terminal node may still have children (a longer
            # symbol name that extends this one); fall through.
        if off >= end:
            continue
        child_count, off = read_uleb(data, off)
        for _ in range(child_count):
            edge_end = data.index(b"\0", off)
            edge = data[off:edge_end].decode()
            off = edge_end + 1
            child_off, off = read_uleb(data, off)
            stack.append((start + child_off, node_prefix + edge))
    return out


def main(path):
    with open(path, "rb") as f:
        data = f.read()
    magic = struct.unpack_from(">I", data, 0)[0]
    if magic != MH_MAGIC_64:
        # also accept little-endian magic
        magic_le = struct.unpack_from("<I", data, 0)[0]
        if magic_le != MH_MAGIC_64:
            print(f"not a 64-bit Mach-O: magic=0x{magic:08x}")
            sys.exit(1)
        endian = "<"
    else:
        endian = ">"
    (magic, cputype, cpusubtype, filetype, ncmds, sizeofcmds, flags) = \
        struct.unpack_from(endian + "IiiIIII", data, 0)
    print(f"file: {path}")
    print(f"  magic=0x{magic:08x} cputype=0x{cputype:08x} "
          f"({CPU_NAMES.get(cputype, '?')}) filetype={filetype} ncmds={ncmds}")
    off = 32
    dylibs = []
    rpaths = []
    idname = None
    export_off = export_size = 0
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from(endian + "II", data, off)
        if cmd == LC_DYLD_EXPORTS_TRIE:
            export_off, export_size = struct.unpack_from(endian + "II", data, off + 8)
        if cmd == LC_DYLD_INFO_ONLY:
            # rebase,bind,weak_bind,lazy_bind,export (off,size pairs)
            export_off, export_size = struct.unpack_from(
                endian + "II", data, off + 8 + 4 * 8)
        if cmd in (LC_LOAD_DYLIB, LC_LOAD_WEAK_DYLIB, LC_REEXPORT_DYLIB, LC_ID_DYLIB):
            name_off = struct.unpack_from(endian + "I", data, off + 8)[0]
            name = data[off + name_off: off + cmdsize].split(b"\0")[0].decode()
            if cmd == LC_ID_DYLIB:
                idname = name
            else:
                kind = {LC_LOAD_DYLIB: "LOAD",
                        LC_LOAD_WEAK_DYLIB: "WEAK",
                        LC_REEXPORT_DYLIB: "REEXPORT"}[cmd]
                dylibs.append((kind, name))
        elif cmd == LC_RPATH:
            name_off = struct.unpack_from(endian + "I", data, off + 8)[0]
            name = data[off + name_off: off + cmdsize].split(b"\0")[0].decode()
            rpaths.append(name)
        off += cmdsize
    print(f"  install_name: {idname}")
    for r in rpaths:
        print(f"  rpath: {r}")
    for kind, name in dylibs:
        print(f"  {kind}: {name}")
    if export_off:
        syms = collect_exports(data, export_off, export_off + export_size)
        print(f"  exports ({len(syms)}):")
        for name, addr in sorted(syms):
            print(f"    {name} @ {addr}")
    else:
        print("  (no LC_DYLD_EXPORTS_TRIE)")
    if os.environ.get("MACHODUMP_ALL"):
        print("  --- all load commands ---")
        off2 = 32
        for _ in range(ncmds):
            cmd, cmdsize = struct.unpack_from(endian + "II", data, off2)
            print(f"    cmd=0x{cmd:08x} size={cmdsize}")
            off2 += cmdsize


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
