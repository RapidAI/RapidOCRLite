#!/usr/bin/env python3
"""Cross-check that every rpocr_* symbol referenced by the JNI library is
exported by the engine library (and vice versa for the ABI surface).

Usage: python check_abi.py <jni-lib> <engine-lib>
Reads ELF dynamic symbol tables (works for .so on any arch).
"""
import struct
import sys

SHT_SYMTAB = 2
SHT_DYNSYM = 11
STB_GLOBAL = 1
STB_WEAK = 2
SHN_UNDEF = 0

EM = {62: "x86-64", 183: "aarch64"}


def read_elf_symbols(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"\x7fELF":
        sys.exit(f"{path}: not an ELF file")
    is64 = data[4] == 2
    little = data[5] == 1
    e = "<" if little else ">"
    if is64:
        (e_type, e_machine, _, e_entry, e_phoff, e_shoff, _, e_ehsize,
         e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx) = \
            struct.unpack_from(e + "HHIQQQIHHHHHH", data, 16)
        shoff_fmt = e + "IIQQQQIIQQ"
    else:
        sys.exit("32-bit ELF not supported")

    sections = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        (sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size,
         sh_link, sh_info, sh_addralign, sh_entsize) = \
            struct.unpack_from(shoff_fmt, data, off)
        sections.append(dict(name_off=sh_name, type=sh_type, offset=sh_offset,
                             size=sh_size, link=sh_link, entsize=sh_entsize))

    # section name string table
    shstr = sections[e_shstrndx]
    shstrtab = data[shstr["offset"]: shstr["offset"] + shstr["size"]]

    def secname(s):
        end = shstrtab.index(b"\0", s["name_off"])
        return shstrtab[s["name_off"]:end].decode()

    defined = {}   # name -> bind info
    undefined = set()
    for s in sections:
        if s["type"] not in (SHT_SYMTAB, SHT_DYNSYM):
            continue
        strtab_sec = sections[s["link"]]
        strtab = data[strtab_sec["offset"]: strtab_sec["offset"] + strtab_sec["size"]]
        count = s["size"] // s["entsize"]
        for j in range(count):
            off = s["offset"] + j * s["entsize"]
            st_name, st_info, st_other, st_shndx, st_value, st_size = \
                struct.unpack_from(e + "IBBHQQ", data, off)
            end = strtab.index(b"\0", st_name)
            name = strtab[st_name:end].decode()
            if not name:
                continue
            bind = st_info >> 4
            if st_shndx == SHN_UNDEF:
                if bind in (STB_GLOBAL, STB_WEAK):
                    undefined.add(name)
            else:
                if bind in (STB_GLOBAL, STB_WEAK):
                    defined.setdefault(name, st_value)
    return dict(machine=EM.get(e_machine, str(e_machine)),
                defined=defined, undefined=undefined)


def main():
    jni_path, eng_path = sys.argv[1], sys.argv[2]
    jni = read_elf_symbols(jni_path)
    eng = read_elf_symbols(eng_path)

    print(f"jni:    {jni_path} [{jni['machine']}]")
    print(f"engine: {eng_path} [{eng['machine']}]")
    assert jni["machine"] == eng["machine"], "architecture mismatch!"

    rpocr_und = sorted(n for n in jni["undefined"] if n.startswith("rpocr_"))
    print(f"\nrpocr_* symbols referenced by jni lib: {len(rpocr_und)}")
    missing = [n for n in rpocr_und if n not in eng["defined"]]
    for n in rpocr_und:
        mark = "OK " if n in eng["defined"] else "MISSING"
        print(f"  [{mark}] {n}")

    eng_rpocr = sorted(n for n in eng["defined"] if n.startswith("rpocr_"))
    print(f"\nrpocr_* symbols exported by engine: {len(eng_rpocr)}")
    unused = [n for n in eng_rpocr if n not in jni["undefined"]]
    print("  (not referenced by jni lib: %s)" %
          (", ".join(unused) if unused else "none"))

    # other undefined symbols must come from libc / libstdc++ / JVM
    others = sorted(n for n in jni["undefined"]
                    if not n.startswith("rpocr_") and not n.startswith("_Z")
                    and not n.startswith("__"))
    print(f"\nother undefined (libc/JVM): {len(others)}")
    print("  " + ", ".join(others))

    if missing:
        print("\nRESULT: FAIL - unresolved engine symbols")
        sys.exit(1)
    print("\nRESULT: PASS - all engine symbols resolve")


if __name__ == "__main__":
    main()
