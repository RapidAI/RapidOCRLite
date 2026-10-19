#!/bin/sh
# Build the RapidOCRLite Java binding: JNI libraries + java classes + jar.
#
# Platforms (PLATFORMS env var, space separated; default: all):
#   win-x86_64     MinGW-w64 g++           (needs MINGW, static runtime)
#   linux-x86_64   zig c++ -target x86_64-linux-gnu
#   linux-aarch64  zig c++ -target aarch64-linux-gnu
#   macos-aarch64  zig c++ -target aarch64-macos
#
# Requires JDK 17 (JAVA_HOME). MINGW/ZIG are auto-located. A platform whose
# SDK directory under sdk/ is missing is skipped with a warning.
#
# PACKAGE_ONLY=1 skips the native builds and only runs javac + jar over
# whatever is already in build/native/<platform>/ (used by the CI package
# job, which downloads the per-platform native artifacts first). Each
# platform's build produces a self-contained build/native/<platform>/
# holding the JNI library AND the engine library under the exact name the
# jar ships it as.
set -e
cd "$(dirname "$0")"

JAVA_HOME="${JAVA_HOME:-/c/Program Files/Eclipse Adoptium/jdk-17.0.19.10-hotspot}"
# A Windows-style JAVA_HOME (C:\...) cannot be used from an MSYS/Git Bash sh
# ($JAVA_HOME/bin/javac would be a literal backslash path); convert it when
# cygpath is available (MSYS/Git Bash/Cygwin all ship it).
case "$JAVA_HOME" in
    [A-Za-z]:\\*) command -v cygpath >/dev/null 2>&1 && JAVA_HOME=$(cygpath -u "$JAVA_HOME") ;;
esac
PLATFORMS="${PLATFORMS:-win-x86_64 linux-x86_64 linux-aarch64 macos-aarch64}"
PACKAGE_ONLY="${PACKAGE_ONLY:-0}"
VERSION=5.0.0a11
JAR="dist/rapidocrlite-java-$VERSION.jar"
JNI_SRC="src/native/rapidocr_jni.cpp"

[ -x "$JAVA_HOME/bin/javac.exe" ] || [ -x "$JAVA_HOME/bin/javac" ] || \
    { echo "missing JDK: $JAVA_HOME (set JAVA_HOME=...)"; exit 1; }
JAVAC="$JAVA_HOME/bin/javac"; [ -x "$JAVAC" ] || JAVAC="$JAVA_HOME/bin/javac.exe"
JARTOOL="$JAVA_HOME/bin/jar"; [ -x "$JARTOOL" ] || JARTOOL="$JAVA_HOME/bin/jar.exe"

has_platform() {
    for p in $PLATFORMS; do [ "$p" = "$1" ] && return 0; done
    return 1
}

skip() { echo "SKIP $1: $2"; }

# ---------------------------------------------------------------- toolchains
MINGW="${MINGW:-}"
if [ -z "$MINGW" ]; then
    for d in /c/Users/ma139/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64 \
             /c/msys64/mingw64; do
        [ -x "$d/bin/g++.exe" ] && MINGW="$d" && break
    done
fi

ZIG="${ZIG:-}"
if [ -z "$ZIG" ]; then
    ZIG=$(ls /c/Users/ma139/AppData/Local/Microsoft/WinGet/Packages/zig.zig_Microsoft.Winget.Source_8wekyb3d8bbwe/zig-x86_64-windows-*/zig.exe 2>/dev/null | head -1)
fi
[ -n "$ZIG" ] || ZIG=$(command -v zig 2>/dev/null || true)

# ------------------------------------------------------------ native builds
# Each platform lands in build/native/<platform>/ with the two files the jar
# ships: the JNI library plus the engine library under its jar name.
if [ "$PACKAGE_ONLY" = "1" ]; then
    echo "== 1/3 native libraries (PACKAGE_ONLY: reusing build/native/)"
    [ -d build/native ] || { echo "PACKAGE_ONLY=1 but build/native/ is empty"; exit 1; }
else
    echo "== 1/3 native libraries"
    rm -rf build/native && mkdir -p build/native

    if has_platform win-x86_64; then
        if [ ! -f sdk/win_amd64/lib/rapidocrlite.lib ]; then
            skip win-x86_64 "sdk/win_amd64/lib/rapidocrlite.lib missing"
        elif [ -z "$MINGW" ]; then
            skip win-x86_64 "MinGW g++ not found (set MINGW=...)"
        else
            echo "== win-x86_64 (MinGW g++)"
            mkdir -p build/native/win-x86_64
            # RPOCR_SHARED: RPOCR_API expands to dllimport; link the MSVC
            # import lib directly (GNU ld accepts it). -static -static-libgcc
            # -static-libstdc++: no MinGW runtime DLLs on the target machine,
            # so the jar only needs rapidocrlite.dll (shipped next to the jni dll).
            "$MINGW/bin/g++.exe" -O2 -std=c++17 -shared -Wall -DRPOCR_SHARED \
                -static -static-libgcc -static-libstdc++ \
                -I"$JAVA_HOME/include" -I"$JAVA_HOME/include/win32" \
                -Isdk/win_amd64/include \
                -o build/native/win-x86_64/rapidocrlite_jni.dll "$JNI_SRC" \
                sdk/win_amd64/lib/rapidocrlite.lib -lwinhttp
            cp sdk/win_amd64/bin/rapidocrlite.dll build/native/win-x86_64/
        fi
    fi

    # zig_cross <platform> <zig-target> <jni-header-dir> <sdk-dir> \
    #           <engine-src-file> <jar-engine-name> <out-file>
    zig_cross() {
        plat="$1"; target="$2"; hdr="$3"; sdkdir="$4"; enginesrc="$5"; \
        enginedst="$6"; outfile="$7"
        if [ ! -f "$sdkdir/include/rapidocrlite/rapidocrlite.h" ]; then
            skip "$plat" "$sdkdir missing (download the SDK tarball)"
            return 0
        fi
        if [ -z "$ZIG" ]; then
            skip "$plat" "zig not found (set ZIG=...)"
            return 0
        fi
        echo "== $plat (zig c++ -target $target)"
        mkdir -p "build/native/$plat"
        # The engine library is passed as a positional input (zig mangles -L
        # paths with drive letters on Windows). Its SONAME / install name is
        # recorded as the dependency, so the loader resolves it from the
        # already-loaded engine module (or via $ORIGIN / @loader_path rpath).
        "$ZIG" c++ -shared -fPIC -O2 -std=c++17 -target "$target" \
            -I"$JAVA_HOME/include" -I"jni-headers/$hdr" -I"$sdkdir/include" \
            -o "build/native/$plat/$outfile" "$JNI_SRC" \
            "$sdkdir/lib/$enginesrc" \
            -Wl,-rpath,'$ORIGIN' -Wl,-rpath,@loader_path \
            -Wl,-soname,"$outfile" -Wl,-install_name,"$outfile"
        # ship the real engine file under the name the loader expects
        # (Linux: soname librapidocrlite.so.5; macOS: install name
        # librapidocrlite.5.dylib — the SDK's unversioned names are symlinks
        # that Git Bash tar materializes as empty files).
        cp "$sdkdir/lib/$enginesrc" "build/native/$plat/$enginedst"
    }

    if has_platform linux-x86_64; then
        zig_cross linux-x86_64 x86_64-linux-gnu linux \
            sdk/linux_x86_64/RapidOCRLite-SDK-linux_x86_64-5.0.0a11 \
            librapidocrlite.so.5.0.0 librapidocrlite.so.5 \
            librapidocrlite_jni.so
    fi
    if has_platform linux-aarch64; then
        zig_cross linux-aarch64 aarch64-linux-gnu linux \
            sdk/linux_aarch64/RapidOCRLite-SDK-linux_aarch64-5.0.0a11 \
            librapidocrlite.so.5.0.0 librapidocrlite.so.5 \
            librapidocrlite_jni.so
    fi
    if has_platform macos-aarch64; then
        zig_cross macos-aarch64 aarch64-macos darwin \
            sdk/macosx_arm64/RapidOCRLite-SDK-macosx_arm64-5.0.0a11 \
            librapidocrlite.5.0.0.dylib librapidocrlite.5.dylib \
            librapidocrlite_jni.dylib
    fi
fi

# ---------------------------------------------------------------- java build
echo "== 2/3 java classes"
rm -rf build/classes && mkdir -p build/classes
"$JAVAC" -encoding UTF-8 -d build/classes $(find src/java -name '*.java')

# ------------------------------------------------------------------ jar
echo "== 3/3 package jar"
mkdir -p build/classes/native dist
found=0
for plat in win-x86_64 linux-x86_64 linux-aarch64 macos-aarch64; do
    src="build/native/$plat"
    [ -d "$src" ] || continue
    mkdir -p "build/classes/native/$plat"
    cp "$src"/* "build/classes/native/$plat/"
    found=1
done
[ "$found" = "1" ] || { echo "no native libraries found under build/native/"; exit 1; }
rm -f "$JAR"
"$JARTOOL" --create --file "$JAR" \
    --main-class ai.rapidocr.lite.RapidOcrDemo \
    -C build/classes .

echo "build ok:"
echo "  $JAR"
ls -la "$JAR"
echo "native contents:"
"$JARTOOL" --list --file "$JAR" | grep "^native/" | sort
