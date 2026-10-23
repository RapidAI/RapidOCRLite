# RapidOCRLite Java 绑定（rapidocrlite-java）

对 [RapidOCRLite v5.0.0a15](https://github.com/RapidAI/RapidOCRLite/releases/tag/v5.0.0a15)
官方 SDK 的 Java 封装（JNI），让 Java 程序可以直接调用 OCR 引擎，支持
**Windows / Linux / macOS 三大平台**。**分发物是一个可运行 jar，内置各平台原生库，
目标机器无需安装任何工具链或配置环境变量。**

| 平台 | JNI 库 | 引擎库 | 工具链 |
|---|---|---|---|
| Windows x64 | `rapidocrlite_jni.dll` | `rapidocrlite.dll` | MinGW-w64 g++（运行时全静态） |
| Linux x64 | `librapidocrlite_jni.so` | `librapidocrlite.so.5` | zig c++（libc++ 静态） |
| Linux arm64 | `librapidocrlite_jni.so` | `librapidocrlite.so.5` | zig c++（libc++ 静态） |
| Linux loongarch64 | `librapidocrlite_jni.so` | `librapidocrlite.so.5` | zig c++（libc++ 静态；仅 QEMU 模拟器验证） |
| macOS arm64 | `librapidocrlite_jni.dylib` | `librapidocrlite.5.dylib` | zig c++（libc++ 静态） |
| macOS x86_64 | `librapidocrlite_jni.dylib` | `librapidocrlite.5.dylib` | zig c++（libc++ 静态） |

## 快速开始

```bash
# 打印引擎 CPU 信息（验证原生库能加载）
java -jar rapidocrlite-java-5.0.0a15.jar --cpu

# Indra lite 模型（传包含 meta.json 的模型目录）
java -Dsun.stdout.encoding=UTF-8 -jar rapidocrlite-java-5.0.0a15.jar \
    --lite models/lite poster.jpg

# PP-OCRv6 模型（det/rec onnx + 字典）
java -jar rapidocrlite-java-5.0.0a15.jar \
    --ppocr PP-OCRv6_det_small.onnx PP-OCRv6_rec_small.onnx ppocrv6_dict.txt doc.jpg

# 网络图片（http/https URL 直接下载识别）
java -jar rapidocrlite-java-5.0.0a15.jar \
    --lite models/lite https://example.com/poster.jpg
```

> JDK 17 中文输出加 `-Dsun.stdout.encoding=UTF-8`（JDK 18+ 为 `-Dstdout.encoding=UTF-8`）。

## 在自己的代码中使用

把 jar 加入 classpath 即可（无其他依赖，纯 JDK）：

```java
import ai.rapidocr.lite.*;

// PP-OCRv6
try (RapidOcr ocr = RapidOcr.forPPOCR(
        "models/PP-OCRv6_det_small.onnx",
        "models/PP-OCRv6_rec_small.onnx",
        "models/ppocrv6_dict.txt")) {
    for (TextLine line : ocr.recognize(Path.of("poster.jpg"))) {
        System.out.printf("%.3f  %s%n", line.confidence(), line.text());
    }
}

// Indra lite
try (RapidOcr ocr = RapidOcr.forLite("models/lite")) {
    List<TextLine> lines = ocr.recognize(
            new URL("https://example.com/doc.jpg"));   // 也支持本地 Path
    for (TextLine line : lines) {
        float[] topLeft = line.box()[0];               // TL, TR, BR, BL 四点
        System.out.println(line.confidence() + " " + line.text()
                + " @" + topLeft[0] + "," + topLeft[1]);
    }
}

// 批量（一次调用识别多张，结果按输入顺序返回）
List<List<TextLine>> pages = ocr.recognizeBatch(List.of(img1, img2));

// PPM 走 SDK 自带的 rpocr_load_ppm（无损回归测试用）
List<TextLine> lines = ocr.recognizePpm("tests/lite_crop.ppm");

// 单词框 / 单字框（与 rapidocr main 的 return_word_box / return_single_char_box 相同）
for (TextLine line : ocr.recognize(Path.of("poster.jpg"))) {
    for (Word w : line.words()) {                      // charBoxes() 为逐字框，words(true) 同
        int[][] b = w.box();                           // 4×2 整数角点
        System.out.println(w.text() + " " + w.confidence() + " @" + b[0][0] + "," + b[0][1]);
    }
}

// 引擎信息
RapidOcr.cpuInfo();      // isa=avx512, threads=16, ...
RapidOcr.lastError();    // 最近一次引擎错误
```

### API 一览

| 类 / 方法 | 说明 |
|---|---|
| `RapidOcr.forPPOCR(det, rec, dict)` / `forLite(dir)` | 创建引擎；失败抛 `IllegalStateException`（含 `rpocr_last_error()`） |
| `RapidOcr.forPPOCR(..., OcrOptions)` / `forLite(dir, OcrOptions)` | 带调参创建 |
| `recognize(BufferedImage)` / `(Path)` / `(URL)` | 单张识别；内部转 **packed RGB**（契约见下） |
| `recognizePpm(path)` | 经 SDK `rpocr_load_ppm` 识别 P6 PPM |
| `recognizeBatch(List<BufferedImage>)` | `rpocr_recognize_batch`，顺序保持 |
| `TextLine` | 不可变结果行：`text()` / `confidence()` / `box()`(4×2) / `bboxX/Y/W/H()` / `clsLabel()` / `clsScore()`；单词框 `words()`（main `return_word_box`）、单字框 `charBoxes()`（`return_single_char_box`）、`words(boolean)`，以及逐字数据 `recSteps()` / `charCols()` / `charProbs()` / `cropWidth()` / `cropHeight()` / `recMaxWhRatio()` |
| `Word` | 单词/单字框：`text()` / `confidence()`（main 的 `round(prob, 5)`）/ `box()`（4×2 整数角点），与 Python 绑定的 `word_results` 在相同像素上逐项一致 |
| `OcrOptions.builder()` | `backend` / `detLimitSideLen` / `recParallelism` / `detThreshold` / `detBoxThreshold` / `detUnclipRatio` |
| `RapidOcr.cpuInfo()` / `lastError()` | 静态工具 |
| `close()` | 释放引擎句柄（`AutoCloseable`，幂等） |

线程安全：单个 `RapidOcr` 实例的方法均 `synchronized`，可多线程共用；
需要并行时建议每线程一个实例。

## 打包内容与原生库加载

```
rapidocrlite-java-5.0.0a15.jar（约 7.3 MB）
├── ai/rapidocr/lite/*.class              # Java API + 演示 Main-Class
└── native/
    ├── win-x86_64/    rapidocrlite_jni.dll + rapidocrlite.dll
    ├── linux-x86_64/  librapidocrlite_jni.so + librapidocrlite.so.5
    ├── linux-aarch64/ librapidocrlite_jni.so + librapidocrlite.so.5
    ├── linux-loongarch64/ librapidocrlite_jni.so + librapidocrlite.so.5
    ├── macos-aarch64/ librapidocrlite_jni.dylib + librapidocrlite.5.dylib
    └── macos-x86_64/  librapidocrlite_jni.dylib + librapidocrlite.5.dylib
```

`NativeLoader` 按 `os.name` + `os.arch` 检测平台（`win-x86_64` / `linux-x86_64` /
`linux-aarch64` / `linux-loongarch64`（`os.arch` 为 `loongarch64`）/ `macos-aarch64` / `macos-x86_64`），加载顺序：

1. 系统属性 `-Drapidocr.lite.native=<库文件或目录>` 显式指定；
2. 从 jar 提取当前平台的引擎库 + JNI 库到临时目录，**先 `System.load` 引擎库**，再加载
   JNI 库（JVM 不会把库所在目录加入依赖搜索路径，先装引擎可让依赖从已加载模块解析；
   Linux 侧 JNI 库另带 `$ORIGIN` rpath、macOS 侧带 `@loader_path` rpath 双保险）；
3. 回退 `System.loadLibrary("rapidocrlite_jni")`（库在 `java.library.path` 时）。

各平台原生库的依赖都收敛到"引擎库 + 系统库"：Windows 侧静态链接 MinGW 运行时
（不依赖 libgcc/libstdc++ DLL），Linux/macOS 侧静态链接 libc++（只依赖 libc /
libSystem 与引擎库）。

像素契约：`rpocr_image.rgb` 是 packed **RGB**（row-major，w×h×3）。Java 侧把
`BufferedImage` 转 packed RGB（`TYPE_3BYTE_BGR` 有快路径；带 Alpha 的图按白底合成）。

## 构建

```bash
sh build.sh                        # 全部平台（SDK 齐备时）
PLATFORMS=linux-x86_64 sh build.sh # 只构建指定平台
PACKAGE_ONLY=1 sh build.sh         # 只打包已有 build/native/（CI 合并打包用）
```

依赖：
- JDK 17（`JAVA_HOME`）；
- Windows 平台：MinGW-w64 g++（WinLibs UCRT 版，`MINGW=` 指定，默认自动探测）；
- Linux/macOS 平台：[zig](https://ziglang.org) 0.17+（`ZIG=` 指定，默认自动探测）；
- 各平台官方 SDK（`sdk/` 已加入 .gitignore，需自行下载，见下文"获取 SDK"）。

可用环境变量：`JAVA_HOME=`、`MINGW=`、`ZIG=`、`PLATFORMS=`、`PACKAGE_ONLY=1`
（跳过原生构建，只对已有的 `build/native/<platform>/` 跑 javac + jar）。
产物在 `dist/rapidocrlite-java-<ver>.jar`。

### 获取 SDK

从 [RapidOCRLite v5.0.0a15 release](https://github.com/RapidAI/RapidOCRLite/releases/tag/v5.0.0a15)
下载对应平台的 SDK 解压到 `sdk/` 下（目录名须与 `build.sh` 期望的一致；CI 中由仓库根目录的
`.github/workflows/build-wheels.yml` 自动完成）：

```bash
# Linux x64 / arm64、macOS arm64 / x86_64：tar.gz 自带顶层目录，直接解压，勿加 --strip-components
mkdir -p sdk/linux_x86_64 sdk/linux_aarch64 sdk/macosx_arm64 sdk/macosx_x86_64
curl -sSL -o /tmp/sdk.tar.gz \
  https://github.com/RapidAI/RapidOCRLite/releases/download/v5.0.0a15/RapidOCRLite-SDK-linux_x86_64-5.0.0a15.tar.gz
tar xzf /tmp/sdk.tar.gz -C sdk/linux_x86_64
# linux_aarch64 / macosx_arm64 / macosx_x86_64 同理：
#   RapidOCRLite-SDK-linux_aarch64-5.0.0a15.tar.gz → sdk/linux_aarch64
#   RapidOCRLite-SDK-linux_loongarch64-5.0.0a15.tar.gz → sdk/linux_loongarch64
#   RapidOCRLite-SDK-macosx_arm64-5.0.0a15.tar.gz  → sdk/macosx_arm64
#   RapidOCRLite-SDK-macosx_x86_64-5.0.0a15.tar.gz → sdk/macosx_x86_64

# Windows x64：zip 解包后，把其中 bin/lib/include 复制到 sdk/win_amd64/
curl -sSL -o sdk.zip \
  https://github.com/RapidAI/RapidOCRLite/releases/download/v5.0.0a15/RapidOCRLite-SDK-win_amd64-5.0.0a15.zip
```

```
java/
├── build.sh                              # 一键构建（各平台 JNI 库 → javac → jar）
├── jni-headers/{linux,darwin}/jni_md.h   # 跨平台 JNI 头（OpenJDK 官方同款）
├── tools/check_abi.py                    # 静态校验：JNI 库引用的引擎符号可解析
├── tools/machodump.py                    # Mach-O load command / 导出符号解析
├── src/java/ai/rapidocr/lite/            # Java API
│   ├── RapidOcr.java                     #   门面类 + native 方法声明
│   ├── TextLine.java / CpuInfo.java      #   结果与信息类型
│   ├── OcrOptions.java                   #   调参（builder）
│   ├── NativeLoader.java                 #   平台检测 + jar 内库提取加载
│   └── RapidOcrDemo.java                 #   jar 的 Main-Class 演示
├── src/native/rapidocr_jni.cpp           # JNI 桥接层（可移植 C++，无平台专有代码）
├── sdk/                                  # 各平台官方 SDK（.gitignore，见"获取 SDK"）
└── dist/                                 # 构建产物（.gitignore；CI 产出五平台 fat jar）
```

CI workflow 与 SDK / wheel 同属一条流水线：仓库根目录 `.github/workflows/build-wheels.yml`
（推送 `main` 只构建；推送 `v5*` 标签即构建并发布 SDK、wheel 与本 jar，无需单独的 `java-*` 标签）。

CI（`.github/workflows/build-wheels.yml`）中 Java 相关的 job 分两类：

- **五个平台构建 job**（ubuntu-latest / ubuntu-24.04-arm / macos-latest×2 /
  windows-latest）：直接下载本次运行产出的 SDK artifact（不再依赖 release 上的
  tarball，因而不会与 SDK 上传竞争），在真实 runner 上构建对应平台原生库，跑
  `--cpu` 冒烟测试，并上传原生库 artifact（`java-native-<platform>`）；
- **`java-package` job**（ubuntu-latest，`needs` 全部五个构建 job）：下载各平台
  原生库 artifact 到 `build/native/<platform>/`，以 `PACKAGE_ONLY=1 sh build.sh`
  只跑 javac + jar，**合并产出六平台全量 fat jar**（artifact
  `jar-all-platforms`），再跑一次 `--cpu` 冒烟验证多平台 jar 能正确选中
  当前平台的资源。该 job 不需要 zig / MinGW 工具链，也不需要下载任何 SDK；
- **`attach-java` job**：把 fat jar 上传到本次 release（与 SDK、wheel 同一个
  GitHub Release）。

arm64 Linux runner 对公开仓库免费。

## 验证结果

**Windows 11 x64（AMD Zen4，JDK 17）——全功能实测：**

| 测试 | 结果 |
|---|---|
| 原生加载（jar 内库提取，平台检测） | OK，`isa=avx512 threads=16` |
| lite + 中文海报（本地 JPEG） | 21 行，与 C ABI 真 RGB 测试逐行一致（约 240 ms） |
| lite + 网络 URL 下载识别 | 21 行（2583 ms，含下载与 ImageIO 解码） |
| lite + 交换通道 PPM | 16 行 = 官方参考 JSON 行数（参考按 BGR 契约生成，自洽） |
| PP-OCRv6 + 英文文档 | 4 行，文本与 C ABI 测试逐字一致（约 250 ms） |
| 批量接口（3 张混合） | 21/4/21 行，顺序保持，与单张一致（约 230 ms） |
| OcrOptions 变体（recParallelism=2, detLimitSideLen=736） | 正常 |
| close() / isClosed() / lastError() | 正常 |

**Linux x64 / arm64、macOS arm64——结构 + 符号级验证（本机无对应系统，未跑运行时）：**

| 检查 | 结果 |
|---|---|
| ELF/Mach-O 结构（机器类型、filetype、install_name） | 正确 |
| 依赖收敛 | Linux：`librapidocrlite.so.5` + libc/libpthread/libdl；macOS：`@rpath/librapidocrlite.5.dylib` + libSystem |
| SONAME / install name 与官方引擎一致 | 一致（引擎先加载即可解析） |
| rpath | Linux `$ORIGIN`、macOS `@loader_path`（引擎库与 JNI 库同目录时自解析） |
| JNI 导出 | Linux：10 个 `Java_*`；macOS：导出 trie 含 10 个 `Java_*` + `JNI_OnLoad/OnUnload` |
| 引擎符号交叉校验（tools/check_abi.py） | 10/10 `rpocr_*` 引用全部可被引擎库解析 |
| 运行时冒烟 | 由 CI 四平台 runner 执行（见上） |

**Linux loongarch64——仅 QEMU 用户模式模拟器验证（无真机测试、无性能数据）：**

CI job `java linux-loongarch64 (cross + QEMU)` 用 zig（`-target loongarch64-linux-gnu`）
交叉编译 JNI 库，同一个 jar 先在 x86_64 runner 上识别 `docs/lite_examples/images/ch_en_num.jpg`
（PP-OCRv6 tiny 模型），再在 QEMU loong64（`ghcr.io/loong64/debian`，按 digest 固定，
Debian `openjdk-21-jre-headless`）中识别同一张图：`isa=lasx`，15 行文本、置信度与坐标
与 x86_64 输出逐行一致。

## 注意事项

- **模型需自备**：SDK 不含模型。PP-OCRv6 从 ModelScope `RapidAI/RapidOCR` 下载；
  Indra lite 模型在官方 sdist（`rapidocrlite-5.0.0a15.tar.gz`）的
  `python/src/rapidocr/models/lite/` 内（约 18 MB）。
- **lite 模型许可证为 modified OpenRAIL-M**（非 Apache-2.0）：营收/资金超 2000 万美元
  或用于竞争产品时受限，商用前请读 `models/lite/LICENSE`。
- 本绑定代码按 Apache-2.0 发布，与上游仓库一致。
- 引擎默认 CPU 后端；GPU 后端（Vulkan）需对应 SDK 构建，官方 CPU SDK 仅含 CPU 内核。
- Linux 引擎库要求 glibc ≥ 2.14 时代（`GLIBC_2.16` 封顶）；JNI 库不引入更高要求。
