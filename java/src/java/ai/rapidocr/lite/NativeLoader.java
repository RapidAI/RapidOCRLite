package ai.rapidocr.lite;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Locale;

/**
 * Loads the native layer from inside the JAR.
 *
 * <p>The JAR carries one {@code /native/<platform>/} directory per supported
 * platform ({@code win-x86_64}, {@code linux-x86_64}, {@code linux-aarch64},
 * {@code linux-loongarch64},
 * {@code macos-aarch64}, {@code macos-x86_64}); each holds the JNI library
 * plus the matching RapidOCRLite engine library.
 *
 * <p>Resolution order:
 * <ol>
 *   <li>system property {@code rapidocr.lite.native} pointing at the JNI
 *       library (or a directory containing it),</li>
 *   <li>extract both libraries from the JAR to a temp dir and load the JNI
 *       library by absolute path. The engine library is loaded first: the JVM
 *       does not add a library's own directory to the dependency search path,
 *       so pre-loading the engine by absolute path makes the JNI library's
 *       dependency resolve from the already-loaded module (this matters on
 *       Windows and Linux; macOS resolves it via {@code @loader_path}),</li>
 *   <li>fallback to {@code System.loadLibrary("rapidocrlite_jni")}.</li>
 * </ol>
 */
final class NativeLoader {

    /** platform -> { jni library file name, engine library file name }. */
    private static final String[][] PLATFORMS = {
        // key,             jni file,                  engine file
        {"win-x86_64",      "rapidocrlite_jni.dll",    "rapidocrlite.dll"},
        {"linux-x86_64",    "librapidocrlite_jni.so",  "librapidocrlite.so.5"},
        {"linux-aarch64",   "librapidocrlite_jni.so",  "librapidocrlite.so.5"},
        {"linux-loongarch64", "librapidocrlite_jni.so", "librapidocrlite.so.5"},
        {"macos-aarch64",   "librapidocrlite_jni.dylib", "librapidocrlite.5.dylib"},
        {"macos-x86_64",    "librapidocrlite_jni.dylib", "librapidocrlite.5.dylib"},
    };

    private NativeLoader() {
    }

    /** Detected platform key, e.g. {@code linux-x86_64}; null when unsupported. */
    static String platform() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        boolean x64 = arch.equals("amd64") || arch.equals("x86_64");
        boolean arm64 = arch.equals("aarch64") || arch.equals("arm64");
        // OpenJDK reports "loongarch64" (some vendor builds: "loong64").
        boolean la64 = arch.equals("loongarch64") || arch.equals("loong64");
        if (os.contains("win")) {
            return x64 ? "win-x86_64" : null;
        }
        if (os.contains("linux")) {
            if (x64) {
                return "linux-x86_64";
            }
            if (arm64) {
                return "linux-aarch64";
            }
            if (la64) {
                return "linux-loongarch64";
            }
            return null;
        }
        if (os.contains("mac") || os.contains("darwin")) {
            if (arm64) {
                return "macos-aarch64";
            }
            if (x64) {
                return "macos-x86_64";
            }
            return null;
        }
        return null;
    }

    private static String jniName(String platform) {
        for (String[] row : PLATFORMS) {
            if (row[0].equals(platform)) {
                return row[1];
            }
        }
        return null;
    }

    private static String engineName(String platform) {
        for (String[] row : PLATFORMS) {
            if (row[0].equals(platform)) {
                return row[2];
            }
        }
        return null;
    }

    private static String supportedList() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < PLATFORMS.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(PLATFORMS[i][0]);
        }
        return sb.toString();
    }

    static synchronized void load() {
        String platform = platform();
        if (platform == null) {
            throw new UnsatisfiedLinkError(
                    "RapidOCRLite has no native build for os.name="
                    + System.getProperty("os.name") + " os.arch="
                    + System.getProperty("os.arch")
                    + " (supported: " + supportedList() + "). "
                    + "Use -Drapidocr.lite.native=<path> to point at your own build.");
        }
        String jniName = jniName(platform);
        String engineName = engineName(platform);
        String resourceDir = "/native/" + platform + "/";
        Throwable last = null;

        // 1. explicit override
        String override = System.getProperty("rapidocr.lite.native");
        if (override != null && !override.isEmpty()) {
            Path p = Paths.get(override);
            if (Files.isDirectory(p)) {
                p = p.resolve(jniName);
            }
            try {
                System.load(p.toAbsolutePath().toString());
                return;
            } catch (Throwable t) {
                last = t;
            }
        }

        // 2. extract from the JAR
        try {
            Path dir = Files.createTempDirectory("rapidocrlite-native-");
            Path engine = dir.resolve(engineName);
            Path jni = dir.resolve(jniName);
            extract(resourceDir, engineName, engine);
            extract(resourceDir, jniName, jni);
            engine.toFile().deleteOnExit();
            jni.toFile().deleteOnExit();
            // Load the engine first: the JVM does not add the jni library's
            // own directory to the dependency search path, so loading the
            // engine by absolute path first makes the jni library's
            // dependency resolve from the already-loaded module.
            if (Files.isRegularFile(engine)) {
                System.load(engine.toAbsolutePath().toString());
            }
            System.load(jni.toAbsolutePath().toString());
            return;
        } catch (Throwable t) {
            last = t;
        }

        // 3. java.library.path
        try {
            System.loadLibrary("rapidocrlite_jni");
            return;
        } catch (Throwable t) {
            last = t;
        }

        throw new UnsatisfiedLinkError(
                "cannot load " + jniName + " for " + platform
                + " (last error: " + last + "). "
                + "Use -Drapidocr.lite.native=<path> to point at your own build.");
    }

    private static void extract(String resourceDir, String name, Path target)
            throws IOException {
        try (InputStream in = NativeLoader.class.getResourceAsStream(resourceDir + name)) {
            if (in == null) {
                if (name.startsWith("librapidocrlite_jni")
                        || name.equals("rapidocrlite_jni.dll")) {
                    throw new IOException("JAR is missing " + resourceDir + name);
                }
                return; // engine library may be provided next to the jni library
            }
            Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }
}
