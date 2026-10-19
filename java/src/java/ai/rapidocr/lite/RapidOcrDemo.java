package ai.rapidocr.lite;

import java.io.IOException;
import java.io.PrintStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * Runnable demo packaged as the jar's Main-Class.
 *
 * <pre>
 * java -jar rapidocrlite-java-5.0.0a10.jar [--lite <dir> | --ppocr <det> <rec> <dict>]
 *                                          [--cpu] [image-or-url ...]
 * </pre>
 *
 * Without image arguments it recognizes one built-in sample URL
 * (requires network); with arguments each path or http(s) URL is recognized.
 */
public final class RapidOcrDemo {

    private static final PrintStream OUT = System.out;

    private RapidOcrDemo() {
    }

    public static void main(String[] args) throws Exception {
        String liteDir = null;
        String det = null;
        String rec = null;
        String dict = null;
        boolean cpuInfo = false;
        java.util.List<String> images = new java.util.ArrayList<String>();

        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if ("--lite".equals(a) && i + 1 < args.length) {
                liteDir = args[++i];
            } else if ("--ppocr".equals(a) && i + 3 < args.length) {
                det = args[++i];
                rec = args[++i];
                dict = args[++i];
            } else if ("--cpu".equals(a)) {
                cpuInfo = true;
            } else if ("--help".equals(a) || "-h".equals(a)) {
                usage();
                return;
            } else {
                images.add(a);
            }
        }

        if (cpuInfo) {
            OUT.println("engine cpu info: " + RapidOcr.cpuInfo());
        }
        if (liteDir == null && det == null && images.isEmpty() && !cpuInfo) {
            usage();
            return;
        }

        RapidOcr ocr;
        if (liteDir != null) {
            ocr = RapidOcr.forLite(liteDir);
        } else if (det != null) {
            ocr = RapidOcr.forPPOCR(det, rec, dict);
        } else {
            usage();
            return;
        }
        OUT.println("engine ready: " + ocr.getClass().getSimpleName()
                + " cpu=" + RapidOcr.cpuInfo().activeIsa());

        if (images.isEmpty()) {
            images.add("https://raw.githubusercontent.com/RapidAI/RapidOCR/main/"
                    + "python/tests/test_files/ch_en_num.jpg");
        }

        int failures = 0;
        for (String spec : images) {
            try {
                OUT.println("== " + spec);
                long t0 = System.nanoTime();
                List<TextLine> lines = recognize(ocr, spec);
                long ms = (System.nanoTime() - t0) / 1_000_000;
                for (TextLine line : lines) {
                    float[] tl = line.box()[0];
                    OUT.printf("  [%5.3f] (%6.1f,%6.1f) %s%n",
                            line.confidence(), tl[0], tl[1], line.text());
                }
                OUT.println("  -> " + lines.size() + " line(s) in " + ms + " ms");
            } catch (Exception e) {
                failures++;
                OUT.println("  ERROR: " + e);
            }
        }
        ocr.close();
        if (failures > 0) {
            System.exit(1);
        }
    }

    private static List<TextLine> recognize(RapidOcr ocr, String spec) throws IOException {
        if (spec.startsWith("http://") || spec.startsWith("https://")) {
            return ocr.recognize(new URL(spec));
        }
        if (spec.toLowerCase().endsWith(".ppm")) {
            return ocr.recognizePpm(spec);
        }
        Path p = Paths.get(spec);
        if (!Files.isRegularFile(p)) {
            throw new IOException("no such file: " + spec);
        }
        return ocr.recognize(p);
    }

    private static void usage() {
        OUT.println("usage: java -jar rapidocrlite-java.jar "
                + "[--lite <modelDir> | --ppocr <det.onnx> <rec.onnx> <dict.txt>] "
                + "[--cpu] [image-or-url ...]");
        OUT.println("  --lite <dir>    Indra lite model directory (contains meta.json)");
        OUT.println("  --ppocr ...     PP-OCRv6 det/rec onnx + dictionary");
        OUT.println("  --cpu           print engine CPU info and exit");
        OUT.println("  images          file paths, http(s) URLs, or .ppm files");
        OUT.println("  no images       recognizes one built-in sample URL (needs network)");
    }
}
