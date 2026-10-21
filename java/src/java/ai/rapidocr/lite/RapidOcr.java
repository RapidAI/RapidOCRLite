package ai.rapidocr.lite;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import javax.imageio.ImageIO;

/**
 * Java binding for the RapidOCRLite C ABI (librapidocrlite).
 *
 * <p>Works on Windows, Linux and macOS: the jar ships the JNI bridge and the
 * matching engine library for each platform under {@code /native/<platform>/};
 * {@link NativeLoader} picks the current one.
 *
 * <pre>{@code
 * // PP-OCRv6: det.onnx + rec.onnx + dict.txt
 * try (RapidOcr ocr = RapidOcr.forPPOCR("models/PP-OCRv6_det_small.onnx",
 *                                       "models/PP-OCRv6_rec_small.onnx",
 *                                       "models/ppocrv6_dict.txt")) {
 *     for (TextLine line : ocr.recognize(Path.of("poster.jpg"))) {
 *         System.out.println(line.confidence() + " " + line.text());
 *     }
 * }
 *
 * // Indra lite: a directory containing meta.json
 * try (RapidOcr ocr = RapidOcr.forLite("models/lite")) {
 *     for (TextLine line : ocr.recognize(new URL("https://.../doc.jpg"))) {
 *         System.out.println(line.confidence() + " " + line.text());
 *     }
 * }
 * }</pre>
 *
 * Pixel contract: {@code recognize} converts images to packed RGB
 * (row-major, width*height*3), matching {@code rpocr_image.rgb}.
 */
public final class RapidOcr implements AutoCloseable {

    static {
        NativeLoader.load();
    }

    private long handle;
    private volatile boolean closed;

    private RapidOcr(long handle) {
        this.handle = handle;
    }

    /** PP-OCRv6 engine: pass det.onnx, rec.onnx and the dictionary. */
    public static RapidOcr forPPOCR(String detOnnx, String recOnnx, String dictTxt) {
        return create(nativeCreatePPOCR(detOnnx, recOnnx, dictTxt));
    }

    /** PP-OCRv6 engine with tunables. */
    public static RapidOcr forPPOCR(String detOnnx, String recOnnx, String dictTxt,
                                    OcrOptions options) {
        Objects.requireNonNull(options, "options");
        if (options.clsModel() != null) {
            return create(nativeCreatePPOCRCls(detOnnx, recOnnx, dictTxt, options.nativeArgs(),
                    options.clsModel()));
        }
        return create(nativeCreatePPOCR(detOnnx, recOnnx, dictTxt, options.nativeArgs()));
    }

    /** Indra lite engine: pass the model directory (contains meta.json). */
    public static RapidOcr forLite(String modelDir) {
        return create(nativeCreateLite(modelDir));
    }

    /** Indra lite engine with tunables. */
    public static RapidOcr forLite(String modelDir, OcrOptions options) {
        Objects.requireNonNull(options, "options");
        return create(nativeCreateLite(modelDir, options.nativeArgs()));
    }

    private static RapidOcr create(long h) {
        if (h == 0L) {
            throw new IllegalStateException("rpocr_create failed: " + lastError());
        }
        return new RapidOcr(h);
    }

    /** Recognize a BufferedImage. Thread-safe per engine handle. */
    public synchronized List<TextLine> recognize(BufferedImage image) {
        ensureOpen();
        int w = image.getWidth();
        int h = image.getHeight();
        if (w <= 0 || h <= 0) {
            return Collections.emptyList();
        }
        byte[] rgb = toPackedRgb(image);
        TextLine[] lines = nativeRecognize(handle, rgb, w, h);
        return lines == null ? Collections.<TextLine>emptyList() : Arrays.asList(lines);
    }

    /** Recognize an image file (JPEG/PNG/BMP/... via ImageIO). */
    public List<TextLine> recognize(Path imageFile) throws IOException {
        BufferedImage img = ImageIO.read(imageFile.toFile());
        if (img == null) {
            throw new IOException("unsupported or corrupt image: " + imageFile);
        }
        return recognize(img);
    }

    /**
     * Download (or open) an image from a URL and recognize it.
     * Works with http(s) URLs and {@code file:} URLs.
     */
    public List<TextLine> recognize(URL url) throws IOException {
        try (InputStream in = url.openStream()) {
            BufferedImage img = ImageIO.read(in);
            if (img == null) {
                throw new IOException("unsupported or corrupt image: " + url);
            }
            return recognize(img);
        }
    }

    /**
     * Recognize an uncompressed PPM (P6) through the SDK's own
     * {@code rpocr_load_ppm}. Useful for byte-exact regression tests.
     */
    public synchronized List<TextLine> recognizePpm(String ppmPath) {
        ensureOpen();
        TextLine[] lines = nativeRecognizePpm(handle, ppmPath);
        return lines == null ? Collections.<TextLine>emptyList() : Arrays.asList(lines);
    }

    /**
     * Recognize several images in one call (rpocr_recognize_batch).
     * The result list has one entry per input image, in input order.
     */
    public synchronized List<List<TextLine>> recognizeBatch(List<? extends BufferedImage> images) {
        ensureOpen();
        if (images == null || images.isEmpty()) {
            return Collections.emptyList();
        }
        int n = images.size();
        byte[][] rgb = new byte[n][];
        int[] ws = new int[n];
        int[] hs = new int[n];
        for (int i = 0; i < n; i++) {
            BufferedImage img = images.get(i);
            ws[i] = img.getWidth();
            hs[i] = img.getHeight();
            rgb[i] = toPackedRgb(img);
        }
        TextLine[][] out = nativeRecognizeBatch(handle, rgb, ws, hs);
        if (out == null) {
            return Collections.emptyList();
        }
        List<List<TextLine>> result = new ArrayList<List<TextLine>>(n);
        for (TextLine[] page : out) {
            result.add(page == null
                    ? Collections.<TextLine>emptyList()
                    : Arrays.asList(page));
        }
        return result;
    }

    /** Last engine-side error message (thread-local in the SDK). */
    public static String lastError() {
        String e = nativeLastError();
        return e == null ? "" : e;
    }

    /** CPU capabilities of the loaded engine binary. */
    public static CpuInfo cpuInfo() {
        return nativeCpuInfo();
    }

    public synchronized boolean isClosed() {
        return closed;
    }

    /** Frees the engine handle. Idempotent. */
    @Override
    public synchronized void close() {
        if (!closed) {
            closed = true;
            long h = handle;
            handle = 0L;
            if (h != 0L) {
                nativeDestroy(h);
            }
        }
    }

    private void ensureOpen() {
        if (closed || handle == 0L) {
            throw new IllegalStateException("engine is closed");
        }
    }

    // ------------------------------------------------------------------
    // image conversion
    // ------------------------------------------------------------------

    /**
     * Converts any BufferedImage to packed RGB. Fast path for the common
     * TYPE_3BYTE_BGR / TYPE_INT_ARGB layouts; transparent pixels are
     * composited over white so OCR sees text, not alpha.
     */
    private static byte[] toPackedRgb(BufferedImage img) {
        final int w = img.getWidth();
        final int h = img.getHeight();
        final byte[] out = new byte[w * h * 3];
        final boolean hasAlpha = img.getColorModel().hasAlpha();

        if (img.getType() == BufferedImage.TYPE_3BYTE_BGR) {
            byte[] px = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
            for (int src = 0, dst = 0; dst < out.length; src += 3, dst += 3) {
                out[dst] = px[src + 2];     // B -> R
                out[dst + 1] = px[src + 1]; // G
                out[dst + 2] = px[src];     // R -> B
            }
            return out;
        }

        int i = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int p = img.getRGB(x, y);
                int r = (p >> 16) & 0xff;
                int g = (p >> 8) & 0xff;
                int b = p & 0xff;
                if (hasAlpha) {
                    int a = (p >>> 24) & 0xff;
                    if (a != 0xff) {
                        r = (r * a + 255 * (255 - a)) / 255;
                        g = (g * a + 255 * (255 - a)) / 255;
                        b = (b * a + 255 * (255 - a)) / 255;
                    }
                }
                out[i++] = (byte) r;
                out[i++] = (byte) g;
                out[i++] = (byte) b;
            }
        }
        return out;
    }

    // ------------------------------------------------------------------
    // JNI
    // ------------------------------------------------------------------

    private static native long nativeCreatePPOCR(String det, String rec, String dict);

    private static native long nativeCreatePPOCR(String det, String rec, String dict,
                                                 double[] options);

    private static native long nativeCreatePPOCRCls(String det, String rec, String dict,
                                                    double[] options, String cls);

    private static native long nativeCreateLite(String dir);

    private static native long nativeCreateLite(String dir, double[] options);

    private static native void nativeDestroy(long handle);

    private static native TextLine[] nativeRecognize(long handle, byte[] rgb,
                                                     int width, int height);

    private static native TextLine[] nativeRecognizePpm(long handle, String path);

    private static native TextLine[][] nativeRecognizeBatch(long handle, byte[][] images,
                                                            int[] widths, int[] heights);

    private static native String nativeLastError();

    private static native CpuInfo nativeCpuInfo();
}
