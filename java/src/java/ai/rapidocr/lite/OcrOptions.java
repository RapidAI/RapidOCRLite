package ai.rapidocr.lite;

/**
 * Tunables for engine creation. Every field is optional; NaN / 0 means
 * "use the SDK default", so the zero instance equals the SDK defaults.
 *
 * <p>Usage:
 * <pre>{@code
 * OcrOptions opts = OcrOptions.builder()
 *         .backend(Backend.CPU)
 *         .detLimitSideLen(960)
 *         .recParallelism(4)
 *         .build();
 * RapidOcr ocr = RapidOcr.forLite("models/lite", opts);
 * }</pre>
 */
public final class OcrOptions {

    public enum Backend {
        CPU,
        GPU,
        HYBRID;

        int nativeValue() {
            return ordinal();
        }
    }

    /** NaN means "SDK default". */
    private final float detThreshold;
    private final float detBoxThreshold;
    private final float detUnclipRatio;
    /** 0 means "SDK default". */
    private final int detLimitSideLen;
    /** 0 means "SDK default". */
    private final int recParallelism;
    private final Backend backend;
    /** Text-line orientation classifier (rapidocr main Global.use_cls), on by default. */
    private final boolean useCls;
    /** Classifier ONNX (ch_ppocr_mobile_v2.0_cls_mobile.onnx); null: none (or PPOCR_CLS_MODEL). */
    private final String clsModel;
    /** NaN means "SDK default" (0.9). */
    private final float clsThresh;

    private OcrOptions(Builder b) {
        this.useCls = b.useCls;
        this.clsModel = b.clsModel;
        this.clsThresh = b.clsThresh;
        this.detThreshold = b.detThreshold;
        this.detBoxThreshold = b.detBoxThreshold;
        this.detUnclipRatio = b.detUnclipRatio;
        this.detLimitSideLen = b.detLimitSideLen;
        this.recParallelism = b.recParallelism;
        this.backend = b.backend;
    }

    public static Builder builder() {
        return new Builder();
    }

    public Backend backend() {
        return backend;
    }

    public int detLimitSideLen() {
        return detLimitSideLen;
    }

    public int recParallelism() {
        return recParallelism;
    }

    public float detThreshold() {
        return detThreshold;
    }

    public float detBoxThreshold() {
        return detBoxThreshold;
    }

    public float detUnclipRatio() {
        return detUnclipRatio;
    }

    public boolean useCls() {
        return useCls;
    }

    public String clsModel() {
        return clsModel;
    }

    public float clsThresh() {
        return clsThresh;
    }

    /**
     * Packed as a double[] for the JNI layer; NaN entries keep SDK defaults.
     * Layout: backend, detLimitSideLen, recParallelism,
     * detThreshold, detBoxThreshold, detUnclipRatio, useCls, clsThresh.
     */
    double[] nativeArgs() {
        return new double[]{
                backend.nativeValue(),
                detLimitSideLen,
                recParallelism,
                detThreshold,
                detBoxThreshold,
                detUnclipRatio,
                useCls ? 1 : 0,
                clsThresh,
        };
    }

    public static final class Builder {
        private float detThreshold = Float.NaN;
        private float detBoxThreshold = Float.NaN;
        private float detUnclipRatio = Float.NaN;
        private int detLimitSideLen;
        private int recParallelism;
        private Backend backend = Backend.CPU;
        private boolean useCls = true;
        private String clsModel;
        private float clsThresh = Float.NaN;

        private Builder() {
        }

        /** Detector binarization threshold. */
        public Builder detThreshold(float v) {
            this.detThreshold = v;
            return this;
        }

        /** Detector box score threshold. */
        public Builder detBoxThreshold(float v) {
            this.detBoxThreshold = v;
            return this;
        }

        /** Detector unclip ratio. */
        public Builder detUnclipRatio(float v) {
            this.detUnclipRatio = v;
            return this;
        }

        /** Long-side cap for detection; 0 keeps the SDK default. */
        public Builder detLimitSideLen(int v) {
            this.detLimitSideLen = v;
            return this;
        }

        /** Recognition parallelism; 0 keeps the SDK default. */
        public Builder recParallelism(int v) {
            this.recParallelism = v;
            return this;
        }

        /**
         * Text-line orientation classification (0/180) between detection and
         * recognition, on by default as in rapidocr main. It runs only with a
         * classifier model ({@link #clsModel(String)} or PPOCR_CLS_MODEL); the
         * lite model ignores it. PPOCR_USE_CLS=0/1 overrides it.
         */
        public Builder useCls(boolean v) {
            this.useCls = v;
            return this;
        }

        /** Path of ch_ppocr_mobile_v2.0_cls_mobile.onnx (PP-OCRv4 Cls in rapidocr main). */
        public Builder clsModel(String path) {
            this.clsModel = path;
            return this;
        }

        /** Rotate a line when "180" scores above this; NaN keeps 0.9. */
        public Builder clsThresh(float v) {
            this.clsThresh = v;
            return this;
        }

        public Builder backend(Backend v) {
            this.backend = v;
            return this;
        }

        public OcrOptions build() {
            return new OcrOptions(this);
        }
    }
}
