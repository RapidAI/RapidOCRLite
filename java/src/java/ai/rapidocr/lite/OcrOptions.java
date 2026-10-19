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

    private OcrOptions(Builder b) {
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

    /**
     * Packed as a double[] for the JNI layer; NaN entries keep SDK defaults.
     * Layout: backend, detLimitSideLen, recParallelism,
     * detThreshold, detBoxThreshold, detUnclipRatio.
     */
    double[] nativeArgs() {
        return new double[]{
                backend.nativeValue(),
                detLimitSideLen,
                recParallelism,
                detThreshold,
                detBoxThreshold,
                detUnclipRatio,
        };
    }

    public static final class Builder {
        private float detThreshold = Float.NaN;
        private float detBoxThreshold = Float.NaN;
        private float detUnclipRatio = Float.NaN;
        private int detLimitSideLen;
        private int recParallelism;
        private Backend backend = Backend.CPU;

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

        public Builder backend(Backend v) {
            this.backend = v;
            return this;
        }

        public OcrOptions build() {
            return new OcrOptions(this);
        }
    }
}
