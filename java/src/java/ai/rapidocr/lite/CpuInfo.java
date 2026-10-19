package ai.rapidocr.lite;

/**
 * CPU capabilities reported by the SDK (rpocr_query_cpu_info).
 */
public final class CpuInfo {

    private final boolean avx2Compiled;
    private final boolean avx512Compiled;
    private final boolean neonCompiled;
    private final int threads;
    private final String activeIsa;

    CpuInfo(boolean avx2Compiled, boolean avx512Compiled, boolean neonCompiled,
            int threads, String activeIsa) {
        this.avx2Compiled = avx2Compiled;
        this.avx512Compiled = avx512Compiled;
        this.neonCompiled = neonCompiled;
        this.threads = threads;
        this.activeIsa = activeIsa;
    }

    /** Whether the binary contains AVX2 kernels. */
    public boolean avx2Compiled() {
        return avx2Compiled;
    }

    /** Whether the binary contains AVX-512 kernels. */
    public boolean avx512Compiled() {
        return avx512Compiled;
    }

    /** Whether the binary contains NEON kernels. */
    public boolean neonCompiled() {
        return neonCompiled;
    }

    /** Thread count the engine will use (0 = auto). */
    public int threads() {
        return threads;
    }

    /** "avx512", "avx2", "neon", "lasx", "lsx" or "scalar". */
    public String activeIsa() {
        return activeIsa;
    }

    @Override
    public String toString() {
        return "CpuInfo{isa=" + activeIsa + ", threads=" + threads
                + ", avx2_compiled=" + avx2Compiled
                + ", avx512_compiled=" + avx512Compiled
                + ", neon_compiled=" + neonCompiled + "}";
    }
}
