# RapidOCRLite vs rapidocr 3.10.0 CPU speed — progress log
Target: >=20% faster than 3.10.0 (ORT 1.31) on det-only / rec-only / full, PP-OCRv6 small, x86 (AVX-512 + PPOCR_FORCE_ISA=avx2) and M4; results unchanged.

## Pushed (RapidOCRLite main, author znsoftm noreply)
- 20dedea FTZ/DAZ during inference.
- cbee092 im2col+GEMM for rec-height 3x3 convs (x86 rec stem 4.1->1.4 ms 1T); bounded yield-spin pool (PPOCR_PARALLEL_SPIN_US=200);
  CTC vocab split on AVX2/NEON (AVX2 rec 16.2->12.2 ms, M4 rec 30->15 ms); Apple Accelerate cblas_sgemm for pointwise /
  expand-GELU-project / CTC head / im2col convs (PPOCR_DISABLE_ACCELERATE, -DPPOCR_USE_ACCELERATE=OFF). CI green (sdk, vulkan, metal).
- d576ebb Apple: tiled expand GEMMs over pool, parallel CTC row reduction.
Verification each step: SWHL det 212 pages full-mode texts+boxes identical to previous main (x86 AVX-512, AVX2, M4);
rec 310 lines identical (EM 0.8355 x86 / 0.8323 M4 PNG-loaded); pytest 60 pass; tiny refcheck 0 mismatch avx512/avx2/scalar.

## Numbers (s/img, both hosts heavily loaded by other users' jobs: x86 load 6-13, M4 load 10-15 -> noisy)
x86 (interleaved, d576ebb):  3.x det .29-.35 rec .0072-.0102 full .39-.49 | Lite AVX-512 det .150-.164 rec .0091-.0116 full .221-.228 | AVX2 det .171-.185 rec .0104-.0123 full .231-.254
M4 (5 interleaved reps):    3.x det .145-.163 rec .0070-.0086 full .360-.395 | Lite det .143-.175 rec .0127-.0175 full .193-.232
Quiet M4 earlier (20 min after reboot): 3.x det .113 rec .0058 | Lite det .100 rec .0102.

## Status vs target
- full: met everywhere (x86 ~2x, M4 ~1.7x).
- det: x86 met (~2x); M4 ~parity/ -11% (NOT met).
- rec-only: NOT met. x86 ~parity; M4 ~1.7x slower. ORT on M4 uses SME/KleidiAI (Conv 384->768 @240px 0.35 ms 1T = ~400 GFLOPS).

## Next steps
1. M4 rec: per-op multi-thread overhead (expand blocks 0.35-0.5 ms each x12 with Accelerate small GEMMs); try VECLIB_MAXIMUM_THREADS / single big GEMM per block, P-core-only thread default (sysctl hw.perflevel0.logicalcpu), MaxPoolConcatConv/Conv.1 via Accelerate.
2. x86 rec: ExpandGeluProjectAdd kernel (109 vs ORT 170 GFLOPS); executor overhead (~1-2 ms between op sum and rec_run).
3. M4 det: depthwise 5x5 (Conv.78 6.5 ms), ConvTranspose (10 ms), Concat.2 conv.
4. Final quiet-host 5-rep medians, README/README-CN tables, CI links.
Tools: bench/perf.py, final_x86.sh, macsync.sh, medprof.py, Mac ~/work/lite-perf (run.sh 3x|src|c1, final.sh, dump_mac.py+cmp_dump.py).

## 16:55 M4 thread experiments (load ~4-8)
3.x rec .0072-.0073 | Lite default .0126-.0129 | PPOCR_BENCH_THREADS=4 .0109-.0115 | VECLIB_MAXIMUM_THREADS=1 no change.
=> P-core-only default would give ~10%; gap is structural (ORT SME kernels scale 12 ms 1T -> 5.3 ms 4T; Lite 15.5 -> ~9-11).
README tables NOT yet updated (targets unmet, hosts noisy).

## 17:25 final (b9f3866 README pushed)
x86 5-rep medians: 3.x det .3215 rec .0078 full .3909 | AVX-512 .1515/.0097/.2209 | AVX2 .1668/.0109/.2503
M4 3-rep medians:  3.x det .1465 rec .0072 full .3604 | Lite .1425/.0126/.1931
CI: cbee092, d576ebb sdk/vulkan/metal green (wheels superseded/in progress).
Tried and rejected: pure-pause spin (worse under load), 4-thread default on M4 (rec +10% but det -15%), VECLIB_MAXIMUM_THREADS=1 (no change).
Unmet: rec-only everywhere, M4 det (+3% only). Needs: SME kernels on M4 (or fewer, larger Accelerate calls), x86 GEMM micro-kernel ~1.5x, ~2 ms executor overhead in rec_run vs op sum.

## 17:40 round 3
- KleidiAI v1.30.0 (ORT pin, sha 36aebdd) SME2 f32 GEMM benchmarked on M4 vs Accelerate: same or slower
  (768x240x384: SME 0.20 ms vs Accel 0.11; 40x18710x384: 4.7 vs 1.3 ms), bit-identical. => GEMM was not the bottleneck; not integrated.
- Real M4 bottleneck: scalar std::erf GELU on NEON + scalar exp in CTC + small-tile GEMMs + E-core stragglers.
- Pushed: NEON vector exact GELU, vvexpf CTC, single GEMM per small block, P-core default threads. M4 rec 11.1 -> 8.3 ms (3.x 7.2).
- pushed CTC chunked head (M4 rec 7.7 vs 3.x 7.15). NOTE: -inf is poison under -ffast-math; use lowest().
- pushed ConvTranspose GEMM + NEON dw 16-wide: M4 det ops 65->50 ms. Next: Concat.2 3x3 s1 via 9 shifted GEMMs over padded input (no im2col).
- tried 9-tap shifted strided GEMM (no im2col) on Apple: slower (Concat.2 6.0->6.9, stem 3.0->5.6), reverted.

## 18:19 M4 at c9af2fa (3 interleaved runs, load 6-7 from other users; s/img or s/line)
| mode | 3.10.0 median | Lite median | speedup |
|---|---|---|---|
| det | 0.1140 | 0.0926 | 1.23x |
| rec | 0.00585 | 0.00534 | 1.10x (target 1.2x not met) |
| full | 0.3145 | 0.1089 | 2.9x |
CI c9af2fa: sdk/metal/vulkan green; wheels 3291253 failed on Windows with HTTP 503 during a download (infra), c9af2fa wheels pending.
x86 GEMM (MLAS-class) and executor-overhead items not started this round.
