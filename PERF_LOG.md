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
- 9bf4a41 AVX-512 expand weight-stationary 8x32 blocking: single-thread Conv.49 2.95->2.3 ms, bit-identical. AVX-512 12-row variant slower; AVX2 6x16 port neutral/slower (reverted). Executor overhead investigation: the '2 ms' gap was profiler printing; real RunCtcTop1 wall == sum of ops (no executor overhead to cut).

## 19:10 status (main b4cebec)
- 9bf4a41 AVX-512 expand weight-stationary blocking (bit-identical); b4cebec dynamic pool chunks (identical outputs x86+M4).
- Executor overhead: measured RunCtcTop1 wall == sum of ops; earlier "2 ms gap" was profiler printing. Nothing to cut.
- Tried+reverted: AVX-512 12-row tile (spills), AVX2 6x16 blocked port (neutral), row-split expand parallelism (helps Conv.49 only, noise-level net).
- M4 @ b4cebec (2 runs, load 6-7): 3.x det .114 rec .00584 full .314 | Lite det .082 (1.40x) rec .00544 (1.07x) full .107 (2.9x)
- 19:45 x86 final_x86.sh @ b4cebec, NOT quiet (load 7-14 on 8 vCPU), medians of 3-4: 3.x det .388 rec .0096 full .521 | AVX-512 det .158 rec .0093 full .222 | AVX2 det .209 rec .0122 full .265. Paired-run rec AVX-512 vs 3.x: 10.2/11.2, 8.7/7.9, 9.9/12.7, 8.1/8.0 ms. README not updated (no quiet host).
- 19:40 MLAS-style packed weights [rows/8][K][8] for AVX-512 blocked expand: Conv.49 ST 2.30 -> 2.24 ms (noise, load 12) -> not kept. Broadcast loads were already L1-resident.
- 20:05 M4 QUIET 5-run medians @ b4cebec (idle load 1.6): 3.x det .1145 rec .00587 full .3142 | Lite det .0817 (1.40x) rec .00543 (1.08x) full .1075 (2.92x)
- im2col conv scratch made thread_local (was fresh zeroed vector per call, up to 0.9 MB): M4 MaxPool.0_concat_conv 0.39 -> 0.20 ms; rec 5.43 -> 5.37 ms, det .0817 -> .0810. Identical outputs x86 (512/AVX2) + M4.
- M4 thread/chunk/spin sweeps: all ~5.4 ms (4 threads best). CTC head 0.57 ms is bandwidth on 28 MB fp32 weights.
- 20:17 README/README-CN M4 rows updated (ff59346). x86 still loaded (load 6-10) -> x86 rows not re-measured. Open: x86 rec-only (AVX2 esp.), M4 rec needs ~10% more.
- 22:15 RELEASED 5.0.0a7 (473d185, tag v5.0.0a7 annotated): PyPI 25 wheels + sdist; GH prerelease with wheels, 7 SDK archives, SHA256SUMS. Tag-push SDK attach raced release creation ('release not found'); fixed via documented workflow_dispatch release_tag=v5.0.0a7 (success).
- 21:55 b74b8a9: build-sdk creates the release from the existing tag if missing (--verify-tag, retry) before SDK upload.
- AVX2 single-thread rec profile (total 45.9 ms vs AVX-512 ~25): Conv.51 5.1, Conv.49 5.0 (340 MFLOP -> ~68 GFLOPS), CTC 4.8 (~120 GFLOPS, packed kernel near peak), mid expand blocks ~2.4-2.6 each, Conv.45 1.9.
  Expand kernels at ~60% of AVX2 FMA peak; previous 6x16 weight-blocked port was neutral, so weight streaming is not the limiter. Note 3.x on this host runs MLAS AVX-512 regardless of PPOCR_FORCE_ISA, so forced-AVX2 Lite competes with half-width SIMD.
- x86 host load swings 3-14; 3.x rec itself ranged 8-20 ms/line in the same hour -> multi-thread comparisons not reproducible today.
- 22:30 pointwise 1x1: AVX2 4x24 (12 acc) and AVX-512 8x32 (16 acc) register tiles, bit-identical. Single-thread rec: AVX2 Conv.45 1.90->1.17, Conv.18 0.90->0.55 ms; AVX-512 Conv.45 1.07->0.60, Conv.18 0.58->0.29 ms. 212/310 identical both ISAs.

- x86 GEMM-accumulate tiles (im2col stem / Conv.1 / Conv.2): AVX-512 4x64 block (`PPOCR_DISABLE_AVX512_GEMM_ACC4`), AVX2 6x16 block (`PPOCR_DISABLE_AVX2_GEMM_ACC6`). Single-thread rec profile, loaded host: AVX-512 MaxPool.0_concat_conv 2.1 -> ~1.65-1.85 ms; AVX2 ~2.2 -> ~1.95-2.09 ms. Bit-identical: 0/212 full text diffs, 0 box diff, 0/310 rec diffs on both ISAs.

- Rejected: AVX2 expand 6x16 tile in Avx2ExpandProjectSpatial16 (expand + project). Single-thread rec went 45.4/44.2 -> 47.9/46.7 ms, slower (register pressure). Matches the earlier 6x16 rejection; the 4x16 tile stays.
- x86 rec-only per line at bc0f518, multi-thread, loaded host (load 6-8, not README quality), 3 interleaved rounds: 3.10.0 ORT 14.7/14.5/15.1 ms; Lite AVX-512 13.7/13.7/17.1 ms; Lite forced AVX2 17.0/18.8/21.4 ms. Rec-only is still short of the 20% target on x86, and forced AVX2 is slower than ORT (which uses MLAS AVX-512 on this host).
- Rejected: AVX2 expand 4x24-lane spatial tile (12 accumulators, 16 ymm live). Single-thread rec 45.1/47.4 -> 69.9/52.6 ms, expand blocks 5.2 -> 6.5-8.8 ms (register spills plus more scalar tail work at SVTR widths). Not repeating the per-hidden-block packed weights: the identical [rows/8][K][8] packing was already measured as neutral/slower. The broadcasts are memory-operand vbroadcastss, so packing only shortens the weight streams, and the weights are already L1/L2 resident. The x86 expand blocks are now limited by FMA/load ports at their current tile shape.

## 2026-10-15 x86 rec-only structural investigation (gcc Release build, host load 1-4)
- CI: 87c68bb is fully green (build wheels, build sdk, metal, lavapipe). Wheels runs on b764697/bc0f518/d4d610f were cancelled by newer pushes (concurrency).
- Per-line rec, 310 lines, same PP-OCRv6 rec small model for both, multi-thread: 3.10.0 ORT 7.0-7.7 ms (one 10.7 outlier); Lite main AVX-512 7.1-8.4 ms; Lite main forced AVX2 9.0-10.7 ms. Shipped 5.0.0a7 wheel: 8.0-10.1 ms (same run as ORT 6.8-9.0 ms).
- Single thread (ORT intra/inter=1 vs PPOCR_BENCH_THREADS=1): ORT 19.9/20.5 ms, Lite 25.8 ms. Lite scales better (3.5x on 8 cores vs ~2.8x), so the remaining gap is per-kernel speed, not threading.
- Op families at 48x320, single thread: Lite expand/GELU/project blocks 11.9 ms vs ORT 1x1 Conv+Gelu+Add of about 11 ms. Lite stem/other convs (Conv.0/1/2, MaxPool concat conv, Conv.45, dw+pw) 5.0 ms vs ORT about 2.8 ms (NCHWc). CTC head 2.3 vs 2.5 ms (Lite ahead).
- Rejected (bit-identical, measured neutral or slower): (a) 3-phase expand split (pack, then expand rows, then project rows, each parallel), neutral end-to-end; (b) MLAS-style output-channel-vector tiles with packed weights (16V oc lanes x P broadcast positions; V/P = 4/6, 2/12, 1/24), no faster than the spatial-lane kernel (1.89-2.69 vs 1.84-1.95 ms Conv.49); (c) the same with K-blocking (kb 64/128/256), slower. The inner loops are clean (no spills, 1 broadcast per 2 FMA), so the expand blocks are FMA-bound on this host.
- Batching lines is not available: use_det=False rec processes one image per call.
- Next: NCHWc-style direct conv for the stem convs (about 2 ms/line single thread gap), then GELU cost inside expand blocks.
- im2col run copies in SmallConvIm2colGemm (`PPOCR_DISABLE_IM2COL_RUNS`): instrumenting showed im2col took about as long as the GEMM in the stem convs. Valid columns are now copied as constant-step runs (memcpy for stride 1, strided loop for stride 2) instead of a per-element select. MaxPool.0_concat_conv at 48x320, 1 thread: 1.14/1.27 -> 0.96/0.99 ms. Bit-identical: 0/212 full, 0 px, 0/310 rec on AVX-512 and AVX2 (baselines in /workspace/bench/baselines).
- Rejected: AVX-512 output-channel-lane GEMM for im2col convs (A^T cached, 3x16 oc x 8 cols): stem convs not faster (MaxPool conv 1.11-1.18 vs 1.07-1.12 ms).
- AVX-512 rec 1x1 convs: 384->384 on a 240-position plane (Conv.45) fell under the plane>=256 cutoff to four-output tiles. Pointwise8 now also covers >=192 channels at plane >= 64 (`PPOCR_DISABLE_AVX512_POINTWISE8_REC`). It also packs activations as [tile][C][32] with a masked tail (`PPOCR_DISABLE_PW_PACK`), so the 384 strided rows stop defeating the prefetcher. Conv.45 at 48x320, 1 thread: 0.83-0.92 -> 0.47-0.50 ms (no pack: 0.54). Bit-identical: 0/212, 0 px, 0/310 on both ISAs.
- Per-line rec at 2d293b0, 310 lines, host load 3-4 (noisy), ms/line. Multi-thread: ORT 7.1-11.5; Lite AVX-512 dev 7.7-9.5 vs main-before 7.4-10.8; Lite AVX2 10.5-12.9. Single thread: ORT 21.0/21.2, Lite AVX-512 25.6/28.0. The stem/1x1 wins (about 0.6 ms/line single thread at 48x320) are below this host's noise end-to-end. Forced AVX2 is about 1.75x the AVX-512 time in the expand blocks (half vector width), so it cannot reach +20% over 3.x, which uses AVX-512 on this host.
- GELU: an output-preserving reduction needs the same op sequence. ErfPs512 is about 0.1 ms/line at 320 (per-op estimate), not worth the risk. Skipped.
