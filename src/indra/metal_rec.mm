#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_rec.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

namespace rpocr::indra {
namespace {
const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct GP { int rows; int K; int N; int act; int resid; int gated; };
inline float hsw(float x) { return x * clamp(x * (1.0f / 6.0f) + 0.5f, 0.0f, 1.0f); }
// Y[r][n] (+)= act(bias[n] + sum_k A[r][k] (* gate[line(r)][k]) * Wt[k][n]); 64x64 tile, 256 threads, 4x4 per thread
kernel void gemm(device const float* A [[buffer(0)]], device const float* Wt [[buffer(1)]],
                 device const float* bias [[buffer(2)]], device float* Y [[buffer(3)]],
                 constant GP& p [[buffer(4)]], device const float* gate [[buffer(5)]],
                 device const int* rline [[buffer(6)]],
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup TG_T As[16][65];
  threadgroup TG_T4 Ws[16][16];
  int r0 = int(tg.y) * 64, n0 = int(tg.x) * 64;
  int tr = int(tid) / 16, tc = int(tid) % 16;
  float4 acc[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
  for (int k0 = 0; k0 < p.K; k0 += 16) {
    for (int i = int(tid); i < 1024; i += 256) {
      int r = i / 16, k = i % 16, gr = r0 + r, gk = k0 + k;
      float v = 0.0f;
      if (gr < p.rows && gk < p.K) {
        v = A[gr * p.K + gk];
        if (p.gated) v *= gate[rline[gr] * p.K + gk];
      }
      As[k][r] = TG_T(v);
    }
    {
      int k = int(tid) / 16, c = int(tid) % 16, gk = k0 + k, gn = n0 + c * 4;
      float4 w = float4(0.0f);
      if (gk < p.K && gn + 3 < p.N) w = *(device const float4*)(Wt + gk * p.N + gn);
      Ws[k][c] = TG_T4(w);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 0; k < 16; ++k) {
      float4 w = float4(Ws[k][tc]);
      for (int i = 0; i < 4; ++i) acc[i] = fma(float4(As[k][tr * 4 + i]), w, acc[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int i = 0; i < 4; ++i) {
    int r = r0 + tr * 4 + i;
    if (r >= p.rows) continue;
    for (int j = 0; j < 4; ++j) {
      int n = n0 + tc * 4 + j;
      if (n >= p.N) continue;
      float x = acc[i][j] + bias[n];
      if (p.act == 1) x = hsw(x);
      if (p.resid) Y[r * p.N + n] += x; else Y[r * p.N + n] = x;
    }
  }
}
// simdgroup-matrix version: 64x64 tile, 4 simdgroups (128 threads) each 32x32 = 4x4 8x8 blocks; K % 16 == 0
kernel void gemm_sg(device const float* A [[buffer(0)]], device const float* Wt [[buffer(1)]],
                    device const float* bias [[buffer(2)]], device float* Y [[buffer(3)]],
                    constant GP& p [[buffer(4)]], device const float* gate [[buffer(5)]],
                    device const int* rline [[buffer(6)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint sgi [[simdgroup_index_in_threadgroup]]) {
  threadgroup float As[64 * 16];
  threadgroup float Ws[16 * 64];
  threadgroup float Cs[64 * 64];
  int r0 = int(tg.y) * 64, n0 = int(tg.x) * 64;
  int sr = int(sgi / 2) * 32, sn = int(sgi % 2) * 32;
  simdgroup_float8x8 acc[4][4];
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (int k0 = 0; k0 < p.K; k0 += 16) {
    for (int i = int(tid); i < 1024; i += 128) {
      int r = i / 16, k = i % 16, gr = r0 + r, gk = k0 + k;
      float v = 0.0f;
      if (gr < p.rows) {
        v = A[gr * p.K + gk];
        if (p.gated) v *= gate[rline[gr] * p.K + gk];
      }
      As[i] = v;
    }
    for (int i = int(tid); i < 1024; i += 128) {
      int k = i / 64, n = i % 64, gn = n0 + n;
      Ws[i] = gn < p.N ? Wt[(k0 + k) * p.N + gn] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < 16; kk += 8) {
      simdgroup_float8x8 a[4], b[4];
      for (int i = 0; i < 4; ++i) simdgroup_load(a[i], As + (sr + i * 8) * 16 + kk, 16);
      for (int j = 0; j < 4; ++j) simdgroup_load(b[j], Ws + kk * 64 + sn + j * 8, 64);
      for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) simdgroup_store(acc[i][j], Cs + (sr + i * 8) * 64 + sn + j * 8, 64);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(tid); i < 4096; i += 128) {
    int r = r0 + i / 64, n = n0 + i % 64;
    if (r >= p.rows || n >= p.N) continue;
    float x = Cs[i] + bias[n];
    if (p.act == 1) x = hsw(x);
    if (p.resid) Y[r * p.N + n] += x; else Y[r * p.N + n] = x;
  }
}
struct DP { int rows; int C; };
// depthwise k=9 along each line (zero outside the line)
kernel void dw9(device const float* h [[buffer(0)]], device const float* wT [[buffer(1)]],
                device const float* b [[buffer(2)]], device float* t [[buffer(3)]], constant DP& p [[buffer(4)]],
                device const int* rline [[buffer(5)]], device const int* starts [[buffer(6)]],
                uint2 gid [[thread_position_in_grid]]) {
  int c = int(gid.x), r = int(gid.y);
  if (c >= p.C || r >= p.rows) return;
  int l = rline[r], s = starts[l], e = starts[l + 1];
  float acc = b[c];
  for (int j = 0; j < 9; ++j) {
    int q = r + j - 4;
    if (q >= s && q < e) acc = fma(h[q * p.C + c], wT[j * p.C + c], acc);
  }
  t[r * p.C + c] = acc;
}
struct SP { int E; int S; };
// per line: mean over rows -> fc1 relu -> fc2 sigmoid -> gate[line][E]; one threadgroup per line
kernel void se(device const float* e [[buffer(0)]], device const int* starts [[buffer(1)]],
               device const float* f1 [[buffer(2)]], device const float* f1b [[buffer(3)]],
               device const float* f2 [[buffer(4)]], device const float* f2b [[buffer(5)]],
               device float* gate [[buffer(6)]], constant SP& p [[buffer(7)]],
               uint l [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
               uint nt [[threads_per_threadgroup]]) {
  threadgroup float s[512], s1[256];
  int r0 = starts[l], r1 = starts[l + 1];
  float inv = 1.0f / max(1.0f, float(r1 - r0));
  for (int c = int(tid); c < p.E; c += int(nt)) {
    float a = 0.0f;
    for (int r = r0; r < r1; ++r) a += e[r * p.E + c];
    s[c] = a * inv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int o = int(tid); o < p.S; o += int(nt)) {
    float a = f1b[o];
    for (int c = 0; c < p.E; ++c) a = fma(f1[o * p.E + c], s[c], a);
    s1[o] = max(a, 0.0f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int o = int(tid); o < p.E; o += int(nt)) {
    float a = f2b[o];
    for (int c = 0; c < p.S; ++c) a = fma(f2[o * p.S + c], s1[c], a);
    gate[l * p.E + o] = 1.0f / (1.0f + exp(-a));
  }
}
#define RB 32
#define C4 32
struct HP { int rows; int V; int ldh4; int pad; };
kernel void head(device const float4* h [[buffer(0)]], device const float4* W [[buffer(1)]],
                 device const float* bias [[buffer(2)]], device int* ids [[buffer(3)]],
                 device float* conf [[buffer(4)]], constant HP& p [[buffer(5)]],
                 uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                 uint nt [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup float4 hs[RB * C4];
  threadgroup float tm[32 * RB], ts[32 * RB];
  threadgroup int ta[32 * RB];
  int r0 = int(tg) * RB;
  int nr = min(RB, p.rows - r0);
  for (uint i = tid; i < uint(RB * C4); i += nt) {
    int r = int(i) / C4, k = int(i) % C4;
    hs[i] = r < nr ? h[(r0 + r) * p.ldh4 + k] : float4(0.0f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m[RB], s[RB]; int a[RB];
  for (int r = 0; r < RB; ++r) { m[r] = -INFINITY; s[r] = 0.0f; a[r] = 0; }
  for (int v = int(tid); v < p.V; v += int(nt)) {
    float acc[RB];
    float bv = bias[v];
    for (int r = 0; r < RB; ++r) acc[r] = bv;
    device const float4* wr = W + v * C4;
    for (int k = 0; k < C4; ++k) {
      float4 w = wr[k];
      for (int r = 0; r < RB; ++r) acc[r] += dot(hs[r * C4 + k], w);
    }
    for (int r = 0; r < RB; ++r) {
      float x = acc[r];
      if (x > m[r]) { s[r] = s[r] * exp(m[r] - x) + 1.0f; m[r] = x; a[r] = v; }
      else s[r] += exp(x - m[r]);
    }
  }
  for (int r = 0; r < RB; ++r) {
    float mx = simd_max(m[r]);
    int am = simd_min(m[r] == mx ? a[r] : 0x7fffffff);
    float sm = simd_sum(m[r] == -INFINITY ? 0.0f : s[r] * exp(m[r] - mx));
    if (lane == 0) { tm[sg * RB + r] = mx; ta[sg * RB + r] = am; ts[sg * RB + r] = sm; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(tid) < nr) {
    int r = int(tid);
    float mx = -INFINITY; int am = 0x7fffffff;
    for (uint g = 0; g < nsg; ++g) mx = max(mx, tm[g * RB + r]);
    float sm = 0.0f;
    for (uint g = 0; g < nsg; ++g) {
      if (tm[g * RB + r] == mx) am = min(am, ta[g * RB + r]);
      if (tm[g * RB + r] != -INFINITY) sm += ts[g * RB + r] * exp(tm[g * RB + r] - mx);
    }
    ids[r0 + r] = am; conf[r0 + r] = 1.0f / sm;
  }
}
// CTC head with simdgroup matrices: 32 rows per threadgroup (128 threads), V in chunks of 64;
// W padded to a multiple of 64 rows. Online max / first-argmax / sum(exp) per row.
kernel void head_sg(device const float* h [[buffer(0)]], device const float* W [[buffer(1)]],
                    device const float* bias [[buffer(2)]], device int* ids [[buffer(3)]],
                    device float* conf [[buffer(4)]], constant HP& p [[buffer(5)]],
                    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint sgi [[simdgroup_index_in_threadgroup]]) {
  threadgroup float hs[32 * 128];
  threadgroup float L[32 * 64];
  int r0 = int(tg) * 32;
  int nr = min(32, p.rows - r0);
  for (int i = int(tid); i < 32 * 128; i += 128) { int r = i / 128; hs[i] = r < nr ? h[(r0 + r) * 128 + i % 128] : 0.0f; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int myr = int(tid) / 4, q = int(tid) % 4;
  float m = -INFINITY, s = 0.0f; int a = 0;
  int Vp = (p.V + 63) / 64 * 64;
  for (int v0 = 0; v0 < Vp; v0 += 64) {
    simdgroup_float8x8 acc[4][2];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k = 0; k < 128; k += 8) {
      simdgroup_float8x8 A[4], B[2];
      for (int i = 0; i < 4; ++i) simdgroup_load(A[i], hs + i * 8 * 128 + k, 128);
      for (int j = 0; j < 2; ++j)
        simdgroup_load(B[j], W + ulong(v0 + int(sgi) * 16 + j * 8) * 128 + k, 128, ulong2(0, 0), true);
      for (int i = 0; i < 4; ++i) for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], A[i], B[j], acc[i][j]);
    }
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 2; ++j) simdgroup_store(acc[i][j], L + i * 8 * 64 + int(sgi) * 16 + j * 8, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c = q * 16; c < q * 16 + 16; ++c) {
      int v = v0 + c;
      if (v >= p.V) break;
      float x = L[myr * 64 + c] + bias[v];
      if (x > m) { s = s * exp(m - x) + 1.0f; m = x; a = v; }
      else s += exp(x - m);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int off = 1; off < 4; off <<= 1) {
    float m2 = simd_shuffle_xor(m, ushort(off)), s2 = simd_shuffle_xor(s, ushort(off));
    int a2 = simd_shuffle_xor(a, ushort(off));
    float mx = max(m, m2);
    float sn = (m == -INFINITY ? 0.0f : s * exp(m - mx)) + (m2 == -INFINITY ? 0.0f : s2 * exp(m2 - mx));
    int an = (m == mx && m2 == mx) ? min(a, a2) : (m == mx ? a : a2);
    m = mx; s = sn; a = an;
  }
  if (q == 0 && myr < nr) { ids[r0 + myr] = a; conf[r0 + myr] = 1.0f / s; }
}
)MSL";

std::vector<float> Transpose(const std::vector<float>& w, int N, int K) {  // [N][K] -> [K][N]
  std::vector<float> t(size_t(N) * K);
  for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) t[size_t(k) * N + n] = w[size_t(n) * K + k];
  return t;
}
}  // namespace

struct MetalRec::Impl {
  id<MTLDevice> dev; id<MTLCommandQueue> q;
  id<MTLComputePipelineState> pg, pd, ps, ph;
  bool sgm = true, gsg = false;
  id<MTLComputePipelineState> phs;
  double tbody = 0, thead = 0; long nrows = 0;
  ~Impl() { if (nrows) fprintf(stderr, "metal rec gpu: body %.1f ms head %.1f ms rows %ld\n", tbody * 1e3, thead * 1e3, nrows); }
  id<MTLBuffer> stem, stemb, headW, headb;
  struct B { id<MTLBuffer> dwT, dwb, ex, exb, f1, f1b, f2, f2b, pr, prb; };
  std::vector<B> blocks;
  id<MTLBuffer> patch, h, t, e, gate, rline, starts, ids, conf;
  size_t cap_rows = 0, cap_lines = 0;
  int K0, C, E, S, V;
  std::mutex mu;
  id<MTLBuffer> Buf(const std::vector<float>& v) {
    return [dev newBufferWithBytes:v.data() length:v.size() * 4 options:MTLResourceStorageModeShared];
  }
  id<MTLBuffer> New(size_t bytes) { return [dev newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared]; }
};

MetalRec::MetalRec(const std::vector<float>& stem, const std::vector<float>& stemb, int K0, int C, int E, int S,
                   const std::vector<MetalRecBlock>& blocks, const float* headW, const float* headb, int V)
    : p_(new Impl) {
  if (C != 128 || E % 16 || K0 % 16 || E % 4 || C % 4 || E > 512 || S > 256) throw std::runtime_error("metal rec: unsupported dims");
  auto& P = *p_;
  P.K0 = K0; P.C = C; P.E = E; P.S = S; P.V = V;
  @autoreleasepool {
    P.dev = MTLCreateSystemDefaultDevice();
    if (!P.dev) throw std::runtime_error("metal rec: no Metal device");
    MTLCompileOptions* opt = [MTLCompileOptions new];
    opt.fastMathEnabled = NO;
    NSError* err = nil;
    // PPOCR_INDRA_METAL_FP16=1 (opt-in, not bit-exact): gemm tiles staged in threadgroup memory as half, FP32 accumulate.
    const char* fe = std::getenv("PPOCR_INDRA_METAL_FP16");
    std::string src = std::string((fe && fe[0] == '1') ? "#define TG_T half\n#define TG_T4 half4\n" : "#define TG_T float\n#define TG_T4 float4\n") + kSrc;
    id<MTLLibrary> lib = [P.dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:opt error:&err];
    if (!lib) throw std::runtime_error(std::string("metal rec: compile failed: ") + [[err localizedDescription] UTF8String]);
    auto pipe = [&](NSString* n) {
      id<MTLComputePipelineState> s = [P.dev newComputePipelineStateWithFunction:[lib newFunctionWithName:n] error:nil];
      if (!s) throw std::runtime_error("metal rec: pipeline failed");
      return s;
    };
    P.gsg = std::getenv("PPOCR_INDRA_METAL_GEMM_SG") != nullptr; P.pg = pipe(P.gsg ? @"gemm_sg" : @"gemm"); { const char* e = std::getenv("PPOCR_INDRA_METAL_SIMPLE"); P.sgm = !(e && *e); } P.pd = pipe(@"dw9"); P.ps = pipe(@"se"); P.ph = pipe(@"head"); P.phs = pipe(@"head_sg");
    P.q = [P.dev newCommandQueue];
    P.stem = P.Buf(Transpose(stem, C, K0)); P.stemb = P.Buf(std::vector<float>(stemb.begin(), stemb.begin() + C));
    for (auto& b : blocks) {
      Impl::B g;
      g.dwT = P.Buf(b.dwT); g.dwb = P.Buf(b.dwb);
      g.ex = P.Buf(Transpose(b.ex, E, C)); g.exb = P.Buf(std::vector<float>(b.exb.begin(), b.exb.begin() + E));
      g.f1 = P.Buf(b.f1); g.f1b = P.Buf(b.f1b); g.f2 = P.Buf(b.f2); g.f2b = P.Buf(b.f2b);
      g.pr = P.Buf(Transpose(b.pr, C, E)); g.prb = P.Buf(std::vector<float>(b.prb.begin(), b.prb.begin() + C));
      P.blocks.push_back(g);
    }
    {
      std::vector<float> hwp(size_t((V + 63) / 64 * 64) * C, 0.f);
      std::copy(headW, headW + size_t(V) * C, hwp.begin());
      P.headW = P.Buf(hwp);
    }
    P.headb = [P.dev newBufferWithBytes:headb length:size_t(V) * 4 options:MTLResourceStorageModeShared];
  }
}
MetalRec::~MetalRec() = default;

void MetalRec::Run(const float* patches, int rows, const std::vector<int>& starts, int64_t* ids, float* conf) {
  if (rows <= 0) return;
  auto& P = *p_;
  std::lock_guard<std::mutex> lk(P.mu);
  int B = int(starts.size()) - 1;
  @autoreleasepool {
    if (size_t(rows) > P.cap_rows) {
      size_t r = size_t(rows);
      P.patch = P.New(r * P.K0 * 4); P.h = P.New(r * P.C * 4); P.t = P.New(r * P.C * 4); P.e = P.New(r * P.E * 4);
      P.rline = P.New(r * 4); P.ids = P.New(r * 4); P.conf = P.New(r * 4); P.cap_rows = r;
    }
    if (size_t(B + 1) > P.cap_lines) {
      P.gate = P.New(size_t(B + 1) * P.E * 4); P.starts = P.New(size_t(B + 1) * 4); P.cap_lines = size_t(B + 1);
    }
    std::memcpy([P.patch contents], patches, size_t(rows) * P.K0 * 4);
    std::memcpy([P.starts contents], starts.data(), size_t(B + 1) * 4);
    int* rl = static_cast<int*>([P.rline contents]);
    for (int l = 0; l < B; ++l) for (int r = starts[l]; r < starts[l + 1]; ++r) rl[r] = l;
    id<MTLCommandBuffer> cb = [P.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    auto gemm = [&](id<MTLBuffer> A, id<MTLBuffer> Wt, id<MTLBuffer> bias, id<MTLBuffer> Y, int K, int N, int act, int resid, int gated) {
      struct { int rows, K, N, act, resid, gated; } gp = {rows, K, N, act, resid, gated};
      [enc setComputePipelineState:P.pg];
      [enc setBuffer:A offset:0 atIndex:0]; [enc setBuffer:Wt offset:0 atIndex:1]; [enc setBuffer:bias offset:0 atIndex:2];
      [enc setBuffer:Y offset:0 atIndex:3]; [enc setBytes:&gp length:sizeof gp atIndex:4];
      [enc setBuffer:P.gate offset:0 atIndex:5]; [enc setBuffer:P.rline offset:0 atIndex:6];
      [enc dispatchThreadgroups:MTLSizeMake((N + 63) / 64, (rows + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(P.gsg ? 128 : 256, 1, 1)];
    };
    gemm(P.patch, P.stem, P.stemb, P.h, P.K0, P.C, 1, 0, 0);
    for (auto& b : P.blocks) {
      struct { int rows, C; } dp = {rows, P.C};
      [enc setComputePipelineState:P.pd];
      [enc setBuffer:P.h offset:0 atIndex:0]; [enc setBuffer:b.dwT offset:0 atIndex:1]; [enc setBuffer:b.dwb offset:0 atIndex:2];
      [enc setBuffer:P.t offset:0 atIndex:3]; [enc setBytes:&dp length:sizeof dp atIndex:4];
      [enc setBuffer:P.rline offset:0 atIndex:5]; [enc setBuffer:P.starts offset:0 atIndex:6];
      [enc dispatchThreads:MTLSizeMake(P.C, rows, 1) threadsPerThreadgroup:MTLSizeMake(P.C, 2, 1)];
      gemm(P.t, b.ex, b.exb, P.e, P.C, P.E, 1, 0, 0);
      struct { int E, S; } sp = {P.E, P.S};
      [enc setComputePipelineState:P.ps];
      [enc setBuffer:P.e offset:0 atIndex:0]; [enc setBuffer:P.starts offset:0 atIndex:1];
      [enc setBuffer:b.f1 offset:0 atIndex:2]; [enc setBuffer:b.f1b offset:0 atIndex:3];
      [enc setBuffer:b.f2 offset:0 atIndex:4]; [enc setBuffer:b.f2b offset:0 atIndex:5];
      [enc setBuffer:P.gate offset:0 atIndex:6]; [enc setBytes:&sp length:sizeof sp atIndex:7];
      [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      gemm(P.e, b.pr, b.prb, P.h, P.E, P.C, 0, 1, 1);
    }
    static const bool prof = std::getenv("PPOCR_INDRA_METAL_PROF") != nullptr;
    if (prof) {
      [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
      P.tbody += cb.GPUEndTime - cb.GPUStartTime;
      cb = [P.q commandBuffer]; enc = [cb computeCommandEncoder];
    }
    struct { int rows, V, ldh4, pad; } hp = {rows, P.V, P.C / 4, 0};
    [enc setComputePipelineState:P.ph];
    [enc setBuffer:P.h offset:0 atIndex:0]; [enc setBuffer:P.headW offset:0 atIndex:1]; [enc setBuffer:P.headb offset:0 atIndex:2];
    [enc setBuffer:P.ids offset:0 atIndex:3]; [enc setBuffer:P.conf offset:0 atIndex:4]; [enc setBytes:&hp length:sizeof hp atIndex:5];
    if (P.sgm) {
      [enc setComputePipelineState:P.phs];
      [enc dispatchThreadgroups:MTLSizeMake((rows + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
      NSUInteger nt = std::min<NSUInteger>(256, P.ph.maxTotalThreadsPerThreadgroup);
      [enc dispatchThreadgroups:MTLSizeMake((rows + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(nt, 1, 1)];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("metal rec: command buffer failed");
    if (prof) { P.thead += cb.GPUEndTime - cb.GPUStartTime; P.nrows += rows; }
    const int* gi = static_cast<const int*>([P.ids contents]);
    const float* gc = static_cast<const float*>([P.conf contents]);
    for (int r = 0; r < rows; ++r) { ids[r] = gi[r]; conf[r] = gc[r]; }
  }
}
}  // namespace rpocr::indra
