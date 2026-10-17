#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_head.hpp"
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace rpocr::indra {
namespace {
const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
#define RB 32
#define C4 32   // C = 128 floats = 32 float4
struct P { int rows; int V; int ldh4; int pad; };
kernel void head(device const float4* h [[buffer(0)]], device const float4* W [[buffer(1)]],
                 device const float* bias [[buffer(2)]], device int* ids [[buffer(3)]],
                 device float* conf [[buffer(4)]], constant P& p [[buffer(5)]],
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
  // reduce within the simdgroup, then across simdgroups
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
)MSL";
}  // namespace

struct MetalHead::Impl {
  id<MTLDevice> dev; id<MTLCommandQueue> q; id<MTLComputePipelineState> ps;
  id<MTLBuffer> W, b, h, ids, conf;
  int V, C; size_t hcap = 0, ocap = 0;
  std::mutex mu;
  std::string name;
};

MetalHead::MetalHead(const float* W, const float* b, int V, int C) : p_(new Impl) {
  if (C != 128) throw std::runtime_error("metal head: C must be 128");
  @autoreleasepool {
    p_->dev = MTLCreateSystemDefaultDevice();
    if (!p_->dev) throw std::runtime_error("metal head: no Metal device");
    p_->name = [[p_->dev name] UTF8String];
    MTLCompileOptions* opt = [MTLCompileOptions new];
    opt.fastMathEnabled = NO;  // exact exp/compare semantics for argmax/conf parity with the CPU path
    NSError* err = nil;
    id<MTLLibrary> lib = [p_->dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:opt error:&err];
    if (!lib) throw std::runtime_error(std::string("metal head: compile failed: ") + [[err localizedDescription] UTF8String]);
    id<MTLFunction> fn = [lib newFunctionWithName:@"head"];
    p_->ps = [p_->dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p_->ps) throw std::runtime_error("metal head: pipeline failed");
    p_->q = [p_->dev newCommandQueue];
    p_->W = [p_->dev newBufferWithBytes:W length:size_t(V) * C * 4 options:MTLResourceStorageModeShared];
    p_->b = [p_->dev newBufferWithBytes:b length:size_t(V) * 4 options:MTLResourceStorageModeShared];
    p_->V = V; p_->C = C;
  }
}
MetalHead::~MetalHead() = default;
const char* MetalHead::device() const { return p_->name.c_str(); }

void MetalHead::Run(const float* h, int rows, int ldh, int64_t* ids, float* conf) {
  if (rows <= 0) return;
  std::lock_guard<std::mutex> g(p_->mu);
  @autoreleasepool {
    size_t hb = size_t(rows) * p_->C * 4;
    if (hb > p_->hcap) { p_->h = [p_->dev newBufferWithLength:hb options:MTLResourceStorageModeShared]; p_->hcap = hb; }
    if (size_t(rows) * 4 > p_->ocap) {
      p_->ids = [p_->dev newBufferWithLength:size_t(rows) * 4 options:MTLResourceStorageModeShared];
      p_->conf = [p_->dev newBufferWithLength:size_t(rows) * 4 options:MTLResourceStorageModeShared];
      p_->ocap = size_t(rows) * 4;
    }
    float* hd = static_cast<float*>([p_->h contents]);
    for (int r = 0; r < rows; ++r) std::memcpy(hd + size_t(r) * p_->C, h + size_t(r) * ldh, size_t(p_->C) * 4);
    struct { int rows, V, ldh4, pad; } prm = {rows, p_->V, p_->C / 4, 0};
    id<MTLCommandBuffer> cb = [p_->q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p_->ps];
    [enc setBuffer:p_->h offset:0 atIndex:0];
    [enc setBuffer:p_->W offset:0 atIndex:1];
    [enc setBuffer:p_->b offset:0 atIndex:2];
    [enc setBuffer:p_->ids offset:0 atIndex:3];
    [enc setBuffer:p_->conf offset:0 atIndex:4];
    [enc setBytes:&prm length:sizeof prm atIndex:5];
    NSUInteger tgs = (NSUInteger(rows) + 31) / 32;
    NSUInteger nt = std::min<NSUInteger>(256, p_->ps.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(nt, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("metal head: command buffer failed");
    const int* gi = static_cast<const int*>([p_->ids contents]);
    const float* gc = static_cast<const float*>([p_->conf contents]);
    for (int r = 0; r < rows; ++r) { ids[r] = gi[r]; conf[r] = gc[r]; }
  }
}
}  // namespace rpocr::indra
