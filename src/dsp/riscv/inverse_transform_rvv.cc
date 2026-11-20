// Copyright 2026 RISC-V RVV optimization work.
// RVV 1.0 inverse transform kernels (8bpp). Bit-exact port of the NEON DCT-4
// path. Only kTransform1dDct/kTransform1dSize4 row+column slots are replaced.
#include <riscv_vector.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "src/dsp/constants.h"
#include "src/dsp/dsp.h"
#include "src/dsp/inverse_transform.h"
#include "src/utils/array_2d.h"
#include "src/utils/common.h"
namespace libgav1 {
namespace dsp {
namespace {
#include "src/dsp/inverse_transform.inc"

inline bool HasRVVCpu() {
  static int cached = -1;
  if (cached < 0) {
    int ok = 0;
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
      char line[512];
      while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "isa", 3) == 0) {
          const char* p = strstr(line, "rv64");
          if (p) { p += 4; while (*p >= 'a' && *p <= 'z') {
            if (*p == 'v') { ok = 1; break; } ++p; } }
          break;
        }
      }
      fclose(f);
    }
    cached = ok;
  }
  return cached != 0;
}

inline vint16m1_t Qrdmulh(const vint16m1_t a, const int16_t mult, size_t vm) {
  const vint32m2_t aw = __riscv_vwmul_vx_i32m2(a, mult, vm);
  const vint32m2_t rnd = __riscv_vadd_vx_i32m2(aw, 2048, vm);
  return __riscv_vnclip_wx_i16m1(rnd, 12, __RISCV_VXRM_RDN, vm);
}

inline void Butterfly(vint16m1_t& a, vint16m1_t& b, int angle, bool flip,
                      size_t vm) {
  const int16_t cos128 = Cos128(angle);
  const int16_t sin128 = Sin128(angle);
  const vint32m2_t ax = __riscv_vwmul_vx_i32m2(a, cos128, vm);
  const vint32m2_t ay = __riscv_vwmul_vx_i32m2(a, sin128, vm);
  const vint32m2_t bx = __riscv_vwmul_vx_i32m2(b, sin128, vm);
  const vint32m2_t by = __riscv_vwmul_vx_i32m2(b, cos128, vm);
  const vint32m2_t x = __riscv_vsub_vv_i32m2(ax, bx, vm);
  const vint32m2_t y = __riscv_vadd_vv_i32m2(ay, by, vm);
  const vint16m1_t xo = __riscv_vnclip_wx_i16m1(
      __riscv_vadd_vx_i32m2(x, 2048, vm), 12, __RISCV_VXRM_RDN, vm);
  const vint16m1_t yo = __riscv_vnclip_wx_i16m1(
      __riscv_vadd_vx_i32m2(y, 2048, vm), 12, __RISCV_VXRM_RDN, vm);
  if (flip) { a = yo; b = xo; } else { a = xo; b = yo; }
}

inline void Hadamard(vint16m1_t& a, vint16m1_t& b, bool flip, size_t vm) {
  const vint16m1_t x = __riscv_vsadd_vv_i16m1(a, b, vm);
  const vint16m1_t y = __riscv_vssub_vv_i16m1(a, b, vm);
  if (flip) { a = y; b = x; } else { a = x; b = y; }
}

inline void Dct4(vint16m1_t& s0, vint16m1_t& s1, vint16m1_t& s2,
                 vint16m1_t& s3, size_t vm) {
  const vint16m1_t t = s1; s1 = s2; s2 = t;
  Butterfly(s0, s1, 32, true, vm);
  Butterfly(s2, s3, 48, false, vm);
  Hadamard(s0, s3, false, vm);
  Hadamard(s1, s2, false, vm);
}

void Dct4RowDcOnly(int16_t* src, int tx_width, bool should_round, int rs) {
  int16_t v = src[0];
  if (should_round) v = (int16_t)((int32_t(v) * 2896 + 2048) >> 12);
  const int16_t c = Cos128(32);
  int16_t r = (int16_t)((int32_t(v) * c + 2048) >> 12);
  if (rs > 0) r = (int16_t)((int32_t(r) + (1 << (rs - 1))) >> rs);
  for (int j = 0; j < tx_width; ++j) src[j] = r;
}

void Dct4Row(TransformType, TransformSize tx_size, int h, void* src_buffer,
              int, int, void*) {
  auto* src = static_cast<int16_t*>(src_buffer);
  const int tx_height = kTransformHeight[tx_size];
  const int tx_width = kTransformWidth[tx_size];
  const bool should_round = (tx_height == 8);
  const int row_shift = (tx_height == 16) ? 1 : 0;
  if (h == 1) { Dct4RowDcOnly(src, tx_width, should_round, row_shift); return; }
  if (should_round) {
    size_t n = size_t(h) * tx_width, i = 0;
    while (i < n) { size_t vm = __riscv_vsetvl_e16m1(n - i);
      __riscv_vse16_v_i16m1(src + i, Qrdmulh(__riscv_vle16_v_i16m1(src + i, vm), 2896, vm), vm);
      i += vm; }
  }
  size_t done = 0;
  while (done < size_t(h)) {
    size_t vm = __riscv_vsetvl_e16m1(size_t(h) - done);
    int16_t* bp = src + done * 4;
    const ptrdiff_t bs = 8;
    vint16m1_t s0 = __riscv_vlse16_v_i16m1(bp, bs, vm);
    vint16m1_t s1 = __riscv_vlse16_v_i16m1(bp + 1, bs, vm);
    vint16m1_t s2 = __riscv_vlse16_v_i16m1(bp + 2, bs, vm);
    vint16m1_t s3 = __riscv_vlse16_v_i16m1(bp + 3, bs, vm);
    Dct4(s0, s1, s2, s3, vm);
    __riscv_vsse16_v_i16m1(bp, bs, s0, vm);
    __riscv_vsse16_v_i16m1(bp + 1, bs, s1, vm);
    __riscv_vsse16_v_i16m1(bp + 2, bs, s2, vm);
    __riscv_vsse16_v_i16m1(bp + 3, bs, s3, vm);
    done += vm;
  }
  if (row_shift > 0) {
    size_t n = size_t(h) * 4, i = 0;
    while (i < n) { size_t vm = __riscv_vsetvl_e16m1(n - i);
      const vint16m1_t a = __riscv_vle16_v_i16m1(src + i, vm);
      const vint32m2_t aw = __riscv_vwmul_vx_i32m2(a, 1, vm);
      const vint32m2_t sh = __riscv_vsra_vx_i32m2(__riscv_vadd_vx_i32m2(aw, 1, vm), 1, vm);
      __riscv_vse16_v_i16m1(src + i, __riscv_vnclip_wx_i16m1(sh, 0, __RISCV_VXRM_RDN, vm), vm);
      i += vm; }
  }
}

void Dct4ColDcOnly(int16_t* src, int tx_height, int tx_width) {
  const int16_t c = Cos128(32);
  for (int j = 0; j < tx_width; ++j) src[j] = (int16_t)((int32_t(src[j]) * c + 2048) >> 12);
  for (int y = 1; y < tx_height; ++y) memcpy(&src[y * tx_width], src, tx_width * sizeof(src[0]));
}

void Dct4Column(TransformType tx_type, TransformSize tx_size, int h,
                 void* src_buffer, int start_x, int start_y, void* dst_frame) {
  auto* src = static_cast<int16_t*>(src_buffer);
  const int tx_width = kTransformWidth[tx_size];
  const int tx_height = kTransformHeight[tx_size];
  if (kTransformFlipColumnsMask.Contains(tx_type)) {
    for (int i = 0; i < tx_height; ++i) { int16_t* r = src + i * tx_width;
      std::swap(r[0], r[3]); std::swap(r[1], r[2]); }
  }
  if (h == 1) { Dct4ColDcOnly(src, tx_height, tx_width); }
  else {
    size_t done = 0;
    while (done < size_t(tx_width)) {
      size_t vm = __riscv_vsetvl_e16m1(size_t(tx_width) - done);
      int16_t* cp = src + done;
      vint16m1_t s0 = __riscv_vle16_v_i16m1(cp, vm);
      vint16m1_t s1 = __riscv_vle16_v_i16m1(cp + tx_width, vm);
      vint16m1_t s2 = __riscv_vle16_v_i16m1(cp + 2 * tx_width, vm);
      vint16m1_t s3 = __riscv_vle16_v_i16m1(cp + 3 * tx_width, vm);
      Dct4(s0, s1, s2, s3, vm);
      __riscv_vse16_v_i16m1(cp, s0, vm);
      __riscv_vse16_v_i16m1(cp + tx_width, s1, vm);
      __riscv_vse16_v_i16m1(cp + 2 * tx_width, s2, vm);
      __riscv_vse16_v_i16m1(cp + 3 * tx_width, s3, vm);
      done += vm;
    }
  }
  auto& frame = *static_cast<Array2DView<uint8_t>*>(dst_frame);
  const bool flip_rows = kTransformFlipRowsMask.Contains(tx_type);
  for (int i = 0; i < tx_height; ++i) {
    const int row = flip_rows ? (tx_height - i - 1) * 4 : i * 4;
    uint8_t* dst = frame[start_y + i] + start_x;
    for (int j = 0; j < 4; ++j) {
      const int a = (int(src[row + j]) + 8) >> 4;
      const int v = int(dst[j]) + a;
      dst[j] = (uint8_t)std::max(0, std::min(255, v));
    }
  }
}

}  // namespace

void InverseTransformInit_RVV();
void InverseTransformInit_RVV() {
  if (!HasRVVCpu()) return;
  Dsp* const dsp = dsp_internal::GetWritableDspTable(kBitdepth8);
  assert(dsp != nullptr);
  dsp->inverse_transforms[kTransform1dDct][kTransform1dSize4][kRow] = Dct4Row;
  dsp->inverse_transforms[kTransform1dDct][kTransform1dSize4][kColumn] = Dct4Column;
}

}  // namespace dsp
}  // namespace libgav1
