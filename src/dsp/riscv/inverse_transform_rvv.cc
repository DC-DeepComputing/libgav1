// Copyright 2026 RISC-V RVV optimization work.
// RVV 1.0 inverse transform kernels (8bpp). Bit-exact port of the NEON DCT-4/8
// paths. Only kTransform1dDct/kTransform1dSize4 and Size8 row+column slots are
// replaced; all other transforms keep the C implementation.
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
  // C reference HadamardRotation_C(a,b,flip): if (flip) swap(a,b); then
  // dst[a] = dst[a]+dst[b]; dst[b] = dst[a]-dst[b] (on swapped indices).
  // Unrolled:  !flip -> a_new = a+b, b_new = a-b
  //            flip  -> a_new = b-a, b_new = a+b
  const vint16m1_t sum = __riscv_vsadd_vv_i16m1(a, b, vm);  // a+b
  if (flip) {
    const vint16m1_t bma = __riscv_vssub_vv_i16m1(b, a, vm);  // b-a
    a = bma;
    b = sum;
  } else {
    const vint16m1_t anb = __riscv_vssub_vv_i16m1(a, b, vm);  // a-b
    a = sum;
    b = anb;
  }
}

// DCT-4 stages 12/17 shared by the 4-point and 8-point transforms (no
// bit-reverse here; callers do their own stage-1 permutation).
inline void Dct4Core(vint16m1_t& s0, vint16m1_t& s1, vint16m1_t& s2,
                     vint16m1_t& s3, size_t vm) {
  Butterfly(s0, s1, 32, true, vm);
  Butterfly(s2, s3, 48, false, vm);
  Hadamard(s0, s3, false, vm);
  Hadamard(s1, s2, false, vm);
}

// Four-point 1D DCT: stage1 bit-reverse [0,2,1,3] + Dct4Core.
inline void Dct4(vint16m1_t& s0, vint16m1_t& s1, vint16m1_t& s2,
                 vint16m1_t& s3, size_t vm) {
  const vint16m1_t t = s1; s1 = s2; s2 = t;
  Dct4Core(s0, s1, s2, s3, vm);
}

// DCT-8 stages (NEON Dct8Stages): stage8 butterfly(4,7,56,F)/(5,6,24,F);
// stage13 hadamard(4,5,F)/(6,7,T); stage18 butterfly(6,5,32,T);
// stage22 hadamard(0,7)/(1,6)/(2,5)/(3,4).
inline void Dct8Core(vint16m1_t& s4, vint16m1_t& s5, vint16m1_t& s6,
                     vint16m1_t& s7, vint16m1_t& s0, vint16m1_t& s1,
                     vint16m1_t& s2, vint16m1_t& s3, size_t vm) {
  Butterfly(s4, s7, 56, false, vm);
  Butterfly(s5, s6, 24, false, vm);
  Hadamard(s4, s5, false, vm);
  Hadamard(s6, s7, true, vm);
  Butterfly(s6, s5, 32, true, vm);
  Hadamard(s0, s7, false, vm);
  Hadamard(s1, s6, false, vm);
  Hadamard(s2, s5, false, vm);
  Hadamard(s3, s4, false, vm);
}

// Eight-point 1D DCT: stage1 bit-reverse [0,4,2,6,1,5,3,7] + Dct4Core + Dct8.
inline void Dct8(vint16m1_t& s0, vint16m1_t& s1, vint16m1_t& s2,
                 vint16m1_t& s3, vint16m1_t& s4, vint16m1_t& s5,
                 vint16m1_t& s6, vint16m1_t& s7, size_t vm) {
  // s[i] = x[i]; permute to [x0,x4,x2,x6,x1,x5,x3,x7]: swap s1<->s4, s3<->s6.
  const vint16m1_t t1 = s1; s1 = s4; s4 = t1;
  const vint16m1_t t3 = s3; s3 = s6; s6 = t3;
  Dct4Core(s0, s1, s2, s3, vm);
  Dct8Core(s4, s5, s6, s7, s0, s1, s2, s3, vm);
}

void RowDcOnly(int16_t* src, int tx_width, bool should_round, int rs) {
  int16_t v = src[0];
  if (should_round) v = (int16_t)((int32_t(v) * 2896 + 2048) >> 12);
  const int16_t c = Cos128(32);
  int16_t r = (int16_t)((int32_t(v) * c + 2048) >> 12);
  if (rs > 0) r = (int16_t)((int32_t(r) + (1 << (rs - 1))) >> rs);
  for (int j = 0; j < tx_width; ++j) src[j] = r;
}

void ColDcOnly(int16_t* src, int tx_height, int tx_width) {
  const int16_t c = Cos128(32);
  for (int j = 0; j < tx_width; ++j)
    src[j] = (int16_t)((int32_t(src[j]) * c + 2048) >> 12);
  for (int y = 1; y < tx_height; ++y)
    memcpy(&src[y * tx_width], src, tx_width * sizeof(src[0]));
}

// Vector rounding right shift by sh over n contiguous int16 elements.
void RoundShiftRight(int16_t* src, size_t n, int sh) {
  size_t i = 0;
  while (i < n) {
    size_t vm = __riscv_vsetvl_e16m1(n - i);
    const vint16m1_t a = __riscv_vle16_v_i16m1(src + i, vm);
    const vint32m2_t aw = __riscv_vwmul_vx_i32m2(a, 1, vm);
    const vint32m2_t r = __riscv_vsra_vx_i32m2(
        __riscv_vadd_vx_i32m2(aw, 1 << (sh - 1), vm), sh, vm);
    __riscv_vse16_v_i16m1(
        src + i, __riscv_vnclip_wx_i16m1(r, 0, __RISCV_VXRM_RDN, vm), vm);
    i += vm;
  }
}

// Multiply by 2896/4096 rounded over n contiguous int16 elements.
void ApplyRowRounding(int16_t* src, size_t n) {
  size_t i = 0;
  while (i < n) {
    const size_t vm = __riscv_vsetvl_e16m1(n - i);
    __riscv_vse16_v_i16m1(
        src + i, Qrdmulh(__riscv_vle16_v_i16m1(src + i, vm), 2896, vm), vm);
    i += vm;
  }
}

void Dct4Row(TransformType, TransformSize tx_size, int h, void* src_buffer,
             int, int, void*) {
  auto* src = static_cast<int16_t*>(src_buffer);
  const int tx_height = kTransformHeight[tx_size];
  const int tx_width = kTransformWidth[tx_size];
  const bool should_round = (tx_height == 8);
  const int row_shift = (tx_height == 16) ? 1 : 0;
  if (h == 1) { RowDcOnly(src, tx_width, should_round, row_shift); return; }
  if (should_round) ApplyRowRounding(src, size_t(h) * tx_width);
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
  if (row_shift > 0) RoundShiftRight(src, size_t(h) * 4, row_shift);
}

void Dct8Row(TransformType, TransformSize tx_size, int h, void* src_buffer,
             int, int, void*) {
  auto* src = static_cast<int16_t*>(src_buffer);
  const int tx_width = kTransformWidth[tx_size];
  const bool should_round = kShouldRound[tx_size];
  const int row_shift = kTransformRowShift[tx_size];
  if (h == 1) { RowDcOnly(src, tx_width, should_round, row_shift); return; }
  if (should_round) ApplyRowRounding(src, size_t(h) * tx_width);
  size_t done = 0;
  while (done < size_t(h)) {
    size_t vm = __riscv_vsetvl_e16m1(size_t(h) - done);
    int16_t* bp = src + done * 8;
    const ptrdiff_t bs = 16;  // 8 int16 elements per row
    vint16m1_t s0 = __riscv_vlse16_v_i16m1(bp, bs, vm);
    vint16m1_t s1 = __riscv_vlse16_v_i16m1(bp + 1, bs, vm);
    vint16m1_t s2 = __riscv_vlse16_v_i16m1(bp + 2, bs, vm);
    vint16m1_t s3 = __riscv_vlse16_v_i16m1(bp + 3, bs, vm);
    vint16m1_t s4 = __riscv_vlse16_v_i16m1(bp + 4, bs, vm);
    vint16m1_t s5 = __riscv_vlse16_v_i16m1(bp + 5, bs, vm);
    vint16m1_t s6 = __riscv_vlse16_v_i16m1(bp + 6, bs, vm);
    vint16m1_t s7 = __riscv_vlse16_v_i16m1(bp + 7, bs, vm);
    Dct8(s0, s1, s2, s3, s4, s5, s6, s7, vm);
    __riscv_vsse16_v_i16m1(bp, bs, s0, vm);
    __riscv_vsse16_v_i16m1(bp + 1, bs, s1, vm);
    __riscv_vsse16_v_i16m1(bp + 2, bs, s2, vm);
    __riscv_vsse16_v_i16m1(bp + 3, bs, s3, vm);
    __riscv_vsse16_v_i16m1(bp + 4, bs, s4, vm);
    __riscv_vsse16_v_i16m1(bp + 5, bs, s5, vm);
    __riscv_vsse16_v_i16m1(bp + 6, bs, s6, vm);
    __riscv_vsse16_v_i16m1(bp + 7, bs, s7, vm);
    done += vm;
  }
  if (row_shift > 0) RoundShiftRight(src, size_t(h) * 8, row_shift);
}

void Dct4Column(TransformType tx_type, TransformSize tx_size, int h,
                void* src_buffer, int start_x, int start_y, void* dst_frame) {
  auto* src = static_cast<int16_t*>(src_buffer);
  const int tx_width = kTransformWidth[tx_size];
  const int tx_height = kTransformHeight[tx_size];
  if (kTransformFlipColumnsMask.Contains(tx_type)) {
    for (int i = 0; i < tx_height; ++i) {
      int16_t* r = src + i * tx_width;
      for (int k = 0; k < tx_width / 2; ++k) std::swap(r[k], r[tx_width - 1 - k]);
    }
  }
  if (h == 1) { ColDcOnly(src, tx_height, tx_width); }
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
  // flip_rows only applies when the column transform is ADST (C reference:
  // transform1d_type == kTransform1dAdst && FlipRowsMask). The DCT column
  // slots are only invoked for tx types whose column transform is DCT, and
  // none of them are in kTransformFlipRowsMask, so rows are never flipped.
  for (int i = 0; i < tx_height; ++i) {
    const int row = i * tx_width;
    uint8_t* dst = frame[start_y + i] + start_x;
    for (int j = 0; j < tx_width; ++j) {
      const int a = (int(src[row + j]) + 8) >> 4;
      const int v = int(dst[j]) + a;
      dst[j] = (uint8_t)std::max(0, std::min(255, v));
    }
  }
}

void Dct8Column(TransformType tx_type, TransformSize tx_size, int h,
                void* src_buffer, int start_x, int start_y, void* dst_frame) {
  auto* src = static_cast<int16_t*>(src_buffer);
  const int tx_width = kTransformWidth[tx_size];
  const int tx_height = kTransformHeight[tx_size];
  if (kTransformFlipColumnsMask.Contains(tx_type)) {
    for (int i = 0; i < tx_height; ++i) {
      int16_t* r = src + i * tx_width;
      for (int k = 0; k < tx_width / 2; ++k) std::swap(r[k], r[tx_width - 1 - k]);
    }
  }
  if (h == 1) { ColDcOnly(src, tx_height, tx_width); }
  else {
    size_t done = 0;
    while (done < size_t(tx_width)) {
      size_t vm = __riscv_vsetvl_e16m1(size_t(tx_width) - done);
      int16_t* cp = src + done;
      vint16m1_t s0 = __riscv_vle16_v_i16m1(cp, vm);
      vint16m1_t s1 = __riscv_vle16_v_i16m1(cp + tx_width, vm);
      vint16m1_t s2 = __riscv_vle16_v_i16m1(cp + 2 * tx_width, vm);
      vint16m1_t s3 = __riscv_vle16_v_i16m1(cp + 3 * tx_width, vm);
      vint16m1_t s4 = __riscv_vle16_v_i16m1(cp + 4 * tx_width, vm);
      vint16m1_t s5 = __riscv_vle16_v_i16m1(cp + 5 * tx_width, vm);
      vint16m1_t s6 = __riscv_vle16_v_i16m1(cp + 6 * tx_width, vm);
      vint16m1_t s7 = __riscv_vle16_v_i16m1(cp + 7 * tx_width, vm);
      Dct8(s0, s1, s2, s3, s4, s5, s6, s7, vm);
      __riscv_vse16_v_i16m1(cp, s0, vm);
      __riscv_vse16_v_i16m1(cp + tx_width, s1, vm);
      __riscv_vse16_v_i16m1(cp + 2 * tx_width, s2, vm);
      __riscv_vse16_v_i16m1(cp + 3 * tx_width, s3, vm);
      __riscv_vse16_v_i16m1(cp + 4 * tx_width, s4, vm);
      __riscv_vse16_v_i16m1(cp + 5 * tx_width, s5, vm);
      __riscv_vse16_v_i16m1(cp + 6 * tx_width, s6, vm);
      __riscv_vse16_v_i16m1(cp + 7 * tx_width, s7, vm);
      done += vm;
    }
  }
  auto& frame = *static_cast<Array2DView<uint8_t>*>(dst_frame);
  // See Dct4Column: DCT column slots never flip rows.
  for (int i = 0; i < tx_height; ++i) {
    const int row = i * tx_width;
    uint8_t* dst = frame[start_y + i] + start_x;
    for (int j = 0; j < tx_width; ++j) {
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
  dsp->inverse_transforms[kTransform1dDct][kTransform1dSize8][kRow] = Dct8Row;
  dsp->inverse_transforms[kTransform1dDct][kTransform1dSize8][kColumn] = Dct8Column;
}

}  // namespace dsp
}  // namespace libgav1


