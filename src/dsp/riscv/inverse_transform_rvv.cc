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

// Vectorized StoreToFrameWithRound for one frame row: residual =
// (src+8)>>4 (arithmetic; range guaranteed by the transform like NEON
// vrshr_n_s16), then frame + residual clamped to [0, 255].
void StoreToFrameRow(const int16_t* src_row, uint8_t* dst, int width) {
  size_t i = 0;
  const size_t n = size_t(width);
  while (i < n) {
    const size_t vm = __riscv_vsetvl_e16m1(n - i);
    vint16m1_t r = __riscv_vle16_v_i16m1(src_row + i, vm);
    r = __riscv_vsra_vx_i16m1(__riscv_vadd_vx_i16m1(r, 8, vm), 4, vm);
    const vuint8mf2_t fb = __riscv_vle8_v_u8mf2(dst + i, vm);
    vint16m1_t sum = __riscv_vadd_vv_i16m1(
        r, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(fb, vm)),
        vm);
    sum = __riscv_vmin_vx_i16m1(sum, 255, vm);
    sum = __riscv_vmax_vx_i16m1(sum, 0, vm);
    __riscv_vse8_v_u8mf2(
        dst + i,
        __riscv_vnclipu_wx_u8mf2(__riscv_vreinterpret_v_i16m1_u16m1(sum), 0,
                                 __RISCV_VXRM_RDN, vm),
        vm);
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
    // Rows are structs of 4 contiguous int16 fields: vlseg4 deinterleaves the
    // 4 columns in one instruction (replaces 4 strided loads).
    vint16m1x4_t t = __riscv_vlseg4e16_v_i16m1x4(bp, vm);
    vint16m1_t s0 = __riscv_vget_v_i16m1x4_i16m1(t, 0);
    vint16m1_t s1 = __riscv_vget_v_i16m1x4_i16m1(t, 1);
    vint16m1_t s2 = __riscv_vget_v_i16m1x4_i16m1(t, 2);
    vint16m1_t s3 = __riscv_vget_v_i16m1x4_i16m1(t, 3);
    Dct4(s0, s1, s2, s3, vm);
    __riscv_vsseg4e16_v_i16m1x4(
        bp, __riscv_vcreate_v_i16m1x4(s0, s1, s2, s3), vm);
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
    vint16m1x8_t t = __riscv_vlseg8e16_v_i16m1x8(bp, vm);
    vint16m1_t s0 = __riscv_vget_v_i16m1x8_i16m1(t, 0);
    vint16m1_t s1 = __riscv_vget_v_i16m1x8_i16m1(t, 1);
    vint16m1_t s2 = __riscv_vget_v_i16m1x8_i16m1(t, 2);
    vint16m1_t s3 = __riscv_vget_v_i16m1x8_i16m1(t, 3);
    vint16m1_t s4 = __riscv_vget_v_i16m1x8_i16m1(t, 4);
    vint16m1_t s5 = __riscv_vget_v_i16m1x8_i16m1(t, 5);
    vint16m1_t s6 = __riscv_vget_v_i16m1x8_i16m1(t, 6);
    vint16m1_t s7 = __riscv_vget_v_i16m1x8_i16m1(t, 7);
    Dct8(s0, s1, s2, s3, s4, s5, s6, s7, vm);
    __riscv_vsseg8e16_v_i16m1x8(
        bp, __riscv_vcreate_v_i16m1x8(s0, s1, s2, s3, s4, s5, s6, s7), vm);
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
    StoreToFrameRow(src + i * tx_width, frame[start_y + i] + start_x, tx_width);
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
    StoreToFrameRow(src + i * tx_width, frame[start_y + i] + start_x, tx_width);
  }
}

//------------------------------------------------------------------------------
// ADST (asymmetric discrete sine transforms), 4- and 8-point. Bit-exact port
// of the NEON Adst4/Adst8 paths (i32-domain multiply-add, +2048 rounding bias,
// saturating narrowing shift of 12; saturating hadamard; saturating negate in
// the output permutation).

inline vint16m1_t Rnd12Sat(const vint32m2_t x, size_t vm) {
  return __riscv_vnclip_wx_i16m1(__riscv_vadd_vx_i32m2(x, 2048, vm), 12,
                                 __RISCV_VXRM_RDN, vm);
}

inline vint16m1_t NegSat(const vint16m1_t v, size_t vm) {
  const vint16m1_t zero = __riscv_vmv_v_x_i16m1(0, vm);
  return __riscv_vssub_vv_i16m1(zero, v, vm);
}

// Four-point ADST on x0..x3 (each a vector of independent lanes).
inline void Adst4(vint16m1_t& x0, vint16m1_t& x1, vint16m1_t& x2,
                  vint16m1_t& x3, size_t vm) {
  constexpr int16_t m0 = 1321, m1 = 2482, m2 = 3344, m3 = 3803;
  const vint32m2_t s5 = __riscv_vwmul_vx_i32m2(x3, m1, vm);
  const vint32m2_t s6 = __riscv_vwmul_vx_i32m2(x3, m3, vm);
  const vint32m2_t a7 = __riscv_vwsub_vv_i32m2(x0, x2, vm);
  const vint32m2_t b7 =
      __riscv_vadd_vv_i32m2(a7, __riscv_vsext_vf2_i32m2(x3, vm), vm);
  vint32m2_t s0 = __riscv_vwmul_vx_i32m2(x0, m0, vm);
  vint32m2_t s1 = __riscv_vwmul_vx_i32m2(x0, m1, vm);
  s0 = __riscv_vadd_vv_i32m2(s0, __riscv_vwmul_vx_i32m2(x2, m3, vm), vm);
  s1 = __riscv_vsub_vv_i32m2(s1, __riscv_vwmul_vx_i32m2(x2, m0, vm), vm);
  const vint32m2_t s3 = __riscv_vwmul_vx_i32m2(x1, m2, vm);
  const vint32m2_t s2 = __riscv_vmul_vx_i32m2(b7, m2, vm);
  s0 = __riscv_vadd_vv_i32m2(s0, s5, vm);
  s1 = __riscv_vsub_vv_i32m2(s1, s6, vm);
  x0 = Rnd12Sat(__riscv_vadd_vv_i32m2(s0, s3, vm), vm);
  x1 = Rnd12Sat(__riscv_vadd_vv_i32m2(s1, s3, vm), vm);
  x2 = Rnd12Sat(s2, vm);
  x3 = Rnd12Sat(
      __riscv_vsub_vv_i32m2(__riscv_vadd_vv_i32m2(s0, s1, vm), s3, vm), vm);
}

// Eight-point ADST on s0..s7 (already input-permuted). Output permutation is
// applied on exit: out = [s0,-s4,s6,-s2,s3,-s7,s5,-s1].
inline void Adst8(vint16m1_t& s0, vint16m1_t& s1, vint16m1_t& s2,
                  vint16m1_t& s3, vint16m1_t& s4, vint16m1_t& s5,
                  vint16m1_t& s6, vint16m1_t& s7, size_t vm) {
  // stage 2.
  Butterfly(s0, s1, 60, true, vm);
  Butterfly(s2, s3, 44, true, vm);
  Butterfly(s4, s5, 28, true, vm);
  Butterfly(s6, s7, 12, true, vm);
  // stage 3.
  Hadamard(s0, s4, false, vm);
  Hadamard(s1, s5, false, vm);
  Hadamard(s2, s6, false, vm);
  Hadamard(s3, s7, false, vm);
  // stage 4.
  Butterfly(s4, s5, 48, true, vm);
  Butterfly(s7, s6, 16, true, vm);
  // stage 5.
  Hadamard(s0, s2, false, vm);
  Hadamard(s4, s6, false, vm);
  Hadamard(s1, s3, false, vm);
  Hadamard(s5, s7, false, vm);
  // stage 6.
  Butterfly(s2, s3, 32, true, vm);
  Butterfly(s6, s7, 32, true, vm);
  // stage 7: output permutation.
  const vint16m1_t t0 = s0, t1 = s1, t2 = s2, t3 = s3;
  const vint16m1_t t4 = s4, t5 = s5, t6 = s6, t7 = s7;
  s0 = t0;
  s1 = NegSat(t4, vm);
  s2 = t6;
  s3 = NegSat(t2, vm);
  s4 = t3;
  s5 = NegSat(t7, vm);
  s6 = t5;
  s7 = NegSat(t1, vm);
}

inline int32_t R12(int32_t v) { return (v + 2048) >> 12; }

inline int32_t Clamp16(int32_t v) {
  v -= (v == 0x8000);
  return v;
}

// Single-lane DC-only ADST-4: out[k] for k in 0..3 (scalar, bit-exact vs
// Adst4DcOnly_C).
inline void Adst4DcLane(int16_t v, bool should_round, int row_shift,
                        int16_t out[4]) {
  if (should_round) v = (int16_t)((int32_t(v) * 2896 + 2048) >> 12);
  int32_t s0 = int32_t(v) * 1321;
  int32_t s1 = int32_t(v) * 2482;
  int32_t s2 = int32_t(v) * 3344;
  int32_t d0 = Clamp16(R12(s0));
  int32_t d1 = Clamp16(R12(s1));
  int32_t d2 = R12(s2);
  int32_t d3 = Clamp16(R12(s0 + s1));
  if (row_shift > 0) {
    d0 = (d0 + 1) >> 1;
    d1 = (d1 + 1) >> 1;
    d2 = (d2 + 1) >> 1;
    d3 = (d3 + 1) >> 1;
  }
  out[0] = (int16_t)d0;
  out[1] = (int16_t)d1;
  out[2] = (int16_t)d2;
  out[3] = (int16_t)d3;
}

// Scalar butterfly rotation for the DC-only ADST-8 path (mirrors
// ButterflyRotation_C / FirstIsZero in i32 domain, rounding each stage).
inline void ScalarButterfly(int32_t t[8], int a, int b, int angle, bool flip,
                            bool a_is_zero) {
  int32_t x, y;
  if (a_is_zero) {
    x = t[b] * -Sin128(angle);
    y = t[b] * Cos128(angle);
  } else {
    x = t[a] * Cos128(angle) - t[b] * Sin128(angle);
    y = t[a] * Sin128(angle) + t[b] * Cos128(angle);
  }
  t[a] = R12(flip ? y : x);
  t[b] = R12(flip ? x : y);
}

// Single-lane DC-only ADST-8 (scalar, bit-exact vs Adst8DcOnly_C).
inline void Adst8DcLane(int16_t v, bool should_round, int row_shift,
                        int16_t out[8]) {
  int32_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  t[1] = v;
  if (should_round) t[1] = (int32_t(t[1]) * 2896 + 2048) >> 12;
  // stage 2 (a=0 is zero).
  ScalarButterfly(t, 0, 1, 60, true, true);
  // stage 3.
  t[4] = t[0];
  t[5] = t[1];
  // stage 4.
  ScalarButterfly(t, 4, 5, 48, true, false);
  // stage 5.
  t[2] = t[0];
  t[3] = t[1];
  t[6] = t[4];
  t[7] = t[5];
  // stage 6.
  ScalarButterfly(t, 2, 3, 32, true, false);
  ScalarButterfly(t, 6, 7, 32, true, false);
  // stage 7: output permutation.
  int32_t o[8] = {t[0], -t[4], t[6], -t[2], t[3], -t[7], t[5], -t[1]};
  for (int i = 0; i < 8; ++i) {
    o[i] -= (o[i] == 0x8000);
    if (row_shift > 0) o[i] = (o[i] + (1 << (row_shift - 1))) >> row_shift;
    out[i] = (int16_t)o[i];
  }
}

void Adst4Row(TransformType, TransformSize tx_size, int h, void* src_buffer,
              int, int, void*) {
  auto* src = static_cast<int16_t*>(src_buffer);
  const int tx_height = kTransformHeight[tx_size];
  const int tx_width = kTransformWidth[tx_size];
  const bool should_round = (tx_height == 8);
  const int row_shift = (tx_height == 16) ? 1 : 0;
  if (h == 1) {
    int16_t out[4];
    Adst4DcLane(src[0], should_round, row_shift, out);
    for (int j = 0; j < tx_width; ++j) src[j] = out[j];
    return;
  }
  if (should_round) ApplyRowRounding(src, size_t(h) * tx_width);
  size_t done = 0;
  while (done < size_t(h)) {
    size_t vm = __riscv_vsetvl_e16m1(size_t(h) - done);
    int16_t* bp = src + done * 4;
    vint16m1x4_t t = __riscv_vlseg4e16_v_i16m1x4(bp, vm);
    vint16m1_t x0 = __riscv_vget_v_i16m1x4_i16m1(t, 0);
    vint16m1_t x1 = __riscv_vget_v_i16m1x4_i16m1(t, 1);
    vint16m1_t x2 = __riscv_vget_v_i16m1x4_i16m1(t, 2);
    vint16m1_t x3 = __riscv_vget_v_i16m1x4_i16m1(t, 3);
    Adst4(x0, x1, x2, x3, vm);
    __riscv_vsseg4e16_v_i16m1x4(
        bp, __riscv_vcreate_v_i16m1x4(x0, x1, x2, x3), vm);
    done += vm;
  }
  if (row_shift > 0) RoundShiftRight(src, size_t(h) * 4, row_shift);
}

void Adst4Column(TransformType tx_type, TransformSize tx_size, int h,
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
  if (h == 1) {
    for (int j = 0; j < tx_width; ++j) {
      int16_t out[4];
      Adst4DcLane(src[j], /*should_round=*/false, /*row_shift=*/0, out);
      for (int y = 0; y < 4; ++y) src[y * tx_width + j] = out[y];
    }
  } else {
    size_t done = 0;
    while (done < size_t(tx_width)) {
      size_t vm = __riscv_vsetvl_e16m1(size_t(tx_width) - done);
      int16_t* cp = src + done;
      vint16m1_t x0 = __riscv_vle16_v_i16m1(cp, vm);
      vint16m1_t x1 = __riscv_vle16_v_i16m1(cp + tx_width, vm);
      vint16m1_t x2 = __riscv_vle16_v_i16m1(cp + 2 * tx_width, vm);
      vint16m1_t x3 = __riscv_vle16_v_i16m1(cp + 3 * tx_width, vm);
      Adst4(x0, x1, x2, x3, vm);
      __riscv_vse16_v_i16m1(cp, x0, vm);
      __riscv_vse16_v_i16m1(cp + tx_width, x1, vm);
      __riscv_vse16_v_i16m1(cp + 2 * tx_width, x2, vm);
      __riscv_vse16_v_i16m1(cp + 3 * tx_width, x3, vm);
      done += vm;
    }
  }
  auto& frame = *static_cast<Array2DView<uint8_t>*>(dst_frame);
  // ADST column slots DO flip rows (unlike DCT): the C reference enables
  // flip_rows when transform1d_type == kTransform1dAdst.
  const bool flip_rows = kTransformFlipRowsMask.Contains(tx_type);
  for (int i = 0; i < tx_height; ++i) {
    const int row = (flip_rows ? (tx_height - i - 1) : i) * tx_width;
    StoreToFrameRow(src + row, frame[start_y + i] + start_x, tx_width);
  }
}

void Adst8Row(TransformType, TransformSize tx_size, int h, void* src_buffer,
              int, int, void*) {
  auto* src = static_cast<int16_t*>(src_buffer);
  const int tx_width = kTransformWidth[tx_size];
  const bool should_round = kShouldRound[tx_size];
  const int row_shift = kTransformRowShift[tx_size];
  if (h == 1) {
    int16_t out[8];
    Adst8DcLane(src[0], should_round, row_shift, out);
    for (int j = 0; j < tx_width; ++j) src[j] = out[j];
    return;
  }
  if (should_round) ApplyRowRounding(src, size_t(h) * tx_width);
  // Input permutation [x7,x0,x5,x2,x3,x4,x1,x6] applied to the vlseg8 fields
  // (vget requires compile-time indices).
  size_t done = 0;
  while (done < size_t(h)) {
    size_t vm = __riscv_vsetvl_e16m1(size_t(h) - done);
    int16_t* bp = src + done * 8;
    vint16m1x8_t t = __riscv_vlseg8e16_v_i16m1x8(bp, vm);
    vint16m1_t s0 = __riscv_vget_v_i16m1x8_i16m1(t, 7);
    vint16m1_t s1 = __riscv_vget_v_i16m1x8_i16m1(t, 0);
    vint16m1_t s2 = __riscv_vget_v_i16m1x8_i16m1(t, 5);
    vint16m1_t s3 = __riscv_vget_v_i16m1x8_i16m1(t, 2);
    vint16m1_t s4 = __riscv_vget_v_i16m1x8_i16m1(t, 3);
    vint16m1_t s5 = __riscv_vget_v_i16m1x8_i16m1(t, 4);
    vint16m1_t s6 = __riscv_vget_v_i16m1x8_i16m1(t, 1);
    vint16m1_t s7 = __riscv_vget_v_i16m1x8_i16m1(t, 6);
    Adst8(s0, s1, s2, s3, s4, s5, s6, s7, vm);
    __riscv_vsseg8e16_v_i16m1x8(
        bp, __riscv_vcreate_v_i16m1x8(s0, s1, s2, s3, s4, s5, s6, s7), vm);
    done += vm;
  }
  if (row_shift > 0) RoundShiftRight(src, size_t(h) * 8, row_shift);
}

void Adst8Column(TransformType tx_type, TransformSize tx_size, int h,
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
  if (h == 1) {
    for (int j = 0; j < tx_width; ++j) {
      int16_t out[8];
      Adst8DcLane(src[j], /*should_round=*/false, /*row_shift=*/0, out);
      for (int y = 0; y < 8; ++y) src[y * tx_width + j] = out[y];
    }
  } else {
    static const int kIn[8] = {7, 0, 5, 2, 3, 4, 1, 6};
    size_t done = 0;
    while (done < size_t(tx_width)) {
      size_t vm = __riscv_vsetvl_e16m1(size_t(tx_width) - done);
      int16_t* cp = src + done;
      vint16m1_t s0 = __riscv_vle16_v_i16m1(cp + kIn[0] * tx_width, vm);
      vint16m1_t s1 = __riscv_vle16_v_i16m1(cp + kIn[1] * tx_width, vm);
      vint16m1_t s2 = __riscv_vle16_v_i16m1(cp + kIn[2] * tx_width, vm);
      vint16m1_t s3 = __riscv_vle16_v_i16m1(cp + kIn[3] * tx_width, vm);
      vint16m1_t s4 = __riscv_vle16_v_i16m1(cp + kIn[4] * tx_width, vm);
      vint16m1_t s5 = __riscv_vle16_v_i16m1(cp + kIn[5] * tx_width, vm);
      vint16m1_t s6 = __riscv_vle16_v_i16m1(cp + kIn[6] * tx_width, vm);
      vint16m1_t s7 = __riscv_vle16_v_i16m1(cp + kIn[7] * tx_width, vm);
      Adst8(s0, s1, s2, s3, s4, s5, s6, s7, vm);
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
  const bool flip_rows = kTransformFlipRowsMask.Contains(tx_type);
  for (int i = 0; i < tx_height; ++i) {
    const int row = (flip_rows ? (tx_height - i - 1) : i) * tx_width;
    StoreToFrameRow(src + row, frame[start_y + i] + start_x, tx_width);
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
  dsp->inverse_transforms[kTransform1dAdst][kTransform1dSize4][kRow] = Adst4Row;
  dsp->inverse_transforms[kTransform1dAdst][kTransform1dSize4][kColumn] = Adst4Column;
  dsp->inverse_transforms[kTransform1dAdst][kTransform1dSize8][kRow] = Adst8Row;
  dsp->inverse_transforms[kTransform1dAdst][kTransform1dSize8][kColumn] = Adst8Column;
}

}  // namespace dsp
}  // namespace libgav1


