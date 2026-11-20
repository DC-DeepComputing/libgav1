// Copyright 2026 RISC-V RVV optimization work.
//
// RVV 1.0 filter-intra prediction kernel (8bpp). The C reference produces
// the 8 pixels of each 4x2 sub-block from 7 neighborhood pixels with 8
// different 7-tap filters. We compute those 8 outputs as 8 RVV lanes of a
// single multiply-accumulate chain. The filter taps (int8_t[5][8][8], only
// 7 of 8 taps used) are lazily transposed to int16_t[5][7][8] so every tap
// vector loads contiguously with lane == output index.
#include <riscv_vector.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "src/dsp/constants.h"
#include "src/dsp/dsp.h"
#include "src/utils/common.h"
#include "src/utils/constants.h"

namespace libgav1 {
namespace dsp {
namespace {

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
          if (p) {
            p += 4;
            while (*p >= 'a' && *p <= 'z') {
              if (*p == 'v') { ok = 1; break; }
              ++p;
            }
          }
          break;
        }
      }
      fclose(f);
    }
    cached = ok;
  }
  return cached != 0;
}

constexpr int kSubVl = 8;  // one 4x2 sub-block == 8 output lanes

// Lazily transposed taps: table[pred][tap][i] = kFilterIntraTaps[pred][i][tap].
// C++11 magic static => thread-safe one-time init.
const int16_t* TransposedTaps(int pred) {
  static int16_t table[kNumFilterIntraPredictors][7][kSubVl];
  static const bool init = [] {
    for (int p = 0; p < kNumFilterIntraPredictors; ++p) {
      for (int k = 0; k < 7; ++k) {
        for (int i = 0; i < kSubVl; ++i) {
          table[p][k][i] = kFilterIntraTaps[p][i][k];
        }
      }
    }
    return true;
  }();
  (void)init;
  return &table[pred][0][0];
}

// value = sum_k taps[k] * p[k] over 8 lanes, then
// Clip3(RightShiftWithRoundingSigned(value, 4), 0, 255). Returns the 8
// clamped outputs as an 8-lane u8 vector (u8mf2). GCC forbids arrays of
// RVV vector types, so the 7 broadcast inputs are named arguments.
inline vuint8mf2_t Filter8(const int16_t* taps, const vint16m1_t p0,
                           const vint16m1_t p1, const vint16m1_t p2,
                           const vint16m1_t p3, const vint16m1_t p4,
                           const vint16m1_t p5, const vint16m1_t p6) {
  vint32m2_t acc = __riscv_vwmul_vv_i32m2(
      __riscv_vle16_v_i16m1(taps + 0 * kSubVl, kSubVl), p0, kSubVl);
  acc = __riscv_vwmacc_vv_i32m2(
      acc, __riscv_vle16_v_i16m1(taps + 1 * kSubVl, kSubVl), p1, kSubVl);
  acc = __riscv_vwmacc_vv_i32m2(
      acc, __riscv_vle16_v_i16m1(taps + 2 * kSubVl, kSubVl), p2, kSubVl);
  acc = __riscv_vwmacc_vv_i32m2(
      acc, __riscv_vle16_v_i16m1(taps + 3 * kSubVl, kSubVl), p3, kSubVl);
  acc = __riscv_vwmacc_vv_i32m2(
      acc, __riscv_vle16_v_i16m1(taps + 4 * kSubVl, kSubVl), p4, kSubVl);
  acc = __riscv_vwmacc_vv_i32m2(
      acc, __riscv_vle16_v_i16m1(taps + 5 * kSubVl, kSubVl), p5, kSubVl);
  acc = __riscv_vwmacc_vv_i32m2(
      acc, __riscv_vle16_v_i16m1(taps + 6 * kSubVl, kSubVl), p6, kSubVl);
  // RightShiftWithRoundingSigned(x, 4) = ((x + (x>>31)) + 8) >> 4.
  const vint32m2_t sign = __riscv_vsra_vx_i32m2(acc, 31, kSubVl);
  acc = __riscv_vsra_vx_i32m2(
      __riscv_vadd_vx_i32m2(__riscv_vadd_vv_i32m2(acc, sign, kSubVl), 8,
                            kSubVl),
      4, kSubVl);
  acc = __riscv_vmin_vx_i32m2(acc, 255, kSubVl);
  acc = __riscv_vmax_vx_i32m2(acc, 0, kSubVl);
  const vint16m1_t acc16 =
      __riscv_vnclip_wx_i16m1(acc, 0, __RISCV_VXRM_RDN, kSubVl);
  return __riscv_vnclipu_wx_u8mf2(
      __riscv_vreinterpret_v_i16m1_u16m1(acc16), 0, __RISCV_VXRM_RDN, kSubVl);
}

void FilterIntraPredictor_RVV(void* LIBGAV1_RESTRICT const dest,
                              ptrdiff_t stride,
                              const void* LIBGAV1_RESTRICT const top_row,
                              const void* LIBGAV1_RESTRICT const left_column,
                              const FilterIntraPredictor pred, const int width,
                              const int height) {
  const auto* const top = static_cast<const uint8_t*>(top_row);
  const auto* const left = static_cast<const uint8_t*>(left_column);

  uint8_t buffer[3][33];  // cache 2 rows + top & left boundaries
  memcpy(buffer[0], &top[-1], (width + 1) * sizeof(top[0]));

  auto* dst = static_cast<uint8_t*>(dest);
  stride /= sizeof(uint8_t);
  const int16_t* taps = TransposedTaps(pred);
  int row0 = 0, row2 = 2;
  int ystep = 1;
  int y = 0;
  do {
    buffer[1][0] = left[y];
    buffer[row2][0] = left[y + 1];
    int x = 1;
    do {
      const int16_t p0 = buffer[row0][x - 1];  // top-left
      const int16_t p1 = buffer[row0][x + 0];  // top 0
      const int16_t p2 = buffer[row0][x + 1];  // top 1
      const int16_t p3 = buffer[row0][x + 2];  // top 2
      const int16_t p4 = buffer[row0][x + 3];  // top 3
      const int16_t p5 = buffer[1][x - 1];     // left 0
      const int16_t p6 = buffer[row2][x - 1];  // left 1
      const vint16m1_t b0 = __riscv_vmv_v_x_i16m1(p0, kSubVl);
      const vint16m1_t b1 = __riscv_vmv_v_x_i16m1(p1, kSubVl);
      const vint16m1_t b2 = __riscv_vmv_v_x_i16m1(p2, kSubVl);
      const vint16m1_t b3 = __riscv_vmv_v_x_i16m1(p3, kSubVl);
      const vint16m1_t b4 = __riscv_vmv_v_x_i16m1(p4, kSubVl);
      const vint16m1_t b5 = __riscv_vmv_v_x_i16m1(p5, kSubVl);
      const vint16m1_t b6 = __riscv_vmv_v_x_i16m1(p6, kSubVl);
      const vuint8mf2_t out = Filter8(taps, b0, b1, b2, b3, b4, b5, b6);
      // Lanes 0..3 -> buffer[1][x..x+3]; lanes 4..7 ->
      // buffer[1+ystep][x..x+3] (row2 when ystep==1, row0 when ystep==-1).
      __riscv_vse8_v_u8mf2(&buffer[1][x], out, 4);
      const vuint8mf2_t hi = __riscv_vslidedown_vx_u8mf2(out, 4, 4);
      __riscv_vse8_v_u8mf2(&buffer[1 + ystep][x], hi, 4);
      x += 4;
    } while (x < width);
    memcpy(dst, &buffer[1][1], width * sizeof(dst[0]));
    dst += stride;
    memcpy(dst, &buffer[row2][1], width * sizeof(dst[0]));
    dst += stride;

    // The final row becomes the top for the next pass.
    row0 ^= 2;
    row2 ^= 2;
    ystep = -ystep;
    y += 2;
  } while (y < height);
}

}  // namespace

void FilterIntraInit_RVV();
void FilterIntraInit_RVV() {
  if (!HasRVVCpu()) return;
  Dsp* const dsp = dsp_internal::GetWritableDspTable(kBitdepth8);
  assert(dsp != nullptr);
  dsp->filter_intra_predictor = FilterIntraPredictor_RVV;
}

}  // namespace dsp
}  // namespace libgav1
