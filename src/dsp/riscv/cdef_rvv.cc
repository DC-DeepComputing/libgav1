// Copyright 2026 RISC-V RVV optimization work.
//
// RVV CDEF filter kernels (8bpp). Lanes run along the row direction;
// the constrain/clip arithmetic mirrors dsp/x86/cdef_sse4.cc, which is
// bit-exact vs the scalar C reference (integer ops, no reordering).
#include <riscv_vector.h>

#include <cstdio>

#include <algorithm>

#include "src/dsp/cdef.h"
#include "src/dsp/constants.h"
#include "src/dsp/dsp.h"
#include "src/utils/common.h"

namespace libgav1 {
namespace dsp {
namespace {

#include "src/dsp/cdef.inc"

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
              if (*p == 'v') {
                ok = 1;
                break;
              }
              p++;
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

constexpr int kRVVL = 8;  // fixed 8 lanes

inline vuint16m1_t LoadRow8(const uint16_t* p, size_t vl) {
  return __riscv_vle16_v_u16m1(p, vl);
}

// width==4: lanes 0..3 from row y, lanes 4..7 from row y+1.
inline vuint16m1_t LoadRow4(const uint16_t* p, ptrdiff_t stride, size_t vl) {
  vuint16m1_t v = __riscv_vlmul_ext_v_u16mf2_u16m1(
      __riscv_vle16_v_u16mf2(p, 4));
  vuint16m1_t hi = __riscv_vlmul_ext_v_u16mf2_u16m1(
      __riscv_vle16_v_u16mf2(p + stride, 4));
  return __riscv_vslideup_vx_u16m1(v, hi, 4, vl);
}

// constrain(val vs ref) * tap accumulated into sum (i16).
inline void ConstrainTap(vuint16m1_t val, vuint16m1_t ref, int threshold,
                         int shift, int tap, vint16m1_t* sum, size_t vl) {
  const vint16m1_t diff =
      __riscv_vsub_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(val),
                             __riscv_vreinterpret_v_u16m1_i16m1(ref), vl);
  const vuint16m1_t abs = __riscv_vreinterpret_v_i16m1_u16m1(
      __riscv_vmax_vv_i16m1(diff, __riscv_vneg_v_i16m1(diff, vl), vl));
  const vuint16m1_t sh = __riscv_vsrl_vx_u16m1(abs, (unsigned)shift, vl);
  const vuint16m1_t thr = __riscv_vmv_v_x_u16m1((uint16_t)threshold, vl);
  const vuint16m1_t t = __riscv_vssubu_vv_u16m1(thr, sh, vl);
  const vuint16m1_t clamp = __riscv_vminu_vv_u16m1(t, abs, vl);
  const vbool16_t neg = __riscv_vmslt_vx_i16m1_b16(
      diff, 0, vl);
  vint16m1_t c = __riscv_vreinterpret_v_u16m1_i16m1(clamp);
  c = __riscv_vmerge_vvm_i16m1(
      c, __riscv_vneg_v_i16m1(c, vl), neg, vl);
  *sum = __riscv_vadd_vv_i16m1(
      *sum, __riscv_vmul_vx_i16m1(c, (int16_t)tap, vl), vl);
}

template <int width, bool enable_primary, bool enable_secondary>
void CdefFilter_RVV(const uint16_t* LIBGAV1_RESTRICT src,
                    const ptrdiff_t src_stride, const int block_height,
                    const int primary_strength, const int secondary_strength,
                    const int damping, const int direction,
                    void* LIBGAV1_RESTRICT const dest,
                    const ptrdiff_t dest_stride) {
  constexpr bool clipping_required = enable_primary && enable_secondary;
  constexpr int kSecTap[2] = {kCdefSecondaryTap0, kCdefSecondaryTap1};

  const int p_shift =
      std::max(0, damping - FloorLog2((int32_t)primary_strength));
  const int s_shift = damping - FloorLog2((int32_t)secondary_strength);
  const int p_thr = primary_strength;
  const int s_thr = secondary_strength;
  const int p_tap0 = kCdefPrimaryTaps[primary_strength & 1][0];
  const int p_tap1 = kCdefPrimaryTaps[primary_strength & 1][1];

  const size_t vl = kRVVL;
  auto* dst = static_cast<uint8_t*>(dest);
  const ptrdiff_t dst_stride = dest_stride;

  int y = block_height;
  do {
    vuint16m1_t pixel = (width == 8) ? LoadRow8(src, vl)
                                     : LoadRow4(src, src_stride, vl);
    vuint16m1_t minv = pixel;
    vuint16m1_t maxv = pixel;
    vint16m1_t sum = __riscv_vmv_v_x_i16m1(0, vl);

    // primary taps: 2 directions x 2 signs
    if (enable_primary) {
      int t = 0;
      do {
        const int tap = t ? p_tap1 : p_tap0;
        int s = 0;
        do {
          const int sign = s ? 1 : -1;
          const int dy = sign * kCdefDirections[direction][t][0];
          const int dx = sign * kCdefDirections[direction][t][1];
          vuint16m1_t val = (width == 8)
                                ? LoadRow8(src + dy * src_stride + dx, vl)
                                : LoadRow4(src + dy * src_stride + dx,
                                           src_stride, vl);
          if (clipping_required) {
            minv = __riscv_vminu_vv_u16m1(minv, val, vl);
            const vuint16m1_t lo8 =
                __riscv_vand_vx_u16m1(val, 0xFF, vl);
            maxv = __riscv_vmaxu_vv_u16m1(maxv, lo8, vl);
          }
          ConstrainTap(val, pixel, p_thr, p_shift, tap, &sum, vl);
          s ^= 1;
        } while (s);
      } while (++t < 2);
    }

    // secondary taps: directions +/- 2, 2 directions x 2 signs x 2 taps
    if (enable_secondary) {
      int d = 0;
      do {
        const int dir = (d == 0) ? direction + 2 : direction - 2;
        int t = 0;
        do {
          int s = 0;
          do {
            const int sign = s ? 1 : -1;
            const int dy = sign * kCdefDirections[dir][t][0];
            const int dx = sign * kCdefDirections[dir][t][1];
            vuint16m1_t val = (width == 8)
                                  ? LoadRow8(src + dy * src_stride + dx, vl)
                                  : LoadRow4(src + dy * src_stride + dx,
                                             src_stride, vl);
            if (clipping_required) {
              minv = __riscv_vminu_vv_u16m1(minv, val, vl);
              const vuint16m1_t lo8 =
                  __riscv_vand_vx_u16m1(val, 0xFF, vl);
              maxv = __riscv_vmaxu_vv_u16m1(maxv, lo8, vl);
            }
            ConstrainTap(val, pixel, s_thr, s_shift, kSecTap[t], &sum, vl);
            s ^= 1;
          } while (s);
        } while (++t < 2);
      } while (++d < 2);
    }

    // offset = (8 + sum - (sum < 0)) >> 4
    vint16m1_t tmp = __riscv_vadd_vx_i16m1(sum, 8, vl);
    const vbool16_t isneg = __riscv_vmslt_vx_i16m1_b16(sum, 0, vl);
    tmp = __riscv_vmerge_vvm_i16m1(tmp, __riscv_vsub_vx_i16m1(tmp, 1, vl),
                                   isneg, vl);
    const vint16m1_t offi = __riscv_vsra_vx_i16m1(tmp, 4, vl);
    const vuint16m1_t off = __riscv_vreinterpret_v_i16m1_u16m1(offi);
    vuint16m1_t out = __riscv_vadd_vv_u16m1(pixel, off, vl);
    if (clipping_required) {
      out = __riscv_vminu_vv_u16m1(out, maxv, vl);
      out = __riscv_vmaxu_vv_u16m1(out, minv, vl);
    }
    const vuint8m1_t out8 = __riscv_vlmul_ext_v_u8mf2_u8m1(
        __riscv_vncvt_x_x_w_u8mf2(out, vl));
    if (width == 8) {
      __riscv_vse8_v_u8m1(dst, out8, vl);
    } else {
      __riscv_vse8_v_u8mf2(dst, __riscv_vlmul_trunc_v_u8m1_u8mf2(out8), 4);
      __riscv_vse8_v_u8mf2(dst + dst_stride,
                           __riscv_vlmul_trunc_v_u8m1_u8mf2(
                               __riscv_vslidedown_vx_u8m1(out8, 4, vl)),
                           4);
    }

    src += src_stride * ((width == 4) ? 2 : 1);
    dst += dst_stride * ((width == 4) ? 2 : 1);
  } while ((y -= (width == 4) ? 2 : 1) > 0);
}

}  // namespace

bool CpuHasRVV() { return HasRVVCpu(); }

void CdefInit_RVV() {
  if (!HasRVVCpu()) return;
  Dsp* const dsp = dsp_internal::GetWritableDspTable(8);
  if (dsp == nullptr) return;
  dsp->cdef_filters[0][0] = CdefFilter_RVV<4, true, true>;
  dsp->cdef_filters[0][1] = CdefFilter_RVV<4, true, false>;
  dsp->cdef_filters[0][2] = CdefFilter_RVV<4, false, true>;
  dsp->cdef_filters[1][0] = CdefFilter_RVV<8, true, true>;
  dsp->cdef_filters[1][1] = CdefFilter_RVV<8, true, false>;
  dsp->cdef_filters[1][2] = CdefFilter_RVV<8, false, true>;
}

}  // namespace dsp
}  // namespace libgav1
