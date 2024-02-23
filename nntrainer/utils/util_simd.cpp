// SPDX-License-Identifier: Apache-2.0
/**
 * @file	util_simd.cpp
 * @date	09 Jan 2024
 * @brief	This is a collection of simd util functions
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <cmath>
#include <util_simd.h>
#ifdef USE_NEON
#include <util_simd_neon.h>
#endif

namespace nntrainer {


constexpr auto compute::the_instance = nullptr;

compute::compute() {
} // Do nothing

compute::~compute() {
  // Do nothing
}

void compute::calc_trigonometric_vals_dup(unsigned int N_half, float *angles,
    float *cos_, float *sin_, unsigned int alpha) {
  for (unsigned int j = 0; j < N_half; ++j) {
    float angle = alpha * angles[j];
    (*cos)[alpha][j] = std::cos(angle);
    (*cos)[alpha][j + N_half] = std::cos(angle); // repeated 2 times

    (*sin)[alpha][j] = std::sin(angle);
    (*sin)[alpha][j + N_half] = std::sin(angle); // repeated 2 times
  }
}

void compute::swish(const unsigned int N, float *X, float *Y, float *Z)
{
  unsigned int i = 0;
  while (i < N) {
    X[i] = (Y[i] / (1.f + std::exp(-Y[i]))) * Z[i];
    ++i;
  }
}

}

#ifdef ENABLE_FP16

void compute::compute_rotary_embedding_value(unsigned int dim, unsigned int half_,
                                    unsigned int w, _FP16 *in, _FP16 *out,
                                    float *cos_, float *sin_) {
  throw std::invalid_argument(
    "Error: No implementation of rotary embedding layer incremental_forwarding "
    "with SIMD acceleration except for NEON!");
}

void compute::swish(const unsigned int N, _FP16 *X, _FP16 *Y, _FP16 *Z) {
  unsigned int i = 0;
  while (i < N) {
    X[i] =
      (Y[i] / static_cast<_FP16>(1.f + std::exp(static_cast<float>(-Y[i])))) *
      Z[i];
    ++i;
  }
}
#endif

compute *compute::get_instance()
{
  // Return the derived instance of the proper arch.
  if (the_instance)
    return the_instance;
#ifdef USE_NEON
  the_instance = new compute_neon();
#elif USE_AVX
  the_instance = new compute_avx();
#elif USE_RVV
  the_instance = new compute_rvv();
#else
  the_instance = new compute();
#endif
  return the_instance;
}

} // namespace nntrainer
