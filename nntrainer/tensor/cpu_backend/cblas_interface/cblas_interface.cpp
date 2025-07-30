// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file cblas_interface.cpp
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Function interface to use cblas lib from cpu_backend with Windows x64 optimizations
 *
 */

#include <cblas.h>
#include <cblas_interface.h>
#include <algorithm>
#include <thread>

namespace nntrainer {

// Windows x64 optimization: Intelligent threading for transformer workloads
namespace x64_threading {
    // Based on windows-native.ini: openblas-num-threads = 6
    constexpr int MAX_THREADS = 6;
    constexpr int MIN_ELEMENTS_PER_THREAD = 4096; // Minimum work per thread
    
    // Transformer workload characteristics
    constexpr size_t ATTENTION_THRESHOLD = 512 * 768;     // BERT-base attention
    constexpr size_t FEEDFORWARD_THRESHOLD = 768 * 3072;  // BERT-base FF
    constexpr size_t LARGE_MODEL_THRESHOLD = 1024 * 4096; // GPT-2 large
    
    static inline int get_optimal_threads_for_sgemm(unsigned int M, unsigned int N, unsigned int K) {
        size_t total_ops = static_cast<size_t>(M) * N * K * 2; // 2 ops per fma
        
        // For very small matrices (attention heads), use fewer threads
        if (total_ops < MIN_ELEMENTS_PER_THREAD * 2) {
            return 1;
        }
        
        // Attention QKV projections: typically square-ish matrices
        if (M <= 512 && N <= 1024 && K <= 1024) {
            return std::min(3, MAX_THREADS); // 3 threads for attention
        }
        
        // Feed-forward layers: larger, asymmetric matrices
        if (N >= 3072 || total_ops >= FEEDFORWARD_THRESHOLD) {
            return MAX_THREADS; // Use all 6 threads for FF layers
        }
        
        // Large transformer layers
        if (total_ops >= LARGE_MODEL_THRESHOLD) {
            return MAX_THREADS;
        }
        
        // Default: scale with problem size
        int optimal = static_cast<int>(total_ops / MIN_ELEMENTS_PER_THREAD);
        return std::min(std::max(optimal, 1), MAX_THREADS);
    }
    
    static inline int get_optimal_threads_for_vector(unsigned int N) {
        // Vector operations are memory-bound, use moderate threading
        if (N < MIN_ELEMENTS_PER_THREAD) {
            return 1;
        }
        
        // For transformer sequence lengths (512, 768, 1024, 2048)
        if (N >= 2048) {
            return std::min(4, MAX_THREADS);
        } else if (N >= 768) {
            return std::min(3, MAX_THREADS);
        } else {
            return std::min(2, MAX_THREADS);
        }
    }
}

void __cblas_saxpy(const unsigned int N, const float alpha, const float *X,
                   const unsigned int incX, float *Y, const unsigned int incY) {
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#else
  // Windows x64 optimization: Vector-specific threading
  int threads = x64_threading::get_optimal_threads_for_vector(N);
  openblas_set_num_threads(threads);
#endif
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}

void __cblas_sgemv(const unsigned int TStorageOrder, bool TransA,
                   const unsigned int M, const unsigned int N,
                   const float alpha, const float *A, const unsigned int lda,
                   const float *X, const unsigned int incX, const float beta,
                   float *Y, const unsigned int incY) {
  CBLAS_TRANSPOSE transA = TransA ? CblasTrans : CblasNoTrans;
  CBLAS_ORDER order = TStorageOrder ? CblasColMajor : CblasRowMajor;
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#else
  // Windows x64 optimization: Matrix-vector specific threading
  size_t total_elements = static_cast<size_t>(M) * N;
  int threads = x64_threading::get_optimal_threads_for_vector(total_elements);
  openblas_set_num_threads(threads);
#endif
  cblas_sgemv(order, transA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

float __cblas_sdot(const unsigned int N, const float *X,
                   const unsigned int incX, const float *Y,
                   const unsigned int incY) {
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#else
  // Windows x64 optimization: Dot products benefit from fewer threads due to reduction
  int threads = std::min(x64_threading::get_optimal_threads_for_vector(N), 3);
  openblas_set_num_threads(threads);
#endif
  return cblas_sdot(N, X, incX, Y, incY);
}

void __cblas_scopy(const unsigned int N, const float *X,
                   const unsigned int incX, float *Y, const unsigned int incY) {
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#else
  // Windows x64 optimization: Memory bandwidth limited, moderate threading
  int threads = x64_threading::get_optimal_threads_for_vector(N);
  openblas_set_num_threads(threads);
#endif
  cblas_scopy(N, X, incX, Y, incY);
}

void __cblas_sscal(const unsigned int N, const float alpha, float *X,
                   const unsigned int incX) {
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#else
  // Windows x64 optimization: Scaling is memory-bound
  int threads = x64_threading::get_optimal_threads_for_vector(N);
  openblas_set_num_threads(threads);
#endif
  cblas_sscal(N, alpha, X, incX);
}

float __cblas_snrm2(const unsigned int N, const float *X,
                    const unsigned int incX) {
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#else
  // Windows x64 optimization: Norm involves reduction, limit threads
  int threads = std::min(x64_threading::get_optimal_threads_for_vector(N), 4);
  openblas_set_num_threads(threads);
#endif
  return cblas_snrm2(N, X, incX);
}

void __cblas_sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
                   const unsigned int M, const unsigned int N,
                   const unsigned int K, const float alpha, const float *A,
                   const unsigned int lda, const float *B,
                   const unsigned int ldb, const float beta, float *C,
                   const unsigned int ldc) {
  CBLAS_TRANSPOSE transA = TransA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB = TransB ? CblasTrans : CblasNoTrans;
  CBLAS_ORDER order = TStorageOrder ? CblasColMajor : CblasRowMajor;
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#else
  // Windows x64 optimization: Transformer-aware SGEMM threading
  int threads = x64_threading::get_optimal_threads_for_sgemm(M, N, K);
  openblas_set_num_threads(threads);
#endif
  cblas_sgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
}

unsigned int __cblas_isamax(const unsigned int N, const float *X,
                            const unsigned int incX) {
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#else
  // Windows x64 optimization: Max-finding is reduction operation
  int threads = std::min(x64_threading::get_optimal_threads_for_vector(N), 3);
  openblas_set_num_threads(threads);
#endif
  return cblas_isamax(N, X, incX);
}
} // namespace nntrainer
