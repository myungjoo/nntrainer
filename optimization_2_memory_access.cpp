/**
 * Optimization 2: Memory Access Pattern Optimization
 * 
 * This optimization improves memory coalescing, cache line utilization,
 * and implements NEON intrinsics for vectorized operations.
 * 
 * Expected Performance Gain: 2-3x improvement in memory bandwidth utilization
 */

#include <arm_neon.h>
#include <algorithm>
#include <cstring>
#include <memory>

// Cache-friendly blocked matrix multiplication with NEON
void fp16_gemm_blocked_neon(const __fp16* A, const __fp16* B, __fp16* C,
                           int M, int N, int K) {
    constexpr int BLOCK_SIZE = 64;  // Optimized for L1 cache (32KB)
    constexpr int VECTOR_SIZE = 8;   // NEON float16x8_t
    
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int bi = 0; bi < M; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
                int block_m = std::min(BLOCK_SIZE, M - bi);
                int block_n = std::min(BLOCK_SIZE, N - bj);
                int block_k = std::min(BLOCK_SIZE, K - bk);
                
                // Process block with NEON intrinsics
                for (int i = 0; i < block_m; i++) {
                    for (int j = 0; j < block_n; j += VECTOR_SIZE) {
                        if (j + VECTOR_SIZE <= block_n) {
                            float16x8_t c_vec = vld1q_f16(&C[(bi + i) * N + bj + j]);
                            
                            for (int k = 0; k < block_k; k++) {
                                float16x8_t a_vec = vdupq_n_f16(A[(bi + i) * K + bk + k]);
                                float16x8_t b_vec = vld1q_f16(&B[(bk + k) * N + bj + j]);
                                c_vec = vfmaq_f16(c_vec, a_vec, b_vec);
                            }
                            
                            vst1q_f16(&C[(bi + i) * N + bj + j], c_vec);
                        } else {
                            // Handle remaining elements
                            for (int jj = j; jj < block_n; jj++) {
                                __fp16 sum = C[(bi + i) * N + bj + jj];
                                for (int k = 0; k < block_k; k++) {
                                    sum += A[(bi + i) * K + bk + k] * B[(bk + k) * N + bj + jj];
                                }
                                C[(bi + i) * N + bj + jj] = sum;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Prefetch-optimized version with software prefetching
void fp16_gemm_prefetch(const __fp16* A, const __fp16* B, __fp16* C,
                       int M, int N, int K) {
    constexpr int PREFETCH_DISTANCE = 512;
    constexpr int UNROLL_FACTOR = 4;
    
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        // Prefetch next cache lines
        if (i + 1 < M) {
            __builtin_prefetch(&A[(i + 1) * K], 0, 3);
        }
        
        for (int j = 0; j < N; j += 8) {
            // Prefetch B matrix data
            if (j + PREFETCH_DISTANCE < N) {
                __builtin_prefetch(&B[0 * N + j + PREFETCH_DISTANCE], 0, 3);
            }
            
            if (j + 8 <= N) {
                float16x8_t c_vec = vld1q_f16(&C[i * N + j]);
                
                for (int k = 0; k < K; k += UNROLL_FACTOR) {
                    int k_limit = std::min(k + UNROLL_FACTOR, K);
                    
                    for (int kk = k; kk < k_limit; kk++) {
                        float16x8_t a_vec = vdupq_n_f16(A[i * K + kk]);
                        float16x8_t b_vec = vld1q_f16(&B[kk * N + j]);
                        c_vec = vfmaq_f16(c_vec, a_vec, b_vec);
                    }
                }
                
                vst1q_f16(&C[i * N + j], c_vec);
            } else {
                // Handle remaining elements
                for (int jj = j; jj < N; jj++) {
                    __fp16 sum = C[i * N + jj];
                    for (int k = 0; k < K; k++) {
                        sum += A[i * K + k] * B[k * N + jj];
                    }
                    C[i * N + jj] = sum;
                }
            }
        }
    }
}

// Memory-aligned matrix operations
class AlignedMatrixOps {
private:
    static constexpr size_t CACHE_LINE_SIZE = 64;
    static constexpr size_t ALIGNMENT = 32;  // 256-bit alignment for NEON
    
public:
    // Allocate aligned memory for optimal SIMD access
    template<typename T>
    static T* allocate_aligned(size_t count) {
        void* ptr = nullptr;
        size_t bytes = count * sizeof(T);
        
        if (posix_memalign(&ptr, ALIGNMENT, bytes) != 0) {
            throw std::bad_alloc();
        }
        
        return static_cast<T*>(ptr);
    }
    
    // Copy matrix with optimal memory access pattern
    static void copy_matrix_optimized(const __fp16* src, __fp16* dst, 
                                    int rows, int cols, int src_stride, int dst_stride) {
        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            const __fp16* src_row = &src[i * src_stride];
            __fp16* dst_row = &dst[i * dst_stride];
            
            int j = 0;
            // Vectorized copy
            for (; j + 8 <= cols; j += 8) {
                float16x8_t data = vld1q_f16(&src_row[j]);
                vst1q_f16(&dst_row[j], data);
            }
            
            // Handle remaining elements
            for (; j < cols; j++) {
                dst_row[j] = src_row[j];
            }
        }
    }
    
    // Transpose matrix with cache-friendly access
    static void transpose_fp16(const __fp16* src, __fp16* dst, int M, int N) {
        constexpr int TILE_SIZE = 32;
        
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < M; i += TILE_SIZE) {
            for (int j = 0; j < N; j += TILE_SIZE) {
                int i_end = std::min(i + TILE_SIZE, M);
                int j_end = std::min(j + TILE_SIZE, N);
                
                // Transpose tile
                for (int ii = i; ii < i_end; ii++) {
                    for (int jj = j; jj < j_end; jj += 8) {
                        if (jj + 8 <= j_end) {
                            float16x8_t data = vld1q_f16(&src[ii * N + jj]);
                            
                            // Store transposed (this is a simplified version)
                            for (int k = 0; k < 8 && jj + k < j_end; k++) {
                                dst[(jj + k) * M + ii] = vgetq_lane_f16(data, k);
                            }
                        } else {
                            for (int jjj = jj; jjj < j_end; jjj++) {
                                dst[jjj * M + ii] = src[ii * N + jjj];
                            }
                        }
                    }
                }
            }
        }
    }
};

// Cache-aware GEMM with multiple optimization levels
void fp16_gemm_cache_aware(const __fp16* A, const __fp16* B, __fp16* C,
                          int M, int N, int K) {
    // L1 cache optimized blocking
    constexpr int L1_BLOCK = 64;
    // L2 cache optimized blocking  
    constexpr int L2_BLOCK = 256;
    
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < M; ii += L2_BLOCK) {
        for (int jj = 0; jj < N; jj += L2_BLOCK) {
            for (int kk = 0; kk < K; kk += L2_BLOCK) {
                // L2 block boundaries
                int i_end = std::min(ii + L2_BLOCK, M);
                int j_end = std::min(jj + L2_BLOCK, N);
                int k_end = std::min(kk + L2_BLOCK, K);
                
                // L1 blocking within L2 blocks
                for (int i = ii; i < i_end; i += L1_BLOCK) {
                    for (int j = jj; j < j_end; j += L1_BLOCK) {
                        for (int k = kk; k < k_end; k += L1_BLOCK) {
                            fp16_gemm_blocked_neon(&A[i * K + k], &B[k * N + j], &C[i * N + j],
                                                  std::min(L1_BLOCK, i_end - i),
                                                  std::min(L1_BLOCK, j_end - j),
                                                  std::min(L1_BLOCK, k_end - k));
                        }
                    }
                }
            }
        }
    }
}

// Streaming memory access for large matrices
void fp16_gemm_streaming(const __fp16* A, const __fp16* B, __fp16* C,
                        int M, int N, int K) {
    constexpr int STREAM_BLOCK = 128;
    
    #pragma omp parallel for
    for (int i = 0; i < M; i += STREAM_BLOCK) {
        int i_end = std::min(i + STREAM_BLOCK, M);
        
        for (int ii = i; ii < i_end; ii++) {
            // Prefetch next row
            if (ii + 1 < i_end) {
                __builtin_prefetch(&A[(ii + 1) * K], 0, 1);
            }
            
            for (int j = 0; j < N; j += 8) {
                if (j + 8 <= N) {
                    float16x8_t c_vec = vdupq_n_f16(0);
                    
                    for (int k = 0; k < K; k++) {
                        // Non-temporal hint for large matrices
                        __builtin_prefetch(&B[(k + 64) * N + j], 0, 0);
                        
                        float16x8_t a_vec = vdupq_n_f16(A[ii * K + k]);
                        float16x8_t b_vec = vld1q_f16(&B[k * N + j]);
                        c_vec = vfmaq_f16(c_vec, a_vec, b_vec);
                    }
                    
                    // Non-temporal store for streaming
                    vst1q_f16(&C[ii * N + j], c_vec);
                } else {
                    // Handle remaining elements
                    for (int jj = j; jj < N; jj++) {
                        __fp16 sum = 0;
                        for (int k = 0; k < K; k++) {
                            sum += A[ii * K + k] * B[k * N + jj];
                        }
                        C[ii * N + jj] = sum;
                    }
                }
            }
        }
    }
}

// Benchmark function for memory access optimizations
#include <chrono>
#include <iostream>
#include <iomanip>

void benchmark_memory_access() {
    const int M = 1024, N = 1024, K = 1024;
    
    // Allocate aligned matrices
    auto A = std::unique_ptr<__fp16[], decltype(&free)>(
        AlignedMatrixOps::allocate_aligned<__fp16>(M * K), &free);
    auto B = std::unique_ptr<__fp16[], decltype(&free)>(
        AlignedMatrixOps::allocate_aligned<__fp16>(K * N), &free);
    auto C1 = std::unique_ptr<__fp16[], decltype(&free)>(
        AlignedMatrixOps::allocate_aligned<__fp16>(M * N), &free);
    auto C2 = std::unique_ptr<__fp16[], decltype(&free)>(
        AlignedMatrixOps::allocate_aligned<__fp16>(M * N), &free);
    auto C3 = std::unique_ptr<__fp16[], decltype(&free)>(
        AlignedMatrixOps::allocate_aligned<__fp16>(M * N), &free);
    auto C4 = std::unique_ptr<__fp16[], decltype(&free)>(
        AlignedMatrixOps::allocate_aligned<__fp16>(M * N), &free);
    
    // Initialize matrices
    for (int i = 0; i < M * K; i++) {
        A[i] = (__fp16)(rand() / (float)RAND_MAX - 0.5f);
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (__fp16)(rand() / (float)RAND_MAX - 0.5f);
    }
    
    std::cout << std::fixed << std::setprecision(2);
    
    // Benchmark blocked NEON implementation
    auto start = std::chrono::high_resolution_clock::now();
    fp16_gemm_blocked_neon(A.get(), B.get(), C1.get(), M, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Benchmark prefetch implementation
    start = std::chrono::high_resolution_clock::now();
    fp16_gemm_prefetch(A.get(), B.get(), C2.get(), M, N, K);
    end = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Benchmark cache-aware implementation
    start = std::chrono::high_resolution_clock::now();
    fp16_gemm_cache_aware(A.get(), B.get(), C3.get(), M, N, K);
    end = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Benchmark streaming implementation
    start = std::chrono::high_resolution_clock::now();
    fp16_gemm_streaming(A.get(), B.get(), C4.get(), M, N, K);
    end = std::chrono::high_resolution_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double gflops = (2.0 * M * N * K) / 1e9;
    
    std::cout << "Matrix size: " << M << "x" << N << "x" << K << "\n";
    std::cout << "Blocked NEON:  " << duration1.count() << " μs, " 
              << gflops / (duration1.count() / 1e6) << " GFLOPS\n";
    std::cout << "Prefetch:      " << duration2.count() << " μs, " 
              << gflops / (duration2.count() / 1e6) << " GFLOPS\n";
    std::cout << "Cache-aware:   " << duration3.count() << " μs, " 
              << gflops / (duration3.count() / 1e6) << " GFLOPS\n";
    std::cout << "Streaming:     " << duration4.count() << " μs, " 
              << gflops / (duration4.count() / 1e6) << " GFLOPS\n";
}