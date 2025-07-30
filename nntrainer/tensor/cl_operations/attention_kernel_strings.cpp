// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Yash Singh <yash.singh@samsung.com>
 *
 * @file	attention_kernel_strings.cpp
 * @date	2 April 2025
 * @brief	All attention OpenCL kernel strings
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Yash Singh <yash.singh@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "attention_kernel_strings.h"

namespace nntrainer {

const std::string &getRotaryEmbClKernel() {
  static const std::string rotary_emb_cl_kernel_ = R"(
  __kernel void rotary_emb_cl(__global float *input,
                                        __global float *output,
                                        __global float *freqs_cos,
                                        __global float *freqs_sin,
                                        __global float *cos_,
                                        __global float *sin_,
                                        unsigned int batch,
                                        unsigned int channel,
                                        unsigned int height,
                                        unsigned int width,
                                        unsigned int dim,
                                        unsigned int half_,
                                        unsigned int max_timestep,
                                        unsigned int from,
                                        unsigned int offsetFreqsSin,
                                        unsigned int offsetSin) {
      
      // Use 2D work group for better parallelization
      unsigned int b = get_global_id(0);
      unsigned int c = get_global_id(1);
      unsigned int local_b = get_local_id(0);
      unsigned int local_c = get_local_id(1);
      
      // Use local memory for cos/sin values to reduce global memory access
      __local float local_cos[256]; // Shared across work group
      __local float local_sin[256];
      
      if(b < batch && c < channel){
        for (unsigned int h = 0; h < height; h++) {
          if (from + h < max_timestep) {
            unsigned idx = (from + h)*dim;
            
            // Cooperative loading of cos/sin values
            for (unsigned int i = local_b; i < dim; i += get_local_size(0)) {
              if (i < 256) { // Safety check for local memory bounds
                local_cos[i] = freqs_cos[idx + i];
                local_sin[i] = freqs_sin[idx + i + offsetFreqsSin];
              }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Process width in vectorized chunks
            for (unsigned int w = 0; w < width; w += dim) {
              // Vectorized processing using float4 where possible
              for (unsigned int k = 0; k < dim; k += 4) {
                if (k + 3 < dim) {
                  // Process 4 elements at once
                  float4 values, transformed_values, cos_vals, sin_vals;
                  
                  unsigned int base_idx = b * channel * height * width + c * height * width + h * width + w;
                  
                  values.x = input[base_idx + k];
                  values.y = input[base_idx + k + 1];
                  values.z = input[base_idx + k + 2];  
                  values.w = input[base_idx + k + 3];
                  
                  // Efficient rotary transformation
                  if (k < half_) {
                    transformed_values.x = -input[base_idx + k + half_];
                    transformed_values.y = -input[base_idx + k + 1 + half_];
                    transformed_values.z = -input[base_idx + k + 2 + half_];
                    transformed_values.w = -input[base_idx + k + 3 + half_];
                  } else {
                    transformed_values.x = input[base_idx + k - half_];
                    transformed_values.y = input[base_idx + k + 1 - half_];
                    transformed_values.z = input[base_idx + k + 2 - half_];
                    transformed_values.w = input[base_idx + k + 3 - half_];
                  }
                  
                  // Use local memory values
                  cos_vals.x = (k < 256) ? local_cos[k] : cos_[k];
                  cos_vals.y = (k + 1 < 256) ? local_cos[k + 1] : cos_[k + 1];
                  cos_vals.z = (k + 2 < 256) ? local_cos[k + 2] : cos_[k + 2];
                  cos_vals.w = (k + 3 < 256) ? local_cos[k + 3] : cos_[k + 3];
                  
                  sin_vals.x = (k < 256) ? local_sin[k] : sin_[k + offsetSin];
                  sin_vals.y = (k + 1 < 256) ? local_sin[k + 1] : sin_[k + 1 + offsetSin];
                  sin_vals.z = (k + 2 < 256) ? local_sin[k + 2] : sin_[k + 2 + offsetSin];
                  sin_vals.w = (k + 3 < 256) ? local_sin[k + 3] : sin_[k + 3 + offsetSin];
                  
                  // Vectorized computation
                  float4 result = values * cos_vals + transformed_values * sin_vals;
                  
                  // Store results
                  output[base_idx + k] = result.x;
                  output[base_idx + k + 1] = result.y;
                  output[base_idx + k + 2] = result.z;
                  output[base_idx + k + 3] = result.w;
                } else {
                  // Handle remaining elements
                  for (unsigned int rem = k; rem < dim; rem++) {
                    unsigned int span = w + rem;
                    float value = input[b * channel * height * width + c * height * width + h * width + span];
                    float transformed_value;
                    
                    if (rem < half_) {
                      transformed_value = -input[b * channel * height * width + c * height * width + h * width + span + half_];
                    } else {
                      transformed_value = input[b * channel * height * width + c * height * width + h * width + span - half_];
                    }
                    
                    float cos_val = (rem < 256) ? local_cos[rem] : cos_[rem];
                    float sin_val = (rem < 256) ? local_sin[rem] : sin_[rem + offsetSin];
                    
                    output[b * channel * height * width + c * height * width + h * width + span] = 
                      value * cos_val + transformed_value * sin_val;
                  }
                }
              }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
          }
        }
      }
  }
  )";
  return rotary_emb_cl_kernel_;
}

#ifdef ENABLE_FP16

const std::string &getRotaryEmbClKernelFP16() {
  static const std::string rotary_emb_cl_kernel_fp16_ = R"(
  
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    
  __kernel void rotary_emb_cl_fp16(__global half *input,
                                        __global half *output,
                                        __global float *freqs_cos,
                                        __global float *freqs_sin,
                                        __global float *cos_,
                                        __global float *sin_,
                                        unsigned int batch,
                                        unsigned int channel,
                                        unsigned int height,
                                        unsigned int width,
                                        unsigned int dim,
                                        unsigned int half_,
                                        unsigned int max_timestep,
                                        unsigned int from,
                                        unsigned int offsetFreqsSin,
                                        unsigned int offsetSin) {
      __global float *cos_ptr = cos_;
      __global float *sin_ptr = sin_;
  
      float value = 0.0f;
      float transformed_value = 0.0f;
  
      unsigned int b = get_global_id(0);
      unsigned int c = get_global_id(1);
      
      if(b < batch && c < channel){
        for (unsigned int h = 0; h < height; h++) {
          if (from + h < max_timestep) {
            unsigned idx = (from + h)*dim;
            for(int i = idx; i < idx + dim; i++ ){
              cos_ptr[i - idx] = freqs_cos[i];
              sin_ptr[i - idx + offsetSin] = freqs_sin[i + offsetFreqsSin];
            }
          }
  
          for (unsigned int w = 0; w < width; w = w + dim) {
            for (unsigned int k = 0; k < dim; k++) {
              unsigned int span = w + k;
              value = (float)input[b * channel * height * width + c * height * width + h * width + span];
              if (k < half_) {
                transformed_value = -1.0f * (float)input[b * channel * height * width + c * height * width + h * width + span + half_];
              } else {
                transformed_value = (float)input[b * channel * height * width + c * height * width + h * width + span - half_];
              }
              value = value * cos_ptr[k] + transformed_value * sin_ptr[k + offsetSin];
              output[b * channel * height * width + c * height * width + h * width + span] = (half)value;
            }
          }
        }
      }
  }
  )";
  return rotary_emb_cl_kernel_fp16_;
}
#endif

} // namespace nntrainer
