// Copyright 2021 karlluo. All rights reserved.
//
// Author: karlluo
#include <cuda_runtime.h>
#include <curand.h>

#include <chrono>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <random>
#include <type_traits>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>

#include <common/common_utils.hpp>

#define BLOCKSIZE 32

template <typename T>
void InitInputs(const size_t data_numbers, const size_t keys_numbers,
                thrust::host_vector<T> &h_inputs_data,
                thrust::host_vector<T> &h_keys,
                thrust::device_vector<T> &d_inputs_data,
                thrust::device_vector<T> &d_keys) {
  curandGenerator_t curand_gen_handler;
  auto seed_time = std::chrono::system_clock::now();
  // Generating random uint64_t array on device for search
  COMMON_CURAND_CHECK(
      curandCreateGenerator(&curand_gen_handler, CURAND_RNG_QUASI_SOBOL64));
  COMMON_CURAND_CHECK(curandSetGeneratorOffset(
      curand_gen_handler, std::chrono::system_clock::to_time_t(seed_time)));
  COMMON_CURAND_CHECK(
      curandSetQuasiRandomGeneratorDimensions(curand_gen_handler, BLOCKSIZE));
  COMMON_CURAND_CHECK(curandGenerateLongLong(
      curand_gen_handler,
      reinterpret_cast<unsigned long long *>(
          thrust::raw_pointer_cast(d_inputs_data.data())),
      data_numbers));
  COMMON_CUDA_CHECK(cudaDeviceSynchronize());
  COMMON_CURAND_CHECK(
      curandGenerateLongLong(curand_gen_handler,
                             reinterpret_cast<unsigned long long *>(
                                 thrust::raw_pointer_cast(d_keys.data())),
                             keys_numbers));
  COMMON_CUDA_CHECK(cudaDeviceSynchronize());
  COMMON_CURAND_CHECK(curandDestroyGenerator(curand_gen_handler));
}

// cache for boundary keys indexed by threadId shared int cache[BLOCKSIZE+2] ;
// index to subset for current iteration shared int range offset;

__shared__ int cache[BLOCKSIZE + 2];
__shared__ int range_offset;

template <typename T>
__global__ void pary_search_gpu(const T *__restrict__ data,
                                const T *__restrict__ search_keys,
                                size_t range_length, T *result) {
  size_t search_key = range_length;
  size_t old_range_length = range_start;
}

int main(int argc, char *argv[]) {
  // Just use GPU 0 for demo
  cudaSetDevice(0);
  // 500MB numbers need for search
  size_t DATA_NUMBERS = 500 * 1024 * 1024;
  // 10MB numbers keys for search
  size_t KEYS_NUMBERS = 10 * 1024 * 1024;

  thrust::host_vector<uint64_t> h_inputs_data(DATA_NUMBERS);
  thrust::host_vector<uint64_t> h_keys(KEYS_NUMBERS);
  thrust::device_vector<uint64_t> d_inputs_data(h_inputs_data);
  thrust::device_vector<uint64_t> d_keys(h_keys);
  thrust::device_vector<uint64_t> d_result(KEYS_NUMBERS);

  cudaStream_t cuda_stream;
  COMMON_CUDA_CHECK(cudaStreamCreate(&cuda_stream));

  // init random numbers for demo
  InitInputs<uint64_t>(DATA_NUMBERS, KEYS_NUMBERS, h_inputs_data, h_keys,
                       d_inputs_data, d_keys);

  // pary_search_gpu<uint64_t><<<1, 1, 0, cuda_stream>>>(
  //     thrust::raw_pointer_cast(d_inputs_data.data()),
  //     thrust::raw_pointer_cast(d_keys.data()), DATA_NUMBERS,
  //     thrust::raw_pointer_cast(d_result.data()));

  COMMON_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  COMMON_CUDA_CHECK(cudaDeviceSynchronize());
  COMMON_CUDA_CHECK(cudaStreamDestroy(cuda_stream));
}