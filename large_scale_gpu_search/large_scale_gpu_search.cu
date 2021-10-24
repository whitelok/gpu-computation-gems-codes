// Copyright 2021 karlluo. All rights reserved.
//
// Author: karlluo
#include <cuda_runtime.h>
#include <curand.h>

#include <cstdint>
#include <iostream>
#include <random>
#include <type_traits>
#include <ctime>

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
  // Generating random uint64_t for search
  COMMON_CURAND_CHECK(
      curandCreateGenerator(&curand_gen_handler, CURAND_RNG_QUASI_SOBOL64));
  curandSetGeneratorOffset (curand_gen_handler, std::time(static_cast<int>(keys_numbers)));
  curandSetQuasiRandomGeneratorDimensions(curand_gen_handler, keys_numbers);
  COMMON_CURAND_CHECK(curandGenerateLongLong(
      curand_gen_handler,
      reinterpret_cast<unsigned long long *>(
          thrust::raw_pointer_cast(d_inputs_data.data())),
      data_numbers));
  COMMON_CUDA_CHECK(cudaDeviceSynchronize());
}

// cache for boundary keys indexed by threadId shared int cache[BLOCKSIZE+2] ;
// index to subset for current iteration shared int range offset;

__shared__ int cache[BLOCKSIZE + 2];

int main(int argc, char *argv[]) {
  // 500MB numbers need for search
  size_t DATA_NUMBERS = 500 * 1024 * 1024;
  // 10MB numbers keys for search
  size_t KEYS_NUMBERS = 10 * 1024 * 1024;

  thrust::host_vector<uint64_t> h_inputs_data(DATA_NUMBERS);
  thrust::host_vector<uint64_t> h_keys(KEYS_NUMBERS);
  thrust::device_vector<uint64_t> d_inputs_data(h_inputs_data);
  thrust::device_vector<uint64_t> d_keys(h_keys);
  // init random numbers for demo
  InitInputs<uint64_t>(DATA_NUMBERS, KEYS_NUMBERS, h_inputs_data, h_keys,
                       d_inputs_data, d_keys);
}