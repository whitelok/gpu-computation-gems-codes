// Copyright 2021 karlluo. All rights reserved.
//
// P-ary Search on Sorted Lists
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

// according CUDA docs, each warp has 32 threads
// block size = 32 can be schedule by 1 warp
// it is the best way to launch our kernel
#define BLOCKSIZE 32

template <typename T>
void InitInputs(const size_t data_numbers, const size_t keys_numbers,
                thrust::host_vector<T> &h_inputs_data,
                thrust::host_vector<T> &h_keys,
                thrust::device_vector<T> &d_inputs_data,
                thrust::device_vector<T> &d_keys) {
  for (uint64_t idx = 0; idx < data_numbers; ++idx) {
    h_inputs_data[idx] = idx + 1;
  }
  std::random_device rd;
  std::mt19937_64 generator(rd());
  std::uniform_int_distribution<uint64_t> uint64_dist;
  for (uint64_t idx = 0; idx < keys_numbers; ++idx) {
    h_keys[idx] = uint64_dist(generator) % data_numbers;
  }

  thrust::copy(h_inputs_data.begin(), h_inputs_data.end(),
               d_inputs_data.begin());
  thrust::copy(h_keys.begin(), h_keys.end(), d_keys.begin());
}

// cache for boundary keys indexed by threadId shared int cache[BLOCKSIZE+2]
// index to subset for current iteration shared int range offset
// each block respond to one key searching
template <typename T>
__global__ void pary_search_gpu_kernel(const T *__restrict__ data,
                                       const T *__restrict__ search_keys,
                                       const T invalid_key_tag,
                                       size_t range_length, T *result) {
  __shared__ T cache[BLOCKSIZE + 2];
  __shared__ size_t range_offset;
  // size_t old_range_length = range_start;
  // initialize search range using a single thread
  if (threadIdx.x == 0) {
    range_offset = 0;
    cache[BLOCKSIZE] = invalid_key_tag;
    cache[BLOCKSIZE + 1] = search_keys[blockIdx.x];
  }
  __synchthreads();
  T search_key = cache[BLOCKSIZE + 1];
  // while (range_length > BLOCKSIZE) {
  //   range_length = range_length / BLOCKSIZE;
  //   // check for division underflow
  //   if (range_length * BLOCKSIZE < old_range_length) {
  //     range_length += 1;
  //   }
  //   old_range_length = range_length;
  //   // cache the boundary keys
  //   range_start = range_offset + threadIdx.x * range_length;
  //   cache[threadIdx.x] = data[range_start];
  // }
}

int main(int argc, char *argv[]) {
  // Just use GPU 0 for demo
  cudaSetDevice(0);
  // 500MB numbers need for search
  size_t DATA_NUMBERS = 500 * 1024 * 1024;
  // 1KB numbers keys for search
  size_t KEYS_NUMBERS = 1 * 1024;

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

  pary_search_gpu_kernel<uint64_t><<<KEYS_NUMBERS, BLOCKSIZE, 0, cuda_stream>>>(
      thrust::raw_pointer_cast(d_inputs_data.data()),
      thrust::raw_pointer_cast(d_keys.data()),
      std::numeric_limits<uint64_t>::max(), DATA_NUMBERS,
      thrust::raw_pointer_cast(d_result.data()));

  COMMON_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  COMMON_CUDA_CHECK(cudaDeviceSynchronize());
  COMMON_CUDA_CHECK(cudaStreamDestroy(cuda_stream));
}