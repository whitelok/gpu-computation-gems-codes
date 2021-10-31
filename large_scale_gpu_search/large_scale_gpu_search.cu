// Copyright 2021 karlluo. All rights reserved.
//
// P-ary Search on Sorted Lists
//
// Author: whitelok@gmail.com
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
                                       size_t range_length, size_t *result) {
  __shared__ T cache[BLOCKSIZE + 2];
  __shared__ size_t range_offset;
  size_t range_start = 0;
  size_t old_range_length = range_start;
  // NOTE(whitelok): initialize search range using a single thread
  if (threadIdx.x == 0) {
    range_offset = 0;
    cache[BLOCKSIZE] = invalid_key_tag;
    // means each block own's one search key
    cache[BLOCKSIZE + 1] = search_keys[blockIdx.x];
  }
  __syncthreads();
  // load search key to each thread register
  T search_key = cache[BLOCKSIZE + 1];

  while (range_length > BLOCKSIZE) {
    range_length = range_length / BLOCKSIZE;
    // check for division underflow
    if (range_length * BLOCKSIZE < old_range_length) {
      range_length += 1;
    }
    old_range_length = range_length;
    // cache the boundary keys
    range_start = range_offset + threadIdx.x * range_length;
    cache[threadIdx.x] = data[range_start];
    __syncthreads();

    // if the seached key is within this thread’s subset,
    // make it the one for the next iteration
    if (search_key >= cache[threadIdx.x] &&
        search_key < cache[threadIdx.x + 1]) {
      range_offset = range_start;
    }
    // all threads need to start next iteration with the new subset
    __syncthreads();
  }
  // store search result
  range_start = range_offset + threadIdx.x;
  if (search_key == data[range_start]) {
    result[blockIdx.x] = range_start;
  }
}

template <typename KeyType>
void implement(const size_t DATA_NUMBERS, const size_t KEYS_NUMBERS,
               const size_t rounds) {
  // Just use GPU 0 for demo
  cudaSetDevice(0);
  cudaEvent_t t_start, t_stop;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_stop);

  // std::cout << "Searching " << std::endl;
  std::cout << "Searching " << KEYS_NUMBERS << " keys in " << DATA_NUMBERS
            << " numbers data with " << rounds << " rounds " << std::endl;

  thrust::host_vector<KeyType> h_inputs_data(DATA_NUMBERS);
  thrust::host_vector<KeyType> h_keys(KEYS_NUMBERS);
  thrust::host_vector<size_t> h_gpu_search_result(KEYS_NUMBERS);
  thrust::device_vector<KeyType> d_inputs_data(h_inputs_data);
  thrust::device_vector<KeyType> d_keys(h_keys);
  // the result is a key index
  thrust::device_vector<size_t> d_result(KEYS_NUMBERS);

  cudaStream_t cuda_stream;
  COMMON_CUDA_CHECK(cudaStreamCreate(&cuda_stream));

  // init random numbers for demo
  InitInputs<KeyType>(DATA_NUMBERS, KEYS_NUMBERS, h_inputs_data, h_keys,
                       d_inputs_data, d_keys);

  // Warm up
  for (size_t round = 0; round < 100; ++round) {
    pary_search_gpu_kernel<KeyType>
        <<<KEYS_NUMBERS, BLOCKSIZE, 0, cuda_stream>>>(
            thrust::raw_pointer_cast(d_inputs_data.data()),
            thrust::raw_pointer_cast(d_keys.data()),
            std::numeric_limits<KeyType>::max(), DATA_NUMBERS,
            thrust::raw_pointer_cast(d_result.data()));
  }

  cudaEventRecord(t_start);
  for (size_t round = 0; round < rounds; ++round) {
    pary_search_gpu_kernel<KeyType>
        <<<KEYS_NUMBERS, BLOCKSIZE, 0, cuda_stream>>>(
            thrust::raw_pointer_cast(d_inputs_data.data()),
            thrust::raw_pointer_cast(d_keys.data()),
            std::numeric_limits<KeyType>::max(), DATA_NUMBERS,
            thrust::raw_pointer_cast(d_result.data()));
  }
  cudaEventRecord(t_stop);

  COMMON_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

  cudaEventSynchronize(t_stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, t_start, t_stop);

  // copy back the result
  thrust::copy(d_result.begin(), d_result.end(), h_gpu_search_result.begin());

  std::string is_equal = "true";
  for (size_t idx = 0; idx < KEYS_NUMBERS; ++idx) {
    if (d_inputs_data[h_gpu_search_result[idx]] != h_keys[idx]) {
      is_equal = "false";
    }
  }
  std::cout << "Correct? : " << is_equal << std::endl;

  std::cout << "One round search average elasped: "
            << static_cast<float>(milliseconds / rounds) << " ms" << std::endl;

  COMMON_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  COMMON_CUDA_CHECK(cudaDeviceSynchronize());
  COMMON_CUDA_CHECK(cudaStreamDestroy(cuda_stream));
}

int main(int argc, char *argv[]) {
  size_t rounds = 100;

  // 500M numbers need for search
  size_t DATA_NUMBERS = 500 * 1024 * 1024;
  // 1K numbers keys for search
  size_t KEYS_NUMBERS = 1 * 1024;
  implement<uint64_t>(DATA_NUMBERS, KEYS_NUMBERS, rounds);

  // 1024M numbers need for search
  DATA_NUMBERS = 1024 * 1024 * 1024;
  // 1K numbers keys for search
  KEYS_NUMBERS = 1 * 1024;
  implement<uint64_t>(DATA_NUMBERS, KEYS_NUMBERS, rounds);

  // 500M numbers need for search
  DATA_NUMBERS = 500 * 1024 * 1024;
  // 1K numbers keys for search
  KEYS_NUMBERS = 16 * 1024;
  implement<uint64_t>(DATA_NUMBERS, KEYS_NUMBERS, rounds);

  // 500M numbers need for search
  DATA_NUMBERS = 500 * 1024 * 1024;
  // 1M numbers keys for search
  KEYS_NUMBERS = 1024 * 1024;
  implement<uint64_t>(DATA_NUMBERS, KEYS_NUMBERS, rounds);

  // 1024M numbers need for search
  DATA_NUMBERS = 1024 * 1024 * 1024;
  // 1K numbers keys for search
  KEYS_NUMBERS = 16 * 1024;
  implement<uint64_t>(DATA_NUMBERS, KEYS_NUMBERS, rounds);

  // 1024M numbers need for search
  DATA_NUMBERS = 1024 * 1024 * 1024;
  // 1M numbers keys for search
  KEYS_NUMBERS = 1024 * 1024;
  implement<uint64_t>(DATA_NUMBERS, KEYS_NUMBERS, rounds);

  // 500M numbers need for search
  size_t DATA_NUMBERS = 500 * 1024 * 1024;
  // 1K numbers keys for search
  size_t KEYS_NUMBERS = 1 * 1024;
  implement<uint32_t>(DATA_NUMBERS, KEYS_NUMBERS, rounds);

  // 1024M numbers need for search
  DATA_NUMBERS = 1024 * 1024 * 1024;
  // 1K numbers keys for search
  KEYS_NUMBERS = 1 * 1024;
  implement<uint32_t>(DATA_NUMBERS, KEYS_NUMBERS, rounds);

  // 500M numbers need for search
  DATA_NUMBERS = 500 * 1024 * 1024;
  // 1K numbers keys for search
  KEYS_NUMBERS = 16 * 1024;
  implement<uint32_t>(DATA_NUMBERS, KEYS_NUMBERS, rounds);

  // 500M numbers need for search
  DATA_NUMBERS = 500 * 1024 * 1024;
  // 1M numbers keys for search
  KEYS_NUMBERS = 1024 * 1024;
  implement<uint32_t>(DATA_NUMBERS, KEYS_NUMBERS, rounds);

  // 1024M numbers need for search
  DATA_NUMBERS = 1024 * 1024 * 1024;
  // 1K numbers keys for search
  KEYS_NUMBERS = 16 * 1024;
  implement<uint32_t>(DATA_NUMBERS, KEYS_NUMBERS, rounds);

  // 1024M numbers need for search
  DATA_NUMBERS = 1024 * 1024 * 1024;
  // 1M numbers keys for search
  KEYS_NUMBERS = 1024 * 1024;
  implement<uint32_t>(DATA_NUMBERS, KEYS_NUMBERS, rounds);
}