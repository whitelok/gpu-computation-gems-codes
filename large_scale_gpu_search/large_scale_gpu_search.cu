// Copyright 2021 karlluo. All rights reserved.
//
// Author: karlluo
#include <cstdint>
#include <random>
#include <type_traits>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>

#define BLOCKSIZE 32

template <typename T>
void InitInputs(const size_t data_numbers, const size_t keys_numbers,
                thrust::host_vector<T> &h_inputs_data,
                thrust::host_vector<T> &h_keys,
                thrust::device_vector<T> &d_inputs_data,
                thrust::device_vector<T> &d_keys) {
  std::random_device rd;
  std::mt19937 gen(rd());
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
  thrust::device_vector<uint64_t> d_inputs_data;
  thrust::device_vector<uint64_t> d_keys;
  // init random numbers for demo
  InitInputs<uint64_t>(h_inputs_data, h_keys, d_inputs_data, d_keys);
}