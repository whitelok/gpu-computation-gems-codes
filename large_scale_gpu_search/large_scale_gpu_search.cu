// Copyright 2021 karlluo. All rights reserved.
//
// Author: karlluo
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>

template <typename T>
void InitInputs(thrust::host_vector<T> &h_inputs_data,
                thrust::host_vector<T> &h_keys,
                thrust::device_vector<T> &d_inputs_data,
                thrust::device_vector<T> &d_keys) {}

int main(int argc, char *argv[]) {
  thrust::host_vector<uint64_t> h_inputs_data;
  thrust::host_vector<uint64_t> h_keys;
  thrust::device_vector<uint64_t> d_inputs_data;
  thrust::device_vector<uint64_t> d_keys;
  InitInputs<uint64_t>(h_inputs_data, h_keys, d_inputs_data, d_keys);
}