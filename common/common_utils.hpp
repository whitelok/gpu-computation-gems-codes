#pragma once

#include <cstdlib>
#include <cxxabi.h>
#include <iostream>
#include <string>

#include <cuda_runtime_api.h>
#include <curand.h>

#define COMMON_CUDA_CHECK(x)                                                   \
  do {                                                                         \
    cudaError_t retval = (x);                                                  \
    if (retval != cudaSuccess) {                                               \
      std::cerr << "Error at " << std::to_string(__LINE__) << ": "             \
                << cudaGetErrorString(retval) << std::endl;                    \
    }                                                                          \
  } while (0)

#define COMMON_CURAND_CHECK(x)                                                 \
  do {                                                                         \
    curandStatus_t retval = (x);                                               \
    if (retval != CURAND_STATUS_SUCCESS) {                                     \
      std::cerr << "Error at " << std::to_string(__LINE__) << ": "             \
                << curandGetErrorString(retval) << std::endl;                  \
    }                                                                          \
  } while (0)

// more friendly to clang-format than <<<>>>
#define CUDA_LAUNCH_KERNEL(kernel, grid_dims, block_dims, shm_size, stream,    \
                           ...)                                                \
  kernel<<<grid_dims, block_dims, shm_size, stream>>>(__VA_ARGS__);            \
  CudaCheck(CODE_LOCATION, cudaGetLastError())

static const char *curandGetErrorString(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";

  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";

  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";

  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";

  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";

  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";

  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";

  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";

  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";

  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";

  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}

template <typename T> std::string type_name() {
  int status;
  std::string tname = typeid(T).name();
  char *demangled_name =
      abi::__cxa_demangle(tname.c_str(), NULL, NULL, &status);
  if (status == 0) {
    tname = demangled_name;
    std::free(demangled_name);
  }
  return tname;
}