#pragma once

#include <iostream>

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

// template <typename Error>
// void CudaCheck(const char* code_location, Error error);

// #define CUDA_CHECK(...) CudaCheck(CODE_LOCATION, __VA_ARGS__)

// more friendly to clang-format than <<<>>>
#define CUDA_LAUNCH_KERNEL(kernel, grid_dims, block_dims, shm_size, stream, \
                           ...)                                             \
  kernel<<<grid_dims, block_dims, shm_size, stream>>>(__VA_ARGS__);         \
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

// template <>
// void CudaCheck(const char* code_location, cudaError_t error) {
//   CHECK_EQ_F(error, cudaSuccess, "{} {}", code_location,
//              cudaGetErrorString(error));
// }

// template <>
// void CudaCheck(const char* code_location, cublasStatus_t error) {
//   CHECK_EQ_F(error, CUBLAS_STATUS_SUCCESS, "{} {}", code_location,
//              CublasGetErrorString(error));
// }

// template <>
// void CudaCheck(const char* code_location, cudnnStatus_t error) {
//   CHECK_EQ_F(error, CUDNN_STATUS_SUCCESS, "{} {}", code_location,
//              cudnnGetErrorString(error));
// }

// template <>
// void CudaCheck(const char* code_location, curandStatus error) {
//   CHECK_EQ_F(error, CURAND_STATUS_SUCCESS, "{} {}", code_location,
//              CurandGetErrorString(error));
// }

// template <>
// void CudaCheck(const char* code_location, ncclResult_t error) {
//   CHECK_EQ_F(error, ncclSuccess, "{} {}", code_location,
//              ncclGetErrorString(error));
// }