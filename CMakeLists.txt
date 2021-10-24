cmake_minimum_required(VERSION 3.8)
project(gpu LANGUAGES C CXX CUDA)

# adding external package
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/external)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_STANDARD 17)

if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
  set(CMAKE_CUDA_STANDARD 17)
else()
  set(CMAKE_CUDA_STANDARD 14)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -D_GLIBCXX_USE_CXX11_ABI=1 \
                     -Wno-unknown-pragmas")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true \
                      --expt-extended-lambda --expt-relaxed-constexpr \
                      -Xcompiler -Wall")

# 61 include P4/P40
# 70 include V100
# 75 include T4
# 80 include A100
foreach(SM 61 70 75)
  message(STATUS "Assign GPU architecture (sm=${SM})")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
      -gencode=arch=compute_${SM},code=\\\"sm_${SM},compute_${SM}\\\"")
endforeach()
# for Numerous 050 or lower version, CUDA is not support sm_80
if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
  message(STATUS "Assign GPU architecture (sm=80)")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
      -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\"")
endif()

if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
  # Policy CMP0104 is not set: CMAKE_CUDA_ARCHITECTURES now detected for NVCC,
  # empty CUDA_ARCHITECTURES not allowed.  Run "cmake --help-policy CMP0104"
  # for policy details.  Use the cmake_policy command to set the policy and
  set(CMAKE_CUDA_ARCHITECTURES "61;70;75;80")
endif()

# cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND THIRD_PARTY_LIBS ${CUDA_LIBRARIES}
                             ${CUDA_CUBLAS_LIBRARIES}
                             ${CUDA_curand_LIBRARY}
                             ${CUDA_nvToolsExt_LIBRARY})

# cudnn
find_package(CUDNN REQUIRED)
include_directories(${CUDNN_INCLUDE_DIR})
list(APPEND THIRD_PARTY_LIBS ${CUDNN_LIBRARIES})

# Large-Scale GPU Search
add_subdirectory(large_scale_gpu_search)