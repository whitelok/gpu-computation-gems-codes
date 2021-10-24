file(GLOB_RECURSE LARGE_SCALE_GPU_SEARCH_SRCS ${PROJECT_SOURCE_DIR}/large_scale_gpu_search/*.cc
  ${PROJECT_SOURCE_DIR}/large_scale_gpu_search/*.cu)

message(STATUS "Compile source: ${LARGE_SCALE_GPU_SEARCH_SRCS}")

add_executable(large_scale_gpu_search ${LARGE_SCALE_GPU_SEARCH_SRCS})
set_target_properties(
  large_scale_gpu_search
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)