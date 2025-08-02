cmake_minimum_required(VERSION 3.16)

project(be-cuda-proj LANGUAGES CXX CUDA)

# Abseil requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O2 -fopenmp")
set(CMAKE_CUDA_FLAGS "-O2")


# Process Abseil's CMake build system
add_subdirectory(abseil-cpp)

add_executable(batched_be batched_be.cu)
add_executable(batched_be_aggressive batched_be_aggressive.cu)

target_link_libraries(batched_be absl::flat_hash_map)
set_target_properties(batched_be PROPERTIES CUDA_ARCHITECTURES "80;86")
target_link_libraries(batched_be_aggressive absl::flat_hash_map)
set_target_properties(batched_be_aggressive PROPERTIES CUDA_ARCHITECTURES "80;86")

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
  set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
