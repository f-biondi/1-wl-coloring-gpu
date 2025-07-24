cmake_minimum_required(VERSION 3.16)

project(be-cuda-proj LANGUAGES CXX CUDA)

# Abseil requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O2 -fopenmp")
set(CMAKE_CUDA_FLAGS "-O2")


# Process Abseil's CMake build system
add_subdirectory(abseil-cpp)

add_executable(cuda_be cuda_be.cu)
add_executable(cuda_be_multi cuda_be_multi.cu)
add_executable(cuda_be_time cuda_be_time.cu)

target_link_libraries(cuda_be absl::flat_hash_map)
set_target_properties(cuda_be PROPERTIES CUDA_ARCHITECTURES "80;86")
target_link_libraries(cuda_be_multi absl::flat_hash_map)
set_target_properties(cuda_be_multi PROPERTIES CUDA_ARCHITECTURES "80;86")
target_link_libraries(cuda_be_time absl::flat_hash_map)
set_target_properties(cuda_be_time PROPERTIES CUDA_ARCHITECTURES "80;86")

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
  set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
