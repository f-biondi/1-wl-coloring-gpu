cmake_minimum_required(VERSION 3.16)

project(be-cuda-proj LANGUAGES CXX CUDA)

# Abseil requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O2")
set(CMAKE_CUDA_FLAGS "-O2")


# Process Abseil's CMake build system
add_subdirectory(abseil-cpp)

add_executable(cuda_be cuda_be.cu)

# Declare dependency on the absl::strings library
target_link_libraries(cuda_be absl::flat_hash_map)
set_target_properties(cuda_be PROPERTIES CUDA_ARCHITECTURES "80;86")
