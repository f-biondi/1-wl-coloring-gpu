cmake_minimum_required(VERSION 3.16)

project(be-proj)

# Abseil requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O2")

# Process Abseil's CMake build system
add_subdirectory(abseil-cpp)

add_executable(random_be random_be.cpp)

target_link_libraries(random_be absl::flat_hash_map)
