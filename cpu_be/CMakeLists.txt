cmake_minimum_required(VERSION 3.16)

project(be-proj)

# Abseil requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O2")

# Process Abseil's CMake build system
add_subdirectory(abseil-cpp)

add_executable(be be.cpp)
add_executable(be_time be_time.cpp)

target_link_libraries(be absl::flat_hash_map)
target_link_libraries(be_time absl::flat_hash_map)
