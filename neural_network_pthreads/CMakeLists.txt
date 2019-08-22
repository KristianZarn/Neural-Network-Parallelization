cmake_minimum_required(VERSION 3.5)
project(neural_network_pthreads_dev)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

add_executable(demo demo.cpp)
add_executable(benchmark benchmark.cpp)
add_library(nnlib neuralnetwork_pthreads.cpp helpers.cpp readwrite.cpp)

target_link_libraries(demo nnlib)
target_link_libraries(benchmark nnlib)