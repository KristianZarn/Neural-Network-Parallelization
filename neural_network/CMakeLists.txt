cmake_minimum_required(VERSION 3.6)
project(neural_network_dev)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY})
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(demo demo.cpp)
add_executable(benchmark benchmark.cpp)
add_library(nnlib neuralnetwork.cpp helpers.cpp readwrite.cpp)

target_link_libraries(demo nnlib)
target_link_libraries(benchmark nnlib)
