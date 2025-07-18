cmake_minimum_required(VERSION 3.12)
project(NeuralNetwork)

set (CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/include/Eigen)

file(GLOB SOURCES "src/*.cpp")

add_executable(neural_network ${SOURCES})