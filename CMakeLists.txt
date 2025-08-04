cmake_minimum_required(VERSION 3.12)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -Wall")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")

include_directories(${CMAKE_SOURCE_DIR}/include/Eigen)

file(GLOB SOURCES "src/*.cpp" "src/training_sets/*.cpp")

add_executable(neural_network ${SOURCES})