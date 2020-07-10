cmake_minimum_required(VERSION 3.16)
project(FCANN)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(fcai)

add_executable(FCANN main.cpp)
target_link_libraries(FCANN fcai)
