cmake_minimum_required(VERSION 3.16)
project(FCANN)

OPTION(DEBUG_MODE "Enable debug output [default: OFF]" ON)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-attributes")

IF(DEBUG_MODE MATCHES "ON")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC")
ENDIF(DEBUG_MODE MATCHES "ON")


add_subdirectory(fcai)

add_executable(FCANN main.cpp network.cpp network.h)
target_link_libraries(FCANN fcai)
