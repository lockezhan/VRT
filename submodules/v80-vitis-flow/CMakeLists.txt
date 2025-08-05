# ##################################################################################################
#  The MIT License (MIT)
#  Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software
#  and associated documentation files (the "Software"), to deal in the Software without restriction,
#  including without limitation the rights to use, copy, modify, merge, publish, distribute,
#  sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in all copies or
#  substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ##################################################################################################

cmake_minimum_required(VERSION 3.10)
project(v80++-linker)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include/xml_parser /usr/include/libxml2 /usr/include/jsoncpp ${CMAKE_SOURCE_DIR}/include/bd_builder/ ${CMAKE_SOURCE_DIR}/include/arg_parser/ ${CMAKE_SOURCE_DIR}/include/ ${CMAKE_SOURCE_DIR}/include/utils/ ${CMAKE_SOURCE_DIR}/include/sw_emu/)

file(GLOB SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/src/xml_parser/*.cpp ${CMAKE_SOURCE_DIR}/src/arg_parser/*.cpp ${CMAKE_SOURCE_DIR}/src/bd_builder/*.cpp ${CMAKE_SOURCE_DIR}/src/utils/*.cpp ${CMAKE_SOURCE_DIR}/src/sw_emu/*.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} xml2 jsoncpp zmq)


