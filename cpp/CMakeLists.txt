cmake_minimum_required(VERSION 3.16)
project(clink CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

add_library(clink src/crc16.cpp src/fir_q15.cpp src/regfile.cpp)
target_include_directories(clink PUBLIC include)

add_executable(clink_tool tools/clink_tool.cpp)
target_link_libraries(clink_tool PRIVATE clink)

enable_testing()
add_executable(clink_tests tests/test_main.cpp tests/test_crc_fir.cpp tests/test_regfile.cpp)
target_link_libraries(clink_tests PRIVATE clink)
add_test(NAME clink_tests COMMAND clink_tests)
