cmake_minimum_required(VERSION 3.20)

project(red_black_tree
  VERSION 0.1.0
  DESCRIPTION "Red-Black tree implementation in C++"
  LANGUAGES CXX
)

find_package(GTest CONFIG REQUIRED)

# Header-only library target for red_black_tree
add_library(red_black_tree INTERFACE)
target_include_directories(red_black_tree INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(red_black_tree INTERFACE cxx_std_20)

# Tests
enable_testing()
add_executable(test_red_black_tree tests/test_red_black_tree.cpp)
target_link_libraries(test_red_black_tree PRIVATE red_black_tree GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_red_black_tree)
