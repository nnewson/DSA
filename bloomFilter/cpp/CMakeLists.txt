cmake_minimum_required(VERSION 3.20)

project(bloom_filter
  VERSION 0.1.0
  DESCRIPTION "Bloom filter implementation in C++"
  LANGUAGES CXX
)

find_package(xxHash CONFIG REQUIRED)
find_package(murmurhash CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

# Header-only library target for bloom_filter
add_library(bloom_filter INTERFACE)
target_include_directories(bloom_filter INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(bloom_filter INTERFACE xxHash::xxhash murmurhash::murmurhash)
target_compile_features(bloom_filter INTERFACE cxx_std_20)

# Tests
enable_testing()
add_executable(test_bloom_filter tests/test_bloom_filter.cpp)
target_link_libraries(test_bloom_filter PRIVATE bloom_filter GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_bloom_filter)
