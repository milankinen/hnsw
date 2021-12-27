cmake_minimum_required(VERSION 3.21)
project(hnsw)

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  SET(CMAKE_CXX_FLAGS "-Ofast -DNDEBUG -std=c++17 -DHAVE_CXX0X -openmp -march=native -fpic -ftree-vectorize")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  SET(CMAKE_CXX_FLAGS "-Ofast -lrt -DNDEBUG -std=c++17 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  SET(CMAKE_CXX_FLAGS "-Ofast -lrt -DNDEBUG -std=c++17 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize")
endif ()

## Load test dependencies
include(InstallRequiredSystemLibraries)
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

FetchContent_MakeAvailable(googletest)

## Library target
include_directories(src)
add_library(hnsw
    src/hnsw.cpp)

## Tests target
enable_testing()
add_executable(hnsw_tests
    test/example_tests.cpp)

target_link_libraries(
    hnsw_tests
    hnsw
    gtest_main
)
include(GoogleTest)
gtest_discover_tests(hnsw_tests)
