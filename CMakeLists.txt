cmake_minimum_required(VERSION 3.3)
project(obamadb)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Set the include directories to the project root directory and the root of the build tree (where
# generated headers will go).
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Setup for third-party libs.
#
set(THIRD_PARTY_SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party")
include(ExternalProject)

# Add gtest unit-testing framework.
set(gtest_force_shared_crt ON CACHE BOOL "Link gtest against shared DLLs on Windows")
add_subdirectory ("${THIRD_PARTY_SOURCE_DIR}/googletest/googletest"
        "${CMAKE_CURRENT_BINARY_DIR}/third_party/googletest/googletest")
include_directories(SYSTEM "${THIRD_PARTY_SOURCE_DIR}/googletest/googletest/include")
enable_testing()

# Add Gflags. Glog depends on gflags.
add_subdirectory ("${THIRD_PARTY_SOURCE_DIR}/gflags" "${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags")
include_directories("${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include")

# Add Glog logging framework.
add_subdirectory ("${THIRD_PARTY_SOURCE_DIR}/glog" "${CMAKE_CURRENT_BINARY_DIR}/third_party/glog")
include_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party/glog)
include_directories(${THIRD_PARTY_SOURCE_DIR}/glog/src)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party)

# Include libraries
#
add_subdirectory(storage)

add_executable(obamadb_main main.cpp)
target_link_libraries(obamadb_main
        glog
        gflags
        obamadb_storage_DataBlock
        obamadb_storage_DataView
        obamadb_storage_IO
        obamadb_storage_SparseDataBlock
        obamadb_storage_StorageConstants
        obamadb_storage_ThreadPool
        obamadb_storage_Matrix
        obamadb_storage_MLTask
        obamadb_storage_Utils
        obamadb_storage_tests_StorageTestHelpers)
