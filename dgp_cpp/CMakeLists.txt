cmake_minimum_required(VERSION 3.16)
project(dgp_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
# optimisation flags differ between MSVC and GCC/Clang.
if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
  # /arch:AVX2 helps Eigen vectorise; comment out if your CPU lacks AVX2.
  string(APPEND CMAKE_CXX_FLAGS_RELEASE " /arch:AVX2")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  # -march=native gives Eigen the best vectorisation on your own CPU.
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
  if(HAS_MARCH_NATIVE)
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -march=native")
  endif()
endif()

# Eigen: use a system install if present, else fetch 3.4.0
# If Eigen was installed with vcpkg, configure with:
#   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
# and this find_package will locate it (the FetchContent fallback won't run).
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
  message(STATUS "system Eigen3 not found; fetching 3.4.0")
  include(FetchContent)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0)
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(eigen)
endif()

find_package(OpenMP)

add_library(dgp_core src/tooling.cpp src/mcmc.cpp src/dgp.cpp)
target_include_directories(dgp_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dgp_core PUBLIC Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(dgp_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(dgp_demo src/main.cpp)
target_link_libraries(dgp_demo PRIVATE dgp_core)