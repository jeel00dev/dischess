cmake_minimum_required(VERSION 3.20)
project(DisEngine LANGUAGES CXX)

# DisEngine: NNUE chess engine + Vulkan training pipeline.
#
#   source/engine/     UCI engine (movegen, search, NNUE inference)
#   source/training/   data pipeline, classical labeler, Vulkan trainer, self-play RL
#
# Build:   cmake -S source -B <build-dir> -DCMAKE_BUILD_TYPE=Release && cmake --build <build-dir> -j
# Test:    ctest --test-dir <build-dir> --output-on-failure
# The tournament executable engine/disengine is produced by source/scripts/build_engine.sh.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(DIS_NATIVE "Tune for the build machine (-march=native, enables AVX2 NNUE path)" ON)
option(DIS_BUILD_TRAINING "Build the Vulkan training tools" ON)
option(DIS_BUILD_TESTS "Build tests" ON)

if(DIS_NATIVE)
  set(DIS_ARCH_FLAGS -march=native)
else()
  set(DIS_ARCH_FLAGS -march=x86-64-v3)
endif()

find_package(Threads REQUIRED)

# Link-time optimisation: lets the compiler inline the small move-generation and
# attack helpers across files (measurable nps gain in the engine).
include(CheckIPOSupported)
check_ipo_supported(RESULT DIS_IPO OUTPUT DIS_IPO_MSG)
if(DIS_IPO)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# ---------------------------------------------------------------- engine core
set(ENGINE_DIR ${CMAKE_SOURCE_DIR}/engine)
add_library(dis_core STATIC
  ${ENGINE_DIR}/chess.cpp
  ${ENGINE_DIR}/search.cpp
  ${ENGINE_DIR}/v7-core/network.cpp
  ${ENGINE_DIR}/v7-core/features.cpp
  ${ENGINE_DIR}/v7-core/architecture.cpp)
target_include_directories(dis_core PUBLIC ${ENGINE_DIR})
target_compile_definitions(dis_core PUBLIC DISENGINE_NET_V7)
target_compile_options(dis_core PUBLIC ${DIS_ARCH_FLAGS} -Wall -Wextra -Wno-sign-compare -Wno-missing-field-initializers)
target_link_libraries(dis_core PUBLIC Threads::Threads)

add_executable(disengine-core ${ENGINE_DIR}/main.cpp)
target_link_libraries(disengine-core PRIVATE dis_core)

if(DIS_BUILD_TESTS)
  enable_testing()
endif()
if(DIS_BUILD_TRAINING)
  add_subdirectory(training)
endif()
