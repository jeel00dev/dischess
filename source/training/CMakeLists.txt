find_package(Vulkan REQUIRED)
find_program(GLSLC glslc REQUIRED)

# ---- compute shaders -> SPIR-V embedded as C arrays (tools are relocatable)
set(SHADERS forward dense_partial reduce_dense sparse_grad adam)
set(SHADER_OUT ${CMAKE_CURRENT_BINARY_DIR}/spirv)
file(MAKE_DIRECTORY ${SHADER_OUT})
set(SPIRV_INCS)
foreach(s ${SHADERS})
  set(src ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${s}.comp)
  set(out ${SHADER_OUT}/${s}.spv.inc)
  add_custom_command(
    OUTPUT ${out}
    COMMAND ${GLSLC} --target-env=vulkan1.2 -O -mfmt=num -o ${out} ${src}
    DEPENDS ${src} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/common.glsl
    COMMENT "glslc ${s}.comp")
  list(APPEND SPIRV_INCS ${out})
endforeach()
add_custom_target(dis_shaders DEPENDS ${SPIRV_INCS})

# ---- pipeline library (CPU parts)
add_library(dis_pipeline STATIC
  src/common/records.cpp
  src/common/pgn.cpp
  src/hce/hce.cpp
  src/selfplay/selfplay.cpp)
target_link_libraries(dis_pipeline PUBLIC dis_core)
target_include_directories(dis_pipeline PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---- GPU trainer library
add_library(dis_trainer STATIC
  src/gpu/vulkan.cpp
  src/train/batch.cpp
  src/train/trainer.cpp
  src/train/loader.cpp)
add_dependencies(dis_trainer dis_shaders)
target_include_directories(dis_trainer PRIVATE ${SHADER_OUT})
target_link_libraries(dis_trainer PUBLIC dis_pipeline Vulkan::Vulkan)

# ---- tools
function(dis_tool name src)
  add_executable(${name} tools/${src})
  target_link_libraries(${name} PRIVATE ${ARGN})
endfunction()
dis_tool(dis-pgn-filter pgn_filter.cpp dis_pipeline)
dis_tool(dis-texel      texel.cpp      dis_pipeline)
dis_tool(dis-label      label.cpp      dis_pipeline)
dis_tool(dis-data       data.cpp       dis_pipeline)
dis_tool(dis-train      train.cpp      dis_trainer)
dis_tool(dis-rl         rl.cpp         dis_trainer)
dis_tool(dis-model      model.cpp      dis_trainer)

# ---- tests
if(DIS_BUILD_TESTS)
  # Original float checkpoint of the shipped network (weights.float + Adam moments),
  # used by test_format to prove format/initialisation/quantisation identity.
  set(DIS_REFERENCE_MODEL_DIR "${CMAKE_SOURCE_DIR}/../../chess_engine_workspace/checkpoints/linear2048-epoch4"
      CACHE PATH "directory holding the reference linear2048 checkpoint")
  enable_testing()
  function(dis_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE ${ARGN})
    target_compile_definitions(${name} PRIVATE DIS_SOURCE_ROOT="${CMAKE_SOURCE_DIR}"
                                                DIS_REFERENCE_MODEL_DIR="${DIS_REFERENCE_MODEL_DIR}")
    add_test(NAME ${name} COMMAND ${name})
  endfunction()
  dis_test(test_chess    dis_pipeline)
  dis_test(test_features dis_pipeline)
  dis_test(test_hce      dis_pipeline)
  dis_test(test_gpu      dis_trainer)
  dis_test(test_format   dis_trainer)
  set_tests_properties(test_gpu PROPERTIES TIMEOUT 600)
endif()
