cmake_minimum_required(VERSION 3.16)
project(EdgeInfer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/EdgeInferOptions.cmake)
include(cmake/EdgeInferLlama.cmake)

add_executable(edgeinfer
    src/cli/main.cpp
)

target_include_directories(edgeinfer
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(edgeinfer PRIVATE
    EDGEINFER_LLAMA_CPP_COMMIT="${EDGEINFER_LLAMA_CPP_COMMIT}"
)

target_link_libraries(edgeinfer PRIVATE llama)

target_compile_options(edgeinfer PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

if(EDGEINFER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
