if(NOT DEFINED EDGEINFER_SOURCE_DIR OR NOT DEFINED EDGEINFER_EXPECTED_COMMIT)
    message(FATAL_ERROR "verify_llama_submodule.cmake requires EDGEINFER_SOURCE_DIR and EDGEINFER_EXPECTED_COMMIT")
endif()

find_package(Git QUIET)
if(NOT Git_FOUND)
    message(FATAL_ERROR "Git is required to verify the llama.cpp submodule pin")
endif()

set(_submodule_dir "${EDGEINFER_SOURCE_DIR}/third_party/llama.cpp")
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${_submodule_dir}" rev-parse HEAD
    OUTPUT_VARIABLE _head
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _res
)
if(NOT _res EQUAL 0)
    message(FATAL_ERROR "Failed to read llama.cpp submodule HEAD")
endif()

if(NOT _head STREQUAL EDGEINFER_EXPECTED_COMMIT)
    message(FATAL_ERROR
        "llama.cpp submodule pin mismatch at test time.\n"
        "  Expected: ${EDGEINFER_EXPECTED_COMMIT}\n"
        "  HEAD:     ${_head}")
endif()

message(STATUS "llama.cpp submodule pin verified: ${EDGEINFER_EXPECTED_COMMIT}")
