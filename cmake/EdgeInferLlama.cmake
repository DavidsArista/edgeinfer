function(edgeinfer_read_llama_cpp_gitlink out_var)
    find_package(Git QUIET)
    if(NOT Git_FOUND)
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    foreach(_ref IN ITEMS ":third_party/llama.cpp" "HEAD:third_party/llama.cpp")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_SOURCE_DIR}" rev-parse "${_ref}"
            OUTPUT_VARIABLE _gitlink
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _res
        )
        if(_res EQUAL 0)
            string(LENGTH "${_gitlink}" _gitlink_len)
            if(_gitlink_len EQUAL 40 AND _gitlink MATCHES "^[0-9a-f]+$")
                set(${out_var} "${_gitlink}" PARENT_SCOPE)
                return()
            endif()
        endif()
    endforeach()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_SOURCE_DIR}" ls-files --stage third_party/llama.cpp
        OUTPUT_VARIABLE _stage_line
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _res
    )
    if(_res EQUAL 0 AND _stage_line MATCHES "^160000 ([0-9a-f]+) ")
        string(LENGTH "${CMAKE_MATCH_1}" _gitlink_len)
        if(_gitlink_len EQUAL 40)
            set(${out_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
            return()
        endif()
    endif()

    set(${out_var} "" PARENT_SCOPE)
endfunction()

function(edgeinfer_read_llama_cpp_head out_var)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
        message(FATAL_ERROR
            "llama.cpp submodule is missing. Run: git submodule update --init --recursive")
    endif()

    find_package(Git QUIET)
    if(Git_FOUND)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_SOURCE_DIR}/third_party/llama.cpp" rev-parse HEAD
            OUTPUT_VARIABLE _head
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _res
        )
        if(_res EQUAL 0)
            set(${out_var} "${_head}" PARENT_SCOPE)
            return()
        endif()
    endif()

    set(${out_var} "unknown" PARENT_SCOPE)
endfunction()

edgeinfer_read_llama_cpp_gitlink(EDGEINFER_LLAMA_CPP_GITLINK)
edgeinfer_read_llama_cpp_head(EDGEINFER_LLAMA_CPP_HEAD)

if(EDGEINFER_LLAMA_CPP_GITLINK AND NOT EDGEINFER_LLAMA_CPP_GITLINK STREQUAL EDGEINFER_LLAMA_CPP_HEAD)
    message(FATAL_ERROR
        "llama.cpp submodule drift detected.\n"
        "  Expected (superproject gitlink): ${EDGEINFER_LLAMA_CPP_GITLINK}\n"
        "  Checked out HEAD:                ${EDGEINFER_LLAMA_CPP_HEAD}\n"
        "Run: git submodule update --init --recursive")
endif()

if(EDGEINFER_LLAMA_CPP_GITLINK)
    set(EDGEINFER_LLAMA_CPP_COMMIT "${EDGEINFER_LLAMA_CPP_GITLINK}")
else()
    set(EDGEINFER_LLAMA_CPP_COMMIT "${EDGEINFER_LLAMA_CPP_HEAD}")
    message(WARNING
        "Could not read llama.cpp gitlink from superproject; "
        "using submodule HEAD ${EDGEINFER_LLAMA_CPP_COMMIT}")
endif()

message(STATUS "Pinned llama.cpp commit: ${EDGEINFER_LLAMA_CPP_COMMIT}")

# Library-only integration: link against llama, not llama-cli or other tools.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)

add_subdirectory(
    "${CMAKE_SOURCE_DIR}/third_party/llama.cpp"
    "${CMAKE_BINARY_DIR}/third_party/llama.cpp"
    EXCLUDE_FROM_ALL
)

if(NOT TARGET llama)
    message(FATAL_ERROR "llama.cpp integration failed: 'llama' target not found")
endif()
