add_executable(llama_link_test llama_link_test.cpp)
target_link_libraries(llama_link_test PRIVATE llama)
target_compile_options(llama_link_test PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

add_test(
    NAME edgeinfer_version_smoke
    COMMAND edgeinfer --version
)

set_tests_properties(edgeinfer_version_smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "edgeinfer 0\\.1\\.0"
)

add_test(
    NAME edgeinfer_llama_revision_smoke
    COMMAND edgeinfer --version
)

# EDGEINFER_LLAMA_CPP_COMMIT is the superproject gitlink verified at configure time.
set_tests_properties(edgeinfer_llama_revision_smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "llama\\.cpp ${EDGEINFER_LLAMA_CPP_COMMIT}"
)

if(EDGEINFER_LLAMA_CPP_GITLINK)
    add_test(
        NAME llama_submodule_pin_smoke
        COMMAND ${CMAKE_COMMAND}
            -DEDGEINFER_SOURCE_DIR=${CMAKE_SOURCE_DIR}
            -DEDGEINFER_EXPECTED_COMMIT=${EDGEINFER_LLAMA_CPP_GITLINK}
            -P ${CMAKE_SOURCE_DIR}/cmake/verify_llama_submodule.cmake
    )
endif()

add_test(
    NAME llama_link_smoke
    COMMAND llama_link_test
)

set_tests_properties(llama_link_smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "llama_link_ok"
)
