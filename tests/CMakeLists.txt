add_executable(llama_link_test llama_link_test.cpp)
target_link_libraries(llama_link_test PRIVATE llama)
target_compile_options(llama_link_test PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

add_executable(tokenization_test tokenization_test.cpp)
target_link_libraries(tokenization_test PRIVATE edgeinfer_core)
target_compile_options(tokenization_test PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

add_executable(generation_test generation_test.cpp)
target_link_libraries(generation_test PRIVATE edgeinfer_core)
target_compile_options(generation_test PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

set(EDGEINFER_TEST_MODEL
    "${CMAKE_SOURCE_DIR}/models/qwen--qwen2.5-0.5b-instruct/qwen--qwen2.5-0.5b-instruct-q4_k_m.gguf"
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

add_test(
    NAME edgeinfer_inspect_invalid_model
    COMMAND edgeinfer inspect --model /nonexistent/edgeinfer-model.gguf
)
set_tests_properties(edgeinfer_inspect_invalid_model PROPERTIES
    WILL_FAIL TRUE
)

add_test(
    NAME edgeinfer_inspect_invalid_threads
    COMMAND edgeinfer inspect --model /nonexistent/edgeinfer-model.gguf --threads 9999999999
)
set_tests_properties(edgeinfer_inspect_invalid_threads PROPERTIES
    WILL_FAIL TRUE
)

add_test(
    NAME edgeinfer_inspect_invalid_context_size
    COMMAND edgeinfer inspect --model /nonexistent/edgeinfer-model.gguf --context-size 2147483648
)
set_tests_properties(edgeinfer_inspect_invalid_context_size PROPERTIES
    WILL_FAIL TRUE
)

add_test(
    NAME edgeinfer_run_missing_prompt
    COMMAND edgeinfer run --model /nonexistent/edgeinfer-model.gguf
)
set_tests_properties(edgeinfer_run_missing_prompt PROPERTIES
    WILL_FAIL TRUE
)

add_test(
    NAME edgeinfer_run_invalid_max_tokens
    COMMAND edgeinfer run --model /nonexistent/edgeinfer-model.gguf --prompt hi --max-tokens 9999999999
)
set_tests_properties(edgeinfer_run_invalid_max_tokens PROPERTIES
    WILL_FAIL TRUE
)

if(EXISTS "${EDGEINFER_TEST_MODEL}")
    add_test(
        NAME tokenization_round_trip
        COMMAND tokenization_test
    )
    set_tests_properties(tokenization_round_trip PROPERTIES
        ENVIRONMENT "EDGEINFER_TEST_MODEL=${EDGEINFER_TEST_MODEL}"
        PASS_REGULAR_EXPRESSION "tokenization_test_ok"
    )

    add_test(
        NAME edgeinfer_inspect_smoke
        COMMAND edgeinfer inspect --model ${EDGEINFER_TEST_MODEL}
    )
    set_tests_properties(edgeinfer_inspect_smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "backend: llama-cpu"
    )

    add_test(
        NAME generation_greedy_and_limits
        COMMAND generation_test
    )
    set_tests_properties(generation_greedy_and_limits PROPERTIES
        ENVIRONMENT "EDGEINFER_TEST_MODEL=${EDGEINFER_TEST_MODEL}"
        PASS_REGULAR_EXPRESSION "generation_test_ok"
    )

    add_test(
        NAME edgeinfer_run_smoke
        COMMAND edgeinfer run --model ${EDGEINFER_TEST_MODEL}
            --prompt "Reply with exactly: EDGEINFER_OK"
            --max-tokens 12 --threads 2 --context-size 512 --temperature 0
    )
    set_tests_properties(edgeinfer_run_smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "EDGEINFER_OK"
    )
endif()
