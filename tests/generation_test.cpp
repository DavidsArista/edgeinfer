#include <cstdlib>
#include <iostream>
#include <string>

#include "edgeinfer/backend_config.hpp"
#include "edgeinfer/generation.hpp"
#include "edgeinfer/inference_backend.hpp"

namespace {

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "generation_test failed: " << message << '\n';
        return false;
    }
    return true;
}

std::string test_model_path() {
    if (const char* env_path = std::getenv("EDGEINFER_TEST_MODEL")) {
        return env_path;
    }
    return "models/qwen--qwen2.5-0.5b-instruct/qwen--qwen2.5-0.5b-instruct-q4_k_m.gguf";
}

std::unique_ptr<edgeinfer::InferenceBackend> load_backend(
    const edgeinfer::BackendConfig& config,
    std::string* error) {
    return edgeinfer::create_llama_backend(config, error);
}

bool run_generation(
    const edgeinfer::BackendConfig& config,
    const std::string& prompt,
    const edgeinfer::GenerationConfig& generation,
    edgeinfer::GenerationResult* result,
    std::string* error) {
    std::unique_ptr<edgeinfer::InferenceBackend> backend = load_backend(config, error);
    if (!backend) {
        return false;
    }
    return backend->generate(prompt, generation, *result, error);
}

}  // namespace

int main() {
    const std::string model_path = test_model_path();
    edgeinfer::BackendConfig config;
    config.model_path = model_path;
    config.context_size = 512;
    config.threads = 2;

    edgeinfer::GenerationConfig greedy;
    greedy.max_tokens = 8;
    greedy.temperature = 0.0f;
    greedy.use_chat_template = true;

    edgeinfer::GenerationResult first;
    edgeinfer::GenerationResult second;
    std::string error;

    if (!run_generation(config, "Reply with exactly: EDGEINFER_OK", greedy, &first, &error)) {
        std::cerr << "generation_test failed: greedy generation failed: " << error << '\n';
        return EXIT_FAILURE;
    }
    if (!expect(!first.generated_tokens.empty(), "greedy generation produced no tokens")) {
        return EXIT_FAILURE;
    }

    if (!run_generation(config, "Reply with exactly: EDGEINFER_OK", greedy, &second, &error)) {
        std::cerr << "generation_test failed: second greedy generation failed: " << error << '\n';
        return EXIT_FAILURE;
    }
    if (!expect(first.generated_tokens == second.generated_tokens,
                "greedy generation is not reproducible")) {
        return EXIT_FAILURE;
    }
    if (!expect(first.text == second.text, "greedy text output is not reproducible")) {
        std::cerr << "  first:  " << first.text << '\n';
        std::cerr << "  second: " << second.text << '\n';
        return EXIT_FAILURE;
    }

    edgeinfer::GenerationConfig limited;
    limited.max_tokens = 3;
    limited.temperature = 0.0f;
    limited.use_chat_template = false;

    edgeinfer::GenerationResult limited_result;
    if (!run_generation(config, "one two three four five six seven eight", limited,
                        &limited_result, &error)) {
        std::cerr << "generation_test failed: max token generation failed: " << error << '\n';
        return EXIT_FAILURE;
    }
    if (!expect(limited_result.generated_tokens.size() <= limited.max_tokens,
                "generated more tokens than max_tokens")) {
        return EXIT_FAILURE;
    }
    if (!expect(limited_result.stopped_on_max_tokens || limited_result.stopped_on_eos,
                "generation did not stop on max_tokens or eos")) {
        return EXIT_FAILURE;
    }

    edgeinfer::BackendConfig narrow = config;
    narrow.context_size = 512;

    edgeinfer::GenerationConfig overflow_gen;
    overflow_gen.max_tokens = 1024;
    overflow_gen.use_chat_template = false;

    std::unique_ptr<edgeinfer::InferenceBackend> narrow_backend =
        load_backend(narrow, &error);
    if (!expect(narrow_backend != nullptr,
                ("narrow context backend load failed: " + error).c_str())) {
        return EXIT_FAILURE;
    }

    edgeinfer::GenerationResult overflow_result;
    const bool overflow_ok =
        narrow_backend->generate("hello", overflow_gen, overflow_result, &error);
    if (!expect(!overflow_ok, "context overflow should fail generation")) {
        return EXIT_FAILURE;
    }
    if (!expect(error.find("exceeds context size") != std::string::npos,
                "context overflow error should mention context size")) {
        std::cerr << "  error: " << error << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "generation_test_ok\n";
    return EXIT_SUCCESS;
}
