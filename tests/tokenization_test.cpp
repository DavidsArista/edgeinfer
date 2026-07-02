#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "edgeinfer/backend_config.hpp"
#include "edgeinfer/inference_backend.hpp"

namespace {

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "tokenization_test failed: " << message << '\n';
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

}  // namespace

int main() {
    const std::string model_path = test_model_path();
    edgeinfer::BackendConfig config;
    config.model_path = model_path;
    config.context_size = 512;
    config.threads = 2;

    std::string error;
    std::unique_ptr<edgeinfer::InferenceBackend> backend =
        edgeinfer::create_llama_backend(config, &error);
    if (!expect(backend != nullptr, ("backend load failed: " + error).c_str())) {
        return EXIT_FAILURE;
    }

    const std::string normal_text = "EdgeInfer token round-trip check.";
    const std::vector<int32_t> normal_tokens =
        backend->tokenize(normal_text, /*add_special=*/false, /*parse_special=*/false);
    if (!expect(!normal_tokens.empty(), "normal text tokenization returned no tokens")) {
        return EXIT_FAILURE;
    }

    const std::string normal_round_trip =
        backend->detokenize(normal_tokens, /*remove_special=*/false, /*unparse_special=*/false);
    if (!expect(normal_round_trip == normal_text, "normal text round-trip mismatch")) {
        std::cerr << "  expected: " << normal_text << '\n';
        std::cerr << "  actual:   " << normal_round_trip << '\n';
        return EXIT_FAILURE;
    }

  // Edge case: whitespace-only input should tokenize deterministically and round-trip.
    const std::string edge_text = "   \t\n";
    const std::vector<int32_t> edge_tokens =
        backend->tokenize(edge_text, /*add_special=*/false, /*parse_special=*/false);
    if (!expect(!edge_tokens.empty(), "whitespace edge case returned no tokens")) {
        return EXIT_FAILURE;
    }

    const std::string edge_round_trip =
        backend->detokenize(edge_tokens, /*remove_special=*/false, /*unparse_special=*/false);
    if (!expect(edge_round_trip == edge_text, "whitespace edge case round-trip mismatch")) {
        std::cerr << "  expected bytes: " << edge_text.size() << '\n';
        std::cerr << "  actual bytes:   " << edge_round_trip.size() << '\n';
        return EXIT_FAILURE;
    }

    if (backend->metadata().has_chat_template) {
        const std::vector<edgeinfer::ChatMessage> messages = {
            {"user", "Say hello in one word."},
        };
        const std::string formatted =
            backend->apply_chat_template(messages, /*add_generation_prompt=*/true);
        const std::vector<int32_t> chat_tokens =
            backend->tokenize(formatted, /*add_special=*/true, /*parse_special=*/true);
        if (!expect(!chat_tokens.empty(), "chat template tokenization returned no tokens")) {
            return EXIT_FAILURE;
        }
    }

    std::string invalid_error;
    edgeinfer::BackendConfig invalid_config;
    invalid_config.model_path = "/nonexistent/edgeinfer-model.gguf";
    std::unique_ptr<edgeinfer::InferenceBackend> invalid_backend =
        edgeinfer::create_llama_backend(invalid_config, &invalid_error);
    if (!expect(invalid_backend == nullptr, "invalid model path should fail to load")) {
        return EXIT_FAILURE;
    }
    if (!expect(!invalid_error.empty(), "invalid model path should set an error message")) {
        return EXIT_FAILURE;
    }

    std::cout << "tokenization_test_ok\n";
    return EXIT_SUCCESS;
}
