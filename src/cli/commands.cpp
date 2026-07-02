#include "edgeinfer/cli/commands.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "edgeinfer/inference_backend.hpp"
#include "edgeinfer/version.hpp"

namespace edgeinfer {

void print_usage() {
    std::cout << "EdgeInfer " << kVersion << "\n"
              << "Usage:\n"
              << "  edgeinfer --version\n"
              << "  edgeinfer backends\n"
              << "  edgeinfer inspect --model PATH [--context-size N] [--threads N]\n"
              << "  edgeinfer run --model PATH --prompt TEXT [--backend llama-cpu]\n"
              << "      [--context-size N] [--threads N] [--max-tokens N]\n"
              << "      [--temperature F] [--seed N] [--no-chat-template]\n";
}

namespace {

void print_kv(const char* key, const std::string& value) {
    std::cout << key << ": " << (value.empty() ? "(unknown)" : value) << '\n';
}

void print_kv_u64(const char* key, uint64_t value) {
    std::cout << key << ": " << value << '\n';
}

void print_kv_i32(const char* key, int32_t value) {
    std::cout << key << ": " << value << '\n';
}

}  // namespace

int run_inspect_command(const BackendConfig& config) {
    std::string error;
    std::unique_ptr<InferenceBackend> backend = create_llama_backend(config, &error);
    if (!backend) {
        std::cerr << "inspect failed: " << error << '\n';
        return 1;
    }

    const ModelMetadata& metadata = backend->metadata();
    std::cout << "backend: llama-cpu\n";
    print_kv("model_path", metadata.model_path);
    print_kv("description", metadata.description);
    print_kv("architecture", metadata.architecture);
    print_kv("name", metadata.name);
    print_kv("quantization", metadata.quantization);
    print_kv_i32("context_train", metadata.context_train);
    print_kv_i32("n_layer", metadata.n_layer);
    print_kv_i32("n_embd", metadata.n_embd);
    print_kv_i32("vocab_size", metadata.vocab_size);
    print_kv_u64("model_size_bytes", metadata.model_size_bytes);
    print_kv_u64("param_count", metadata.param_count);
    std::cout << "has_chat_template: " << (metadata.has_chat_template ? "true" : "false") << '\n';

    if (metadata.has_chat_template) {
        try {
            const std::vector<ChatMessage> messages = {
                {"user", "Hello"},
            };
            const std::string formatted =
                backend->apply_chat_template(messages, /*add_generation_prompt=*/true);
            const std::vector<int32_t> tokens =
                backend->tokenize(formatted, /*add_special=*/true, /*parse_special=*/true);
            std::cout << "chat_template_probe_tokens: " << tokens.size() << '\n';
        } catch (const std::exception& ex) {
            std::cerr << "chat template probe failed: " << ex.what() << '\n';
            return 1;
        }
    }

    return 0;
}

int run_backends_command() {
    std::cout << "llama-cpu  available  llama.cpp library backend (CPU)\n";
    return 0;
}

int run_generate_command(const BackendConfig& config, const GenerationConfig& generation,
                         const std::string& prompt) {
    std::string error;
    std::unique_ptr<InferenceBackend> backend = create_llama_backend(config, &error);
    if (!backend) {
        std::cerr << "run failed: " << error << '\n';
        return 1;
    }

    GenerationConfig streaming_config = generation;
    streaming_config.on_token = [](const std::string& piece) {
        std::cout << piece << std::flush;
    };

    GenerationResult result;
    if (!backend->generate(prompt, streaming_config, result, &error)) {
        std::cerr << "run failed: " << error << '\n';
        return 1;
    }

    if (!result.text.empty() && result.text.back() != '\n') {
        std::cout << '\n';
    }
    return 0;
}

}  // namespace edgeinfer
