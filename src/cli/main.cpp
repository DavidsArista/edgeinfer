#include <iostream>
#include <string>

#include "edgeinfer/backend_config.hpp"
#include "edgeinfer/cli/commands.hpp"
#include "edgeinfer/cli/parse_args.hpp"
#include "edgeinfer/generation.hpp"
#include "edgeinfer/llama_build_info.hpp"
#include "edgeinfer/version.hpp"

namespace {

int run_inspect(int argc, char** argv) {
    edgeinfer::BackendConfig config;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            config.model_path = argv[++i];
            continue;
        }
        if (arg == "--context-size" && i + 1 < argc) {
            if (!edgeinfer::parse_context_size(argv[++i], &config.context_size)) {
                std::cerr << "invalid --context-size value (must be 1-"
                          << edgeinfer::kMaxContextSize << ")\n";
                return 1;
            }
            continue;
        }
        if (arg == "--threads" && i + 1 < argc) {
            if (!edgeinfer::parse_threads(argv[++i], &config.threads)) {
                std::cerr << "invalid --threads value (must be 1-" << edgeinfer::kMaxThreads
                          << ")\n";
                return 1;
            }
            continue;
        }
        std::cerr << "unknown or incomplete inspect argument: " << arg << '\n';
        edgeinfer::print_usage();
        return 1;
    }

    if (config.model_path.empty()) {
        std::cerr << "inspect requires --model PATH\n";
        edgeinfer::print_usage();
        return 1;
    }

    return edgeinfer::run_inspect_command(config);
}

int run_generate(int argc, char** argv) {
    edgeinfer::BackendConfig config;
    edgeinfer::GenerationConfig generation;
    std::string prompt;
    std::string backend = "llama-cpu";

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            config.model_path = argv[++i];
            continue;
        }
        if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
            continue;
        }
        if (arg == "--backend" && i + 1 < argc) {
            backend = argv[++i];
            continue;
        }
        if (arg == "--context-size" && i + 1 < argc) {
            if (!edgeinfer::parse_context_size(argv[++i], &config.context_size)) {
                std::cerr << "invalid --context-size value (must be 1-"
                          << edgeinfer::kMaxContextSize << ")\n";
                return 1;
            }
            continue;
        }
        if (arg == "--threads" && i + 1 < argc) {
            if (!edgeinfer::parse_threads(argv[++i], &config.threads)) {
                std::cerr << "invalid --threads value (must be 1-" << edgeinfer::kMaxThreads
                          << ")\n";
                return 1;
            }
            continue;
        }
        if (arg == "--max-tokens" && i + 1 < argc) {
            if (!edgeinfer::parse_max_tokens(argv[++i], &generation.max_tokens)) {
                std::cerr << "invalid --max-tokens value (must be 1-"
                          << edgeinfer::kMaxGenerationTokens << ")\n";
                return 1;
            }
            continue;
        }
        if (arg == "--temperature" && i + 1 < argc) {
            if (!edgeinfer::parse_temperature(argv[++i], &generation.temperature)) {
                std::cerr << "invalid --temperature value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--seed" && i + 1 < argc) {
            if (!edgeinfer::parse_seed(argv[++i], &generation.seed)) {
                std::cerr << "invalid --seed value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--no-chat-template") {
            generation.use_chat_template = false;
            continue;
        }
        std::cerr << "unknown or incomplete run argument: " << arg << '\n';
        edgeinfer::print_usage();
        return 1;
    }

    if (config.model_path.empty()) {
        std::cerr << "run requires --model PATH\n";
        edgeinfer::print_usage();
        return 1;
    }
    if (prompt.empty()) {
        std::cerr << "run requires --prompt TEXT\n";
        edgeinfer::print_usage();
        return 1;
    }
    if (backend != "llama-cpu") {
        std::cerr << "unsupported backend: " << backend << " (available: llama-cpu)\n";
        return 1;
    }

    return edgeinfer::run_generate_command(config, generation, prompt);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        edgeinfer::print_usage();
        return 0;
    }

    const std::string command = argv[1];
    if (command == "--version" || command == "-v") {
        std::cout << "edgeinfer " << edgeinfer::kVersion << '\n'
                  << "llama.cpp " << edgeinfer::kLlamaCppCommit << '\n';
        return 0;
    }
    if (command == "--help" || command == "-h") {
        edgeinfer::print_usage();
        return 0;
    }
    if (command == "backends") {
        return edgeinfer::run_backends_command();
    }
    if (command == "inspect") {
        return run_inspect(argc, argv);
    }
    if (command == "run") {
        return run_generate(argc, argv);
    }

    std::cerr << "Unknown argument: " << command << '\n';
    edgeinfer::print_usage();
    return 1;
}
