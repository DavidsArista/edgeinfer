#include <iostream>
#include <string>

#include "edgeinfer/llama_build_info.hpp"
#include "edgeinfer/version.hpp"

namespace {

void print_usage() {
    std::cout << "EdgeInfer " << edgeinfer::kVersion << "\n"
              << "Usage: edgeinfer [--version]\n";
}

void print_version() {
    std::cout << "edgeinfer " << edgeinfer::kVersion << "\n"
              << "llama.cpp " << edgeinfer::kLlamaCppCommit << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        const std::string arg = argv[1];
        if (arg == "--version" || arg == "-v") {
            print_version();
            return 0;
        }
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
        std::cerr << "Unknown argument: " << arg << "\n";
        print_usage();
        return 1;
    }

    print_usage();
    return 0;
}
