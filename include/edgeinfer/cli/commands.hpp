#pragma once

#include "edgeinfer/backend_config.hpp"
#include "edgeinfer/generation.hpp"

namespace edgeinfer {

int run_inspect_command(const BackendConfig& config);
int run_backends_command();
int run_generate_command(const BackendConfig& config, const GenerationConfig& generation,
                         const std::string& prompt);
void print_usage();

}  // namespace edgeinfer
