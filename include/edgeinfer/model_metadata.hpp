#pragma once

#include <cstdint>
#include <string>

namespace edgeinfer {

struct ModelMetadata {
    std::string model_path;
    std::string description;
    std::string architecture;
    std::string name;
    std::string quantization;
    int32_t context_train = 0;
    int32_t n_layer = 0;
    int32_t n_embd = 0;
    int32_t vocab_size = 0;
    uint64_t model_size_bytes = 0;
    uint64_t param_count = 0;
    bool has_chat_template = false;
};

}  // namespace edgeinfer
