#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace edgeinfer {

struct GenerationConfig {
    uint32_t max_tokens = 128;
    float temperature = 0.0f;
    uint32_t seed = 0xFFFFFFFF;
    bool use_chat_template = true;
    std::function<void(const std::string&)> on_token;
};

struct GenerationResult {
    std::string text;
    std::vector<int32_t> prompt_tokens;
    std::vector<int32_t> generated_tokens;
    bool stopped_on_eos = false;
    bool stopped_on_max_tokens = false;
};

}  // namespace edgeinfer
