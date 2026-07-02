#pragma once

#include <cstdint>
#include <string>

namespace edgeinfer {

struct BackendConfig {
    std::string model_path;
    uint32_t context_size = 2048;
    int32_t threads = 4;
    bool vocab_only = false;
};

}  // namespace edgeinfer
