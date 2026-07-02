#pragma once

#include <cstdint>

namespace edgeinfer {

constexpr uint32_t kMaxContextSize = 1'048'576;
constexpr uint32_t kMaxThreads = 256;
constexpr uint32_t kMaxGenerationTokens = 1'048'576;

bool parse_decimal_u64(const char* text, uint64_t* out);
bool parse_threads(const char* text, int32_t* out);
bool parse_context_size(const char* text, uint32_t* out);
bool parse_max_tokens(const char* text, uint32_t* out);
bool parse_temperature(const char* text, float* out);
bool parse_seed(const char* text, uint32_t* out);

}  // namespace edgeinfer
