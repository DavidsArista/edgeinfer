#include "edgeinfer/cli/parse_args.hpp"

#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>

namespace edgeinfer {

bool parse_decimal_u64(const char* text, uint64_t* out) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    const char* end = text + std::strlen(text);
    const auto result = std::from_chars(text, end, *out);
    return result.ec == std::errc{} && result.ptr == end;
}

bool parse_threads(const char* text, int32_t* out) {
    uint64_t value = 0;
    if (!parse_decimal_u64(text, &value) || value == 0 || value > kMaxThreads) {
        return false;
    }
    *out = static_cast<int32_t>(value);
    return true;
}

bool parse_context_size(const char* text, uint32_t* out) {
    uint64_t value = 0;
    if (!parse_decimal_u64(text, &value) || value == 0 || value > kMaxContextSize) {
        return false;
    }
    *out = static_cast<uint32_t>(value);
    return true;
}

bool parse_max_tokens(const char* text, uint32_t* out) {
    uint64_t value = 0;
    if (!parse_decimal_u64(text, &value) || value == 0 || value > kMaxGenerationTokens) {
        return false;
    }
    *out = static_cast<uint32_t>(value);
    return true;
}

bool parse_temperature(const char* text, float* out) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    const char* end = text + std::strlen(text);
    const auto result = std::from_chars(text, end, *out);
    if (result.ec != std::errc{} || result.ptr != end) {
        return false;
    }
    if (!std::isfinite(*out) || *out < 0.0f) {
        return false;
    }
    return true;
}

bool parse_seed(const char* text, uint32_t* out) {
    uint64_t value = 0;
    if (!parse_decimal_u64(text, &value) || value > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    *out = static_cast<uint32_t>(value);
    return true;
}

}  // namespace edgeinfer
