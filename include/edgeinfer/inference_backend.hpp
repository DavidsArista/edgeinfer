#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "edgeinfer/backend_config.hpp"
#include "edgeinfer/generation.hpp"
#include "edgeinfer/model_metadata.hpp"

namespace edgeinfer {

struct ChatMessage {
    std::string role;
    std::string content;
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    virtual const ModelMetadata& metadata() const = 0;

    virtual std::vector<int32_t> tokenize(
        const std::string& text,
        bool add_special,
        bool parse_special) const = 0;

    virtual std::string detokenize(
        const std::vector<int32_t>& tokens,
        bool remove_special,
        bool unparse_special) const = 0;

    virtual std::string apply_chat_template(
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt) const = 0;

    virtual bool generate(
        const std::string& prompt,
        const GenerationConfig& config,
        GenerationResult& result,
        std::string* error_out = nullptr) = 0;
};

std::unique_ptr<InferenceBackend> create_llama_backend(
    const BackendConfig& config,
    std::string* error_out = nullptr);

}  // namespace edgeinfer
