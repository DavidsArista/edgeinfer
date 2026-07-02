#include "edgeinfer/inference_backend.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "llama.h"

namespace edgeinfer {
namespace {

class LlamaRuntime {
public:
    LlamaRuntime() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (refcount_++ == 0) {
            ggml_backend_load_all();
            llama_backend_init();
        }
    }

    ~LlamaRuntime() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (--refcount_ == 0) {
            llama_backend_free();
        }
    }

    LlamaRuntime(const LlamaRuntime&) = delete;
    LlamaRuntime& operator=(const LlamaRuntime&) = delete;

private:
    static inline std::mutex mutex_;
    static inline int refcount_ = 0;
};

struct LlamaModelDeleter {
    void operator()(llama_model* model) const noexcept {
        if (model != nullptr) {
            llama_model_free(model);
        }
    }
};

struct LlamaContextDeleter {
    void operator()(llama_context* context) const noexcept {
        if (context != nullptr) {
            llama_free(context);
        }
    }
};

struct LlamaSamplerDeleter {
    void operator()(llama_sampler* sampler) const noexcept {
        if (sampler != nullptr) {
            llama_sampler_free(sampler);
        }
    }
};

using LlamaModelPtr = std::unique_ptr<llama_model, LlamaModelDeleter>;
using LlamaContextPtr = std::unique_ptr<llama_context, LlamaContextDeleter>;
using LlamaSamplerPtr = std::unique_ptr<llama_sampler, LlamaSamplerDeleter>;

std::string meta_value_or_empty(const llama_model* model, const char* key) {
    char buffer[512];
    const int32_t length = llama_model_meta_val_str(model, key, buffer, sizeof(buffer));
    if (length < 0) {
        return {};
    }
    return std::string(buffer);
}

ModelMetadata collect_metadata(const llama_model* model, const std::string& model_path) {
    ModelMetadata metadata;
    metadata.model_path = model_path;
    metadata.architecture = meta_value_or_empty(model, "general.architecture");
    metadata.name = meta_value_or_empty(model, "general.name");
    metadata.quantization = meta_value_or_empty(model, "general.file_type");

    char description[256];
    if (llama_model_desc(model, description, sizeof(description)) > 0) {
        metadata.description = description;
    }

    metadata.context_train = llama_model_n_ctx_train(model);
    metadata.n_layer = llama_model_n_layer(model);
    metadata.n_embd = llama_model_n_embd(model);
    metadata.model_size_bytes = llama_model_size(model);
    metadata.param_count = llama_model_n_params(model);

    const llama_vocab* vocab = llama_model_get_vocab(model);
    metadata.vocab_size = llama_vocab_n_tokens(vocab);
    metadata.has_chat_template = llama_model_chat_template(model, nullptr) != nullptr;
    return metadata;
}

class LlamaInferenceBackend final : public InferenceBackend {
public:
    LlamaInferenceBackend(
        std::shared_ptr<LlamaRuntime> runtime,
        LlamaModelPtr model,
        LlamaContextPtr context,
        ModelMetadata metadata)
        : runtime_(std::move(runtime)),
          model_(std::move(model)),
          context_(std::move(context)),
          metadata_(std::move(metadata)) {}

    const ModelMetadata& metadata() const override { return metadata_; }

    std::vector<int32_t> tokenize(
        const std::string& text,
        bool add_special,
        bool parse_special) const override {
        const llama_vocab* vocab = llama_model_get_vocab(model_.get());
        const int32_t token_count =
            -llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), nullptr, 0,
                            add_special, parse_special);
        if (token_count < 0) {
            throw std::runtime_error("tokenization failed: buffer sizing returned " +
                                     std::to_string(token_count));
        }

        std::vector<int32_t> tokens(static_cast<size_t>(token_count));
        const int32_t written = llama_tokenize(
            vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
            static_cast<int32_t>(tokens.size()), add_special, parse_special);
        if (written < 0) {
            throw std::runtime_error("tokenization failed while writing tokens");
        }
        tokens.resize(static_cast<size_t>(written));
        return tokens;
    }

    std::string detokenize(
        const std::vector<int32_t>& tokens,
        bool remove_special,
        bool unparse_special) const override {
        const llama_vocab* vocab = llama_model_get_vocab(model_.get());
        const int32_t byte_count = -llama_detokenize(
            vocab, tokens.data(), static_cast<int32_t>(tokens.size()), nullptr, 0, remove_special,
            unparse_special);
        if (byte_count < 0) {
            throw std::runtime_error("detokenization failed: buffer sizing returned " +
                                     std::to_string(byte_count));
        }

        std::string text(static_cast<size_t>(byte_count), '\0');
        const int32_t written = llama_detokenize(
            vocab, tokens.data(), static_cast<int32_t>(tokens.size()), text.data(),
            static_cast<int32_t>(text.size()), remove_special, unparse_special);
        if (written < 0) {
            throw std::runtime_error("detokenization failed while writing text");
        }
        text.resize(static_cast<size_t>(written));
        return text;
    }

    std::string apply_chat_template(
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt) const override {
        const char* tmpl = llama_model_chat_template(model_.get(), nullptr);
        if (tmpl == nullptr) {
            throw std::runtime_error("model does not provide a chat template");
        }

        std::vector<llama_chat_message> llama_messages;
        llama_messages.reserve(messages.size());
        for (const ChatMessage& message : messages) {
            llama_messages.push_back({message.role.c_str(), message.content.c_str()});
        }

        int32_t buffer_size = 4096;
        for (int attempt = 0; attempt < 4; ++attempt) {
            std::vector<char> buffer(static_cast<size_t>(buffer_size), '\0');
            const int32_t written = llama_chat_apply_template(
                tmpl, llama_messages.data(), llama_messages.size(), add_generation_prompt,
                buffer.data(), buffer_size);
            if (written < 0) {
                throw std::runtime_error("chat template application failed");
            }
            if (written <= buffer_size) {
                return std::string(buffer.data(), static_cast<size_t>(written));
            }
            buffer_size = written + 1;
        }

        throw std::runtime_error("chat template output exceeded buffer growth limit");
    }

    bool generate(
        const std::string& prompt,
        const GenerationConfig& config,
        GenerationResult& result,
        std::string* error_out) override {
        auto fail = [&](const std::string& message) {
            if (error_out != nullptr) {
                *error_out = message;
            }
            return false;
        };

        llama_context* ctx = context_.get();
        const llama_vocab* vocab = llama_model_get_vocab(model_.get());
        const uint32_t n_ctx = llama_n_ctx(ctx);

        std::string prompt_text;
        if (config.use_chat_template && metadata_.has_chat_template) {
            try {
                prompt_text = apply_chat_template({{"user", prompt}}, true);
            } catch (const std::exception& ex) {
                return fail(std::string("chat template failed: ") + ex.what());
            }
        } else {
            prompt_text = prompt;
        }

        const bool is_first =
            llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;
        std::vector<int32_t> prompt_tokens;
        try {
            prompt_tokens = tokenize(prompt_text, is_first, true);
        } catch (const std::exception& ex) {
            return fail(std::string("tokenization failed: ") + ex.what());
        }

        if (prompt_tokens.empty()) {
            return fail("prompt tokenization produced no tokens");
        }
        if (prompt_tokens.size() >= n_ctx) {
            return fail("prompt token count (" + std::to_string(prompt_tokens.size()) +
                        ") exceeds context size (" + std::to_string(n_ctx) + ")");
        }
        if (prompt_tokens.size() + config.max_tokens > n_ctx) {
            return fail("prompt token count (" + std::to_string(prompt_tokens.size()) +
                        ") plus max_tokens (" + std::to_string(config.max_tokens) +
                        ") exceeds context size (" + std::to_string(n_ctx) + ")");
        }

        const uint32_t n_batch = llama_n_batch(ctx);
        if (static_cast<uint32_t>(prompt_tokens.size()) > n_batch) {
            return fail("prompt token count (" + std::to_string(prompt_tokens.size()) +
                        ") exceeds batch size (" + std::to_string(n_batch) + ")");
        }

        LlamaSamplerPtr sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        if (config.temperature <= 0.0f) {
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(config.temperature));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(config.seed));
        }

        result = GenerationResult{};
        result.prompt_tokens = std::move(prompt_tokens);

        llama_batch batch = llama_batch_get_one(
            reinterpret_cast<llama_token*>(result.prompt_tokens.data()),
            static_cast<int32_t>(result.prompt_tokens.size()));

        const int32_t n_ctx_used_before_prefill =
            llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
        if (static_cast<uint32_t>(n_ctx_used_before_prefill) +
                static_cast<uint32_t>(batch.n_tokens) >
            n_ctx) {
            return fail("context overflow before prefill decode");
        }

        if (llama_decode(ctx, batch) != 0) {
            return fail("prefill decode failed");
        }

        for (uint32_t i = 0; i < config.max_tokens; ++i) {
            const llama_token new_token = llama_sampler_sample(sampler.get(), ctx, -1);
            if (llama_vocab_is_eog(vocab, new_token)) {
                result.stopped_on_eos = true;
                break;
            }

            result.generated_tokens.push_back(new_token);

            char buffer[256];
            const int32_t piece_len =
                llama_token_to_piece(vocab, new_token, buffer, sizeof(buffer), 0, true);
            if (piece_len < 0) {
                return fail("detokenization failed during generation");
            }
            const std::string piece(buffer, static_cast<size_t>(piece_len));
            result.text += piece;
            if (config.on_token) {
                config.on_token(piece);
            }

            if (i + 1 >= config.max_tokens) {
                result.stopped_on_max_tokens = true;
                break;
            }

            llama_token decode_token = new_token;
            batch = llama_batch_get_one(&decode_token, 1);
            const int32_t n_ctx_used =
                llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
            if (static_cast<uint32_t>(n_ctx_used) + static_cast<uint32_t>(batch.n_tokens) >
                n_ctx) {
                return fail("context overflow during decode");
            }
            if (llama_decode(ctx, batch) != 0) {
                return fail("decode failed during generation");
            }
        }

        return true;
    }

private:
    std::shared_ptr<LlamaRuntime> runtime_;
    LlamaModelPtr model_;
    LlamaContextPtr context_;
    ModelMetadata metadata_;
};

}  // namespace

std::unique_ptr<InferenceBackend> create_llama_backend(
    const BackendConfig& config,
    std::string* error_out) {
    auto fail = [&](const std::string& message) {
        if (error_out != nullptr) {
            *error_out = message;
        }
        return std::unique_ptr<InferenceBackend>{};
    };

    if (config.model_path.empty()) {
        return fail("model path is empty");
    }

    FILE* model_file = std::fopen(config.model_path.c_str(), "rb");
    if (model_file == nullptr) {
        return fail("model file does not exist or is not readable: " + config.model_path);
    }
    std::fclose(model_file);

    llama_log_set(
        [](enum ggml_log_level level, const char* text, void* /*user_data*/) {
            if (level >= GGML_LOG_LEVEL_ERROR && text != nullptr) {
                std::fwrite(text, 1, std::strlen(text), stderr);
            }
        },
        nullptr);

    auto runtime = std::make_shared<LlamaRuntime>();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.vocab_only = config.vocab_only;
    model_params.use_mmap = true;

    llama_model* raw_model =
        llama_model_load_from_file(config.model_path.c_str(), model_params);
    if (raw_model == nullptr) {
        return fail("failed to load model (invalid or incompatible GGUF): " + config.model_path);
    }
    LlamaModelPtr model(raw_model);

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = config.context_size;
    context_params.n_batch = config.context_size;
    context_params.n_threads = config.threads;
    context_params.n_threads_batch = config.threads;

    llama_context* raw_context = llama_init_from_model(model.get(), context_params);
    if (raw_context == nullptr) {
        return fail("failed to create llama context for model: " + config.model_path);
    }
    LlamaContextPtr context(raw_context);

    ModelMetadata metadata = collect_metadata(model.get(), config.model_path);
    return std::make_unique<LlamaInferenceBackend>(
        std::move(runtime), std::move(model), std::move(context), std::move(metadata));
}

}  // namespace edgeinfer
