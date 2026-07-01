# Model Pipeline

EdgeInfer prepares models with `tools/prepare_model.py`, using the pinned llama.cpp submodule for conversion and quantization.

## Default target

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Quantization: `Q4_K_M`
- Intermediate precision: `f16` GGUF before quantization

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r tools/requirements-model.txt
```

The script builds `llama-quantize` and `llama-tokenize` from `third_party/llama.cpp` on first use (under `third_party/llama.cpp/build-tools/`).

## Prepare the model

```bash
python tools/prepare_model.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir models \
  --outtype f16 \
  --quantization Q4_K_M
```

Re-running without `--force` is safe: an existing successful manifest and output files are reused.

Use `--dry-run` to print the planned download/conversion/quantization commands without writing artifacts.

## Outputs

For the default model, artifacts land under:

```text
models/qwen--qwen2.5-0.5b-instruct/
  hf_source/                         # downloaded Hugging Face snapshot
  qwen--qwen2.5-0.5b-instruct-f16.gguf
  qwen--qwen2.5-0.5b-instruct-q4_k_m.gguf
  manifest.json
```

`manifest.json` records source model/revision, config/tokenizer hashes, commands, tool versions, timestamps, file sizes, output hashes, and smoke-test status.

## Smoke test

After quantization, the pipeline runs:

```bash
llama-tokenize --model <quantized.gguf> --prompt hello --log-disable
```

This verifies the GGUF loads through the pinned llama.cpp runtime tooling.

## Notes

- Model weights and manifests stay under `models/` and are gitignored.
- Do not substitute a pre-quantized download for the full convert-and-quantize path when completing pipeline steps.
