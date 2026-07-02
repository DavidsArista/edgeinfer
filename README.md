# EdgeInfer

Reproducible embedded-LLM inference and benchmarking: Hugging Face models through conversion, native C++ runtimes, ARM64 deployment, and verified accelerator backends.

## Status

Step 4 complete: `edgeinfer run` performs greedy/sampled autoregressive generation through the llama.cpp library API. Benchmark instrumentation begins in Step 5.

## Clone and dependencies

```bash
git clone --recurse-submodules <repository-url>
cd edgeinfer
```

If the repository is already cloned:

```bash
git submodule update --init --recursive
```

Pinned llama.cpp revision is recorded in `docs/architecture.md` and printed by `edgeinfer --version`.

## Requirements

- macOS on Apple Silicon (primary development host)
- CMake 3.16+
- C++17 compiler (Apple Clang or LLVM)
- Python 3.11+

## Python environment

Create and activate a project-local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

No Python packages are required for the C++ build. Model preparation uses:

```bash
python -m pip install -r tools/requirements-model.txt
```

See `docs/model-pipeline.md` for the full `prepare_model.py` workflow.

Capture host tool versions as JSON:

```bash
python tools/environment_report.py > environment.json
```

## Build (host)

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/edgeinfer --version
```

## Run inference (host)

After preparing a model (see `docs/model-pipeline.md`):

```bash
./build/edgeinfer run \
  --model models/qwen--qwen2.5-0.5b-instruct/qwen--qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --prompt "What is 2+2? Reply with one number." \
  --backend llama-cpu \
  --threads 4 \
  --context-size 2048 \
  --max-tokens 32 \
  --temperature 0
```

`--temperature 0` selects greedy decoding. Set `--temperature` above `0` and optionally `--seed` for sampling.

Expected output includes:

```text
edgeinfer 0.1.0
llama.cpp <pinned-commit-sha>
```

## Repository layout

See `docs/architecture.md` for structure, dependency pinning, and planned backends.

## License

MIT. See `LICENSE`.
