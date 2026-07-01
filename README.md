# EdgeInfer

Reproducible embedded-LLM inference and benchmarking: Hugging Face models through conversion, native C++ runtimes, ARM64 deployment, and verified accelerator backends.

## Status

Step 2 complete: `tools/prepare_model.py` converts Hugging Face models to quantized GGUF artifacts. Native C++ load/tokenize begins in Step 3.

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

Expected output includes:

```text
edgeinfer 0.1.0
llama.cpp <pinned-commit-sha>
```

## Repository layout

See `docs/architecture.md` for structure, dependency pinning, and planned backends.

## License

MIT. See `LICENSE`.
