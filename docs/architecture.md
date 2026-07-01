# EdgeInfer Architecture

## Overview

EdgeInfer is a native C++ CLI and benchmarking harness around multiple inference backends. The baseline path uses llama.cpp on CPU; later steps add ONNX Runtime GenAI and Qualcomm QNN/HTP with explicit backend reporting.

```text
tools/prepare_model.py  →  GGUF / ONNX artifacts
        ↓
edgeinfer CLI (C++)  →  backend interface
        ↓
llama.cpp CPU  |  ONNX Runtime GenAI  |  QNN/HTP
        ↓
benchmark JSON schema + Python orchestration
```

## Repository structure

The layout matches the project plan in `CLAUDE.md`. Source is grouped by concern:

- `src/cli/` — command-line entry points
- `src/core/` — shared types and orchestration (added in later steps)
- `src/backends/` — llama.cpp and ONNX backends behind a common interface
- `src/benchmark/` — timing, memory, and result serialization
- `src/platform/` — host-specific RSS and device probes
- `tools/` — model preparation and benchmark orchestration

## Dependency pinning: llama.cpp

**Decision (Step 0):** pin llama.cpp as a **Git submodule** at `third_party/llama.cpp`.

**Pinned revision (Step 1):** `0eca4d490e591d4e93058d07540cf47278a72577`

Rationale:

- Exact commit hash is recorded in the superproject and reproducible after `git submodule update --init`.
- CMake can add the upstream target as a subdirectory without vendoring a copy in the main tree.
- Official conversion and quantize tools stay aligned with the linked library revision.

Step 1 links the EdgeInfer binary against the `llama` CMake target with tools, examples, and tests disabled. EdgeInfer does not shell out to `llama-cli` for inference.

CMake configure fails if the checked-out submodule HEAD does not match the superproject gitlink for `third_party/llama.cpp`. Run `git submodule update --init --recursive` to restore the pinned revision.

## Build system

CMake 3.16+ builds a host `edgeinfer` executable. Android NDK cross-compilation is added in Step 8 without breaking the host target.

## Backends (planned)

| Backend | Step | Notes |
|---------|------|-------|
| llama.cpp CPU | 1–4 | Linked via `third_party/llama.cpp` submodule; library API only |
| ONNX Runtime GenAI CPU | 10 | Second backend behind the same interface |
| QNN / HTP | 12 | Requires device/SDK evidence; no simulated acceleration |

## Deviations

None yet. Document any future structural changes here with justification.
