# AGENTS.md

This document provides guidance for AI coding agents working in this repository.

## Project Overview

Qwen3-ASR-0.6B-CPU is an Automatic Speech Recognition (ASR) project using the Qwen3-ASR model with CPU inference via `chatllm.cpp`. It includes Chinese Inverse Text Normalization (ITN) for converting spoken numbers to written form.

## Build Commands

### C++ (chatllm.cpp)

The project builds both a CLI executable (`main`) and a shared library (`libchatllm.so`) for Python bindings.

```bash
# Build everything (executable + shared lib)
cd chatllm.cpp && mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

**Note:** The shared library `libchatllm.so` is automatically output to `chatllm.cpp/bindings/` by the CMake configuration.

### Clean Build

```bash
cd chatllm.cpp/build && make clean && cd .. && rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc)
```

## Test Commands

### Full Pipeline (Integration Test)

The primary integration test runs ASR on a sample file, performs ITN, and benchmarks performance.

```bash
# Run the full ASR pipeline (Streaming + ITN + Benchmark)
cd experiments/qwen3-asr-chatllm.cpp
python run_asr.py
```

### Python Tests (Chinese-ITN)

Unit tests for the Inverse Text Normalization module.

```bash
# Run all ITN tests
cd Chinese-ITN
python run_test.py

# Run with pytest (if installed)
pytest -v run_test.py

# Run a single manual test case
python -c "from chinese_itn import chinese_to_num; print(chinese_to_num('一五八七三六九零'))"
```

### C++ Tests

There are no formal unit tests for the C++ core. Use the `main` executable for manual verification.

```bash
# Manual C++ verification
./chatllm.cpp/build/bin/main -m models/qwen3-asr-0.6b-q4_0.bin -n 4 -c 4096 --set language Chinese --interactive
```

## Lint and Type Check Commands

This project does not enforce a strict linting CI, but the following are recommended.

### Python

```bash
# Linting
ruff check Chinese-ITN/ experiments/
flake8 Chinese-ITN/ experiments/

# Type Checking
mypy Chinese-ITN/ experiments/
```

## Code Style Guidelines

### General

- **Submodules:** `chatllm.cpp` and `Chinese-ITN` are git submodules. Modify them only if necessary and ensure changes are atomic.
- **Encoding:** UTF-8 is used throughout.

### Python

#### Imports
Group imports: Standard Library -> Third Party -> Local.

```python
import sys
import os
from pathlib import Path

import numpy as np

from chinese_itn import chinese_to_num
```

#### Naming Conventions
- **Functions/Variables:** `snake_case` (e.g., `run_asr`, `audio_path`)
- **Classes:** `PascalCase` (e.g., `AudioTransformer`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `MODEL_PATH`)

#### Error Handling
Use explicit `try/except` blocks for external operations (I/O, Inference).

```python
try:
    result = lib.chat(chat, input_data)
except Exception as e:
    print(f"Error during inference: {e}")
    # Handle or re-raise gracefully
```

### C++

#### Naming Conventions
- **Namespaces:** `lowercase` (e.g., `chatllm`)
- **Classes:** `PascalCase` (e.g., `ModelRegistry`)
- **Methods/Functions:** `snake_case` (e.g., `load_model`, `forward`)
- **Members:** `snake_case_` (trailing underscore for private members)

#### Class Structure
```cpp
class AudioProcessor {
public:
    AudioProcessor();
    void process_chunk(const std::vector<float>& data);

private:
    int sample_rate_;
    std::vector<float> buffer_;
};
```

## Project Structure

```
├── chatllm.cpp/                # C++ inference engine (submodule)
│   ├── src/                    # Core C++ source
│   ├── bindings/               # Python bindings (libchatllm.so lands here)
│   └── CMakeLists.txt          # Build config
├── Chinese-ITN/                # Chinese ITN module (submodule)
│   ├── chinese_itn.py          # Core logic
│   └── run_test.py             # Test runner
├── experiments/
│   └── qwen3-asr-chatllm.cpp/  # Main experiment scripts
│       ├── run_asr.py          # Entry point for ASR pipeline
│       └── README.md           # Experiment docs
├── models/                     # Model weights (*.bin)
└── samples/                    # Sample audio (*.wav)
```

## Key Files

| File | Purpose |
|------|---------|
| `experiments/qwen3-asr-chatllm.cpp/run_asr.py` | **Main Entry Point**. Runs ASR inference & benchmarks. |
| `chatllm.cpp/bindings/chatllm.py` | Python wrapper for the C++ shared library. |
| `Chinese-ITN/chinese_itn.py` | Logic for converting Chinese numerals to digits. |
| `chatllm.cpp/src/main.cpp` | CLI entry point for C++ executable. |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BUILD_THREADS` | `$(nproc)` | Threads for compilation |
| `INFERENCE_THREADS` | `6` | Threads for CPU inference |
| `CONTEXT_LENGTH` | `4096` | Context window size |
| `ASR_LANGUAGE` | `Chinese` | Target language for Qwen3-ASR |
