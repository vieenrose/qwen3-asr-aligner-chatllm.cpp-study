---
title: Qwen3-ASR 0.6B CPU
emoji: "\U0001F399"
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Speech Recognition with Forced Alignment (CPU Inference)
---

# Qwen3-ASR 0.6B CPU - ASR with Forced Alignment Study

A research project implementing **Automatic Speech Recognition (ASR)** with **forced alignment** using the Qwen3-ASR model with **CPU-only inference** via a customized fork of `chatllm.cpp`.

## Overview

This repository documents a systematic study progressing from basic ASR pipeline development to production-ready HuggingFace Spaces deployments. The project explores memory behavior, performance optimization, and practical deployment strategies for CPU-based speech recognition.

**Live Demos:**
- **Exp-7 (Short Audio)**: https://huggingface.co/spaces/Luigi/Qwen3-ASR-0.6B-CPU
- **Exp-8 (Long Audio + VAD)**: https://huggingface.co/spaces/Luigi/Qwen3-ASR-VAD

## Study Goals

1. **Build CPU-only ASR pipeline** using `chatllm.cpp` Python bindings (not CLI)
2. **Investigate memory behavior** in repeated inference scenarios
3. **Integrate forced alignment** for word-level timestamps
4. **Support hours-long audio** via VAD-based chunking
5. **Deploy as interactive WebUI** on HuggingFace Spaces

## Key Findings

### Memory Behavior

| Finding | Description |
|---------|-------------|
| **restart() vs destroy/recreate** | `restart()` has **192x less overhead** than destroying and recreating `ChatLLM` objects |
| **Memory pooling** | Memory stabilizes at ~4GB after warmup (not a leak - it's pooling behavior) |
| **Model recommendation** | 0.6B model is faster, more accurate, and releases memory better than 1.7B |

### Performance Optimizations

| Optimization | Impact |
|--------------|--------|
| **Batch model loading** | 3x faster for long audio (78 loads → 2 loads) |
| **ChatLLMStreamer** | True token-by-token streaming (vs burst output) |
| **VAD chunking** | Handles hours-long audio with ~20s chunks |

### chatllm.cpp Bug Fixes

6 bugs were discovered and fixed in the forked `chatllm.cpp` repository:

| # | Bug | Fix |
|---|-----|-----|
| 1 | `free(): invalid pointer` crash | Heap allocation for `pos_helper` in `src/layers.h` |
| 2 | Empty transcription (missing language param) | New API: `chatllm_set_additional_args` |
| 3 | Python binding double-encoding | Type check before encoding in `chatllm.py` |
| 4 | Audio loading failure (`popen` mode) | Use "r" instead of "rb" in `audio_process.cpp` |
| 5 | Inference state persistence | Exposed `chatllm_destroy`, `chatllm_history_set_cursor` |
| 6 | ForcedAligner crash with `parent_id` gaps | Handle gaps with placeholders in `models/qwen.cpp` |

See [ISSUES.md](ISSUES.md) for detailed documentation.

## Experiments Overview

| Exp | Goal | Key Achievement |
|-----|------|-----------------|
| **1** | ASR pipeline with Python bindings | Streaming transcription, ITN, zh-TW, 0% WER |
| **2** | Compare memory: destroy vs restart() | Found `restart()` is 192x more efficient |
| **3** | Model switching benchmark (0.6B ↔ 1.7B) | Memory pools stabilize at ~4GB (no leak) |
| **4** | Forced Aligner memory test | Two inference methods compared |
| **5** | ASR + Aligner chain | One model at a time constraint satisfied |
| **6** | Full zh-TW pipeline with SRT | Complete pipeline: ffmpeg → ASR → ITN → Aligner → SRT |
| **7** | HuggingFace Spaces deployment | Interactive Gradio WebUI for short audio |
| **8** | VAD chunking for long audio | TEN VAD + progressive SRT generation |

### Phase 1: Foundation (Exp 1-4)

Building and testing the core ASR infrastructure:
- Python bindings for `chatllm.cpp`
- Memory leak investigation
- Model switching benchmarks
- Forced aligner integration

### Phase 2: Integration (Exp 5-6)

Combining components into complete pipelines:
- ASR → ITN → Aligner chain
- zh-TW conversion via OpenCC
- SRT subtitle generation
- Live streaming display

### Phase 3: Deployment (Exp 7-8)

Production-ready deployments:
- Gradio WebUI with streaming support
- VAD-based chunking for hours-long audio
- Progressive SRT generation

## Project Structure

```
Qwen3-ASR-0.6B-CPU/
├── chatllm.cpp/               # Forked C++ inference engine
│   ├── src/                   # Core source (with bug fixes)
│   ├── bindings/              # Python bindings (libchatllm.so, chatllm.py)
│   └── EXP1_CHANGES.md        # Fork modifications documentation
├── Chinese-ITN/               # Chinese ITN submodule
│   └── chinese_itn.py         # Number normalization (254 test cases)
├── models/                    # Model weights (via Git LFS)
│   ├── qwen3-asr-0.6b-q4_0.bin        # 514 MB - ASR model
│   ├── qwen3-asr-1.7b-q4_0.bin        # 1.27 GB - ASR model (alternative)
│   └── qwen3-forced-aligner-0.6b-q4_0.bin  # 503 MB - Aligner model
├── samples/                   # Test audio files
├── experiments/               # 8 experiment directories
│   ├── qwen3-asr-chatllm.cpp/         # Exp 1: ASR pipeline
│   ├── two-ways-of-repeated-inference/ # Exp 2: Memory comparison
│   ├── model-switch/                  # Exp 3: Model switching
│   ├── qwen3-aligner-chatllm.cpp/     # Exp 4: Aligner test
│   ├── exp-5/                         # Exp 5: ASR + Aligner chain
│   ├── exp-6/                         # Exp 6: Full zh-TW pipeline
│   ├── exp-7/                         # Exp 7: HF Spaces deployment
│   └── exp-8/                         # Exp 8: VAD chunking
├── ISSUES.md                  # Bug documentation
├── AGENTS.md                  # AI coding agent instructions
└── README.md                  # This file
```

## Key Components

### chatllm.cpp (Forked Submodule)

- **Upstream**: https://github.com/foldl/chatllm.cpp
- **Fork**: https://github.com/vieenrose/chatllm.cpp (branch: `feature/exp1-qwen3-asr`)
- **Purpose**: C++ inference engine for LLM/ASR models on CPU

Key modifications for ASR support:
- Memory safety fixes
- Python binding improvements
- ASR model purpose support
- Audio processing fixes

See [chatllm.cpp/EXP1_CHANGES.md](chatllm.cpp/EXP1_CHANGES.md) for details.

### Chinese-ITN (Submodule)

Chinese Inverse Text Normalization - converts spoken numbers to written form:
- "九百九十九" → "999"
- "一万两千三百四十五" → "12345"
- 254 test cases with 100% accuracy

### Models

| Model | File | Size | Purpose |
|-------|------|------|---------|
| Qwen3-ASR 0.6B | `qwen3-asr-0.6b-q4_0.bin` | 514 MB | Speech recognition (recommended) |
| Qwen3-ASR 1.7B | `qwen3-asr-1.7b-q4_0.bin` | 1.27 GB | Speech recognition (alternative) |
| Qwen3 Forced Aligner 0.6B | `qwen3-forced-aligner-0.6b-q4_0.bin` | 503 MB | Word-level timestamps |

Models are downloaded from HuggingFace: `Luigi/Qwen3-ASR-0.6B-chatllm-quantized`

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for containerized builds)
- Git LFS (for model weights)

### Build

```bash
# Clone with submodules
git clone --recursive https://github.com/vieenrose/qwen3-asr-aligner-chatllm.cpp-study.git
cd qwen3-asr-aligner-chatllm.cpp-study

# Build chatllm.cpp
cd chatllm.cpp && mkdir -p build && cd build
cmake .. && make -j$(nproc) libchatllm
```

### Run Experiments

```bash
# Exp 1: Basic ASR pipeline
cd experiments/qwen3-asr-chatllm.cpp
python run_asr.py

# Exp 6: Full pipeline with SRT output
cd experiments/exp-6
python pipeline.py ../../samples/meeting-1min5s.mp3 -o output.srt

# Exp 7/8: Deploy to HuggingFace Spaces
# See experiments/exp-7/README.md or experiments/exp-8/README.md
```

### Docker Build

```bash
# Build exp-7 container
cd experiments/exp-7
docker build -t qwen3-asr-exp7 .
docker run -p 7860:7860 qwen3-asr-exp7
```

## Known Issues

All discovered issues have been fixed and documented in [ISSUES.md](ISSUES.md):
- Memory safety issues in attention layers
- Python binding encoding bugs
- Audio processing compatibility
- ForcedAligner parent_id gap handling

## Technologies Used

| Technology | Purpose |
|------------|---------|
| **Qwen3-ASR** | Speech recognition model |
| **Qwen3-ForcedAligner** | Word-level timestamp alignment |
| **chatllm.cpp** | CPU inference engine |
| **TEN VAD** | Voice activity detection |
| **OpenCC** | zh-CN to zh-TW conversion |
| **Jieba** | Chinese word segmentation |
| **Gradio** | WebUI framework |

## References

- [Qwen3-Audio GitHub](https://github.com/QwenLM/Qwen3-Audio)
- [chatllm.cpp (upstream)](https://github.com/foldl/chatllm.cpp)
- [chatllm.cpp (fork with ASR support)](https://github.com/vieenrose/chatllm.cpp)
- [TEN VAD](https://github.com/TEN-framework/ten-vad)

## License

MIT
