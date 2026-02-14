# Qwen3-ASR-0.6B-CPU

This repository implements Automatic Speech Recognition (ASR) using the **Qwen3-ASR** model with CPU inference powered by **chatllm.cpp**. It features Chinese Inverse Text Normalization (ITN) and streaming transcription capabilities.

## Repository Structure

- `chatllm.cpp/`: Submodule for the C++ inference engine.
- `Chinese-ITN/`: Submodule for Chinese Inverse Text Normalization.
- `models/`: Directory for model weights (`qwen3-asr-0.6b-q4_0.bin`).
- `samples/`: Sample audio files for testing.
- `experiments/`: Contains experiment implementations and benchmarks.

## Experiments

### Experiment 1: Qwen3-ASR with chatllm.cpp Python Bindings

Located in: [`experiments/qwen3-asr-chatllm.cpp/`](experiments/qwen3-asr-chatllm.cpp/)

**Goal:** Implement a complete ASR pipeline using `chatllm.cpp` Python bindings instead of the CLI executable.

**Features:**
- **Streaming ASR:** Real-time transcription output.
- **Text Processing:** `zh-CN` to `zh-TW` conversion (via `opencc`) and Inverse Text Normalization (ITN).
- **Benchmarking:** Measures Time To First Token (TTFT), generation speed, memory usage, and Word Error Rate (WER).
- **Memory Safety:** Includes leak detection tests over multiple iterations.
- **Containerization:** Full Docker support for reproducible builds.

For detailed usage instructions, please refer to the [Experiment 1 README](experiments/qwen3-asr-chatllm.cpp/README.md).

### Experiment 2: Extend Experiement 1 to Compare Two Ways of Repeated Inference against Memory Leak
## Way 1: Destroy and Re-create the Object
## Way 2: Use restart() instead
