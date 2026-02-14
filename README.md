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

### Experiment 2: Compare Two Ways of Repeated Inference for Memory Leak

Located in: [`experiments/two-ways-of-repeated-inference/`](experiments/two-ways-of-repeated-inference/)

**Goal:** Compare memory behavior of two approaches to repeated inference:

| Method | Description |
|--------|-------------|
| **Way 1** | Destroy and Re-create `ChatLLM` object each iteration |
| **Way 2** | Reuse `ChatLLM` object, call `restart()` to reset context |

**Features:**
- **Side-by-side Comparison:** Runs both methods with 100 iterations each
- **Memory Tracking:** Measures RSS memory growth per iteration
- **Benchmark Metrics:** TTFT, generation speed, WER for both methods
- **Containerization:** Standalone Docker image

For detailed usage instructions, please refer to the [Experiment 2 README](experiments/two-ways-of-repeated-inference/README.md).

### Experiment 3: Test Model Swiching for Memory Leak
Benchmark for 100-iteation repeated inference. 
For each iteration, 
load Qwen3-ASR 0.6B model to transcribe phoneNumber1-zh-TW.wav, then unload it. 
Then load Qwen3-ASR 1.7B model to do the same, then unload it. 
0. Develope this experiment in experiments/model-switch/ from a fresh copy of qwen3-asr-chatllm.cpp, as standalone and independant project from exp1
1. Record memory overhead for each iteration, before, after loading and unloading each of models
2. Benchmark also reports performance metrics like WER, TTFT, generation speed, etc. 

### Experiment 4: Test Forced Aligner for Memory Leak
Like exp2, but with Qwen3 Forced Aligenr ‘qwen3-forced-aligner-0.6b-q4_0.bin’ in models, you have to alignment with input audio jfk.wav and transcript in jfk.txt for 100 iteration in two ways: 1. load then unload model 2. keep the model but call restart() between 2 inference.
Report also, TTFT, generation speed, memory usage.
