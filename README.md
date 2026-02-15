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

# Qwen3-ASR 0.6B CPU

Automatic Speech Recognition (ASR) using the **Qwen3-ASR** model with CPU inference powered by **chatllm.cpp**.

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
You have to perform alignment inference unquely via direct Python binding to call ChatLLM.cpp C++ library similarly to Qwen3-ASR DEMO before. 

### Experiment 5: Test Qwen3-ASR 0.6B + Forced Aligner for Memory Leak
Exp 5 is like combinaison of exp 1 and 4. You will perform 100 iterations of inferece 
to test memory leak. For each of iteration, transcribe jfk.wav with Qwen3-ASR 0.6B via 
Python binding to obtain transcript then use Qwen3 forced aligner to peform alignment on
transcript generated from Qwen3-ASR and jfk.wav. 

There is a constraint that You can only load one model at a time, so you are supposed to for each of iteration, load ASR model to transcribe, then unload it to load aligner to make alignment before unloading it for next iteration.

As before, report also performance metrics for each of models, like TTFT, genration speed. 
For ASR model, you report WER, and for alignment result, you manually verify its correctness.

You have to develop all your work in experiments/exp-5 as standalone project independent from others.
I already copied qwen3-asr-chatllm.cpp to exp-5 under experiments to help your start.

### Experiment 6: Extend exp 5 to support zh-TW

under experiments/exp-6, 
you extend exp-5 to follow a more refined pipeline schematised below in Python

any audio -> pyffmpeg -> wav
wav -> Qwen3-ASR 0.6B Q4_0 -> live transcript in streaming, detected language 
full transcript -> Chinese ITN -> itn-transcript
itn-transcript -> Jieba -> tokenzied-transcript
tokenzied-transcript -> Forced Aligner -> aligned-transcript
aligned-transcript -> opencc-python-reimplemented -> zh-TW transcript

1. test with ./samples/meeting-1min5s.mp3
2. you can only load one llm model at a time
3. show performance benchmark: TTFT, generation speed, memory overhead
4. inference with python binding uniquely to use ChatLLM.cpp C++ library
   built from https://github.com/vieenrose/chatllm.cpp/tree/feature/exp1-qwen3-asr
   where memory leak during load and unload models has been fixed
5. transcript result have to be shown in live streaming on screen
6. show alignment result in style of .srt on screen in the end

### Experiment 7: HuggingFace Spaces Deployment

Located in: [`experiments/exp-7/`](experiments/exp-7/)

**Goal:** Deploy the exp-6 pipeline to HuggingFace Spaces as an interactive WebUI.

**Features:**
- **Audio Upload**: Upload audio files via browser or select from samples
- **Live Streaming**: Character-by-character transcript display
- **Language Detection**: Shows detected language (Chinese, English, Japanese, Korean)
- **zh-TW Output**: Traditional Chinese conversion via OpenCC
- **SRT Preview**: View alignment results in subtitle format
- **SRT Download**: Export subtitles as .srt file
- **Sample Audio**: Quick demo with news-zh.mp3 and phoneNumber1-zh-TW.wav

**Technical Details:**
- **Framework**: Gradio 4.x with streaming support
- **Inference**: chatllm.cpp built from [vieenrose fork](https://github.com/vieenrose/chatllm.cpp/tree/feature/exp1-qwen3-asr)
- **Memory**: ~1.5GB peak (one model loaded at a time)
- **SDK**: Docker (port 7860)

For detailed usage instructions, please refer to the [Experiment 7 README](experiments/exp-7/README.md).

### Experiment 8: Extend Experiment 7 to handle hours-long audio
Start from a fresh copy of experiments/exp-7 named exp-8, 
use ten-vad via ONNXruntime to segment input audio into chunks of around 20 seconds,
so that each boundary between each pair of consecutive chunks is speech-aware,
then run pipeline you devleloped in exp-7 to perform pipeline chunk by chunk, 
with propre result accumulation and live streaming transcript display
