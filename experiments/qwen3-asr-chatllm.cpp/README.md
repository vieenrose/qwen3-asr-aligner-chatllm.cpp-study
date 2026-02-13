# exp1: Qwen3-ASR with chatllm.cpp

## Overview

This experiment sets up a complete ASR (Automatic Speech Recognition) pipeline using the Qwen3-ASR model with CPU inference via chatllm.cpp. It includes Docker containerization, streaming transcription with zh-TW output, Chinese Inverse Text Normalization (ITN), and comprehensive benchmarking.

## Files

- `.env` - Environment configuration (build threads, inference threads, context length)
- `Dockerfile` - Docker image definition with minimal Python base
- `requirements.txt` - Python dependencies (opencc-python-reimplemented, psutil)
- `run_asr.py` - Main ASR inference script with benchmarking

## Setup

### Docker Build

```bash
cd /home/luigi/Qwen3-ASR-0.6B-CPU
docker build -t qwen3-asr-chatllm -f experiments/qwen3-asr-chatllm.cpp/Dockerfile .
```

### Docker Run

```bash
docker run --rm -v $(pwd)/experiments/qwen3-asr-chatllm.cpp/results:/app/experiment/results qwen3-asr-chatllm
```

### Local Run (without Docker)

```bash
cd /home/luigi/Qwen3-ASR-0.6B-CPU
pip install -r experiments/qwen3-asr-chatllm.cpp/requirements.txt
cd experiments/qwen3-asr-chatllm.cpp
python run_asr.py
```

## Benchmark Results

The `run_asr.py` script produces:

1. **Time to 1st token** - Time from inference start to first output token
2. **Generation speed** - Tokens generated per second
3. **Total time** - Complete inference time
4. **Peak memory usage** - RSS memory peak (MB)
5. **WER** - Word Error Rate vs ground truth (`0900073331`)

Results are saved to `asr_results.json` (includes per-iteration stats and summary).

## Transcription Flow

1. Load audio file (`phoneNumber1-zh-TW.wav`)
2. Send to Qwen3-ASR model via chatllm.py binding
3. Stream output chunks in real-time
4. Convert each chunk: zh-CN (Simplified) → zh-TW (Traditional TW)
5. Print streaming zh-TW output
6. After completion: Apply ITN (e.g., "一五八七" → "1587")
7. Display final ITN result

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| BUILD_THREADS | 8 | Threads for chatllm.cpp compilation |
| INFERENCE_THREADS | 4 | Threads for inference |
| CONTEXT_LENGTH | 4096 | Context window size for inference |
