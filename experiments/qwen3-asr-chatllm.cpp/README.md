# exp1: Qwen3-ASR with chatllm.cpp

## Overview

This experiment sets up a complete ASR (Automatic Speech Recognition) pipeline using the Qwen3-ASR model with CPU inference via chatllm.cpp. It includes Docker containerization, streaming transcription with zh-TW output, Chinese Inverse Text Normalization (ITN), and comprehensive benchmarking.

## Files

- `Dockerfile` - Docker image definition with minimal Python base
- `requirements.txt` - Python dependencies (opencc-python-reimplemented, psutil)
- `run_asr.py` - Main ASR inference script with benchmarking

## Setup

### Prerequisites

Ensure the model file exists:
```bash
ls models/qwen3-asr-0.6b-q4_0.bin
```

### Docker Build

```bash
cd /home/luigi/Qwen3-ASR-0.6B-CPU
docker build --build-arg BUILD_THREADS=22 -t qwen3-asr-exp1:latest -f experiments/qwen3-asr-chatllm.cpp/Dockerfile .
```

For faster builds, adjust `BUILD_THREADS` to match your CPU cores.

### Docker Run

Mount the entire workspace and run the benchmark:

```bash
docker run --rm \
  -v /home/luigi/Qwen3-ASR-0.6B-CPU:/workspace \
  -w /workspace \
  qwen3-asr-exp1:latest \
  bash -c "cd experiments/qwen3-asr-chatllm.cpp && python run_asr.py"
```

Results will be saved to `/workspace/experiments/qwen3-asr-chatllm.cpp/results/asr_results.json`.

### Local Run (without Docker)

```bash
# 1. Build the shared library
cd /home/luigi/Qwen3-ASR-0.6B-CPU/chatllm.cpp
mkdir -p build && cd build
cmake .. && make libchatllm

# 2. Install Python dependencies
cd /home/luigi/Qwen3-ASR-0.6B-CPU
pip install -r experiments/qwen3-asr-chatllm.cpp/requirements.txt

# 3. Run the benchmark
cd experiments/qwen3-asr-chatllm.cpp
python run_asr.py
```

## Benchmark Results

The `run_asr.py` script produces:

1. **TTFT (Time to First Token)** - Time from inference start to first output token
2. **Speed** - Characters generated per second
3. **Memory RSS** - Peak memory usage (MB)
4. **WER** - Word Error Rate vs ground truth (`0900073331`)

Results are saved to `results/asr_results.json` (includes per-iteration stats and summary).

### Sample Output

```
============================================================
Qwen3-ASR Inference Benchmark (exp1)
============================================================
Model: /workspace/models/qwen3-asr-0.6b-q4_0.bin
Audio: /workspace/samples/phoneNumber1-zh-TW.wav
Ground Truth: 0900073331
Inference Threads: 6
Iterations: 10

--- Iteration 1/10 ---
Transcribing audio (streaming output)...
零九零零零七三三三一
Result: 0900073331
TTFT: 4349.01 ms, Speed: 147686.76 chars/sec, WER: 0.00%, Memory: 1356.57 MB
...
```

## Transcription Flow

1. Load audio file (`samples/phoneNumber1-zh-TW.wav`)
2. Send to Qwen3-ASR model via chatllm.py binding
3. Stream output chunks in real-time
4. Convert each chunk: zh-CN (Simplified) -> zh-TW (Traditional TW)
5. Print streaming zh-TW output
6. After completion: Apply ITN (e.g., "零九零零" -> "0900")
7. Display final ITN result

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| BUILD_THREADS | 22 | Threads for chatllm.cpp compilation |
| INFERENCE_THREADS | 6 | Threads for CPU inference |
| CONTEXT_LENGTH | 4096 | Context window size for inference |

## Troubleshooting

### Docker build fails: "models/ directory not found"

Ensure `.dockerignore` includes an exception for the models directory:
```
# .dockerignore should contain:
!models/
```

### Library not found: libchatllm.so

For local runs, ensure the library is built:
```bash
cd chatllm.cpp/build && cmake .. && make libchatllm
```

The library should appear at `chatllm.cpp/bindings/libchatllm.so`.

## Known Issues

### Memory Leak (~50MB/iteration)

The current implementation shows memory growth of ~50MB per iteration due to GGML resources not being fully freed between runs. This is being tracked for a future fix. See `MEMORY_LEAK_ANALYSIS.md` for details.
