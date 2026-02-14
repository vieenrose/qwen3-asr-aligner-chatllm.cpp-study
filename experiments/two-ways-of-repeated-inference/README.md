# exp2: Compare Two Ways of Repeated Inference for Memory Leak

## Overview

This experiment compares memory behavior of two approaches to repeated inference with the Qwen3-ASR model via chatllm.cpp Python bindings.

| Method | Description |
|--------|-------------|
| **Way 1** | Destroy and Re-create `ChatLLM` object each iteration |
| **Way 2** | Reuse `ChatLLM` object, call `restart()` to reset context |

## Files

- `Dockerfile` - Docker image definition
- `requirements.txt` - Python dependencies (opencc-python-reimplemented, psutil)
- `run_asr.py` - Basic ASR inference (copied from exp1, for verification)
- `run_comparison.py` - Main comparison script for two methods

## Setup

### Prerequisites

Ensure the model file exists:
```bash
ls models/qwen3-asr-0.6b-q4_0.bin
```

### Docker Build

```bash
cd /home/luigi/Qwen3-ASR-0.6B-CPU
docker build --build-arg BUILD_THREADS=22 -t qwen3-asr-exp2:latest -f experiments/two-ways-of-repeated-inference/Dockerfile .
```

For faster builds, adjust `BUILD_THREADS` to match your CPU cores.

### Docker Run

Mount the entire workspace and run the comparison:

```bash
docker run --rm \
  -v /home/luigi/Qwen3-ASR-0.6B-CPU:/workspace \
  -w /workspace \
  qwen3-asr-exp2:latest \
  bash -c "cd experiments/two-ways-of-repeated-inference && python run_comparison.py"
```

Results will be saved to `/workspace/experiments/two-ways-of-repeated-inference/results/comparison_results.json`.

### Local Run (without Docker)

```bash
# 1. Build the shared library
cd /home/luigi/Qwen3-ASR-0.6B-CPU/chatllm.cpp
mkdir -p build && cd build
cmake .. && make libchatllm

# 2. Install Python dependencies
cd /home/luigi/Qwen3-ASR-0.6B-CPU
pip install -r experiments/two-ways-of-repeated-inference/requirements.txt

# 3. Run the comparison
cd experiments/two-ways-of-repeated-inference
python run_comparison.py
```

## Benchmark Results

The `run_comparison.py` script runs 100 iterations per method and produces:

1. **Memory Growth** - Total RSS growth from first to last iteration
2. **Growth Rate** - Memory growth per iteration (MB/iter)
3. **TTFT (Time to First Token)** - Time from inference start to first output token
4. **Speed** - Characters generated per second
5. **WER** - Word Error Rate vs ground truth (`0900073331`)

Results are saved to `results/comparison_results.json` (includes per-iteration stats and summary).

### Sample Output

```
======================================================================
Experiment 2: Memory Leak Comparison
======================================================================
Model: /workspace/models/qwen3-asr-0.6b-q4_0.bin
Audio: /workspace/samples/phoneNumber1-zh-TW.wav
Ground Truth: 0900073331
Inference Threads: 6
Iterations per method: 100

======================================================================
Method 1: Destroy + Recreate ChatLLM each iteration
======================================================================
  Iteration 100/100

======================================================================
Method 2: Reuse ChatLLM, call restart() each iteration
======================================================================
  Iteration 100/100

======================================================================
Comparison Results (100 iterations)
======================================================================

Metric                      Destroy+Recreate            Restart()
----------------------------------------------------------------------
Initial Memory (MB)               1356.05               1356.05
Final Memory (MB)                 1489.66               1489.66
Memory Growth (MB)                 133.61                 0.00
Growth Rate (MB/iter)               1.34                 0.00
Avg TTFT (ms)                     6665.00              6665.00
Avg Speed (chars/sec)           144000.00            144000.00
Final WER (%)                        0.00                 0.00

Results saved to .../results/comparison_results.json
```

## Transcription Flow

Both methods follow the same transcription flow:

1. Load audio file (`samples/phoneNumber1-zh-TW.wav`)
2. Send to Qwen3-ASR model via chatllm.py binding
3. Stream output chunks
4. Convert: zh-CN (Simplified) -> zh-TW (Traditional TW)
5. Apply ITN (e.g., "零九零零" -> "0900")

The difference is how the ChatLLM object is managed between iterations.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| BUILD_THREADS | 22 | Threads for chatllm.cpp compilation |
| INFERENCE_THREADS | 6 | Threads for CPU inference |
| CONTEXT_LENGTH | 4096 | Context window size for inference |

## Troubleshooting

### Library not found: libchatllm.so

For local runs, ensure the library is built:
```bash
cd chatllm.cpp/build && cmake .. && make libchatllm
```

The library should appear at `chatllm.cpp/bindings/libchatllm.so`.
