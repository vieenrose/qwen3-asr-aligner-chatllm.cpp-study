# exp4: Forced Aligner Memory Leak Comparison

## Overview

This experiment compares memory behavior of two approaches to repeated forced alignment inference with the Qwen3-ForcedAligner model via chatllm.cpp Python bindings.

| Method | Description |
|--------|-------------|
| **Way 1** | Destroy and Re-create `ChatLLM` object each iteration |
| **Way 2** | Reuse `ChatLLM` object, call `restart()` to reset context |

## What is Forced Alignment?

Forced alignment takes audio + transcript and outputs word-level timestamps:

```
Input:  audio.wav + "And so, my fellow Americans..."
Output: [{"start": 0.0, "end": 500.0, "text": "And"}, ...]
```

## Files

- `Dockerfile` - Docker image definition
- `requirements.txt` - Python dependencies (psutil)
- `run_alignment.py` - Main comparison script for two methods

## Setup

### Prerequisites

Ensure the model file exists:
```bash
ls models/qwen3-forced-aligner-0.6b-q4_0.bin
ls samples/jfk.wav
ls samples/jfk.txt
```

### Docker Build

```bash
cd /home/luigi/Qwen3-ASR-0.6B-CPU
docker build --build-arg BUILD_THREADS=22 -t qwen3-aligner-exp4:latest -f experiments/qwen3-aligner-chatllm.cpp/Dockerfile .
```

For faster builds, adjust `BUILD_THREADS` to match your CPU cores.

### Docker Run

Mount the entire workspace and run the comparison:

```bash
docker run --rm \
  -v /home/luigi/Qwen3-ASR-0.6B-CPU:/workspace \
  -w /workspace \
  qwen3-aligner-exp4:latest \
  bash -c "cd experiments/qwen3-aligner-chatllm.cpp && python run_alignment.py"
```

Results will be saved to `/workspace/experiments/qwen3-aligner-chatllm.cpp/results/alignment_comparison_results.json`.

### Local Run (without Docker)

```bash
# 1. Build the shared library
cd /home/luigi/Qwen3-ASR-0.6B-CPU/chatllm.cpp
mkdir -p build && cd build
cmake .. && make libchatllm

# 2. Install Python dependencies
cd /home/luigi/Qwen3-ASR-0.6B-CPU
pip install -r experiments/qwen3-aligner-chatllm.cpp/requirements.txt

# 3. Run the comparison
cd experiments/qwen3-aligner-chatllm.cpp
python run_alignment.py
```

## Benchmark Results

The `run_alignment.py` script runs 100 iterations per method and produces:

1. **Memory Growth** - Total RSS growth from first to last iteration
2. **Growth Rate** - Memory growth per iteration (MB/iter)
3. **TTFT (Time to First Token)** - Time from inference start to first output token
4. **Speed** - Characters generated per second

Results are saved to `results/alignment_comparison_results.json` (includes per-iteration stats and summary).

### Sample Output

```
======================================================================
Experiment 4: Forced Aligner Memory Comparison
======================================================================
Model: /workspace/models/qwen3-forced-aligner-0.6b-q4_0.bin
Audio: /workspace/samples/jfk.wav
Transcript: And so, my fellow Americans, ask not what yo...
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
Comparison Results Summary
======================================================================

Metric                                Destroy+Recreate         Restart()
----------------------------------------------------------------------
Initial Memory (MB)                         XXXX.XX          XXXX.XX
Final Memory (MB)                           XXXX.XX          XXXX.XX
Memory Growth (MB)                          XXXX.XX          XXXX.XX
Growth Rate (MB/iter)                         XX.XX            XX.XX
----------------------------------------------------------------------
Avg TTFT (ms)                               XXXX.XX          XXXX.XX
Avg Speed (chars/sec)                     XXXXXX.XX        XXXXXX.XX

Results saved to .../results/alignment_comparison_results.json
```

## Alignment Flow

1. Load audio file (`samples/jfk.wav`)
2. Load transcript (`samples/jfk.txt`)
3. Send to Qwen3-ForcedAligner model via chatllm.py binding
4. Stream output (JSON format with timestamps)

The difference between methods is how the ChatLLM object is managed between iterations.

## Model Configuration

The ForcedAligner uses these settings:

| Setting | Value | Description |
|---------|-------|-------------|
| `language` | `English` | Word splitting behavior |
| `format` | `json` | Output format (json or srt) |

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
