# exp3: Model Switching Memory Benchmark

## Overview

This experiment benchmarks memory behavior when repeatedly switching between two ASR models. For each iteration:
1. Load Qwen3-ASR 0.6B → transcribe → unload
2. Load Qwen3-ASR 1.7B → transcribe → unload

Tracks memory at each phase: before load, after load, after inference, after unload.

## Files

- `Dockerfile` - Docker image definition
- `requirements.txt` - Python dependencies (opencc-python-reimplemented, psutil)
- `run_asr.py` - Basic ASR inference (copied from exp1, for verification)
- `run_model_switch.py` - Main model switching benchmark script

## Setup

### Prerequisites

Ensure both model files exist:
```bash
ls models/qwen3-asr-0.6b-q4_0.bin
ls models/qwen3-asr-1.7b-q4_0.bin
```

### Docker Build

```bash
cd /home/luigi/Qwen3-ASR-0.6B-CPU
docker build --build-arg BUILD_THREADS=22 -t qwen3-asr-exp3:latest -f experiments/model-switch/Dockerfile .
```

For faster builds, adjust `BUILD_THREADS` to match your CPU cores.

### Docker Run

Mount the entire workspace and run the benchmark:

```bash
docker run --rm \
  -v /home/luigi/Qwen3-ASR-0.6B-CPU:/workspace \
  -w /workspace \
  qwen3-asr-exp3:latest \
  bash -c "cd experiments/model-switch && python run_model_switch.py"
```

Results will be saved to `/workspace/experiments/model-switch/results/model_switch_results.json`.

### Local Run (without Docker)

```bash
# 1. Build the shared library
cd /home/luigi/Qwen3-ASR-0.6B-CPU/chatllm.cpp
mkdir -p build && cd build
cmake .. && make libchatllm

# 2. Install Python dependencies
cd /home/luigi/Qwen3-ASR-0.6B-CPU
pip install -r experiments/model-switch/requirements.txt

# 3. Run the benchmark
cd experiments/model-switch
python run_model_switch.py
```

## Benchmark Results

The `run_model_switch.py` script runs 100 iterations and produces:

### Per-Model Metrics
- **Load Overhead (MB)** - Memory increase after loading model
- **Unload Release (MB)** - Memory decrease after unloading model
- **Net Leak per Switch (MB)** - Memory not released after unload
- **TTFT (ms)** - Time to first token
- **Speed (chars/sec)** - Generation speed
- **WER (%)** - Word Error Rate

### Overall Metrics
- **Initial/Final Memory** - Total memory at start and end
- **Total Growth** - Overall memory growth
- **Growth Rate** - Memory growth per iteration

### Sample Output

```
======================================================================
Experiment 3: Model Switching Memory Benchmark
======================================================================
Model 1 (0.6B): /workspace/models/qwen3-asr-0.6b-q4_0.bin
Model 2 (1.7B): /workspace/models/qwen3-asr-1.7b-q4_0.bin
Audio: /workspace/samples/phoneNumber1-zh-TW.wav
Ground Truth: 0900073331
Inference Threads: 6
Iterations: 100

======================================================================
Iteration 1/100
======================================================================

[0.6B] Loading model...
[0.6B] Transcribing...
零九零零零七三三三一
[0.6B] Result: 0900073331
[0.6B] TTFT: 4000.00ms, Speed: 150000.00 chars/sec, WER: 0.00%
[0.6B] Memory: 800.0 → 1500.0 (load) → 1500.0 (infer) → 850.0 (unload)

[1.7B] Loading model...
[1.7B] Transcribing...
零九零零零七三三三一
[1.7B] Result: 0900073331
[1.7B] TTFT: 5000.00ms, Speed: 180000.00 chars/sec, WER: 0.00%
[1.7B] Memory: 850.0 → 2500.0 (load) → 2500.0 (infer) → 900.0 (unload)

======================================================================
Summary (100 iterations)
======================================================================

Metric                                  0.6B Model       1.7B Model
----------------------------------------------------------------------
Avg Load Overhead (MB)                       700.00          1650.00
Avg Unload Release (MB)                      650.00          1600.00
Avg Net Leak per Switch (MB)                  50.00            50.00
----------------------------------------------------------------------
Avg TTFT (ms)                               4000.00          5000.00
Avg Speed (chars/sec)                     150000.00         180000.00
Avg WER (%)                                    0.00             0.00

======================================================================
Overall Memory
======================================================================
Initial Memory (MB)                          800.00
Final Memory (MB)                           5800.00
Total Growth (MB)                           5000.00
Growth Rate (MB/iter)                          50.00
```

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
