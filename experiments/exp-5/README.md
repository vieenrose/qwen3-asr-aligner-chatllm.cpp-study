# exp5: ASR + Forced Aligner Memory Test

## Overview

This experiment tests memory behavior when chaining ASR and Forced Aligner inference. **Constraint: Only one model loaded at a time.**

For each iteration:
1. Load ASR 0.6B → transcribe jfk.wav → unload
2. Load Aligner 0.6B → align transcript → unload
3. Track memory at each phase

## Files

- `Dockerfile` - Docker image definition
- `requirements.txt` - Python dependencies (psutil)
- `run_asr_align.py` - Main experiment script

## Setup

### Prerequisites

Ensure both model files exist:
```bash
ls models/qwen3-asr-0.6b-q4_0.bin
ls models/qwen3-forced-aligner-0.6b-q4_0.bin
ls samples/jfk.wav
ls samples/jfk.txt
```

### Docker Build

```bash
cd /home/luigi/Qwen3-ASR-0.6B-CPU
docker build --build-arg BUILD_THREADS=22 -t qwen3-asr-exp5:latest -f experiments/exp-5/Dockerfile .
```

### Docker Run

```bash
docker run --rm \
  -v /home/luigi/Qwen3-ASR-0.6B-CPU:/workspace \
  -w /workspace \
  qwen3-asr-exp5:latest \
  bash -c "cd experiments/exp-5 && python run_asr_align.py"
```

Results saved to `/workspace/experiments/exp-5/results/asr_align_results.json`.

### Local Run

```bash
# 1. Build the shared library
cd /home/luigi/Qwen3-ASR-0.6B-CPU/chatllm.cpp
mkdir -p build && cd build
cmake .. && make libchatllm

# 2. Install dependencies
cd /home/luigi/Qwen3-ASR-0.6B-CPU
pip install -r experiments/exp-5/requirements.txt

# 3. Run
cd experiments/exp-5
python run_asr_align.py
```

## Benchmark Results

The script runs 100 iterations and produces:

### Per-Model Metrics
- **TTFT** - Time to first token
- **Speed** - Characters per second
- **WER** - Word error rate (ASR only)

### Memory Metrics
- Load/unload overhead per model
- Net memory change per iteration
- Total growth rate

### Sample Output

```
======================================================================
Experiment 5: ASR + Forced Aligner Memory Test
======================================================================
ASR Model: qwen3-asr-0.6b-q4_0.bin
Aligner Model: qwen3-forced-aligner-0.6b-q4_0.bin
Audio: samples/jfk.wav
Ground Truth: And so, my fellow Americans...
Iterations: 100

======================================================================
Iteration 1/100
======================================================================

[ASR] Loading model...
[ASR] Transcribing...
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
[ASR] Transcript: And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
[ASR] TTFT: 4000.00ms, Speed: 150000.00 chars/sec, WER: 0.00%

[Aligner] Loading model...
[Aligner] Aligning transcript with audio...
[{"start":320,"end":560,"text":"And"},...]
[Summary] Memory: 34.0 → 550.0 MB (delta: +516.0 MB)

======================================================================
Summary (100 iterations)
======================================================================

Phase                            Load Overhead   Net Overhead
----------------------------------------------------------------------
ASR                                    XXX.XX          XXX.XX
Aligner                                XXX.XX          XXX.XX

Metric                                    ASR       Aligner
----------------------------------------------------------------------
Avg TTFT (ms)                           XXXX.XX       XXXX.XX
Avg WER (%)                                X.XX          N/A

======================================================================
Overall Memory
======================================================================
Initial Memory (MB)                        XX.XX
Final Memory (MB)                        XXXX.XX
Total Growth (MB)                        XXXX.XX
Growth Rate (MB/iter)                       X.XX
```

## Flow Per Iteration

```
┌─────────────────────────────────────────────────────────────┐
│ 1. mem_baseline = get_memory()                              │
│                                                             │
│ 2. Load ASR 0.6B                                            │
│ 3. Transcribe jfk.wav → transcript                          │
│ 4. Unload ASR                                               │
│                                                             │
│ 5. Load Aligner 0.6B                                        │
│ 6. Align (transcript + jfk.wav) → timestamps                │
│ 7. Unload Aligner                                           │
│                                                             │
│ 8. Record: memory deltas, metrics                           │
└─────────────────────────────────────────────────────────────┘
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| BUILD_THREADS | 22 | Threads for compilation |
| INFERENCE_THREADS | 6 | Threads for CPU inference |
| CONTEXT_LENGTH | 4096 | Context window size |

## Troubleshooting

### Library not found: libchatllm.so

```bash
cd chatllm.cpp/build && cmake .. && make libchatllm
```
