# exp6: ASR + Forced Aligner Pipeline (zh-TW)

## Overview

Complete ASR pipeline with zh-TW output and SRT generation.

### Pipeline Flow

```
meeting-1min5s.mp3 (65s, Chinese)
       ↓
   ffmpeg → WAV (16kHz mono)
       ↓
   Qwen3-ASR 0.6B → live streaming transcript (zh-CN)
       ↓
   Chinese-ITN → number normalization
       ↓
   Jieba → word tokenization (filter punctuation)
       ↓
   Forced Aligner (delimiter="|") → word timestamps
       ↓
   OpenCC → zh-TW conversion
       ↓
   SRT output
```

## Features

- **Live Streaming**: Real-time transcript output
- **ITN**: Chinese number normalization
- **Tokenization**: Jieba word segmentation (punctuation filtered)
- **Alignment**: Word-level timestamps
- **zh-TW Output**: Traditional Chinese with OpenCC
- **SRT Format**: Standard subtitle format

## Files

- `run_pipeline.py` - Main pipeline script
- `Dockerfile` - Uses vieenrose/chatllm.cpp fork (memory leak fixed)
- `requirements.txt` - Dependencies

## Setup

### Prerequisites

```bash
ls models/qwen3-asr-0.6b-q4_0.bin
ls models/qwen3-forced-aligner-0.6b-q4_0.bin
ls samples/meeting-1min5s.mp3
```

### Docker Build

```bash
cd /home/luigi/Qwen3-ASR-0.6B-CPU
docker build --build-arg BUILD_THREADS=22 -t qwen3-asr-exp6:latest -f experiments/exp-6/Dockerfile .
```

### Docker Run

```bash
docker run --rm \
  -v /home/luigi/Qwen3-ASR-0.6B-CPU:/workspace \
  -w /workspace \
  qwen3-asr-exp6:latest \
  bash -c "cd experiments/exp-6 && python run_pipeline.py"
```

Results saved to:
- `/workspace/experiments/exp-6/results/output.srt`
- `/workspace/experiments/exp-6/results/pipeline_results.json`

## Output

### Sample SRT Output

```
1
00:00:00,320 --> 00:00:01,280
有關

2
00:00:01,280 --> 00:00:02,160
緯創

3
00:00:02,160 --> 00:00:03,040
資通
...
```

### Performance Metrics

| Metric | Description |
|--------|-------------|
| ASR TTFT | Time to first transcript token |
| ASR Speed | Characters per second |
| Aligner TTFT | Time to first alignment token |
| Total Time | End-to-end pipeline duration |
| Peak Memory | Maximum RSS during pipeline |

## Technical Details

### Forced Aligner Delimiter

Tokens are joined with `|` delimiter and aligner is configured with:
```
--set delimiter |
```

This ensures word-level timestamps match Jieba tokens.

### Punctuation Filtering

Jieba tokens that contain only punctuation are filtered:
```python
punctuation = set('，。、；：！？「」『』（）""''…—·')
filtered = [t for t in tokens if not all(c in punctuation for c in t)]
```

### OpenCC Conversion

Simplified Chinese (ASR output) → Traditional Chinese (zh-TW):
```python
converter = opencc.OpenCC('s2twp')
zh_tw = converter.convert(zh_cn)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| BUILD_THREADS | 22 | Threads for compilation |
| INFERENCE_THREADS | 6 | Threads for CPU inference |
| CONTEXT_LENGTH | 4096 | Context window size |

## chatllm.cpp Source

Uses fork with memory leak fixes:
```
https://github.com/vieenrose/chatllm.cpp/tree/feature/exp1-qwen3-asr
```
