---
title: Qwen3-ASR VAD
emoji: "\U0001F399"
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Speech Recognition with VAD Chunking for Long Audio
---

# Qwen3-ASR 0.6B - Speech Recognition with VAD Chunking

Automatic Speech Recognition (ASR) using the **Qwen3-ASR** model with **TEN VAD** for handling long audio files.

## Features

- **VAD Chunking**: Automatic speech detection and chunking for hours-long audio
- **Live Streaming Transcription**: Real-time character-by-character output
- **Chunk Progress**: Visual progress indicator showing current chunk and time range
- **Language Detection**: Automatic detection of Chinese, English, Japanese, Korean
- **Inverse Text Normalization (ITN)**: Converts spoken numbers to written form
- **Forced Alignment**: Word-level timestamps synchronized with audio
- **zh-TW Conversion**: Traditional Chinese output via OpenCC
- **SRT Export**: Progressive subtitle generation with download

## How It Works

For audio **longer than 30 seconds**:

1. **Speech Detection**: TEN VAD identifies speech segments
2. **Smart Chunking**: Audio split into ~20s chunks at silence boundaries
3. **Sequential Processing**: Each chunk processed: ASR → ITN → Alignment
4. **Result Accumulation**: Timestamps offset and merged
5. **Progressive SRT**: Subtitles available as chunks complete

For audio **shorter than 30 seconds**: Single-pass pipeline (no VAD overhead)

## Usage

1. **Upload** an audio file (any length, MP3/WAV/etc.) or select a sample
2. Click **Transcribe** to start
3. Watch **chunk progress** and **live transcript** streaming
4. View **ITN result** and **zh-TW result** (shown at end)
5. **Download SRT** file for subtitles

## Technical Details

| Component | Technology |
|-----------|------------|
| **VAD** | TEN VAD via ONNX Runtime |
| **ASR Model** | Qwen3-ASR 0.6B (Q4_0) |
| **Aligner Model** | Qwen3 Forced Aligner 0.6B (Q4_0) |
| **Inference** | chatllm.cpp (CPU-only) |
| **Memory** | ~1.5GB peak |

## Chunking Strategy

- **Target chunk**: ~20 seconds
- **Max chunk**: 30 seconds (hard limit)
- **Boundary**: Speech-aware (splits at silence)
- **Silence handling**: Skipped in SRT output

## Pipeline

```
Long Audio
    ↓
[TEN VAD] → Speech Segments
    ↓
[Chunker] → ~20s Chunks
    ↓
For each chunk:
    [ASR] → [ITN] → [Jieba] → [Aligner]
    ↓
Accumulate (with timestamp offsets)
    ↓
[OpenCC] → zh-TW SRT
```

## Sample Audio

- **News (Chinese)**: ~2.7 minutes (uses VAD chunking)
- **Phone Number (zh-TW)**: ~30 seconds (single-pass, no chunking)

## Links

- [Qwen3-Audio GitHub](https://github.com/QwenLM/Qwen3-Audio)
- [TEN VAD GitHub](https://github.com/TEN-framework/ten-vad)
- [chatllm.cpp Fork](https://github.com/vieenrose/chatllm.cpp)
