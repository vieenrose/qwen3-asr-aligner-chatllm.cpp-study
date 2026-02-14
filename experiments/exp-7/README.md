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

# Qwen3-ASR 0.6B - Speech Recognition with Forced Alignment

Automatic Speech Recognition (ASR) using the **Qwen3-ASR** model with CPU inference powered by **chatllm.cpp**.

## Features

- **Live Streaming Transcription**: Real-time character-by-character output
- **Language Detection**: Automatic detection of Chinese, English, Japanese, Korean
- **Inverse Text Normalization (ITN)**: Converts spoken numbers to written form
- **Forced Alignment**: Word-level timestamps synchronized with audio
- **zh-TW Conversion**: Traditional Chinese output via OpenCC
- **SRT Export**: Download subtitles for video editing

## Usage

1. **Upload** an audio file (MP3, WAV, etc.) or select a sample
2. Click **Transcribe** to start
3. Watch the **live transcript** stream in real-time
4. View the **detected language** and **zh-TW result**
5. **Download SRT** file for subtitles

## Technical Details

- **ASR Model**: Qwen3-ASR 0.6B (Q4_0 quantization)
- **Aligner Model**: Qwen3 Forced Aligner 0.6B (Q4_0 quantization)
- **Inference Engine**: [chatllm.cpp](https://github.com/vieenrose/chatllm.cpp) (CPU-only)
- **Memory**: ~1.5GB peak (one model loaded at a time)

## Pipeline

```
Audio -> WAV -> ASR (streaming) -> ITN -> Jieba -> Aligner -> OpenCC -> SRT
```

## Sample Audio

- **News (Chinese)**: ~2.7 minutes of Chinese news broadcast
- **Phone Number (zh-TW)**: ~30 seconds of Taiwanese Mandarin

## Links

- [Qwen3-Audio GitHub](https://github.com/QwenLM/Qwen3-Audio)
- [Qwen3-Audio Paper](https://arxiv.org/abs/2502.09345)
