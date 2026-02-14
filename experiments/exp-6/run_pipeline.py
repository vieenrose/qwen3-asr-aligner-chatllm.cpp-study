#!/usr/bin/env python3
# coding: utf-8
'''
Experiment 6: ASR + Forced Aligner Pipeline (zh-TW)

Pipeline:
1. MP3 → WAV (ffmpeg)
2. Qwen3-ASR → live streaming transcript
3. Chinese-ITN → number normalization
4. Jieba → word tokenization (filter punctuation)
5. Forced Aligner (delimiter="|") → word timestamps
6. OpenCC → zh-TW conversion
7. SRT output
'''

import os
import sys
import time
import json
import psutil
import subprocess
import tempfile
import re
from pathlib import Path
from queue import Queue, Empty
from typing import List, Tuple, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASR_MODEL_PATH = PROJECT_ROOT / 'models' / 'qwen3-asr-0.6b-q4_0.bin'
ALIGNER_MODEL_PATH = PROJECT_ROOT / 'models' / 'qwen3-forced-aligner-0.6b-q4_0.bin'
INPUT_AUDIO = PROJECT_ROOT / 'samples' / 'meeting-1min5s.mp3'
GROUND_TRUTH_PATH = PROJECT_ROOT / 'samples' / 'meeting-1min5s.txt'

INFERENCE_THREADS = int(os.getenv('INFERENCE_THREADS', '6'))
CONTEXT_LENGTH = os.getenv('CONTEXT_LENGTH', '4096')

sys.path.insert(0, str(PROJECT_ROOT / 'chatllm.cpp' / 'bindings'))
sys.path.insert(0, str(PROJECT_ROOT / 'chatllm.cpp' / 'scripts'))

from chatllm import LibChatLLM, ChatLLM, LLMChatChunk, LLMChatDone

import jieba
import opencc

sys.path.insert(0, str(PROJECT_ROOT / 'Chinese-ITN'))
try:
    from chinese_itn import chinese_to_num
    ITN_AVAILABLE = True
except ImportError:
    ITN_AVAILABLE = False


def get_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def convert_to_wav(input_path: str, output_path: str) -> float:
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-ar', '16000',
        '-ac', '1',
        '-acodec', 'pcm_s16le',
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', output_path],
        capture_output=True, text=True, check=True
    )
    return float(result.stdout.strip())


def get_asr_params() -> List[str]:
    return [
        '-m', str(ASR_MODEL_PATH),
        '-n', str(INFERENCE_THREADS),
        '-c', CONTEXT_LENGTH,
        '--format', 'chat',
        '--set', 'language', 'auto',
        '--multimedia_file_tags', '{{', '}}'
    ]


def get_aligner_params() -> List[str]:
    return [
        '-m', str(ALIGNER_MODEL_PATH),
        '-n', str(INFERENCE_THREADS),
        '-c', CONTEXT_LENGTH,
        '--multimedia_file_tags', '{{', '}}',
        '--set', 'language', 'Chinese',
        '--set', 'delimiter', '|',
        '--set', 'format', 'json'
    ]


def clean_asr_output(raw_output: str) -> str:
    for tag in ['<asr_text>', '</asr_text>', 'language']:
        raw_output = raw_output.replace(tag, '')
    output = raw_output.strip()
    if output.startswith('Chinese') or output.startswith('English'):
        for lang in ['Chinese', 'English', 'Japanese', 'Korean', 'auto']:
            if output.startswith(lang):
                output = output[len(lang):].strip()
                break
    return output


def run_asr(lib: LibChatLLM, wav_path: str) -> Tuple[str, float, float, str]:
    print("\n[ASR] Transcribing (live streaming)...")
    print("-" * 60)
    
    llm_params = get_asr_params()
    chat = ChatLLM(lib, llm_params)
    
    queue = Queue()
    chat.out_queue = queue
    
    user_input = f'{{{{audio:{wav_path}}}}}'
    
    chunks = []
    first_token_time = None
    start_time = time.time()
    
    result = lib.chat(chat._chat, user_input)
    if result != 0:
        raise Exception(f"Chat failed: {result}")
    
    while True:
        try:
            item = queue.get(timeout=120.0)
            if isinstance(item, LLMChatChunk):
                chunk = item.chunk
                if chunk is None or chunk in ['<asr_text>', '</asr_text>', 'language']:
                    continue
                if first_token_time is None:
                    first_token_time = time.time()
                chunks.append(chunk)
                print(f"\r[LIVE] {''.join(chunks)[-60:]}", end='', flush=True)
            elif isinstance(item, LLMChatDone):
                break
        except Empty:
            break
    
    print()
    print("-" * 60)
    
    raw_output = ''.join(chunks)
    end_time = time.time()
    
    transcript = clean_asr_output(raw_output)
    
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
    gen_time = end_time - first_token_time if first_token_time else end_time - start_time
    speed = len(raw_output) / gen_time if gen_time > 0 else 0
    
    chat.destroy()
    del chat
    
    return transcript, ttft_ms, speed, raw_output


def apply_itn(text: str) -> str:
    if ITN_AVAILABLE:
        return chinese_to_num(text)
    return text


def tokenize_with_jieba(text: str) -> List[str]:
    print("\n[Jieba] Tokenizing...")
    
    raw_tokens = list(jieba.cut(text))
    
    punctuation = set('，。、；：！？「」『』（）""''…—·,.!?;:\"\'()[]{}')
    
    filtered_tokens = []
    filtered_count = 0
    for token in raw_tokens:
        token = token.strip()
        if not token:
            continue
        if all(c in punctuation or c.isspace() for c in token):
            filtered_count += 1
            continue
        filtered_tokens.append(token)
    
    print(f"[Jieba] Raw tokens: {len(raw_tokens)}, after filtering: {len(filtered_tokens)} (removed {filtered_count} punctuation-only)")
    
    return filtered_tokens


def run_alignment(lib: LibChatLLM, wav_path: str, tokens: List[str]) -> Tuple[List[Dict], float, float]:
    print(f"\n[Aligner] Aligning {len(tokens)} words...")
    
    delimiter = '|'
    tokenized_text = delimiter.join(tokens)
    
    llm_params = get_aligner_params()
    chat = ChatLLM(lib, llm_params)
    
    queue = Queue()
    chat.out_queue = queue
    
    user_input = f'{{{{audio:{wav_path}}}}} {tokenized_text}'
    
    chunks = []
    first_token_time = None
    start_time = time.time()
    
    result = lib.chat(chat._chat, user_input)
    if result != 0:
        raise Exception(f"Alignment failed: {result}")
    
    while True:
        try:
            item = queue.get(timeout=120.0)
            if isinstance(item, LLMChatChunk):
                chunk = item.chunk
                if chunk is None:
                    continue
                if first_token_time is None:
                    first_token_time = time.time()
                chunks.append(chunk)
            elif isinstance(item, LLMChatDone):
                break
        except Empty:
            break
    
    output = ''.join(chunks)
    end_time = time.time()
    
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
    gen_time = end_time - first_token_time if first_token_time else end_time - start_time
    speed = len(output) / gen_time if gen_time > 0 else 0
    
    chat.destroy()
    del chat
    
    try:
        alignment = json.loads(output)
    except json.JSONDecodeError:
        print(f"[Aligner] Warning: Failed to parse JSON, raw output length: {len(output)}")
        alignment = []
    
    return alignment, ttft_ms, speed


def convert_to_zh_tw(text: str) -> str:
    converter = opencc.OpenCC('s2twp')
    return converter.convert(text)


def generate_srt(alignment: List[Dict], output_path: str):
    print(f"\n[SRT] Generating {len(alignment)} entries...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, entry in enumerate(alignment, 1):
            start_ms = entry.get('start', 0)
            end_ms = entry.get('end', 0)
            text = entry.get('text', '')
            
            text_zh_tw = convert_to_zh_tw(text)
            
            start_h = start_ms // 3600000
            start_m = (start_ms % 3600000) // 60000
            start_s = (start_ms % 60000) // 1000
            start_ms_rem = start_ms % 1000
            
            end_h = end_ms // 3600000
            end_m = (end_ms % 3600000) // 60000
            end_s = (end_ms % 60000) // 1000
            end_ms_rem = end_ms % 1000
            
            f.write(f"{i}\n")
            f.write(f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms_rem:03d} --> ")
            f.write(f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms_rem:03d}\n")
            f.write(f"{text_zh_tw}\n\n")
    
    print(f"[SRT] Saved to: {output_path}")


def print_srt_preview(srt_path: str, lines: int = 20):
    print("\n" + "=" * 70)
    print("SRT Output Preview (zh-TW)")
    print("=" * 70)
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    entries = content.strip().split('\n\n')
    
    for entry in entries[:lines // 4]:
        print(entry)
        print()
    
    if len(entries) > lines // 4:
        print(f"... ({len(entries) - lines // 4} more entries)")


def run_pipeline():
    print("=" * 70)
    print("Experiment 6: ASR + Forced Aligner Pipeline (zh-TW)")
    print("=" * 70)
    print(f"Input: {INPUT_AUDIO}")
    print(f"ASR Model: {ASR_MODEL_PATH}")
    print(f"Aligner Model: {ALIGNER_MODEL_PATH}")
    print(f"Inference Threads: {INFERENCE_THREADS}")
    print()
    
    mem_start = get_memory_mb()
    pipeline_start = time.time()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, 'audio.wav')
        
        print("[FFmpeg] Converting MP3 to WAV...")
        duration = convert_to_wav(str(INPUT_AUDIO), wav_path)
        print(f"[FFmpeg] Duration: {duration:.2f} seconds")
        print(f"[FFmpeg] Saved to: {wav_path}")
        
        mem_after_ffmpeg = get_memory_mb()
        
        bindings_path = str(PROJECT_ROOT / 'chatllm.cpp' / 'bindings')
        
        print("\n[ASR] Loading model...")
        lib_asr = LibChatLLM(bindings_path)
        
        mem_after_asr_load = get_memory_mb()
        
        transcript, asr_ttft, asr_speed, raw_output = run_asr(lib_asr, wav_path)
        
        del lib_asr
        
        mem_after_asr = get_memory_mb()
        
        print(f"\n[ASR] Transcript length: {len(transcript)} chars")
        print(f"[ASR] TTFT: {asr_ttft:.2f} ms")
        print(f"[ASR] Speed: {asr_speed:.2f} chars/sec")
        
        print("\n[ITN] Normalizing numbers...")
        itn_transcript = apply_itn(transcript)
        print(f"[ITN] Result length: {len(itn_transcript)} chars")
        
        tokens = tokenize_with_jieba(itn_transcript)
        
        print("\n[Aligner] Loading model...")
        lib_aligner = LibChatLLM(bindings_path)
        
        mem_after_aligner_load = get_memory_mb()
        
        alignment, aligner_ttft, aligner_speed = run_alignment(lib_aligner, wav_path, tokens)
        
        del lib_aligner
        
        mem_end = get_memory_mb()
        
        print(f"\n[Aligner] Aligned {len(alignment)} words")
        print(f"[Aligner] TTFT: {aligner_ttft:.2f} ms")
        print(f"[Aligner] Speed: {aligner_speed:.2f} chars/sec")
        
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        srt_path = results_dir / 'output.srt'
        
        generate_srt(alignment, str(srt_path))
        
        print_srt_preview(str(srt_path))
        
        pipeline_end = time.time()
        total_time = pipeline_end - pipeline_start
        
        print("\n" + "=" * 70)
        print("Performance Summary")
        print("=" * 70)
        print(f"{'Metric':<30} {'Value':>20}")
        print("-" * 70)
        print(f"{'Audio Duration':<30} {duration:>18.2f} s")
        print(f"{'ASR TTFT':<30} {asr_ttft:>18.2f} ms")
        print(f"{'ASR Speed':<30} {asr_speed:>18.2f} chars/sec")
        print(f"{'Aligner TTFT':<30} {aligner_ttft:>18.2f} ms")
        print(f"{'Aligner Speed':<30} {aligner_speed:>18.2f} chars/sec")
        print(f"{'Total Pipeline Time':<30} {total_time:>18.2f} s")
        print("-" * 70)
        print(f"{'Initial Memory':<30} {mem_start:>18.2f} MB")
        print(f"{'Peak Memory':<30} {max(mem_after_asr_load, mem_after_aligner_load):>18.2f} MB")
        print(f"{'Final Memory':<30} {mem_end:>18.2f} MB")
        print()
        
        json_output = results_dir / 'pipeline_results.json'
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump({
                'input': str(INPUT_AUDIO),
                'duration_seconds': duration,
                'transcript_zh_cn': transcript,
                'transcript_after_itn': itn_transcript,
                'token_count': len(tokens),
                'alignment_count': len(alignment),
                'metrics': {
                    'asr_ttft_ms': asr_ttft,
                    'asr_speed_chars_per_sec': asr_speed,
                    'aligner_ttft_ms': aligner_ttft,
                    'aligner_speed_chars_per_sec': aligner_speed,
                    'total_pipeline_time_sec': total_time,
                    'memory_start_mb': mem_start,
                    'memory_peak_mb': max(mem_after_asr_load, mem_after_aligner_load),
                    'memory_end_mb': mem_end
                },
                'alignment': alignment
            }, f, ensure_ascii=False, indent=2)
        
        print(f"JSON results saved to: {json_output}")


if __name__ == '__main__':
    run_pipeline()
