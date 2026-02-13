#!/usr/bin/env python3
# coding: utf-8
'''
Qwen3-ASR Inference using chatllm.cpp Python binding with streaming output.
'''

import os
import sys
import time
import json
import base64
import tracemalloc
import psutil
from pathlib import Path
from queue import Queue, Empty
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'qwen3-asr-0.6b-q4_0.bin'
AUDIO_PATH = PROJECT_ROOT / 'samples' / 'phoneNumber1-zh-TW.wav'
GROUND_TRUTH_PATH = PROJECT_ROOT / 'samples' / 'phoneNumber1-zh-TW.txt'

ENV_FILE = Path(__file__).parent / '.env'
if ENV_FILE.exists():
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

INFERENCE_THREADS = int(os.getenv('INFERENCE_THREADS', '6'))
CONTEXT_LENGTH = os.getenv('CONTEXT_LENGTH', '4096')
ASR_LANGUAGE = os.getenv('ASR_LANGUAGE', 'Chinese')

sys.path.insert(0, str(PROJECT_ROOT / 'chatllm.cpp' / 'bindings'))
sys.path.insert(0, str(PROJECT_ROOT / 'chatllm.cpp' / 'scripts'))

from chatllm import LibChatLLM, ChatLLM, LLMChatChunk, LLMChatDone

try:
    import opencc
    OPENCC_AVAILABLE = True
except ImportError:
    OPENCC_AVAILABLE = False

try:
    sys.path.insert(0, str(PROJECT_ROOT / 'Chinese-ITN'))
    from chinese_itn import chinese_to_num
    ITN_AVAILABLE = True
except ImportError:
    ITN_AVAILABLE = False


def load_ground_truth():
    with open(GROUND_TRUTH_PATH) as f:
        return f.read().strip()


def calculate_wer(reference, hypothesis):
    ref_words = list(reference)
    hyp_words = list(hypothesis)
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    edit_distance = dp[m][n]
    wer = edit_distance / max(len(ref_words), 1) if len(ref_words) > 0 else 0
    return wer * 100


def run_asr(iterations: int = 1):
    ground_truth = load_ground_truth()
    model_path = str(MODEL_PATH)
    audio_path = str(AUDIO_PATH)

    print("=" * 60)
    print("Qwen3-ASR Inference Benchmark (exp1)")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Audio: {audio_path}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Inference Threads: {INFERENCE_THREADS}")
    print(f"Iterations: {iterations}")
    print()

    process = psutil.Process(os.getpid())

    # Create LibChatLLM and ChatLLM once to reuse them
    bindings_path = str(PROJECT_ROOT / 'chatllm.cpp' / 'bindings')
    lib = LibChatLLM(bindings_path)
    
    llm_params = [
        '-m', model_path,
        '-n', str(INFERENCE_THREADS),
        '-c', CONTEXT_LENGTH,
        '--format', 'chat',
        '--set', 'language', ASR_LANGUAGE,
        '--multimedia_file_tags', '{{', '}}'
    ]
    chat = None

    chat = None
    try:
        all_results = []
        for i in range(iterations):
            print(f"--- Iteration {i+1}/{iterations} ---")
            
            # Create a NEW ChatLLM object per iteration
            # This reuses the same LibChatLLM (weights are loaded once in GGML)
            # but creates/destroys the Chat object to test for memory leaks
            # and avoid the context state issue.
            chat = ChatLLM(lib, llm_params)
            queue = Queue()
            chat.out_queue = queue
            
            print("Transcribing audio (streaming output)...")
            start_time = time.time()
            
            user_input = '{{audio:' + audio_path + '}}'
            result = lib.chat(chat._chat, user_input)
            if result != 0: raise Exception(f"Chat failed: {result}")

            chunks = []
            first_token_time = None
            while True:
                try:
                    item = queue.get(timeout=30.0)
                    if isinstance(item, LLMChatChunk):
                        chunk = item.chunk
                        if chunk in ['language', '<asr_text>', '</asr_text>'] or chunk is None:
                            continue
                        if first_token_time is None:
                            first_token_time = time.time()
                        chunks.append(chunk)
                        print(chunk, end='', flush=True)
                    elif isinstance(item, LLMChatDone):
                        break
                except Empty:
                    break
            
            output = ''.join(chunks)
            print()
            
            end_time = time.time()
            current_memory = process.memory_info().rss / 1024 / 1024
            
            total_time = end_time - start_time
            gen_time = end_time - first_token_time if first_token_time else total_time

            if OPENCC_AVAILABLE:
                try:
                    converter = opencc.OpenCC('s2twp')
                    output_zh_tw = converter.convert(output)
                except Exception:
                    output_zh_tw = output
            else:
                output_zh_tw = output

            if ITN_AVAILABLE:
                itn_result = chinese_to_num(output_zh_tw)
            else:
                itn_result = output_zh_tw

            wer = calculate_wer(ground_truth, itn_result)
            char_count = len(output.replace(' ', ''))
            gen_speed = char_count / gen_time if gen_time > 0 else 0
            ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0

            print(f"Result: {itn_result}")
            print(f"TTFT: {ttft_ms:.2f} ms, Speed: {gen_speed:.2f} chars/sec, WER: {wer:.2f}%, Memory: {current_memory:.2f} MB")
            print()

            all_results.append({
                'iteration': i + 1,
                'ttft_ms': ttft_ms,
                'gen_speed': gen_speed,
                'memory_mb': current_memory,
                'wer': wer
            })
            
            # CRITICAL: Destroy the chat object to free memory!
            chat.destroy()
            del chat

        print("=" * 60)
        print("Final Benchmark Summary")
        print("=" * 60)
        avg_ttft = sum(r['ttft_ms'] for r in all_results) / iterations
        avg_speed = sum(r['gen_speed'] for r in all_results) / iterations
        final_memory = all_results[-1]['memory_mb']
        
        print(f"Average TTFT: {avg_ttft:.2f} ms")
        print(f"Average Speed: {avg_speed:.2f} chars/sec")
        print(f"Final Memory RSS: {final_memory:.2f} MB")
        print(f"Final WER: {all_results[-1]['wer']:.2f}%")
        
        output_file = Path(__file__).parent / 'asr_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'summary': {
                    'avg_ttft_ms': avg_ttft,
                    'avg_speed_chars_per_sec': avg_speed,
                    'final_memory_mb': final_memory,
                    'final_wer_percent': all_results[-1]['wer']
                },
                'iterations': all_results
            }, f, indent=2)
        print(f"Results saved to {output_file}")

        return all_results

    except Exception as e:
        print(f"ERROR: {e}")
        raise e


if __name__ == '__main__':
    run_asr(iterations=10)
