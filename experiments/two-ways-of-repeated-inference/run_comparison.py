#!/usr/bin/env python3
# coding: utf-8
'''
Experiment 2: Compare two methods of repeated inference for memory leak behavior.

Way 1: Destroy and Re-create ChatLLM object each iteration
Way 2: Reuse ChatLLM object, call restart() to reset context
'''

import os
import sys
import time
import json
import psutil
from pathlib import Path
from queue import Queue, Empty
from typing import List, Dict, Any, Tuple

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


def get_llm_params():
    return [
        '-m', str(MODEL_PATH),
        '-n', str(INFERENCE_THREADS),
        '-c', CONTEXT_LENGTH,
        '--format', 'chat',
        '--set', 'language', ASR_LANGUAGE,
        '--multimedia_file_tags', '{{', '}}'
    ]


def transcribe_audio(lib, chat, audio_path: str, print_output: bool = False) -> Tuple[str, float, float, str]:
    queue = Queue()
    chat.out_queue = queue
    
    user_input = '{{audio:' + audio_path + '}}'
    
    chunks = []
    first_token_time = None
    start_time = time.time()
    
    result = lib.chat(chat._chat, user_input)
    if result != 0:
        raise Exception(f"Chat failed: {result}")
    
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
                if print_output:
                    print(chunk, end='', flush=True)
            elif isinstance(item, LLMChatDone):
                break
        except Empty:
            break
    
    output = ''.join(chunks)
    end_time = time.time()
    
    if print_output:
        print()
    
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
    
    char_count = len(output.replace(' ', ''))
    gen_speed = char_count / gen_time if gen_time > 0 else 0
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
    
    return itn_result, ttft_ms, gen_speed, output_zh_tw


def run_method_destroy_recreate(iterations: int = 100, print_transcripts: bool = True) -> List[Dict[str, Any]]:
    print("=" * 70)
    print("Method 1: Destroy + Recreate ChatLLM each iteration")
    print("=" * 70)
    
    ground_truth = load_ground_truth()
    process = psutil.Process(os.getpid())
    bindings_path = str(PROJECT_ROOT / 'chatllm.cpp' / 'bindings')
    lib = LibChatLLM(bindings_path)
    llm_params = get_llm_params()
    
    results = []
    
    for i in range(iterations):
        if print_transcripts:
            print(f"\n--- Iteration {i+1}/{iterations} ---")
        
        chat = ChatLLM(lib, llm_params)
        itn_result, ttft_ms, gen_speed, zh_tw = transcribe_audio(
            lib, chat, str(AUDIO_PATH), print_output=print_transcripts
        )
        
        current_memory = process.memory_info().rss / 1024 / 1024
        wer = calculate_wer(ground_truth, itn_result)
        
        if print_transcripts:
            print(f"Result: {itn_result}")
            print(f"TTFT: {ttft_ms:.2f} ms, Speed: {gen_speed:.2f} chars/sec, WER: {wer:.2f}%, Memory: {current_memory:.2f} MB")
        else:
            print(f"\r  Iteration {i+1}/{iterations} - TTFT: {ttft_ms:.0f}ms, Mem: {current_memory:.0f}MB", end='', flush=True)
        
        results.append({
            'iteration': i + 1,
            'transcript_zh_tw': zh_tw,
            'itn_result': itn_result,
            'ttft_ms': ttft_ms,
            'gen_speed': gen_speed,
            'memory_mb': current_memory,
            'wer': wer
        })
        
        chat.destroy()
        del chat
    
    print()
    return results


def run_method_restart(iterations: int = 100, print_transcripts: bool = True) -> List[Dict[str, Any]]:
    print("=" * 70)
    print("Method 2: Reuse ChatLLM, call restart() each iteration")
    print("=" * 70)
    
    ground_truth = load_ground_truth()
    process = psutil.Process(os.getpid())
    bindings_path = str(PROJECT_ROOT / 'chatllm.cpp' / 'bindings')
    lib = LibChatLLM(bindings_path)
    llm_params = get_llm_params()
    
    results = []
    
    chat = ChatLLM(lib, llm_params)
    
    for i in range(iterations):
        if print_transcripts:
            print(f"\n--- Iteration {i+1}/{iterations} ---")
        
        itn_result, ttft_ms, gen_speed, zh_tw = transcribe_audio(
            lib, chat, str(AUDIO_PATH), print_output=print_transcripts
        )
        
        current_memory = process.memory_info().rss / 1024 / 1024
        wer = calculate_wer(ground_truth, itn_result)
        
        if print_transcripts:
            print(f"Result: {itn_result}")
            print(f"TTFT: {ttft_ms:.2f} ms, Speed: {gen_speed:.2f} chars/sec, WER: {wer:.2f}%, Memory: {current_memory:.2f} MB")
        else:
            print(f"\r  Iteration {i+1}/{iterations} - TTFT: {ttft_ms:.0f}ms, Mem: {current_memory:.0f}MB", end='', flush=True)
        
        results.append({
            'iteration': i + 1,
            'transcript_zh_tw': zh_tw,
            'itn_result': itn_result,
            'ttft_ms': ttft_ms,
            'gen_speed': gen_speed,
            'memory_mb': current_memory,
            'wer': wer
        })
        
        chat.restart()
    
    chat.destroy()
    del chat
    
    print()
    return results


def print_comparison_table(results_way1: List[Dict], results_way2: List[Dict]):
    initial_mem_w1 = results_way1[0]['memory_mb']
    final_mem_w1 = results_way1[-1]['memory_mb']
    growth_w1 = final_mem_w1 - initial_mem_w1
    growth_rate_w1 = growth_w1 / len(results_way1)
    avg_ttft_w1 = sum(r['ttft_ms'] for r in results_way1) / len(results_way1)
    avg_speed_w1 = sum(r['gen_speed'] for r in results_way1) / len(results_way1)
    avg_wer_w1 = sum(r['wer'] for r in results_way1) / len(results_way1)
    
    initial_mem_w2 = results_way2[0]['memory_mb']
    final_mem_w2 = results_way2[-1]['memory_mb']
    growth_w2 = final_mem_w2 - initial_mem_w2
    growth_rate_w2 = growth_w2 / len(results_way2)
    avg_ttft_w2 = sum(r['ttft_ms'] for r in results_way2) / len(results_way2)
    avg_speed_w2 = sum(r['gen_speed'] for r in results_way2) / len(results_way2)
    avg_wer_w2 = sum(r['wer'] for r in results_way2) / len(results_way2)
    
    print()
    print("=" * 70)
    print("Comparison Results Summary")
    print("=" * 70)
    print()
    print(f"{'Metric':<30} {'Destroy+Recreate':>18} {'Restart()':>18}")
    print("-" * 70)
    print(f"{'Initial Memory (MB)':<30} {initial_mem_w1:>18.2f} {initial_mem_w2:>18.2f}")
    print(f"{'Final Memory (MB)':<30} {final_mem_w1:>18.2f} {final_mem_w2:>18.2f}")
    print(f"{'Memory Growth (MB)':<30} {growth_w1:>18.2f} {growth_w2:>18.2f}")
    print(f"{'Growth Rate (MB/iter)':<30} {growth_rate_w1:>18.4f} {growth_rate_w2:>18.4f}")
    print("-" * 70)
    print(f"{'Avg TTFT (ms)':<30} {avg_ttft_w1:>18.2f} {avg_ttft_w2:>18.2f}")
    print(f"{'Avg Speed (chars/sec)':<30} {avg_speed_w1:>18.2f} {avg_speed_w2:>18.2f}")
    print(f"{'Avg WER (%)':<30} {avg_wer_w1:>18.2f} {avg_wer_w2:>18.2f}")
    print()
    
    return {
        'destroy_recreate': {
            'initial_memory_mb': initial_mem_w1,
            'final_memory_mb': final_mem_w1,
            'memory_growth_mb': growth_w1,
            'growth_rate_mb_per_iter': growth_rate_w1,
            'avg_ttft_ms': avg_ttft_w1,
            'avg_speed_chars_per_sec': avg_speed_w1,
            'avg_wer_percent': avg_wer_w1
        },
        'restart': {
            'initial_memory_mb': initial_mem_w2,
            'final_memory_mb': final_mem_w2,
            'memory_growth_mb': growth_w2,
            'growth_rate_mb_per_iter': growth_rate_w2,
            'avg_ttft_ms': avg_ttft_w2,
            'avg_speed_chars_per_sec': avg_speed_w2,
            'avg_wer_percent': avg_wer_w2
        }
    }


def run_comparison(iterations: int = 100, print_transcripts: bool = True):
    print("=" * 70)
    print("Experiment 2: Memory Leak Comparison")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Audio: {AUDIO_PATH}")
    print(f"Ground Truth: {load_ground_truth()}")
    print(f"Inference Threads: {INFERENCE_THREADS}")
    print(f"Iterations per method: {iterations}")
    print()
    
    results_way1 = run_method_destroy_recreate(iterations, print_transcripts)
    results_way2 = run_method_restart(iterations, print_transcripts)
    
    summary = print_comparison_table(results_way1, results_way2)
    
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / 'comparison_results.json'
    
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'iterations': iterations,
                'model': str(MODEL_PATH),
                'audio': str(AUDIO_PATH),
                'inference_threads': INFERENCE_THREADS
            },
            'summary': summary,
            'iterations': {
                'destroy_recreate': results_way1,
                'restart': results_way2
            }
        }, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    return summary


if __name__ == '__main__':
    run_comparison(iterations=100, print_transcripts=True)
