#!/usr/bin/env python3
# coding: utf-8
'''
Experiment 4: Compare two methods of repeated forced alignment inference for memory behavior.

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
MODEL_PATH = PROJECT_ROOT / 'models' / 'qwen3-forced-aligner-0.6b-q4_0.bin'
AUDIO_PATH = PROJECT_ROOT / 'samples' / 'jfk.wav'
TRANSCRIPT_PATH = PROJECT_ROOT / 'samples' / 'jfk.txt'

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

sys.path.insert(0, str(PROJECT_ROOT / 'chatllm.cpp' / 'bindings'))
sys.path.insert(0, str(PROJECT_ROOT / 'chatllm.cpp' / 'scripts'))

from chatllm import LibChatLLM, ChatLLM, LLMChatChunk, LLMChatDone


def load_transcript() -> str:
    with open(TRANSCRIPT_PATH) as f:
        return f.read().strip()


def get_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_llm_params() -> List[str]:
    return [
        '-m', str(MODEL_PATH),
        '-n', str(INFERENCE_THREADS),
        '-c', CONTEXT_LENGTH,
        '--multimedia_file_tags', '{{', '}}',
        '--set', 'language', 'English',
        '--set', 'format', 'json'
    ]


def run_alignment(lib: LibChatLLM, chat: ChatLLM, audio_path: str, transcript: str,
                  print_output: bool = False) -> Tuple[str, float, float]:
    queue = Queue()
    chat.out_queue = queue
    
    user_input = f'{{{{audio:{audio_path}}}}} {transcript}'
    
    chunks = []
    first_token_time = None
    start_time = time.time()
    
    result = lib.chat(chat._chat, user_input)
    if result != 0:
        raise Exception(f"Chat failed: {result}")
    
    while True:
        try:
            item = queue.get(timeout=60.0)
            if isinstance(item, LLMChatChunk):
                chunk = item.chunk
                if chunk is None:
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
    
    char_count = len(output.replace(' ', '').replace('\n', ''))
    gen_speed = char_count / gen_time if gen_time > 0 else 0
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
    
    return output, ttft_ms, gen_speed


def run_method_destroy_recreate(iterations: int = 100, print_output: bool = True) -> List[Dict[str, Any]]:
    print("=" * 70)
    print("Method 1: Destroy + Recreate ChatLLM each iteration")
    print("=" * 70)
    
    transcript = load_transcript()
    process = psutil.Process(os.getpid())
    bindings_path = str(PROJECT_ROOT / 'chatllm.cpp' / 'bindings')
    lib = LibChatLLM(bindings_path)
    llm_params = get_llm_params()
    
    results = []
    
    for i in range(iterations):
        if print_output:
            print(f"\n--- Iteration {i+1}/{iterations} ---")
        
        mem_before = get_memory_mb()
        chat = ChatLLM(lib, llm_params)
        mem_after_load = get_memory_mb()
        
        alignment_result, ttft_ms, gen_speed = run_alignment(
            lib, chat, str(AUDIO_PATH), transcript, print_output=print_output
        )
        
        mem_after_infer = get_memory_mb()
        
        chat.destroy()
        del chat
        
        mem_after_unload = get_memory_mb()
        
        if print_output:
            print(f"Alignment output length: {len(alignment_result)} chars")
            print(f"TTFT: {ttft_ms:.2f} ms, Speed: {gen_speed:.2f} chars/sec")
            print(f"Memory: {mem_before:.1f} → {mem_after_load:.1f} (load) → {mem_after_infer:.1f} (infer) → {mem_after_unload:.1f} (unload)")
        else:
            print(f"\r  Iteration {i+1}/{iterations} - TTFT: {ttft_ms:.0f}ms, Mem: {mem_after_unload:.0f}MB", end='', flush=True)
        
        results.append({
            'iteration': i + 1,
            'alignment_result': alignment_result[:200] if len(alignment_result) > 200 else alignment_result,
            'ttft_ms': ttft_ms,
            'gen_speed': gen_speed,
            'mem_before_mb': mem_before,
            'mem_after_load_mb': mem_after_load,
            'mem_after_infer_mb': mem_after_infer,
            'mem_after_unload_mb': mem_after_unload,
            'load_overhead_mb': mem_after_load - mem_before,
            'unload_release_mb': mem_after_load - mem_after_unload,
            'net_leak_mb': mem_after_unload - mem_before
        })
    
    del lib
    print()
    return results


def run_method_restart(iterations: int = 100, print_output: bool = True) -> List[Dict[str, Any]]:
    print("=" * 70)
    print("Method 2: Reuse ChatLLM, call restart() each iteration")
    print("=" * 70)
    
    transcript = load_transcript()
    process = psutil.Process(os.getpid())
    bindings_path = str(PROJECT_ROOT / 'chatllm.cpp' / 'bindings')
    lib = LibChatLLM(bindings_path)
    llm_params = get_llm_params()
    
    results = []
    
    mem_before_create = get_memory_mb()
    chat = ChatLLM(lib, llm_params)
    mem_after_create = get_memory_mb()
    
    if print_output:
        print(f"Initial ChatLLM creation: {mem_before_create:.1f} → {mem_after_create:.1f} MB (+{mem_after_create - mem_before_create:.1f} MB)")
    
    for i in range(iterations):
        if print_output:
            print(f"\n--- Iteration {i+1}/{iterations} ---")
        
        mem_before = get_memory_mb()
        
        alignment_result, ttft_ms, gen_speed = run_alignment(
            lib, chat, str(AUDIO_PATH), transcript, print_output=print_output
        )
        
        mem_after_infer = get_memory_mb()
        
        if print_output:
            print(f"Alignment output length: {len(alignment_result)} chars")
            print(f"TTFT: {ttft_ms:.2f} ms, Speed: {gen_speed:.2f} chars/sec")
            print(f"Memory: {mem_before:.1f} → {mem_after_infer:.1f} MB")
        else:
            print(f"\r  Iteration {i+1}/{iterations} - TTFT: {ttft_ms:.0f}ms, Mem: {mem_after_infer:.0f}MB", end='', flush=True)
        
        results.append({
            'iteration': i + 1,
            'alignment_result': alignment_result[:200] if len(alignment_result) > 200 else alignment_result,
            'ttft_ms': ttft_ms,
            'gen_speed': gen_speed,
            'mem_before_mb': mem_before,
            'mem_after_infer_mb': mem_after_infer,
            'net_growth_mb': mem_after_infer - mem_before
        })
        
        chat.restart()
    
    chat.destroy()
    del chat
    del lib
    
    print()
    return results


def print_comparison_table(results_way1: List[Dict], results_way2: List[Dict]):
    initial_mem_w1 = results_way1[0]['mem_before_mb']
    final_mem_w1 = results_way1[-1]['mem_after_unload_mb']
    growth_w1 = final_mem_w1 - initial_mem_w1
    growth_rate_w1 = growth_w1 / len(results_way1)
    avg_ttft_w1 = sum(r['ttft_ms'] for r in results_way1) / len(results_way1)
    avg_speed_w1 = sum(r['gen_speed'] for r in results_way1) / len(results_way1)
    avg_leak_w1 = sum(r['net_leak_mb'] for r in results_way1) / len(results_way1)
    
    initial_mem_w2 = results_way2[0]['mem_before_mb']
    final_mem_w2 = results_way2[-1]['mem_after_infer_mb']
    growth_w2 = final_mem_w2 - initial_mem_w2
    growth_rate_w2 = growth_w2 / len(results_way2)
    avg_ttft_w2 = sum(r['ttft_ms'] for r in results_way2) / len(results_way2)
    avg_speed_w2 = sum(r['gen_speed'] for r in results_way2) / len(results_way2)
    avg_growth_w2 = sum(r['net_growth_mb'] for r in results_way2) / len(results_way2)
    
    print()
    print("=" * 70)
    print("Comparison Results Summary")
    print("=" * 70)
    print()
    print(f"{'Metric':<35} {'Destroy+Recreate':>18} {'Restart()':>18}")
    print("-" * 70)
    print(f"{'Initial Memory (MB)':<35} {initial_mem_w1:>18.2f} {initial_mem_w2:>18.2f}")
    print(f"{'Final Memory (MB)':<35} {final_mem_w1:>18.2f} {final_mem_w2:>18.2f}")
    print(f"{'Memory Growth (MB)':<35} {growth_w1:>18.2f} {growth_w2:>18.2f}")
    print(f"{'Growth Rate (MB/iter)':<35} {growth_rate_w1:>18.4f} {growth_rate_w2:>18.4f}")
    print(f"{'Avg Net Leak/Growth per Iter (MB)':<35} {avg_leak_w1:>18.4f} {avg_growth_w2:>18.4f}")
    print("-" * 70)
    print(f"{'Avg TTFT (ms)':<35} {avg_ttft_w1:>18.2f} {avg_ttft_w2:>18.2f}")
    print(f"{'Avg Speed (chars/sec)':<35} {avg_speed_w1:>18.2f} {avg_speed_w2:>18.2f}")
    print()
    
    return {
        'destroy_recreate': {
            'initial_memory_mb': initial_mem_w1,
            'final_memory_mb': final_mem_w1,
            'memory_growth_mb': growth_w1,
            'growth_rate_mb_per_iter': growth_rate_w1,
            'avg_net_leak_mb': avg_leak_w1,
            'avg_ttft_ms': avg_ttft_w1,
            'avg_speed_chars_per_sec': avg_speed_w1
        },
        'restart': {
            'initial_memory_mb': initial_mem_w2,
            'final_memory_mb': final_mem_w2,
            'memory_growth_mb': growth_w2,
            'growth_rate_mb_per_iter': growth_rate_w2,
            'avg_net_growth_mb': avg_growth_w2,
            'avg_ttft_ms': avg_ttft_w2,
            'avg_speed_chars_per_sec': avg_speed_w2
        }
    }


def run_comparison(iterations: int = 100, print_output: bool = True):
    print("=" * 70)
    print("Experiment 4: Forced Aligner Memory Comparison")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Audio: {AUDIO_PATH}")
    print(f"Transcript: {load_transcript()[:50]}...")
    print(f"Inference Threads: {INFERENCE_THREADS}")
    print(f"Iterations per method: {iterations}")
    print()
    
    results_way1 = run_method_destroy_recreate(iterations, print_output)
    results_way2 = run_method_restart(iterations, print_output)
    
    summary = print_comparison_table(results_way1, results_way2)
    
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / 'alignment_comparison_results.json'
    
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'iterations': iterations,
                'model': str(MODEL_PATH),
                'audio': str(AUDIO_PATH),
                'transcript': load_transcript(),
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
    run_comparison(iterations=100, print_output=True)
