#!/usr/bin/env python3
# coding: utf-8
'''
Experiment 3: Model Switching Memory Benchmark

For each iteration:
1. Load Qwen3-ASR 0.6B → transcribe → unload
2. Load Qwen3-ASR 1.7B → transcribe → unload
3. Track memory at each phase: before load, after load, after unload
'''

import os
import sys
import time
import json
import psutil
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, Any, Tuple, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_06B_PATH = PROJECT_ROOT / 'models' / 'qwen3-asr-0.6b-q4_0.bin'
MODEL_17B_PATH = PROJECT_ROOT / 'models' / 'qwen3-asr-1.7b-q4_0.bin'
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


def load_ground_truth() -> str:
    with open(GROUND_TRUTH_PATH) as f:
        return f.read().strip()


def calculate_wer(reference: str, hypothesis: str) -> float:
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


def get_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_llm_params(model_path: Path) -> List[str]:
    return [
        '-m', str(model_path),
        '-n', str(INFERENCE_THREADS),
        '-c', CONTEXT_LENGTH,
        '--format', 'chat',
        '--set', 'language', ASR_LANGUAGE,
        '--multimedia_file_tags', '{{', '}}'
    ]


def transcribe_audio(lib: LibChatLLM, chat: ChatLLM, audio_path: str, 
                     print_output: bool = False) -> Tuple[str, float, float, str]:
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
            item = queue.get(timeout=60.0)
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


def run_model_switch_benchmark(iterations: int = 100, print_transcripts: bool = True) -> Dict[str, Any]:
    print("=" * 70)
    print("Experiment 3: Model Switching Memory Benchmark")
    print("=" * 70)
    print(f"Model 1 (0.6B): {MODEL_06B_PATH}")
    print(f"Model 2 (1.7B): {MODEL_17B_PATH}")
    print(f"Audio: {AUDIO_PATH}")
    print(f"Ground Truth: {load_ground_truth()}")
    print(f"Inference Threads: {INFERENCE_THREADS}")
    print(f"Iterations: {iterations}")
    print()
    
    ground_truth = load_ground_truth()
    bindings_path = str(PROJECT_ROOT / 'chatllm.cpp' / 'bindings')
    
    all_results = []
    
    for i in range(iterations):
        if print_transcripts:
            print(f"\n{'='*70}")
            print(f"Iteration {i+1}/{iterations}")
            print("=" * 70)
        
        iter_result = {
            'iteration': i + 1,
            'model_06b': {},
            'model_17b': {}
        }
        
        mem_baseline = get_memory_mb()
        iter_result['mem_baseline_mb'] = mem_baseline
        
        # Create fresh lib for this iteration
        lib = LibChatLLM(bindings_path)
        
        # ===== Model 0.6B =====
        if print_transcripts:
            print(f"\n[0.6B] Loading model...")
        
        mem_before_load_06b = get_memory_mb()
        
        llm_params_06b = get_llm_params(MODEL_06B_PATH)
        chat_06b = ChatLLM(lib, llm_params_06b)
        
        mem_after_load_06b = get_memory_mb()
        
        if print_transcripts:
            print(f"[0.6B] Transcribing...")
        
        itn_06b, ttft_06b, speed_06b, zh_tw_06b = transcribe_audio(
            lib, chat_06b, str(AUDIO_PATH), print_output=print_transcripts
        )
        wer_06b = calculate_wer(ground_truth, itn_06b)
        
        if print_transcripts:
            print(f"[0.6B] Result: {itn_06b}")
            print(f"[0.6B] TTFT: {ttft_06b:.2f}ms, Speed: {speed_06b:.2f} chars/sec, WER: {wer_06b:.2f}%")
        
        mem_after_infer_06b = get_memory_mb()
        
        chat_06b.destroy()
        del chat_06b
        
        mem_after_unload_06b = get_memory_mb()
        
        if print_transcripts:
            print(f"[0.6B] Memory: {mem_before_load_06b:.1f} → {mem_after_load_06b:.1f} (load) → {mem_after_infer_06b:.1f} (infer) → {mem_after_unload_06b:.1f} (unload)")
        
        iter_result['model_06b'] = {
            'transcript_zh_tw': zh_tw_06b,
            'itn_result': itn_06b,
            'ttft_ms': ttft_06b,
            'gen_speed': speed_06b,
            'wer': wer_06b,
            'mem_before_load_mb': mem_before_load_06b,
            'mem_after_load_mb': mem_after_load_06b,
            'mem_after_infer_mb': mem_after_infer_06b,
            'mem_after_unload_mb': mem_after_unload_06b,
            'load_overhead_mb': mem_after_load_06b - mem_before_load_06b,
            'unload_release_mb': mem_after_load_06b - mem_after_unload_06b,
            'net_leak_mb': mem_after_unload_06b - mem_before_load_06b
        }
        
        # ===== Model 1.7B =====
        if print_transcripts:
            print(f"\n[1.7B] Loading model...")
        
        mem_before_load_17b = get_memory_mb()
        
        llm_params_17b = get_llm_params(MODEL_17B_PATH)
        chat_17b = ChatLLM(lib, llm_params_17b)
        
        mem_after_load_17b = get_memory_mb()
        
        if print_transcripts:
            print(f"[1.7B] Transcribing...")
        
        itn_17b, ttft_17b, speed_17b, zh_tw_17b = transcribe_audio(
            lib, chat_17b, str(AUDIO_PATH), print_output=print_transcripts
        )
        wer_17b = calculate_wer(ground_truth, itn_17b)
        
        if print_transcripts:
            print(f"[1.7B] Result: {itn_17b}")
            print(f"[1.7B] TTFT: {ttft_17b:.2f}ms, Speed: {speed_17b:.2f} chars/sec, WER: {wer_17b:.2f}%")
        
        mem_after_infer_17b = get_memory_mb()
        
        chat_17b.destroy()
        del chat_17b
        
        mem_after_unload_17b = get_memory_mb()
        
        if print_transcripts:
            print(f"[1.7B] Memory: {mem_before_load_17b:.1f} → {mem_after_load_17b:.1f} (load) → {mem_after_infer_17b:.1f} (infer) → {mem_after_unload_17b:.1f} (unload)")
        
        iter_result['model_17b'] = {
            'transcript_zh_tw': zh_tw_17b,
            'itn_result': itn_17b,
            'ttft_ms': ttft_17b,
            'gen_speed': speed_17b,
            'wer': wer_17b,
            'mem_before_load_mb': mem_before_load_17b,
            'mem_after_load_mb': mem_after_load_17b,
            'mem_after_infer_mb': mem_after_infer_17b,
            'mem_after_unload_mb': mem_after_unload_17b,
            'load_overhead_mb': mem_after_load_17b - mem_before_load_17b,
            'unload_release_mb': mem_after_load_17b - mem_after_unload_17b,
            'net_leak_mb': mem_after_unload_17b - mem_before_load_17b
        }
        
        del lib
        
        iter_result['mem_end_mb'] = get_memory_mb()
        iter_result['total_leak_mb'] = iter_result['mem_end_mb'] - iter_result['mem_baseline_mb']
        
        all_results.append(iter_result)
        
        if not print_transcripts:
            print(f"\r  Iteration {i+1}/{iterations} - Mem: {iter_result['mem_end_mb']:.0f}MB", end='', flush=True)
    
    print()
    
    summary = print_summary_table(all_results)
    
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / 'model_switch_results.json'
    
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'iterations': iterations,
                'model_06b': str(MODEL_06B_PATH),
                'model_17b': str(MODEL_17B_PATH),
                'audio': str(AUDIO_PATH),
                'inference_threads': INFERENCE_THREADS
            },
            'summary': summary,
            'iterations': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return summary


def print_summary_table(results: List[Dict]) -> Dict[str, Any]:
    n = len(results)
    
    # 0.6B stats
    avg_load_06b = sum(r['model_06b']['load_overhead_mb'] for r in results) / n
    avg_unload_06b = sum(r['model_06b']['unload_release_mb'] for r in results) / n
    avg_leak_06b = sum(r['model_06b']['net_leak_mb'] for r in results) / n
    avg_ttft_06b = sum(r['model_06b']['ttft_ms'] for r in results) / n
    avg_speed_06b = sum(r['model_06b']['gen_speed'] for r in results) / n
    avg_wer_06b = sum(r['model_06b']['wer'] for r in results) / n
    
    # 1.7B stats
    avg_load_17b = sum(r['model_17b']['load_overhead_mb'] for r in results) / n
    avg_unload_17b = sum(r['model_17b']['unload_release_mb'] for r in results) / n
    avg_leak_17b = sum(r['model_17b']['net_leak_mb'] for r in results) / n
    avg_ttft_17b = sum(r['model_17b']['ttft_ms'] for r in results) / n
    avg_speed_17b = sum(r['model_17b']['gen_speed'] for r in results) / n
    avg_wer_17b = sum(r['model_17b']['wer'] for r in results) / n
    
    # Overall memory
    initial_mem = results[0]['mem_baseline_mb']
    final_mem = results[-1]['mem_end_mb']
    total_growth = final_mem - initial_mem
    growth_rate = total_growth / n
    
    print()
    print("=" * 70)
    print(f"Summary ({n} iterations)")
    print("=" * 70)
    print()
    print(f"{'Metric':<35} {'0.6B Model':>15} {'1.7B Model':>15}")
    print("-" * 70)
    print(f"{'Avg Load Overhead (MB)':<35} {avg_load_06b:>15.2f} {avg_load_17b:>15.2f}")
    print(f"{'Avg Unload Release (MB)':<35} {avg_unload_06b:>15.2f} {avg_unload_17b:>15.2f}")
    print(f"{'Avg Net Leak per Switch (MB)':<35} {avg_leak_06b:>15.2f} {avg_leak_17b:>15.2f}")
    print("-" * 70)
    print(f"{'Avg TTFT (ms)':<35} {avg_ttft_06b:>15.2f} {avg_ttft_17b:>15.2f}")
    print(f"{'Avg Speed (chars/sec)':<35} {avg_speed_06b:>15.2f} {avg_speed_17b:>15.2f}")
    print(f"{'Avg WER (%)':<35} {avg_wer_06b:>15.2f} {avg_wer_17b:>15.2f}")
    print()
    print("=" * 70)
    print("Overall Memory")
    print("=" * 70)
    print(f"{'Initial Memory (MB)':<35} {initial_mem:>15.2f}")
    print(f"{'Final Memory (MB)':<35} {final_mem:>15.2f}")
    print(f"{'Total Growth (MB)':<35} {total_growth:>15.2f}")
    print(f"{'Growth Rate (MB/iter)':<35} {growth_rate:>15.4f}")
    print()
    
    return {
        'model_06b': {
            'avg_load_overhead_mb': avg_load_06b,
            'avg_unload_release_mb': avg_unload_06b,
            'avg_net_leak_mb': avg_leak_06b,
            'avg_ttft_ms': avg_ttft_06b,
            'avg_speed_chars_per_sec': avg_speed_06b,
            'avg_wer_percent': avg_wer_06b
        },
        'model_17b': {
            'avg_load_overhead_mb': avg_load_17b,
            'avg_unload_release_mb': avg_unload_17b,
            'avg_net_leak_mb': avg_leak_17b,
            'avg_ttft_ms': avg_ttft_17b,
            'avg_speed_chars_per_sec': avg_speed_17b,
            'avg_wer_percent': avg_wer_17b
        },
        'overall': {
            'initial_memory_mb': initial_mem,
            'final_memory_mb': final_mem,
            'total_growth_mb': total_growth,
            'growth_rate_mb_per_iter': growth_rate
        }
    }


if __name__ == '__main__':
    run_model_switch_benchmark(iterations=100, print_transcripts=True)
