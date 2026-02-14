#!/usr/bin/env python3
# coding: utf-8
'''
Experiment 5: ASR + Forced Aligner Memory Test

For each iteration:
1. Load ASR 0.6B → transcribe jfk.wav → unload
2. Load Aligner 0.6B → align transcript → unload
3. Track memory at each phase

Constraint: Only one model loaded at a time.
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
ASR_MODEL_PATH = PROJECT_ROOT / 'models' / 'qwen3-asr-0.6b-q4_0.bin'
ALIGNER_MODEL_PATH = PROJECT_ROOT / 'models' / 'qwen3-forced-aligner-0.6b-q4_0.bin'
AUDIO_PATH = PROJECT_ROOT / 'samples' / 'jfk.wav'
GROUND_TRUTH_PATH = PROJECT_ROOT / 'samples' / 'jfk.txt'

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


def load_ground_truth() -> str:
    with open(GROUND_TRUTH_PATH) as f:
        return f.read().strip()


def get_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def calculate_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
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
        '--set', 'language', 'English',
        '--set', 'format', 'json'
    ]


def run_inference(lib: LibChatLLM, chat: ChatLLM, user_input: str,
                  print_output: bool = False) -> Tuple[str, float, float]:
    queue = Queue()
    chat.out_queue = queue
    
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


def clean_asr_output(raw_output: str) -> str:
    for tag in ['<asr_text>', '</asr_text>', 'language']:
        raw_output = raw_output.replace(tag, '')
    output = raw_output.strip()
    if output.startswith('English'):
        output = output[7:].strip()
    return output


def run_asr(lib: LibChatLLM, audio_path: str, print_output: bool = False) -> Tuple[str, float, float]:
    user_input = f'{{{{audio:{audio_path}}}}}'
    raw_output, ttft_ms, gen_speed = run_inference(lib, None, user_input, print_output)
    transcript = clean_asr_output(raw_output)
    return transcript, ttft_ms, gen_speed


def run_alignment(lib: LibChatLLM, chat: ChatLLM, audio_path: str, transcript: str,
                  print_output: bool = False) -> Tuple[str, float, float]:
    user_input = f'{{{{audio:{audio_path}}}}} {transcript}'
    output, ttft_ms, gen_speed = run_inference(lib, chat, user_input, print_output)
    return output, ttft_ms, gen_speed


class ChatWrapper:
    def __init__(self, lib, params):
        self.lib = lib
        self.params = params
        self.chat = None
        self.out_queue = None
    
    def create(self):
        self.chat = ChatLLM(self.lib, self.params)
        self.chat.out_queue = self.chat.out_queue if hasattr(self.chat, 'out_queue') else None
    
    def destroy(self):
        if self.chat:
            self.chat.destroy()
            del self.chat
            self.chat = None


def run_asr_align_iteration(lib: LibChatLLM, audio_path: str, ground_truth: str,
                            print_output: bool = False) -> Dict[str, Any]:
    result = {}
    
    mem_baseline = get_memory_mb()
    result['mem_baseline_mb'] = mem_baseline
    
    # ===== ASR Phase =====
    if print_output:
        print("\n[ASR] Loading model...")
    
    mem_before_asr = get_memory_mb()
    
    asr_chat = ChatLLM(lib, get_asr_params())
    
    mem_after_asr_load = get_memory_mb()
    result['asr_load_overhead_mb'] = mem_after_asr_load - mem_before_asr
    
    if print_output:
        print("[ASR] Transcribing...")
    
    user_input = f'{{{{audio:{audio_path}}}}}'
    queue = Queue()
    asr_chat.out_queue = queue
    
    asr_chunks = []
    asr_first_token = None
    asr_start = time.time()
    
    lib.chat(asr_chat._chat, user_input)
    
    while True:
        try:
            item = queue.get(timeout=60.0)
            if isinstance(item, LLMChatChunk):
                chunk = item.chunk
                if chunk is None or chunk in ['<asr_text>', '</asr_text>', 'language']:
                    continue
                if asr_first_token is None:
                    asr_first_token = time.time()
                asr_chunks.append(chunk)
                if print_output:
                    print(chunk, end='', flush=True)
            elif isinstance(item, LLMChatDone):
                break
        except Empty:
            break
    
    if print_output:
        print()
    
    asr_raw = ''.join(asr_chunks)
    asr_end = time.time()
    
    transcript = clean_asr_output(asr_raw)
    asr_ttft = (asr_first_token - asr_start) * 1000 if asr_first_token else 0
    asr_speed = len(asr_raw) / (asr_end - asr_first_token) if asr_first_token else 0
    asr_wer = calculate_wer(ground_truth, transcript)
    
    mem_after_asr_infer = get_memory_mb()
    
    if print_output:
        print(f"[ASR] Transcript: {transcript}")
        print(f"[ASR] TTFT: {asr_ttft:.2f}ms, Speed: {asr_speed:.2f} chars/sec, WER: {asr_wer:.2f}%")
    
    asr_chat.destroy()
    del asr_chat
    
    mem_after_asr_unload = get_memory_mb()
    result['asr_unload_release_mb'] = mem_after_asr_load - mem_after_asr_unload
    result['asr_net_overhead_mb'] = mem_after_asr_unload - mem_before_asr
    
    result['asr'] = {
        'transcript': transcript,
        'ttft_ms': asr_ttft,
        'gen_speed': asr_speed,
        'wer': asr_wer,
        'mem_before_mb': mem_before_asr,
        'mem_after_load_mb': mem_after_asr_load,
        'mem_after_infer_mb': mem_after_asr_infer,
        'mem_after_unload_mb': mem_after_asr_unload
    }
    
    # ===== Aligner Phase =====
    if print_output:
        print("\n[Aligner] Loading model...")
    
    mem_before_aligner = get_memory_mb()
    
    aligner_chat = ChatLLM(lib, get_aligner_params())
    
    mem_after_aligner_load = get_memory_mb()
    result['aligner_load_overhead_mb'] = mem_after_aligner_load - mem_before_aligner
    
    if print_output:
        print("[Aligner] Aligning transcript with audio...")
    
    aligner_queue = Queue()
    aligner_chat.out_queue = aligner_queue
    
    aligner_input = f'{{{{audio:{audio_path}}}}} {transcript}'
    
    aligner_chunks = []
    aligner_first_token = None
    aligner_start = time.time()
    
    lib.chat(aligner_chat._chat, aligner_input)
    
    while True:
        try:
            item = aligner_queue.get(timeout=60.0)
            if isinstance(item, LLMChatChunk):
                chunk = item.chunk
                if chunk is None:
                    continue
                if aligner_first_token is None:
                    aligner_first_token = time.time()
                aligner_chunks.append(chunk)
                if print_output:
                    print(chunk, end='', flush=True)
            elif isinstance(item, LLMChatDone):
                break
        except Empty:
            break
    
    if print_output:
        print()
    
    alignment_output = ''.join(aligner_chunks)
    aligner_end = time.time()
    
    aligner_ttft = (aligner_first_token - aligner_start) * 1000 if aligner_first_token else 0
    aligner_speed = len(alignment_output) / (aligner_end - aligner_first_token) if aligner_first_token else 0
    
    mem_after_aligner_infer = get_memory_mb()
    
    aligner_chat.destroy()
    del aligner_chat
    
    mem_after_aligner_unload = get_memory_mb()
    result['aligner_unload_release_mb'] = mem_after_aligner_load - mem_after_aligner_unload
    result['aligner_net_overhead_mb'] = mem_after_aligner_unload - mem_before_aligner
    
    result['aligner'] = {
        'alignment_output': alignment_output[:500] if len(alignment_output) > 500 else alignment_output,
        'ttft_ms': aligner_ttft,
        'gen_speed': aligner_speed,
        'mem_before_mb': mem_before_aligner,
        'mem_after_load_mb': mem_after_aligner_load,
        'mem_after_infer_mb': mem_after_aligner_infer,
        'mem_after_unload_mb': mem_after_aligner_unload
    }
    
    result['mem_end_mb'] = get_memory_mb()
    result['total_memory_delta_mb'] = result['mem_end_mb'] - result['mem_baseline_mb']
    
    return result


def run_experiment(iterations: int = 100, print_output: bool = True):
    print("=" * 70)
    print("Experiment 5: ASR + Forced Aligner Memory Test")
    print("=" * 70)
    print(f"ASR Model: {ASR_MODEL_PATH}")
    print(f"Aligner Model: {ALIGNER_MODEL_PATH}")
    print(f"Audio: {AUDIO_PATH}")
    ground_truth = load_ground_truth()
    print(f"Ground Truth: {ground_truth}")
    print(f"Inference Threads: {INFERENCE_THREADS}")
    print(f"Iterations: {iterations}")
    print()
    
    bindings_path = str(PROJECT_ROOT / 'chatllm.cpp' / 'bindings')
    
    all_results = []
    
    for i in range(iterations):
        print(f"\n{'='*70}")
        print(f"Iteration {i+1}/{iterations}")
        print("=" * 70)
        
        lib = LibChatLLM(bindings_path)
        
        result = run_asr_align_iteration(lib, str(AUDIO_PATH), ground_truth, print_output)
        result['iteration'] = i + 1
        
        if not print_output:
            print(f"  Iteration {i+1}/{iterations} - Mem delta: {result['total_memory_delta_mb']:.1f}MB")
        
        print(f"\n[Summary] Memory: {result['mem_baseline_mb']:.1f} → {result['mem_end_mb']:.1f} MB (delta: +{result['total_memory_delta_mb']:.1f} MB)")
        
        all_results.append(result)
        
        del lib
    
    summary = print_summary_table(all_results)
    
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / 'asr_align_results.json'
    
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'iterations': iterations,
                'asr_model': str(ASR_MODEL_PATH),
                'aligner_model': str(ALIGNER_MODEL_PATH),
                'audio': str(AUDIO_PATH),
                'ground_truth': ground_truth,
                'inference_threads': INFERENCE_THREADS
            },
            'summary': summary,
            'iterations': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return summary


def print_summary_table(results: List[Dict]) -> Dict:
    n = len(results)
    
    # ASR stats
    avg_asr_load = sum(r['asr_load_overhead_mb'] for r in results) / n
    avg_asr_unload = sum(r['asr_unload_release_mb'] for r in results) / n
    avg_asr_net = sum(r['asr_net_overhead_mb'] for r in results) / n
    avg_asr_ttft = sum(r['asr']['ttft_ms'] for r in results) / n
    avg_asr_wer = sum(r['asr']['wer'] for r in results) / n
    
    # Aligner stats
    avg_aligner_load = sum(r['aligner_load_overhead_mb'] for r in results) / n
    avg_aligner_unload = sum(r['aligner_unload_release_mb'] for r in results) / n
    avg_aligner_net = sum(r['aligner_net_overhead_mb'] for r in results) / n
    avg_aligner_ttft = sum(r['aligner']['ttft_ms'] for r in results) / n
    
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
    print(f"{'Phase':<30} {'Load Overhead':>15} {'Net Overhead':>15}")
    print("-" * 70)
    print(f"{'ASR':<30} {avg_asr_load:>15.2f} {avg_asr_net:>15.2f}")
    print(f"{'Aligner':<30} {avg_aligner_load:>15.2f} {avg_aligner_net:>15.2f}")
    print()
    print(f"{'Metric':<30} {'ASR':>15} {'Aligner':>15}")
    print("-" * 70)
    print(f"{'Avg TTFT (ms)':<30} {avg_asr_ttft:>15.2f} {avg_aligner_ttft:>15.2f}")
    print(f"{'Avg WER (%)':<30} {avg_asr_wer:>15.2f} {'N/A':>15}")
    print()
    print("=" * 70)
    print("Overall Memory")
    print("=" * 70)
    print(f"{'Initial Memory (MB)':<30} {initial_mem:>15.2f}")
    print(f"{'Final Memory (MB)':<30} {final_mem:>15.2f}")
    print(f"{'Total Growth (MB)':<30} {total_growth:>15.2f}")
    print(f"{'Growth Rate (MB/iter)':<30} {growth_rate:>15.4f}")
    print()
    
    return {
        'asr': {
            'avg_load_overhead_mb': avg_asr_load,
            'avg_net_overhead_mb': avg_asr_net,
            'avg_ttft_ms': avg_asr_ttft,
            'avg_wer_percent': avg_asr_wer
        },
        'aligner': {
            'avg_load_overhead_mb': avg_aligner_load,
            'avg_net_overhead_mb': avg_aligner_net,
            'avg_ttft_ms': avg_aligner_ttft
        },
        'overall': {
            'initial_memory_mb': initial_mem,
            'final_memory_mb': final_mem,
            'total_growth_mb': total_growth,
            'growth_rate_mb_per_iter': growth_rate
        }
    }


if __name__ == '__main__':
    run_experiment(iterations=100, print_output=True)
