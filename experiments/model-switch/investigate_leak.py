#!/usr/bin/env python3
# coding: utf-8
'''
Memory Leak Investigation Tool for exp3

Uses Valgrind to detect memory leaks in the chatllm.cpp library
when switching between models.
'''

import os
import sys
import subprocess
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_06B_PATH = PROJECT_ROOT / 'models' / 'qwen3-asr-0.6b-q4_0.bin'
MODEL_17B_PATH = PROJECT_ROOT / 'models' / 'qwen3-asr-1.7b-q4_0.bin'
AUDIO_PATH = PROJECT_ROOT / 'samples' / 'phoneNumber1-zh-TW.wav'

INFERENCE_THREADS = int(os.getenv('INFERENCE_THREADS', '6'))
CONTEXT_LENGTH = os.getenv('CONTEXT_LENGTH', '4096')
ASR_LANGUAGE = os.getenv('ASR_LANGUAGE', 'Chinese')

sys.path.insert(0, str(PROJECT_ROOT / 'chatllm.cpp' / 'bindings'))
sys.path.insert(0, str(PROJECT_ROOT / 'chatllm.cpp' / 'scripts'))

from chatllm import LibChatLLM, ChatLLM, LLMChatChunk, LLMChatDone
from queue import Queue, Empty


def get_llm_params(model_path: Path):
    return [
        '-m', str(model_path),
        '-n', str(INFERENCE_THREADS),
        '-c', CONTEXT_LENGTH,
        '--format', 'chat',
        '--set', 'language', ASR_LANGUAGE,
        '--multimedia_file_tags', '{{', '}}'
    ]


def transcribe_once(lib, chat, audio_path: str):
    queue = Queue()
    chat.out_queue = queue
    
    user_input = '{{audio:' + audio_path + '}}'
    result = lib.chat(chat._chat, user_input)
    if result != 0:
        raise Exception(f"Chat failed: {result}")
    
    chunks = []
    while True:
        try:
            item = queue.get(timeout=60.0)
            if isinstance(item, LLMChatChunk):
                chunk = item.chunk
                if chunk in ['language', '<asr_text>', '</asr_text>'] or chunk is None:
                    continue
                chunks.append(chunk)
            elif isinstance(item, LLMChatDone):
                break
        except Empty:
            break
    
    return ''.join(chunks)


def test_single_model_load_unload(model_path: Path, iterations: int = 3):
    print(f"\n{'='*70}")
    print(f"Testing: {model_path.name} ({iterations} iterations)")
    print('='*70)
    
    bindings_path = str(PROJECT_ROOT / 'chatllm.cpp' / 'bindings')
    
    for i in range(iterations):
        lib = LibChatLLM(bindings_path)
        llm_params = get_llm_params(model_path)
        chat = ChatLLM(lib, llm_params)
        
        output = transcribe_once(lib, chat, str(AUDIO_PATH))
        print(f"  Iter {i+1}: {output[:20]}...")
        
        chat.destroy()
        del chat
        del lib
    
    print(f"  Completed {iterations} iterations")


def test_model_switching(iterations: int = 3):
    print(f"\n{'='*70}")
    print(f"Testing: Model switching ({iterations} iterations)")
    print('='*70)
    
    bindings_path = str(PROJECT_ROOT / 'chatllm.cpp' / 'bindings')
    
    for i in range(iterations):
        lib = LibChatLLM(bindings_path)
        
        # Load 0.6B
        chat_06b = ChatLLM(lib, get_llm_params(MODEL_06B_PATH))
        output_06b = transcribe_once(lib, chat_06b, str(AUDIO_PATH))
        print(f"  Iter {i+1} [0.6B]: {output_06b[:15]}...")
        chat_06b.destroy()
        del chat_06b
        
        # Load 1.7B
        chat_17b = ChatLLM(lib, get_llm_params(MODEL_17B_PATH))
        output_17b = transcribe_once(lib, chat_17b, str(AUDIO_PATH))
        print(f"  Iter {i+1} [1.7B]: {output_17b[:15]}...")
        chat_17b.destroy()
        del chat_17b
        
        del lib
    
    print(f"  Completed {iterations} switching iterations")


def run_with_valgrind(test_func, output_file: str = "valgrind_report.txt"):
    print("\n" + "="*70)
    print("Running with Valgrind memory leak check")
    print("="*70)
    print("This will be SLOW (10-100x slower than normal)")
    print()
    
    valgrind_cmd = [
        'valgrind',
        '--leak-check=full',
        '--show-leak-kinds=all',
        '--track-origins=yes',
        '--verbose',
        f'--log-file={output_file}',
        sys.executable,
        __file__,
        '--no-valgrind',
        '--quick-test'
    ]
    
    result = subprocess.run(valgrind_cmd, capture_output=True, text=True)
    
    print(f"\nValgrind completed. Report saved to: {output_file}")
    return output_file


def parse_valgrind_report(report_file: str):
    print(f"\n{'='*70}")
    print("Valgrind Report Summary")
    print('='*70)
    
    with open(report_file, 'r') as f:
        content = f.read()
    
    leaked_bytes = 0
    leaked_blocks = 0
    
    for line in content.split('\n'):
        if 'definitely lost:' in line.lower():
            print(f"  {line.strip()}")
            match = re.search(r'(\d+)\s*bytes.*(\d+)\s*blocks', line)
            if match:
                leaked_bytes += int(match.group(1))
                leaked_blocks += int(match.group(2))
        elif 'indirectly lost:' in line.lower():
            print(f"  {line.strip()}")
        elif 'possibly lost:' in line.lower():
            print(f"  {line.strip()}")
        elif 'still reachable:' in line.lower():
            print(f"  {line.strip()}")
        elif 'in use at exit:' in line.lower():
            print(f"  {line.strip()}")
    
    print(f"\n  Total definitely leaked: {leaked_bytes} bytes in {leaked_blocks} blocks")
    
    return {
        'leaked_bytes': leaked_bytes,
        'leaked_blocks': leaked_blocks
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Memory Leak Investigation')
    parser.add_argument('--valgrind', action='store_true', help='Run with Valgrind')
    parser.add_argument('--no-valgrind', dest='no_valgrind', action='store_true', help='Run without Valgrind (internal)')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test (2 iterations)')
    parser.add_argument('--full-test', action='store_true', help='Run full test (10 iterations)')
    args = parser.parse_args()
    
    iterations = 2 if args.quick_test else (10 if args.full_test else 3)
    
    if args.valgrind:
        report_file = str(Path(__file__).parent / 'results' / 'valgrind_report.txt')
        Path(__file__).parent.joinpath('results').mkdir(exist_ok=True)
        run_with_valgrind(None, report_file)
        parse_valgrind_report(report_file)
    else:
        print("Running memory leak tests (without Valgrind)...")
        test_single_model_load_unload(MODEL_06B_PATH, iterations)
        test_single_model_load_unload(MODEL_17B_PATH, iterations)
        test_model_switching(iterations)
        print("\n" + "="*70)
        print("Tests completed. To run with Valgrind, use: python investigate_leak.py --valgrind")
        print("="*70)


if __name__ == '__main__':
    main()
