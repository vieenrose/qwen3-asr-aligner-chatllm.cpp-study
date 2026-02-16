#!/usr/bin/env python3
# coding: utf-8
'''
Experiment 7: ASR + Forced Aligner Pipeline with Streaming Support

Pipeline:
1. Audio -> WAV (ffmpeg)
2. Qwen3-ASR -> live streaming transcript + detected language
3. Chinese-ITN -> number normalization
4. Jieba -> word tokenization (filter punctuation)
5. Forced Aligner (delimiter="|") -> word timestamps
6. OpenCC -> zh-TW conversion
7. SRT output

Generator-based for real-time UI updates.
'''

import os
import sys
import time
import json
import tempfile
import subprocess
from pathlib import Path
from queue import Queue, Empty
from typing import Generator, List, Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASR_MODEL_PATH = PROJECT_ROOT / 'models' / 'qwen3-asr-0.6b-q4_0.bin'
ALIGNER_MODEL_PATH = PROJECT_ROOT / 'models' / 'qwen3-forced-aligner-0.6b-q4_0.bin'

INFERENCE_THREADS = int(os.getenv('INFERENCE_THREADS', '6'))
CONTEXT_LENGTH = os.getenv('CONTEXT_LENGTH', '4096')

sys.path.insert(0, str(PROJECT_ROOT / 'chatllm.cpp' / 'bindings'))
sys.path.insert(0, str(PROJECT_ROOT / 'chatllm.cpp' / 'scripts'))

from chatllm import LibChatLLM, ChatLLM, ChatLLMStreamer, LLMChatChunk, LLMChatDone

CHUNK_TIMEOUT_SEC = 300  # 5 minutes timeout

import jieba
import opencc

sys.path.insert(0, str(PROJECT_ROOT / 'Chinese-ITN'))
try:
    from chinese_itn import chinese_to_num
    ITN_AVAILABLE = True
except ImportError:
    ITN_AVAILABLE = False


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


def extract_language(raw_output: str) -> Tuple[str, str]:
    output = raw_output.strip()
    detected_lang = "Unknown"
    
    for lang in ['Chinese', 'English', 'Japanese', 'Korean', 'auto']:
        if output.startswith(lang):
            detected_lang = lang
            output = output[len(lang):].strip()
            break
    
    for tag in ['<asr_text>', '</asr_text>', 'language']:
        output = output.replace(tag, '')
    
    return detected_lang, output.strip()


def apply_itn(text: str) -> str:
    if ITN_AVAILABLE:
        return chinese_to_num(text)
    return text


def tokenize_with_jieba(text: str) -> List[str]:
    raw_tokens = list(jieba.cut(text))
    
    punctuation = set('，。、；：！？「」『』（）""''…—·,.!?;:\"\'()[]{}')
    
    filtered_tokens = []
    for token in raw_tokens:
        token = token.strip()
        if not token:
            continue
        if all(c in punctuation or c.isspace() for c in token):
            continue
        filtered_tokens.append(token)
    
    return filtered_tokens


def convert_to_zh_tw(text: str) -> str:
    converter = opencc.OpenCC('s2twp')
    return converter.convert(text)


def generate_srt_content(alignment: List[Dict]) -> str:
    lines = []
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
        
        lines.append(f"{i}")
        lines.append(f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms_rem:03d} --> "
                     f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms_rem:03d}")
        lines.append(text_zh_tw)
        lines.append("")
    
    return "\n".join(lines)


def run_pipeline_streaming(audio_path: str, hint_text: str = "") -> Generator[Dict[str, Any], None, None]:
    '''
    Generator that yields status updates for real-time UI updates.
    
    Yields dicts with keys:
    - stage: current stage name
    - message: human-readable status message
    - text: partial/complete text (for transcript stages)
    - language: detected language (after ASR)
    - srt: SRT content (at end)
    - metrics: performance metrics (at end)
    '''
    
    metrics = {
        'audio_duration_sec': 0,
        'asr_ttft_ms': 0,
        'asr_speed_chars_per_sec': 0,
        'aligner_ttft_ms': 0,
        'aligner_speed_chars_per_sec': 0,
        'total_time_sec': 0,
    }
    
    pipeline_start = time.time()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, 'audio.wav')
        
        yield {'stage': 'converting', 'message': 'Converting audio to WAV format...'}
        
        duration = convert_to_wav(audio_path, wav_path)
        metrics['audio_duration_sec'] = duration
        
        yield {'stage': 'converting', 'message': f'Audio duration: {duration:.2f}s'}
        
        bindings_path = str(PROJECT_ROOT / 'chatllm.cpp' / 'bindings')
        
        yield {'stage': 'loading_asr', 'message': 'Loading ASR model...'}
        
        lib_asr = LibChatLLM(bindings_path)
        
        llm_params = get_asr_params()
        chat = ChatLLM(lib_asr, llm_params)
        
        streamer = ChatLLMStreamer(chat)
        
        if hint_text and hint_text.strip():
            user_input = f'{hint_text.strip()} {{{{audio:{wav_path}}}}}'
        else:
            user_input = f'{{{{audio:{wav_path}}}}}'
        
        yield {'stage': 'transcribing', 'message': 'Transcribing...', 'text': '', 'language': ''}
        
        asr_start = time.time()
        first_token_time = None
        text_acc = ''
        detected_lang = 'Unknown'
        
        try:
            for output in streamer.chat(user_input):
                if time.time() - asr_start > CHUNK_TIMEOUT_SEC:
                    yield {'stage': 'error', 'message': f'ASR timed out after {CHUNK_TIMEOUT_SEC}s'}
                    return
                
                if isinstance(output, str):
                    text_acc += output
                    
                    for tag in ['<asr_text>', '</asr_text>', 'language']:
                        text_acc = text_acc.replace(tag, '')
                    
                    detected_lang, filtered = extract_language(text_acc)
                    
                    if first_token_time is None and output.strip():
                        first_token_time = time.time()
                    
                    yield {
                        'stage': 'transcribing',
                        'message': 'Transcribing...',
                        'text': filtered,
                        'language': detected_lang
                    }
        finally:
            streamer.terminate()
            chat.destroy()
        
        asr_end = time.time()
        
        detected_lang, transcript = extract_language(text_acc)
        
        if first_token_time:
            metrics['asr_ttft_ms'] = (first_token_time - asr_start) * 1000
            gen_time = asr_end - first_token_time
            metrics['asr_speed_chars_per_sec'] = len(text_acc) / gen_time if gen_time > 0 else 0
        
        del chat
        del lib_asr
        
        yield {
            'stage': 'transcribed',
            'message': 'Transcription complete',
            'text': transcript,
            'language': detected_lang
        }
        
        yield {'stage': 'itn', 'message': 'Applying Inverse Text Normalization...', 'text': transcript}
        
        itn_transcript = apply_itn(transcript)
        zh_tw_transcript = convert_to_zh_tw(itn_transcript)
        
        yield {
            'stage': 'itn_done',
            'message': 'ITN complete',
            'text': itn_transcript,
            'zh_tw_text': zh_tw_transcript,
            'language': detected_lang
        }
        
        yield {'stage': 'tokenizing', 'message': 'Tokenizing with Jieba...'}
        
        tokens = tokenize_with_jieba(itn_transcript)
        
        yield {'stage': 'tokenized', 'message': f'Tokenized: {len(tokens)} words'}
        
        yield {'stage': 'loading_aligner', 'message': 'Loading Forced Aligner model...'}
        
        lib_aligner = LibChatLLM(bindings_path)
        
        delimiter = '|'
        tokenized_text = delimiter.join(tokens)
        
        llm_params = get_aligner_params()
        chat = ChatLLM(lib_aligner, llm_params)
        
        streamer = ChatLLMStreamer(chat)
        
        user_input = f'{{{{audio:{wav_path}}}}} {tokenized_text}'
        
        yield {'stage': 'aligning', 'message': 'Aligning words to audio...'}
        
        align_start = time.time()
        first_token_time = None
        align_text_acc = ''
        
        try:
            for output in streamer.chat(user_input):
                if time.time() - align_start > CHUNK_TIMEOUT_SEC:
                    yield {'stage': 'error', 'message': f'Alignment timed out after {CHUNK_TIMEOUT_SEC}s'}
                    return
                
                if isinstance(output, str):
                    align_text_acc += output
                    if first_token_time is None:
                        first_token_time = time.time()
        finally:
            streamer.terminate()
            chat.destroy()
        
        align_end = time.time()
        
        if first_token_time:
            metrics['aligner_ttft_ms'] = (first_token_time - align_start) * 1000
            gen_time = align_end - first_token_time
            metrics['aligner_speed_chars_per_sec'] = len(align_text_acc) / gen_time if gen_time > 0 else 0
        
        del chat
        del lib_aligner
        
        try:
            alignment = json.loads(align_text_acc)
        except json.JSONDecodeError:
            alignment = []
        
        yield {'stage': 'aligned', 'message': f'Aligned {len(alignment)} words'}
        
        srt_content = generate_srt_content(alignment)
        
        pipeline_end = time.time()
        metrics['total_time_sec'] = pipeline_end - pipeline_start
        
        yield {
            'stage': 'done',
            'message': 'Pipeline complete!',
            'text': transcript,
            'itn_text': itn_transcript,
            'zh_tw_text': zh_tw_transcript,
            'language': detected_lang,
            'srt': srt_content,
            'alignment_count': len(alignment),
            'metrics': metrics
        }


def run_pipeline_simple(audio_path: str, hint_text: str = "") -> Dict[str, Any]:
    '''
    Non-streaming version that returns final results.
    Useful for testing or when streaming is not needed.
    '''
    result = {
        'transcript': '',
        'itn_transcript': '',
        'zh_tw_transcript': '',
        'language': 'Unknown',
        'srt': '',
        'alignment': [],
        'metrics': {}
    }
    
    for update in run_pipeline_streaming(audio_path, hint_text):
        if update['stage'] == 'transcribed':
            result['transcript'] = update.get('text', '')
            result['language'] = update.get('language', 'Unknown')
        elif update['stage'] == 'itn_done':
            result['itn_transcript'] = update.get('text', '')
        elif update['stage'] == 'done':
            result['zh_tw_transcript'] = update.get('zh_tw_text', '')
            result['srt'] = update.get('srt', '')
            result['alignment'] = update.get('alignment', [])
            result['metrics'] = update.get('metrics', {})
        elif update['stage'] == 'error':
            result['error'] = update.get('message', 'Unknown error')
            return result
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ASR + Forced Aligner Pipeline')
    parser.add_argument('audio', help='Path to audio file')
    parser.add_argument('--output', '-o', help='Output SRT file path')
    args = parser.parse_args()
    
    for update in run_pipeline_streaming(args.audio):
        stage = update.get('stage', '')
        message = update.get('message', '')
        
        if stage == 'transcribing':
            text = update.get('text', '')
            lang = update.get('language', '')
            print(f"\r[{stage}] lang={lang} text={text[-50:]}", end='', flush=True)
        elif stage == 'done':
            print(f"\n\n{message}")
            print(f"Language: {update.get('language')}")
            print(f"Metrics: {update.get('metrics')}")
            
            if args.output:
                srt_path = Path(args.output)
                srt_path.write_text(update.get('srt', ''), encoding='utf-8')
                print(f"SRT saved to: {srt_path}")
        else:
            print(f"[{stage}] {message}")
