#!/usr/bin/env python3
# coding: utf-8
'''
Experiment 8: Chunked ASR + Forced Aligner Pipeline

Extends exp-7 to handle long audio by:
1. Detecting speech segments with TEN VAD
2. Chunking audio into ~20s segments with speech-aware boundaries
3. Processing each chunk through the ASR + alignment pipeline
4. Accumulating results with proper timestamp offsets

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
from typing import Generator, List, Dict, Any, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASR_MODEL_PATH = PROJECT_ROOT / 'models' / 'qwen3-asr-0.6b-q4_0.bin'
ALIGNER_MODEL_PATH = PROJECT_ROOT / 'models' / 'qwen3-forced-aligner-0.6b-q4_0.bin'
VAD_MODEL_PATH = PROJECT_ROOT / 'models' / 'ten-vad.onnx'

INFERENCE_THREADS = int(os.getenv('INFERENCE_THREADS', '6'))
CONTEXT_LENGTH = os.getenv('CONTEXT_LENGTH', '4096')

CHUNK_THRESHOLD_SEC = 30.0
TARGET_CHUNK_DURATION_SEC = 20.0
MAX_CHUNK_DURATION_SEC = 30.0

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

from chunker import AudioChunker, get_audio_duration


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


def format_timestamp(ms: int) -> str:
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms_rem = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms_rem:03d}"


def generate_srt_content(alignment: List[Dict], offset_ms: int = 0) -> str:
    lines = []
    for i, entry in enumerate(alignment, 1):
        start_ms = entry.get('start', 0) + offset_ms
        end_ms = entry.get('end', 0) + offset_ms
        text = entry.get('text', '')
        
        text_zh_tw = convert_to_zh_tw(text)
        
        lines.append(f"{i}")
        lines.append(f"{format_timestamp(start_ms)} --> {format_timestamp(end_ms)}")
        lines.append(text_zh_tw)
        lines.append("")
    
    return "\n".join(lines)


def run_single_chunk_asr(
    wav_path: str,
    bindings_path: str,
    lib_asr: LibChatLLM
) -> Generator[Dict[str, Any], None, None]:
    '''Run ASR on a single chunk. Returns (transcript, detected_lang).'''
    
    llm_params = get_asr_params()
    chat = ChatLLM(lib_asr, llm_params)
    
    queue = Queue()
    chat.out_queue = queue
    
    user_input = f'{{{{audio:{wav_path}}}}}'
    
    result = lib_asr.chat(chat._chat, user_input)
    if result != 0:
        chat.destroy()
        yield {'stage': 'error', 'message': f'ASR failed with code: {result}'}
        return
    
    chunks = []
    while True:
        try:
            item = queue.get(timeout=300.0)
            if isinstance(item, LLMChatChunk):
                chunk = item.chunk
                if chunk is None or chunk in ['<asr_text>', '</asr_text>', 'language']:
                    continue
                chunks.append(chunk)
                raw_output = ''.join(chunks)
                detected_lang, _ = extract_language(raw_output)
                yield {
                    'stage': 'chunk_transcribing',
                    'text': raw_output,
                    'language': detected_lang
                }
            elif isinstance(item, LLMChatDone):
                break
        except Empty:
            break
    
    raw_output = ''.join(chunks)
    detected_lang, transcript = extract_language(raw_output)
    
    chat.destroy()
    
    yield {
        'stage': 'chunk_transcribed',
        'text': transcript,
        'language': detected_lang
    }


def run_single_chunk_alignment(
    wav_path: str,
    itn_transcript: str,
    bindings_path: str,
    lib_aligner: LibChatLLM
) -> Generator[Dict[str, Any], None, None]:
    '''Run forced alignment on a single chunk. Returns alignment list.'''
    
    tokens = tokenize_with_jieba(itn_transcript)
    
    if not tokens:
        yield {'stage': 'chunk_aligned', 'alignment': []}
        return
    
    llm_params = get_aligner_params()
    chat = ChatLLM(lib_aligner, llm_params)
    
    queue = Queue()
    chat.out_queue = queue
    
    delimiter = '|'
    tokenized_text = delimiter.join(tokens)
    user_input = f'{{{{audio:{wav_path}}}}} {tokenized_text}'
    
    result = lib_aligner.chat(chat._chat, user_input)
    if result != 0:
        chat.destroy()
        yield {'stage': 'error', 'message': f'Alignment failed with code: {result}'}
        return
    
    chunks = []
    while True:
        try:
            item = queue.get(timeout=300.0)
            if isinstance(item, LLMChatChunk):
                chunk = item.chunk
                if chunk is None:
                    continue
                chunks.append(chunk)
            elif isinstance(item, LLMChatDone):
                break
        except Empty:
            break
    
    output = ''.join(chunks)
    chat.destroy()
    
    try:
        alignment = json.loads(output)
    except json.JSONDecodeError:
        alignment = []
    
    yield {'stage': 'chunk_aligned', 'alignment': alignment}


def run_chunked_pipeline_streaming(audio_path: str) -> Generator[Dict[str, Any], None, None]:
    '''
    Chunked pipeline for long audio.
    
    1. Detect speech segments with VAD
    2. Create chunks (~20s each)
    3. Process each chunk: ASR -> ITN -> Alignment
    4. Accumulate results with timestamp offsets
    '''
    
    metrics = {
        'audio_duration_sec': 0,
        'total_chunks': 0,
        'total_asr_chars': 0,
        'total_alignment_words': 0,
        'avg_asr_ttft_ms': 0,
        'avg_asr_speed_chars_per_sec': 0,
        'total_time_sec': 0,
    }
    
    pipeline_start = time.time()
    
    accumulated_transcript = ""
    accumulated_itn = ""
    accumulated_alignment = []
    detected_language = "Unknown"
    
    asr_ttft_samples = []
    asr_speed_samples = []
    
    yield {'stage': 'preparing', 'message': 'Analyzing audio and detecting speech segments...'}
    
    try:
        chunker = AudioChunker(
            audio_path,
            target_chunk_duration_s=TARGET_CHUNK_DURATION_SEC,
            max_chunk_duration_s=MAX_CHUNK_DURATION_SEC,
            vad_model_path=str(VAD_MODEL_PATH) if VAD_MODEL_PATH.exists() else None
        )
        chunks = chunker.prepare()
    except Exception as e:
        yield {'stage': 'error', 'message': f'Failed to prepare chunks: {e}'}
        return
    
    metrics['audio_duration_sec'] = chunker.total_duration_s
    metrics['total_chunks'] = len(chunks)
    
    if not chunks:
        yield {'stage': 'error', 'message': 'No speech detected in audio'}
        chunker.cleanup()
        return
    
    yield {
        'stage': 'chunks_ready',
        'message': f'Found {len(chunks)} speech chunks ({chunker.total_duration_s:.1f}s total)',
        'total_chunks': len(chunks),
        'audio_duration': chunker.total_duration_s
    }
    
    bindings_path = str(PROJECT_ROOT / 'chatllm.cpp' / 'bindings')
    
    yield {'stage': 'loading_asr', 'message': 'Loading ASR model...'}
    
    lib_asr = LibChatLLM(bindings_path)
    
    yield {'stage': 'loading_aligner', 'message': 'Loading Forced Aligner model...'}
    
    lib_aligner = LibChatLLM(bindings_path)
    
    for chunk_idx, (start_ms, end_ms) in enumerate(chunks):
        chunk_info = chunker.get_chunk_info(chunk_idx)
        
        yield {
            'stage': 'processing_chunk',
            'message': f'Processing chunk {chunk_idx + 1}/{len(chunks)} ({chunk_info["start_s"]:.0f}s - {chunk_info["end_s"]:.0f}s)',
            'chunk_index': chunk_idx,
            'total_chunks': len(chunks),
            'chunk_start_s': chunk_info['start_s'],
            'chunk_end_s': chunk_info['end_s']
        }
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            chunk_wav_path = tmp.name
        
        try:
            chunker.get_chunk_audio(chunk_idx, chunk_wav_path)
            
            chunk_asr_start = time.time()
            first_token_time = None
            chunk_transcript = ""
            
            for update in run_single_chunk_asr(chunk_wav_path, bindings_path, lib_asr):
                if update['stage'] == 'chunk_transcribing':
                    partial_text = update.get('text', '')
                    detected_language = update.get('language', detected_language)
                    
                    yield {
                        'stage': 'chunk_transcribing',
                        'message': f'Chunk {chunk_idx + 1}/{len(chunks)}: Transcribing...',
                        'partial_transcript': partial_text,
                        'accumulated_transcript': accumulated_transcript,
                        'language': detected_language,
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks)
                    }
                    
                    if first_token_time is None and partial_text:
                        first_token_time = time.time()
                    
                elif update['stage'] == 'chunk_transcribed':
                    chunk_transcript = update.get('text', '')
                    detected_language = update.get('language', detected_language)
                    
                elif update['stage'] == 'error':
                    yield update
                    return
            
            chunk_asr_end = time.time()
            
            if first_token_time:
                ttft_ms = (first_token_time - chunk_asr_start) * 1000
                gen_time = chunk_asr_end - first_token_time
                speed = len(chunk_transcript) / gen_time if gen_time > 0 else 0
                asr_ttft_samples.append(ttft_ms)
                asr_speed_samples.append(speed)
            
            accumulated_transcript += chunk_transcript
            
            yield {
                'stage': 'chunk_itn',
                'message': f'Chunk {chunk_idx + 1}/{len(chunks)}: Applying ITN...',
                'accumulated_transcript': accumulated_transcript,
                'chunk_index': chunk_idx,
                'total_chunks': len(chunks)
            }
            
            chunk_itn = apply_itn(chunk_transcript)
            accumulated_itn += chunk_itn
            
            yield {
                'stage': 'chunk_itn_done',
                'message': f'Chunk {chunk_idx + 1}/{len(chunks)}: ITN complete',
                'accumulated_itn': accumulated_itn,
                'chunk_index': chunk_idx,
                'total_chunks': len(chunks)
            }
            
            yield {
                'stage': 'chunk_aligning',
                'message': f'Chunk {chunk_idx + 1}/{len(chunks)}: Aligning...',
                'chunk_index': chunk_idx,
                'total_chunks': len(chunks)
            }
            
            for update in run_single_chunk_alignment(chunk_wav_path, chunk_itn, bindings_path, lib_aligner):
                if update['stage'] == 'chunk_aligned':
                    chunk_alignment = update.get('alignment', [])
                    
                    for entry in chunk_alignment:
                        entry['start'] = entry.get('start', 0) + start_ms
                        entry['end'] = entry.get('end', 0) + start_ms
                    
                    accumulated_alignment.extend(chunk_alignment)
                    metrics['total_alignment_words'] += len(chunk_alignment)
                    
                elif update['stage'] == 'error':
                    yield update
                    return
            
            yield {
                'stage': 'chunk_complete',
                'message': f'Chunk {chunk_idx + 1}/{len(chunks)}: Complete',
                'accumulated_transcript': accumulated_transcript,
                'accumulated_itn': accumulated_itn,
                'alignment_count': len(accumulated_alignment),
                'chunk_index': chunk_idx,
                'total_chunks': len(chunks),
                'srt': generate_srt_content(accumulated_alignment)
            }
            
        finally:
            if os.path.exists(chunk_wav_path):
                os.unlink(chunk_wav_path)
    
    lib_asr = None
    lib_aligner = None
    
    chunker.cleanup()
    
    zh_tw_transcript = convert_to_zh_tw(accumulated_itn)
    
    metrics['total_asr_chars'] = len(accumulated_transcript)
    metrics['avg_asr_ttft_ms'] = sum(asr_ttft_samples) / len(asr_ttft_samples) if asr_ttft_samples else 0
    metrics['avg_asr_speed_chars_per_sec'] = sum(asr_speed_samples) / len(asr_speed_samples) if asr_speed_samples else 0
    metrics['total_time_sec'] = time.time() - pipeline_start
    
    final_srt = generate_srt_content(accumulated_alignment)
    
    yield {
        'stage': 'done',
        'message': f'Complete! Processed {len(chunks)} chunks in {metrics["total_time_sec"]:.1f}s',
        'text': accumulated_transcript,
        'itn_text': accumulated_itn,
        'zh_tw_text': zh_tw_transcript,
        'language': detected_language,
        'srt': final_srt,
        'alignment_count': len(accumulated_alignment),
        'metrics': metrics
    }


def run_pipeline_streaming(audio_path: str) -> Generator[Dict[str, Any], None, None]:
    '''
    Main entry point - routes to chunked or single-pass based on duration.
    '''
    duration = get_audio_duration(audio_path)
    
    if duration < CHUNK_THRESHOLD_SEC:
        yield from run_single_pass_pipeline_streaming(audio_path)
    else:
        yield from run_chunked_pipeline_streaming(audio_path)


def run_single_pass_pipeline_streaming(audio_path: str) -> Generator[Dict[str, Any], None, None]:
    '''
    Single-pass pipeline for short audio (<30s).
    Same as exp-7 logic.
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
        
        queue = Queue()
        chat.out_queue = queue
        
        user_input = f'{{{{audio:{wav_path}}}}}'
        
        yield {'stage': 'transcribing', 'message': 'Transcribing...', 'text': '', 'language': ''}
        
        result = lib_asr.chat(chat._chat, user_input)
        if result != 0:
            chat.destroy()
            del chat
            del lib_asr
            yield {'stage': 'error', 'message': f'ASR failed with code: {result}'}
            return
        
        chunks = []
        first_token_time = None
        start_time = time.time()
        
        while True:
            try:
                item = queue.get(timeout=300.0)
                if isinstance(item, LLMChatChunk):
                    chunk = item.chunk
                    if chunk is None or chunk in ['<asr_text>', '</asr_text>', 'language']:
                        continue
                    if first_token_time is None:
                        first_token_time = time.time()
                    chunks.append(chunk)
                    raw_output = ''.join(chunks)
                    detected_lang, _ = extract_language(raw_output)
                    yield {
                        'stage': 'transcribing',
                        'message': 'Transcribing...',
                        'text': raw_output,
                        'language': detected_lang
                    }
                elif isinstance(item, LLMChatDone):
                    break
            except Empty:
                break
        
        raw_output = ''.join(chunks)
        end_time = time.time()
        
        detected_lang, transcript = extract_language(raw_output)
        
        if first_token_time:
            metrics['asr_ttft_ms'] = (first_token_time - start_time) * 1000
            gen_time = end_time - first_token_time
            metrics['asr_speed_chars_per_sec'] = len(raw_output) / gen_time if gen_time > 0 else 0
        
        chat.destroy()
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
        
        queue = Queue()
        chat.out_queue = queue
        
        user_input = f'{{{{audio:{wav_path}}}}} {tokenized_text}'
        
        yield {'stage': 'aligning', 'message': 'Aligning words to audio...'}
        
        result = lib_aligner.chat(chat._chat, user_input)
        if result != 0:
            chat.destroy()
            del chat
            del lib_aligner
            yield {'stage': 'error', 'message': f'Alignment failed with code: {result}'}
            return
        
        chunks = []
        first_token_time = None
        start_time = time.time()
        
        while True:
            try:
                item = queue.get(timeout=300.0)
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
        
        if first_token_time:
            metrics['aligner_ttft_ms'] = (first_token_time - start_time) * 1000
            gen_time = end_time - first_token_time
            metrics['aligner_speed_chars_per_sec'] = len(output) / gen_time if gen_time > 0 else 0
        
        chat.destroy()
        del chat
        del lib_aligner
        
        try:
            alignment = json.loads(output)
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


def run_pipeline_simple(audio_path: str) -> Dict[str, Any]:
    '''
    Non-streaming version that returns final results.
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
    
    for update in run_pipeline_streaming(audio_path):
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
    
    parser = argparse.ArgumentParser(description='Chunked ASR + Forced Aligner Pipeline')
    parser.add_argument('audio', help='Path to audio file')
    parser.add_argument('--output', '-o', help='Output SRT file path')
    args = parser.parse_args()
    
    for update in run_pipeline_streaming(args.audio):
        stage = update.get('stage', '')
        message = update.get('message', '')
        
        if stage == 'chunk_transcribing':
            partial = update.get('partial_transcript', '')
            print(f"\r[Chunk {update.get('chunk_index', 0) + 1}] {partial[-50:]}", end='', flush=True)
        elif stage == 'done':
            print(f"\n\n{message}")
            print(f"Language: {update.get('language')}")
            print(f"Metrics: {update.get('metrics')}")
            
            if args.output:
                srt_path = Path(args.output)
                srt_path.write_text(update.get('srt', ''), encoding='utf-8')
                print(f"SRT saved to: {srt_path}")
        elif stage in ['chunks_ready', 'processing_chunk', 'chunk_complete']:
            print(f"\n[{stage}] {message}")
