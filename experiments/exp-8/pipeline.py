#!/usr/bin/env python3
# coding: utf-8
'''
Experiment 8: Chunked ASR + Forced Aligner Pipeline

Extends exp-7 to handle long audio by:
1. Detecting speech segments with TEN VAD
2. Chunking audio into ~20s segments with speech-aware boundaries
3. Processing each chunk through the ASR + alignment pipeline
4. Accumulating results with proper timestamp offsets

Optimized: Models loaded once per phase (not per chunk).
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

PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT', str(Path(__file__).resolve().parent.parent.parent)))
ASR_MODEL_PATH = PROJECT_ROOT / 'models' / 'qwen3-asr-0.6b-q4_0.bin'
ALIGNER_MODEL_PATH = PROJECT_ROOT / 'models' / 'qwen3-forced-aligner-0.6b-q4_0.bin'

INFERENCE_THREADS = int(os.getenv('INFERENCE_THREADS', '6'))
CONTEXT_LENGTH = os.getenv('CONTEXT_LENGTH', '4096')

CHUNK_THRESHOLD_SEC = 30.0
TARGET_CHUNK_DURATION_SEC = 20.0
MAX_CHUNK_DURATION_SEC = 30.0

DEBUG = os.getenv('DEBUG', '1') == '1'

def _debug(msg: str):
    if DEBUG:
        print(f"[PIPELINE] {msg}", flush=True)

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


def run_batch_asr(
    chunk_data: List[Dict],
    bindings_path: str,
    pipeline_start: float
) -> Generator[Dict[str, Any], None, None]:
    '''
    Run ASR on all chunks with model loaded once.
    
    Args:
        chunk_data: List of chunk dicts with 'wav_path', 'start_ms', 'end_ms'
        bindings_path: Path to chatllm bindings
        pipeline_start: Start time for debug logging
    
    Yields:
        Per-chunk progress and results
    '''
    _debug(f"run_batch_asr: Loading ASR model ONCE for {len(chunk_data)} chunks")
    load_start = time.time()
    
    lib_asr = LibChatLLM(bindings_path)
    llm_params = get_asr_params()
    chat = ChatLLM(lib_asr, llm_params)
    
    load_time = time.time() - load_start
    _debug(f"run_batch_asr: Model loaded in {load_time:.1f}s")
    
    accumulated_transcript = ""
    accumulated_itn = ""
    detected_language = "Unknown"
    asr_ttft_samples = []
    asr_speed_samples = []
    
    for chunk_idx, chunk in enumerate(chunk_data):
        wav_path = chunk['wav_path']
        chunk_num = chunk_idx + 1
        total_chunks = len(chunk_data)
        
        _debug(f"run_batch_asr: Chunk {chunk_num}/{total_chunks} - restart()")
        chat.restart()
        
        queue = Queue()
        chat.out_queue = queue
        
        user_input = f'{{{{audio:{wav_path}}}}}'
        
        chunk_asr_start = time.time()
        first_token_time = None
        chunks = []
        
        result = lib_asr.chat(chat._chat, user_input)
        if result != 0:
            chat.destroy()
            yield {'stage': 'error', 'message': f'ASR failed with code: {result}'}
            return
        
        while True:
            try:
                item = queue.get(timeout=300.0)
                if isinstance(item, LLMChatChunk):
                    chunk_text = item.chunk
                    if chunk_text is None or chunk_text in ['<asr_text>', '</asr_text>', 'language']:
                        continue
                    chunks.append(chunk_text)
                    raw_output = ''.join(chunks)
                    detected_lang, _ = extract_language(raw_output)
                    detected_language = detected_lang
                    
                    if first_token_time is None and chunk_text:
                        first_token_time = time.time()
                    
                    yield {
                        'stage': 'chunk_transcribing',
                        'message': f'Chunk {chunk_num}/{total_chunks}: Transcribing...',
                        'partial_transcript': raw_output,
                        'accumulated_transcript': accumulated_transcript,
                        'language': detected_language,
                        'chunk_index': chunk_idx,
                        'total_chunks': total_chunks
                    }
                elif isinstance(item, LLMChatDone):
                    break
            except Empty:
                break
        
        chunk_asr_end = time.time()
        raw_output = ''.join(chunks)
        _, chunk_transcript = extract_language(raw_output)
        
        if first_token_time:
            ttft_ms = (first_token_time - chunk_asr_start) * 1000
            gen_time = chunk_asr_end - first_token_time
            speed = len(chunk_transcript) / gen_time if gen_time > 0 else 0
            asr_ttft_samples.append(ttft_ms)
            asr_speed_samples.append(speed)
        
        _debug(f"run_batch_asr: Chunk {chunk_num} done, {len(chunk_transcript)} chars in {chunk_asr_end - chunk_asr_start:.1f}s")
        
        accumulated_transcript += chunk_transcript
        
        yield {
            'stage': 'chunk_itn',
            'message': f'Chunk {chunk_num}/{total_chunks}: Applying ITN...',
            'accumulated_transcript': accumulated_transcript,
            'chunk_index': chunk_idx,
            'total_chunks': total_chunks
        }
        
        chunk_itn = apply_itn(chunk_transcript)
        accumulated_itn += chunk_itn
        
        yield {
            'stage': 'chunk_itn_done',
            'message': f'Chunk {chunk_num}/{total_chunks}: ITN complete',
            'chunk_transcript': chunk_transcript,
            'chunk_itn': chunk_itn,
            'accumulated_transcript': accumulated_transcript,
            'accumulated_itn': accumulated_itn,
            'language': detected_language,
            'chunk_index': chunk_idx,
            'total_chunks': total_chunks
        }
    
    _debug(f"run_batch_asr: All chunks done, destroying model")
    chat.destroy()
    
    yield {
        'stage': 'batch_asr_complete',
        'accumulated_transcript': accumulated_transcript,
        'accumulated_itn': accumulated_itn,
        'language': detected_language,
        'asr_ttft_samples': asr_ttft_samples,
        'asr_speed_samples': asr_speed_samples,
        'model_load_time': load_time
    }


def run_batch_alignment(
    chunk_data: List[Dict],
    bindings_path: str,
    pipeline_start: float
) -> Generator[Dict[str, Any], None, None]:
    '''
    Run alignment on all chunks with model loaded once.
    
    Args:
        chunk_data: List of chunk dicts with 'wav_path', 'start_ms', 'end_ms', 'itn_transcript'
        bindings_path: Path to chatllm bindings
        pipeline_start: Start time for debug logging
    
    Yields:
        Per-chunk progress and accumulated SRT
    '''
    _debug(f"run_batch_alignment: Loading Aligner model ONCE for {len(chunk_data)} chunks")
    load_start = time.time()
    
    lib_aligner = LibChatLLM(bindings_path)
    llm_params = get_aligner_params()
    chat = ChatLLM(lib_aligner, llm_params)
    
    load_time = time.time() - load_start
    _debug(f"run_batch_alignment: Model loaded in {load_time:.1f}s")
    
    accumulated_alignment = []
    
    for chunk_idx, chunk in enumerate(chunk_data):
        wav_path = chunk['wav_path']
        itn_transcript = chunk['itn_transcript']
        start_ms = chunk['start_ms']
        chunk_num = chunk_idx + 1
        total_chunks = len(chunk_data)
        
        tokens = tokenize_with_jieba(itn_transcript)
        
        if not tokens:
            _debug(f"run_batch_alignment: Chunk {chunk_num} - no tokens, skipping")
            yield {
                'stage': 'chunk_aligned',
                'chunk_index': chunk_idx,
                'total_chunks': total_chunks,
                'alignment_count': len(accumulated_alignment),
                'srt': generate_srt_content(accumulated_alignment)
            }
            continue
        
        _debug(f"run_batch_alignment: Chunk {chunk_num}/{total_chunks} - {len(tokens)} tokens, restart()")
        chat.restart()
        
        queue = Queue()
        chat.out_queue = queue
        
        delimiter = '|'
        tokenized_text = delimiter.join(tokens)
        user_input = f'{{{{audio:{wav_path}}}}} {tokenized_text}'
        
        yield {
            'stage': 'chunk_aligning',
            'message': f'Chunk {chunk_num}/{total_chunks}: Aligning...',
            'chunk_index': chunk_idx,
            'total_chunks': total_chunks
        }
        
        align_start = time.time()
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
                    chunk_text = item.chunk
                    if chunk_text is None:
                        continue
                    chunks.append(chunk_text)
                elif isinstance(item, LLMChatDone):
                    break
            except Empty:
                break
        
        output = ''.join(chunks)
        
        try:
            chunk_alignment = json.loads(output)
        except json.JSONDecodeError:
            chunk_alignment = []
        
        for entry in chunk_alignment:
            entry['start'] = entry.get('start', 0) + start_ms
            entry['end'] = entry.get('end', 0) + start_ms
        
        accumulated_alignment.extend(chunk_alignment)
        
        _debug(f"run_batch_alignment: Chunk {chunk_num} done, {len(chunk_alignment)} words in {time.time()-align_start:.1f}s")
        
        yield {
            'stage': 'chunk_complete',
            'message': f'Chunk {chunk_num}/{total_chunks}: Complete',
            'alignment_count': len(accumulated_alignment),
            'chunk_index': chunk_idx,
            'total_chunks': total_chunks,
            'srt': generate_srt_content(accumulated_alignment)
        }
    
    _debug(f"run_batch_alignment: All chunks done, destroying model")
    chat.destroy()
    
    yield {
        'stage': 'batch_alignment_complete',
        'alignment': accumulated_alignment,
        'model_load_time': load_time
    }


def run_chunked_pipeline_streaming(audio_path: str) -> Generator[Dict[str, Any], None, None]:
    '''
    Chunked pipeline for long audio (optimized: models loaded once per phase).
    
    Phase 1: Load ASR → Process all chunks → Unload ASR
    Phase 2: Load Aligner → Process all chunks → Unload Aligner
    
    Chunk WAV files are cached between phases.
    '''
    
    metrics = {
        'audio_duration_sec': 0,
        'total_chunks': 0,
        'total_asr_chars': 0,
        'total_alignment_words': 0,
        'avg_asr_ttft_ms': 0,
        'avg_asr_speed_chars_per_sec': 0,
        'asr_phase_time_sec': 0,
        'alignment_phase_time_sec': 0,
        'model_load_time_asr_sec': 0,
        'model_load_time_aligner_sec': 0,
        'total_time_sec': 0,
    }
    
    pipeline_start = time.time()
    _debug(f"=== PIPELINE START: {audio_path} ===")
    
    _debug(f"+0.0s: Yielding 'preparing'")
    yield {'stage': 'preparing', 'message': 'Analyzing audio and detecting speech segments...'}
    
    _debug(f"+{time.time()-pipeline_start:.1f}s: Starting VAD/chunking")
    try:
        chunker = AudioChunker(
            audio_path,
            target_chunk_duration_s=TARGET_CHUNK_DURATION_SEC,
            max_chunk_duration_s=MAX_CHUNK_DURATION_SEC
        )
        chunks = chunker.prepare()
    except Exception as e:
        _debug(f"+{time.time()-pipeline_start:.1f}s: EXCEPTION in chunking: {e}")
        yield {'stage': 'error', 'message': f'Failed to prepare chunks: {e}'}
        return
    
    _debug(f"+{time.time()-pipeline_start:.1f}s: VAD/chunking complete - {len(chunks)} chunks")
    
    metrics['audio_duration_sec'] = chunker.total_duration_s
    metrics['total_chunks'] = len(chunks)
    
    if not chunks:
        _debug(f"+{time.time()-pipeline_start:.1f}s: No speech detected")
        yield {'stage': 'error', 'message': 'No speech detected in audio'}
        chunker.cleanup()
        return
    
    _debug(f"+{time.time()-pipeline_start:.1f}s: Yielding 'chunks_ready'")
    yield {
        'stage': 'chunks_ready',
        'message': f'Found {len(chunks)} speech chunks ({chunker.total_duration_s:.1f}s total)',
        'total_chunks': len(chunks),
        'audio_duration': chunker.total_duration_s
    }
    
    chunk_data = []
    for chunk_idx, (start_ms, end_ms) in enumerate(chunks):
        chunk_info = chunker.get_chunk_info(chunk_idx)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            chunk_wav_path = tmp.name
        
        _debug(f"+{time.time()-pipeline_start:.1f}s: Extracting chunk {chunk_idx+1}/{len(chunks)} audio")
        chunker.get_chunk_audio(chunk_idx, chunk_wav_path)
        
        chunk_data.append({
            'index': chunk_idx,
            'start_ms': start_ms,
            'end_ms': end_ms,
            'wav_path': chunk_wav_path,
            'transcript': '',
            'itn_transcript': '',
            'alignment': []
        })
    
    chunker.cleanup()
    _debug(f"+{time.time()-pipeline_start:.1f}s: All chunk WAVs extracted")
    
    bindings_path = str(PROJECT_ROOT / 'chatllm.cpp' / 'bindings')
    
    _debug(f"+{time.time()-pipeline_start:.1f}s: === PHASE 1: ASR ===")
    _debug(f"+{time.time()-pipeline_start:.1f}s: Yielding 'loading_asr'")
    yield {'stage': 'loading_asr', 'message': 'Loading ASR model...'}
    
    asr_phase_start = time.time()
    asr_ttft_samples = []
    asr_speed_samples = []
    accumulated_transcript = ""
    accumulated_itn = ""
    detected_language = "Unknown"
    
    for update in run_batch_asr(chunk_data, bindings_path, pipeline_start):
        if update['stage'] == 'chunk_transcribing':
            yield {
                'stage': 'chunk_transcribing',
                'message': update['message'],
                'partial_transcript': update['partial_transcript'],
                'accumulated_transcript': update['accumulated_transcript'],
                'language': update['language'],
                'chunk_index': update['chunk_index'],
                'total_chunks': update['total_chunks']
            }
        
        elif update['stage'] == 'chunk_itn':
            yield update
        
        elif update['stage'] == 'chunk_itn_done':
            chunk_idx = update['chunk_index']
            chunk_data[chunk_idx]['transcript'] = update['chunk_transcript']
            chunk_data[chunk_idx]['itn_transcript'] = update['chunk_itn']
            accumulated_transcript = update['accumulated_transcript']
            accumulated_itn = update['accumulated_itn']
            detected_language = update['language']
            
            yield update
        
        elif update['stage'] == 'batch_asr_complete':
            accumulated_transcript = update['accumulated_transcript']
            accumulated_itn = update['accumulated_itn']
            detected_language = update['language']
            asr_ttft_samples = update['asr_ttft_samples']
            asr_speed_samples = update['asr_speed_samples']
            metrics['model_load_time_asr_sec'] = update['model_load_time']
        
        elif update['stage'] == 'error':
            _cleanup_chunk_wavs(chunk_data)
            yield update
            return
    
    asr_phase_end = time.time()
    metrics['asr_phase_time_sec'] = asr_phase_end - asr_phase_start
    _debug(f"+{time.time()-pipeline_start:.1f}s: === PHASE 1 COMPLETE ({metrics['asr_phase_time_sec']:.1f}s) ===")
    
    zh_tw_transcript = convert_to_zh_tw(accumulated_itn)
    
    _debug(f"+{time.time()-pipeline_start:.1f}s: === PHASE 2: ALIGNMENT ===")
    _debug(f"+{time.time()-pipeline_start:.1f}s: Yielding 'loading_aligner'")
    yield {'stage': 'loading_aligner', 'message': 'Loading Forced Aligner model...'}
    
    align_phase_start = time.time()
    accumulated_alignment = []
    
    for update in run_batch_alignment(chunk_data, bindings_path, pipeline_start):
        if update['stage'] == 'chunk_aligning':
            yield update
        
        elif update['stage'] == 'chunk_aligned':
            yield {
                'stage': 'chunk_complete',
                'message': f"Chunk {update['chunk_index']+1}/{update['total_chunks']}: Complete (no tokens)",
                'alignment_count': update['alignment_count'],
                'chunk_index': update['chunk_index'],
                'total_chunks': update['total_chunks'],
                'srt': update['srt']
            }
        
        elif update['stage'] == 'chunk_complete':
            accumulated_alignment = generate_srt_content([]) 
            yield update
        
        elif update['stage'] == 'batch_alignment_complete':
            accumulated_alignment = update['alignment']
            metrics['model_load_time_aligner_sec'] = update['model_load_time']
        
        elif update['stage'] == 'error':
            _cleanup_chunk_wavs(chunk_data)
            yield update
            return
    
    align_phase_end = time.time()
    metrics['alignment_phase_time_sec'] = align_phase_end - align_phase_start
    _debug(f"+{time.time()-pipeline_start:.1f}s: === PHASE 2 COMPLETE ({metrics['alignment_phase_time_sec']:.1f}s) ===")
    
    _cleanup_chunk_wavs(chunk_data)
    _debug(f"+{time.time()-pipeline_start:.1f}s: Cleaned up chunk WAVs")
    
    metrics['total_asr_chars'] = len(accumulated_transcript)
    metrics['total_alignment_words'] = len(accumulated_alignment)
    metrics['avg_asr_ttft_ms'] = sum(asr_ttft_samples) / len(asr_ttft_samples) if asr_ttft_samples else 0
    metrics['avg_asr_speed_chars_per_sec'] = sum(asr_speed_samples) / len(asr_speed_samples) if asr_speed_samples else 0
    metrics['total_time_sec'] = time.time() - pipeline_start
    
    final_srt = generate_srt_content(accumulated_alignment)
    
    _debug(f"+{time.time()-pipeline_start:.1f}s: === PIPELINE COMPLETE ===")
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


def _cleanup_chunk_wavs(chunk_data: List[Dict]):
    '''Clean up temporary chunk WAV files.'''
    for chunk in chunk_data:
        wav_path = chunk.get('wav_path')
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except:
                pass


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
