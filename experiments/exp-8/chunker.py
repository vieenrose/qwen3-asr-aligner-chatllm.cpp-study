#!/usr/bin/env python3
# coding: utf-8
'''
Speech-Aware Audio Chunker

Uses VAD to segment audio into ~20 second chunks with speech-aware boundaries.
'''

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

from vad import detect_speech_segments


DEFAULT_TARGET_CHUNK_DURATION_MS = 20_000
DEFAULT_MAX_CHUNK_DURATION_MS = 30_000
DEFAULT_MIN_CHUNK_DURATION_MS = 5_000


def convert_to_wav(input_path: str, output_path: str) -> float:
    '''Convert audio to 16kHz mono WAV.'''
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


def get_audio_duration(audio_path: str) -> float:
    '''Get audio duration in seconds.'''
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
        capture_output=True, text=True, check=True
    )
    return float(result.stdout.strip())


def extract_chunk_audio(
    input_wav: str,
    output_path: str,
    start_ms: int,
    end_ms: int
) -> str:
    '''Extract a chunk of audio from WAV file.'''
    start_sec = start_ms / 1000.0
    duration_sec = (end_ms - start_ms) / 1000.0
    
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_sec),
        '-i', input_wav,
        '-t', str(duration_sec),
        '-acodec', 'pcm_s16le',
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def create_chunks_from_segments(
    speech_segments: List[Tuple[int, int]],
    total_duration_ms: int,
    target_duration_ms: int = DEFAULT_TARGET_CHUNK_DURATION_MS,
    max_duration_ms: int = DEFAULT_MAX_CHUNK_DURATION_MS,
    min_duration_ms: int = DEFAULT_MIN_CHUNK_DURATION_MS
) -> List[Tuple[int, int]]:
    '''
    Group speech segments into chunks of ~target_duration.
    
    Strategy:
    - Accumulate speech segments until reaching target_duration
    - Split at silence gaps when possible (soft limit)
    - Enforce hard max_duration limit
    - Merge short remaining chunks
    '''
    if not speech_segments:
        return []
    
    chunks = []
    current_chunk_start = None
    current_chunk_end = None
    
    for seg_start, seg_end in speech_segments:
        if current_chunk_start is None:
            current_chunk_start = seg_start
            current_chunk_end = seg_end
            continue
        
        gap = seg_start - current_chunk_end
        potential_duration = seg_end - current_chunk_start
        
        if potential_duration <= target_duration_ms:
            current_chunk_end = seg_end
        elif potential_duration <= max_duration_ms and gap < 500:
            current_chunk_end = seg_end
        else:
            if current_chunk_end - current_chunk_start >= min_duration_ms:
                chunks.append((current_chunk_start, current_chunk_end))
            current_chunk_start = seg_start
            current_chunk_end = seg_end
    
    if current_chunk_start is not None:
        if chunks and (current_chunk_end - current_chunk_start) < min_duration_ms:
            last_start, last_end = chunks[-1]
            chunks[-1] = (last_start, current_chunk_end)
        else:
            chunks.append((current_chunk_start, current_chunk_end))
    
    return chunks


def chunk_audio(
    audio_path: str,
    target_chunk_duration_s: float = 20.0,
    max_chunk_duration_s: float = 30.0,
    min_chunk_duration_s: float = 5.0,
    vad_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int]], str]:
    '''
    Main chunking function.
    
    Args:
        audio_path: Path to input audio file (any format)
        target_chunk_duration_s: Target chunk duration in seconds
        max_chunk_duration_s: Maximum chunk duration
        min_chunk_duration_s: Minimum chunk duration
        vad_threshold: VAD speech probability threshold
        
    Returns:
        (chunks, wav_path): List of (start_ms, end_ms) tuples and path to converted WAV
    '''
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        wav_path = tmp.name
    
    duration_s = convert_to_wav(audio_path, wav_path)
    duration_ms = int(duration_s * 1000)
    
    speech_segments = detect_speech_segments(
        wav_path,
        threshold=vad_threshold
    )
    
    chunks = create_chunks_from_segments(
        speech_segments,
        total_duration_ms=duration_ms,
        target_duration_ms=int(target_chunk_duration_s * 1000),
        max_duration_ms=int(max_chunk_duration_s * 1000),
        min_duration_ms=int(min_chunk_duration_s * 1000)
    )
    
    return chunks, wav_path


class AudioChunker:
    '''
    Audio chunker that manages chunk extraction.
    '''
    
    def __init__(
        self,
        audio_path: str,
        target_chunk_duration_s: float = 20.0,
        max_chunk_duration_s: float = 30.0,
        vad_threshold: float = 0.5
    ):
        self.audio_path = audio_path
        self.target_chunk_duration_s = target_chunk_duration_s
        self.max_chunk_duration_s = max_chunk_duration_s
        self.vad_threshold = vad_threshold
        
        self.chunks = []
        self.wav_path = None
        self.total_duration_s = 0
        
    def prepare(self):
        '''Prepare chunks and convert audio to WAV.'''
        self.chunks, self.wav_path = chunk_audio(
            self.audio_path,
            target_chunk_duration_s=self.target_chunk_duration_s,
            max_chunk_duration_s=self.max_chunk_duration_s,
            vad_threshold=self.vad_threshold
        )
        self.total_duration_s = get_audio_duration(self.wav_path)
        return self.chunks
    
    def get_chunk_audio(self, chunk_index: int, output_path: str = None) -> str:
        '''Extract audio for a specific chunk.'''
        if chunk_index < 0 or chunk_index >= len(self.chunks):
            raise IndexError(f"Chunk index {chunk_index} out of range")
        
        start_ms, end_ms = self.chunks[chunk_index]
        
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.wav')
        
        return extract_chunk_audio(self.wav_path, output_path, start_ms, end_ms)
    
    def get_chunk_info(self, chunk_index: int) -> dict:
        '''Get information about a specific chunk.'''
        if chunk_index < 0 or chunk_index >= len(self.chunks):
            raise IndexError(f"Chunk index {chunk_index} out of range")
        
        start_ms, end_ms = self.chunks[chunk_index]
        return {
            'index': chunk_index,
            'total_chunks': len(self.chunks),
            'start_ms': start_ms,
            'end_ms': end_ms,
            'start_s': start_ms / 1000,
            'end_s': end_ms / 1000,
            'duration_s': (end_ms - start_ms) / 1000
        }
    
    def cleanup(self):
        '''Clean up temporary files.'''
        if self.wav_path and os.path.exists(self.wav_path):
            try:
                os.unlink(self.wav_path)
            except:
                pass
            self.wav_path = None


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python chunker.py <audio_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    print(f"Chunking: {audio_path}")
    
    chunker = AudioChunker(audio_path)
    chunks = chunker.prepare()
    
    print(f"\nTotal duration: {chunker.total_duration_s:.2f}s")
    print(f"Found {len(chunks)} chunks:\n")
    
    for i, (start_ms, end_ms) in enumerate(chunks, 1):
        info = chunker.get_chunk_info(i - 1)
        print(f"  Chunk {i}/{len(chunks)}: {info['start_s']:.1f}s - {info['end_s']:.1f}s ({info['duration_s']:.1f}s)")
    
    chunker.cleanup()
