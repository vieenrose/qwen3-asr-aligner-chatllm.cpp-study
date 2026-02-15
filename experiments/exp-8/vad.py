#!/usr/bin/env python3
# coding: utf-8
'''
TEN VAD Wrapper for Voice Activity Detection

Uses TEN VAD via pip package (wraps native library).
Install: pip install ten-vad
'''

import os
import wave
from pathlib import Path
from typing import List, Tuple
import numpy as np

try:
    from ten_vad import TenVad as TenVadNative
    TEN_VAD_AVAILABLE = True
except ImportError:
    TEN_VAD_AVAILABLE = False


class TenVAD:
    '''
    TEN VAD wrapper.
    
    Usage:
        vad = TenVAD(hop_size=256, threshold=0.5)
        for frame in audio_frames:
            prob, is_speech = vad.process(frame)
    '''
    
    def __init__(self, hop_size: int = 256, threshold: float = 0.5):
        if not TEN_VAD_AVAILABLE:
            raise ImportError("ten-vad is required. Install with: pip install ten-vad")
        
        self.hop_size = hop_size
        self.threshold = threshold
        self.sample_rate = 16000
        
        self.vad = TenVadNative(hop_size, threshold)
    
    def reset(self):
        '''Reset internal state for new audio stream.'''
        self.vad = TenVadNative(self.hop_size, self.threshold)
    
    def process(self, audio_frame: np.ndarray) -> Tuple[float, bool]:
        '''
        Process a single audio frame.
        
        Args:
            audio_frame: numpy array of int16 samples, length = hop_size
            
        Returns:
            (probability, is_speech): speech probability [0-1] and boolean flag
        '''
        if len(audio_frame) != self.hop_size:
            raise ValueError(f"Expected frame size {self.hop_size}, got {len(audio_frame)}")
        
        audio_int16 = audio_frame.astype(np.int16)
        prob, flag = self.vad.process(audio_int16)
        
        return float(prob), bool(flag)
    
    def process_file(self, wav_path: str) -> List[Tuple[int, float, bool]]:
        '''
        Process an entire WAV file.
        
        Args:
            wav_path: Path to WAV file (16kHz, mono)
            
        Returns:
            List of (sample_offset, probability, is_speech) tuples
        '''
        self.reset()
        
        with wave.open(wav_path, 'rb') as wf:
            if wf.getframerate() != self.sample_rate:
                raise ValueError(f"Expected {self.sample_rate}Hz, got {wf.getframerate()}Hz")
            if wf.getnchannels() != 1:
                raise ValueError("Expected mono audio")
            
            results = []
            sample_offset = 0
            
            while True:
                frames = wf.readframes(self.hop_size)
                if len(frames) == 0:
                    break
                
                n_samples = len(frames) // 2
                if n_samples < self.hop_size:
                    frames += b'\x00' * (self.hop_size * 2 - len(frames))
                
                audio_frame = np.frombuffer(frames, dtype=np.int16)[:self.hop_size]
                prob, is_speech = self.process(audio_frame)
                
                results.append((sample_offset, prob, is_speech))
                sample_offset += self.hop_size
            
            return results


def detect_speech_segments(
    wav_path: str,
    hop_size: int = 256,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100
) -> List[Tuple[int, int]]:
    '''
    Detect speech segments in audio file.
    
    Args:
        wav_path: Path to WAV file (16kHz, mono)
        hop_size: Frame size in samples (256 = 16ms at 16kHz)
        threshold: Speech probability threshold
        min_speech_duration_ms: Minimum speech segment duration
        min_silence_duration_ms: Minimum silence gap to split segments
        
    Returns:
        List of (start_ms, end_ms) tuples for speech segments
    '''
    vad = TenVAD(hop_size=hop_size, threshold=threshold)
    results = vad.process_file(wav_path)
    
    hop_ms = int(hop_size * 1000 / 16000)
    min_speech_frames = min_speech_duration_ms // hop_ms
    min_silence_frames = min_silence_duration_ms // hop_ms
    
    segments = []
    in_speech = False
    speech_start = 0
    silence_count = 0
    sample_offset = 0
    
    for sample_offset, prob, is_speech in results:
        time_ms = int(sample_offset * 1000 / 16000)
        
        if is_speech:
            if not in_speech:
                in_speech = True
                speech_start = time_ms
            silence_count = 0
        else:
            if in_speech:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    speech_end = time_ms - (silence_count * hop_ms)
                    if (speech_end - speech_start) >= min_speech_duration_ms:
                        segments.append((speech_start, speech_end))
                    in_speech = False
                    silence_count = 0
    
    if in_speech:
        speech_end = int(sample_offset * 1000 / 16000) if sample_offset > 0 else 0
        if (speech_end - speech_start) >= min_speech_duration_ms:
            segments.append((speech_start, speech_end))
    
    return segments


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vad.py <audio.wav>")
        sys.exit(1)
    
    wav_path = sys.argv[1]
    print(f"Processing: {wav_path}")
    
    segments = detect_speech_segments(wav_path)
    
    print(f"\nFound {len(segments)} speech segments:")
    for i, (start_ms, end_ms) in enumerate(segments, 1):
        duration_s = (end_ms - start_ms) / 1000
        print(f"  {i}. {start_ms/1000:.2f}s - {end_ms/1000:.2f}s ({duration_s:.2f}s)")
