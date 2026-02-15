#!/usr/bin/env python3
# coding: utf-8
'''
TEN VAD ONNX Wrapper for Voice Activity Detection

Uses TEN VAD via ONNX Runtime for speech detection.
Model: TEN-framework/ten-vad on HuggingFace
'''

import os
import wave
import struct
from pathlib import Path
from typing import List, Tuple
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


DEFAULT_VAD_MODEL_PATH = Path('/app/models/ten-vad.onnx')
if not DEFAULT_VAD_MODEL_PATH.exists():
    DEFAULT_VAD_MODEL_PATH = Path(__file__).parent / 'models' / 'ten-vad.onnx'


class TenVAD:
    '''
    TEN VAD wrapper using ONNX Runtime.
    
    Usage:
        vad = TenVAD(hop_size=256, threshold=0.5)
        for frame in audio_frames:
            prob, is_speech = vad.process(frame)
    '''
    
    def __init__(self, hop_size: int = 256, threshold: float = 0.5, model_path: str = None):
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")
        
        self.hop_size = hop_size
        self.threshold = threshold
        self.sample_rate = 16000
        
        model_path = model_path or str(DEFAULT_VAD_MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"VAD model not found at {model_path}")
        
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        self.reset()
    
    def reset(self):
        '''Reset internal state for new audio stream.'''
        self.h1 = np.zeros((2, 1, 64), dtype=np.float32)
        self.h2 = np.zeros((2, 1, 128), dtype=np.float32)
    
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
        
        audio_float = audio_frame.astype(np.float32) / 32768.0
        audio_input = audio_float.reshape(1, -1)
        
        outputs = self.session.run(
            None,
            {
                self.input_name: audio_input,
                'h1': self.h1,
                'h2': self.h2
            }
        )
        
        prob = float(outputs[0][0][0])
        is_speech = prob > self.threshold
        
        self.h1 = outputs[1]
        self.h2 = outputs[2]
        
        return prob, is_speech
    
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
    model_path: str = None,
    hop_size: int = 256,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100
) -> List[Tuple[int, int]]:
    '''
    Detect speech segments in audio file.
    
    Args:
        wav_path: Path to WAV file (16kHz, mono)
        model_path: Path to VAD ONNX model
        hop_size: Frame size in samples (256 = 16ms at 16kHz)
        threshold: Speech probability threshold
        min_speech_duration_ms: Minimum speech segment duration
        min_silence_duration_ms: Minimum silence gap to split segments
        
    Returns:
        List of (start_ms, end_ms) tuples for speech segments
    '''
    vad = TenVAD(hop_size=hop_size, threshold=threshold, model_path=model_path)
    results = vad.process_file(wav_path)
    
    hop_ms = int(hop_size * 1000 / 16000)
    min_speech_frames = min_speech_duration_ms // hop_ms
    min_silence_frames = min_silence_duration_ms // hop_ms
    
    segments = []
    in_speech = False
    speech_start = 0
    silence_count = 0
    
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
        speech_end = int(sample_offset * 1000 / 16000)
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
