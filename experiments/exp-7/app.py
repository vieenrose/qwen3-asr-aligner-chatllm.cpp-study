#!/usr/bin/env python3
# coding: utf-8
'''
Experiment 7: HuggingFace Spaces WebUI for Qwen3-ASR

Gradio WebUI with:
- Audio upload + sample audio buttons
- Live streaming transcript (character-by-character)
- Detected language display
- zh-TW conversion result
- SRT preview + download
- Performance metrics
'''

import os
import sys
import gradio as gr
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = PROJECT_ROOT / 'samples'

SAMPLE_AUDIOS = {}
if (SAMPLES_DIR / 'phoneNumber1-zh-TW.wav').exists():
    SAMPLE_AUDIOS["Phone Number (zh-TW, 30s)"] = str(SAMPLES_DIR / 'phoneNumber1-zh-TW.wav')
if (SAMPLES_DIR / 'news-zh.mp3').exists():
    SAMPLE_AUDIOS["News (Chinese, 2.7 min)"] = str(SAMPLES_DIR / 'news-zh.mp3')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import run_pipeline_streaming


def transcribe_audio(audio_path: str):
    '''
    Generator function that yields updates for Gradio streaming.
    Returns tuple of updates for all output components.
    '''
    if audio_path is None:
        yield (
            "Please upload an audio file or select a sample.",
            "", "", "", "", "", None, ""
        )
        return
    
    transcript = ""
    language = "Detecting..."
    itn_text = ""
    zh_tw_text = ""
    srt_content = ""
    metrics_text = ""
    srt_file = None
    
    yield (
        "Starting pipeline...",
        transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text
    )
    
    try:
        for update in run_pipeline_streaming(audio_path):
            stage = update.get('stage', '')
            message = update.get('message', '')
            
            if stage == 'converting':
                status = f"Converting: {message}"
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text)
            
            elif stage == 'loading_asr':
                status = "Loading ASR model..."
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text)
            
            elif stage == 'transcribing':
                transcript = update.get('text', '')
                language = update.get('language', 'Detecting...')
                status = "Transcribing..."
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text)
            
            elif stage == 'transcribed':
                transcript = update.get('text', '')
                language = update.get('language', 'Unknown')
                status = "Transcription complete"
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text)
            
            elif stage == 'itn':
                status = "Applying Inverse Text Normalization..."
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text)
            
            elif stage == 'itn_done':
                itn_text = update.get('text', '')
                status = "ITN complete"
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text)
            
            elif stage == 'tokenizing':
                status = "Tokenizing with Jieba..."
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text)
            
            elif stage == 'loading_aligner':
                status = "Loading Forced Aligner..."
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text)
            
            elif stage == 'aligning':
                status = "Aligning words to audio..."
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text)
            
            elif stage == 'aligned':
                count = update.get('alignment_count', 0)
                status = f"Aligned {count} words"
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text)
            
            elif stage == 'done':
                zh_tw_text = update.get('zh_tw_text', '')
                print(f"[DEBUG app.py] received zh_tw_text: {zh_tw_text[:100] if zh_tw_text else 'EMPTY'}")
                srt_content = update.get('srt', '')
                metrics = update.get('metrics', {})
                
                if metrics:
                    metrics_text = (
                        f"Audio Duration: {metrics.get('audio_duration_sec', 0):.2f}s\n"
                        f"ASR TTFT: {metrics.get('asr_ttft_ms', 0):.0f}ms\n"
                        f"ASR Speed: {metrics.get('asr_speed_chars_per_sec', 0):.0f} chars/sec\n"
                        f"Aligner TTFT: {metrics.get('aligner_ttft_ms', 0):.0f}ms\n"
                        f"Total Time: {metrics.get('total_time_sec', 0):.2f}s"
                    )
                
                if srt_content:
                    srt_path = Path('/tmp/output.srt')
                    srt_path.write_text(srt_content, encoding='utf-8')
                    srt_file = str(srt_path)
                
                status = "Done!"
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text)
            
            elif stage == 'error':
                status = f"Error: {message}"
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text)
                return
    
    except Exception as e:
        yield (
            f"Error: {str(e)}",
            transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text
        )


def load_sample(sample_name: str):
    '''Return the path to a sample audio file.'''
    return SAMPLE_AUDIOS.get(sample_name)


with gr.Blocks(
    title="Qwen3-ASR - Speech Recognition"
) as app:
    
    gr.Markdown(
        """
        # Qwen3-ASR 0.6B - Speech Recognition with Forced Alignment
        
        Upload an audio file or select a sample below. The system will:
        1. Transcribe speech using Qwen3-ASR (streaming)
        2. Normalize numbers (Chinese ITN)
        3. Align words to timestamps
        4. Convert to Traditional Chinese (zh-TW)
        5. Generate SRT subtitles
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            audio_input = gr.Audio(
                label="Upload Audio",
                type="filepath",
                sources=["upload", "microphone"]
            )
        
        with gr.Column(scale=1):
            gr.Markdown("**Quick Demo Samples**")
            sample_dropdown = gr.Dropdown(
                choices=list(SAMPLE_AUDIOS.keys()),
                label="Select Sample",
                value=None
            )
            load_sample_btn = gr.Button("Load Sample", variant="secondary")
    
    with gr.Row():
        transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")
    
    with gr.Row():
        status_output = gr.Textbox(
            label="Status",
            value="Ready",
            interactive=False,
            elem_classes=["status-box"]
        )
        
        language_output = gr.Textbox(
            label="Detected Language",
            value="-",
            interactive=False,
            elem_classes=["language-box"]
        )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Live Transcript (Streaming)")
            transcript_output = gr.Textbox(
                label="Raw Transcript",
                value="",
                interactive=False,
                lines=8,
                show_label=False
            )
        
        with gr.Column():
            gr.Markdown("### ITN Result")
            itn_output = gr.Textbox(
                label="After ITN",
                value="",
                interactive=False,
                lines=8,
                show_label=False
            )
    
    with gr.Row():
        gr.Markdown("### zh-TW Result (Traditional Chinese)")
        zh_tw_output = gr.Textbox(
            label="zh-TW",
            value="",
            interactive=False,
            lines=4,
            show_label=False
        )
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### SRT Subtitles")
            srt_preview = gr.Textbox(
                label="SRT",
                value="",
                interactive=False,
                lines=10,
                show_label=False
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### Performance")
            metrics_output = gr.Textbox(
                label="Metrics",
                value="",
                interactive=False,
                lines=6,
                show_label=False
            )
            
            srt_download = gr.File(
                label="Download SRT",
                visible=True
            )
    
    load_sample_btn.click(
        fn=load_sample,
        inputs=[sample_dropdown],
        outputs=[audio_input]
    )
    
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input],
        outputs=[
            status_output,
            transcript_output,
            language_output,
            itn_output,
            zh_tw_output,
            srt_preview,
            srt_download,
            metrics_output
        ]
    )
    
    gr.Markdown(
        """
        ---
        **About:** Qwen3-ASR 0.6B with CPU inference via chatllm.cpp
        | [GitHub](https://github.com/QwenLM/Qwen3-Audio)
        """
    )


if __name__ == '__main__':
    app.launch(server_name='0.0.0.0', server_port=7860, allowed_paths=['/app/samples'])
