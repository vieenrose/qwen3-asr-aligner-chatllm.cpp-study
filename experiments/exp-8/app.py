#!/usr/bin/env python3
# coding: utf-8
'''
Experiment 8: HuggingFace Spaces WebUI for Qwen3-ASR with VAD Chunking

Gradio WebUI with:
- Audio upload + sample audio buttons
- VAD-based chunking for long audio
- Chunk progress display
- Live streaming transcript (character-by-character)
- Detected language display
- zh-TW conversion result (shown at end)
- SRT preview + download (progressive)
- Performance metrics
'''

import os
import sys
import gradio as gr
from pathlib import Path

PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT', str(Path(__file__).resolve().parent.parent.parent)))
SAMPLES_DIR = PROJECT_ROOT / 'samples'

SAMPLE_AUDIOS = {}
SAMPLE_HINTS = {}

if (SAMPLES_DIR / 'phoneNumber1-zh-TW.wav').exists():
    SAMPLE_AUDIOS["Phone Number (zh-TW, 30s)"] = str(SAMPLES_DIR / 'phoneNumber1-zh-TW.wav')
    SAMPLE_HINTS["Phone Number (zh-TW, 30s)"] = "台灣國語電話號碼唸讀"
if (SAMPLES_DIR / 'news-zh.mp3').exists():
    SAMPLE_AUDIOS["News (Chinese, 2.7 min)"] = str(SAMPLES_DIR / 'news-zh.mp3')
    SAMPLE_HINTS["News (Chinese, 2.7 min)"] = "新聞播報，包含經濟、政治或社會議題"
if (SAMPLES_DIR / 'gettysburg_address_lincoln_64kb.mp3').exists():
    SAMPLE_AUDIOS["Gettysburg Address (English, 2.7 min)"] = str(SAMPLES_DIR / 'gettysburg_address_lincoln_64kb.mp3')
    SAMPLE_HINTS["Gettysburg Address (English, 2.7 min)"] = "English speech, American history"
if (SAMPLES_DIR / 'SteveJobsSpeech_64kb.mp3').exists():
    SAMPLE_AUDIOS["Steve Jobs Speech (English, 17 min)"] = str(SAMPLES_DIR / 'SteveJobsSpeech_64kb.mp3')
    SAMPLE_HINTS["Steve Jobs Speech (English, 17 min)"] = "English speech, technology keynote"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import run_pipeline_streaming


def format_time(seconds: float) -> str:
    '''Format seconds as MM:SS or HH:MM:SS'''
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}:{m:02d}:{s:02d}"
    else:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}:{s:02d}"


def transcribe_audio(audio_path: str, hint_text: str):
    '''
    Generator function that yields updates for Gradio streaming.
    Returns tuple of updates for all output components.
    '''
    if audio_path is None:
        yield (
            "Please upload an audio file or select a sample.",
            "", "", "", "", "", None, "", ""
        )
        return
    
    transcript = ""
    language = "Detecting..."
    itn_text = ""
    zh_tw_text = ""
    srt_content = ""
    metrics_text = ""
    srt_file = None
    
    chunk_progress = ""
    total_chunks = 0
    
    yield (
        "Starting pipeline...",
        transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text
    )
    
    try:
        for update in run_pipeline_streaming(audio_path, hint_text):
            stage = update.get('stage', '')
            message = update.get('message', '')
            
            if stage == 'preparing':
                status = "Analyzing audio..."
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text)
            
            elif stage == 'chunks_ready':
                total_chunks = update.get('total_chunks', 0)
                audio_dur = update.get('audio_duration', 0)
                status = f"Found {total_chunks} chunks ({format_time(audio_dur)} total)"
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text)
            
            elif stage == 'loading_asr':
                status = "Loading ASR model..."
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text)
            
            elif stage == 'loading_aligner':
                status = "Loading Forced Aligner..."
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text)
            
            elif stage == 'processing_chunk':
                chunk_idx = update.get('chunk_index', 0) + 1
                total = update.get('total_chunks', 1)
                start_s = update.get('chunk_start_s', 0)
                end_s = update.get('chunk_end_s', 0)
                chunk_progress = f"Chunk {chunk_idx}/{total}"
                status = f"Processing {chunk_progress} ({format_time(start_s)} - {format_time(end_s)})"
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text)
            
            elif stage == 'chunk_transcribing':
                partial = update.get('partial_transcript', '')
                accumulated = update.get('accumulated_transcript', '')
                language = update.get('language', language)
                chunk_idx = update.get('chunk_index', 0) + 1
                total = update.get('total_chunks', 1)
                status = f"Transcribing {chunk_progress}..."
                transcript = accumulated + partial  # Show previous chunks + current streaming
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text)
            
            elif stage == 'chunk_itn':
                accumulated = update.get('accumulated_transcript', transcript)
                status = f"ITN {chunk_progress}..."
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text)
            
            elif stage == 'chunk_itn_done':
                itn_text = update.get('accumulated_itn', itn_text)
                status = f"ITN {chunk_progress} done"
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text)
            
            elif stage == 'chunk_aligning':
                status = f"Aligning {chunk_progress}..."
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text)
            
            elif stage == 'chunk_complete':
                transcript = update.get('accumulated_transcript', transcript)
                itn_text = update.get('accumulated_itn', itn_text)
                srt_content = update.get('srt', srt_content)
                align_count = update.get('alignment_count', 0)
                status = f"Chunk {update.get('chunk_index', 0) + 1}/{update.get('total_chunks', 1)} complete ({align_count} words)"
                
                if srt_content:
                    srt_path = Path('/tmp/output.srt')
                    srt_path.write_text(srt_content, encoding='utf-8')
                    srt_file = str(srt_path)
                
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text)
            
            elif stage == 'converting':
                status = f"Converting: {message}"
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text)
            
            elif stage == 'transcribing':
                transcript = update.get('text', '')
                language = update.get('language', 'Detecting...')
                status = "Transcribing..."
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text)
            
            elif stage == 'transcribed':
                transcript = update.get('text', '')
                language = update.get('language', 'Unknown')
                status = "Transcription complete"
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text)
            
            elif stage == 'itn':
                status = "Applying Inverse Text Normalization..."
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text)
            
            elif stage == 'itn_done':
                itn_text = update.get('text', '')
                status = "ITN complete"
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text)
            
            elif stage == 'tokenizing':
                status = "Tokenizing with Jieba..."
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text)
            
            elif stage == 'tokenized':
                status = message
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text)
            
            elif stage == 'aligning':
                status = "Aligning words to audio..."
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text)
            
            elif stage == 'aligned':
                count = update.get('alignment_count', 0)
                status = f"Aligned {count} words"
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, hint_text)
            
            elif stage == 'done':
                zh_tw_text = update.get('zh_tw_text', '')
                srt_content = update.get('srt', '')
                metrics = update.get('metrics', {})
                
                if metrics:
                    if 'total_chunks' in metrics:
                        metrics_text = (
                            f"Chunks: {metrics.get('total_chunks', 0)}\n"
                            f"Audio Duration: {format_time(metrics.get('audio_duration_sec', 0))}\n"
                            f"Total Chars: {metrics.get('total_asr_chars', 0)}\n"
                            f"Avg TTFT: {metrics.get('avg_asr_ttft_ms', 0):.0f}ms\n"
                            f"Avg Speed: {metrics.get('avg_asr_speed_chars_per_sec', 0):.0f} chars/sec\n"
                            f"Alignment Words: {metrics.get('total_alignment_words', 0)}\n"
                            f"Total Time: {format_time(metrics.get('total_time_sec', 0))}"
                        )
                    else:
                        metrics_text = (
                            f"Audio Duration: {format_time(metrics.get('audio_duration_sec', 0))}\n"
                            f"ASR TTFT: {metrics.get('asr_ttft_ms', 0):.0f}ms\n"
                            f"ASR Speed: {metrics.get('asr_speed_chars_per_sec', 0):.0f} chars/sec\n"
                            f"Aligner TTFT: {metrics.get('aligner_ttft_ms', 0):.0f}ms\n"
                            f"Total Time: {format_time(metrics.get('total_time_sec', 0))}"
                        )
                
                if srt_content:
                    srt_path = Path('/tmp/output.srt')
                    srt_path.write_text(srt_content, encoding='utf-8')
                    srt_file = str(srt_path)
                
                status = f"Done! {message}"
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, "")
            
            elif stage == 'error':
                status = f"Error: {message}"
                yield (status, transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, "")
                return
    
    except Exception as e:
        yield (
            f"Error: {str(e)}",
            transcript, language, itn_text, zh_tw_text, srt_content, srt_file, metrics_text, ""
        )


def load_sample(sample_name: str):
    '''Return the path to a sample audio file and its associated hint text.'''
    return SAMPLE_AUDIOS.get(sample_name), SAMPLE_HINTS.get(sample_name, "")


with gr.Blocks(
    title="Lightweight Long-Audio ASR Stack: Qwen3 ASR + Ten-VAD + Qwen3 Aligner on ChatLLM.cpp"
) as app:
    
    gr.Markdown(
        """
        # Lightweight Long-Audio ASR Stack: Qwen3 ASR + Ten-VAD + Qwen3 Aligner on ChatLLM.cpp
        
        Upload an audio file or select a sample below. For long audio (>30s), the system will:
        1. Detect speech segments using TEN VAD
        2. Chunk audio into ~20s segments with speech-aware boundaries
        3. Process each chunk: Transcribe → ITN → Align
        4. Generate SRT subtitles with proper timestamps
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
        with gr.Column(scale=2):
            hint_input = gr.Textbox(
                label="Hint Text (optional)",
                placeholder="e.g., 台灣國語電話號碼唸讀",
                value="",
                lines=1,
                info="Provide context to improve ASR accuracy"
            )
        
        with gr.Column(scale=1):
            pass
    
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
                lines=8,
                show_label=False
            )
            
            srt_download = gr.File(
                label="Download SRT",
                visible=True
            )
    
    load_sample_btn.click(
        fn=load_sample,
        inputs=[sample_dropdown],
        outputs=[audio_input, hint_input]
    )
    
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, hint_input],
        outputs=[
            status_output,
            transcript_output,
            language_output,
            itn_output,
            zh_tw_output,
            srt_preview,
            srt_download,
            metrics_output,
            hint_input
        ]
    )
    
    gr.Markdown(
        """
        ---
        **About:** Qwen3-ASR 0.6B with CPU inference via chatllm.cpp
        | VAD: TEN VAD via ONNX Runtime
        | [GitHub](https://github.com/QwenLM/Qwen3-Audio)
        """
    )


if __name__ == '__main__':
    app.launch(server_name='0.0.0.0', server_port=7860, allowed_paths=['/app/samples'])
