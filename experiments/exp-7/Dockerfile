FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ARG CHATLLM_REPO=https://github.com/vieenrose/chatllm.cpp.git
ARG CHATLLM_BRANCH=feature/exp1-qwen3-asr

RUN git clone --depth 1 -b ${CHATLLM_BRANCH} --recursive ${CHATLLM_REPO} /app/chatllm.cpp

ARG BUILD_THREADS=22
RUN rm -rf /app/chatllm.cpp/build && \
    mkdir -p /app/chatllm.cpp/build && \
    cd /app/chatllm.cpp/build && \
    cmake .. && \
    make -j${BUILD_THREADS} libchatllm

RUN mkdir -p /app/lib && \
    cp -P /app/chatllm.cpp/build/lib/*.so* /app/lib/ 2>/dev/null || true

ENV LD_LIBRARY_PATH=/app/lib

RUN pip install --no-cache-dir huggingface_hub && \
    mkdir -p /app/models && \
    huggingface-cli download Luigi/Qwen3-ASR-0.6B-chatllm-quantized \
        qwen3-asr-0.6b-q4_0.bin \
        qwen3-forced-aligner-0.6b-q4_0.bin \
        --local-dir /app/models \
        --local-dir-use-symlinks False

COPY Chinese-ITN/ /app/Chinese-ITN/

COPY experiments/exp-7/ /app/experiments/exp-7/

RUN mkdir -p /app/samples
COPY samples/phoneNumber1-zh-TW.wav /app/samples/
COPY samples/news-zh.mp3 /app/samples/

RUN mkdir -p /app/bindings && \
    cp /app/chatllm.cpp/bindings/libchatllm.so /app/bindings/ && \
    cp /app/chatllm.cpp/bindings/chatllm.py /app/bindings/

RUN pip install --no-cache-dir -r /app/experiments/exp-7/requirements.txt

WORKDIR /app/experiments/exp-7

ENV PYTHONPATH=/app/bindings:/app/Chinese-ITN:/app/chatllm.cpp/scripts

EXPOSE 7860

CMD ["python", "app.py"]
