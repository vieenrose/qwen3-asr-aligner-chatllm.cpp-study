FROM python:3.11-slim

WORKDIR /app

# Install build dependencies and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Clone from fork with memory leak fix
ARG CHATLLM_REPO=https://github.com/vieenrose/chatllm.cpp.git
ARG CHATLLM_BRANCH=feature/exp1-qwen3-asr

RUN git clone --depth 1 -b ${CHATLLM_BRANCH} --recursive ${CHATLLM_REPO} /app/chatllm.cpp

# Build chatllm.cpp
ARG BUILD_THREADS=22
RUN rm -rf /app/chatllm.cpp/build && \
    mkdir -p /app/chatllm.cpp/build && \
    cd /app/chatllm.cpp/build && \
    cmake .. && \
    make -j${BUILD_THREADS} libchatllm

# Copy built libraries
RUN mkdir -p /app/lib && \
    cp -P /app/chatllm.cpp/build/lib/*.so* /app/lib/ 2>/dev/null || true

ENV LD_LIBRARY_PATH=/app/lib

# Copy model files
COPY models/ /app/models/

# Copy Chinese-ITN submodule
COPY Chinese-ITN/ /app/Chinese-ITN/

# Copy experiment files
COPY experiments/exp-6/ /app/experiments/exp-6/

# Copy sample audio
COPY samples/ /app/samples/

# Create bindings dir and copy the built library
RUN mkdir -p /app/bindings && \
    cp /app/chatllm.cpp/bindings/libchatllm.so /app/bindings/ && \
    cp /app/chatllm.cpp/bindings/chatllm.py /app/bindings/

RUN pip install --no-cache-dir -r /app/experiments/exp-6/requirements.txt

WORKDIR /app/experiments/exp-6

ENV PYTHONPATH=/app/bindings:/app/Chinese-ITN:/app/chatllm.cpp/scripts

CMD ["python", "run_pipeline.py"]
