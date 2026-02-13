FROM python:3.11-slim

WORKDIR /app

# Install build dependencies and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy chatllm.cpp source and submodules
COPY chatllm.cpp/ /app/chatllm.cpp/

# Build chatllm.cpp FIRST (build arg for thread count)
# This builds libchatllm.so and puts it in /app/chatllm.cpp/bindings/
ARG BUILD_THREADS=22
RUN rm -rf /app/chatllm.cpp/build && \
    mkdir -p /app/chatllm.cpp/build && \
    cd /app/chatllm.cpp/build && \
    cmake .. && \
    make -j${BUILD_THREADS} libchatllm

# Copy built libraries to /app/lib AFTER build
RUN mkdir -p /app/lib && \
    cp -P /app/chatllm.cpp/build/lib/*.so* /app/lib/ 2>/dev/null || true

ENV LD_LIBRARY_PATH=/app/lib

# Copy model file
COPY models/ /app/models/

# Copy Chinese-ITN submodule
COPY Chinese-ITN/ /app/Chinese-ITN/

# Copy experiment files
COPY experiments/qwen3-asr-chatllm.cpp/ /app/experiments/qwen3-asr-chatllm.cpp/

# Copy sample audio
COPY samples/ /app/samples/

# Create bindings dir and copy the BUILT libchatllm.so (from build output in bindings/)
RUN mkdir -p /app/bindings && \
    cp /app/chatllm.cpp/bindings/libchatllm.so /app/bindings/ && \
    cp /app/chatllm.cpp/bindings/chatllm.py /app/bindings/

RUN pip install --no-cache-dir -r /app/experiments/qwen3-asr-chatllm.cpp/requirements.txt

WORKDIR /app/experiments/qwen3-asr-chatllm.cpp

ENV PYTHONPATH=/app/bindings:/app/Chinese-ITN:/app/chatllm.cpp/scripts

CMD ["python", "run_asr.py"]
