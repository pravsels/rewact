# NVIDIA NGC PyTorch image (works on both ARM64 and x86_64 with CUDA)
FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install additional system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg libgl1 \
        build-essential cmake git pkg-config \
        libavcodec-dev libavformat-dev libswscale-dev libavfilter-dev libavdevice-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install lerobot and common deps (numpy<2 first to avoid pyarrow incompatibility)
RUN python -m pip install "numpy<2"

# Build/Install decord from source for arm64, or from pip for amd64
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        git clone --recursive https://github.com/dmlc/decord.git && \
        cd decord && mkdir build && cd build && \
        cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release && \
        make -j$(nproc) && \
        cd ../python && python3 setup.py install && \
        cd ../.. && rm -rf decord; \
    else \
        python -m pip install decord; \
    fi

# Copy packages first (before installing dependencies to use Docker layer caching)
COPY sam3 /workspace/sam3
COPY vision /workspace/vision
COPY rewact_tools /workspace/rewact_tools
COPY lerobot_policy_rewact /workspace/lerobot_policy_rewact

# Install all packages in editable mode (lerobot>=0.4 installed as dependency)
# Also install common dependencies in one go
RUN python -m pip install \
    safetensors>=0.7.0 \
    wandb \
    pycocotools \
    opencv-python==4.10.0.84 \
    av \
    draccus \
    tqdm \
    && python -m pip install -v -e /workspace/sam3 \
    && python -m pip install -v -e /workspace/vision \
    && python -m pip install -v -e /workspace/rewact_tools \
    && python -m pip install -v -e /workspace/lerobot_policy_rewact

WORKDIR /workspace

CMD ["/bin/bash"]
