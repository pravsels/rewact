# NVIDIA NGC PyTorch image (25.01 uses Python 3.12)
FROM nvcr.io/nvidia/pytorch:25.01-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_BREAK_SYSTEM_PACKAGES=1

# Install additional system deps + uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg libgl1 build-essential cmake git pkg-config \
        libavcodec-dev libavformat-dev libswscale-dev libavfilter-dev libavdevice-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv

WORKDIR /workspace

# 1. Create constraints to protect base image packages (exclude torch*/numpy due to version conflicts).
RUN python3 <<'EOF' > /workspace/constraints.txt
import importlib.metadata
for dist in importlib.metadata.distributions():
    name = dist.metadata.get("Name")
    if name and any(x in name.lower() for x in ["nvidia", "tensorrt"]) and "torch" not in name.lower():
        print(f"{name}=={dist.version}")
EOF

# 2. Build/Install decord (source for arm64, pip for amd64)
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        git clone --recursive https://github.com/dmlc/decord.git && \
        cd decord && mkdir build && cd build && \
        cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release && \
        make -j$(nproc) && \
        cd ../python && pip install . && \
        cd ../.. && rm -rf decord; \
    else \
        uv pip install --system --prerelease=allow -c /workspace/constraints.txt decord; \
    fi

# 3. Copy and install local packages in editable mode (uv handles resolution).
COPY sam3 /workspace/sam3
COPY dinov3 /workspace/dinov3
COPY rewact_tools /workspace/rewact_tools
COPY lerobot_policy_rewact /workspace/lerobot_policy_rewact

RUN uv pip install --system --prerelease=allow -c /workspace/constraints.txt \
    -e /workspace/sam3 \
    -e /workspace/dinov3 \
    -e /workspace/rewact_tools \
    -e /workspace/lerobot_policy_rewact && \
    pip install --force-reinstall --no-cache-dir pycocotools

WORKDIR /workspace/repo
CMD ["/bin/bash"]
