#!/bin/bash
# Run container locally for testing

IMAGE_NAME="${1:-rewact_amd64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

docker run --gpus all --rm -it \
    -v "${REPO_DIR}:/workspace/repo" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -e "HF_HOME=/root/.cache/huggingface" \
    -e "PYTHONPATH=/workspace/repo:${PYTHONPATH}" \
    "$IMAGE_NAME"

