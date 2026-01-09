#!/bin/bash
# Run container locally for testing

IMAGE_NAME="${1:-rewact_amd64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

docker run --gpus all --rm -it \
    -v "${REPO_DIR}:/workspace/repo" \
    -v "${REPO_DIR}/rewact_tools:/workspace/rewact_tools" \
    -v "${REPO_DIR}/lerobot_policy_rewact:/workspace/lerobot_policy_rewact" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -e "HF_HOME=/root/.cache/huggingface" \
    -e "PYTHONPATH=/workspace/repo:${PYTHONPATH}" \
    "$IMAGE_NAME"

