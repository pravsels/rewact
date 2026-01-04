#!/bin/bash
set -e

# Select platform: set to either "arm64" or "amd64"
PLATFORM="${1:-arm64}"
IMAGE_NAME="rewact_${PLATFORM}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SAM3_DIR="$(dirname "$REPO_DIR")/sam3"

[ -d "$SAM3_DIR" ] || { echo "SAM3 repo not found at $SAM3_DIR"; exit 1; }

BUILD_DIR=$(mktemp -d)
trap "rm -rf $BUILD_DIR" EXIT

# Copy all package directories
cp -r "$REPO_DIR/lerobot_policy_rewact" "$BUILD_DIR/lerobot_policy_rewact"
cp -r "$REPO_DIR/rewact_tools" "$BUILD_DIR/rewact_tools"
cp -r "$REPO_DIR/vision" "$BUILD_DIR/vision"
cp -r "$SAM3_DIR" "$BUILD_DIR/sam3"
cp "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR/.dockerignore" "$BUILD_DIR/"

docker buildx build \
    --platform "linux/${PLATFORM}" \
    -t "$IMAGE_NAME" \
    --load \
    -f "$BUILD_DIR/Dockerfile" \
    "$BUILD_DIR"

echo "Built: $IMAGE_NAME"
