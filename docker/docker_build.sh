#!/bin/bash
set -e

# Select platform: set to either "arm64" or "amd64"
PLATFORM="${1:-arm64}"
IMAGE_NAME="rewact_${PLATFORM}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SAM3_DIR="$(dirname "$REPO_DIR")/sam3"
DINOV3_DIR="$(dirname "$REPO_DIR")/dinov3"

[ -d "$SAM3_DIR" ] || { echo "SAM3 repo not found at $SAM3_DIR"; exit 1; }
[ -d "$DINOV3_DIR" ] || { echo "DINOV3 repo not found at $DINOV3_DIR"; exit 1; }

BUILD_DIR=$(mktemp -d)
trap "rm -rf $BUILD_DIR" EXIT

# Copy all package directories
cp -r "$REPO_DIR/lerobot_policy_rewact" "$BUILD_DIR/lerobot_policy_rewact"
cp -r "$REPO_DIR/rewact_tools" "$BUILD_DIR/rewact_tools"
cp -r "$SAM3_DIR" "$BUILD_DIR/sam3"
cp -r "$DINOV3_DIR" "$BUILD_DIR/dinov3"
cp "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR/.dockerignore" "$BUILD_DIR/"

# Relax numpy constraint in sam3 (requires <2 but lerobot needs >=2)
sed -i 's/"numpy>=1.26,<2"/"numpy>=1.26"/' "$BUILD_DIR/sam3/pyproject.toml"


if [ "$PLATFORM" = "amd64" ]; then
    # Native build - use regular docker build for simpler caching
    docker build \
        -t "$IMAGE_NAME" \
        -f "$BUILD_DIR/Dockerfile" \
        "$BUILD_DIR"
else
    # Cross-platform build - requires buildx
docker buildx build \
    --platform "linux/${PLATFORM}" \
    -t "$IMAGE_NAME" \
    --load \
    -f "$BUILD_DIR/Dockerfile" \
    "$BUILD_DIR"
fi

echo "Built: $IMAGE_NAME"
