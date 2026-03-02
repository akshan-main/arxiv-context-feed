#!/bin/bash
# Download Qwen2.5-14B Q4_K_M GGUF model for local inference.
# Stores in ~/.cache/arxiv-feed/models/ (~8.5GB)
# Skips download if file already exists.

set -euo pipefail

MODEL_DIR="${HOME}/.cache/arxiv-feed/models"
MODEL_NAME="qwen2.5-14b-instruct-q4_k_m.gguf"
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

# HuggingFace URL for the GGUF file
HF_REPO="Qwen/Qwen2.5-14B-Instruct-GGUF"
HF_FILE="qwen2.5-14b-instruct-q4_k_m.gguf"
DOWNLOAD_URL="https://huggingface.co/${HF_REPO}/resolve/main/${HF_FILE}"

echo "Model: ${MODEL_NAME}"
echo "Target: ${MODEL_PATH}"

# Check if already downloaded
if [ -f "${MODEL_PATH}" ]; then
    SIZE=$(du -sh "${MODEL_PATH}" | cut -f1)
    echo "Model already exists (${SIZE}). Skipping download."
    exit 0
fi

# Create directory
mkdir -p "${MODEL_DIR}"

echo "Downloading from HuggingFace (~8.5GB)..."
echo "URL: ${DOWNLOAD_URL}"

# Download with resume support
if command -v wget &> /dev/null; then
    wget -c -O "${MODEL_PATH}" "${DOWNLOAD_URL}"
elif command -v curl &> /dev/null; then
    curl -C - -L -o "${MODEL_PATH}" "${DOWNLOAD_URL}"
else
    echo "Error: wget or curl is required"
    exit 1
fi

SIZE=$(du -sh "${MODEL_PATH}" | cut -f1)
echo "Download complete: ${MODEL_PATH} (${SIZE})"
