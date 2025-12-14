#!/usr/bin/env bash
set -euo pipefail

# Start script for the Vosk service.
# It will download a default model into /models if the directory is empty,
# then exec the uvicorn server.

MODELPATH=${VOSK_MODEL_PATH:-/models}
MODEL_URL=${VOSK_MODEL_URL:-https://alphacephei.com/vosk/models/vosk-model-small-es-0.22.zip}
MODEL_ZIP="/tmp/vosk_model.zip"

echo "[start.sh] MODELPATH=${MODELPATH}"

mkdir -p "${MODELPATH}"

if [ -z "$(ls -A ${MODELPATH})" ]; then
  echo "[start.sh] /models is empty — downloading default Vosk model"
  echo "[start.sh] MODEL_URL=${MODEL_URL}"
  if command -v wget >/dev/null 2>&1; then
    wget -O "${MODEL_ZIP}" "${MODEL_URL}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -o "${MODEL_ZIP}" "${MODEL_URL}"
  else
    echo "[start.sh] Error: neither wget nor curl is installed"
    exit 1
  fi

  echo "[start.sh] Unpacking model to ${MODELPATH}"
  mkdir -p /tmp/vosk_model_unpack
  unzip -q "${MODEL_ZIP}" -d /tmp/vosk_model_unpack
  # Move unpacked folder contents into MODELPATH
  # If the zip contains a top-level folder, move its contents
  shopt -s dotglob
  mv /tmp/vosk_model_unpack/*/* "${MODELPATH}/" 2>/dev/null || mv /tmp/vosk_model_unpack/* "${MODELPATH}/"
  rm -rf /tmp/vosk_model_unpack
  rm -f "${MODEL_ZIP}"
  echo "[start.sh] Model downloaded and installed into ${MODELPATH}"
else
  echo "[start.sh] /models not empty — skipping download"
fi

echo "[start.sh] Starting uvicorn"
exec "uvicorn" "src.main:app" "--host" "0.0.0.0" "--port" "8000"
