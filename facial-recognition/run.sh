#!/bin/bash
set -e

# Read configuration from add-on options using jq
OPTIONS_FILE="/data/options.json"

# Read all configuration options
HARDWARE_PRESET=$(jq -r '.hardware_preset // "balanced"' "$OPTIONS_FILE")
DISTANCE_THRESHOLD=$(jq -r '.distance_threshold' "$OPTIONS_FILE")
MIN_FACE_CONFIDENCE=$(jq -r '.min_face_confidence' "$OPTIONS_FILE")
MIN_FACE_SIZE=$(jq -r '.min_face_size' "$OPTIONS_FILE")
MODEL_NAME=$(jq -r '.model_name' "$OPTIONS_FILE")
DETECTOR_BACKEND=$(jq -r '.detector_backend' "$OPTIONS_FILE")
ENSEMBLE_MODELS=$(jq -r '.ensemble_models // "ArcFace,Facenet512,VGG-Face"' "$OPTIONS_FILE")

# Set environment variables
export HARDWARE_PRESET="${HARDWARE_PRESET}"
export DISTANCE_THRESHOLD="${DISTANCE_THRESHOLD}"
export MIN_FACE_CONFIDENCE="${MIN_FACE_CONFIDENCE}"
export MIN_FACE_SIZE="${MIN_FACE_SIZE}"
export MODEL_NAME="${MODEL_NAME}"
export DETECTOR_BACKEND="${DETECTOR_BACKEND}"
export ENSEMBLE_MODELS="${ENSEMBLE_MODELS}"
export FACES_DIR="/homeassistant/camera_faces"
export HOST="0.0.0.0"
export PORT="8100"

# Create faces directory if it doesn't exist
mkdir -p /homeassistant/camera_faces

echo "Starting Facial Recognition Server..."
echo "Hardware preset: ${HARDWARE_PRESET}"
echo "Faces directory: ${FACES_DIR}"
echo "Distance threshold: ${DISTANCE_THRESHOLD}"

if [ "${HARDWARE_PRESET}" = "ensemble" ]; then
    echo "ENSEMBLE MODE: Using multiple models for voting"
    echo "Ensemble models: ${ENSEMBLE_MODELS}"
elif [ "${HARDWARE_PRESET}" = "custom" ]; then
    echo "Custom model: ${MODEL_NAME}"
    echo "Custom detector: ${DETECTOR_BACKEND}"
else
    echo "Model/detector will be auto-selected based on preset"
fi

# Start the server
exec python3 /server.py
