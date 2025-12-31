#!/bin/bash
set -e

# Read configuration from add-on options using jq
OPTIONS_FILE="/data/options.json"

DISTANCE_THRESHOLD=$(jq -r '.distance_threshold' "$OPTIONS_FILE")
MIN_FACE_CONFIDENCE=$(jq -r '.min_face_confidence' "$OPTIONS_FILE")
MIN_FACE_SIZE=$(jq -r '.min_face_size' "$OPTIONS_FILE")
MODEL_NAME=$(jq -r '.model_name' "$OPTIONS_FILE")
DETECTOR_BACKEND=$(jq -r '.detector_backend' "$OPTIONS_FILE")

# Set environment variables
export DISTANCE_THRESHOLD="${DISTANCE_THRESHOLD}"
export MIN_FACE_CONFIDENCE="${MIN_FACE_CONFIDENCE}"
export MIN_FACE_SIZE="${MIN_FACE_SIZE}"
export MODEL_NAME="${MODEL_NAME}"
export DETECTOR_BACKEND="${DETECTOR_BACKEND}"
export FACES_DIR="/config/camera_faces"
export HOST="0.0.0.0"
export PORT="8100"

# Create faces directory if it doesn't exist
mkdir -p /config/camera_faces

echo "Starting Facial Recognition Server..."
echo "Faces directory: ${FACES_DIR}"
echo "Model: ${MODEL_NAME}"
echo "Distance threshold: ${DISTANCE_THRESHOLD}"
echo "Detector: ${DETECTOR_BACKEND}"

# Start the server
exec python3 /server.py
