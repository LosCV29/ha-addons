"""
Facial Recognition API Server for Home Assistant Add-on
DeepFace-based face detection and recognition
"""

import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# MODEL AND DETECTOR INFORMATION
# =============================================================================

# Recognition models with their characteristics
# RAM usage is approximate and includes model weights
MODEL_INFO = {
    "GhostFaceNet": {
        "embedding_size": 512,
        "accuracy": "excellent",
        "accuracy_score": 99.6,  # LFW benchmark approximate
        "speed": "medium",
        "ram_mb": 1800,
        "description": "Newest model with excellent accuracy, efficient architecture"
    },
    "ArcFace": {
        "embedding_size": 512,
        "accuracy": "excellent",
        "accuracy_score": 99.5,
        "speed": "medium",
        "ram_mb": 1500,
        "description": "State-of-the-art accuracy, recommended for high-end hardware"
    },
    "Facenet512": {
        "embedding_size": 512,
        "accuracy": "excellent",
        "accuracy_score": 99.4,
        "speed": "medium",
        "ram_mb": 1200,
        "description": "Excellent accuracy with good efficiency"
    },
    "VGG-Face": {
        "embedding_size": 2622,
        "accuracy": "very_good",
        "accuracy_score": 98.9,
        "speed": "slow",
        "ram_mb": 2500,
        "description": "Large model, high accuracy but resource intensive"
    },
    "Facenet": {
        "embedding_size": 128,
        "accuracy": "good",
        "accuracy_score": 99.2,
        "speed": "fast",
        "ram_mb": 500,
        "description": "Compact model, good for mid-range hardware"
    },
    "SFace": {
        "embedding_size": 128,
        "accuracy": "good",
        "accuracy_score": 99.0,
        "speed": "fastest",
        "ram_mb": 300,
        "description": "Lightweight model, ideal for Raspberry Pi and low-power devices"
    },
    "Dlib": {
        "embedding_size": 128,
        "accuracy": "good",
        "accuracy_score": 99.1,
        "speed": "medium",
        "ram_mb": 600,
        "description": "Classic reliable model with consistent performance"
    },
    "OpenFace": {
        "embedding_size": 128,
        "accuracy": "fair",
        "accuracy_score": 93.8,
        "speed": "fast",
        "ram_mb": 400,
        "description": "Older model, fast but less accurate"
    },
    "DeepID": {
        "embedding_size": 160,
        "accuracy": "fair",
        "accuracy_score": 97.5,
        "speed": "fast",
        "ram_mb": 350,
        "description": "Compact older model"
    }
}

# Face detector backends with their characteristics
DETECTOR_INFO = {
    "retinaface": {
        "accuracy": "best",
        "accuracy_score": 95,
        "speed": "slow",
        "ram_mb": 500,
        "description": "Best face detection accuracy, handles angles well"
    },
    "mtcnn": {
        "accuracy": "better",
        "accuracy_score": 90,
        "speed": "medium",
        "ram_mb": 200,
        "description": "Good balance of accuracy and speed"
    },
    "ssd": {
        "accuracy": "good",
        "accuracy_score": 85,
        "speed": "fast",
        "ram_mb": 150,
        "description": "Fast detection, good for real-time"
    },
    "mediapipe": {
        "accuracy": "good",
        "accuracy_score": 85,
        "speed": "fast",
        "ram_mb": 100,
        "description": "Google's lightweight detector, efficient"
    },
    "opencv": {
        "accuracy": "basic",
        "accuracy_score": 70,
        "speed": "fastest",
        "ram_mb": 50,
        "description": "Haar cascade, fastest but least accurate"
    },
    "yolov8": {
        "accuracy": "good",
        "accuracy_score": 88,
        "speed": "fast",
        "ram_mb": 300,
        "description": "Modern object detection, requires ultralytics"
    }
}

# Hardware presets - model + detector combinations
HARDWARE_PRESETS = {
    "lightweight": {
        "model_name": "SFace",
        "detector_backend": "mediapipe",
        "description": "Raspberry Pi / Low-power devices (~400MB RAM)",
        "estimated_ram_mb": 400,
        "ensemble": False
    },
    "balanced": {
        "model_name": "Facenet512",
        "detector_backend": "ssd",
        "description": "Mid-range hardware (~1.4GB RAM)",
        "estimated_ram_mb": 1350,
        "ensemble": False
    },
    "accuracy": {
        "model_name": "ArcFace",
        "detector_backend": "retinaface",
        "description": "High-end hardware, maximum accuracy (~2GB RAM)",
        "estimated_ram_mb": 2000,
        "ensemble": False
    },
    "ultra": {
        "model_name": "GhostFaceNet",
        "detector_backend": "retinaface",
        "description": "Maximum single-model accuracy (~2.3GB RAM)",
        "estimated_ram_mb": 2300,
        "ensemble": False
    },
    "ensemble": {
        "model_name": "ArcFace",  # Primary model for single-model fallback
        "detector_backend": "retinaface",
        "description": "Multi-model voting for highest accuracy (~6GB+ RAM)",
        "estimated_ram_mb": 6000,
        "ensemble": True,
        "ensemble_models": ["ArcFace", "Facenet512", "VGG-Face"]
    }
}

# Default ensemble models if not specified
DEFAULT_ENSEMBLE_MODELS = ["ArcFace", "Facenet512", "VGG-Face"]


def get_available_ram_mb() -> int:
    """Get available system RAM in MB."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    # Value is in kB
                    kb = int(line.split()[1])
                    return kb // 1024
    except Exception:
        pass
    return 0


def get_total_ram_mb() -> int:
    """Get total system RAM in MB."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    kb = int(line.split()[1])
                    return kb // 1024
    except Exception:
        pass
    return 0


def resolve_hardware_preset(
    preset: str,
    manual_model: str,
    manual_detector: str,
    manual_ensemble_models: str | None = None
) -> tuple[str, str, str, bool, list[str]]:
    """Resolve hardware preset to actual model and detector.

    Args:
        preset: The hardware preset (auto, lightweight, balanced, accuracy, ultra, ensemble, custom)
        manual_model: User-specified model (used if preset is custom)
        manual_detector: User-specified detector (used if preset is custom)
        manual_ensemble_models: Comma-separated list of models for ensemble mode

    Returns:
        Tuple of (model_name, detector_backend, preset_used, is_ensemble, ensemble_models)
    """
    if preset == "custom":
        return manual_model, manual_detector, "custom", False, []

    if preset == "auto":
        # Auto-select based on available RAM
        total_ram = get_total_ram_mb()
        logger.info(f"Auto-detecting hardware: {total_ram}MB total RAM")

        if total_ram < 2000:  # Less than 2GB
            preset = "lightweight"
            logger.info("Auto-selected: lightweight preset (low RAM detected)")
        elif total_ram < 4000:  # 2-4GB
            preset = "balanced"
            logger.info("Auto-selected: balanced preset (mid-range RAM detected)")
        elif total_ram < 8000:  # 4-8GB
            preset = "accuracy"
            logger.info("Auto-selected: accuracy preset (high RAM detected)")
        elif total_ram < 12000:  # 8-12GB
            preset = "ultra"
            logger.info("Auto-selected: ultra preset (very high RAM detected)")
        else:  # 12GB+
            preset = "ensemble"
            logger.info("Auto-selected: ensemble preset (massive RAM detected!)")

    if preset in HARDWARE_PRESETS:
        config = HARDWARE_PRESETS[preset]
        is_ensemble = config.get("ensemble", False)

        # Get ensemble models
        if is_ensemble:
            if manual_ensemble_models:
                # Parse user-specified ensemble models
                ensemble_models = [m.strip() for m in manual_ensemble_models.split(",")]
                # Validate models exist
                ensemble_models = [m for m in ensemble_models if m in MODEL_INFO]
                if not ensemble_models:
                    ensemble_models = DEFAULT_ENSEMBLE_MODELS
            else:
                ensemble_models = config.get("ensemble_models", DEFAULT_ENSEMBLE_MODELS)
        else:
            ensemble_models = []

        return config["model_name"], config["detector_backend"], preset, is_ensemble, ensemble_models

    # Fallback to balanced
    logger.warning(f"Unknown preset '{preset}', falling back to balanced")
    config = HARDWARE_PRESETS["balanced"]
    return config["model_name"], config["detector_backend"], "balanced", False, []

# =============================================================================
# CONFIGURATION - From environment variables (set by run.sh from add-on config)
# =============================================================================

FACES_DIR = os.environ.get("FACES_DIR", "/share/faces")
DISTANCE_THRESHOLD = float(os.environ.get("DISTANCE_THRESHOLD", "0.45"))
MIN_FACE_CONFIDENCE = float(os.environ.get("MIN_FACE_CONFIDENCE", "0.80"))
MIN_FACE_SIZE = int(os.environ.get("MIN_FACE_SIZE", "40"))
MAX_CONSIDERATION_DISTANCE = float(os.environ.get("MAX_CONSIDERATION_DISTANCE", "0.60"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8100"))

# Hardware preset handling
HARDWARE_PRESET = os.environ.get("HARDWARE_PRESET", "balanced")
_MANUAL_MODEL = os.environ.get("MODEL_NAME", "Facenet512")
_MANUAL_DETECTOR = os.environ.get("DETECTOR_BACKEND", "retinaface")
_MANUAL_ENSEMBLE_MODELS = os.environ.get("ENSEMBLE_MODELS", None)

# Resolve the actual model and detector based on preset
MODEL_NAME, DETECTOR_BACKEND, ACTIVE_PRESET, ENSEMBLE_MODE, ENSEMBLE_MODELS = resolve_hardware_preset(
    HARDWARE_PRESET, _MANUAL_MODEL, _MANUAL_DETECTOR, _MANUAL_ENSEMBLE_MODELS
)

# Embedding cache configuration
EMBEDDINGS_CACHE_FILE = os.path.join(FACES_DIR, ".embeddings_cache.json")

# =============================================================================
# GLOBALS
# =============================================================================

app = FastAPI(
    title="Facial Recognition API",
    description="Home Assistant Add-on for facial recognition",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single model mode: {person_name: [embeddings]}
known_faces: dict[str, list[np.ndarray]] = {}

# Ensemble mode: {model_name: {person_name: [embeddings]}}
ensemble_faces: dict[str, dict[str, list[np.ndarray]]] = {}

deepface = None


# =============================================================================
# MODELS
# =============================================================================

class IdentifyRequest(BaseModel):
    image_base64: str = Field(validation_alias=AliasChoices('image_base64', 'image'))
    tolerance: float | None = None


class IdentifyResponse(BaseModel):
    success: bool
    faces_detected: int
    people: list[dict[str, Any]]
    summary: str
    error: str | None = None


class StatusResponse(BaseModel):
    status: str
    known_people: list[str]
    total_embeddings: int
    faces_dir: str
    model: str
    threshold: float
    detector: str
    min_confidence: float
    min_face_size: int
    # New fields for model/detector info
    hardware_preset: str
    model_info: dict[str, Any]
    detector_info: dict[str, Any]
    system_ram_mb: int
    available_ram_mb: int
    estimated_usage_mb: int
    available_models: list[str]
    available_detectors: list[str]
    # Ensemble mode info
    ensemble_mode: bool
    ensemble_models: list[str]
    ensemble_embeddings: dict[str, int]  # model_name -> count of embeddings


# =============================================================================
# FACE RECOGNITION FUNCTIONS
# =============================================================================

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine distance between two vectors."""
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    return 1 - similarity


# =============================================================================
# EMBEDDING CACHE FUNCTIONS
# =============================================================================

def get_image_files_info(faces_path: Path) -> dict[str, dict[str, float]]:
    """Get all image files and their modification timestamps.

    Returns:
        Dict mapping person_name -> {image_filename: mtime}
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files_info = {}

    if not faces_path.exists():
        return files_info

    for person_dir in faces_path.iterdir():
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        files_info[person_name] = {}

        for image_file in person_dir.iterdir():
            if image_file.suffix.lower() not in image_extensions:
                continue
            files_info[person_name][image_file.name] = image_file.stat().st_mtime

    return files_info


def save_embeddings_cache(
    faces: dict[str, list[np.ndarray]],
    files_info: dict[str, dict[str, float]],
    embeddings_by_file: dict[str, dict[str, list[float]]]
) -> None:
    """Save embeddings cache to disk.

    Args:
        faces: Dict mapping person_name -> list of numpy arrays
        files_info: Dict mapping person_name -> {filename: mtime}
        embeddings_by_file: Dict mapping person_name -> {filename: embedding_list}
    """
    cache_data = {
        "model_name": MODEL_NAME,
        "detector_backend": DETECTOR_BACKEND,
        "files_info": files_info,
        "embeddings": embeddings_by_file
    }

    try:
        cache_path = Path(EMBEDDINGS_CACHE_FILE)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        logger.info(f"Saved embeddings cache to {EMBEDDINGS_CACHE_FILE}")
    except Exception as e:
        logger.warning(f"Failed to save embeddings cache: {e}")


def load_embeddings_cache() -> tuple[dict[str, list[np.ndarray]], dict[str, dict[str, list[float]]], set[tuple[str, str]]] | None:
    """Load embeddings from cache if valid.

    Returns:
        Tuple of (faces_dict, embeddings_by_file, files_to_skip) if cache is valid,
        None if cache is invalid or doesn't exist.
        files_to_skip contains (person_name, filename) tuples for files that are cached.
    """
    cache_path = Path(EMBEDDINGS_CACHE_FILE)

    if not cache_path.exists():
        logger.info("No embeddings cache found")
        return None

    try:
        with open(cache_path, "r") as f:
            cache_data = json.load(f)

        # Validate model and detector match
        if cache_data.get("model_name") != MODEL_NAME:
            logger.info(f"Cache model mismatch: {cache_data.get('model_name')} != {MODEL_NAME}")
            return None

        if cache_data.get("detector_backend") != DETECTOR_BACKEND:
            logger.info(f"Cache detector mismatch: {cache_data.get('detector_backend')} != {DETECTOR_BACKEND}")
            return None

        cached_files_info = cache_data.get("files_info", {})
        cached_embeddings = cache_data.get("embeddings", {})

        # Get current files info
        faces_path = Path(FACES_DIR)
        current_files_info = get_image_files_info(faces_path)

        # Build faces dict from cache, tracking which files are still valid
        faces = {}
        valid_embeddings_by_file = {}
        files_to_skip = set()

        for person_name, person_embeddings in cached_embeddings.items():
            # Check if person still exists
            if person_name not in current_files_info:
                logger.info(f"Person {person_name} removed, skipping cached embeddings")
                continue

            current_person_files = current_files_info[person_name]
            cached_person_files = cached_files_info.get(person_name, {})

            valid_embeddings = []
            valid_embeddings_files = {}

            for filename, embedding in person_embeddings.items():
                # Check if file still exists with same timestamp
                if filename not in current_person_files:
                    logger.info(f"File {filename} for {person_name} removed")
                    continue

                current_mtime = current_person_files[filename]
                cached_mtime = cached_person_files.get(filename)

                if cached_mtime is None or abs(current_mtime - cached_mtime) > 1:
                    logger.info(f"File {filename} for {person_name} modified, will regenerate")
                    continue

                # Cache entry is valid
                valid_embeddings.append(np.array(embedding))
                valid_embeddings_files[filename] = embedding
                files_to_skip.add((person_name, filename))

            if valid_embeddings:
                faces[person_name] = valid_embeddings
                valid_embeddings_by_file[person_name] = valid_embeddings_files

        logger.info(f"Loaded {sum(len(e) for e in faces.values())} embeddings from cache for {len(faces)} people")
        return faces, valid_embeddings_by_file, files_to_skip

    except Exception as e:
        logger.warning(f"Failed to load embeddings cache: {e}")
        return None


def load_known_faces(force_reload: bool = False) -> dict[str, list[np.ndarray]]:
    """Load all reference faces from the faces directory.

    Uses caching to avoid regenerating embeddings for unchanged images.

    Args:
        force_reload: If True, ignore cache and regenerate all embeddings.
    """
    global deepface

    try:
        from deepface import DeepFace
        deepface = DeepFace
        logger.info(f"DeepFace loaded successfully, using {MODEL_NAME} model")
    except ImportError as e:
        logger.error(f"DeepFace not installed: {e}")
        return {}

    faces_path = Path(FACES_DIR)

    if not faces_path.exists():
        logger.warning(f"Faces directory does not exist: {FACES_DIR}")
        logger.info(f"Creating directory: {FACES_DIR}")
        faces_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Add photos to {FACES_DIR}/PersonName/ folders")
        return {}

    # Try to load from cache first
    faces = {}
    embeddings_by_file = {}
    files_to_skip = set()

    if not force_reload:
        cache_result = load_embeddings_cache()
        if cache_result:
            faces, embeddings_by_file, files_to_skip = cache_result
            if files_to_skip:
                logger.info(f"Using {len(files_to_skip)} cached embeddings")

    # Get current files info for saving cache later
    current_files_info = get_image_files_info(faces_path)

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    new_embeddings_generated = False

    for person_dir in faces_path.iterdir():
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name

        # Initialize if person is new
        if person_name not in faces:
            faces[person_name] = []
        if person_name not in embeddings_by_file:
            embeddings_by_file[person_name] = {}

        for image_file in person_dir.iterdir():
            if image_file.suffix.lower() not in image_extensions:
                continue

            # Skip if already loaded from cache
            if (person_name, image_file.name) in files_to_skip:
                logger.info(f"  ✓ Using cached embedding for {image_file.name} ({person_name})")
                continue

            try:
                logger.info(f"Loading {image_file.name} for {person_name}...")

                result = deepface.represent(
                    img_path=str(image_file),
                    model_name=MODEL_NAME,
                    enforce_detection=True,
                    detector_backend=DETECTOR_BACKEND
                )

                if result and len(result) > 0:
                    embedding = np.array(result[0]["embedding"])
                    faces[person_name].append(embedding)
                    embeddings_by_file[person_name][image_file.name] = embedding.tolist()
                    new_embeddings_generated = True
                    logger.info(f"  ✓ Loaded embedding from {image_file.name}")
                else:
                    logger.warning(f"  ✗ No face found in {image_file.name}")

            except Exception as e:
                logger.warning(f"  ✗ Error loading {image_file}: {e}")

    # Remove people with no embeddings
    faces = {k: v for k, v in faces.items() if v}
    embeddings_by_file = {k: v for k, v in embeddings_by_file.items() if v}

    # Log summary for each person
    for person_name, person_embeddings in faces.items():
        logger.info(f"Loaded {len(person_embeddings)} embeddings for {person_name}")

    logger.info(f"Total: {len(faces)} people loaded")

    # Save updated cache if new embeddings were generated
    if new_embeddings_generated or force_reload:
        save_embeddings_cache(faces, current_files_info, embeddings_by_file)

    return faces


def load_ensemble_faces(force_reload: bool = False) -> dict[str, dict[str, list[np.ndarray]]]:
    """Load faces for all models in ensemble mode.

    Returns:
        Dict mapping model_name -> {person_name: [embeddings]}
    """
    global deepface

    try:
        from deepface import DeepFace
        deepface = DeepFace
    except ImportError as e:
        logger.error(f"DeepFace not installed: {e}")
        return {}

    faces_path = Path(FACES_DIR)

    if not faces_path.exists():
        logger.warning(f"Faces directory does not exist: {FACES_DIR}")
        faces_path.mkdir(parents=True, exist_ok=True)
        return {}

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_model_faces = {}

    # Load embeddings for each model in the ensemble
    for model_name in ENSEMBLE_MODELS:
        logger.info(f"")
        logger.info(f"Loading embeddings for model: {model_name}")
        logger.info("-" * 40)

        model_faces = {}

        for person_dir in faces_path.iterdir():
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            model_faces[person_name] = []

            for image_file in person_dir.iterdir():
                if image_file.suffix.lower() not in image_extensions:
                    continue

                try:
                    result = deepface.represent(
                        img_path=str(image_file),
                        model_name=model_name,
                        enforce_detection=True,
                        detector_backend=DETECTOR_BACKEND
                    )

                    if result and len(result) > 0:
                        embedding = np.array(result[0]["embedding"])
                        model_faces[person_name].append(embedding)
                        logger.info(f"  ✓ {person_name}/{image_file.name}")

                except Exception as e:
                    logger.warning(f"  ✗ {person_name}/{image_file.name}: {e}")

        # Remove people with no embeddings
        model_faces = {k: v for k, v in model_faces.items() if v}
        all_model_faces[model_name] = model_faces

        total = sum(len(e) for e in model_faces.values())
        logger.info(f"  Total: {total} embeddings for {len(model_faces)} people")

    return all_model_faces


def identify_with_ensemble(image_bytes: bytes, threshold: float) -> dict[str, Any]:
    """Identify faces using ensemble voting across multiple models."""
    global deepface, ensemble_faces

    if deepface is None:
        return {
            "success": False,
            "faces_detected": 0,
            "people": [],
            "summary": "DeepFace not loaded",
            "error": "DeepFace not initialized"
        }

    if not ensemble_faces:
        return {
            "success": True,
            "faces_detected": 0,
            "people": [],
            "summary": f"No ensemble faces loaded. Add photos to {FACES_DIR}/PersonName/",
            "error": None
        }

    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))

        if img.mode != "RGB":
            img = img.convert("RGB")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            img.save(tmp_path, "JPEG")

        try:
            # Extract faces first
            logger.info(f"[ENSEMBLE] Extracting faces with {DETECTOR_BACKEND}...")
            face_objs = deepface.extract_faces(
                img_path=tmp_path,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True
            )

            # Filter valid faces
            valid_faces = []
            for face_obj in face_objs:
                confidence = face_obj.get("confidence", 0)
                facial_area = face_obj.get("facial_area", {})
                width = facial_area.get("w", 0)
                height = facial_area.get("h", 0)

                if confidence >= MIN_FACE_CONFIDENCE and width >= MIN_FACE_SIZE and height >= MIN_FACE_SIZE:
                    valid_faces.append(face_obj)

            if not valid_faces:
                return {
                    "success": True,
                    "faces_detected": 0,
                    "people": [],
                    "summary": "No faces detected"
                }

            # For each detected face, get votes from all models
            identified = []

            for face_idx in range(len(valid_faces)):
                votes = {}  # person_name -> list of (distance, model_name)

                for model_name in ENSEMBLE_MODELS:
                    if model_name not in ensemble_faces:
                        continue

                    model_known_faces = ensemble_faces[model_name]

                    try:
                        # Get embedding for this model
                        results = deepface.represent(
                            img_path=tmp_path,
                            model_name=model_name,
                            enforce_detection=False,
                            detector_backend=DETECTOR_BACKEND
                        )

                        if not results or face_idx >= len(results):
                            continue

                        embedding = np.array(results[face_idx]["embedding"])

                        # Find best match for this model
                        best_match = None
                        best_distance = float("inf")

                        for person_name, known_embeddings in model_known_faces.items():
                            for known_emb in known_embeddings:
                                distance = cosine_distance(embedding, known_emb)
                                if distance < best_distance:
                                    best_distance = distance
                                    best_match = person_name

                        if best_match and best_distance <= threshold:
                            if best_match not in votes:
                                votes[best_match] = []
                            votes[best_match].append((best_distance, model_name))
                            logger.info(f"  [ENSEMBLE] {model_name} votes: {best_match} (d={best_distance:.3f})")

                    except Exception as e:
                        logger.warning(f"  [ENSEMBLE] {model_name} error: {e}")

                # Determine winner by vote count, then by average distance
                if votes:
                    # Sort by: number of votes (desc), then average distance (asc)
                    vote_scores = []
                    for person_name, model_votes in votes.items():
                        num_votes = len(model_votes)
                        avg_distance = sum(d for d, _ in model_votes) / num_votes
                        vote_scores.append((person_name, num_votes, avg_distance, model_votes))

                    vote_scores.sort(key=lambda x: (-x[1], x[2]))  # More votes better, lower distance better
                    winner = vote_scores[0]

                    person_name = winner[0]
                    num_votes = winner[1]
                    avg_distance = winner[2]
                    voting_models = [m for _, m in winner[3]]

                    confidence = round((1 - avg_distance) * 100, 1)

                    identified.append({
                        "name": person_name,
                        "confidence": confidence,
                        "distance": round(avg_distance, 3),
                        "status": "identified",
                        "votes": num_votes,
                        "total_models": len(ENSEMBLE_MODELS),
                        "voting_models": voting_models
                    })
                    logger.info(f"  [ENSEMBLE] Winner: {person_name} ({num_votes}/{len(ENSEMBLE_MODELS)} votes, {confidence}%)")
                else:
                    identified.append({
                        "name": "Unknown",
                        "confidence": 0,
                        "distance": None,
                        "status": "unknown",
                        "votes": 0,
                        "total_models": len(ENSEMBLE_MODELS),
                        "voting_models": []
                    })

        finally:
            os.unlink(tmp_path)

        # Build summary
        names = [p["name"] for p in identified]
        known_names = [n for n in names if n != "Unknown"]
        unknown_count = names.count("Unknown")

        if known_names and unknown_count:
            summary = f"[ENSEMBLE] Detected: {', '.join(known_names)} and {unknown_count} unknown"
        elif known_names:
            summary = f"[ENSEMBLE] Detected: {', '.join(known_names)}"
        elif unknown_count:
            summary = f"[ENSEMBLE] {unknown_count} unknown person(s)"
        else:
            summary = "[ENSEMBLE] No faces identified"

        return {
            "success": True,
            "faces_detected": len(valid_faces),
            "people": identified,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"[ENSEMBLE] Error: {e}", exc_info=True)
        return {
            "success": False,
            "faces_detected": 0,
            "people": [],
            "summary": f"Error: {str(e)}",
            "error": str(e)
        }


def identify_faces_in_image(image_bytes: bytes, threshold: float) -> dict[str, Any]:
    """Identify faces in an image."""
    global deepface, known_faces

    if deepface is None:
        return {
            "success": False,
            "faces_detected": 0,
            "people": [],
            "summary": "DeepFace not loaded",
            "error": "DeepFace not initialized"
        }

    if not known_faces:
        return {
            "success": True,
            "faces_detected": 0,
            "people": [],
            "summary": f"No known faces loaded. Add photos to {FACES_DIR}/PersonName/",
            "error": None
        }

    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))

        if img.mode != "RGB":
            img = img.convert("RGB")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            img.save(tmp_path, "JPEG")

        try:
            # Extract faces
            logger.info(f"Extracting faces with {DETECTOR_BACKEND} detector...")
            face_objs = deepface.extract_faces(
                img_path=tmp_path,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True
            )

            logger.info(f"Raw detections: {len(face_objs)} potential faces found")

            # Filter faces
            valid_faces = []
            for i, face_obj in enumerate(face_objs):
                confidence = face_obj.get("confidence", 0)
                facial_area = face_obj.get("facial_area", {})
                width = facial_area.get("w", 0)
                height = facial_area.get("h", 0)

                logger.info(f"Face {i+1}: confidence={confidence:.3f}, size={width}x{height}")

                if confidence < MIN_FACE_CONFIDENCE:
                    logger.info(f"  ✗ Rejected: confidence {confidence:.3f} < {MIN_FACE_CONFIDENCE}")
                    continue

                if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
                    logger.info(f"  ✗ Rejected: size {width}x{height} < {MIN_FACE_SIZE}px")
                    continue

                if width > 0 and height > 0:
                    aspect_ratio = width / height
                    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                        logger.info(f"  ✗ Rejected: aspect ratio {aspect_ratio:.2f} out of range")
                        continue

                logger.info(f"  ✓ Accepted: valid face detection")
                valid_faces.append(face_obj)

            if not valid_faces:
                logger.info("No valid faces after filtering")
                return {
                    "success": True,
                    "faces_detected": 0,
                    "people": [],
                    "summary": "No faces detected"
                }

            # Get embeddings
            results = deepface.represent(
                img_path=tmp_path,
                model_name=MODEL_NAME,
                enforce_detection=False,
                detector_backend=DETECTOR_BACKEND
            )

        finally:
            os.unlink(tmp_path)

        if not results:
            return {
                "success": True,
                "faces_detected": 0,
                "people": [],
                "summary": "No faces detected"
            }

        results_to_process = results[:len(valid_faces)]

        # Identify each face
        identified = []
        for idx, face_result in enumerate(results_to_process):
            embedding = np.array(face_result["embedding"])

            best_match = None
            best_distance = float("inf")

            for person_name, known_embeddings in known_faces.items():
                for known_emb in known_embeddings:
                    distance = cosine_distance(embedding, known_emb)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = person_name

            logger.info(f"Face {idx+1}: best_match={best_match}, distance={best_distance:.3f}, threshold={threshold}")

            if best_match and best_distance <= threshold and best_distance <= MAX_CONSIDERATION_DISTANCE:
                confidence = round((1 - best_distance) * 100, 1)
                identified.append({
                    "name": best_match,
                    "confidence": confidence,
                    "distance": round(best_distance, 3),
                    "status": "identified"
                })
                logger.info(f"  → Identified as {best_match} ({confidence}%)")
            else:
                confidence = round((1 - best_distance) * 100, 1) if best_distance != float("inf") else 0
                identified.append({
                    "name": "Unknown",
                    "confidence": confidence,
                    "distance": round(best_distance, 3) if best_distance != float("inf") else None,
                    "status": "unknown",
                    "closest_match": best_match
                })
                logger.info(f"  → Unknown person (closest: {best_match})")

        # Build summary
        names = [p["name"] for p in identified]
        known_names = [n for n in names if n != "Unknown"]
        unknown_count = names.count("Unknown")

        if known_names and unknown_count:
            summary = f"Detected: {', '.join(known_names)} and {unknown_count} unknown"
        elif known_names:
            summary = f"Detected: {', '.join(known_names)}"
        elif unknown_count:
            summary = f"Unknown person detected" if unknown_count == 1 else f"{unknown_count} unknown people"
        else:
            summary = "No faces identified"

        return {
            "success": True,
            "faces_detected": len(valid_faces),
            "people": identified,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Error identifying faces: {e}", exc_info=True)
        return {
            "success": False,
            "faces_detected": 0,
            "people": [],
            "summary": f"Error: {str(e)}",
            "error": str(e)
        }


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load known faces on startup."""
    global known_faces, ensemble_faces

    # Get system info
    total_ram = get_total_ram_mb()
    available_ram = get_available_ram_mb()
    model_info = MODEL_INFO.get(MODEL_NAME, {})
    detector_info = DETECTOR_INFO.get(DETECTOR_BACKEND, {})

    logger.info("=" * 60)
    logger.info("Facial Recognition Add-on Starting")
    logger.info("=" * 60)
    logger.info("")
    logger.info("SYSTEM INFO:")
    logger.info(f"  Total RAM: {total_ram}MB")
    logger.info(f"  Available RAM: {available_ram}MB")
    logger.info("")
    logger.info("CONFIGURATION:")
    logger.info(f"  Hardware preset: {ACTIVE_PRESET}")
    logger.info(f"  Faces directory: {FACES_DIR}")

    if ENSEMBLE_MODE:
        # Ensemble mode
        logger.info("")
        logger.info("ENSEMBLE MODE ENABLED!")
        logger.info(f"  Models: {', '.join(ENSEMBLE_MODELS)}")
        estimated_ram = sum(MODEL_INFO.get(m, {}).get("ram_mb", 0) for m in ENSEMBLE_MODELS)
        estimated_ram += detector_info.get("ram_mb", 0)
        logger.info(f"  Face detector: {DETECTOR_BACKEND}")
        logger.info(f"  Estimated total RAM usage: ~{estimated_ram}MB")
        logger.info("")
        logger.info("Loading embeddings for all ensemble models...")
        logger.info("(This may take a while with multiple models)")
    else:
        # Single model mode
        estimated_ram = model_info.get("ram_mb", 0) + detector_info.get("ram_mb", 0)
        logger.info("")
        logger.info("MODEL SETTINGS:")
        logger.info(f"  Recognition model: {MODEL_NAME}")
        logger.info(f"    - Accuracy: {model_info.get('accuracy', 'unknown')} ({model_info.get('accuracy_score', 'N/A')}% LFW)")
        logger.info(f"    - Speed: {model_info.get('speed', 'unknown')}")
        logger.info(f"    - RAM: ~{model_info.get('ram_mb', 'N/A')}MB")
        logger.info(f"  Face detector: {DETECTOR_BACKEND}")
        logger.info(f"    - Accuracy: {detector_info.get('accuracy', 'unknown')}")
        logger.info(f"    - Speed: {detector_info.get('speed', 'unknown')}")
        logger.info(f"    - RAM: ~{detector_info.get('ram_mb', 'N/A')}MB")
        logger.info(f"  Estimated total RAM usage: ~{estimated_ram}MB")

    logger.info("")
    logger.info("THRESHOLDS:")
    logger.info(f"  Distance threshold: {DISTANCE_THRESHOLD}")
    logger.info(f"  Min face confidence: {MIN_FACE_CONFIDENCE}")
    logger.info(f"  Min face size: {MIN_FACE_SIZE}px")
    logger.info("=" * 60)

    # Load faces based on mode
    if ENSEMBLE_MODE:
        ensemble_faces = load_ensemble_faces()
        if not ensemble_faces or not any(ensemble_faces.values()):
            logger.info("")
            logger.info("No faces loaded! To add people:")
            logger.info(f"1. Go to {FACES_DIR} via Samba or SSH")
            logger.info(f"2. Create a folder for each person (e.g., {FACES_DIR}/John/")
            logger.info("3. Add 3-5 photos of their face to the folder")
            logger.info("4. Restart this add-on or call POST /reload")
            logger.info("")
        else:
            # Also populate known_faces with the primary model for backward compat
            if ENSEMBLE_MODELS and ENSEMBLE_MODELS[0] in ensemble_faces:
                known_faces = ensemble_faces[ENSEMBLE_MODELS[0]]
    else:
        known_faces = load_known_faces()
        if not known_faces:
            logger.info("")
            logger.info("No faces loaded! To add people:")
            logger.info(f"1. Go to {FACES_DIR} via Samba or SSH")
            logger.info(f"2. Create a folder for each person (e.g., {FACES_DIR}/John/")
            logger.info("3. Add 3-5 photos of their face to the folder")
            logger.info("4. Restart this add-on or call POST /reload")
            logger.info("")

    logger.info("=" * 60)
    logger.info(f"Server ready on http://{HOST}:{PORT}")
    if ENSEMBLE_MODE:
        logger.info(f"Running in ENSEMBLE mode with {len(ENSEMBLE_MODELS)} models")
    logger.info("=" * 60)


@app.get("/", response_model=StatusResponse)
@app.get("/status", response_model=StatusResponse)
async def status():
    """Get server status and configuration info."""
    # Calculate estimated RAM usage
    detector_ram = DETECTOR_INFO.get(DETECTOR_BACKEND, {}).get("ram_mb", 0)

    if ENSEMBLE_MODE:
        model_ram = sum(MODEL_INFO.get(m, {}).get("ram_mb", 0) for m in ENSEMBLE_MODELS)
        # Get people from first model in ensemble
        if ensemble_faces and ENSEMBLE_MODELS:
            first_model = ENSEMBLE_MODELS[0]
            people = list(ensemble_faces.get(first_model, {}).keys())
            total_emb = sum(
                sum(len(e) for e in model_faces.values())
                for model_faces in ensemble_faces.values()
            )
        else:
            people = []
            total_emb = 0
        ensemble_emb_counts = {
            m: sum(len(e) for e in faces.values())
            for m, faces in ensemble_faces.items()
        }
    else:
        model_ram = MODEL_INFO.get(MODEL_NAME, {}).get("ram_mb", 0)
        people = list(known_faces.keys())
        total_emb = sum(len(e) for e in known_faces.values())
        ensemble_emb_counts = {}

    estimated_usage = model_ram + detector_ram

    return StatusResponse(
        status="running",
        known_people=people,
        total_embeddings=total_emb,
        faces_dir=FACES_DIR,
        model=MODEL_NAME if not ENSEMBLE_MODE else f"ENSEMBLE ({len(ENSEMBLE_MODELS)} models)",
        threshold=DISTANCE_THRESHOLD,
        detector=DETECTOR_BACKEND,
        min_confidence=MIN_FACE_CONFIDENCE,
        min_face_size=MIN_FACE_SIZE,
        # Model/detector info
        hardware_preset=ACTIVE_PRESET,
        model_info=MODEL_INFO.get(MODEL_NAME, {}),
        detector_info=DETECTOR_INFO.get(DETECTOR_BACKEND, {}),
        system_ram_mb=get_total_ram_mb(),
        available_ram_mb=get_available_ram_mb(),
        estimated_usage_mb=estimated_usage,
        available_models=list(MODEL_INFO.keys()),
        available_detectors=list(DETECTOR_INFO.keys()),
        # Ensemble info
        ensemble_mode=ENSEMBLE_MODE,
        ensemble_models=ENSEMBLE_MODELS,
        ensemble_embeddings=ensemble_emb_counts
    )


@app.get("/models")
async def list_models():
    """List all available recognition models with their characteristics."""
    return {
        "models": MODEL_INFO,
        "current_model": MODEL_NAME,
        "presets": HARDWARE_PRESETS
    }


@app.get("/detectors")
async def list_detectors():
    """List all available face detector backends with their characteristics."""
    return {
        "detectors": DETECTOR_INFO,
        "current_detector": DETECTOR_BACKEND
    }


@app.post("/identify", response_model=IdentifyResponse)
async def identify_base64(request: IdentifyRequest):
    """Identify faces in a base64-encoded image."""
    try:
        image_bytes = base64.b64decode(request.image_base64)
        threshold = request.tolerance or DISTANCE_THRESHOLD

        # Use ensemble or single model based on mode
        if ENSEMBLE_MODE:
            result = identify_with_ensemble(image_bytes, threshold)
        else:
            result = identify_faces_in_image(image_bytes, threshold)

        return IdentifyResponse(**result)
    except Exception as e:
        logger.error(f"Error in /identify: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload")
async def reload_faces(force: bool = False):
    """Reload reference faces from disk.

    Args:
        force: If True, ignore cache and regenerate all embeddings.
    """
    global known_faces
    if force:
        logger.info("Force reloading known faces (ignoring cache)...")
    else:
        logger.info("Reloading known faces...")
    known_faces = load_known_faces(force_reload=force)
    return {
        "success": True,
        "known_people": list(known_faces.keys()),
        "total_embeddings": sum(len(e) for e in known_faces.values()),
        "cache_cleared": force
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
