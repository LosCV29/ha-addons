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
# CONFIGURATION - From environment variables (set by run.sh from add-on config)
# =============================================================================

FACES_DIR = os.environ.get("FACES_DIR", "/share/faces")
DISTANCE_THRESHOLD = float(os.environ.get("DISTANCE_THRESHOLD", "0.45"))
MODEL_NAME = os.environ.get("MODEL_NAME", "Facenet512")
DETECTOR_BACKEND = os.environ.get("DETECTOR_BACKEND", "retinaface")
MIN_FACE_CONFIDENCE = float(os.environ.get("MIN_FACE_CONFIDENCE", "0.80"))
MIN_FACE_SIZE = int(os.environ.get("MIN_FACE_SIZE", "40"))
MAX_CONSIDERATION_DISTANCE = float(os.environ.get("MAX_CONSIDERATION_DISTANCE", "0.60"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8100"))

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

known_faces: dict[str, list[np.ndarray]] = {}
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
    global known_faces
    logger.info("=" * 60)
    logger.info("Facial Recognition Add-on Starting")
    logger.info(f"Faces directory: {FACES_DIR}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Distance threshold: {DISTANCE_THRESHOLD}")
    logger.info(f"Detector: {DETECTOR_BACKEND}")
    logger.info(f"Min face confidence: {MIN_FACE_CONFIDENCE}")
    logger.info(f"Min face size: {MIN_FACE_SIZE}px")
    logger.info("=" * 60)
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
    logger.info("=" * 60)


@app.get("/", response_model=StatusResponse)
@app.get("/status", response_model=StatusResponse)
async def status():
    """Get server status."""
    return StatusResponse(
        status="running",
        known_people=list(known_faces.keys()),
        total_embeddings=sum(len(e) for e in known_faces.values()),
        faces_dir=FACES_DIR,
        model=MODEL_NAME,
        threshold=DISTANCE_THRESHOLD,
        detector=DETECTOR_BACKEND,
        min_confidence=MIN_FACE_CONFIDENCE,
        min_face_size=MIN_FACE_SIZE
    )


@app.post("/identify", response_model=IdentifyResponse)
async def identify_base64(request: IdentifyRequest):
    """Identify faces in a base64-encoded image."""
    try:
        image_bytes = base64.b64decode(request.image_base64)
        threshold = request.tolerance or DISTANCE_THRESHOLD
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
