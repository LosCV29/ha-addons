"""HA Video Vision - AI Camera Analysis with Auto-Discovery and Facial Recognition."""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import aiofiles
import aiohttp
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.components.camera import async_get_image, async_get_stream_source

from .const import (
    DOMAIN,
    # Provider
    CONF_PROVIDER,
    CONF_API_KEY,
    CONF_PROVIDER_CONFIGS,
    CONF_DEFAULT_PROVIDER,
    PROVIDER_LOCAL,
    PROVIDER_GOOGLE,
    PROVIDER_OPENROUTER,
    PROVIDER_BASE_URLS,
    PROVIDER_DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    # AI Settings
    CONF_VLLM_URL,
    CONF_VLLM_MODEL,
    CONF_VLLM_MAX_TOKENS,
    CONF_VLLM_TEMPERATURE,
    DEFAULT_VLLM_URL,
    DEFAULT_VLLM_MODEL,
    DEFAULT_VLLM_MAX_TOKENS,
    DEFAULT_VLLM_TEMPERATURE,
    # DeepFace
    CONF_DEEPFACE_URL,
    CONF_DEEPFACE_ENABLED,
    CONF_DEEPFACE_TOLERANCE,
    DEFAULT_DEEPFACE_URL,
    DEFAULT_DEEPFACE_ENABLED,
    DEFAULT_DEEPFACE_TOLERANCE,
    # Cameras - Auto-Discovery
    CONF_SELECTED_CAMERAS,
    DEFAULT_SELECTED_CAMERAS,
    CONF_CAMERA_ALIASES,
    DEFAULT_CAMERA_ALIASES,
    # Video
    CONF_VIDEO_DURATION,
    CONF_VIDEO_WIDTH,
    DEFAULT_VIDEO_DURATION,
    DEFAULT_VIDEO_WIDTH,
    # Snapshot
    CONF_SNAPSHOT_DIR,
    CONF_SNAPSHOT_QUALITY,
    DEFAULT_SNAPSHOT_DIR,
    DEFAULT_SNAPSHOT_QUALITY,
    # Services
    SERVICE_ANALYZE_CAMERA,
    SERVICE_RECORD_CLIP,
    SERVICE_IDENTIFY_FACES,
    # Attributes
    ATTR_CAMERA,
    ATTR_DURATION,
    ATTR_USER_QUERY,
    ATTR_IMAGE_PATH,
    ATTR_IMAGE_BASE64,
)

_LOGGER = logging.getLogger(__name__)

# Bundled blueprints
BLUEPRINTS = [
    {
        "domain": "automation",
        "filename": "camera_alert.yaml",
    },
]


async def async_import_blueprints(hass: HomeAssistant) -> None:
    """Import bundled blueprints to the user's blueprints directory."""
    try:
        integration_dir = Path(__file__).parent
        blueprints_source = integration_dir / "blueprints"
        blueprints_target = Path(hass.config.path("blueprints"))

        for blueprint in BLUEPRINTS:
            domain = blueprint["domain"]
            filename = blueprint["filename"]

            source_file = blueprints_source / domain / filename
            target_dir = blueprints_target / domain / DOMAIN
            target_file = target_dir / filename

            if not source_file.exists():
                _LOGGER.warning("Blueprint not found: %s", source_file)
                continue

            await hass.async_add_executor_job(
                lambda: target_dir.mkdir(parents=True, exist_ok=True)
            )

            should_copy = False
            if not target_file.exists():
                should_copy = True
                _LOGGER.info("Installing blueprint: %s", filename)
            else:
                source_mtime = source_file.stat().st_mtime
                target_mtime = target_file.stat().st_mtime
                if source_mtime > target_mtime:
                    should_copy = True
                    _LOGGER.info("Updating blueprint: %s", filename)

            if should_copy:
                await hass.async_add_executor_job(
                    shutil.copy2, source_file, target_file
                )
                _LOGGER.info("Blueprint installed: %s -> %s", filename, target_file)

    except Exception as e:
        _LOGGER.warning("Failed to import blueprints: %s", e)


# Service schemas
SERVICE_ANALYZE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CAMERA): cv.string,
        vol.Optional(ATTR_DURATION, default=3): vol.All(vol.Coerce(int), vol.Range(min=1, max=10)),
        vol.Optional(ATTR_USER_QUERY, default=""): cv.string,
    }
)

SERVICE_RECORD_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CAMERA): cv.string,
        vol.Optional(ATTR_DURATION, default=3): vol.All(vol.Coerce(int), vol.Range(min=1, max=10)),
    }
)

SERVICE_IDENTIFY_FACES_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_CAMERA): cv.string,
        vol.Optional(ATTR_IMAGE_PATH): cv.string,
        vol.Optional(ATTR_IMAGE_BASE64): cv.string,
    }
)


async def async_setup(hass: HomeAssistant, config: dict[str, Any]) -> bool:
    """Set up the HA Video Vision component."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry to new version."""
    _LOGGER.info("Migrating HA Video Vision config entry from version %s", config_entry.version)

    if config_entry.version < 5:
        new_data = {**config_entry.data}
        new_options = {**config_entry.options}

        if CONF_SELECTED_CAMERAS not in new_options and CONF_SELECTED_CAMERAS not in new_data:
            new_options[CONF_SELECTED_CAMERAS] = []

        # Add DeepFace defaults for migration
        if CONF_DEEPFACE_URL not in new_options and CONF_DEEPFACE_URL not in new_data:
            new_options[CONF_DEEPFACE_ENABLED] = DEFAULT_DEEPFACE_ENABLED
            new_options[CONF_DEEPFACE_URL] = DEFAULT_DEEPFACE_URL

        hass.config_entries.async_update_entry(
            config_entry,
            data=new_data,
            options=new_options,
            version=5,
        )
        _LOGGER.info("Migration to version 5 (DeepFace support) successful")

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up HA Video Vision from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    await async_import_blueprints(hass)

    config = {**entry.data, **entry.options}

    analyzer = VideoAnalyzer(hass, config)
    hass.data[DOMAIN][entry.entry_id] = {
        "config": config,
        "analyzer": analyzer,
    }

    async def handle_analyze_camera(call: ServiceCall) -> dict[str, Any]:
        """Handle analyze_camera service call."""
        camera = call.data[ATTR_CAMERA]
        duration = call.data.get(ATTR_DURATION, 3)
        user_query = call.data.get(ATTR_USER_QUERY, "")
        return await analyzer.analyze_camera(camera, duration, user_query)

    async def handle_record_clip(call: ServiceCall) -> dict[str, Any]:
        """Handle record_clip service call."""
        camera = call.data[ATTR_CAMERA]
        duration = call.data.get(ATTR_DURATION, 3)
        return await analyzer.record_clip(camera, duration)

    async def handle_identify_faces(call: ServiceCall) -> dict[str, Any]:
        """Handle identify_faces service call."""
        camera = call.data.get(ATTR_CAMERA)
        image_path = call.data.get(ATTR_IMAGE_PATH)
        image_base64 = call.data.get(ATTR_IMAGE_BASE64)
        return await analyzer.identify_faces(
            camera=camera,
            image_path=image_path,
            image_base64=image_base64,
        )

    hass.services.async_register(
        DOMAIN,
        SERVICE_ANALYZE_CAMERA,
        handle_analyze_camera,
        schema=SERVICE_ANALYZE_SCHEMA,
        supports_response=True,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_RECORD_CLIP,
        handle_record_clip,
        schema=SERVICE_RECORD_SCHEMA,
        supports_response=True,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_IDENTIFY_FACES,
        handle_identify_faces,
        schema=SERVICE_IDENTIFY_FACES_SCHEMA,
        supports_response=True,
    )

    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    _LOGGER.info(
        "HA Video Vision setup complete - Cameras: %d, DeepFace: %s",
        len(config.get(CONF_SELECTED_CAMERAS, [])),
        "enabled" if config.get(CONF_DEEPFACE_ENABLED) else "disabled"
    )
    return True


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    config = {**entry.data, **entry.options}
    hass.data[DOMAIN][entry.entry_id]["config"] = config
    hass.data[DOMAIN][entry.entry_id]["analyzer"].update_config(config)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    hass.services.async_remove(DOMAIN, SERVICE_ANALYZE_CAMERA)
    hass.services.async_remove(DOMAIN, SERVICE_RECORD_CLIP)
    hass.services.async_remove(DOMAIN, SERVICE_IDENTIFY_FACES)

    hass.data[DOMAIN].pop(entry.entry_id, None)
    return True


class VideoAnalyzer:
    """Class to handle video analysis with auto-discovered cameras and facial recognition."""

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]) -> None:
        """Initialize the analyzer."""
        self.hass = hass
        self._session = async_get_clientsession(hass)
        self.update_config(config)

    def update_config(self, config: dict[str, Any]) -> None:
        """Update configuration."""
        # Provider settings
        self.provider = config.get(CONF_DEFAULT_PROVIDER, config.get(CONF_PROVIDER, DEFAULT_PROVIDER))
        self.provider_configs = config.get(CONF_PROVIDER_CONFIGS, {})

        active_config = self.provider_configs.get(self.provider, {})

        if active_config:
            self.api_key = active_config.get("api_key", "")
            self.vllm_model = active_config.get("model", PROVIDER_DEFAULT_MODELS.get(self.provider, ""))
            self.base_url = active_config.get("base_url", PROVIDER_BASE_URLS.get(self.provider, ""))
        else:
            self.api_key = config.get(CONF_API_KEY, "")
            self.vllm_model = config.get(CONF_VLLM_MODEL, PROVIDER_DEFAULT_MODELS.get(self.provider, DEFAULT_VLLM_MODEL))

            if self.provider == PROVIDER_LOCAL:
                self.base_url = config.get(CONF_VLLM_URL, DEFAULT_VLLM_URL)
            else:
                self.base_url = PROVIDER_BASE_URLS.get(self.provider, DEFAULT_VLLM_URL)

        # AI settings
        self.vllm_max_tokens = config.get(CONF_VLLM_MAX_TOKENS, DEFAULT_VLLM_MAX_TOKENS)
        self.vllm_temperature = config.get(CONF_VLLM_TEMPERATURE, DEFAULT_VLLM_TEMPERATURE)

        # DeepFace settings
        self.deepface_enabled = config.get(CONF_DEEPFACE_ENABLED, DEFAULT_DEEPFACE_ENABLED)
        self.deepface_url = config.get(CONF_DEEPFACE_URL, DEFAULT_DEEPFACE_URL)
        self.deepface_tolerance = config.get(CONF_DEEPFACE_TOLERANCE, DEFAULT_DEEPFACE_TOLERANCE)

        # Auto-discovered cameras
        self.selected_cameras = config.get(CONF_SELECTED_CAMERAS, DEFAULT_SELECTED_CAMERAS)
        self.camera_aliases = config.get(CONF_CAMERA_ALIASES, DEFAULT_CAMERA_ALIASES)

        # Video settings
        self.video_duration = config.get(CONF_VIDEO_DURATION, DEFAULT_VIDEO_DURATION)
        self.video_width = config.get(CONF_VIDEO_WIDTH, DEFAULT_VIDEO_WIDTH)

        # Snapshot settings
        self.snapshot_dir = config.get(CONF_SNAPSHOT_DIR, DEFAULT_SNAPSHOT_DIR)
        self.snapshot_quality = config.get(CONF_SNAPSHOT_QUALITY, DEFAULT_SNAPSHOT_QUALITY)

        _LOGGER.info(
            "HA Video Vision config - Provider: %s, Cameras: %d, DeepFace: %s (%s)",
            self.provider, len(self.selected_cameras),
            "enabled" if self.deepface_enabled else "disabled",
            self.deepface_url if self.deepface_enabled else "N/A"
        )

    def _get_effective_provider(self) -> tuple[str, str, str]:
        """Get the effective provider."""
        return (self.provider, self.vllm_model, self.api_key)

    def _normalize_name(self, name: str) -> str:
        """Normalize a name for comparison."""
        import re
        normalized = name.lower().strip()
        normalized = re.sub(r'[_\-]+', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def _find_camera_entity(self, camera_input: str) -> str | None:
        """Find camera entity ID by alias, name, entity_id, or friendly name."""
        camera_input_norm = self._normalize_name(camera_input)
        camera_input_lower = camera_input.lower().strip()

        # Check voice aliases first
        for alias, entity_id in self.camera_aliases.items():
            alias_norm = self._normalize_name(alias)
            if alias_norm == camera_input_norm:
                return entity_id
            if alias_norm in camera_input_norm:
                return entity_id
            if camera_input_norm in alias_norm:
                return entity_id

        camera_matches = []

        for entity_id in self.selected_cameras:
            state = self.hass.states.get(entity_id)
            if not state:
                continue

            friendly_name = state.attributes.get("friendly_name", "")
            entity_suffix = entity_id.replace("camera.", "")

            camera_matches.append({
                "entity_id": entity_id,
                "friendly_name": friendly_name,
                "friendly_norm": self._normalize_name(friendly_name),
                "entity_suffix": entity_suffix,
                "entity_norm": self._normalize_name(entity_suffix),
            })

        for state in self.hass.states.async_all("camera"):
            entity_id = state.entity_id
            if entity_id in self.selected_cameras:
                continue

            friendly_name = state.attributes.get("friendly_name", "")
            entity_suffix = entity_id.replace("camera.", "")

            camera_matches.append({
                "entity_id": entity_id,
                "friendly_name": friendly_name,
                "friendly_norm": self._normalize_name(friendly_name),
                "entity_suffix": entity_suffix,
                "entity_norm": self._normalize_name(entity_suffix),
            })

        if camera_input_lower.startswith("camera."):
            for cam in camera_matches:
                if cam["entity_id"].lower() == camera_input_lower:
                    return cam["entity_id"]

        for cam in camera_matches:
            if cam["friendly_norm"] == camera_input_norm:
                return cam["entity_id"]

        for cam in camera_matches:
            if cam["entity_norm"] == camera_input_norm:
                return cam["entity_id"]

        for cam in camera_matches:
            if camera_input_norm in cam["friendly_norm"] or cam["friendly_norm"] in camera_input_norm:
                return cam["entity_id"]

        for cam in camera_matches:
            if camera_input_norm in cam["entity_norm"] or cam["entity_norm"] in camera_input_norm:
                return cam["entity_id"]

        input_words = set(camera_input_norm.split())
        for cam in camera_matches:
            friendly_words = set(cam["friendly_norm"].split())
            entity_words = set(cam["entity_norm"].split())

            if input_words & friendly_words:
                return cam["entity_id"]
            if input_words & entity_words:
                return cam["entity_id"]

        return None

    async def _get_camera_snapshot(self, entity_id: str, retries: int = 3, delay: float = 1.0) -> bytes | None:
        """Get camera snapshot with retry logic."""
        last_image = None
        last_error = None

        for attempt in range(retries):
            try:
                if attempt > 0:
                    _LOGGER.debug(
                        "Snapshot retry %d/%d for %s (waiting %.1fs)",
                        attempt + 1, retries, entity_id, delay
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 1.5, 5.0)

                image = await async_get_image(self.hass, entity_id)
                if image and image.content:
                    if last_image and image.content == last_image:
                        _LOGGER.debug(
                            "Snapshot from %s unchanged on attempt %d, retrying...",
                            entity_id, attempt + 1
                        )
                        continue

                    _LOGGER.debug(
                        "Got snapshot from %s on attempt %d (%d bytes)",
                        entity_id, attempt + 1, len(image.content)
                    )
                    return image.content

                last_image = image.content if image else None

            except Exception as e:
                last_error = e
                _LOGGER.debug(
                    "Snapshot attempt %d failed for %s: %s",
                    attempt + 1, entity_id, e
                )

        if last_error:
            _LOGGER.warning(
                "Failed to get fresh snapshot from %s after %d attempts: %s",
                entity_id, retries, last_error
            )
        elif last_image:
            _LOGGER.debug(
                "Returning possibly stale snapshot from %s (unchanged across retries)",
                entity_id
            )
            return last_image
        else:
            _LOGGER.warning(
                "No snapshot available from %s after %d attempts",
                entity_id, retries
            )

        return last_image

    async def _get_stream_url(self, entity_id: str) -> str | None:
        """Get RTSP/stream URL from camera entity."""
        try:
            stream_url = await async_get_stream_source(self.hass, entity_id)
            return stream_url
        except Exception as e:
            _LOGGER.debug("Could not get stream URL for %s: %s", entity_id, e)
            return None

    def _build_ffmpeg_cmd(self, stream_url: str, duration: int, output_path: str) -> list[str]:
        """Build ffmpeg command based on stream type."""
        cmd = ["ffmpeg", "-y"]

        if stream_url.startswith("rtsp://"):
            cmd.extend(["-rtsp_transport", "tcp"])

        cmd.extend(["-i", stream_url])
        cmd.extend([
            "-t", str(duration),
            "-vf", f"scale={self.video_width}:-2",
            "-r", "10",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-an",
            output_path
        ])

        return cmd

    def _build_ffmpeg_frame_cmd(self, stream_url: str, output_path: str) -> list[str]:
        """Build ffmpeg command to extract a single frame."""
        cmd = ["ffmpeg", "-y"]

        if stream_url.startswith("rtsp://"):
            cmd.extend(["-rtsp_transport", "tcp"])

        cmd.extend([
            "-i", stream_url,
            "-frames:v", "1",
            "-vf", f"scale={self.video_width}:-2",
            "-q:v", "2",
            output_path
        ])

        return cmd

    async def _call_deepface(self, image_bytes: bytes) -> dict[str, Any]:
        """Call the DeepFace facial recognition server."""
        if not self.deepface_enabled:
            return {"success": False, "error": "DeepFace not enabled"}

        try:
            url = f"{self.deepface_url.rstrip('/')}/identify"
            image_b64 = base64.b64encode(image_bytes).decode()

            payload = {
                "image_base64": image_b64,
                "tolerance": self.deepface_tolerance,
            }

            _LOGGER.debug("Calling DeepFace at %s", url)

            async with asyncio.timeout(30):
                async with self._session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        _LOGGER.debug("DeepFace response: %s", result)
                        return result
                    else:
                        error = await response.text()
                        _LOGGER.error("DeepFace error %d: %s", response.status, error[:500])
                        return {
                            "success": False,
                            "error": f"DeepFace returned {response.status}: {error[:100]}"
                        }

        except asyncio.TimeoutError:
            _LOGGER.error("DeepFace request timed out")
            return {"success": False, "error": "DeepFace request timed out"}
        except aiohttp.ClientError as e:
            _LOGGER.error("DeepFace connection error: %s", e)
            return {"success": False, "error": f"Connection error: {str(e)}"}
        except Exception as e:
            _LOGGER.error("DeepFace error: %s", e)
            return {"success": False, "error": str(e)}

    async def identify_faces(
        self,
        camera: str | None = None,
        image_path: str | None = None,
        image_base64: str | None = None,
    ) -> dict[str, Any]:
        """Identify faces in an image from camera, file path, or base64 string."""
        if not self.deepface_enabled:
            return {
                "success": False,
                "error": "DeepFace facial recognition is not enabled. Configure it in the integration settings."
            }

        image_bytes = None
        source = None

        # Priority: image_base64 > image_path > camera
        if image_base64:
            try:
                image_bytes = base64.b64decode(image_base64)
                source = "base64"
            except Exception as e:
                return {"success": False, "error": f"Invalid base64 image: {e}"}

        elif image_path:
            try:
                async with aiofiles.open(image_path, 'rb') as f:
                    image_bytes = await f.read()
                source = image_path
            except Exception as e:
                return {"success": False, "error": f"Could not read image file: {e}"}

        elif camera:
            entity_id = self._find_camera_entity(camera)
            if not entity_id:
                available = ", ".join(self.selected_cameras) if self.selected_cameras else "None configured"
                return {
                    "success": False,
                    "error": f"Camera '{camera}' not found. Available: {available}"
                }

            image_bytes = await self._get_camera_snapshot(entity_id, retries=3, delay=1.0)
            if not image_bytes:
                return {"success": False, "error": f"Could not get snapshot from camera {entity_id}"}
            source = entity_id

        else:
            return {
                "success": False,
                "error": "No image source provided. Specify camera, image_path, or image_base64."
            }

        # Call DeepFace
        result = await self._call_deepface(image_bytes)
        result["source"] = source

        return result

    async def record_clip(self, camera_input: str, duration: int = None) -> dict[str, Any]:
        """Record a video clip from camera."""
        duration = duration or self.video_duration

        entity_id = self._find_camera_entity(camera_input)
        if not entity_id:
            available = ", ".join(self.selected_cameras) if self.selected_cameras else "None configured"
            return {
                "success": False,
                "error": f"Camera '{camera_input}' not found. Available: {available}"
            }

        stream_url = await self._get_stream_url(entity_id)
        if not stream_url:
            return {
                "success": False,
                "error": f"Could not get stream URL for {entity_id}. Camera may not support streaming."
            }

        os.makedirs(self.snapshot_dir, exist_ok=True)
        video_path = None

        friendly_name = self.hass.states.get(entity_id).attributes.get("friendly_name", entity_id)
        safe_name = entity_id.replace("camera.", "").replace(".", "_")

        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=self.snapshot_dir) as vf:
                video_path = vf.name

            cmd = self._build_ffmpeg_cmd(stream_url, duration, video_path)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=duration + 15)

            if proc.returncode != 0:
                _LOGGER.error("FFmpeg error: %s", stderr.decode() if stderr else "Unknown")
                return {"success": False, "error": "Failed to record video"}

            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                return {"success": False, "error": "Video file empty"}

            final_path = os.path.join(self.snapshot_dir, f"{safe_name}_clip.mp4")
            os.rename(video_path, final_path)

            return {
                "success": True,
                "camera": entity_id,
                "friendly_name": friendly_name,
                "video_path": final_path,
                "duration": duration,
            }

        except asyncio.TimeoutError:
            return {"success": False, "error": "Recording timed out"}
        except Exception as e:
            _LOGGER.error("Error recording clip: %s", e)
            return {"success": False, "error": str(e)}
        finally:
            if video_path and os.path.exists(video_path) and "clip.mp4" not in video_path:
                try:
                    os.remove(video_path)
                except Exception:
                    pass

    async def _record_video_and_frames(self, entity_id: str, duration: int) -> tuple[bytes | None, bytes | None]:
        """Record video and extract frames from camera entity."""
        stream_url = await self._get_stream_url(entity_id)

        video_bytes = None
        frame_bytes = None

        if not stream_url:
            _LOGGER.warning(
                "No stream URL for %s - using snapshot mode (cloud camera).",
                entity_id
            )
            frame_bytes = await self._get_camera_snapshot(entity_id, retries=4, delay=1.5)
            return video_bytes, frame_bytes

        video_path = None
        frame_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as vf:
                video_path = vf.name
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as ff:
                frame_path = ff.name

            video_cmd = self._build_ffmpeg_cmd(stream_url, duration, video_path)
            frame_cmd = self._build_ffmpeg_frame_cmd(stream_url, frame_path)

            video_proc = await asyncio.create_subprocess_exec(
                *video_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            frame_proc = await asyncio.create_subprocess_exec(
                *frame_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )

            await asyncio.wait_for(video_proc.communicate(), timeout=duration + 15)
            await asyncio.wait_for(frame_proc.wait(), timeout=10)

            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                async with aiofiles.open(video_path, 'rb') as f:
                    video_bytes = await f.read()

            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                async with aiofiles.open(frame_path, 'rb') as f:
                    frame_bytes = await f.read()

            return video_bytes, frame_bytes

        except Exception as e:
            _LOGGER.error("Error recording video from %s: %s", entity_id, e)
            fallback_frame = await self._get_camera_snapshot(entity_id)
            return None, fallback_frame
        finally:
            for path in [video_path, frame_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

    async def analyze_camera(
        self, camera_input: str, duration: int = None, user_query: str = ""
    ) -> dict[str, Any]:
        """Analyze camera using video, AI vision, and optional facial recognition."""
        duration = duration or self.video_duration

        _LOGGER.info(
            "Camera analysis requested - Input: '%s', Provider: %s, DeepFace: %s",
            camera_input, self.provider, "enabled" if self.deepface_enabled else "disabled"
        )

        entity_id = self._find_camera_entity(camera_input)
        if not entity_id:
            available = ", ".join(self.selected_cameras) if self.selected_cameras else "None configured"
            return {
                "success": False,
                "error": f"Camera '{camera_input}' not found. Available: {available}"
            }

        state = self.hass.states.get(entity_id)
        friendly_name = state.attributes.get("friendly_name", entity_id) if state else entity_id
        safe_name = entity_id.replace("camera.", "").replace(".", "_")

        # Record video and get frames
        video_bytes, frame_bytes = await self._record_video_and_frames(entity_id, duration)

        # Run AI analysis and facial recognition in parallel
        ai_task = self._analyze_with_provider(video_bytes, frame_bytes, user_query or self._get_default_prompt())

        # Only run facial recognition if enabled and we have a frame
        face_task = None
        if self.deepface_enabled and frame_bytes:
            face_task = self._call_deepface(frame_bytes)

        # Await results
        if face_task:
            description_result, face_result = await asyncio.gather(ai_task, face_task)
            description, provider_used = description_result
        else:
            description, provider_used = await ai_task
            face_result = None

        _LOGGER.info(
            "Analysis complete for %s - Provider: %s, Faces: %s",
            friendly_name, provider_used,
            face_result.get("faces_detected", 0) if face_result else "N/A"
        )

        # Save snapshot
        snapshot_path = None
        if frame_bytes:
            os.makedirs(self.snapshot_dir, exist_ok=True)
            snapshot_path = os.path.join(self.snapshot_dir, f"{safe_name}_latest.jpg")
            try:
                async with aiofiles.open(snapshot_path, 'wb') as f:
                    await f.write(frame_bytes)
            except Exception as e:
                _LOGGER.error("Failed to save snapshot: %s", e)

        # Build identified_people list from DeepFace results
        identified_people = []
        if face_result and face_result.get("success"):
            for person in face_result.get("people", []):
                if person.get("status") == "identified" and person.get("name") != "Unknown":
                    identified_people.append({
                        "name": person["name"],
                        "confidence": person.get("confidence", 0),
                    })

        # Check for person-related words in AI description
        description_text = description or ""
        person_detected = any(
            word in description_text.lower()
            for word in ["person", "people", "someone", "man", "woman", "child"]
        ) or len(identified_people) > 0

        return {
            "success": True,
            "camera": entity_id,
            "friendly_name": friendly_name,
            "description": description,
            "person_detected": person_detected,
            "identified_people": identified_people,
            "faces_detected": face_result.get("faces_detected", 0) if face_result else 0,
            "face_summary": face_result.get("summary", "") if face_result else "",
            "snapshot_path": snapshot_path,
            "snapshot_url": f"/media/local/ha_video_vision/{safe_name}_latest.jpg" if snapshot_path else None,
            "provider_used": provider_used,
            "default_provider": self.provider,
        }

    def _get_default_prompt(self) -> str:
        """Get the default analysis prompt."""
        return (
            "Describe what you see in this camera feed. "
            "Note any activity, vehicles, or notable events. "
            "Be concise (2-3 sentences). Say 'no activity' if nothing notable is happening."
        )

    async def _analyze_with_provider(
        self, video_bytes: bytes | None, frame_bytes: bytes | None, prompt: str
    ) -> tuple[str, str]:
        """Send video/image to the configured AI provider."""
        effective_provider, effective_model, effective_api_key = self._get_effective_provider()

        media_type = "video" if video_bytes else ("image" if frame_bytes else "none")
        _LOGGER.debug(
            "Sending %s to AI - Provider: %s, Model: %s",
            media_type, effective_provider, effective_model
        )

        if effective_provider == PROVIDER_GOOGLE:
            result = await self._analyze_google(video_bytes, frame_bytes, prompt, effective_model, effective_api_key)
        elif effective_provider == PROVIDER_OPENROUTER:
            result = await self._analyze_openrouter(video_bytes, frame_bytes, prompt, effective_model, effective_api_key)
        elif effective_provider == PROVIDER_LOCAL:
            result = await self._analyze_local(video_bytes, frame_bytes, prompt)
        else:
            result = "Unknown provider configured"

        return result, effective_provider

    async def _analyze_google(
        self, video_bytes: bytes | None, frame_bytes: bytes | None, prompt: str,
        model: str = None, api_key: str = None
    ) -> str:
        """Analyze using Google Gemini."""
        if not video_bytes:
            return "No video available for analysis."

        model = model or self.vllm_model
        api_key = api_key or self.api_key

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

            parts = [{"text": prompt}]
            video_b64 = base64.b64encode(video_bytes).decode()
            parts.insert(0, {
                "inline_data": {
                    "mime_type": "video/mp4",
                    "data": video_b64
                }
            })

            system_instruction = (
                "You are a security camera analyst. Describe what you see accurately and concisely."
            )

            payload = {
                "contents": [{"parts": parts}],
                "systemInstruction": {"parts": [{"text": system_instruction}]},
                "generationConfig": {
                    "temperature": self.vllm_temperature,
                    "maxOutputTokens": self.vllm_max_tokens,
                }
            }

            async with asyncio.timeout(60):
                async with self._session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        candidates = result.get("candidates", [])
                        if not candidates:
                            prompt_feedback = result.get("promptFeedback", {})
                            block_reason = prompt_feedback.get("blockReason")
                            if block_reason:
                                return f"Content blocked: {block_reason}"
                            return "No response from Gemini"

                        candidate = candidates[0]
                        finish_reason = candidate.get("finishReason", "")
                        if finish_reason == "SAFETY":
                            return "Content blocked by safety filters"

                        content = candidate.get("content", {})
                        parts = content.get("parts", [])
                        text_parts = [p.get("text", "") for p in parts if "text" in p]
                        return "".join(text_parts) if text_parts else "No text in response"
                    else:
                        error = await response.text()
                        _LOGGER.error("Gemini error: %s", error[:500])
                        return f"Analysis failed: {response.status}"

        except Exception as e:
            _LOGGER.error("Gemini analysis error: %s", e)
            return f"Analysis error: {str(e)}"

    async def _analyze_openrouter(
        self, video_bytes: bytes | None, frame_bytes: bytes | None, prompt: str,
        model: str = None, api_key: str = None
    ) -> str:
        """Analyze using OpenRouter."""
        if not video_bytes:
            return "No video available for analysis."

        model = model or self.vllm_model
        api_key = api_key or self.api_key

        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            content = []
            video_b64 = base64.b64encode(video_bytes).decode()
            content.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}
            })
            content.append({"type": "text", "text": prompt})

            system_message = (
                "You are a security camera analyst. Describe what you see accurately and concisely."
            )

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": content}
                ],
                "max_tokens": self.vllm_max_tokens,
                "temperature": self.vllm_temperature,
                "provider": {"only": ["Google Vertex"]}
            }

            async with asyncio.timeout(60):
                async with self._session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        choices = result.get("choices", [])
                        if not choices:
                            return "No response from AI"
                        message = choices[0].get("message", {})
                        content = message.get("content", "")
                        return content if content else "No description available"
                    else:
                        error = await response.text()
                        _LOGGER.error("OpenRouter error: %s", error[:500])
                        return f"Analysis failed: {response.status}"

        except Exception as e:
            _LOGGER.error("OpenRouter error: %s", e)
            return f"Analysis error: {str(e)}"

    async def _analyze_local(self, video_bytes: bytes | None, frame_bytes: bytes | None, prompt: str) -> str:
        """Analyze using local vLLM endpoint."""
        if not video_bytes and not frame_bytes:
            return "No video or image available for analysis"

        try:
            url = f"{self.base_url}/chat/completions"

            content = []

            if video_bytes:
                video_b64 = base64.b64encode(video_bytes).decode()
                content.append({
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}
                })
            elif frame_bytes:
                image_b64 = base64.b64encode(frame_bytes).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                })

            content.append({"type": "text", "text": prompt})

            system_message = (
                "You are a security camera analyst. Describe what you see accurately and concisely."
            )

            payload = {
                "model": self.vllm_model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": content}
                ],
                "max_tokens": self.vllm_max_tokens,
                "temperature": self.vllm_temperature,
            }

            async with asyncio.timeout(120):
                async with self._session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        choices = result.get("choices", [])
                        if not choices:
                            return "No response from AI"
                        message = choices[0].get("message", {})
                        content = message.get("content", "")
                        return content if content else "No description available"
                    else:
                        error = await response.text()
                        _LOGGER.error("Local vLLM error: %s", error[:500])
                        return f"Analysis failed: {response.status}"

        except Exception as e:
            _LOGGER.error("Local vLLM error: %s", e)
            return f"Analysis error: {str(e)}"
