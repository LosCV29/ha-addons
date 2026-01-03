# HA Video Vision Add-ons

Home Assistant add-ons and integration for [HA Video Vision](https://github.com/LosCV29/ha-video-vision).

## Contents

This repository contains:

1. **Facial Recognition Add-on** - DeepFace-based facial recognition server
2. **HA Video Vision Integration** - Updated integration with DeepFace support (in `custom_components/ha_video_vision/`)

## Add-ons

### Facial Recognition

[![Open your Home Assistant instance and show the add add-on repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2FLosCV29%2Fha-addons)

DeepFace-based facial recognition server that runs locally on your Home Assistant machine.

**Features:**
- One-click installation
- Runs on localhost:8100 (no network config needed)
- Works with HA Video Vision integration
- All processing done locally (privacy-first)
- Supports multiple face detection backends

## Installation

1. Click the button above, or manually add this repository URL to your Home Assistant Add-on Store:
   ```
   https://github.com/LosCV29/ha-addons
   ```

2. Find "Facial Recognition" in the add-on store and click Install

3. Add photos to `/share/faces/PersonName/` folders

4. Start the add-on

5. Configure HA Video Vision to use `http://localhost:8100`

## Requirements

- Home Assistant OS or Supervised installation
- At least 1GB free RAM
- amd64, aarch64, or armv7 architecture

## Integration with DeepFace

The HA Video Vision integration (in `custom_components/ha_video_vision/`) now supports facial recognition:

### New Features

- **identify_faces service**: Identify faces from camera snapshot, file path, or base64 image
- **Automatic face detection**: When enabled, `analyze_camera` automatically identifies people
- **Response includes `identified_people`**: List of recognized people with confidence scores

### Configuration

1. Go to Settings > Devices & Services > HA Video Vision
2. Click Configure > Configure Facial Recognition
3. Enable facial recognition and enter the DeepFace URL (e.g., `http://localhost:8100`)
4. The integration will test the connection before saving

### Service Example

```yaml
service: ha_video_vision.identify_faces
data:
  camera: front_door
```

Response:
```json
{
  "success": true,
  "faces_detected": 2,
  "people": [
    {"name": "John", "confidence": 92.5, "status": "identified"},
    {"name": "Unknown", "confidence": 45.2, "status": "unknown"}
  ],
  "summary": "Detected: John and 1 unknown"
}
```

## Support

- [HA Video Vision Documentation](https://github.com/LosCV29/ha-video-vision)
- [Issues](https://github.com/LosCV29/ha-addons/issues)
