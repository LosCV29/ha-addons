# Changelog

## [1.0.6] - 2026-01-03

### Fixed
- Changed `homeassistant_config` mapping from read-only to read-write so addon can create faces directory
- Fixed README.md to show correct faces directory path (`/homeassistant/camera_faces/` not `/share/faces/`)

## [1.0.5] - 2026-01-03

### Fixed
- Accept both `image` and `image_base64` field names in /identify endpoint for client compatibility

## [1.0.4] - 2026-01-03

### Fixed
- Fixed faces directory path: now uses `/homeassistant/camera_faces` to match user's actual face storage location
- Added `homeassistant_config` mapping to access Home Assistant config directory
- Updated all documentation to reflect correct paths

## [1.0.0] - 2025-12-31

### Added
- Initial release
- DeepFace-based facial recognition
- Support for multiple face detection backends
- Configurable distance threshold and confidence
- API endpoints: /status, /identify, /reload
- Multi-architecture support (amd64, aarch64, armv7)
