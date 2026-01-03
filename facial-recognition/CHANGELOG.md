# Changelog

## [1.0.4] - 2026-01-03

### Fixed
- Fixed faces directory path mismatch: now correctly uses `/share/faces` to match documentation
- Users following the DOCS.md instructions will now have their faces properly detected

## [1.0.0] - 2025-12-31

### Added
- Initial release
- DeepFace-based facial recognition
- Support for multiple face detection backends
- Configurable distance threshold and confidence
- API endpoints: /status, /identify, /reload
- Multi-architecture support (amd64, aarch64, armv7)
