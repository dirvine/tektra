# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- In-app progress overlay for model downloads and initialization
- Real-time download progress tracking with speed and time remaining
- Progressive UI improvements with modern theme updates
- Multiple progress monitoring utilities for Hugging Face downloads
- Non-blocking model loading with background thread execution
- Reusable progress overlay component for future model switching

### Changed
- Model initialization now shows granular progress instead of hanging
- Progress feedback moved from separate window to in-app overlay
- UI theme updated with improved colors and modern styling
- Chat panel styling enhanced with message bubble design
- Replaced emoji status indicators with pill-style components

### Fixed
- App hanging during model initialization without progress feedback
- Multiple progress windows opening simultaneously
- Incorrect download speed calculations (showing unrealistic speeds)
- Toga style compatibility issues with Pack properties
- Progress tracking stuck at low percentages during downloads

### Technical
- Implemented SimpleProgressIndicator for better UX during downloads
- Added ProgressOverlay component for in-app progress display
- Enhanced SimpleLLM with concurrent.futures for non-blocking loading
- Created comprehensive progress monitoring system for transformers
- Fixed various Toga Pack style property validation errors

## Previous Releases

### [Previous] - 2024-07-14
- refactor: major repository cleanup and dual licensing implementation
- docs(readme): transform README with enterprise positioning and P2P/MPC roadmap
- feat(system): complete Phase 6 integration testing and production readiness
- feat(agents): integrate SmolAgents with Qwen backend for real AI agent execution
- fix: address critical production readiness issues