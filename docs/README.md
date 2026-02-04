# Technical Documentation

This directory contains technical details, architectural decisions, and specifications for "Timeline Plana".

## Contents

- [Architecture](./ARCHITECTURE.md) - Directory structure and core design principles.
- [Specifications](./specs/) - Detailed specs for specific features.
  - [Combat Intel](./specs/combat_intel.md) - Battle state detection (Active, Paused, Slow).
  - [Countdown Monitor](./specs/countdown_monitor.md) - Timer OCR and skew correction pipeline.
  - [Shittim Link](./specs/shittim_link.md) - Backend-to-Frontend communication protocol (30Hz).
  - [Target Acquisition](./specs/target_acquisition.md) - Game process discovery logic.
  - [Visual Intercept](./specs/visual_intercept.md) - High-performance screen capture and OCR pipeline.
