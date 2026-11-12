# Tank2 AI Video Studio

An Android-first, offline-first research project for on-device **Image-to-Video** generation on the Unihertz Tank 2 (MediaTek MT6789 / Helio G99, Mali-G57 MC2, Android 14).

## Project scope
- **Image-to-Video only:** animate a selected photo with a genuine generative video model.
- **Strict local mode:** inference must run on the phone. No cloud inference, remote GPU, or upload of user photos/videos.
- **Transparent capability reporting:** slideshow, pan/zoom, interpolation, or template animation must never be labelled diffusion video generation.

Text-to-Video is explicitly out of scope.

## Current status
This repository contains an Android app shell and a GitHub Actions debug-build workflow. It does **not** yet contain an integrated video model, ONNX weights, or a working generation pipeline. A successful APK build proves only that the Android shell compiles.

## Device constraints
The target device reports Android 14, arm64-v8a, MT6789, about 11 GiB physical RAM, and Mali GPU libraries. Termux Vulkan currently exposes Mesa llvmpipe rather than Mali, and OpenCL reports zero platforms. Android's own graphics driver exists, but usable ML acceleration has not been demonstrated.

## Build
Push to `main` or run the **Android debug build** workflow manually. The workflow builds `app-debug.apk` and uploads it as a GitHub Actions artifact. This is a development APK, not a signed release.

## Model integration rules
1. Do not download model weights automatically without documenting the source, license, size, and checksum.
2. Keep model weights out of Git history; use a separate download/cache or release asset strategy.
3. Validate model architecture, input/output shapes, quantization, runtime operator support, and peak RAM before claiming device compatibility.
4. Keep all inference local and make network access unnecessary after model installation.
5. Do not promise real-time generation. Benchmark actual latency, memory use, thermals, and battery drain on the target phone.

## Planned stages
1. Android shell + reproducible CI build.
2. Device-side runtime capability probe.
3. Select and validate a small, legally distributable Image-to-Video model and runtime.
4. Implement and benchmark real local Image-to-Video inference.
