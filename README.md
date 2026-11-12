# Tank2 AI Video Studio

An Android-first, offline-first research project for on-device **Image-to-Video** generation on the Unihertz Tank 2 (MediaTek MT6789 / Helio G99, Mali-G57 MC2, Android 14).

## Project scope
- **Image-to-Video only:** animate a selected photo with a genuine generative video model.
- **Strict local mode:** inference must run on the phone. No cloud inference, remote GPU, or upload of user photos/videos.
- **Transparent capability reporting:** slideshow, pan/zoom, interpolation, or template animation must never be labelled diffusion video generation.

Text-to-Video is explicitly out of scope.

## Current status
The Android shell builds successfully through GitHub Actions. It is not yet a working generator: no verified Android-compatible model package or real inference pipeline is integrated. A successful APK build proves only that the Android shell compiles.

## MobileI2V investigation (2026-11-12)
The official [hustvl/MobileI2V repository](https://github.com/hustvl/MobileI2V) publishes PyTorch inference instructions using `hybrid_371.pth`, a Python environment, and a CUDA device. The official repository does not provide a ready-to-use Android ONNX package. The paper describes a 270M-parameter mobile-focused model and fast mobile results, but that does not by itself provide a compatible runtime package for this specific Helio G99 device.

A third-party Android port, [himadriHK/MobileI2VConsole](https://github.com/himadriHK/MobileI2VConsole), was audited as a possible starting point. Its own `.codebase/CONCERNS.md` states that all four core inference stages return zero-filled placeholder arrays rather than executing ONNX sessions. Its model download/manifest assumptions and pipeline therefore cannot be treated as a working implementation or copied as-is. This project will not inherit that fake generation path.

## Device constraints
The target device reports Android 14, arm64-v8a, MT6789, about 11 GiB physical RAM, and Mali GPU libraries. Termux Vulkan currently exposes Mesa llvmpipe rather than Mali, and OpenCL reports zero platforms. Android's own graphics driver exists, but usable ML acceleration has not been demonstrated.

## Build
Push to `main` or run the **Android debug build** workflow manually. The workflow builds `app-debug.apk` and uploads it as a GitHub Actions artifact. This is a development APK, not a signed release.

## Model integration acceptance criteria
1. Identify the exact upstream weights, license, file sizes, and SHA-256 hashes.
2. Produce real ONNX/ORT model files from the upstream architecture; keep model binaries out of Git history.
3. Validate every exported graph with ONNX checker and runtime inference using real test tensors.
4. Verify input/output names, shapes, data types, tokenizer, scheduler, and numerical agreement with the reference implementation.
5. Implement real ONNX sessions in Android; never use zero tensors or dummy frames as generated output.
6. Load/unload stages deliberately and measure peak resident memory, latency, thermals, and battery use on the Tank 2.
7. Confirm NNAPI acceleration on this exact device; otherwise report CPU execution and its measured performance.
8. Keep inference local and make network access unnecessary after model installation.

## Planned stages
1. Android shell + reproducible CI build. (Complete)
2. Device-side runtime and accelerator capability probe.
3. Reproducible model conversion and validation.
4. Real Android ONNX inference integration.
5. On-device end-to-end testing and performance/memory qualification.

Do not promise real-time generation or label a build as ready until end-to-end local inference has produced and saved a valid MP4 on the target phone.

## RIFE x2 frame interpolation (2026-11-13)
A reusable Android-side RIFE v4.9 ONNX interpolator has been added at `RifeX2Interpolator.kt`.
- Runtime: Microsoft ONNX Runtime Android 1.28.0, CPU execution with two intra-op threads as the conservative initial setting.
- Model: `edgetools/rife` / `rife49.onnx`, approximately 21.5 MB, downloaded on first use into app-private storage (not bundled in the APK).
- Integrity: SHA-256 is checked before the model is accepted: `76e4cef9ab42fa7dd4e8f6e4aba47462051e3faa969e4bca6479784fbab0ac6f`.
- Inputs: `img0`, `img1`, and midpoint `timestep=0.5`; output: `output`. RGB planar float32 tensors are edge-padded to dimensions divisible by 32 and cropped back to the original frame size.
- Output: one newly inferred midpoint between every adjacent pair. A 17-frame MobileI2V clip becomes 33 frames, keeping the same duration while doubling the frame rate from 8 to 16 FPS.
- Model card states MIT licensing and attributes weights to Practical-RIFE v4.9. See https://huggingface.co/edgetools/rife and retain its attribution when redistributing.

This is an integration component, not yet an end-to-end feature: the current screen's Generate button is still a placeholder, and no MobileI2V-to-RIFE-to-MP4 flow is wired. Android CI compilation and a real-device inference/performance test are required before claiming the feature works on the Tank 2. CPU speed and peak memory at 720p are still unknown.
