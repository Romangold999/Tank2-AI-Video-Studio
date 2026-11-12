# Tank2 AI Video Studio

An Android-first, offline-first research project for on-device **Image-to-Video** generation on the Unihertz Tank 2 (MediaTek MT6789 / Helio G99, Mali-G57 MC2, Android 14).

## Project scope
- **Image-to-Video only:** animate a selected photo with a genuine generative video model.
- **Strict local mode:** inference must run on the phone. No cloud inference, remote GPU, or upload of user photos/videos.
- **Transparent capability reporting:** slideshow, pan/zoom, interpolation, or template animation must never be labelled diffusion video generation.

Text-to-Video is explicitly out of scope.

## Current status (2026-11-13)
The Android app now has a wired local pipeline in code:
1. LTX VAE image encoder (ONNX) -> MobileI2V latent guide.
2. MobileI2V MobileDiT denoising with the 28-step shifted Flow Euler schedule.
3. Turbo-VAED LTX decoder -> 17 frames at 1280x720.
4. RIFE v4.9 ONNX midpoint interpolation -> 16 FPS.
5. Android MediaCodec H.264 MP4 export.

The corrected decoder implementation has compiled successfully in GitHub Actions run [43322366402](https://github.com/Romangold999/Tank2-AI-Video-Studio/actions/runs/43322366402). The workflow uploaded a debug APK artifact. ONNX components and a synthetic composition path have passed CPU-side numerical/probe checks. This does NOT yet prove real-photo generation on Android or performance on the Unihertz TANK 2. The three core ONNX model assets are published in the `tank2-models-v0.1.0` GitHub Release. On the first Generate action, the app downloads them into private app storage and verifies SHA-256 and file sizes; RIFE is downloaded separately on first interpolation. LTX VAE redistribution also requires shipping its Open Weights License and enforcing its use restrictions. Do not call this a finished app until a real image has been processed on the phone and a valid MP4 has been saved.

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

## Remaining acceptance work
1. Complete model-license notices and an in-app enforceable acceptance flow for the LTX VAE license.
2. Publish the three ONNX assets and manifest only after license compliance is implemented.
3. Test model download, real image inference, RIFE x2, and MP4 export on the Unihertz TANK 2.
4. Measure CPU latency, peak RAM, thermal behavior, and battery use; report actual device results.

Do not promise real-time generation or label a build as ready until end-to-end local inference has produced and saved a valid MP4 on the target phone.

## RIFE x2 frame interpolation (2026-11-13)
The Android-side RIFE v4.9 ONNX interpolator is now connected to the generation/export path in `RifeX2Interpolator.kt`.
- Runtime: Microsoft ONNX Runtime Android 1.28.0, CPU execution with two intra-op threads as the conservative initial setting.
- Model: `edgetools/rife` / `rife49.onnx`, approximately 21.5 MB, downloaded on first use into app-private storage (not bundled in the APK).
- Integrity: SHA-256 is checked before the model is accepted: `76e4cef9ab42fa7dd4e8f6e4aba47462051e3faa969e4bca6479784fbab0ac6f`.
- Inputs: `img0`, `img1`, and midpoint `timestep=0.5`; output: `output`. RGB planar float32 tensors are edge-padded to dimensions divisible by 32 and cropped back to the original frame size.
- Output: one newly inferred midpoint between every adjacent pair. A 17-frame MobileI2V clip becomes 34 output frames: 33 sampled/interpolated positions plus a one-frame hold on the final source frame. At 16 FPS this preserves the original 2.125-second duration exactly while doubling the frame rate from 8 to 16 FPS.
- Model card states MIT licensing and attributes weights to Practical-RIFE v4.9. See https://huggingface.co/edgetools/rife and retain its attribution when redistributing.

The RIFE component is wired into the app's Generate pipeline, but end-to-end inference has not yet been proven on the target phone. Android CI success confirms compilation only; CPU speed, peak memory, and the final MP4 must still be checked on the Tank 2.
