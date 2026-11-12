#!/usr/bin/env python3
"""Validate the exact RIFE ONNX model used by the Android interpolator."""
import hashlib
import os
import urllib.request

import numpy as np
import onnxruntime as ort

URL = "https://huggingface.co/edgetools/rife/resolve/main/rife49.onnx"
SHA256 = "76e4cef9ab42fa7dd4e8f6e4aba47462051e3faa969e4bca6479784fbab0ac6f"
PATH = os.path.join("build", "rife49.onnx")


def main():
    os.makedirs("build", exist_ok=True)
    if not os.path.isfile(PATH):
        print("Downloading RIFE v4.9 ONNX...")
        urllib.request.urlretrieve(URL, PATH)
    digest = hashlib.sha256(open(PATH, "rb").read()).hexdigest()
    assert digest == SHA256, f"SHA-256 mismatch: {digest}"
    print(f"MODEL_SHA256=PASS ({digest})")

    session = ort.InferenceSession(PATH, providers=["CPUExecutionProvider"])
    inputs = {item.name: item for item in session.get_inputs()}
    outputs = {item.name: item for item in session.get_outputs()}
    assert set(inputs) == {"img0", "img1", "timestep"}, set(inputs)
    assert "output" in outputs, set(outputs)
    for name in ("img0", "img1"):
        assert inputs[name].type == "tensor(float)", (name, inputs[name].type)
        assert len(inputs[name].shape) == 4, (name, inputs[name].shape)
    assert inputs["timestep"].type == "tensor(float)"
    print("MODEL_IO_METADATA=PASS")
    print("INPUTS=" + repr({k: v.shape for k, v in inputs.items()}))
    print("OUTPUTS=" + repr({k: v.shape for k, v in outputs.items()}))

    # Validate the actual 720p target used by MobileI2V. RIFE requires H/W
    # divisible by 32; 720x1280 already satisfies this.
    height, width = 720, 1280
    rng = np.random.default_rng(20261113)
    frame0 = rng.random((1, 3, height, width), dtype=np.float32)
    frame1 = np.roll(frame0, 3, axis=3).copy()
    timestep = np.array([0.5], dtype=np.float32)
    result = session.run(["output"], {
        "img0": frame0,
        "img1": frame1,
        "timestep": timestep,
    })[0]
    assert result.shape == (1, 3, height, width), result.shape
    assert np.isfinite(result).all(), "Output contains NaN/Inf"
    assert float(result.min()) >= -0.05 and float(result.max()) <= 1.05, (
        float(result.min()), float(result.max())
    )
    print(f"RIFE_720P_CPU_INFERENCE=PASS shape={result.shape}")
    print(f"OUTPUT_RANGE={float(result.min()):.6f}..{float(result.max()):.6f}")
    print("RIFE_ANDROID_CONTRACT=PASS")


if __name__ == "__main__":
    main()
