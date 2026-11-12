#!/usr/bin/env python3
"""Export the open-weights LTX Video VAE image encoder with the Dynamo ONNX exporter.

This is an export compatibility probe, not a claim that the full video pipeline
or Android inference is ready.
"""
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.onnx._internal.exporter import _core, _registration
from diffusers import AutoencoderKLLTXVideo


def main():
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKLLTXVideo.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="vae",
        torch_dtype=torch.float32,
    ).cpu().eval()
    print("TORCH_VERSION=%s" % torch.__version__)
    print("ONNX_VERSION=%s" % onnx.__version__)
    print("LTX_VAE_REPOSITORY=Lightricks/LTX-Video")
    print("LTX_VAE_SCALING_FACTOR=%s" % vae.config.scaling_factor)

    torch.manual_seed(20261112)
    image = torch.rand(1, 3, 1, 256, 256, dtype=torch.float32) * 2.0 - 1.0
    with torch.inference_mode():
        reference = vae.encoder(image).cpu().numpy()

    if reference.shape != (1, 256, 1, 8, 8):
        raise RuntimeError("Unexpected encoder output shape: %s" % (reference.shape,))

    output = out_dir / "ltx_open_vae_encoder.onnx"
    # Bypass torch.onnx.export's compatibility wrapper: in torch 2.6 it
    # attempts an opset conversion that fails when the graph has local
    # ONNX functions. Export the captured program directly and save its
    # native ONNXProgram without that conversion pass.
    exported = torch.export.export(vae.encoder, (image,), strict=False)
    registry = _registration.ONNXRegistry.from_torchlib()
    onnx_program = _core.export(
        exported,
        registry=registry,
        input_names=["image"],
        output_names=["moments"],
        verbose=True,
    )
    onnx_program.save(
        str(output),
        include_initializers=True,
        external_data=False,
    )
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    np.savez_compressed(
        out_dir / "ltx_open_vae_encoder_reference.npz",
        image=image.numpy(),
        torch_output=reference,
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(output), sess_options=options, providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {"image": image.numpy()})[0]
    if actual.shape != reference.shape:
        raise RuntimeError("Output shape mismatch: ORT=%s Torch=%s" % (actual.shape, reference.shape))
    if not np.isfinite(actual).all():
        raise RuntimeError("LTX encoder output contains NaN or infinity")
    max_abs = float(np.max(np.abs(actual - reference)))
    if not np.allclose(actual, reference, rtol=2e-3, atol=2e-4):
        raise RuntimeError("PyTorch/ONNX mismatch: max_abs_error=%.8g" % max_abs)

    print("LTX_ENCODER_ONNX_CHECK=PASS")
    print("LTX_ENCODER_ORT_CPU=PASS")
    print("PYTORCH_ONNX_NUMERICAL_MATCH=PASS")
    print("MAX_ABS_ERROR=%.8g" % max_abs)
    print("LATENT_MOMENTS_SHAPE=%s" % (actual.shape,))


if __name__ == "__main__":
    main()
