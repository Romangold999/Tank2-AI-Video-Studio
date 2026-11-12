#!/usr/bin/env python3
"""Export the official Turbo-VAED LTX decoder and compare ONNX with PyTorch.

The encoder is deliberately not replaced here: MobileI2V requires the matching
LTX Video VAE encoder to convert the user's still image into the same latent
space. This probe covers decoder export/runtime only.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download

TURBO_SOURCE = Path(os.environ.get("TURBO_VAED_SOURCE", "Turbo-VAED")).resolve()
sys.path.insert(0, str(TURBO_SOURCE / "diffusers_vae" / "src"))

from diffusers.models.autoencoders.autoencoder_kl_turbo_vaed import AutoencoderKLTurboVAED


def main():
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = TURBO_SOURCE / "configs" / "Turbo-VAED-LTX.json"
    config = json.loads(config_path.read_text())
    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)

    model = AutoencoderKLTurboVAED(**config).cpu().eval().float()
    checkpoint_path = hf_hub_download(
        repo_id="hustvl/Turbo-VAED", filename="Turbo-VAED-LTX.pth"
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    incompatible = model.decoder.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError("Unexpected Turbo-VAED keys: %s" % incompatible.unexpected_keys)
    if incompatible.missing_keys:
        print("DECODER_MISSING_KEYS=%d" % len(incompatible.missing_keys))
    print("TURBO_VAED_WEIGHTS_LOADED=YES")

    # 3 latent frames decode to 17 video frames; 8x8 latents test the 32x scale.
    torch.manual_seed(20261112)
    latents = torch.randn(1, 128, 3, 8, 8, dtype=torch.float32)
    with torch.inference_mode():
        reference = model.decoder(latents).cpu().numpy()

    output = out_dir / "turbo_vaed_ltx_decoder.onnx"
    torch.onnx.export(
        model.decoder,
        (latents,),
        str(output),
        input_names=["latents"],
        output_names=["video"],
        dynamic_axes={
            "latents": {3: "latent_height", 4: "latent_width"},
            "video": {3: "video_height", 4: "video_width"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    np.savez_compressed(
        out_dir / "turbo_vaed_ltx_reference.npz",
        latents=latents.numpy(),
        torch_output=reference,
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(output), sess_options=options, providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {"latents": latents.numpy()})[0]
    if actual.shape != reference.shape:
        raise RuntimeError("Output shape mismatch: ORT=%s Torch=%s" % (actual.shape, reference.shape))
    if not np.isfinite(actual).all():
        raise RuntimeError("ONNX decoder output contains NaN or infinity")
    max_abs = float(np.max(np.abs(actual - reference)))
    if not np.allclose(actual, reference, rtol=2e-3, atol=2e-4):
        raise RuntimeError("PyTorch/ONNX mismatch: max_abs_error=%.8g" % max_abs)
    if actual.shape[2] != 17 or actual.shape[3:] != (256, 256):
        raise RuntimeError("Unexpected decoded video shape: %s" % (actual.shape,))

    print("TURBO_VAED_ONNX_CHECK=PASS")
    print("TURBO_VAED_ORT_CPU=PASS")
    print("PYTORCH_ONNX_NUMERICAL_MATCH=PASS")
    print("MAX_ABS_ERROR=%.8g" % max_abs)
    print("VIDEO_SHAPE=%s" % (actual.shape,))


if __name__ == "__main__":
    main()
