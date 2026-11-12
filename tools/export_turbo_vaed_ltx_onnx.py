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
    # The official checkpoint also contains global latent statistics and
    # feature-projection heads. Load only tensors owned by the decoder.
    decoder_state = model.decoder.state_dict()
    decoder_weights = {key: value for key, value in state.items() if key in decoder_state}
    if not decoder_weights:
        raise RuntimeError("No decoder weights matched the Turbo-VAED decoder")
    incompatible = model.decoder.load_state_dict(decoder_weights, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError("Unexpected decoder keys after filtering: %s" % incompatible.unexpected_keys)
    if incompatible.missing_keys:
        raise RuntimeError("Turbo-VAED decoder weights are incomplete: %s" % incompatible.missing_keys[:20])
    print("DECODER_WEIGHT_TENSORS_LOADED=%d" % len(decoder_weights))
    print("IGNORED_NON_DECODER_TENSORS=%d" % (len(state) - len(decoder_weights)))
    print("TURBO_VAED_WEIGHTS_LOADED=YES")

    # Target MobileI2V latent grid: 3 temporal frames, 23x40 spatial tokens.
    torch.manual_seed(20261112)
    latents = torch.randn(1, 128, 3, 23, 40, dtype=torch.float32)
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
    if actual.shape[2] != 17 or actual.shape[3:] != (736, 1280):
        raise RuntimeError("Unexpected target decoded video shape: %s" % (actual.shape,))

    print("TURBO_VAED_ONNX_CHECK=PASS")
    print("TURBO_VAED_ORT_CPU=PASS")
    print("PYTORCH_ONNX_NUMERICAL_MATCH=PASS")
    print("MAX_ABS_ERROR=%.8g" % max_abs)
    print("VIDEO_SHAPE=%s" % (actual.shape,))
    print("TARGET_720P_DECODER=PASS")


if __name__ == "__main__":
    main()
