#!/usr/bin/env python3
"""Real-weight MobileI2V MobileDiT ONNX export probe.

This exports ONLY the official denoiser network, not a complete I2V pipeline.
It intentionally fails on checkpoint incompatibility or unsupported graph ops.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import onnx
from huggingface_hub import hf_hub_download

UPSTREAM = Path(os.environ.get("MOBILEI2V_SOURCE", "MobileI2V")).resolve()
sys.path.insert(0, str(UPSTREAM))

from diffusion.model.builder import build_model
from diffusion.utils.config import SanaConfig, model_init_config
import pyrallis


class DenoiserExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, latent, timestep, guide_image, text_features, cond_mask, flow_score):
        return self.model(
            latent, timestep, guide_image, text_features, cond_mask, flow_score,
            mask=None, data_info=None
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/mobiledit_300m.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(7)
    cfg_path = UPSTREAM / "configs/mobilei2v_config/MobileI2V_300M_img512.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)

    config = pyrallis.parse(config_class=SanaConfig, config_path=str(cfg_path))
    # 720x1280 input -> latent grid 23x40 (the official inference code rounds
    # 720/32 upward because it is not divisible by 32); 17 output frames -> 3 latent frames.
    latent_h = config.model.image_height // config.vae.vae_downsample_rate
    if config.model.image_height % config.vae.vae_downsample_rate:
        latent_h += 1
    latent_w = config.model.image_width // config.vae.vae_downsample_rate
    latent_t = config.data.num_frames // 8 + 1
    assert (latent_t, latent_h, latent_w) == (3, 23, 40), (latent_t, latent_h, latent_w)

    model = build_model(
        config.model.model,
        use_fp32_attention=config.model.get("fp32_attention", False),
        **model_init_config(config, latent_size=(latent_h, latent_w)),
    ).cpu().eval().float()

    checkpoint = hf_hub_download(
        repo_id="hustvl/MobileI2V", filename="hybrid_371.pth"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    state.pop("pos_embed", None)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint mismatch: missing=%s unexpected=%s"
            % (incompatible.missing_keys[:20], incompatible.unexpected_keys[:20])
        )

    print("CHECKPOINT_LOADED=YES")
    print("PARAMETERS=%d" % sum(p.numel() for p in model.parameters()))
    print("LATENT_SHAPE=[1,128,3,23,40]")
    print("TEXT_FEATURE_SHAPE=[1,1,300,896]")

    wrapped = DenoiserExport(model).eval()
    inputs = (
        torch.randn(1, 128, 3, 23, 40, dtype=torch.float32),
        torch.ones(1, dtype=torch.float32),
        torch.randn(1, 128, 1, 23, 40, dtype=torch.float32),
        torch.randn(1, 1, 300, 896, dtype=torch.float32),
        torch.ones(1, 2760, dtype=torch.float32),
        torch.ones(1, dtype=torch.float32),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped, inputs, str(output),
        input_names=["latent", "timestep", "guide_image", "text_features", "cond_mask", "flow_score"],
        output_names=["noise_prediction"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    print("ONNX_CHECK=PASS")
    print("ONNX_FILE=%s" % output)
    print("NOTE=This is the denoiser only; VAE encoder/decoder, tokenizer/text encoder, sampler and Android runtime are not included.")


if __name__ == "__main__":
    main()
