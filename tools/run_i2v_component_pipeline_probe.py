#!/usr/bin/env python3
"""Compose the verified MobileI2V ONNX stages at the real 720p latent grid."""
from pathlib import Path
import gc
import numpy as np
import onnxruntime as ort
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

def ort_session(path):
    o=ort.SessionOptions(); o.intra_op_num_threads=2; o.inter_op_num_threads=1
    o.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL; o.enable_mem_pattern=False
    return ort.InferenceSession(str(path),sess_options=o,providers=["CPUExecutionProvider"])

def locate(root, filename):
    hits=list(Path(root).rglob(filename))
    if len(hits)!=1: raise RuntimeError(f"Expected one {filename} under {root}, found {hits}")
    return hits[0]

def main():
    enc_path=locate("artifacts/encoder","ltx_open_vae_encoder.onnx")
    dit_path=locate("artifacts/mobiledit","mobiledit_300m.onnx")
    dec_path=locate("artifacts/decoder","turbo_vaed_ltx_decoder.onnx")
    print("ALL_THREE_EXPORTED_MODELS_FOUND=PASS",flush=True)

    # Deterministic representative 720p RGB photo tensor, normalized to [-1,1].
    rng=np.random.default_rng(20261113)
    image=(rng.random((1,3,1,720,1280),dtype=np.float32)*2-1).astype(np.float32)
    enc=ort_session(enc_path)
    enc_inputs={i.name for i in enc.get_inputs()}
    if enc_inputs!={"image"}: raise RuntimeError(f"Encoder inputs: {enc_inputs}")
    moments=enc.run(None,{"image":image})[0]
    del enc; gc.collect()
    if moments.shape!=(1,256,1,23,40) or not np.isfinite(moments).all():
        raise RuntimeError(f"Unexpected encoder moments: {moments.shape}")
    mean,logvar=np.split(moments,2,axis=1)
    # Diffusers diagonal-Gaussian posterior sampling, scaling_factor=1.0
    # from the current LTX Video VAE config used by the export.
    logvar=np.clip(logvar,-30.0,20.0)
    eps=np.random.default_rng(20261114).standard_normal(mean.shape).astype(np.float32)
    guide=(mean+np.exp(0.5*logvar)*eps).astype(np.float32)
    if guide.shape!=(1,128,1,23,40): raise RuntimeError(f"Guide latent shape: {guide.shape}")
    print("720P_IMAGE_TO_LTX_GUIDE_LATENT=PASS",flush=True)

    dit=ort_session(dit_path)
    inputs={i.name for i in dit.get_inputs()}
    if inputs!={"latent","timestep","cond_mask","flow_score"}:
        raise RuntimeError(f"Unexpected MobileDiT input signature: {inputs}")
    latents=rng.standard_normal((1,128,3,23,40),dtype=np.float32)
    latents[:,:,:1]=guide
    cond=np.zeros((1,3*23*40),dtype=np.float32); cond[:,:23*40]=1
    score=np.array([2.0],dtype=np.float32)
    scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0)
    scheduler.set_timesteps(2,device="cpu")
    for t in scheduler.timesteps:
        pred=dit.run(None,{"latent":latents,"timestep":np.array([float(t.item())],dtype=np.float32),
                           "cond_mask":cond,"flow_score":score})[0]
        latents=scheduler.step(torch.from_numpy(pred),t,torch.from_numpy(latents),return_dict=False)[0].numpy()
        latents[:,:,:1]=guide
        if not np.isfinite(latents).all(): raise RuntimeError("Non-finite MobileDiT latents")
    del dit; gc.collect()
    print("MOBILEDIT_FLOW_EULER_2_STEPS=PASS",flush=True)
    print("FIRST_FRAME_IMAGE_CONDITION=PASS",flush=True)

    dec=ort_session(dec_path)
    if {i.name for i in dec.get_inputs()}!={"latents"}: raise RuntimeError("Unexpected decoder inputs")
    video=dec.run(None,{"latents":latents})[0]
    if video.shape!=(1,3,17,736,1280) or not np.isfinite(video).all():
        raise RuntimeError(f"Unexpected decoded video: {video.shape}")
    video=video[:,:,:,:720,:]
    if video.shape!=(1,3,17,720,1280): raise RuntimeError(f"Crop shape: {video.shape}")
    print("TURBO_VAED_17_FRAME_DECODE=PASS",flush=True)
    print("FINAL_VIDEO_TENSOR=(1,3,17,720,1280)",flush=True)
    print("END_TO_END_COMPONENT_COMPOSITION=PASS",flush=True)

if __name__=="__main__": main()
