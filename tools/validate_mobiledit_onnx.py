#!/usr/bin/env python3
"""Load and execute the exported MobileI2V denoiser with ONNX Runtime CPU.

This is a graph/runtime smoke test only. It does not claim a complete I2V
pipeline: VAE, Qwen tokenizer/text encoder, scheduler and video decoder remain
separate required stages.
"""
import argparse
import sys

import numpy as np
import onnxruntime as ort


EXPECTED = {
    "latent": (1, 128, 3, 23, 40),
    "timestep": (1,),
    "cond_mask": (1, 2760),
    "flow_score": (1,),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/mobiledit_300m.onnx")
    args = parser.parse_args()

    print("ONNXRUNTIME_VERSION=" + ort.__version__)
    # The current upstream Mobiledit block does not consume guide_image or y:
    # image conditioning is injected into latent frame 0 by the sampler.
    print("UPSTREAM_UNUSED_ARGS=guide_image,text_features")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.enable_mem_pattern = False

    session = ort.InferenceSession(
        args.model, sess_options=options, providers=["CPUExecutionProvider"]
    )
    actual = {item.name: item for item in session.get_inputs()}
    if set(actual) != set(EXPECTED):
        raise RuntimeError(
            "Input names differ: expected=%s actual=%s"
            % (sorted(EXPECTED), sorted(actual))
        )

    feed = {}
    rng = np.random.default_rng(20261112)
    for name, shape in EXPECTED.items():
        item = actual[name]
        if item.type != "tensor(float)":
            raise RuntimeError("%s has unexpected type %s" % (name, item.type))
        if tuple(item.shape) != shape:
            raise RuntimeError(
                "%s has unexpected shape %s; expected %s"
                % (name, item.shape, shape)
            )
        feed[name] = rng.standard_normal(shape, dtype=np.float32)

    # Match the official sampler: lock the first latent frame as the image condition.
    feed["cond_mask"] = np.zeros(EXPECTED["cond_mask"], dtype=np.float32)
    feed["cond_mask"][:, : 23 * 40] = 1.0
    feed["timestep"] = np.array([500.0], dtype=np.float32)
    feed["flow_score"] = np.array([2.0], dtype=np.float32)

    print("CONDITION_MASK=FIRST_LATENT_FRAME")
    print("CFG_CAN_BE_SINGLE_PASS=YES (upstream MobileDiT blocks ignore text features)")
    print("ORT_SESSION=CREATED")
    print("INPUT_SIGNATURE=PASS")
    outputs = session.run(None, feed)
    if len(outputs) != 1:
        raise RuntimeError("Expected one denoiser output, got %d" % len(outputs))
    result = outputs[0]
    if result.shape != EXPECTED["latent"]:
        raise RuntimeError(
            "Output shape mismatch: got %s expected %s"
            % (result.shape, EXPECTED["latent"])
        )
    if not np.isfinite(result).all():
        raise RuntimeError("Denoiser output contains NaN or infinity")

    print("ORT_CPU_INFERENCE=PASS")
    print("OUTPUT_SHAPE=%s" % (result.shape,))
    print("OUTPUT_FINITE=YES")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ORT_SMOKE_TEST=FAIL: %s" % exc, file=sys.stderr)
        raise
