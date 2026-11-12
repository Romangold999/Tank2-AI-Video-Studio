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

    reference_path = Path(args.model).with_name("mobiledit_reference.npz")
    if not reference_path.is_file():
        raise FileNotFoundError(
            "PyTorch reference missing: %s" % reference_path
        )
    with np.load(reference_path) as reference:
        feed = {name: reference[name] for name in EXPECTED}
        torch_output = reference["torch_output"]

    for name, shape in EXPECTED.items():
        item = actual[name]
        if item.type != "tensor(float)":
            raise RuntimeError("%s has unexpected type %s" % (name, item.type))
        if tuple(item.shape) != shape:
            raise RuntimeError(
                "%s has unexpected shape %s; expected %s"
                % (name, item.shape, shape)
            )
        if feed[name].shape != shape or feed[name].dtype != np.float32:
            raise RuntimeError(
                "%s reference has invalid shape/dtype: %s %s"
                % (name, feed[name].shape, feed[name].dtype)
            )

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
    if not np.isfinite(torch_output).all():
        raise RuntimeError("PyTorch reference contains NaN or infinity")

    max_abs = float(np.max(np.abs(result - torch_output)))
    max_ref = float(np.max(np.abs(torch_output)))
    if not np.allclose(result, torch_output, rtol=2e-3, atol=2e-4):
        raise RuntimeError(
            "PyTorch/ONNX mismatch: max_abs_error=%.8g max_reference=%.8g"
            % (max_abs, max_ref)
        )

    print("ORT_CPU_INFERENCE=PASS")
    print("PYTORCH_ONNX_NUMERICAL_MATCH=PASS")
    print("MAX_ABS_ERROR=%.8g" % max_abs)
    print("OUTPUT_SHAPE=%s" % (result.shape,))
    print("OUTPUT_FINITE=YES")



if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ORT_SMOKE_TEST=FAIL: %s" % exc, file=sys.stderr)
        raise
