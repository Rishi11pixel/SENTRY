from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "tinyml" / "sentry_tinyml_int8.tflite"


def format_quantization_params(details: dict) -> dict:
    params = details.get("quantization_parameters", {})
    if not params:
        return {
            "scale": None,
            "zero_point": None,
            "quantized_dimension": None,
        }

    return {
        "scale": params.get("scale"),
        "zero_point": params.get("zero_point"),
        "quantized_dimension": params.get("quantized_dimension"),
    }


def inspect_tensors() -> None:
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()

    print(f"Model path: {MODEL_PATH}")
    print(f"Tensor count: {len(interpreter.get_tensor_details())}")
    print()

    print("ALL TENSORS")
    for detail in interpreter.get_tensor_details():
        name = detail["name"]
        index = detail["index"]
        shape = tuple(detail["shape"])
        dtype = detail["dtype"]
        quant_params = format_quantization_params(detail)
        quant_tuple = detail.get("quantization", (None, None, None))

        print(f"- name: {name}")
        print(f"  index: {index}")
        print(f"  shape: {shape}")
        print(f"  dtype: {dtype}")
        print(f"  quantization_parameters: {quant_params}")
        print(f"  quantization tuple: {quant_tuple}")
        print()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("INPUT DETAILS")
    for details in input_details:
        print(f"- dtype: {details['dtype']}")
        print(f"  shape: {tuple(details['shape'])}")
        print(f"  quantization: {details.get('quantization', (None, None, None))}")
        print(f"  quantization_parameters: {format_quantization_params(details)}")
        print()

    print("OUTPUT DETAILS")
    for details in output_details:
        print(f"- dtype: {details['dtype']}")
        print(f"  shape: {tuple(details['shape'])}")
        print(f"  quantization: {details.get('quantization', (None, None, None))}")
        print(f"  quantization_parameters: {format_quantization_params(details)}")
        print()

    print("TENSORS WITH dtype int8")
    int8_tensors = []
    for detail in interpreter.get_tensor_details():
        if detail["dtype"] == np.int8:
            int8_tensors.append(detail["name"])
    if int8_tensors:
        for name in int8_tensors:
            print(f"- {name}")
    else:
        print("- none")
    print()

    print("TENSORS WITH non-zero quantization scales")
    non_zero_scale_tensors = []
    for detail in interpreter.get_tensor_details():
        params = detail.get("quantization_parameters", {})
        scales = params.get("scale")
        if scales is not None:
            if isinstance(scales, (list, tuple, np.ndarray)):
                values = np.asarray(scales, dtype=np.float32)
                if np.any(values != 0):
                    non_zero_scale_tensors.append(detail["name"])
            else:
                if float(scales) != 0.0:
                    non_zero_scale_tensors.append(detail["name"])

    if non_zero_scale_tensors:
        for name in non_zero_scale_tensors:
            print(f"- {name}")
    else:
        print("- none")
    print()

    print("INT8 model inspection complete.")
    print("Model was not modified.")


if __name__ == "__main__":
    inspect_tensors()
