#!/usr/bin/env python3

"""Convert Multinomial HMM model to ONNX model using custom skl2onnx
converter"""

import sys
import pickle

import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import Int32TensorType

import skl2onnx_hmm


if len(sys.argv) < 3:
    print("usage: {} <input_model> <output_model>".format(sys.argv[0]))
    exit(1)

hmmlearn_model_path = sys.argv[1]
onnx_model_path = sys.argv[2]

skl2onnx_hmm.register_converter()

hmm_model = pickle.load(open(hmmlearn_model_path, "rb"))

initial_types_hmm = [("input_states", Int32TensorType([None]))]
output_types_hmm = [("output_states", None)]
onx_hmm = convert_sklearn(hmm_model,
                          name="hmm_model",
                          initial_types=initial_types_hmm,
                          final_types=output_types_hmm)
onnx.checker.check_model(onx_hmm)
with open(onnx_model_path, "wb") as f_output_model:
    f_output_model.write(onx_hmm.SerializeToString())
