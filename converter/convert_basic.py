#!/usr/bin/env python3

"""Convert scikit-learn model to ONNX model"""

import sys
import pickle

import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


if len(sys.argv) < 3:
    print("usage: {} <input_model> <output_model>".format(sys.argv[0]))
    exit(1)

input_model = sys.argv[1]
output_model = sys.argv[2]
options = None
skl_model = pickle.load(open(input_model, "rb"))
if len(sys.argv) > 3 and sys.argv[3] == "no_zipmap":
    options = {id(skl_model): {'zipmap': False}}

initial_type = [('float_input', FloatTensorType([None, 3]))]
onnx_model = convert_sklearn(skl_model, initial_types=initial_type, options=options)
onnx.checker.check_model(onnx_model)
with open(output_model, "wb") as f:
    f.write(onnx_model.SerializeToString())
