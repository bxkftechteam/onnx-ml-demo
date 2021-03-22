#!/usr/bin/env python3

"""Run inference on HMM models"""

import sys
from os import path
import numpy as np
import onnxruntime as ort


if len(sys.argv) < 3:
    print("usage: {} <onnx_model_path> <N>".format(
        sys.argv[0]))
    exit(1)

onnx_model_path = sys.argv[1]
N = int(sys.argv[2])
features = np.random.randint(low=0, high=6, size=N).astype(np.int32)

project_dir = path.dirname(path.dirname(path.abspath(__file__)))
shared_library_path = path.join(project_dir, "cpp/libviterbi.so")
session_options = ort.SessionOptions()
session_options.register_custom_ops_library(shared_library_path)

sess = ort.InferenceSession(onnx_model_path, session_options)
res = sess.run(None, {"input_states": features})
print(res)
