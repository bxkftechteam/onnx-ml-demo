#!/usr/bin/env python3

"""Run inference on ONNX model"""

import sys
import time
import numpy as np
import onnxruntime as ort


if len(sys.argv) < 3:
    print("usage: {} <model_path> <N>".format(sys.argv[0]))
    exit(1)

onnx_model_path = sys.argv[1]
N = int(sys.argv[2])
features = np.random.uniform(low=0.0, high=2.0, size=(N, 3))

sess = ort.InferenceSession(onnx_model_path)
start = time.time()
res = sess.run(None, {"float_input": features.astype(np.float32)})
cost = time.time() - start

if len(sys.argv) < 4 or sys.argv[3] != "q":
    print(res)
print("cost: %d" % (cost * 1000))
