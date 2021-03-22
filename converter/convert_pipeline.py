#!/usr/bin/env python3

"""Convert the entire pipeline to ONNX model"""

import sys
import pickle

import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import Int32TensorType
from skl2onnx.common.data_types import FloatTensorType

import skl2onnx_pipeline
from utils import postprocess_onnx_model


if len(sys.argv) < 5:
    print("usage: {} <scaler_model> <clf_model> <hmm_model> <output_model>".format(sys.argv[0]))
    exit(1)

scaler_model_path = sys.argv[1]
clf_model_path = sys.argv[2]
hmm_model_path = sys.argv[3]
onnx_model_path = sys.argv[4]

scaler_model = pickle.load(open(scaler_model_path, "rb"))
clf_model = pickle.load(open(clf_model_path, "rb"))
hmm_model = pickle.load(open(hmm_model_path, "rb"))
n_features = scaler_model.n_features_in_

skl2onnx_pipeline.register_converter()
pipeline = skl2onnx_pipeline.Pipeline(scaler_model, clf_model, hmm_model)
onnx_model = convert_sklearn(
    pipeline,
    name='pipeline',
    initial_types=[('input', FloatTensorType([None, n_features]))],
    final_types=[
        ('states', Int32TensorType([None])),
        ('probs', FloatTensorType([None]))
    ])

onnx_model = postprocess_onnx_model(onnx_model)
onnx.checker.check_model(onnx_model)
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
