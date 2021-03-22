#!/usr/bin/env python3

"""Convert Multinomial HMM model to ONNX model using onnx helper functions"""

import sys
import pickle
import numpy as np
from onnx import helper, TensorProto, numpy_helper, checker


if len(sys.argv) < 3:
    print("usage: {} <input_model> <output_model>".format(sys.argv[0]))
    exit(1)

hmmlearn_model_path = sys.argv[1]
onnx_model_path = sys.argv[2]


def log_mask_zero(a):
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        return np.log(a).astype(np.float32)


hmm_clf = pickle.load(open(hmmlearn_model_path, "rb"))
log_emissionprob = log_mask_zero(hmm_clf.emissionprob_)
log_startprob = log_mask_zero(hmm_clf.startprob_)
log_transmat_t = log_mask_zero(hmm_clf.transmat_).T

input_states = helper.make_tensor_value_info(
    'input_states', TensorProto.INT32, [None])

output_states = helper.make_tensor_value_info(
    'output_states', TensorProto.INT32, [None])

node_gather = helper.make_node(
    'Gather',  # node name
    inputs=['log_emissionprob', 'input_states'],  # inputs
    outputs=['framelogprob'],  # outputs
    axis=0,
)

node_viterbi = helper.make_node(
    'Viterbi',
    inputs=['log_startprob', 'log_transmat_t', 'framelogprob'],
    outputs=['output_states'],
    domain='ml.hmm'
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    nodes=[node_gather, node_viterbi],
    name='hmm_model',
    inputs=[input_states],
    outputs=[output_states],
    initializer=[
        numpy_helper.from_array(log_emissionprob.T, 'log_emissionprob'),
        numpy_helper.from_array(log_startprob, 'log_startprob'),
        numpy_helper.from_array(log_transmat_t, 'log_transmat_t'),
    ]
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-hmm-converter')
imp = model_def.opset_import.add()
imp.domain = "ml.hmm"
imp.version = 1
checker.check_model(model_def)
with open(onnx_model_path, "wb") as f:
    f.write(model_def.SerializeToString())
