#!/usr/bin/env python3

"""sklearn-onnx converters for converting hmmlearn models to ONNX model

"""

import numpy as np
import skl2onnx
from onnx import TensorProto
from skl2onnx.common import data_types
from hmmlearn.hmm import MultinomialHMM
from hmmlearn.utils import log_mask_zero


def _multinomial_hmm_shape_calculator(operator):
    """Input/output Shape calculator for Multinomial HMM nodes

    """
    input_node = operator.inputs[0]
    if len(input_node.type.shape) != 1 or \
       not isinstance(input_node.type, data_types.Int32TensorType):
        raise ValueError("input node is not a 1-D tensor of int32")
    n_samples = input_node.type.shape[0]
    operator.outputs[0].type = data_types.Int32TensorType([n_samples])


def _multinomial_hmm_converter(scope, operator, container):
    """Convert HMM Multinomial model to ONNX model

    """
    input_states = operator.inputs[0]
    output_states = operator.outputs[0]

    hmm_clf = operator.raw_operator
    if hmm_clf.algorithm != 'viterbi':
        raise NotImplementedError((
            "Multinomial model with {} decoder algorithm "
            "is not supported yet").format(hmm_clf.algorithm))
    log_emissionprob = log_mask_zero(hmm_clf.emissionprob_).astype(np.float32)
    log_startprob = log_mask_zero(hmm_clf.startprob_).astype(np.float32)
    log_transmat_t = log_mask_zero(hmm_clf.transmat_).astype(np.float32).T
    n_components = log_startprob.shape[0]

    log_emissionprob_name = scope.get_unique_variable_name("log_emissionprob")
    log_startprob_name = scope.get_unique_variable_name("log_startprob")
    log_transmat_t_name = scope.get_unique_variable_name("log_transmat_t")
    container.add_initializer(
        log_emissionprob_name, TensorProto.FLOAT,
        list(log_emissionprob.T.shape),
        log_emissionprob.T.flatten())
    container.add_initializer(
        log_startprob_name, TensorProto.FLOAT,
        list(log_startprob.shape),
        log_startprob.flatten())
    container.add_initializer(
        log_transmat_t_name, TensorProto.FLOAT,
        list(log_transmat_t.shape),
        log_transmat_t.flatten())

    op_name_gather = scope.get_unique_operator_name('Gather')
    op_name_viterbi = scope.get_unique_operator_name('Viterbi')
    framelogprob = scope.declare_local_variable(
        'framelogprob',
        data_types.FloatTensorType([None, n_components]))
    container.add_node(
        'Gather',
        inputs=[log_emissionprob_name, input_states.onnx_name],
        outputs=[framelogprob.onnx_name],
        name=op_name_gather,
        axis=0)
    container.add_node(
        'Viterbi',
        inputs=[log_startprob_name,
                log_transmat_t_name,
                framelogprob.onnx_name],
        outputs=[output_states.onnx_name],
        op_domain='ml.hmm',
        op_version=1,
        name=op_name_viterbi)


def register_converter():
    """Register HMM converters to skl2onnx. This function must be called before
    converting HMM models.

    """
    skl2onnx.update_registered_converter(
        MultinomialHMM, 'HMMLearnMultinomialHMM',
        _multinomial_hmm_shape_calculator,
        _multinomial_hmm_converter)
