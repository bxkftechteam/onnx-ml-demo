#!/usr/bin/env python3

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin
from skl2onnx.common import data_types
import skl2onnx
from skl2onnx.algebra import onnx_ops
from onnx import TensorProto

import skl2onnx_hmm


class Pipeline(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_model, clf_model, hmm_model):
        prob_bins = np.array([-np.inf, 0.1, 0.3, 0.5, 0.7, 0.9, np.inf])
        bins_discretizer = KBinsDiscretizer(encode='ordinal')
        bins_discretizer.n_bins_ = np.array([prob_bins.shape[0]])
        bins_discretizer.bin_edges_ = prob_bins.reshape(1, -1)
        TransformerMixin.__init__(self)
        BaseEstimator.__init__(self)
        self.scaler_model_ = scaler_model
        self.clf_model_ = clf_model
        self.hmm_model_ = hmm_model
        self.bins_discretizer_ = bins_discretizer

    def fit(self, X):
        raise NotImplementedError("fit is not implemented")

    def transform(self, X):
        X_scaled = self.scaler_model_.transform(X)
        prob = self.clf_model_.predict_proba(X_scaled)
        prob_positive = prob[:, 1]
        states = self.bins_discretizer_.transform(prob_positive.reshape(-1, 1))
        states = states.flatten().astype(np.int32)
        final_states = self.hmm_model_.predict(states)
        return final_states, prob_positive

    @staticmethod
    def shape_calculator(operator):
        scaler_model = operator.raw_operator.scaler_model_
        clf_model = operator.raw_operator.clf_model_
        n_features = scaler_model.n_features_in_
        n_classes = clf_model.n_classes_
        if n_classes != 2:
            raise ValueError(
                'main classifier is not binary classification model ' +
                '(n_classes = {})'.format(n_classes))
        if len(operator.inputs) != 1:
            raise ValueError('this classifier only accept 1 input')

        input_node = operator.inputs[0]
        if len(input_node.type.shape) != 2 or \
           not isinstance(input_node.type, data_types.FloatTensorType):
            raise ValueError(
                "input node is not a 2-dimensional tensor of float")
        if input_node.type.shape[1] is None:
            input_node.type.shape[1] = n_features
        if input_node.type.shape[1] != n_features:
            raise ValueError(
                "feature size does not match with model ({} vs {})".format(
                    input_node.type.shape[1], n_features))

        N = input_node.type.shape[0]
        operator.outputs[0].type = data_types.Int32TensorType([N])
        operator.outputs[1].type = data_types.FloatTensorType([N])

    @staticmethod
    def converter(scope, operator, container):
        opv = container.target_opset
        clf = operator.raw_operator
        output_states = operator.outputs[0]
        output_prob = operator.outputs[1]

        scaler_model = operator.raw_operator.scaler_model_
        clf_model = operator.raw_operator.clf_model_
        hmm_model = operator.raw_operator.hmm_model_
        n_features = scaler_model.n_features_in_
        n_classes = clf_model.n_classes_

        # First step is the scaler
        alias = skl2onnx.get_model_alias(type(scaler_model))
        scaler_op = scope.declare_local_operator(alias, scaler_model)
        scaler_op.inputs = operator.inputs
        scaler_output = scope.declare_local_variable(
            'scaler_output', data_types.FloatTensorType([None, n_features]))
        scaler_op.outputs.append(scaler_output)
        scaler_op.infer_types()

        # Second step is random forest classifier
        alias = skl2onnx.get_model_alias(type(clf_model))
        classifier_op = scope.declare_local_operator(alias, clf_model)
        classifier_op.inputs.append(scaler_output)
        classifier_output_labels = scope.declare_local_variable(
            'classifier_output_labels',
            data_types.Int32TensorType([None]))
        classifier_output_probabilities = scope.declare_local_variable(
            'classifier_output_probabilities',
            data_types.FloatTensorType([None, n_classes]))
        classifier_op.outputs.append(classifier_output_labels)
        classifier_op.outputs.append(classifier_output_probabilities)
        classifier_op.infer_types()

        # Take the second column from probabilities produced by classifier
        gather_indices_name = scope.get_unique_variable_name('gather_indices')
        container.add_initializer(
            gather_indices_name, TensorProto.INT32,
            [1], [1])
        gather_positive = onnx_ops.OnnxGather(
            classifier_output_probabilities,
            gather_indices_name,
            axis=1,
            output_names=[output_prob],
            op_version=opv)
        gather_positive.set_onnx_name_prefix("gather_positive")
        gather_positive.add_to(scope, container)

        # Translate probabilities of class-1 to discrete integers
        bins_discretizer = clf.bins_discretizer_
        alias = skl2onnx.get_model_alias(type(bins_discretizer))
        discretizer_op = scope.declare_local_operator(alias, bins_discretizer)
        discretizer_op.inputs.append(output_prob)
        states_2d = scope.declare_local_variable(
            'states_2d',
            data_types.FloatTensorType([None, 1]))
        discretizer_op.outputs.append(states_2d)
        discretizer_op.infer_types()

        # Output of discretizer is an N * 1 tensor, we need to flatten it as an
        # N dimensional array
        final_shape_name = scope.get_unique_variable_name("final_shape")
        container.add_initializer(
            final_shape_name, TensorProto.INT64,
            [1], [-1])

        hmm_input_name = scope.declare_local_variable(
            'hmm_input', data_types.Int32TensorType([None]))
        cast = onnx_ops.OnnxCast(
            onnx_ops.OnnxReshape(
                states_2d, final_shape_name,
                op_version=opv),
            to=TensorProto.INT32,
            output_names=[hmm_input_name],
            op_version=opv)
        cast.set_onnx_name_prefix("cast")
        cast.add_to(scope, container)

        # feed reshaped array to HMM
        hmm_output_name = scope.declare_local_variable(
            'hmm_output', data_types.Int32TensorType([None]))
        alias = skl2onnx.get_model_alias(type(hmm_model))
        hmm_op = scope.declare_local_operator(alias, hmm_model)
        hmm_op.inputs.append(hmm_input_name)
        hmm_op.outputs.append(hmm_output_name)
        classifier_op.infer_types()

        # an extra identity
        final_identity = onnx_ops.OnnxIdentity(
            hmm_output_name, output_names=[output_states], op_version=opv)
        final_identity.set_onnx_name_prefix("final_identity")
        final_identity.add_to(scope, container)

    @staticmethod
    def parser(scope, model, inputs):
        """Custom parser for parsing ONNX model outputs

        """
        alias = skl2onnx.get_model_alias(type(model))
        this_op = scope.declare_local_operator(alias, model)
        this_op.inputs = inputs
        this_op.outputs.append(
            scope.declare_local_variable(
                'states', data_types.Int32TensorType([None])))
        this_op.outputs.append(
            scope.declare_local_variable(
                'probs', data_types.FloatTensorType([None])))
        return this_op.outputs


def register_converter():
    """Register phone play classifier to skl2onnx

    """
    skl2onnx_hmm.register_converter()

    skl2onnx.update_registered_converter(
        Pipeline,
        'DemoPipeline',
        lambda operator: Pipeline.shape_calculator(operator),
        lambda scope, operator, container:
        Pipeline.converter(scope, operator, container))

    skl2onnx.update_registered_parser(
        Pipeline,
        lambda scope, model, inputs, custom_parsers:
        Pipeline.parser(scope, model, inputs))
