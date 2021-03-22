#!/usr/bin/env python3

from toposort import toposort_flatten


def postprocess_onnx_model(onnx_model):
    """Tune order of nodes to topological order. We have to do this manually since
    skl2onnx could not emit nodes of custom models in correct order

    """
    node_info = [{
        'name': node.name,
        'in': list(node.input),
        'out': list(node.output)
    } for node in onnx_model.graph.node]

    # resolve node dependencies
    dependencies = dict()
    dependent_nodes_of_value = dict()
    for node in node_info:
        for out in node['out']:
            if out not in dependent_nodes_of_value:
                dependent_nodes_of_value[out] = []
            dependent_nodes_of_value[out].append(node['name'])
    for node in node_info:
        node_dependencies = set()
        for depend_on in node['in']:
            if depend_on in dependent_nodes_of_value:
                node_dependencies.update(dependent_nodes_of_value[depend_on])
        dependencies[node['name']] = node_dependencies

    # topological sort
    ordered_node_names = toposort_flatten(dependencies)
    onnx_model.graph.node.sort(
        key=lambda node: ordered_node_names.index(node.name))
    return onnx_model
