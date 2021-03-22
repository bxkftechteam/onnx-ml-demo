#!/usr/bin/env python3

import os
import sys
import onnx
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer


if len(sys.argv) < 2:
    print("usage: {} <model_path>".format(sys.argv[0]))
    exit(1)

onnx_model_path = sys.argv[1]
model = onnx.load(onnx_model_path)
pydot_graph = GetPydotGraph(model.graph, name=model.graph.name, rankdir="TB",
                            node_producer=GetOpNodeProducer("docstring"))
pydot_graph.write_dot(onnx_model_path + ".dot")
os.system('dot -O -Tpng {}.dot'.format(onnx_model_path))
