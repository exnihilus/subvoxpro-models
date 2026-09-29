"""Small helper to append ONNX nodes with unique, readable names."""
import numpy as np
from onnx import helper, numpy_helper


class GraphBuilder:
    def __init__(self, graph, prefix):
        self.graph = graph
        self.prefix = prefix
        self.nodes = []
        self.constants = {}
        self.counter = 0

    def name(self, hint):
        self.counter += 1
        return f"{self.prefix}/{hint}_{self.counter}"

    def op(self, op_type, inputs, hint=None, outputs=None, **attributes):
        node_name = self.name(hint or op_type)
        outputs = outputs or [node_name + "/output_0"]
        self.nodes.append(helper.make_node(op_type, inputs, outputs, name=node_name, **attributes))
        return outputs[0] if len(outputs) == 1 else outputs

    def constant(self, value, dtype):
        key = (str(dtype), repr(value))
        if key not in self.constants:
            tensor_name = self.name("const")
            array = np.array(value, dtype=dtype)
            self.graph.initializer.append(numpy_helper.from_array(array, tensor_name))
            self.constants[key] = tensor_name
        return self.constants[key]

    def int64(self, value):
        return self.constant(value, np.int64)

    def float32(self, value):
        return self.constant(value, np.float32)
