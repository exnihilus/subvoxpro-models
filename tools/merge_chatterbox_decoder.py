"""Merges the split Chatterbox decoder graphs into a few larger graphs.

Each ONNX Runtime DirectML session keeps its own pool of GPU buffers, sized by the largest phrase it has rendered,
so 24 sessions hold 24 pools: about 7.4 GB after twenty phrase lengths. Six sessions hold about 3.2 GB for the
same work at the same speed, and still load in parallel within a reasonable time (the planner cost that made the
whole graph slow to load grows faster than linearly with its size).

The merged graphs chain the same nodes in the same order and reference the same upstream weights
(conditional_decoder.onnx_data), so only the partition changes: with the random nodes seeded, the waveform matches
the 24-part decoder within 2e-5 (at most one 16-bit step on under 1 % of the samples).

Usage: python merge_chatterbox_decoder.py <folder with conditional_decoder.partNN.onnx> <group count>
Writes conditional_decoder.groupNN.onnx next to the parts.
"""
import glob
import os
import sys

import onnx
from onnx import helper


def merge(part_paths, group_count):
    models = [onnx.load(path, load_external_data=False) for path in part_paths]
    size = (len(models) + group_count - 1) // group_count
    final_outputs = {output.name for output in models[-1].graph.output}
    groups = []
    for start in range(0, len(models), size):
        chunk = models[start:start + size]
        later_inputs = {value.name for model in models[start + size:] for value in model.graph.input}
        produced, inputs, nodes, initializers, outputs = set(), [], [], [], []
        seen_inputs, seen_initializers, seen_outputs = set(), set(), set()
        for model in chunk:
            for value in model.graph.input:
                if value.name not in produced and value.name not in seen_inputs:
                    inputs.append(value)
                    seen_inputs.add(value.name)
            nodes.extend(model.graph.node)
            for tensor in model.graph.initializer:
                if tensor.name not in seen_initializers:
                    initializers.append(tensor)
                    seen_initializers.add(tensor.name)
            produced |= {value.name for value in model.graph.output}
        for model in chunk:
            for value in model.graph.output:
                if value.name in later_inputs | final_outputs and value.name not in seen_outputs:
                    outputs.append(value)
                    seen_outputs.add(value.name)
        graph = helper.make_graph(nodes, f"decoder_group{len(groups)}", inputs, outputs, initializer=initializers)
        groups.append(helper.make_model(graph, opset_imports=chunk[0].opset_import, ir_version=chunk[0].ir_version))
    return groups


def main():
    folder, group_count = sys.argv[1], int(sys.argv[2])
    part_paths = sorted(glob.glob(os.path.join(folder, "conditional_decoder.part*.onnx")))
    for index, model in enumerate(merge(part_paths, group_count)):
        path = os.path.join(folder, f"conditional_decoder.group{index:02d}.onnx")
        onnx.save(model, path)
        print(path, len(model.graph.node), "nodes")


if __name__ == "__main__":
    main()
