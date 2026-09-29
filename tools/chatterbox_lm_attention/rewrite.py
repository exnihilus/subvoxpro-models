"""Rewrites the Chatterbox language model so DirectML can run it.

Every com.microsoft.GroupQueryAttention node (packed QKV, rotary embedding, KV cache) becomes standard ONNX
operators with a fixed-capacity KV cache (see attention.py for the input protocol, including the added
`cache_write_index` input); everything else and the weights are untouched. The rewritten graph is saved next to
the original and keeps referencing the same external weights file, so no weight is duplicated.

Usage: python rewrite.py <language_model.onnx> <output.onnx>
"""
import os
import sys

import onnx
from onnx import TensorProto, helper

from attention import WRITE_INDEX_INPUT, AttentionShape, build_layer_attention, build_shared_context
from graph_builder import GraphBuilder

GQA_OP = "GroupQueryAttention"
MASK_SUBGRAPH_PREFIX = "/model/attn_mask_reformat/"


def attention_shape(node, graph):
    attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    if attributes.get("num_heads") != attributes.get("kv_num_heads"):
        raise ValueError(f"{node.name}: grouped heads are not supported by this rewrite.")
    if attributes.get("do_rotary") != 1 or attributes.get("rotary_interleaved", 0) != 0:
        raise ValueError(f"{node.name}: only non-interleaved rotary attention is supported.")
    if attributes.get("local_window_size", -1) != -1 or attributes.get("softcap", 0.0) != 0.0:
        raise ValueError(f"{node.name}: sliding window and softcap are not supported.")

    past_key = next(i for i in graph.input if i.name == node.input[3])
    head_size = past_key.type.tensor_type.shape.dim[3].dim_value
    return AttentionShape(attributes["num_heads"], head_size, attributes["scale"])


def rewrite(model):
    graph = model.graph
    nodes = list(graph.node)
    attention_nodes = [n for n in nodes if n.op_type == GQA_OP]
    if not attention_nodes:
        raise ValueError("No GroupQueryAttention node found.")

    first = attention_nodes[0]
    graph.input.append(helper.make_tensor_value_info(WRITE_INDEX_INPUT, TensorProto.INT64, [1]))
    shared = GraphBuilder(graph, "/model/standard_attention/shared")
    context = build_shared_context(shared, "attention_mask", "inputs_embeds", first.input[3], first.input[7], first.input[8])

    rewritten = list(shared.nodes)
    for node in nodes:
        if node.name.startswith(MASK_SUBGRAPH_PREFIX):
            continue
        if node.op_type != GQA_OP:
            rewritten.append(node)
            continue

        layer = GraphBuilder(graph, node.name.rsplit("/", 1)[0] + "/standard_attention")
        build_layer_attention(
            layer, attention_shape(node, graph), context,
            qkv=node.input[0], past_key=node.input[3], past_value=node.input[4],
            output=node.output[0], present_key=node.output[1], present_value=node.output[2])
        rewritten.extend(layer.nodes)

    del graph.node[:]
    graph.node.extend(rewritten)
    return len(attention_nodes)


def main(source, destination):
    if os.path.dirname(os.path.abspath(source)) != os.path.dirname(os.path.abspath(destination)):
        raise ValueError("Save next to the source so the external weights file still resolves.")

    model = onnx.load(source, load_external_data=False)
    replaced = rewrite(model)
    onnx.save(model, destination)
    print(f"Replaced {replaced} attention nodes -> {destination} ({os.path.getsize(destination)} bytes)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
