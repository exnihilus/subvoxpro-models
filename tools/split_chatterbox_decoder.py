import onnx, json, sys
from onnx import helper
K = int(sys.argv[1])
m = onnx.load("conditional_decoder.onnx", load_external_data=False)
g = m.graph
nodes = list(g.node)
inits = {t.name: t for t in g.initializer}
vi = {v.name: v for v in list(g.value_info) + list(g.input) + list(g.output)}
graph_inputs = {i.name for i in g.input}
graph_outputs = [o.name for o in g.output]

def outer_refs(sg):
    produced = {i.name for i in sg.input} | {t.name for t in sg.initializer}
    refs = []
    for n in sg.node:
        for x in node_inputs(n):
            if x not in produced:
                refs.append(x)
        produced |= set(n.output)
    return refs

def node_inputs(n):
    names = [x for x in n.input if x]
    for a in n.attribute:
        if a.type == onnx.AttributeProto.GRAPH:
            names += outer_refs(a.g)
        elif a.type == onnx.AttributeProto.GRAPHS:
            for sg in a.graphs:
                names += outer_refs(sg)
    return names

constant_producer = {}
producer = {}
for i, n in enumerate(nodes):
    for o in n.output:
        producer[o] = i
        if n.op_type == "Constant":
            constant_producer[o] = n
work = [i for i, n in enumerate(nodes) if n.op_type != "Constant"]
per = (len(work) + K - 1) // K
bounds = [work[min(c * per, len(work) - 1)] for c in range(K)] + [len(nodes)]
consumers = {}
for i, n in enumerate(nodes):
    for x in node_inputs(n):
        consumers.setdefault(x, []).append(i)
manifest = []
for c in range(K):
    s, e = bounds[c], bounds[c + 1]
    chunk = [n for n in nodes[s:e] if n.op_type != "Constant"]
    needed_consts, inputs, used_inits = [], [], []
    seen = set()
    produced = set()
    for n in chunk:
        for x in node_inputs(n):
            if x in seen or x in produced:
                continue
            seen.add(x)
            if x in inits:
                used_inits.append(inits[x])
            elif x in constant_producer:
                needed_consts.append(constant_producer[x])
            elif x in graph_inputs or producer.get(x, e) < s:
                inputs.append(x)
        produced |= set(n.output)
    outputs = [o for n in chunk for o in n.output if o and (o in graph_outputs or any(j >= e for j in consumers.get(o, [])))]
    def info(name):
        if name not in vi:
            raise SystemExit(f"missing value_info for {name}")
        return vi[name]
    graph = helper.make_graph(needed_consts + chunk, f"decoder_part{c}", [info(x) for x in inputs], [info(x) for x in outputs], initializer=used_inits)
    model = helper.make_model(graph, opset_imports=m.opset_import, ir_version=m.ir_version)
    name = f"conditional_decoder.part{c:02d}.onnx"
    onnx.save(model, name)
    manifest.append({"file": name, "inputs": inputs, "outputs": outputs, "nodes": len(chunk)})
json.dump(manifest, open(f"conditional_decoder.parts.json", "w"), indent=1)
for p in manifest:
    print(p["file"], "nodes", p["nodes"], "in", len(p["inputs"]), "out", len(p["outputs"]))
