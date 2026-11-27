# SPDX-License-Identifier: GPL-3.0-only
"""Build flow-only RIFE models: XeSS frame generation motion from RIFE's flow.

RIFE (Practical-RIFE, hzwer, MIT) estimates, for a time t between two frames,
the flow from the t picture to each frame (F_t->0, F_t->1) and an occlusion
mask, then warps and blends. These models stop before the warp: same inputs as
the models of make_rife_models.py (img0 [1,3,H,W] previous frame, img1
[1,3,H,W] current frame, timestep [1,1,H,W]), two outputs:
  flow [1,4,H,W]: F_t->0 (x, y) then F_t->1 (x, y), in input pixels;
  mask [1,1,H,W]: the blend weight of the warped img0 (sigmoid).
The final flow is the Add that takes the last IFBlock's flow delta (first
output of the last /blockN/Split), the mask is the graph's Sigmoid. Nodes that
only serve the warp and blend are removed. fp16 as in make_rife_models.py.

Veyra runs them at a quarter of the work resolution with t = 0.5 and turns
the two flows into current->previous motion vectors (RifeFlowToMotion.hlsl).

  python make_rife_flow_models.py <dir holding rife_v2/> <output dir> rife_v4.26 [...]
  -> <output dir>/rife_v4.26_flow_fp16.onnx

Requires onnx and onnxruntime 1.24.x (see make_rife_models.py).
"""
import os
import sys
import tempfile

import onnx
from onnx import helper

import make_rife_models


def flow_outputs(graph):
    produced = {o: n for n in graph.node for o in n.output}
    splits = [n for n in graph.node if n.op_type == "Split" and n.name.startswith("/block")]
    if not splits:
        raise SystemExit("no IFBlock Split in the graph")
    delta = splits[-1].output[0]
    adds = [n for n in graph.node if n.op_type == "Add" and delta in n.input]
    sigmoids = [n for n in graph.node if n.op_type == "Sigmoid"]
    if len(adds) != 1 or len(sigmoids) != 1:
        raise SystemExit(f"unexpected graph: {len(adds)} final flow Add, {len(sigmoids)} Sigmoid")
    if produced.get(sigmoids[0].input[0]) is not splits[-1]:
        raise SystemExit("the Sigmoid does not take the last IFBlock's mask")
    return adds[0].output[0], sigmoids[0].output[0]


def prune(graph, outputs):
    produced = {o: n for n in graph.node for o in n.output}
    needed, stack = set(), list(outputs)
    while stack:
        node = produced.get(stack.pop())
        if node is None or id(node) in needed:
            continue
        needed.add(id(node))
        stack.extend(i for i in node.input if i)
    keep = [n for n in graph.node if id(n) in needed]
    del graph.node[:]
    graph.node.extend(keep)
    used = {i for n in keep for i in n.input}
    init = [t for t in graph.initializer if t.name in used]
    del graph.initializer[:]
    graph.initializer.extend(init)


def flow_model(source, target):
    model = onnx.load(source)
    graph = model.graph
    flow, mask = flow_outputs(graph)
    # Rename through Identity so the backend binds stable names.
    graph.node.extend([helper.make_node("Identity", [flow], ["flow"], name="veyra_flow"),
                       helper.make_node("Identity", [mask], ["mask"], name="veyra_mask")])
    del graph.output[:]
    dims = [d.dim_param or d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    graph.output.extend([helper.make_tensor_value_info("flow", onnx.TensorProto.FLOAT, [dims[0], 4] + dims[2:]),
                         helper.make_tensor_value_info("mask", onnx.TensorProto.FLOAT, [dims[0], 1] + dims[2:])])
    prune(graph, ["flow", "mask"])
    del graph.value_info[:]
    onnx.checker.check_model(model)
    # The fp32 step stays out of the output directory (CI copies its *.onnx).
    with tempfile.TemporaryDirectory() as scratch:
        fp32 = os.path.join(scratch, "flow_fp32.onnx")
        onnx.save(model, fp32)
        make_rife_models.convert(fp32, target)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    for name in sys.argv[3:]:
        out = f"{sys.argv[2]}/{name}_flow_fp16.onnx"
        flow_model(f"{sys.argv[1]}/rife_v2/{name}.onnx", out)
        print(f"{name}_flow_fp16.onnx")


if __name__ == "__main__":
    main()
