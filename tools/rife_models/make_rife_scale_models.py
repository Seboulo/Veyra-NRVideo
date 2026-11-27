# SPDX-License-Identifier: GPL-3.0-only
"""Build half-resolution-flow RIFE models for RifeDmlBackend (VEYRA_TEST_RIFE_MODEL).

Practical-RIFE's --scale 0.5 ("recommended for 4K") runs every IFBlock of the flow
network at half the resolution (scale_list 16,8,4,2,1 -> 32,16,8,4,2) while the
final backward warp and merge stay at full resolution. vs-mlrt's ONNX exports fix
scale 1 in the graph; this script doubles the scale of every block:
  - blocks 0-3: input Resize 1/s -> 1/(2s), output Resize s -> 2s, flow * s -> * 2s,
    and the incoming flow * 1/s -> * 1/(2s);
  - block 4 ran at scale 1, so the exporter left out its resizes and multipliers:
    they are inserted (input * 1/2 and Resize 1/2, output Resize 2 and flow * 2).
At 2560x1440 RIFE 4.26 drops from 84 to 30 GMAC (its encoder stays at full
resolution). Tensors need padding to 128 (RifeDmlBackend: "_s05" in the name).

Input: the v2-layout ONNX files of vs-mlrt's external-models release (see
make_rife_models.py). Output: <name>_s05_fp16.onnx (and <name>_fp16.onnx for models
make_rife_models.py does not build), with that script's conversion.

  python make_rife_scale_models.py <dir holding rife_v2/> <output dir> rife_v4.26 [rife_v4.25_lite ...]

Requires onnx and onnxruntime 1.24.x (the float16 tool of 1.29 writes duplicate
node and tensor names, which ONNX Runtime refuses to load).
"""
import os
import sys
import tempfile

import numpy as np
import onnx
from onnx import helper, numpy_helper

import make_rife_models


def half_scale(source, target):
    model = onnx.load(source)
    graph = model.graph
    uses = {}
    for n in graph.node:
        for i in n.input:
            uses[i] = uses.get(i, 0) + 1
    producer = {o: n for n in graph.node for o in n.output}
    constant = {n.output[0]: n for n in graph.node if n.op_type == 'Constant'}

    def scale(name, factor, index=None):
        node = constant[name]
        if uses.get(name, 0) != 1:
            raise ValueError(f'{name} is shared')
        value = numpy_helper.to_array(node.attribute[0].t).copy()
        if index is None:
            value = value * factor
        else:
            value[index] = value[index] * factor
        node.attribute[0].t.CopyFrom(numpy_helper.from_array(value, name=node.attribute[0].t.name))
        return value

    for b in range(4):
        scale(f'/block{b}/Constant_output_0', 0.5, [2, 3])
        scale(f'/block{b}/Constant_1_output_0', 2.0, [2, 3])
        scale(f'/block{b}/Constant_2_output_0', 2.0)
    for b in range(1, 4):
        resize = next(n for n in graph.node if n.name == f'/block{b}/Resize')
        flow_in = producer[producer[resize.input[0]].input[-1]]   # flow * 1/scale enters the concat last
        if flow_in.op_type != 'Mul':
            raise ValueError(f'block{b}: incoming flow is not scaled by a Mul')
        factor = next(i for i in flow_in.input if i in constant)
        if not 0 < float(numpy_helper.to_array(constant[factor].attribute[0].t)) < 1:
            raise ValueError(f'block{b}: unexpected flow factor')
        scale(factor, 0.5)

    first = next(n for n in graph.node if n.name.startswith('/block4/conv0') and n.op_type == 'Conv')
    concat = producer[first.input[0]]
    if concat.op_type != 'Concat' or producer[concat.input[-1]].op_type != 'Add':
        raise ValueError('block4 input is not the expected Concat(..., flow)')
    split = next(n for n in graph.node if n.name == '/block4/Split')

    def const(name, value):
        return helper.make_node('Constant', [], [name], name=name, value=numpy_helper.from_array(np.asarray(value, np.float32), name=name))
    added = {
        'before': [const('/veyra/b4_down', [1, 1, .5, .5]), const('/veyra/b4_up', [1, 1, 2, 2]), const('/veyra/b4_flow_in', .5),
                   const('/veyra/b4_flow_out', np.array([2] * 4 + [1] * 9, np.float32).reshape(1, 13, 1, 1)),
                   helper.make_node('Mul', [concat.input[-1], '/veyra/b4_flow_in'], ['/veyra/b4_flow'], name='/veyra/b4_flow_mul')],
        'after': [helper.make_node('Resize', [concat.output[0], '', '/veyra/b4_down'], ['/veyra/b4_x'], name='/veyra/b4_resize_down',
                                   mode='linear', coordinate_transformation_mode='half_pixel')],
    }
    concat.input[len(concat.input) - 1] = '/veyra/b4_flow'
    first.input[0] = '/veyra/b4_x'
    nodes = []
    for n in graph.node:
        if n is concat:
            nodes += added['before'] + [n] + added['after']
            continue
        if n is split:
            nodes += [helper.make_node('Resize', [n.input[0], '', '/veyra/b4_up'], ['/veyra/b4_up_out'], name='/veyra/b4_resize_up',
                                       mode='linear', coordinate_transformation_mode='half_pixel'),
                      helper.make_node('Mul', ['/veyra/b4_up_out', '/veyra/b4_flow_out'], ['/veyra/b4_out'], name='/veyra/b4_flow_out_mul')]
            n.input[0] = '/veyra/b4_out'
        nodes.append(n)
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, target)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    source, output, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    with tempfile.TemporaryDirectory() as work:
        for name in names:
            path = f'{source}/rife_v2/{name}.onnx'
            if name not in make_rife_models.MODELS:
                make_rife_models.convert(path, f'{output}/{name}_fp16.onnx')
                print(f'{name}_fp16.onnx')
            half = os.path.join(work, f'{name}_s05.onnx')
            half_scale(path, half)
            make_rife_models.convert(half, f'{output}/{name}_s05_fp16.onnx')
            print(f'{name}_s05_fp16.onnx')


if __name__ == '__main__':
    main()
