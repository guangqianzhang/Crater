
from collections import OrderedDict

import numpy as np
import onnx
import onnx_graphsurgeon as gs

tensor0 = gs.Variable("tensor0", np.float32, ["B", 3, 64, 64])
tensor1 = gs.Variable("tensor1", np.float32, ["B", 3, 64, 64])
tensor2 = gs.Variable("tensor2", np.float32, ["B", 3, 64, 64])
tensor3 = gs.Variable("tensor3", np.float32, ["B", 3, 64, 64])
constant0 = gs.Constant(name="constant0", values=np.ones(shape=[1, 1, 1, 1], dtype=np.float32))

node0 = gs.Node("Identity", "myIdentity0", inputs=[tensor0], outputs=[tensor1])
node1 = gs.Node("Add", "myAdd", inputs=[tensor1, constant0], outputs=[tensor2])
node2 = gs.Node("Identity", "myIdentity1", inputs=[tensor2], outputs=[tensor3])

graph = gs.Graph(nodes=[node0, node1, node2], inputs=[tensor0], outputs=[tensor3])
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "model-08-01.onnx")

del graph

# mark some tensors in original graph as input / output tensors so that isolate a subgraph 
# 将原图中的一些张量标记为输入/输出张量，以隔离子图
graph = gs.import_onnx(onnx.load("model-08-01.onnx"))
for node in graph.nodes:
    if node.op == "Add" and node.name == "myAdd":
        graph.inputs = [node.inputs[0]]
        graph.outputs = [node.outputs[0]]

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "model-08-02.onnx")

