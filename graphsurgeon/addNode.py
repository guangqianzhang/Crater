from collections import OrderedDict

import numpy as np
import onnx
import onnx_graphsurgeon as gs

tensor0 = gs.Variable("tensor0", np.float32, ["B", 3, 64, 64])
tensor1 = gs.Variable("tensor1", np.float32, None)
tensor2 = gs.Variable("tensor2", np.float32, None)
#gs.Node(节点类型，节点名称，输入，输出)
node0 = gs.Node("Identity", "myIdentity0", inputs=[tensor0], outputs=[tensor1])
node1 = gs.Node("Identity", "myIdentity1", inputs=[tensor1], outputs=[tensor2])

graph = gs.Graph(nodes=[node0, node1], inputs=[tensor0], outputs=[tensor2])
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "model-02-01.onnx")

del graph

graph = gs.import_onnx(onnx.load("model-02-01.onnx"))  # load the graph from ONNX file
for node in graph.nodes:
    if node.op == "Identity" and node.name == "myIdentity0":  # find the place we want to add ndoe
        constant0 = gs.Constant(name="constant0", values=np.ones(shape=[1, 1, 1, 1], dtype=np.float32))  # construct the new variable and node
        tensor3 = gs.Variable("tensor3", np.float32, None)
        newNode = gs.Node("Add", "myAdd", inputs=[node.outputs[0], constant0], outputs=[tensor3])

        graph.nodes.append(newNode)  # REMEMBER to add the new node into the grap
        index = node.o().inputs.index(node.outputs[0])  # find the next node
        node.o().inputs[index] = tensor3  # replace the input tensor of next node as the new tensor
#注意：node.o()为node的输出node；node.outputs[]为node的输出Tensor
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "model-02-02.onnx")

