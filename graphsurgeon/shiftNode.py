
from collections import OrderedDict

import numpy as np
import onnx
import onnx_graphsurgeon as gs

tensor0 = gs.Variable("tensor0", np.float32, ["B", 3, 64, 64])
tensor1 = gs.Variable("tensor1", np.float32, None)
tensor2 = gs.Variable("tensor2", np.float32, None)
tensor3 = gs.Variable("tensor3", np.float32, None)
constant0 = gs.Constant(name="constant0", values=np.ones(shape=[1, 1, 1, 1], dtype=np.float32))

node0 = gs.Node("Identity", "myIdentity0", inputs=[tensor0], outputs=[tensor1])
node1 = gs.Node("Add", "myAdd", inputs=[tensor1, constant0], outputs=[tensor2])
node2 = gs.Node("Identity", "myIdentity1", inputs=[tensor2], outputs=[tensor3])

graph = gs.Graph(nodes=[node0, node1, node2], inputs=[tensor0], outputs=[tensor3])
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "model-04-01.onnx")

del graph

# replace node by edit the operator type
graph = gs.import_onnx(onnx.load("model-04-01.onnx"))  # load the graph from ONNX file
for node in graph.nodes:
    if node.op == "Add" and node.name == "myAdd":
        node.op = "Sub"
        node.name = "mySub"  # it's OK to change the name of the node or not

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "model-04-02.onnx")

del graph

# repalce node by inserting new node
graph = gs.import_onnx(onnx.load("model-04-01.onnx"))  # load the graph from ONNX file
for node in graph.nodes:
    if node.op == "Add" and node.name == "myAdd":
        newNode = gs.Node("Sub", "mySub", inputs=node.inputs, outputs=node.outputs)
        graph.nodes.append(newNode)
        #node.outputs = [] #如果注释了该行，则会得到第四个图，这个图是错误的，因为Add和Sub的输出都是tensor2，因此需要将Add的输出置空，由于Add没有输出，gs会自动删除Add层。

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "model-04-03.onnx")

