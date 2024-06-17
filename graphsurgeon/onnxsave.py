from collections import OrderedDict

import numpy as np
import onnx
import onnx_graphsurgeon as gs
#定义节点输入的变量
tensor0 = gs.Variable("tensor0", np.float32, ["B", 3, 64, 64])  # define tensor (variable in ONNX)
tensor1 = gs.Variable("tensor1", np.float32, None)  # type or shape of the intermediate tensors can be None
tensor2 = gs.Variable("tensor2", np.float32, None)
tensor3 = gs.Variable("tensor3", np.float32, None)
#定义节点输入的常量
constant0 = gs.Constant(name="constant0", values=np.ones(shape=[1, 3, 3, 3], dtype=np.float32))  # define constant tensor
constant1 = gs.Constant(name="constant1", values=np.ones(shape=[1], dtype=np.float32))
#定义节点
node0 = gs.Node("Conv", "myConv", inputs=[tensor0, constant0], outputs=[tensor1])  # defione node
node0.attrs = OrderedDict([["dilations", [1, 1]], ["kernel_shape", [3, 3]], ["pads", [1, 1, 1, 1]], ["strides", [1, 1]]])  # attribution of the node
node1 = gs.Node("Add", "myAdd", inputs=[tensor1, constant1], outputs=[tensor2])
node2 = gs.Node("Relu", "myRelu", inputs=[tensor2], outputs=[tensor3])
#定义Graph
graph = gs.Graph(nodes=[node0, node1, node2], inputs=[tensor0], outputs=[tensor3])  # define graph
#topsort:有向图的拓扑排序
graph.cleanup().toposort()  # clean the graph before saving as ONNX file
onnx.save(gs.export_onnx(graph), "model-01.onnx")