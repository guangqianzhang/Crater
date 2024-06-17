
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def markGraphOutput(graph, lNode, bMarkOutput=True, bMarkInput=False, lMarkOutput=None, lMarkInput=None, bRemoveOldOutput=True):
    # graph:            The ONNX graph for edition
    # lNode:            The list of nodes we want to mark as output
    # bMarkOutput:      Whether to mark the output tensor(s) of the nodes in the lNode
    # bMarkInput:       Whether to mark the input tensor(s) of the nodes in the lNode
    # lMarkOutput:      The index of output tensor(s) of the node are marked as output, only available when len(lNode) == 1
    # lMarkInput:       The index of input tensor(s) of the node are marked as output, only available when len(lNode) == 1
    # bRemoveOldOutput: Whether to remove the original output of the network (cutting the graph to the node we want to mark to save ytime of building)

    # In most cases, using the first 4 parameters is enough, for example:
    #markGraphOutput(graph, ["/Conv"])                          # mark output tensor of the node "/Conv" as output
    #markGraphOutput(graph, ["/Conv"], False, True)             # mark input tensors of the node "/Conv" (input tensor + weight + bias) as output
    #markGraphOutput(graph, ["/TopK"], lMarkOutput=[1])         # mark the second output tensor of the node "/TopK" as output
    #markGraphOutput(graph, ["/Conv"], bRemoveOldOutput=False)  # mark output tensor of the node "/Conv" as output, and keep the original output of the network
    
# graph:用于编辑的ONNX图形
# lNode:我们想要标记为输出的节点列表
# bMarkOutput:是否标记lNode中节点的输出张量
# bMarkInput:是否在lNode中标记节点的输入张量
# lMarkOutput:节点的输出张量索引被标记为输出，仅当len(lNode) == 1时可用
# lMarkInput:节点的输入张量的索引被标记为输出，仅当len(lNode) == 1时可用
# bRemoveOldOutput:是否删除网络的原始输出(将图切割到我们想要标记的节点以节省构建时间)

#在大多数情况下，使用前4个参数就足够了，例如:
#markGraphOutput(graph， ["/Conv"]) #将节点"/Conv"的输出张量标记为输出
#markGraphOutput(graph， ["/Conv"]， False, True) #标记节点"/Conv"的输入张量(输入张量+权重+偏置)作为输出
#markGraphOutput(graph， ["/TopK"]， lMarkOutput=[1]) #将节点"/TopK"的第二个输出张量标记为输出
#markGraphOutput(graph， ["/Conv"]， bRemoveOldOutput=False) #将节点"/Conv"的输出张量标记为输出，保持网络的原始输出

    if bRemoveOldOutput:
        graph.outputs = []
    for node in graph.nodes:
        if node.name in lNode:
            if bMarkOutput:
                if lMarkOutput is None or len(lNode) > 1:
                    lMarkOutput = range(len(node.outputs))
                for index in lMarkOutput:
                    graph.outputs.append(node.outputs[index])
                    print("Mark node [%s] output tensor [%s]" % (node.name, node.outputs[index].name))
            if bMarkInput:
                if lMarkInput is None or len(lNode) > 1:
                    lMarkInput = range(len(node.inputs))
                for index in lMarkInput:
                    graph.outputs.append(node.inputs[index])
                    print("Mark node [%s] input  tensor [%s]" % (node.name, node.inputs[index].name))

    graph.cleanup().toposort()
    return len(lNode)

def addNode(graph, nodeType, prefix, number, inputList, attribution=None, suffix="", dtype=None, shape=None):
    # ONLY for the node with one output tensor!!

    # graph:        The ONNX graph for edition
    # nodeType:     The type of the node to add, for example, "Concat"
    # prefix:       Optimization type, for example "RemoveLoop"
    # number:       An incremental number to prevent duplicate names
    # inputlist:    The list of input tensors for the node
    # attribution:  The attribution dictionary of the node, for example, OrderedDict([('axis',0)])
    # suffix:       Extra name for marking the tensor, for example "bTensor"
    # dtype:        The data type of the output tensor (optional)
    # shape:        The shape of the output tensor (optional)

#只适用于只有一个输出张量的节点!!

# graph:用于编辑的ONNX图形
# nodeType:要添加的节点类型，例如“Concat”。
# prefix:优化类型，例如“RemoveLoop”
# number:一个增量数字，以防止重复的名称
# inputlist:节点的输入张量列表
# attribution:节点的属性字典，例如OrderedDict([('axis'，0)])
# suffix:用于标记张量的额外名称，例如“bTensor”
# dtype:输出张量的数据类型(可选)
# shape:输出张量的形状(可选)

    tensorName = prefix + "-V-" + str(number) + "-" + nodeType
    nodeName = prefix + "-N-" + str(number) + "-" + nodeType
    if attribution == None:
        attribution = OrderedDict()
    if len(suffix) > 0:
        tensorName += "-" + suffix

    tensor = gs.Variable(tensorName, dtype, shape)
    node = gs.Node(nodeType, nodeName, inputs=inputList, outputs=[tensor], attrs=attribution)
    graph.nodes.append(node)
    return tensor, number + 1






