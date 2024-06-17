
import numpy as np
import onnx
import onnx_graphsurgeon as gs


# Create nodes
# use onnx_graphsurgeon.Graph.register() to register a function as a ndoe
@gs.Graph.register()
def add(self, a, b):
    return self.layer(op="Add", inputs=[a, b], outputs=["myAdd"])

@gs.Graph.register()
def mul(self, a, b):
    return self.layer(op="Mul", inputs=[a, b], outputs=["myMul"])
# 通用矩阵-矩阵乘法（General Matrix-Matrix Multiplication，GEMM）
# a：表示输入矩阵a ; b：表示输入矩阵b ; isTransposeA和isTransposeB：表示是否需要对矩阵a和矩阵b进行转置操作
@gs.Graph.register()
def gemm(self, a, b, isTransposeA=False, isTransposeB=False):
    attrs = {"transA": int(isTransposeA), "transB": int(isTransposeB)}
    return self.layer(op="Gemm", inputs=[a, b], outputs=["myGgemm"], attrs=attrs)

@gs.Graph.register()
def min(self, *args):
    return self.layer(op="Min", inputs=args, outputs=["myMin"])

@gs.Graph.register()
def max(self, *args):
    return self.layer(op="Max", inputs=args, outputs=["myMax"])

# mark version during register a function as a ndoe
@gs.Graph.register(opsets=[11])
def relu(self, a):
    return self.layer(op="Relu", inputs=[a], outputs=["myReLU"])

# register node with the same name but different version
@gs.Graph.register(opsets=[1])
def relu(self, a):
    raise NotImplementedError("This function has not been implemented!")

# Create graph
graph = gs.Graph(opset=11)
tensor0 = gs.Variable(name="tensor0", shape=[64, 64], dtype=np.float32)
tensor1 = gs.Constant(name="tensor1", values=np.ones(shape=(64, 64), dtype=np.float32))
#tensor1 = np.ones(shape=(64, 64), dtype=np.float32) # np.array can also be used but the name of the tensor will be created automatically by ONNX if so
tensor2 = gs.Constant(name="tensor2", values=np.ones((64, 64), dtype=np.float32) * 0.5)
tensor3 = gs.Constant(name="tensor3", values=np.ones(shape=[64, 64], dtype=np.float32))
tensor4 = gs.Constant(name="tensor4", values=np.array([3], dtype=np.float32))
tensor5 = gs.Constant(name="tensor5", values=np.array([-3], dtype=np.float32))

node0 = graph.gemm(tensor1, tensor0, isTransposeB=True)
node1 = graph.add(*node0, tensor2)
node2 = graph.relu(*node1)
node3 = graph.mul(*node2, tensor3)
node4 = graph.min(*node3, tensor4)
node5 = graph.max(*node4, tensor5)

graph.inputs = [tensor0]
graph.outputs = [node5[0]]

graph.inputs[0].dtype = np.dtype(np.float32)
graph.outputs[0].dtype = np.dtype(np.float32)

onnx.save(gs.export_onnx(graph), "model-09-01.onnx")

@gs.Graph.register()
def replaceWithClip(self, inputs, outputs):
    # remove the output tensor of the tail node and the input tensor of the head node
    # 去掉尾部节点的输出张量和头部节点的输入张量
    for inp in inputs:
        inp.outputs.clear()
    for out in outputs:
        out.inputs.clear()
    # inset new node
    return self.layer(op="Clip", inputs=inputs, outputs=outputs)

temp = graph.tensors()

# find the input / outpu tensor that we want to remove, and pass them to function replaceWithClip
inputs = [temp["myMul_6"], temp["tensor5"], temp["tensor4"]]
outputs = [temp["myMax_10"]]
graph.replaceWithClip(inputs, outputs)

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "model-09-02.onnx")

