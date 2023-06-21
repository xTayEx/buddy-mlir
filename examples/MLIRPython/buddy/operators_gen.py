import array

from mlir.ir import (RankedTensorType,
                     F32Type, 
                     DenseElementsAttr, 
                     FloatAttr,
                     Value,
                     Operation)
from mlir.dialects import (arith,
                           linalg,
                           tosa,
                           tensor)

def GenAddOp(node, symbolTable):
  input1 = symbolTable.get(str(node._args[0])) 
  input2 = symbolTable.get(str(node._args[1]))
  op = arith.AddFOp(input1, input2)
  symbolTable[str(node.name)] = op

def GenMatmulOp(node, symbolTable):
  # Get two input values.
  input1 = symbolTable.get(str(node._args[0]))
  input2 = symbolTable.get(str(node._args[1]))
  shp1 = RankedTensorType(input1.type).shape
  shp2 = RankedTensorType(input2.type).shape
  assert len(shp1) == len(shp2)
  f32 = F32Type.get()
  zeroElement = FloatAttr.get(f32, 0.0)
  if len(shp1) == 2:
    # Infer the output sizes.
    size1 = shp1[0]
    size2 = shp2[1]
    sizes = [size1, size2]
    # Generate an output tensor for matmul operation.
    # For example:
    # `arith.constant dense<0.000000e+00> : tensor<3x3xf32>`
    tensorType = RankedTensorType.get(sizes, f32)
    attr = DenseElementsAttr.get_splat(tensorType, zeroElement)
    initResult = arith.ConstantOp(tensorType, attr)
    # Generate matmul operation.
    op = linalg.matmul(input1, input2, outs=[initResult.result])
    symbolTable[str(node.name)] = op
  elif len(shp1) == 3:
    size0 = shp1[0]
    size1 = shp1[1]
    size2 = shp2[2]
    sizes = [size0, size1, size2]
    tensorType = RankedTensorType.get(sizes, f32)
    attr = DenseElementsAttr.get_splat(tensorType, zeroElement)
    initResult = arith.ConstantOp(tensorType, attr)
    op = linalg.batch_matmul(input1, input2, outs=[initResult.result])
    symbolTable[str(node.name)] = op
  else:
    raise NotImplementedError

def GenTransposeOp(node, symbolTable):
  if node.target.__name__ == "transpose":
    input_tensor = symbolTable.get(str(node._args[0]))
    size1 = RankedTensorType(input_tensor.type).shape[0]
    size2 = RankedTensorType(input_tensor.type).shape[1]
    sizes = [size2, size1]

    f32 = F32Type.get()
    transResultTensorType = RankedTensorType.get(sizes, f32)
    permTensorType = RankedTensorType.get([2], f32)
    permContent = memoryview(array.array('i', [1, 0]))
    permAttr = DenseElementsAttr.get(permContent)
    perm = arith.ConstantOp(permTensorType, permAttr)
    op = tosa.TransposeOp(transResultTensorType, input_tensor, perm)
    symbolTable[str(node.name)] = op

def GenSubOp(node, symbolTable):
  input1 = symbolTable.get(str(node._args[0])) 
  input2 = symbolTable.get(str(node._args[1]))
  op = arith.SubFOp(input1, input2)
  symbolTable[str(node.name)] = op

# iadd means in-place add!
def GenIaddOp(node, symbolTable):
  input1 = symbolTable.get(str(node._args[0])) 
  input2 = symbolTable.get(str(node._args[1]))
  op = arith.AddIOp(input1, input2)
  symbolTable[str(node.name)] = op

def GenMulOp(node, symbolTable):
  input1 = symbolTable.get(str(node._args[0])) 
  input2 = symbolTable.get(str(node._args[1]))
  op = arith.MulFOp(input1, input2)
  symbolTable[str(node.name)] = op

def GenTrueDivOp(node, symbolTable):
  input1 = symbolTable.get(str(node._args[0])) 
  input2 = symbolTable.get(str(node._args[1]))
  op = arith.DivFOp(input1, input2)
  symbolTable[str(node.name)] = op

def GenOnesOp(node, symbolTable):
  args = node._args
  # flatten the size tuple
  while isinstance(args[0], tuple):
    args = args[0]
  sizes = list(args)
  f32 = F32Type.get()
  # create the all-one tensor
  allOneTensorType = RankedTensorType.get(sizes, f32)
  oneElementAttr = FloatAttr.get(f32, 1.0)
  allOneAttr = DenseElementsAttr.get_splat(allOneTensorType, oneElementAttr)
  op = arith.ConstantOp(allOneTensorType, allOneAttr)
  symbolTable[str(node.name)] = op

OpCodeGen = {
  'add': GenAddOp,
  'matmul': GenMatmulOp,
  'transpose': GenTransposeOp,
  'sub': GenSubOp,
  'mul': GenMulOp,
  'truediv': GenTrueDivOp,
  'iadd': GenIaddOp,
  'ones': GenOnesOp
}
