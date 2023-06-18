import array

from mlir.ir import RankedTensorType, F32Type, DenseElementsAttr, FloatAttr
from mlir.dialects import arith, linalg, tosa

def GenAddOpCode(node, symbolTable):
  input1 = symbolTable.get(str(node._args[0])) 
  input2 = symbolTable.get(str(node._args[1]))
  op = arith.AddFOp(input1, input2)
  symbolTable[str(node.name)] = op

def GenMatmulOpCode(node, symbolTable):
  # Get two input values.
  input1 = symbolTable.get(str(node._args[0]))
  input2 = symbolTable.get(str(node._args[1]))
  shp1 = RankedTensorType(input1.type).shape
  shp2 = RankedTensorType(input2.type).shape
  assert len(shp1) == len(shp2)
  f32 = F32Type.get()
  zero_element = FloatAttr.get(f32, 0.0)
  if len(shp1) == 2:
    # Infer the output sizes.
    size1 = shp1[0]
    size2 = shp2[1]
    sizes = [size1, size2]
    # Generate an output tensor for matmul operation.
    # For example:
    # `arith.constant dense<0.000000e+00> : tensor<3x3xf32>`
    tensor_type = RankedTensorType.get(sizes, f32)
    attr = DenseElementsAttr.get_splat(tensor_type, zero_element)
    init_result = arith.ConstantOp(tensor_type, attr)
    # Generate matmul operation.
    op = linalg.matmul(input1, input2, outs=[init_result.result])
    symbolTable[str(node.name)] = op
  elif len(shp1) == 3:
    size0 = shp1[0]
    size1 = shp1[1]
    size2 = shp2[2]
    sizes = [size0, size1, size2]
    tensor_type = RankedTensorType.get(sizes, f32)
    attr = DenseElementsAttr.get_splat(tensor_type, zero_element)
    init_result = arith.ConstantOp(tensor_type, attr)
    op = linalg.batch_matmul(input1, input2, outs=[init_result.result])
    symbolTable[str(node.name)] = op
  else:
    raise NotImplementedError

def GenTransposeOpCode(node, symbolTable):
  if node.target.__name__ == "transpose":
    input_tensor = symbolTable.get(str(node._args[0]))
    size1 = RankedTensorType(input_tensor.type).shape[0]
    size2 = RankedTensorType(input_tensor.type).shape[1]
    sizes = [size2, size1]

    f32 = F32Type.get()
    trans_result_tensor_type = RankedTensorType.get(sizes, f32)
    perm_tensor_type = RankedTensorType.get([2], f32)
    perm_content = memoryview(array.array('i', [1, 0]))
    perm_attr = DenseElementsAttr.get(perm_content)
    perm = arith.ConstantOp(perm_tensor_type, perm_attr)
    op = tosa.TransposeOp(trans_result_tensor_type, input_tensor, perm)
    symbolTable[str(node.name)] = op

OpCodeGen = {
  'add': GenAddOpCode,
  'matmul': GenMatmulOpCode,
  'transpose': GenTransposeOpCode
}
