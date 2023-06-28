import array

from mlir.ir import (RankedTensorType,
                     F32Type, 
                     DenseElementsAttr, 
                     FloatAttr)
from mlir.dialects import (arith,
                           linalg,
                           tosa,
                           math)


def _getConstantTensorOp(value: float, sizes: list[int]):
  f32 = F32Type.get()
  constantTensorType = RankedTensorType.get(sizes, f32)
  constantElementAttr = FloatAttr.get(f32, value)
  constantTensorAttr = DenseElementsAttr.get_splat(constantTensorType, constantElementAttr)
  op = arith.ConstantOp(constantTensorType, constantTensorAttr)
  return op


def _broadcast_shape(tensor_input1, tensor_input2):
  shp1 = RankedTensorType(tensor_input1.type).shape
  shp2 = RankedTensorType(tensor_input2.type).shape
  shp1_list = list(shp1)
  shp2_list = list(shp2)
  swapped = False
  if len(shp1_list) < len(shp2_list):
    shp1_list, shp2_list = shp2_list, shp1_list
    swapped = True
  while len(shp2_list) < len(shp1_list):
    shp2_list.insert(0, 1)
  for idx, (dim1, dim2) in enumerate(zip(shp1_list, shp2_list)):
    shp1_list[idx] = shp2_list[idx] = max(dim1, dim2)

  return shp1_list


def AddOp(node, symbolTable):
  input1 = symbolTable.get(str(node._args[0])) 
  input2 = symbolTable.get(str(node._args[1]))
  broadcasted_shp = _broadcast_shape(input1, input2)
  sizes = broadcasted_shp
  f32 = F32Type.get()   
  addResultTensorType = RankedTensorType.get(sizes, f32)
  op = tosa.AddOp(addResultTensorType, input1, input2)
  return op


def MatmulOp(node, symbolTable):
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
    initResult = arith.ConstantOp(tensorType, attr).result
    # Generate matmul operation.
    op = linalg.matmul(input1, input2, outs=[initResult])
    return op
  elif len(shp1) == 3:
    size0 = shp1[0]
    size1 = shp1[1]
    size2 = shp2[2]
    sizes = [size0, size1, size2]
    tensorType = RankedTensorType.get(sizes, f32)
    attr = DenseElementsAttr.get_splat(tensorType, zeroElement)
    initResult = arith.ConstantOp(tensorType, attr)
    op = linalg.batch_matmul(input1, input2, outs=[initResult.result])
    return op
  else:
    raise NotImplementedError


def TransposeOp(node, symbolTable):
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
  return op


def SubOp(node, symbolTable):
  input1 = symbolTable.get(str(node._args[0])) 
  input2 = symbolTable.get(str(node._args[1]))
  op = arith.SubFOp(input1, input2)
  return op


def MulOp(node, symbolTable):
  input1 = symbolTable.get(str(node._args[0])) 
  input2 = symbolTable.get(str(node._args[1]))
  op = arith.MulFOp(input1, input2)
  return op


def TrueDivOp(node, symbolTable):
  input1 = symbolTable.get(str(node._args[0]))
  input2 = symbolTable.get(str(node._args[1]))
  op = arith.DivFOp(input1, input2)
  return op


def OnesOp(node, symbolTable):
  args = node._args
  # flatten the size tuple
  while isinstance(args[0], tuple):
    args = args[0]
  sizes = list(args)
  op = _getConstantTensorOp(1.0, sizes)
  return op


def AddmmOp(node, symbolTable):
  input_ = symbolTable.get(str(node._args[0]))
  mat1 = symbolTable.get(str(node._args[1]))
  mat2 = symbolTable.get(str(node._args[2]))
  mmOp = MatmulOp(node, symbolTable)
  addOp = arith.AddFOp(input_, mmOp.result)
  return addOp


def ErfOp(node, symbolTable):
  input_ = symbolTable.get(str(node._args[0]))
  erfOp = math.ErfOp(input_)
  return erfOp


def TanhOp(node, symbolTable):
  input_ = symbolTable.get(str(node._args[0]))
  tanhOp = math.TanhOp(input_)
  return tanhOp


def ExpOp(node, symbolTable):
  input_ = symbolTable.get(str(node._args[0]))
  expOp = math.ExpOp(input_)
  return expOp


def RsqrtOp(node, symbolTable):
  input_ = symbolTable.get(str(node._args[0]))
  rsqrtOp = math.RsqrtOp(input_)
  return rsqrtOp


# add, addmm, amax, bmm, clone, convert_element_type
# div, embedding, erf, exp, expand, getitem, gt, inductor_lookup_seed
# inductor_random, inductor_seeds, mul, permute, reshape, rsqrt
# select, slice, sub, tanh, unsqueeze, var_mean
OpCodeGen = {
  'add': AddOp,
  'iadd': AddOp,
  'matmul': MatmulOp,
  'transpose': TransposeOp,
  'sub': SubOp,
  'mul': MulOp,
  'truediv': TrueDivOp,
  'ones': OnesOp,
}
