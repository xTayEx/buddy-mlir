import array

from mlir.ir import (RankedTensorType,
                     F32Type,
                     IntegerType,
                     FloatAttr,
                     DenseI64ArrayAttr,
                     DenseElementsAttr,
                     IntegerAttr)
from mlir.dialects import (arith,
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
  if len(shp1) < len(shp2):
    shp1, shp2 = shp2, shp1
  while len(shp2) < len(shp1):
    shp2.insert(0, 1)
  for idx, (dim1, dim2) in enumerate(zip(shp1, shp2)):
    shp1[idx] = shp2[idx] = max(dim1, dim2)

  return shp1


def AddOp(node, symbolTable):
  input1 = symbolTable.get(str(node.args[0]))
  input2 = symbolTable.get(str(node.args[1]))
  broadcasted_shp = _broadcast_shape(input1, input2)
  sizes = broadcasted_shp
  f32 = F32Type.get()   
  addResultTensorType = RankedTensorType.get(sizes, f32)
  op = tosa.AddOp(addResultTensorType, input1, input2)
  return op


def GtOp(node, symbolTable):
  input1 = symbolTable.get(str(node.args[0]))
  input2 = symbolTable.get(str(node.args[1]))
  broadcasted_shp = _broadcast_shape(input1, input2)
  sizes = broadcasted_shp
  f32 = F32Type.get()
  addResultTensorType = RankedTensorType.get(sizes, f32)
  op = tosa.GreaterOp(addResultTensorType, input1, input2)
  return op


def SubOp(node, symbolTable):
  input1 = symbolTable.get(str(node.args[0]))
  input2 = symbolTable.get(str(node.args[1]))
  op = arith.SubFOp(input1, input2)
  return op


def MulOp(node, symbolTable):
  input1 = symbolTable.get(str(node.args[0]))
  input2 = symbolTable.get(str(node.args[1]))
  broadcasted_shp = _broadcast_shape(input1, input2)
  sizes = broadcasted_shp
  f32 = F32Type.get()
  addResultTensorType = RankedTensorType.get(sizes, f32)
  op = tosa.MulOp(addResultTensorType, input1, input2)
  return op


def DivOp(node, symbolTable):
  input1 = symbolTable.get(str(node.args[0]))
  input2 = symbolTable.get(str(node.args[1]))
  broadcasted_shp = _broadcast_shape(input1, input2)
  sizes = broadcasted_shp
  f32 = F32Type.get()
  divResultTensorType = RankedTensorType.get(sizes, f32)
  op = tosa.DivOp(divResultTensorType, input1, input2)
  return op


def ErfOp(node, symbolTable):
  input_ = symbolTable.get(str(node.args[0]))
  erfOp = math.ErfOp(input_)
  return erfOp


def TanhOp(node, symbolTable):
  input1 = symbolTable.get(str(node.args[0]))
  sizes = RankedTensorType(input1.type).shape
  f32 = F32Type.get()
  tanhResultTensorType = RankedTensorType.get(sizes, f32)
  tanhOp = tosa.TanhOp(tanhResultTensorType, input1)
  return tanhOp


def ExpOp(node, symbolTable):
  input1 = symbolTable.get(str(node.args[0]))
  sizes = RankedTensorType(input1.type).shape
  f32 = F32Type.get()
  expResultTensorType = RankedTensorType.get(sizes, f32)
  expOp = tosa.ExpOp(expResultTensorType, input1)
  return expOp


def RsqrtOp(node, symbolTable):
  input1 = symbolTable.get(str(node.args[0]))
  sizes = RankedTensorType(input1.type).shape
  f32 = F32Type.get()
  rsqrtResultTensorType = RankedTensorType.get(sizes, f32)
  rsqrtOp = tosa.RsqrtOp(rsqrtResultTensorType, input1)
  return rsqrtOp


def AmaxOp(node, symbolTable):
  input1 = symbolTable.get(str(node.args[0]))
  dim = symbolTable.get(str(node.args[1]))[0]
  signlessType = IntegerType.get_signless()
  dimAttr = IntegerAttr.get(signlessType, dim)
  op = tosa.ReduceMaxOp(input1, dimAttr)
  return op


def ReshapeOp(node, symbolTable):
  input1 = symbolTable.get(str(node.args[0]))
  newShape = symbolTable.get(str(node.args[1]))
  newShapeContent = array.array('i', newShape)
  newShapeContent = memoryview(newShapeContent)
  newShapeAttr = DenseElementsAttr.get(newShapeContent)
  op = tosa.ReshapeOp(input1, newShapeContent)
  return op


def UnsqueezeOp(node, symbolTable):
  inputTensor = symbolTable.get(str(node.args[0]))
  dim = symbolTable.get(str(node.args[1]))
  sizes = RankedTensorType(inputTensor.type).shape
  sizes.insert(dim, 1)
  newShapeContent = array.array('i', sizes)
  newShapeContent = memoryview(newShapeContent)
  newShapeAttr = DenseElementsAttr.get(newShapeContent)
  op = tosa.ReshapeOp(inputTensor, newShapeAttr)
  return op


def GetItemOp(node, symbolTable):
  inputTensor = symbolTable.get(str(node.args[0]))
  index = node.args[1]
  sizes = RankedTensorType(inputTensor.type).shape

  print(f"len(sizes): {len(sizes)}")
  startContent = array.array('Q', [index, 0])
  startContent = memoryview(startContent)
  startAttr = DenseI64ArrayAttr.get(startContent)

  stridesContent = array.array('Q', [1, sizes[1]])
  stridesContent = memoryview(stridesContent)
  strideAttr = DenseI64ArrayAttr.get(stridesContent)

  f32 = F32Type.get()
  resultSizes = [1, *sizes[1:]]
  resultType = RankedTensorType.get(resultSizes, f32)

  op = tosa.SliceOp(resultType, inputTensor, startAttr, strideAttr)
  return op


# 从文档以及print_tabular的结果来看，dim参数可能是None，此时需要对所有维度一同计算var和mean；
# 也可能是一个包裹在list中的int，指定具体的某个维度；也可能是包裹在list中的tuple，指定具体的某几个维度
def VarMeanOp(node, symbolTable):
  inputTensor = symbolTable.get(str(node.args[0]))
  dim = node.args[1]
  if isinstance(dim, list):
    dim = dim[0]


# add, addmm, amax, bmm, clone, convert_element_type
# div, embedding, erf, exp, expand, getitem, gt, inductor_lookup_seed
# inductor_random, inductor_seeds, mul, permute, reshape, rsqrt
# select, slice, sub, tanh, unsqueeze, var_mean
OpCodeGen = {
  'getitem': GetItemOp,
  'add': AddOp
}
