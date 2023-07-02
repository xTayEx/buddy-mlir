import array

from mlir.ir import (RankedTensorType,
                     F32Type,
                     IntegerType,
                     DenseElementsAttr,
                     DenseIntElementsAttr,
                     FloatAttr,
                     IntegerAttr)
from mlir.dialects import (arith,
                           tosa,
                           tensor,
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
  if len(shp1_list) < len(shp2_list):
    shp1_list, shp2_list = shp2_list, shp1_list
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


def GtOp(node, symbolTable):
  input1 = symbolTable.get(str(node._args[0]))
  input2 = symbolTable.get(str(node._args[1]))
  broadcasted_shp = _broadcast_shape(input1, input2)
  sizes = broadcasted_shp
  f32 = F32Type.get()
  addResultTensorType = RankedTensorType.get(sizes, f32)
  op = tosa.GreaterOp(addResultTensorType, input1, input2)
  return op


def SubOp(node, symbolTable):
  input1 = symbolTable.get(str(node._args[0])) 
  input2 = symbolTable.get(str(node._args[1]))
  op = arith.SubFOp(input1, input2)
  return op


def MulOp(node, symbolTable):
  input1 = symbolTable.get(str(node._args[0])) 
  input2 = symbolTable.get(str(node._args[1]))
  broadcasted_shp = _broadcast_shape(input1, input2)
  sizes = broadcasted_shp
  f32 = F32Type.get()
  addResultTensorType = RankedTensorType.get(sizes, f32)
  op = tosa.MulOp(addResultTensorType, input1, input2)
  return op


def DivOp(node, symbolTable):
  input1 = symbolTable.get(str(node._args[0]))
  input2 = symbolTable.get(str(node._args[1]))
  broadcasted_shp = _broadcast_shape(input1, input2)
  sizes = broadcasted_shp
  f32 = F32Type.get()
  divResultTensorType = RankedTensorType.get(sizes, f32)
  op = tosa.DivOp(divResultTensorType, input1, input2)
  return op


def ErfOp(node, symbolTable):
  input_ = symbolTable.get(str(node._args[0]))
  erfOp = math.ErfOp(input_)
  return erfOp


def TanhOp(node, symbolTable):
  input1 = symbolTable.get(str(node._args[0]))
  sizes = RankedTensorType(input1.type).shape
  f32 = F32Type.get()
  tanhResultTensorType = RankedTensorType.get(sizes, f32)
  tanhOp = tosa.TanhOp(tanhResultTensorType, input1)
  return tanhOp


def ExpOp(node, symbolTable):
  input1 = symbolTable.get(str(node._args[0]))
  sizes = RankedTensorType(input1.type).shape
  f32 = F32Type.get()
  expResultTensorType = RankedTensorType.get(sizes, f32)
  expOp = tosa.ExpOp(expResultTensorType, input1)
  return expOp


def RsqrtOp(node, symbolTable):
  input1 = symbolTable.get(str(node._args[0]))
  sizes = RankedTensorType(input1.type).shape
  f32 = F32Type.get()
  rsqrtResultTensorType = RankedTensorType.get(sizes, f32)
  rsqrtOp = tosa.RsqrtOp(rsqrtResultTensorType, input1)
  return rsqrtOp


def AmaxOp(node, symbolTable):
  input1 = symbolTable.get(str(node._args[0]))
  dim = symbolTable.get(str(node._args[1]))[0]
  signlessType = IntegerType.get_signless()
  dimAttr = IntegerAttr.get(signlessType, dim)
  op = tosa.ReduceMaxOp(input1, dimAttr)
  return op


def ReshapeOp(node, symbolTable):
  input1 = symbolTable.get(str(node._args[0]))
  newShape = symbolTable.get(str(node._args[1]))
  newShapeContent = array.array('Q', newShape)
  newShapeContent = memoryview(newShapeContent)
  op = tosa.ReshapeOp(input1, newShapeContent)
  return op


def UnsqueezeOp(node, symbolTable):
  inputTensor = symbolTable.get(str(node._args[0]))
  dim = symbolTable.get(str(node._args[1]))
  sizes = RankedTensorType(inputTensor.type).shape
  sizes.insert(dim, 1)
  newShapeContent = array.array('Q', sizes)
  newShapeContent = memoryview(newShapeContent)
  op = tosa.ReshapeOp(inputTensor, newShapeContent)
  return op


# TODO: Not sure about the argument type. May fail to run.
def GetItemOp(node, symbolTable):
  inputTensor = symbolTable.get(str(node._args[0]))
  indices = list(symbolTable.get(str(node._args[1])))
  op = tensor.ExtractOp(inputTensor, indices)
  return op


# add, addmm, amax, bmm, clone, convert_element_type
# div, embedding, erf, exp, expand, getitem, gt, inductor_lookup_seed
# inductor_random, inductor_seeds, mul, permute, reshape, rsqrt
# select, slice, sub, tanh, unsqueeze, var_mean
OpCodeGen = {
}
