import array
import math as builtin_math

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
    initResult = arith.ConstantOp(tensorType, attr).result
    # Generate matmul operation.
    op = linalg.matmul(input1, input2, outs=[initResult])
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
  symbolTable[str(node.name)] = _getConstantTensorOp(1.0, sizes)


def GenGeluEstimateOp(node, symbolTable):
  x = symbolTable.get(str(node._args[0]))
  shp = RankedTensorType(x.type).shape
  sizes = list(shp)

  halfConstantOp = _getConstantTensorOp(0.5, sizes)
  twoOverPiConstantOp = _getConstantTensorOp(2 / builtin_math.pi, sizes)
  oneConstantOp = _getConstantTensorOp(1.0, sizes)
  # 0.5 * (1 + tanh(sqrt(2/pi)))
  leftConstantOp = math.SqrtOp(twoOverPiConstantOp.result)
  leftConstantOp = math.TanhOp(leftConstantOp.result)
  leftConstantOp = arith.AddFOp(oneConstantOp.result, leftConstantOp.result)
  leftConstantOp = arith.MulFOp(halfConstantOp.result, leftConstantOp.result)
  leftOp = arith.MulFOp(leftConstantOp.result, x)

  magicNumberConstantOp = _getConstantTensorOp(0.044715, sizes)
  cubexOp = arith.MulFOp(x, x)
  cubexOp = arith.MulFOp(x, cubexOp.result)
  rightOp = arith.AddFOp(x, arith.MulFOp(magicNumberConstantOp.result, cubexOp.result))
  op = arith.MulFOp(leftOp.result, rightOp.result)
  symbolTable[str(node.name)] = op


def GenGeluNoEstimateOp(node, symbolTable):
  x = symbolTable.get(str(node._args[0]))
  shp = RankedTensorType(x.type).shape
  sizes = list(shp)

  halfConstantOp = _getConstantTensorOp(0.5, sizes)
  sqrtTwoReciprocalConstantOp = _getConstantTensorOp(1 / builtin_math.sqrt(2), sizes)
  oneConstantOp = _getConstantTensorOp(1.0, sizes)
  erfOp = math.ErfOp(sqrtTwoReciprocalConstantOp.result)
  phiOp = arith.MulFOp(
    halfConstantOp.result,
    arith.AddFOp(oneConstantOp.result, erfOp.result)
  )
  op = arith.MulFOp(x, phiOp.result)
  symbolTable[str(node.name)] = op


OpCodeGen = {
  'add': GenAddOp,
  'iadd': GenAddOp,
  'matmul': GenMatmulOp,
  'transpose': GenTransposeOp,
  'sub': GenSubOp,
  'mul': GenMulOp,
  'truediv': GenTrueDivOp,
  'ones': GenOnesOp,
  'gelu': GenGeluNoEstimateOp
}
