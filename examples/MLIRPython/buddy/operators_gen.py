import array

import mlir.ir as ir
from mlir.dialects import arith, tosa, math, linalg


def _getConstantTensorOp(value: float, sizes: list[int]):
    f32 = ir.F32Type.get()
    constantTensorType = ir.RankedTensorType.get(sizes, f32)
    constantElementAttr = ir.FloatAttr.get(f32, value)
    constantTensorAttr = ir.DenseElementsAttr.get_splat(
        constantTensorType, constantElementAttr
    )
    op = arith.ConstantOp(constantTensorType, constantTensorAttr)
    return op


def _broadcast_shape(tensor_input1, tensor_input2):
    shp1 = ir.RankedTensorType(tensor_input1.type).shape
    shp2 = ir.RankedTensorType(tensor_input2.type).shape
    if len(shp1) < len(shp2):
        shp1, shp2 = shp2, shp1
    while len(shp2) < len(shp1):
        shp2.insert(0, 1)
    for idx, (dim1, dim2) in enumerate(zip(shp1, shp2)):
        shp1[idx] = shp2[idx] = max(dim1, dim2)

    return shp1


def AddOp(node, symbol_table):
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))
    broadcasted_shp = _broadcast_shape(input1, input2)
    sizes = broadcasted_shp
    f32 = ir.F32Type.get()
    addResultTensorType = ir.RankedTensorType.get(sizes, f32)
    op = tosa.AddOp(addResultTensorType, input1, input2)
    return op


def GtOp(node, symbol_table):
    input1 = symbol_table.get(str(node.args[0]))
    input2 = symbol_table.get(str(node.args[1]))
    broadcasted_shp = _broadcast_shape(input1, input2)
    sizes = broadcasted_shp
    f32 = ir.F32Type.get()
    addResultTensorType = ir.RankedTensorType.get(sizes, f32)
    op = tosa.GreaterOp(addResultTensorType, input1, input2)
    return op


def SubOp(node, symbol_table):
    input1 = symbol_table.get(str(node.args[0]))
    input2 = symbol_table.get(str(node.args[1]))
    op = arith.SubFOp(input1, input2)
    return op


def MulOp(node, symbol_table):
    input1 = symbol_table.get(str(node.args[0]))
    input2 = symbol_table.get(str(node.args[1]))
    broadcasted_shp = _broadcast_shape(input1, input2)
    sizes = broadcasted_shp
    f32 = ir.F32Type.get()
    addResultTensorType = ir.RankedTensorType.get(sizes, f32)
    op = tosa.MulOp(addResultTensorType, input1, input2)
    return op


def DivOp(node, symbol_table):
    input1 = symbol_table.get(str(node.args[0]))
    input2 = symbol_table.get(str(node.args[1]))
    broadcasted_shp = _broadcast_shape(input1, input2)
    sizes = broadcasted_shp
    f32 = ir.F32Type.get()
    divResultTensorType = ir.RankedTensorType.get(sizes, f32)
    op = tosa.DivOp(divResultTensorType, input1, input2)
    return op


def ErfOp(node, symbol_table):
    input_ = symbol_table.get(str(node.args[0]))
    erfOp = math.ErfOp(input_)
    return erfOp


def TanhOp(node, symbol_table):
    input1 = symbol_table.get(str(node.args[0]))
    sizes = ir.RankedTensorType(input1.type).shape
    f32 = ir.F32Type.get()
    tanhResultTensorType = ir.RankedTensorType.get(sizes, f32)
    tanhOp = tosa.TanhOp(tanhResultTensorType, input1)
    return tanhOp


def ExpOp(node, symbol_table):
    input1 = symbol_table.get(str(node.args[0]))
    sizes = ir.RankedTensorType(input1.type).shape
    f32 = ir.F32Type.get()
    expResultTensorType = ir.RankedTensorType.get(sizes, f32)
    expOp = tosa.ExpOp(expResultTensorType, input1)
    return expOp


def RsqrtOp(node, symbol_table):
    input1 = symbol_table.get(str(node.args[0]))
    sizes = ir.RankedTensorType(input1.type).shape
    f32 = ir.F32Type.get()
    rsqrtResultTensorType = ir.RankedTensorType.get(sizes, f32)
    rsqrtOp = tosa.RsqrtOp(rsqrtResultTensorType, input1)
    return rsqrtOp


def AmaxOp(node, symbol_table):
    input1 = symbol_table.get(str(node.args[0]))
    dim = symbol_table.get(str(node.args[1]))[0]
    signlessType = ir.IntegerType.get_signless()
    dimAttr = ir.IntegerAttr.get(signlessType, dim)
    op = tosa.ReduceMaxOp(input1, dimAttr)
    return op


def ReshapeOp(node, symbol_table):
    input1 = symbol_table.get(str(node.args[0]))
    newShape = symbol_table.get(str(node.args[1]))
    newShapeContent = array.array("i", newShape)
    newShapeContent = memoryview(newShapeContent)
    op = tosa.ReshapeOp(input1, newShapeContent)
    return op


def UnsqueezeOp(node, symbol_table):
    inputTensor = symbol_table.get(str(node.args[0]))
    dim = symbol_table.get(str(node.args[1]))
    sizes = ir.RankedTensorType(inputTensor.type).shape
    sizes.insert(dim, 1)
    newShapeContent = array.array("i", sizes)
    newShapeContent = memoryview(newShapeContent)
    newShapeAttr = ir.DenseElementsAttr.get(newShapeContent)
    op = tosa.ReshapeOp(inputTensor, newShapeAttr)
    return op


def VarMeanOp(node, symbol_table):
    def MeanDimOp(_input_tensor: ir.Value, _dim) -> ir.Operation:
        reduce_dim_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(), _dim)
        reduce_sum_op: ir.Operation = tosa.ReduceSumOp(_input_tensor, reduce_dim_attr)

        output_size = []
        tensor_shp = ir.RankedTensorType(_input_tensor.type).shape
        dim_size = tensor_shp[_dim]
        for i, siz in enumerate(dim_size):
            if i == _dim:
                continue
            output_size.append(siz)

        denominator_const_op: ir.Operation = tosa.ConstOp(
            ir.DenseElementsAttr.get(memoryview(array.array("f", [dim_size])))
        )

        denominator_const_reshape_op: ir.Operation = tosa.ReshapeOp(
            denominator_const_op.results[0], memoryview(array.array("i", [1]))
        )

        reciprocal_op: ir.Operation = tosa.ReciprocalOp(
            denominator_const_reshape_op.results[0].type,
            denominator_const_reshape_op.results[0],
        )

        return tosa.MulOp(
            reduce_sum_op.results[0].type,
            reciprocal_op.results[0],
            reduce_sum_op.results[0],
        )

    def VarDimOp(_input_tensor: ir.Value, _mean_tensor: ir.Value, _dim) -> ir.Operation:
        sub_op: ir.Operation = tosa.SubOp(
            _input_tensor.type, _input_tensor, _mean_tensor
        )
        mul_op: ir.Operation = tosa.MulOp(
            _input_tensor.type, sub_op.results[0], sub_op.results[0]
        )

        reduce_dim_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(), _dim)
        reduce_sum_op: ir.Operation = tosa.ReduceSumOp(
            mul_op.results[0], reduce_dim_attr
        )

        tensor_shp = ir.RankedTensorType(_input_tensor.type).shape
        dim_size = tensor_shp[_dim]
        biased_denominator_const_op: ir.Operation = tosa.ConstOp(
            ir.DenseElementsAttr.get(memoryview(array.array("f", [dim_size - 1.0])))
        )
        reciprocal_op: ir.Operation = tosa.ReciprocalOp(
            biased_denominator_const_op.results[0].type,
            biased_denominator_const_op.results[0],
        )

        return tosa.MulOp(
            reduce_sum_op.results[0].type,
            reciprocal_op.results[0],
            reduce_sum_op.results[0],
        )

    mean_input_tensor = symbol_table.get(str(node.args[0]))
    var_input_tensor = symbol_table.get(str(node.args[0]))
    dim = node.args[1]
    if dim is None:
        while True:
            _mean_op = MeanDimOp(mean_input_tensor, 0)
            _var_op = VarDimOp(var_input_tensor, _mean_op.results[0], 0)
            shp = ir.RankedTensorType(_mean_op.result.type).shape
            if len(shp) == 0:
                mean_op = _mean_op
                var_op = _var_op
                break
            mean_input_tensor = _mean_op.results[0]
            var_input_tensor = _var_op.results[0]

    else:
        mean_op = MeanDimOp(mean_input_tensor, dim[0])
        var_op = VarDimOp(var_input_tensor, mean_op.results[0], dim[0])

    return var_op, mean_op


# add, addmm, amax, bmm, clone, convert_element_type
# div, embedding, erf, exp, expand, getitem, gt, inductor_lookup_seed
# inductor_random, inductor_seeds, mul, permute, reshape, rsqrt
# select, slice, sub, tanh, unsqueeze, var_mean
operation_func = {"add": AddOp, "var_mean": VarMeanOp}
