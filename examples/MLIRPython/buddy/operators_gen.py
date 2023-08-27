import array
from typing import Dict, Tuple

import mlir.ir as ir
from mlir.dialects import arith, tosa, math, tensor
import torch


def _get_constant_tensor_op(value: float, sizes: list[int]):
  f32 = ir.F32Type.get()
  constant_tensor_type = ir.RankedTensorType.get(sizes, f32)
  constant_element_attr = ir.FloatAttr.get(f32, value)
  constant_tensor_attr = ir.DenseElementsAttr.get_splat(constant_tensor_type,
                                                        constant_element_attr)
  op = arith.ConstantOp(constant_tensor_type, constant_tensor_attr)
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


def add_op(node, symbol_table):
  input1 = symbol_table.get((str(node.args[0]), 0))
  input2 = symbol_table.get((str(node.args[1]), 0))
  broadcasted_shp = _broadcast_shape(input1, input2)
  sizes = broadcasted_shp
  f32 = ir.F32Type.get()
  add_result_tensor_type = ir.RankedTensorType.get(sizes, f32)
  op = tosa.AddOp(add_result_tensor_type, input1, input2)
  return op


def addmm_op(node: torch.fx.Node,
             symbol_table: Dict[Tuple[str, int], ir.Operation]) -> ir.Operation:
  input_ = symbol_table.get((str(node.args[0]), 0))
  mat1 = symbol_table.get((str(node.args[1]), 0))
  mat2 = symbol_table.get((str(node.args[2]), 0))
  mat1_shp = ir.RankedTensorType.get(mat1).shape
  mat2_shp = ir.RankedTensorType.get(mat2).shape
  result_shp = [mat1_shp[0], mat2_shp[1]]
  f32 = ir.F32Type.get()
  matmul_result_type = ir.RankedTensorType.get(result_shp, f32)
  matmul_op = tosa.MatMulOp(matmul_result_type, mat1, mat2)
  add_result_tensor_type = ir.RankedTensorType.get(result_shp, f32)
  op = tosa.AddOp(add_result_tensor_type, input_, matmul_op.c)
  return op


def gt_op(node, symbol_table):
  input1 = symbol_table.get((str(node.args[0]), 0))
  input2 = symbol_table.get((str(node.args[1]), 0))
  broadcasted_shp = _broadcast_shape(input1, input2)
  sizes = broadcasted_shp
  f32 = ir.F32Type.get()
  add_result_tensor_type = ir.RankedTensorType.get(sizes, f32)
  op = tosa.GreaterOp(add_result_tensor_type, input1, input2)
  return op


def sub_op(node, symbol_table):
  input1 = symbol_table.get((str(node.args[0]), 0))
  input2 = symbol_table.get((str(node.args[1]), 0))
  op = arith.SubFOp(input1, input2)
  return op


def mul_op(node, symbol_table):
  input1 = symbol_table.get((str(node.args[0]), 0))
  input2 = symbol_table.get((str(node.args[1]), 0))
  broadcasted_shp = _broadcast_shape(input1, input2)
  sizes = broadcasted_shp
  f32 = ir.F32Type.get()
  addResultTensorType = ir.RankedTensorType.get(sizes, f32)
  op = tosa.MulOp(addResultTensorType, input1, input2)
  return op


def div_op(node, symbol_table):
  input1 = symbol_table.get((str(node.args[0]), 0))
  input2 = symbol_table.get((str(node.args[1]), 0))
  broadcasted_shp = _broadcast_shape(input1, input2)
  sizes = broadcasted_shp
  f32 = ir.F32Type.get()
  divResultTensorType = ir.RankedTensorType.get(sizes, f32)
  op = tosa.DivOp(divResultTensorType, input1, input2)
  return op


def erf_op(node, symbol_table):
  input_ = symbol_table.get((str(node.args[0]), 0))
  op = math.ErfOp(input_)
  return op


def tanh_op(node, symbol_table):
  input1 = symbol_table.get((str(node.args[0]), 0))
  sizes = ir.RankedTensorType(input1.type).shape
  f32 = ir.F32Type.get()
  tanhResultTensorType = ir.RankedTensorType.get(sizes, f32)
  op = tosa.TanhOp(tanhResultTensorType, input1)
  return op


def exp_op(node, symbol_table):
  input1 = symbol_table.get((str(node.args[0]), 0))
  sizes = ir.RankedTensorType(input1.type).shape
  f32 = ir.F32Type.get()
  expResultTensorType = ir.RankedTensorType.get(sizes, f32)
  op = tosa.ExpOp(expResultTensorType, input1)
  return op


def rsqrt_op(node, symbol_table):
  input1 = symbol_table.get((str(node.args[0]), 0))
  sizes = ir.RankedTensorType(input1.type).shape
  f32 = ir.F32Type.get()
  rsqrt_result_tensor_type = ir.RankedTensorType.get(sizes, f32)
  op = tosa.RsqrtOp(rsqrt_result_tensor_type, input1)
  return op


def amax_op(node, symbol_table):
  input1 = symbol_table.get((str(node.args[0]), 0))
  dim = symbol_table.get(str(node.args[1]))[0]
  signlessType = ir.IntegerType.get_signless()
  dimAttr = ir.IntegerAttr.get(signlessType, dim)
  op = tosa.ReduceMaxOp(input1, dimAttr)
  return op


def reshape_op(node, symbol_table):
  input1 = symbol_table.get((str(node.args[0]), 0))
  new_shape = node.args[1]
  total_size = 1
  now_shape = ir.RankedTensorType(input1.type).shape
  for dim_siz in now_shape:
    total_size *= dim_siz

  neg_one_cnt = 0
  rest_size = 1
  for dim_siz in new_shape:
    if dim_siz == -1:
      neg_one_cnt += 1
      continue
    rest_size *= dim_siz

  if neg_one_cnt != 0:
    if neg_one_cnt > 1 or total_size % rest_size != 0:
      raise ValueError("Can not infer the new shape!")
    infer_dim_size = total_size // rest_size
    for i, _ in enumerate(new_shape):
      if new_shape[i] == -1:
        new_shape[i] = infer_dim_size

  new_shape_content = array.array("i", new_shape)
  new_shape_content = memoryview(new_shape_content)
  op = tosa.ReshapeOp(input1, new_shape_content)

  return op


def unsqueeze_op(node, symbol_table):
  input_tensor = symbol_table.get((str(node.args[0]), 0))
  dim = node.args[1]
  sizes = ir.RankedTensorType(input_tensor.type).shape
  sizes.insert(dim, 1)
  new_shape_content = array.array("i", sizes)
  new_shape_content = memoryview(new_shape_content)
  new_shape_attr = ir.DenseElementsAttr.get(new_shape_content)
  op = tosa.ReshapeOp(input_tensor, new_shape_attr)
  return op


def select_op(node, symbol_table):
  input_tensor = symbol_table.get((str(node.args[0]), 0))
  dim = node.args[1]
  index = node.args[2]

  sizes = ir.RankedTensorType(input_tensor.type).shape

  new_sizes = sizes[:dim] + [1] + sizes[dim + 1:]
  new_sizes_attr = ir._denseI64ArrayAttr(new_sizes, None)

  start = [0] * len(sizes)
  start[dim] = index
  start_attr = ir._denseI64ArrayAttr(start, None)

  f32 = ir.F32Type.get()
  output_type = ir.RankedTensorType.get(new_sizes, f32)
  op = tosa.SliceOp(output_type, input_tensor, start_attr, new_sizes_attr)

  reshape_sizes = sizes[:dim] + sizes[dim + 1:]
  reshape_sizes_content = array.array("Q", reshape_sizes)
  reshape_sizes_content = memoryview(reshape_sizes_content)
  op = tosa.ReshapeOp(op.results[0], reshape_sizes_content)

  return op


def slice_op(node, symbol_table):
  input_tensor = symbol_table.get((str(node.args[0]), 0))
  dim = node.args[1]
  start_idx = node.args[2]
  end_idx = node.args[3]

  sizes = ir.RankedTensorType(input_tensor.type).shape

  if start_idx < 0:
    start_idx += sizes[dim]

  if end_idx < 0:
    end_idx += sizes[dim]

  if start_idx < 0:
    start_idx = 0
  elif start_idx >= sizes[dim]:
    start_idx = sizes[dim]

  if end_idx < start_idx:
    end_idx = start_idx
  elif end_idx >= sizes[dim]:
    end_idx = sizes[dim]

  new_sizes = [x for x in sizes]
  new_sizes[dim] = end_idx - start_idx
  new_sizes_attr = ir._denseI64ArrayAttr(new_sizes, None)

  offsets = [0] * len(sizes)
  offsets[dim] = start_idx
  offsets_attr = ir._denseI64ArrayAttr(offsets, None)

  strides = [1] * len(sizes)
  strides_attr = ir._denseI64ArrayAttr(strides, None)

  f32 = ir.F32Type.get()
  extract_slice_result_type = ir.RankedTensorType.get(new_sizes, f32)
  op = tensor.ExtractSliceOp(extract_slice_result_type, input_tensor, [], [],
                             [], offsets_attr, new_sizes_attr, strides_attr)

  return op


def convert_element_type_op(node, symbol_table):
  # maintain a mapping of torch types and mlir types
  types_mapping = {
      torch.float64: ir.F64Type.get(),
      torch.float32: ir.F32Type.get(),
      torch.float16: ir.F16Type.get()
  }
  input_tensor = symbol_table.get((str(node.args[0]), 0))
  to_cast_type = types_mapping[node.args[1]]
  sizes = ir.RankedTensorType(input_tensor.type).shape
  output_type = ir.RankedTensorType.get(sizes, to_cast_type)
  return tosa.CastOp(output_type, input_tensor)


def clone_op(node, symbol_table):
  input_tensor = symbol_table.get((str(node.args[0]), 0))
  sizes = ir.RankedTensorType(input_tensor.type).shape
  f32 = ir.F32Type.get()
  output_type = ir.RankedTensorType.get(sizes, f32)

  return tosa.IdentityOp(output_type, input_tensor)


def var_mean_op(node, symbol_table):

  def mean_dim_op(_input_tensor: ir.Value, _dim) -> ir.Operation:
    if isinstance(_dim, int):
      _dim = [_dim]

    # `_input_tensor` is the first tensor we need to reduce
    reduce_sum_result = _input_tensor

    # reduce along each dimension in `_dim`
    for _dim_item, _ in enumerate(_dim):
      reduce_dim_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64),
                                           _dim_item)
      reduce_sum_op: ir.Operation = tosa.ReduceSumOp(reduce_sum_result,
                                                     reduce_dim_attr)
      # Next reduction is executed based on this time's reduction result
      reduce_sum_result = reduce_sum_op.results[0]

    tensor_shp = ir.RankedTensorType(_input_tensor.type).shape
    dim_size = 1
    # calculate the total size on all reduction dimensions to get the denominator
    for _dim_item in _dim:
      dim_size *= tensor_shp[_dim_item]

    denominator_const_op: ir.Operation = tosa.ConstOp(
        ir.DenseElementsAttr.get(memoryview(array.array("f", [dim_size]))))

    reciprocal_op: ir.Operation = tosa.ReciprocalOp(
        denominator_const_op.results[0].type,
        denominator_const_op.results[0],
    )

    return tosa.MulOp(
        reduce_sum_op.results[0].type,
        reciprocal_op.results[0],
        reduce_sum_op.results[0],
        ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0),
    )

  def var_dim_op(_input_tensor: ir.Value, _mean_tensor: ir.Value, _dim,
                 _correction) -> ir.Operation:
    if isinstance(_dim, int):
      _dim = [_dim]
    # get (\bar{x} - x_i)
    sub_op: ir.Operation = tosa.SubOp(_input_tensor.type, _input_tensor,
                                      _mean_tensor)

    # get (\bar{x} - x_i) ^ 2
    mul_op: ir.Operation = tosa.MulOp(
        _input_tensor.type,
        sub_op.results[0],
        sub_op.results[0],
        ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0),
    )

    # the result of `mul_op` is the first tensor we need to reduce
    reduce_sum_op = mul_op
    for _dim_item, _ in enumerate(_dim):
      reduce_dim_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64),
                                           _dim_item)
      reduce_sum_op: ir.Operation = tosa.ReduceSumOp(reduce_sum_op.results[0],
                                                     reduce_dim_attr)

    tensor_shp = ir.RankedTensorType(_input_tensor.type).shape
    dim_size = 1
    # calculate the denominator
    for _dim_item in _dim:
      dim_size *= tensor_shp[_dim_item]
    biased_denominator_const_op: ir.Operation = tosa.ConstOp(
        ir.DenseElementsAttr.get(
            memoryview(array.array("f", [dim_size - _correction]))))
    reciprocal_op: ir.Operation = tosa.ReciprocalOp(
        biased_denominator_const_op.results[0].type,
        biased_denominator_const_op.results[0],
    )

    return tosa.MulOp(
        reduce_sum_op.results[0].type,
        reciprocal_op.results[0],
        reduce_sum_op.results[0],
        ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0),
    )

  mean_input_tensor = symbol_table.get((str(node.args[0]), 0))
  var_input_tensor = symbol_table.get((str(node.args[0]), 0))

  kwargs = node.kwargs
  keep_dim = kwargs.get("keep_dim", False)
  correction = kwargs.get("correction", 1.0)

  mean_op = None
  var_op = None
  if len(node.args) == 1:
    calc_dims = range(len(ir.RankedTensorType(mean_input_tensor.type).shape))
  else:
    calc_dims = node.args[1]

  mean_op = mean_dim_op(mean_input_tensor, calc_dims)
  var_op = var_dim_op(var_input_tensor, mean_op.results[0], calc_dims,
                      correction)
  mean_input_tensor = mean_op.results[0]
  var_input_tensor = var_op.results[0]

  if not keep_dim:
    result_shp = ir.RankedTensorType(var_op.results[0].type).shape
    result_shp = [siz for siz in result_shp if siz != 1]
    var_op = tosa.ReshapeOp(var_op.results[0],
                            memoryview(array.array("i", result_shp)))
    mean_op = tosa.ReshapeOp(mean_op.results[0],
                             memoryview(array.array("i", result_shp)))

  return var_op, mean_op


def permute_op(node, symbol_table):
  input_tensor = symbol_table.get((str(node.args[0]), 0))
  perm = node.args[1]
  perm_const_op = tosa.ConstOp(
      ir.DenseElementsAttr.get(memoryview(array.array("i", perm))))
  result_element_type = ir.RankedTensorType(input_tensor.type).element_type
  init_shape = ir.RankedTensorType(input_tensor.type).shape
  new_shape = []
  for perm_item in perm:
    new_shape.append(init_shape[perm_item])

  permute_result_type = ir.RankedTensorType.get(new_shape, result_element_type)
  permute_op = tosa.TransposeOp(permute_result_type, input_tensor,
                                perm_const_op.results[0])
  return permute_op


def embedding_op(node, symbol_table):
  indices = symbol_table.get((str(node.args[1]), 0))
  weight = symbol_table.get((str(node.args[0]), 0))
  padding_idx = None if len(node.args) < 3 else node.args[2]

  indices_size = ir.RankedTensorType(indices.type).shape
  weight_size = ir.RankedTensorType(weight.type).shape
  f32 = ir.F32Type.get()
  assert len(indices_size) == 2

  if indices_size[0] != 1:
    total_size = 1
    for x in indices_size:
      total_size *= x
    indices_reshape_op = tosa.ReshapeOp(
        indices, memoryview(array.array("i", [1, total_size])))
    indices = indices_reshape_op.result
    gather_result_type = ir.RankedTensorType.get(
        [1, total_size, weight_size[1]], f32)
  else:
    gather_result_type = ir.RankedTensorType.get(
        [*indices_size, weight_size[1]], f32)

  weight_reshape_op = tosa.ReshapeOp(
      weight, memoryview(array.array("i", [1, *weight_size])))

  gather_op = tosa.GatherOp(gather_result_type, weight_reshape_op.result,
                            indices)
  op = tosa.ReshapeOp(
      gather_op.output,
      memoryview(array.array("i", [*indices_size, weight_size[1]])))

  return op


# add, addmm, amax, bmm, clone, convert_element_type
# div, embedding, erf, exp, expand, getitem, gt, inductor_lookup_seed
# inductor_random, inductor_seeds, mul, permute, reshape, rsqrt
# select, slice, sub, tanh, unsqueeze, var_mean
operation_func = {
    "add.Tensor": add_op,
    "var_mean.correction": var_mean_op,
    "addmm.default": addmm_op,
    "reshape.default": reshape_op,
    "select.int": select_op,
    "slice.Tensor": slice_op,
    "embedding.default": embedding_op,
    "convert_element_type.default": convert_element_type_op,
    "permute.default": permute_op
}
