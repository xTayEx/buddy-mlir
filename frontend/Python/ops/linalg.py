# ===- linalg.py ---------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# The registry of mappings from Buddy Graph to MLIR linalg dialect operations.
#
# ===---------------------------------------------------------------------------

from typing import Dict, Tuple

import mlir.ir as ir
from mlir.dialects import tosa, linalg, arith, tensor, math
import copy
import numpy
import functools

from ..graph import *
from ..graph.graph import TensorDType
from .utils import *


def add_op(node: AddOp, symbol_table: Dict[Tuple[str, int], ir.Operation]):
    """
    Import tensor add operation.
    From buddy AddOp to MLIR arith `constant` operation.

    Note: this function init an output tensor according input range.

    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation representing the result tensor of two input nodes' add
        result.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    shape = list(node.tensor_meta["shape"])
    if isinstance(node.args[1], str):
        input2 = symbol_table.get((str(node.args[1]), 0))
    else:
        data = [node.args[1]]
        input2_shape = numpy.array(data).shape
        tensor_type = ir.RankedTensorType.get(input2_shape, mlir_dtype)
        element = mlir_element_attr_get(dtype, node.args[1])
        attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
        input2 = arith.ConstantOp(tensor_type, attr).result
    if input1 is None or input2 is None:
        return
    add_result_tensor_type = ir.RankedTensorType.get(shape, mlir_dtype)
    op = tosa.AddOp(
        add_result_tensor_type,
        input1,
        input2,
    )
    return op.result


def arange_op(
    node: ArangeOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import tensor arange operation.
    From buddy ArangeOp to MLIR arith `constant` operation.

    Note: this function init an output tensor according input range.

    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation representing the result tensor of ranging the start
        and end from input node.
    """
    if len(node.args) == 2:
        start = int(node.args[0])
        end = int(node.args[1])
    else:
        start = 0
        end = int(node.args[0])
    stride = 1
    dtype = node.tensor_meta["dtype"]
    shape = list(node.tensor_meta["shape"])
    dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(shape, dtype)
    attr = ir.DenseElementsAttr.get(
        numpy.array([i for i in range(start, end, stride)]),
        signless=True,
        type=tensor_type,
    )
    op = arith.ConstantOp(tensor_type, attr)

    return op


def unsqueeze_op(
    node: UnsqueezeOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the unsqueeze operation.
    From buddy UnsqueezeOp to MLIR TOSA `reshape` operation.

    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the reshape op.
    """
    input_node = symbol_table.get((str(node.args[0]), 0))
    if input_node is None:
        return
    axis = int(node.args[1])
    input_shape = ir.RankedTensorType(input_node.type).shape
    input_shape.insert(axis, 1)
    tensor_type = ir._denseI64ArrayAttr(
        numpy.array(input_shape, dtype=numpy.int64), None
    )
    op = tosa.ReshapeOp(input_node, tensor_type)

    return op


def view_op(
    node: ViewOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor view operation.
    From buddy ViewOp to MLIR TOSA `reshape` operation.

    Note: If the new shape contains one and only one `-1`, the size of the new
    shape will be inferred automatically.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the reshape op.
    """
    input_node = symbol_table.get((str(node.args[0]), 0))
    if input_node is None:
        return
    output_shape = list(node.args[1])
    input_shape = list(ir.RankedTensorType(input_node.type).shape)

    nums = 1
    for i in input_shape:
        nums *= i
    for i in output_shape:
        if i != -1:
            nums //= i
    for i, s in enumerate(output_shape):
        if s == -1:
            output_shape[i] = nums

    tensor_type = ir._denseI64ArrayAttr(
        numpy.array(output_shape, dtype=numpy.int64), None
    )
    op = tosa.ReshapeOp(input_node, tensor_type)

    return op


def embedding_op(
    node: EmbeddingOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the embedding operation.
    From buddy EmbeddingOp to MLIR linalg `generic` operation.

    Note: In this op, input node1's value is as index to get input node2's row
    slice.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, dtype)
    output = tensor.EmptyOp(output_shape, dtype)
    generic_map = ir.AffineMap.get_permutation([0, 1, 2])
    op = linalg.GenericOp(
        [tensor_type],
        [input2],
        [output],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(generic_map.get_submap([0, 1])),
                ir.AffineMapAttr.get(generic_map.get_submap([0, 1, 2])),
            ]
        ),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * 3
        ),
    )
    block = ir.Block.create_at_start(
        op.region,
        [
            ir.RankedTensorType(input2.type).element_type,
            ir.RankedTensorType(output.result.type).element_type,
        ],
    )
    index1 = arith.IndexCastOp(ir.IndexType.get(), block.arguments[0])
    index2 = linalg.IndexOp(ir._i64Attr(2, None))
    value = tensor.ExtractOp(input1, [index1.result, index2.result])
    block.append(index1)
    block.append(index2)
    block.append(value)
    block.append(linalg.YieldOp([value.result]))

    return op


def ones_op(
    node: OnesOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor ones operation.
    From buddy OnesOp to MLIR arith `constant` operation.

    Note: This op, input node1's value is as index to get input node2's row
    slice.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the arith.constant op.
    """
    output_shape = list(node.args[0])
    dtype = node.tensor_meta["dtype"]
    element = mlir_element_attr_get(dtype, 1)
    tensor_type = ir.RankedTensorType.get(output_shape, element.type)
    attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
    op = arith.ConstantOp(tensor_type, attr)

    return op


def full_op(
    node: FullOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor full operation.
    From buddy FullOp to MLIR arith `constant` operation.

    Note: This op, input node1's value is the shape of output tensor, input
    node2's value is the value of all elements in output tensor.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the arith.constant op.
    """
    output_shape = list(node.args[0])
    value = node.args[1]
    dtype = node.tensor_meta["dtype"]
    element = mlir_element_attr_get(dtype, value)
    tensor_type = ir.RankedTensorType.get(output_shape, element.type)
    attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
    op = arith.ConstantOp(tensor_type, attr)

    return op


def lt_op(
    node: LessThanOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor less than operation.
    From buddy LessThanOp to MLIR arith `constant` operation.

    Note: This op, campare two input nodes, and output bool tensor to represent
    compare result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    value = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 2)
    shp1 = list(ir.RankedTensorType(ir.Value(input1).type).shape)
    shp2 = list(ir.RankedTensorType(ir.Value(input2).type).shape)
    dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, dtype)
    output = tensor.EmptyOp(output_shape, dtype)
    if len(shp1) < len(shp2):
        if int(shp1[-1]) > 1 and shp2[-1] == 1:
            generic_map = ir.AffineMap.get_permutation(
                [i for i in range(len(shp2) + 1)]
            )
            op = linalg.GenericOp(
                [tensor_type],
                [input1, input2],
                [output],
                ir.ArrayAttr.get(
                    [
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [
                                    i
                                    for i in range(
                                        len(shp2) - len(shp1), len(shp2)
                                    )
                                ]
                            )
                        ),
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(0, len(shp2) - 1)]
                                + [len(shp2)]
                            )
                        ),
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(0, len(shp2))]
                            )
                        ),
                    ]
                ),
                ir.ArrayAttr.get(
                    [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                    * len(shp2)
                    + [ir.Attribute.parse("#linalg.iterator_type<reduction>")]
                ),
            )
            block = ir.Block.create_at_start(
                op.region,
                [
                    ir.RankedTensorType(input2.type).element_type,
                    ir.RankedTensorType(input2.type).element_type,
                    dtype,
                ],
            )
            if (
                str(ir.RankedTensorType(input2.type).element_type).find("i")
                != -1
            ):
                cmpop = arith.CmpIOp(
                    value, block.arguments[0], block.arguments[1]
                )
            else:
                cmpop = arith.CmpFOp(
                    value, block.arguments[0], block.arguments[1]
                )
            block.append(cmpop)
            block.append(linalg.YieldOp([cmpop.result]))

    return op


def masked_fill_op(
    node: MaskedFillOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor masked fill operation.
    From buddy MaskedFillOp to MLIR linalg `generic` operation.

    Note: This op, input node2 is a bool tensor. Select input node1's value or
    input node3's value by true or false in input node2's value.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))
    if input1 is None or input2 is None:
        return
    dtype = node.tensor_meta["dtype"]
    value = node.args[2]
    attr = mlir_element_attr_get(dtype, value)
    dtype = mlir_element_type_get(dtype)
    value = arith.ConstantOp(dtype, attr)
    output_shape = list(node.tensor_meta["shape"])
    tensor_type = ir.RankedTensorType.get(output_shape, dtype)
    output = tensor.EmptyOp(output_shape, dtype)
    generic_map = ir.AffineMap.get_permutation(
        [i for i in range(len(output_shape))]
    )
    op = linalg.GenericOp(
        [tensor_type],
        [input1, input2],
        [output],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
            ]
        ),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
            * len(output_shape)
        ),
    )
    block = ir.Block.create_at_start(
        op.region,
        [
            ir.RankedTensorType(input1.type).element_type,
            ir.RankedTensorType(input2.type).element_type,
            ir.RankedTensorType(output.result.type).element_type,
        ],
    )
    select_op = arith.SelectOp(block.arguments[1], value, block.arguments[0])
    block.append(select_op)
    block.append(linalg.YieldOp([select_op.result]))

    return op


def slice_op(
    node: SliceOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor slice operation.
    From buddy SliceOp to MLIR tensor `extract_slice` operation.

    Note: This op, get the slice of input node1.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the tensor.extract_slice op.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return
    dim = int(node.args[1])
    start = int(node.args[2])
    end = int(node.args[3])
    input_shape = ir.RankedTensorType(input1.type).shape
    if end > input_shape[dim]:
        end = input_shape[dim]
    if len(node.args) < 5:
        step = 1
    else:
        step = node.args[4]
    offset = [0 for x in input_shape]
    offset[dim] = start
    offset_attr = ir._denseI64ArrayAttr(offset, None)
    output_shape = list(node.tensor_meta["shape"])
    size_attr = ir._denseI64ArrayAttr(output_shape, None)
    stride = [1 for x in output_shape]
    stride[dim] = step
    stride_attr = ir._denseI64ArrayAttr(stride, None)
    dtype = node.tensor_meta["dtype"]
    dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, dtype)

    op = tensor.ExtractSliceOp(
        tensor_type, input1, [], [], [], offset_attr, size_attr, stride_attr
    )

    return op


def expand_op(
    node: ExpandOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor expand operation.
    From buddy ExpandOp to MLIR tensor `extract_slice` operation.

    Note: This op, based on expand shape, create a new tensor and extract slice
    from origin tensor.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the tensor.extract_slice op.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    assert isinstance(node.args[1], list)

    if input1 is None:
        return
    input_shape = ir.RankedTensorType(input1.type).shape
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    dtype = mlir_element_type_get(dtype)
    empty_tensor = tensor.EmptyOp(output_shape, dtype)
    if list(input_shape) == list(node.args[1]):
        offset_attr = ir._denseI64ArrayAttr([0 for x in input_shape], None)
        size_attr = ir._denseI64ArrayAttr(output_shape, None)
        stride_attr = ir._denseI64ArrayAttr([1 for x in input_shape], None)
        tensor_type = ir.RankedTensorType.get(output_shape, dtype)
        extract_tensor = tensor.ExtractSliceOp(
            tensor_type, input1, [], [], [], offset_attr, size_attr, stride_attr
        )
        op = tensor.InsertSliceOp(
            extract_tensor.result,
            empty_tensor.result,
            [],
            [],
            [],
            offset_attr,
            size_attr,
            stride_attr,
        )
    else:
        for i in range(len(input_shape) - 1, -1, -1):
            if input_shape[i] != output_shape[i]:
                for j in range(output_shape[i]):
                    offset = [0 for x in input_shape]
                    offset_attr = ir._denseI64ArrayAttr(offset, None)
                    size_attr = ir._denseI64ArrayAttr(
                        [1] * (i + 1) + [x for x in output_shape[i + 1 :]], None
                    )
                    stride_attr = ir._denseI64ArrayAttr([1] * len(offset), None)
                    tensor_type = ir.RankedTensorType.get(
                        [1] * (i + 1) + [x for x in output_shape[i + 1 :]],
                        dtype,
                    )
                    extract_tensor = tensor.ExtractSliceOp(
                        tensor_type,
                        input1,
                        [],
                        [],
                        [],
                        offset_attr,
                        size_attr,
                        stride_attr,
                    )
                    offset[i] = j
                    offset_attr = ir._denseI64ArrayAttr(offset, None)
                    op = tensor.InsertSliceOp(
                        extract_tensor.result,
                        empty_tensor.result,
                        [],
                        [],
                        [],
                        offset_attr,
                        size_attr,
                        stride_attr,
                    )
                    empty_tensor = op
    return op


def to_copy_op(
    node: ToCopyOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor copy operation.
    From buddy ToCopyOp to MLIR linalg `generic`
    operation.

    Note: This op, will convert input node's value type, such as float32 to
    bool.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]

    if dtype == TensorDType.Bool:
        if str(ir.RankedTensorType(input1.type).element_type) == "f32":
            tensor_type = ir.RankedTensorType.get(
                output_shape, ir.IntegerType.get_signless(1)
            )
            output = tensor.EmptyOp(
                output_shape, ir.IntegerType.get_signless(1)
            )
            generic_map = ir.AffineMap.get_permutation(
                [i for i in range(len(output_shape))]
            )
            op = linalg.GenericOp(
                [tensor_type],
                [input1],
                [output],
                ir.ArrayAttr.get(
                    [
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(len(output_shape))]
                            )
                        ),
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(len(output_shape))]
                            )
                        ),
                    ]
                ),
                ir.ArrayAttr.get(
                    [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                    * len(output_shape)
                ),
            )
            block = ir.Block.create_at_start(
                op.region,
                [
                    ir.RankedTensorType(input1.type).element_type,
                    ir.RankedTensorType(output.result.type).element_type,
                ],
            )
            fptosi_op = arith.FPToSIOp(
                ir.IntegerType.get_signless(32), block.arguments[0]
            )
            trunc_op = arith.TruncIOp(
                ir.IntegerType.get_signless(1), fptosi_op.result
            )
            block.append(fptosi_op)
            block.append(trunc_op)
            block.append(linalg.YieldOp([trunc_op.result]))
    elif dtype == TensorDType.Float32:
        if str(ir.RankedTensorType(input1.type).element_type) == "i1":
            tensor_type = ir.RankedTensorType.get(
                output_shape, ir.F32Type.get()
            )
            output = tensor.EmptyOp(output_shape, ir.F32Type.get())
            generic_map = ir.AffineMap.get_permutation(
                [i for i in range(len(output_shape))]
            )
            op = linalg.GenericOp(
                [tensor_type],
                [input1],
                [output],
                ir.ArrayAttr.get(
                    [
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(len(output_shape))]
                            )
                        ),
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(len(output_shape))]
                            )
                        ),
                    ]
                ),
                ir.ArrayAttr.get(
                    [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                    * len(output_shape)
                ),
            )
            block = ir.Block.create_at_start(
                op.region,
                [
                    ir.RankedTensorType(input1.type).element_type,
                    ir.RankedTensorType(output.result.type).element_type,
                ],
            )
            exti_op = arith.ExtUIOp(
                ir.IntegerType.get_signless(32), block.arguments[0]
            )
            sitofp_op = arith.SIToFPOp(ir.F32Type.get(), exti_op.result)
            block.append(exti_op)
            block.append(sitofp_op)
            block.append(linalg.YieldOp([sitofp_op.result]))

    return op


def rsub_op(
    node: RsubOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor rsub operation.
    From buddy RsubOp to MLIR linalg `generic` operation.

    Note: This op, compute input node1 rsub input node2
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    value = node.args[1]
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    if not isinstance(value, str):
        value = arith.ConstantOp(
            mlir_dtype, mlir_element_attr_get(dtype, value)
        )
        generic_map = ir.AffineMap.get_permutation(
            [i for i in range(len(output_shape))]
        )
        tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
        output = tensor.EmptyOp(output_shape, mlir_dtype)
        op = linalg.GenericOp(
            [tensor_type],
            [input1],
            [output],
            ir.ArrayAttr.get(
                [
                    ir.AffineMapAttr.get(
                        generic_map.get_submap(
                            [i for i in range(len(output_shape))]
                        )
                    ),
                    ir.AffineMapAttr.get(
                        generic_map.get_submap(
                            [i for i in range(len(output_shape))]
                        )
                    ),
                ]
            ),
            ir.ArrayAttr.get(
                [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                * len(output_shape)
            ),
        )
        block = ir.Block.create_at_start(
            op.region,
            [
                ir.RankedTensorType(input1.type).element_type,
                ir.RankedTensorType(output.result.type).element_type,
            ],
        )
        if str(ir.RankedTensorType(input1.type).element_type).find("i") != -1:
            sub_op = arith.SubIOp(value.result, block.arguments[0])
        else:
            sub_op = arith.SubFOp(value.result, block.arguments[0])
        block.append(sub_op)
        block.append(linalg.YieldOp([sub_op.result]))

    return op


def pow_op(
    node: PowOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor copy operation.
    From buddy PowOp to MLIR linalg `generic`
    operation.

    Note: This op, compute input node's power result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return
    value = node.args[1]
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    dtype = mlir_element_type_get(dtype)
    if not isinstance(value, str):
        if abs(int(value) - float(value)) < 1e-6:
            generic_map = ir.AffineMap.get_permutation(
                [i for i in range(len(output_shape))]
            )
            tensor_type = ir.RankedTensorType.get(output_shape, dtype)
            output = tensor.EmptyOp(output_shape, dtype)
            value = arith.ConstantOp(
                ir.IntegerType.get_signless(32),
                ir.IntegerAttr.get(ir.IntegerType.get_signless(32), value),
            )
            op = linalg.GenericOp(
                [tensor_type],
                [input1],
                [output],
                ir.ArrayAttr.get(
                    [
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(len(output_shape))]
                            )
                        ),
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(len(output_shape))]
                            )
                        ),
                    ]
                ),
                ir.ArrayAttr.get(
                    [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                    * len(output_shape)
                ),
            )
            block = ir.Block.create_at_start(
                op.region,
                [
                    ir.RankedTensorType(input1.type).element_type,
                    ir.RankedTensorType(output.result.type).element_type,
                ],
            )
            if (
                str(ir.RankedTensorType(input1.type).element_type).find("i")
                != -1
            ):
                powi_op = math.IPowIOp(block.arguments[0], value.result)
            else:
                powi_op = math.FPowIOp(block.arguments[0], value.result)
            block.append(powi_op)
            block.append(linalg.YieldOp([powi_op.result]))

    return op


def mean_op(
    node: MeanOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor copy operation.
    From buddy MeanOp to MLIR linalg `generic` operation.

    Note: This op, compute input node's mean result in a specified dim.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return
    dims = list(node.args[1])
    keep_dim = bool(node.args[2])
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    element = mlir_element_attr_get(dtype, 0.0)
    attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
    output = arith.ConstantOp(tensor_type, attr)
    assert len(dims) == 1
    for dim in dims:
        if dim < 0:
            dim = len(list(ir.RankedTensorType(input1.type).shape)) + dim
        if keep_dim:
            generic_map = ir.AffineMap.get_permutation(
                [i for i in range(len(output_shape) + 1)]
            )
            tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
            output_map = [i for i in range(len(output_shape))]
            output_map[dim] = len(output_shape)
            loop_type = [
                ir.Attribute.parse("#linalg.iterator_type<parallel>")
            ] * (len(output_shape) + 1)
            loop_type[dim] = ir.Attribute.parse(
                "#linalg.iterator_type<reduction>"
            )
            op = linalg.GenericOp(
                [tensor_type],
                [input1],
                [output],
                ir.ArrayAttr.get(
                    [
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(
                                [i for i in range(len(output_shape))]
                            )
                        ),
                        ir.AffineMapAttr.get(
                            generic_map.get_submap(output_map)
                        ),
                    ]
                ),
                ir.ArrayAttr.get(loop_type),
            )
            block = ir.Block.create_at_start(
                op.region,
                [
                    ir.RankedTensorType(input1.type).element_type,
                    ir.RankedTensorType(output.result.type).element_type,
                ],
            )
            value = arith.ConstantOp(
                mlir_dtype,
                mlir_element_attr_get(
                    dtype, list(ir.RankedTensorType(input1.type).shape)[dim]
                ),
            )
            if (
                str(ir.RankedTensorType(input1.type).element_type).find("i")
                != -1
            ):
                block_div_op = arith.DivSIOp(block.arguments[0], value.result)
                block_add_op = arith.AddIOp(
                    block_div_op.result, block.arguments[1]
                )
            else:
                block_div_op = arith.DivFOp(block.arguments[0], value.result)
                block_add_op = arith.AddFOp(
                    block_div_op.result, block.arguments[1]
                )
            block.append(value)
            block.append(block_div_op)
            block.append(block_add_op)
            block.append(linalg.YieldOp([block_add_op.result]))

    return op


def rsqrt_op(
    node: RsqrtOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor rsqrt operation.
    From buddy RsqrtOp to MLIR linalg `generic` operation.

    Note: This op, compute input node's rsqrt result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 1
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    output = tensor.EmptyOp(output_shape, mlir_dtype)
    generic_map = ir.AffineMap.get_permutation(
        [i for i in range(len(output_shape))]
    )
    op = linalg.GenericOp(
        [tensor_type],
        [input1],
        [output],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
            ]
        ),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
            * len(output_shape)
        ),
    )
    block = ir.Block.create_at_start(
        op.region,
        [
            ir.RankedTensorType(input1.type).element_type,
            ir.RankedTensorType(output.result.type).element_type,
        ],
    )
    math_rsqrt_op = math.RsqrtOp(block.arguments[0])
    block.append(math_rsqrt_op)
    block.append(linalg.YieldOp([math_rsqrt_op.result]))

    return op


def mul_op(
    node: MulOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor mul operation.
    From buddy MulOp to MLIR linalg `generic` operation.

    Note: This op, compute input node's mul result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 2
    input1 = symbol_table.get((str(node.args[0]), 0))
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    shape = list(node.tensor_meta["shape"])
    if isinstance(node.args[1], str):
        input2 = symbol_table.get((str(node.args[1]), 0))
    else:
        data = [node.args[1]]
        input2_shape = numpy.array(data).shape
        tensor_type = ir.RankedTensorType.get(input2_shape, mlir_dtype)
        element = mlir_element_attr_get(dtype, node.args[1])
        attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
        input2 = arith.ConstantOp(tensor_type, attr).result
    if input1 is None or input2 is None:
        return
    mul_result_tensor_type = ir.RankedTensorType.get(shape, mlir_dtype)
    op = tosa.MulOp(
        mul_result_tensor_type,
        input1,
        input2,
        ir.IntegerAttr.get(ir.IntegerType.get_signless(8), 0),
    )
    return op.result


def t_op(
    node: TOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor tanspose operation.
    From buddy TransposeOp to MLIR linalg `generic` operation.

    Note: This op, compute input node's transpose result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 1
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    perm = ir._denseI64ArrayAttr([1, 0], None)
    output = tensor.EmptyOp(output_shape, mlir_dtype)
    op = linalg.transpose(input=input1, outs=[output], permutation=perm)

    return op.result[0]


def matmul_op(
    node: MatmulOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor matmul operation.
    From Buddy MatmulOp to MLIR linalg `matmul` operation.

    Note: This op, compute input node's matrix multiplication result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.matmul op.
    """
    assert len(node.args) == 2
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))
    if input1 is None or input2 is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    element = mlir_element_attr_get(dtype, 0.0)
    attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
    matmul_result_buffer = arith.ConstantOp(tensor_type, attr).result
    op = linalg.matmul(input1, input2, outs=[matmul_result_buffer])
    return op


def transpose_op(
    node: TransposeOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor transpose operation.
    From buddy TransposeSpecificDimOp to MLIR linalg `generic`
    operation.

    Note: This op, compute input node's transpose result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 3
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return
    dim1 = int(node.args[1])
    dim2 = int(node.args[2])
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    output_perm = [i for i in range(len(output_shape))]
    output_perm[dim2], output_perm[dim1] = output_perm[dim1], output_perm[dim2]
    perm = ir._denseI64ArrayAttr(output_perm, None)
    output = tensor.EmptyOp(output_shape, mlir_dtype)
    op = linalg.transpose(input=input1, outs=[output], permutation=perm)

    return op.result[0]


def index_op(
    node: IndexOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor index operation.
    From buddy IndexOp to MLIR linalg `generic`
    operation.

    Note: This op, get input node slice result by input index.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    # Limitation: current implementation doesn't support advanced
    # indexing with multi-dimensional tensor(s).
    # e.g.
    # >>> t = torch.randn((13, 13))
    # >>> ind = torch.tensor([[1, 2], [1, 2]])
    # >>> t[[ind]]
    assert len(node.args) == 2
    to_index_tensor = symbol_table.get((str(node.args[0]), 0))
    assert to_index_tensor is not None
    to_index_tensor_shape = ir.RankedTensorType(to_index_tensor.type).shape
    indices = node.args[1]
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    assert len(indices) <= len(to_index_tensor_shape)

    # Step 1: find offset of the non-none arguments and none arguments.
    none_index_tensor_offset = []
    non_none_index_tensor_offset = []
    for i, tensor_extract_indices in enumerate(indices):
        if tensor_extract_indices is None:
            none_index_tensor_offset.append(i)
        else:
            non_none_index_tensor_offset.append(i)

    # Omit None argument since inductor decomp will help to decompose
    # index op with None argument to trivial index op and several
    # unsqueese ops.
    indices = [index for index in indices if index is not None]

    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    output = tensor.EmptyOp(output_shape, mlir_dtype)
    indexing_tensor_shape = ir.RankedTensorType(
        symbol_table.get((str(indices[0]), 0)).type
    ).shape

    indexing_tensors = [symbol_table.get((str(ind), 0)) for ind in indices]
    # Due to linalg.generic's shape verification logic, we need to reshape
    # the indexing tensor on none-dim. Refer
    # https://github.com/llvm/llvm-project/blob/b47f63d3c8fedf7c98b7f58e892e784fddee4601/mlir/lib/Dialect/Linalg/IR/LinalgInterfaces.cpp#L1132
    # for more details.
    for dim in range(min(len(to_index_tensor_shape), len(node.args[1]))):
        if dim not in none_index_tensor_offset:
            continue

        indexing_tensor_shape.insert(dim, 1)
        for i, ind_tensor in enumerate(indexing_tensors):
            ind_tensor_shape = ir.RankedTensorType(ind_tensor.type).shape
            ind_tensor_shape.insert(dim, 1)
            reshape_result = tosa.ReshapeOp(
                ind_tensor,
                ir._denseI64ArrayAttr(
                    numpy.array(ind_tensor_shape, dtype=numpy.int64), None
                ),
            ).output
            if to_index_tensor_shape[dim] != 1:
                repeat_result_shape = ind_tensor_shape.copy()
                repeat_result_shape[dim] = to_index_tensor_shape[dim]
                const_op_element_type = ir.RankedTensorType(
                    ind_tensor.type
                ).element_type
                const = tosa.ConstOp(
                    ir.DenseElementsAttr.get_splat(
                        ir.RankedTensorType.get(
                            repeat_result_shape, const_op_element_type
                        ),
                        ir.IntegerAttr.get(const_op_element_type, 0),
                    )
                ).output
                broadcast_result_type = ir.RankedTensorType.get(
                    repeat_result_shape, const_op_element_type
                )
                reshape_result = tosa.AddOp(
                    broadcast_result_type, const, reshape_result
                ).output

            indexing_tensors[i] = reshape_result

    # Step 2: prepare affine maps, iterator types and so on.
    generic_map = ir.AffineMap.get_permutation(
        [i for i in range(len(output_shape))]
    )

    # Affine map 1: map for indexing tensor.
    # Affine map 2: map for output tensor
    map_ = [
        ir.AffineMapAttr.get(
            generic_map.get_submap(
                [dim for dim in range(len(indexing_tensor_shape))]
            )
        )
        for _ in range(len(indices))
    ] + [
        ir.AffineMapAttr.get(
            generic_map.get_submap([dim for dim in range(len(output_shape))])
        )
    ]

    op = linalg.GenericOp(
        [tensor_type],
        indexing_tensors,
        [output],
        ir.ArrayAttr.get(map_),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
            * len(output_shape)
        ),
    )
    arguments = [
        ir.RankedTensorType(operand.type).element_type
        for operand in indexing_tensors
    ] + [ir.RankedTensorType(output.result.type).element_type]
    block = ir.Block.create_at_start(op.region, arguments)

    tensor_extract_indices = []

    block_argument_list = list(block.arguments[:-1])
    for dim in range(len(to_index_tensor_shape)):
        if dim in none_index_tensor_offset or len(block_argument_list) == 0:
            # Step 4: Extract index value for the dims with none arguments.
            # None arguments means to extract all the values in the dim.
            # Also, handle dims that are not operated by the index op in
            # current context.
            index_op = linalg.IndexOp(ir._i64Attr(dim, None))
            block.append(index_op)
            tensor_extract_indices.append(index_op.result)

        else:
            # Step 5: Extract index value for the dims with non-none arguments.
            indexcast_op = arith.IndexCastOp(
                ir.IndexType.get(), block_argument_list.pop(0)
            )
            block.append(indexcast_op)
            tensor_extract_indices.append(indexcast_op.result)

    # Step 6: use `tensor.extract` to extract the result tensor.
    value = tensor.ExtractOp(to_index_tensor, tensor_extract_indices)
    block.append(value)
    block.append(linalg.YieldOp([value.result]))

    return op


def neg_op(
    node: NegOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor neg operation.
    From buddy NegOp to MLIR linalg `negf` operation.

    Note: This op, compute input node's neg result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 1
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    output = tensor.EmptyOp(output_shape, mlir_dtype)
    op = linalg.negf(input1, outs=output)

    return op


def cat_op(
    node: CatOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor concate operation.
    From buddy CatOp to MLIR tensor `insert_slice`
    operation.

    Note: This op, concate two input tensor.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the tensor.insert_slice op.
    """
    assert len(node.args) == 2
    input1 = symbol_table.get((str(node.args[0][0]), 0))
    input2 = symbol_table.get((str(node.args[0][1]), 0))
    dim = int(node.args[1])
    if input1 is None or input2 is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    if dim < 0:
        dim = len(output_shape) + dim
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    output = tensor.EmptyOp(output_shape, mlir_dtype)
    offset = [0 for x in output_shape]
    offset_attr = ir._denseI64ArrayAttr(offset, None)
    input1_shape = ir.RankedTensorType(input1.type).shape
    size_attr = ir._denseI64ArrayAttr(input1_shape, None)
    stride_attr = ir._denseI64ArrayAttr([1] * len(offset), None)
    insert_input1 = tensor.InsertSliceOp(
        input1,
        output.result,
        [],
        [],
        [],
        offset_attr,
        size_attr,
        stride_attr,
    )
    offset[dim] += input1_shape[dim]
    offset_attr = ir._denseI64ArrayAttr(offset, None)
    input2_shape = ir.RankedTensorType(input2.type).shape
    size_attr = ir._denseI64ArrayAttr(input2_shape, None)
    insert_input2 = tensor.InsertSliceOp(
        input2,
        insert_input1.result,
        [],
        [],
        [],
        offset_attr,
        size_attr,
        stride_attr,
    )

    return insert_input2


def squeeze_op(
    node: SqueezeOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor squeeze operation.
    From buddy SqueezeOp to MLIR linalg `generic` operation.

    Note: This op, reduce the input tensor's shape dims by specified dim.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 2
    input1 = symbol_table.get((str(node.args[0]), 0))
    dim = int(node.args[1])
    if input1 is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    input1_shape = ir.RankedTensorType(input1.type).shape
    if dim < 0:
        dim = len(input1_shape) + dim
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    output = tensor.EmptyOp(output_shape, mlir_dtype)
    if input1_shape[dim] != 1:
        offset = [0 for x in output_shape]
        offset_attr = ir._denseI64ArrayAttr(offset, None)
        size_attr = ir._denseI64ArrayAttr(input1_shape, None)
        stride_attr = ir._denseI64ArrayAttr([1] * len(offset), None)
        op = tensor.InsertSliceOp(
            input1,
            output.result,
            [],
            [],
            [],
            offset_attr,
            size_attr,
            stride_attr,
        )
    else:
        output_map = ir.AffineMap.get(
            len(output_shape),
            0,
            [ir.AffineExpr.get_dim(i) for i in range(len(output_shape))],
        )
        input1_map = []
        loop_index = 0
        for i in range(len(input1_shape)):
            if len(input1_map) == dim:
                input1_map.append(ir.AffineExpr.get_constant(0))
            else:
                input1_map.append(ir.AffineExpr.get_dim(loop_index))
                loop_index += 1
        input1_map = ir.AffineMap.get(len(output_shape), 0, input1_map)
        op = linalg.GenericOp(
            [tensor_type],
            [input1],
            [output],
            ir.ArrayAttr.get(
                [
                    ir.AffineMapAttr.get(input1_map),
                    ir.AffineMapAttr.get(output_map),
                ]
            ),
            ir.ArrayAttr.get(
                [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
                * len(output_shape)
            ),
        )
        block = ir.Block.create_at_start(
            op.region,
            [
                ir.RankedTensorType(input1.type).element_type,
                ir.RankedTensorType(output.result.type).element_type,
            ],
        )
        block.append(linalg.YieldOp([block.arguments[0]]))

    return op


def batch_matmul_op(
    node: BatchMatmulOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor batch matmul operation.
    From buddy BatchMatmulOp to MLIR linalg `batch_matmul`
    operation.

    Note: This op, compute input node's batch matrix multiplication result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.batch_matmul op.
    """
    assert len(node.args) == 2
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))
    if input1 is None or input2 is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    element = mlir_element_attr_get(dtype, 0)
    attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
    zero_fill = arith.ConstantOp(tensor_type, attr).result
    op = linalg.batch_matmul(input1, input2, outs=[zero_fill])

    return op


def div_op(
    node: DivOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor divsion operation.
    From buddy DivOp to MLIR linalg `generic` operation.

    Note: This op, compute input node's division result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 2
    input1 = symbol_table.get((str(node.args[0]), 0))
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    shape = list(node.tensor_meta["shape"])
    if isinstance(node.args[1], str):
        input2 = symbol_table.get((str(node.args[1]), 0))
    else:
        data = [node.args[1]]
        input2_shape = numpy.array(data).shape
        tensor_type = ir.RankedTensorType.get(input2_shape, mlir_dtype)
        element = mlir_element_attr_get(dtype, node.args[1])
        attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
        input2 = arith.ConstantOp(tensor_type, attr).result
    if input1 is None or input2 is None:
        return
    div_result_tensor_type = ir.RankedTensorType.get(shape, mlir_dtype)
    op = tosa.MulOp(
        div_result_tensor_type,
        input1,
        tosa.ReciprocalOp(input2.type, input2).result,
        ir.IntegerAttr.get(ir.IntegerType.get_signless(8), 0),
    )
    return op.result


def softmax_op(
    node: SoftmaxOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor softmax operation.
    From buddy SoftmaxOp to MLIR linalg `generic` operation.

    Note: This op, compute input node's softmax result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 3
    assert node.args[2] is False
    input1 = symbol_table.get((str(node.args[0]), 0))
    dim = int(node.args[1])
    if input1 is None:
        return
    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    if dim < 0:
        dim += len(output_shape)
    mlir_dtype = mlir_element_type_get(dtype)

    sum_tensor_shape = copy.deepcopy(output_shape)
    sum_tensor_shape[dim] = 1
    sum_tensor_type = ir.RankedTensorType.get(sum_tensor_shape, mlir_dtype)
    element = mlir_element_attr_get(dtype, 0)
    attr = ir.DenseElementsAttr.get_splat(sum_tensor_type, element)
    sum_tensor = arith.ConstantOp(sum_tensor_type, attr).result
    input1_map = [ir.AffineExpr.get_dim(i) for i in range(len(output_shape))]
    input1_map = ir.AffineMap.get(len(output_shape), 0, input1_map)
    sum_tensor_map = [
        ir.AffineExpr.get_dim(i) for i in range(len(output_shape))
    ]
    sum_tensor_map[dim] = ir.AffineExpr.get_constant(0)
    sum_tensor_map = ir.AffineMap.get(len(output_shape), 0, sum_tensor_map)
    loop_type = [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * len(
        output_shape
    )
    loop_type[dim] = ir.Attribute.parse("#linalg.iterator_type<reduction>")
    sum_tensor_op = linalg.GenericOp(
        [sum_tensor_type],
        [input1],
        [sum_tensor],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(input1_map),
                ir.AffineMapAttr.get(sum_tensor_map),
            ]
        ),
        ir.ArrayAttr.get(loop_type),
    )
    block = ir.Block.create_at_start(
        sum_tensor_op.region,
        [
            mlir_dtype,
            mlir_dtype,
        ],
    )
    exp_op = math.ExpOp(block.arguments[0])
    add_op = arith.AddFOp(exp_op.result, block.arguments[1])
    block.append(exp_op)
    block.append(add_op)
    block.append(linalg.YieldOp([add_op.result]))
    result_tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    result_tensor = tensor.EmptyOp(output_shape, mlir_dtype)
    result_tensor_map = [
        ir.AffineExpr.get_dim(i) for i in range(len(output_shape))
    ]
    result_tensor_map = ir.AffineMap.get(
        len(output_shape), 0, result_tensor_map
    )
    op = linalg.GenericOp(
        [result_tensor_type],
        [input1, sum_tensor_op.result],
        [result_tensor.result],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(input1_map),
                ir.AffineMapAttr.get(sum_tensor_map),
                ir.AffineMapAttr.get(result_tensor_map),
            ]
        ),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
            * len(output_shape)
        ),
    )
    block = ir.Block.create_at_start(
        op.region,
        [
            mlir_dtype,
            mlir_dtype,
            mlir_dtype,
        ],
    )
    exp_op = math.ExpOp(block.arguments[0])
    div_op = arith.DivFOp(exp_op.result, block.arguments[1])
    block.append(exp_op)
    block.append(div_op)
    block.append(linalg.YieldOp([div_op.result]))

    return op


def clone_op(
    node: CloneOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor clone operation.
    From buddy CloneOp to MLIR tensor `extract_slice`
    operation.

    Note: This op, clone input tensor to a new tensor.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the tensor.extract_slice op.
    """
    assert len(node.args) == 1
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    offset = [0 for x in output_shape]
    offset_attr = ir._denseI64ArrayAttr(offset, None)
    size_attr = ir._denseI64ArrayAttr(output_shape, None)
    stride = [1 for x in output_shape]
    stride_attr = ir._denseI64ArrayAttr(stride, None)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    op = tensor.ExtractSliceOp(
        tensor_type, input1, [], [], [], offset_attr, size_attr, stride_attr
    )

    return op


def silu_op(
    node: SiluOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor silu activation operation.
    From Buddy SiluOp to MLIR linalg `generic` operation.

    Note: This op, compute input node's silu activation result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 1
    input1 = symbol_table.get((str(node.args[0]), 0))
    if input1 is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    output = tensor.EmptyOp(output_shape, mlir_dtype)
    generic_map = ir.AffineMap.get_permutation(
        [i for i in range(len(output_shape))]
    )
    op = linalg.GenericOp(
        [tensor_type],
        [input1],
        [output],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
            ]
        ),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
            * len(output_shape)
        ),
    )
    block = ir.Block.create_at_start(
        op.region,
        [
            ir.RankedTensorType(input1.type).element_type,
            ir.RankedTensorType(output.result.type).element_type,
        ],
    )
    neg_op = arith.NegFOp(block.arguments[0])
    exp_op = math.ExpOp(neg_op.result)
    one_op = arith.ConstantOp(mlir_dtype, mlir_element_attr_get(dtype, 1))
    add_op = arith.AddFOp(one_op.result, exp_op.result)
    div_op = arith.DivFOp(block.arguments[0], add_op.result)
    block.append(neg_op)
    block.append(exp_op)
    block.append(one_op)
    block.append(add_op)
    block.append(div_op)
    block.append(linalg.YieldOp([div_op.result]))

    return op


def param_extract(
    node: PlaceholderOp,
    offset,
    params_mlir_node,
):
    """
    Extract param from packed params.

    Note: This function extract slice from packed params tensor, and expand
    shape by param node shape.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the tensor.expand_shape op.
    """
    dtype_mapping = {
        TensorDType.Float32: ir.F32Type.get(),
        TensorDType.Int64: ir.IntegerType.get_signless(64),
    }
    tensor_element_type = dtype_mapping[node.tensor_meta["dtype"]]
    output_shape = list(node.tensor_meta["shape"])
    extract_size = functools.reduce(lambda x, y: x * y, output_shape, 1)
    offset_attr = ir._denseI64ArrayAttr([offset], None)
    size_attr = ir._denseI64ArrayAttr([extract_size], None)
    stride = [1]
    stride_attr = ir._denseI64ArrayAttr(stride, None)
    tensor_type = ir.RankedTensorType.get([extract_size], tensor_element_type)
    extract_slice_op = tensor.ExtractSliceOp(
        tensor_type,
        params_mlir_node,
        [],
        [],
        [],
        offset_attr,
        size_attr,
        stride_attr,
    )
    if len(output_shape) == 1 or len(output_shape) == 0:
        return extract_slice_op
    tensor_type = ir.RankedTensorType.get(output_shape, tensor_element_type)
    axis = ir.ArrayAttr.get(
        [
            ir.IntegerAttr.get(ir.IntegerType.get_signless(64), i)
            for i in range(len(output_shape))
        ],
        None,
    )
    axis = ir.ArrayAttr.get([axis], None)
    return tensor.ExpandShapeOp(tensor_type, extract_slice_op.result, axis)


def where_op(
    node: WhereOp,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor where operation.
    From Buddy WhereOp to MLIR linalg `generic` operation.

    Note: This op, compute input node's silu activation result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.generic op.
    """
    assert len(node.args) == 3
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))
    input3 = symbol_table.get((str(node.args[2]), 0))
    if input1 is None or input2 is None or input3 is None:
        return

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    tensor_type = ir.RankedTensorType.get(output_shape, mlir_dtype)
    output = tensor.EmptyOp(output_shape, mlir_dtype)
    generic_map = ir.AffineMap.get_permutation(
        [i for i in range(len(output_shape))]
    )
    op = linalg.GenericOp(
        [tensor_type],
        [input1, input3],
        [output],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
                ir.AffineMapAttr.get(
                    generic_map.get_submap(
                        [i for i in range(len(output_shape))]
                    )
                ),
            ]
        ),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")]
            * len(output_shape)
        ),
    )
    block = ir.Block.create_at_start(
        op.region,
        [
            ir.RankedTensorType(input1.type).element_type,
            ir.RankedTensorType(input3.type).element_type,
            ir.RankedTensorType(output.result.type).element_type,
        ],
    )
    select_op = arith.SelectOp(block.arguments[0], input2, block.arguments[1])
    block.append(select_op)
    block.append(linalg.YieldOp([select_op.result]))

    return op


def scalar_tensor_op(node: ScalarTensorOp, symbol_table):
    """
    Import the tensor Scalar_Tensor operation.
    From Buddy ScalarTensorOp to MLIR arith `ConstantOp` operation.
    """
    assert len(node.args) == 1
    dtype = node.tensor_meta["dtype"]
    attr = mlir_element_attr_get(dtype, node.args[0])
    op = arith.ConstantOp(dtype, attr)

    return op


def slice_scatter_op(node: SliceScatterOp, symbol_table):
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    src_tensor = symbol_table.get((str(node.args[1]), 0))
    dim = node.args[2]
    start = node.args[3]

    input_tensor_shape = ir.RankedTensorType(input_tensor.type).shape
    src_tensor_shape = ir.RankedTensorType(src_tensor.type).shape
    offset = [0] * len(input_tensor_shape)
    offset[dim] = start
    static_offset_attr = ir._denseI64ArrayAttr(offset, None)
    static_sizes_attr = ir._denseI64ArrayAttr(src_tensor_shape, None)
    stride = [1] * len(input_tensor_shape)
    if len(node.args) > 5:
        stride[dim] = node.args[5]
    static_stride_attr = ir._denseI64ArrayAttr(stride, None)

    return tensor.InsertSliceOp(
        src_tensor,
        input_tensor,
        [],
        [],
        [],
        static_offset_attr,
        static_sizes_attr,
        static_stride_attr,
    )


def index_put_op(node: IndexPutOp, symbol_table):
    dest = symbol_table.get((str(node.args[0]), 0))
    index = node.args[1]
    print(index)
    for i in index:
        if i is None:
            print("None")
        else:
            print(ir.RankedTensorType(symbol_table.get((str(i), 0)).type).shape)

    value = symbol_table.get((str(node.args[2]), 0))
    print("dest shape", ir.RankedTensorType(dest.type).shape)
    print("value shape", ir.RankedTensorType(value.type).shape)


ops_registry = {
    "param.extract": param_extract,
    "MatmulOp": matmul_op,
    "ArangeOp": arange_op,
    "UnsqueezeOp": unsqueeze_op,
    "ViewOp": view_op,
    "EmbeddingOp": embedding_op,
    "OnesOp": ones_op,
    "FullOp": full_op,
    "LessThanOp": lt_op,
    "MaskedFillOp": masked_fill_op,
    "SliceOp": slice_op,
    "ExpandOp": expand_op,
    "ToCopyOp": to_copy_op,
    "RsubOp": rsub_op,
    "PowOp": pow_op,
    "MeanOp": mean_op,
    "RsqrtOp": rsqrt_op,
    "MulOp": mul_op,
    "TOp": t_op,
    "TransposeOp": transpose_op,
    "IndexOp": index_op,
    "NegOp": neg_op,
    "CatOp": cat_op,
    "SqueezeOp": squeeze_op,
    "BatchMatmulOp": batch_matmul_op,
    "DivOp": div_op,
    "SoftmaxOp": softmax_op,
    "CloneOp": clone_op,
    "SiluOp": silu_op,
    "AddOp": add_op,
    "WhereOp": where_op,
    "ScalarTensorOp": scalar_tensor_op,
    "SliceScatterOp": slice_scatter_op,
    "IndexPutOp": index_put_op,
}
