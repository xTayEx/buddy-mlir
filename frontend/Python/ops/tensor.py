from mlir import ir
from mlir.dialects import tensor, scf, arith
from buddy.compiler.graph import SliceOp


def slice_op(node: SliceOp, symbol_table):
    """
    Import the slice operation.
    From buddy graph ir's `SliceOp` operator to MLIR TOSA `extract_slice`
    operation.
    """

    def _normlize_index(index: int | str):
        if isinstance(index, int):
            index = (index + sizes[dim]) % sizes[dim]
            index = min(max(0, index), sizes[dim])
        else:
            index = symbol_table.get((str(index), 0))
            minus_one_const = arith.constant(-1)
            lt_op = arith.cmpi("slt", index, minus_one_const)
            if_op = scf.IfOp(lt_op.result, [], hasElse=False)
            if_block = ir.Block.create_at_start(if_op.regions[0], [])
            if_block.append(
                arith.AddIOp(index, arith.constant(sizes[dim])).result
            )
        return index

    # retrieve input tensor and do some normalization
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    dim = node.args[1]
    start_idx = _normlize_index(node.args[2])
    end_idx = _normlize_index(node.args[3])
    sizes = ir.RankedTensorType(input_tensor.type).shape

    has_dynamic_dim = not (
        isinstance(start_idx, int) and isinstance(end_idx, int)
    )

    # prepare new_sizes
    new_sizes = [x for x in sizes]
    if not has_dynamic_dim:
        new_sizes[dim] = end_idx - start_idx
    else:
        new_sizes[dim] = ir.ShapedType.get_dynamic_size()

    new_sizes_attr = ir._denseI64ArrayAttr(new_sizes, None)

    # prepare offsets
    offsets = [0] * len(sizes)
    if not has_dynamic_dim:
        offsets[dim] = start_idx
    else:
        offsets[dim] = ir.ShapedType.get_dynamic_size()

    offsets_attr = ir._denseI64ArrayAttr(offsets, None)

    # prepare strides
    strides = [1] * len(sizes)
    strides_attr = ir._denseI64ArrayAttr(strides, None)

    # construct tensor.ExtractSliceOp
    result_element_type = ir.RankedTensorType(input_tensor.type).element_type
    extract_slice_result_type = ir.RankedTensorType.get(
        new_sizes, result_element_type
    )
    if not has_dynamic_dim:
        return tensor.ExtractSliceOp(
            extract_slice_result_type,
            input_tensor,
            [],
            [],
            [],
            offsets_attr,
            new_sizes_attr,
            strides_attr,
        )
    else:
        return tensor.ExtractSliceOp(
            extract_slice_result_type,
            input_tensor,
            [],
            new_sizes_attr,
            [],
            offsets_attr,
            [],
            strides_attr,
        )


ops_registry = {
    "SliceOp": slice_op,
}
