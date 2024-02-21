# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x, y):
    return x[:, :1] + y


# x = torch.randn(3, 5, 2)
x = torch.arange(10).reshape(2, 5)
y = torch.arange(2).reshape(2, 1)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler.importer(foo, x, y)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

foo_mlir = torch.compile(foo, backend=dynamo_compiler)
assert torch.equal(foo_mlir(x, y), foo(x, y))

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tensor.extract_slice
# CHECK: return %{{.*}} : tensor<2x1xi64>
# CHECK: }
# CHECK: }
