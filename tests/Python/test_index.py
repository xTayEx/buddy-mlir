# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg


def foo(x, y):
    return x[[None, None, y]]


in1 = torch.arange(25).reshape([5, 5])
in2 = torch.tensor([1, 2])
# Initialize the dynamo compiler.
dynamo_compiler_for_test1 = DynamoCompiler(
    primary_registry=linalg.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler_for_test1.importer(foo, in1, in2)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

dynamo_compiler_for_test2 = DynamoCompiler(
    primary_registry=linalg.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

foo_mlir = torch.compile(foo, backend=dynamo_compiler_for_test2)
assert torch.equal(foo(in1, in2), foo_mlir(in1, in2))

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tensor.empty
# CHECK: %{{.*}} = linalg.generic
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
