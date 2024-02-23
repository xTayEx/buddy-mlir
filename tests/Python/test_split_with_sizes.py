# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


def foo(x, t1_add, t2_add):
    t1, t2 = torch.split(x, [1, 4], 1)
    return t1 + t1_add, t2 + t2_add


x = torch.arange(10).reshape(2, 5)
t1_add = torch.arange(2).reshape(2, 1)
t2_add = torch.arange(8).reshape(2, 4)

# Initialize the dynamo compiler.
dynamo_compiler_for_test1 = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

graphs = dynamo_compiler_for_test1.importer(foo, x, t1_add, t2_add)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

dynamo_compiler_for_test2 = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)
foo_mlir = torch.compile(foo, backend=dynamo_compiler_for_test2)
foo_mlir_result = foo_mlir(x, t1_add, t2_add)
foo_result = foo(x, t1_add, t2_add)
assert torch.equal(foo_mlir_result[0], foo_result[0])
assert torch.equal(foo_mlir_result[1], foo_result[1])
# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tosa.slice
# CHECK: return %{{.*}}
# CHECK: }
# CHECK: }
