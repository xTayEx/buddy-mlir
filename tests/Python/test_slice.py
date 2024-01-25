# RUN: %PYTHON %s 2>&1 | FileCheck %s
import random

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa



class Slice(torch.nn.Module):
    def forward(self, x, dim, start_idx, end_idx=4):
        start_idx = start_idx + 1
        return torch.ops.aten.slice(x, dim, start_idx, end_idx)

def foo(x, dim, start_idx, end_idx):
    return torch.ops.aten.slice(x, dim, start_idx, end_idx)

x = torch.randn(3, 5, 2)
dim = 1
start_idx = random.randint(0, 3)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

slice_module = Slice()
torch.compile()

graphs = dynamo_compiler.importer(slice_module, x, dim, start_idx)
assert len(graphs) == 1
graph = graphs[0]
graph.lower_to_top_level_ir()
print(graph._imported_module)

# CHECK: module {
# CHECK-LABEL: func.func @forward
# CHECK: %{{.*}} = tensor.extract_slice
# CHECK: return %{{.*}} : tensor<3x2x2xf32>
# CHECK: }
# CHECK: }
