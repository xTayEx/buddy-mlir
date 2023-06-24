from buddy.compiler import DynamoCompiler
import torch
import torch._dynamo as dynamo

def foo(x):
  return torch.nn.functional.gelu(x)

foo_mlir = dynamo.optimize(DynamoCompiler)(foo)
x = torch.randn(10)
foo_mlir(x)
