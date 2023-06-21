from buddy.compiler import DynamoCompiler
import torch
import torch._dynamo as dynamo

def foo(x, y, scalar):
  return x + scalar * y

foo_mlir = dynamo.optimize(DynamoCompiler)(foo)
scalar = torch.Tensor([2.0] * 10)
in1 = torch.randn(10)
in2 = torch.randn(10)

foo_mlir(in1, scalar, in2)
