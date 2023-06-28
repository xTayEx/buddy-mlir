from buddy.compiler import DynamoCompiler
import torch
import torch._dynamo as dynamo


def foo(x, y, z):
  return x + y + z


foo_mlir = dynamo.optimize(DynamoCompiler)(foo)
in1 = torch.randn(10)
in2 = torch.randn(10)
in3 = torch.randn(10)
foo_mlir(in1, in2, in3)
