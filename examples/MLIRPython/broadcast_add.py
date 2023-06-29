from buddy.compiler import DynamoCompiler
import torch
import torch._dynamo as dynamo

def foo(x, y):
  return x + y

foo_mlir = dynamo.optimize(DynamoCompiler)(foo)
in1 = torch.randn((10, 2))
in2 = torch.randn((1, 2))
foo_mlir(in1, in2)
