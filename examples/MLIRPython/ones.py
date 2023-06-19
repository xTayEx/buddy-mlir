from buddy.compiler import DynamoCompiler
import torch
import torch._dynamo as dynamo

def foo():
  return torch.ones(((2, 3)))

foo_mlir = dynamo.optimize(DynamoCompiler)(foo)
foo_mlir()