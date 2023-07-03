import operator

from buddy.compiler import DynamoCompiler
import torch
import torch._dynamo as dynamo


def foo(tup, idx):
  return operator.getitem(tup, idx)


foo_mlir = dynamo.optimize(DynamoCompiler)(foo)
tup = torch.tensor([[2.0], [1.0]])
foo_mlir(tup, 1)
