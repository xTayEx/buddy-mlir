import torch
import torch._dynamo as dynamo


def foo(x, idx):
  return x[idx]


foo_inductor = dynamo.optimize()(foo)
x = torch.randn(10000)
idx = 1
foo_inductor(x, idx)
