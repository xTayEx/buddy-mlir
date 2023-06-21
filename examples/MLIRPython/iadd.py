import torch
import torch._dynamo as dynamo


def foo(x, y):
  x += y


foo_inductor = dynamo.optimize()(foo)
t1 = torch.randn(10000, 10000)
t2 = torch.randn(10000, 10000)
foo_inductor(t1, t2)
