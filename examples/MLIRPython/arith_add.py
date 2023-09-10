from buddy.compiler import DynamoCompiler
import torch
import torch._dynamo as dynamo


def foo(x, y):
    return x + y


foo_mlir = dynamo.optimize(DynamoCompiler)(foo)
in1 = torch.tensor([1, 2, 3])
in2 = torch.tensor([4, 5, 6])
result = foo_mlir(in1, in2)
print(result)
