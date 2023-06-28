from typing import List

import mlir.dialects.func as func
import torch

import mlir.ir
from mlir.ir import *
from mlir.passmanager import *
from loguru import logger

from .operators_gen import OpCodeGen


def DynamoCompiler(gm: torch.fx.GraphModule, inputs: List[torch.Tensor]):
  print("Custom Compiler from FX Graph to MLIR:")
  print("-------------------------------------------------------------------")
  gm.graph.print_tabular()
  # Initialize the MLIR context.
  ctx = Context()
  with Location.unknown(ctx):
    module = Importer(gm, inputs)
    module = Lowering(module)
  return gm.forward


def Importer(gm: torch.fx.GraphModule, inputs: List[torch.Tensor]):
  # Initialize the symbol table.
  symbolTable = {}
  # Create a module and build the operations into the module.
  module = Module.create()
  with InsertionPoint(module.body):
    # Parse the arguments.
    arguments = []
    for arg in inputs:
      shapeList = list(arg.shape)
      f32 = F32Type.get()
      tensorArg = RankedTensorType.get(shapeList, f32)
      arguments.append(tensorArg)

    # Generate the function.
    @func.FuncOp.from_py_func(*arguments)
    def generated_func(*args):
      # Convert arguments tuple into a list.
      argsList = list(args)
      # Traverse the graph and generate IR.
      for node in gm.graph.nodes:
        if node.op != "output":
          symbolTable[str(node.name)] = CodeGen(node, symbolTable, argsList)
        else:
          symbolTable["output"] = CodeGen(node, symbolTable, argsList)

      return symbolTable.get("output")
  print("-------------------------------------------------------------------")
  print("Printing the symbol table ...")
  for symbol, op in symbolTable.items():
    print(symbol, ": ", op)
  print("-------------------------------------------------------------------")
  print("Printing the generated MLIR ...")
  print(module)
  return module


def CodeGen(node, symbolTable, argsList):
  if node.op == "placeholder":
    # Bind the placeholder with args.
    placeholderName = argsList[0]
    argsList.pop(0)
    return placeholderName
  if node.op == "call_function":
    # Parse a call_function operation.
    opName = node.target.__name__
    return OpCodeGen[opName](node, symbolTable)
  if node.op == "output":
    # Generating return operation.
    ret = symbolTable.get(str(node._args[0][0]))
    return ret


def Lowering(module: Module):
  print("-------------------------------------------------------------------")
  print("Bufferizing the module ...")
  pm = PassManager('builtin.module')
  pm.add("func.func(tosa-to-linalg)")
  pm.add("func.func(tosa-to-tensor)")
  pm.add("empty-tensor-to-alloc-tensor")
  pm.add("convert-elementwise-to-linalg")
  pm.add("arith-bufferize")
  pm.add("func.func(linalg-bufferize)")
  pm.add("func.func(tensor-bufferize)")
  pm.add("func-bufferize")
  pm.run(module.operation)
  print(module)
  print("-------------------------------------------------------------------")
  print("Lowering the module to LLVM dialect ...")
  pm.add("func.func(buffer-deallocation)")
  pm.add("func.func(convert-linalg-to-loops)")
  pm.add("convert-scf-to-cf")
  pm.add("convert-linalg-to-llvm")
  pm.add("convert-arith-to-llvm")
  pm.add("expand-strided-metadata")
  pm.add("finalize-memref-to-llvm")
  pm.add("convert-func-to-llvm")
  pm.add("reconcile-unrealized-casts")
  pm.run(module.operation)
  print(module)
  return module
