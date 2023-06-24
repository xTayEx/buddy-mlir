import time as timer
import logging
import torch
import torch._dynamo
import torch._dynamo.config
from typing import List
import graphviz

torch._dynamo.config.log_level = logging.INFO
torch._dynamo.config.output_code = True

import transformers
from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments, BertConfig

def DrawGraphviz(gm: torch.fx.GraphModule):
  dot = graphviz.Digraph(format="png")
  for node in gm.graph.nodes:
    print(node.op)
    if node.op == "call_function":
      dot.node(str(id(node)), label=node.name, shape="box")
    elif node.op == "call_method":
      dot.node(str(id(node)), label=node.name, shape="parallelogram")
    else:
      dot.node(str(id(node)), label=node.name, shape="ellipse")
  
  for node in gm.graph.nodes:
    for used_node in node._args:
      if isinstance(used_node, torch.fx.Node):
        dot.edge(str(id(used_node)), str(id(node)))
  
  dot.render("transformer_fx_graph", view=False)

def dynamo_debug_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
  DrawGraphviz(gm)
  gm.graph.print_tabular()
  op_set = set()
  for node in gm.graph.nodes:
    if node.op == "call_function":
      op_set.add(node.target.__name__)
  
  print(op_set)

  return gm.forward

def prepare_inputs(**kwargs):
  vocab_size = kwargs['vocab_size']
  batch_size = kwargs['batch_size']
  sequence_length = kwargs['sequence_length']
  input_ids = torch.randint(vocab_size, (batch_size, sequence_length),
                            dtype=torch.long)
  return input_ids

def prepare_train_model(config):
  model = transformers.models.bert.modeling_bert.BertForMaskedLM(config)
  model.train()
  return model

def train(model, inps):
  loss = model(inps, labels=inps)[0]
  loss.backward()

def timed(fn):
  start = timer.time()
  result = fn()
  end = timer.time()
  return result, end - start


N_ITER = 100

def train_bench():
  config_1_lay = BertConfig(num_hidden_layers=1)

  args = {
    'vocab_size': 30522,
    'batch_size': 8,
    'sequence_length':128
  }
    
  inputs = prepare_inputs(**args)
  model = prepare_train_model(config_1_lay)
  opt = torch.optim.Adam(model.parameters())
    
  # timing eager 
  # eager_times = []
  # for i in range(N_ITER):
  #   opt.zero_grad(True)
  #   _, time = timed(lambda: train(model, inputs))
  #   opt.step()
  #   eager_times.append(time)
  #   print(f"eager train time {i}: {time}")
  # print('=' * 20)

  inputs = prepare_inputs(**args)
  model = prepare_train_model(config_1_lay)
  opt = torch.optim.Adam(model.parameters())
  compiled_train = torch.compile(train, backend=dynamo_debug_backend)

  # timing dynamo
  dynamo_times = []
  for i in range(N_ITER):
    opt.zero_grad(True)
    _, time = timed(lambda: compiled_train(model, inputs))
    opt.step()
    dynamo_times.append(time)
    # print(f"dynamo train time {i}: {time}")
  # print('=' * 20)

if __name__ == "__main__":
  train_bench()
