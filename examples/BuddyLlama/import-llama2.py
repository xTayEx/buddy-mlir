# ===- import-llama2.py --------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This is the test of llama2 model.
#
# ===---------------------------------------------------------------------------

import os
import time

import numpy
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch._functorch.aot_autograd import aot_autograd_decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


# Retrieve the LLaMA model path from environment variables.
model_path = os.environ.get("LLAMA_MODEL_PATH")
if model_path is None:
    raise EnvironmentError(
        "The environment variable 'LLAMA_MODEL_PATH' is not set or is invalid."
    )

# Initialize the tokenizer and model from the specified model path.
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path, torchscript=True)
print(type(model))

# Initialize Dynamo Compiler with specific configurations as an importer.
prefill_dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=aot_autograd_decompositions,
)

decode_dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=aot_autograd_decompositions,
)

data = torch.tensor([[1 for _ in range(40)]], dtype=torch.int64)

# TODO: we just need the shape of `past_key_values`. So instead of 
# calling the model, calculate the shape directly is more efficient.
model_output = model(data, use_cache=True, return_dict=True)
past_key_values = model_output.past_key_values

with torch.no_grad():
    prefill_graphs = prefill_dynamo_compiler.importer(model, data)
    assert len(prefill_graphs) == 1
    prefill_graph = prefill_graphs[0]
    prefill_graph.lower_to_top_level_ir(do_params_pack=True)

    decode_graphs = decode_dynamo_compiler.importer(
        model, data, past_key_values=past_key_values
    )
    assert len(decode_graphs) == 1
    decode_graph = decode_graphs[0]
    decode_graph.lower_to_top_level_ir(do_params_pack=True)

params = decode_dynamo_compiler.imported_params[decode_graph]
path_prefix = os.path.dirname(os.path.abspath(__file__))

# Write prefill and decode modules to files.
with open(os.path.join(path_prefix, "llama_prefill.mlir"), "w") as prefill_module_file:
    prefill_module_file.write(str(prefill_graph._imported_module))

with open(os.path.join(path_prefix, "llama_decode.mlir"), "w") as decode_module_file:
    decode_module_file.write(str(decode_graph._imported_module))


# Concatenate all parameters into a single numpy array and write to a file.
all_param = numpy.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
)
all_param.tofile(os.path.join(path_prefix, "arg0.data"))
