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

import numpy
import torch
from pathlib import Path
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch._functorch.aot_autograd import aot_autograd_decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from .model import Transformer


# Retrieve the LLaMA model path from environment variables.
model_path_env = os.environ.get("LLAMA_MODEL_PATH")
if model_path_env is None:
    raise EnvironmentError(
        "The environment variable 'LLAMA_MODEL_PATH' is not set or is invalid."
    )

model_path = Path(model_path_env)

# Initialize the tokenizer and model from the specified model path.
model = Transformer.from_name(str(model_path))
checkpoint = torch.load(str(model_path / "model.pth"), mmap=True, weights_only=True)
model.load_state_dict(checkpoint, assign=True)
model = model.to("cuda")
model = model.eval()
print(model)

# model = LlamaForCausalLM.from_pretrained(model_path, torchscript=True)
# model.config.use_cache = False

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=aot_autograd_decompositions,
)

# Import the model into MLIR module and parameters.
with torch.no_grad():
    data = torch.tensor([[1 for i in range(40)]], dtype=torch.int64)
    graphs = dynamo_compiler.importer(model, data)

assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
graph.lower_to_top_level_ir(True)
path_prefix = os.path.dirname(os.path.abspath(__file__))
# Write the MLIR module to the file.
with open(os.path.join(path_prefix, "llama.mlir"), "w") as module_file:
    print(graph._imported_module, file=module_file)

# Concatenate all parameters into a single numpy array and write to a file.
all_param = numpy.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
)
all_param.tofile(os.path.join(path_prefix, "arg0.data"))
