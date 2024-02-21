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
from pathlib import Path
from typing import List

import torch
from torch._inductor.decomposition import decompositions as inductor_decomp
from torch._functorch.aot_autograd import aot_module_simplified

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from transformer_with_kvcache import Transformer


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    def _compiler(
        _gm: torch.fx.GraphModule, _example_inputs: List[torch.Tensor]
    ):
        # ops_set = set()
        _gm.graph.print_tabular()
        # for node in _gm.graph.nodes:
        #     try:
        #         ops_set.add(node.target.__name__)
        #     except AttributeError:
        #         pass

        # for op in ops_set:
        #     print(op)
        return _gm.forward

    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=_compiler,
        decompositions=inductor_decomp,
    )  # return a python callable


def prefill(
    model_: Transformer,
    x: torch.Tensor,
    input_pos_: torch.Tensor,
    **sampling_kwargs,
) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model_(x, input_pos_)
    # return sample(logits, **sampling_kwargs)[0]
    return logits


MAX_NEW_TOKENS = 40

# Retrieve the LLaMA model path from environment variables.
model_path = os.environ.get("LLAMA_MODEL_PATH")
if model_path is None:
    raise EnvironmentError(
        "The environment variable 'LLAMA_MODEL_PATH' is not set or is invalid."
    )

model = Transformer.from_name(model_path)
checkpoint = torch.load(
    str(Path(model_path) / "model.pth"), mmap=True, weights_only=True
)
model.load_state_dict(checkpoint, assign=True)
model = model.to(device="cpu", dtype=torch.float32)
model.eval()


# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

prompt = torch.tensor([1 for _ in range(10)], dtype=torch.int64)
T = prompt.size(0)
T_new = T + MAX_NEW_TOKENS
max_seq_length = min(T_new, model.config.block_size)

model.setup_caches(1, max_seq_length)
empty = torch.empty(T_new, dtype=torch.int64)
empty[:T] = prompt
seq = empty
input_pos = torch.arange(0, T, dtype=torch.int64)

with torch.no_grad():
    # prefill_opt = torch.compile(prefill, backend=my_compiler)
    # prefill_opt(
    #     model, prompt.view(1, -1), input_pos, temperature=1.0, top_k=None
    # )
    graphs = dynamo_compiler.importer(
        prefill,
        model,
        prompt.view(1, -1),
        input_pos,
        temperature=1.0,
        top_k=None,
    )
    # print(len(graphs))
    # graphs[0].lower_to_top_level_ir(do_params_pack=True)
    # print(graphs[0]._imported_module)

# # Import the model into MLIR module and parameters.
# with torch.no_grad():
#     data = torch.tensor([[1 for i in range(40)]], dtype=torch.int64)
#     graphs = dynamo_compiler.importer(model, data)

# assert len(graphs) == 1
# graph = graphs[0]
# params = dynamo_compiler.imported_params[graph]
# graph.lower_to_top_level_ir(True)
# path_prefix = os.path.dirname(os.path.abspath(__file__))
# # Write the MLIR module to the file.
# with open(os.path.join(path_prefix, "llama.mlir"), "w") as module_file:
#     print(graph._imported_module, file=module_file)

# # Concatenate all parameters into a single numpy array and write to a file.
# all_param = numpy.concatenate(
#     [param.detach().numpy().reshape([-1]) for param in params]
# )
# all_param.tofile(os.path.join(path_prefix, "arg0.data"))
