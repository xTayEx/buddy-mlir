import torch
from typing import List
from torch._functorch.aot_autograd import aot_module_simplified
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from datasets import load_dataset

ops_set = set()


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    def _compiler(_gm, _example_inputs):
        # _gm.print_readable()
        for node in _gm.graph.nodes:
            if str(node.op) != "placeholder":
                continue
            ops_set.add(node.name)

        print(ops_set)
        return gm.forward  # return a python callable

    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=_compiler,
        decompositions=inductor_decomp,
    )


dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
sample = ds[0]["audio"]

processor = WhisperProcessor.from_pretrained("openai/whisper-base")
input_features = processor(
    sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
).input_features
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

# model_opt = torch.compile(model, backend=my_compiler)
# model_opt(input_features)
