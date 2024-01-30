from typing import List
import argparse

import torch
from torch._functorch.aot_autograd import aot_module_simplified
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torch._dynamo as dynamo

torch._dynamo.config.suppress_errors = True

from torch._inductor.decomposition import decompositions as inductor_decomp


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    def _compiler(_gm, _example_inputs):
        opcodes = set()
        ops = set()
        for node in _gm.graph.nodes:
            opcodes.add(node.op)
            if str(node.op) != "call_function":
                continue
            ops.add(str(node.target.__name__))

        print("-------------------------------------------------")
        print(opcodes)
        print(ops)
        return _gm.forward  # return a python callable

    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=_compiler,
        decompositions=inductor_decomp,
    )


class WhisperModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-base"
        )
        self.model.eval()

    def forward(self, input):
        return self.model.generate(input)


parser = argparse.ArgumentParser()
parser.add_argument("--use_generate", action="store_true")
args = parser.parse_args()

# load model and processor
model_path = "openai/whisper-base"
processor = WhisperProcessor.from_pretrained(model_path)


# load dummy dataset and read audio files
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
sample = ds[0]["audio"]
input_features = processor(
    sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
).input_features

if args.use_generate:
    whisper_model = WhisperModel()
else:
    whisper_model = WhisperForConditionalGeneration.from_pretrained(model_path)
    whisper_model.config.forced_decoder_ids = None

model_opt = dynamo.optimize(backend=my_compiler)(whisper_model)
if args.use_generate:
    model_opt(input_features)
else:
    model_opt(input_features, decoder_input_ids=torch.tensor([[50258]]))
