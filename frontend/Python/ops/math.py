# ===- math.py -----------------------------------------------------------------
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
# The registry of mappings from Torch node to MLIR math dialect operations.
#
# ===---------------------------------------------------------------------------

from mlir.dialects import math
from buddy.compiler.graph.operation import ErfOp, SqrtOp


def erf_op(node: ErfOp, symbol_table):
    """
    Import the tensor erf operation.
    From Buddy ErfOp to MLIR TOSA `erf` operation.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.ErfOp(input_tensor)
    return op

def sqrt_op(node: SqrtOp, symbol_table):
    """
    Import the tensor sqrt operation.
    From Buddy SqrtOp to MLIR TOSA `sqrt` operation.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    return math.SqrtOp(input_tensor)


ops_registry = {
    "ErfOp": erf_op,
    "SqrtOp": sqrt_op,
}
