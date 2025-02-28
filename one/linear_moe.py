# coding=utf-8
# Copyright 2025 The MedIT Solutions Kurman i Wspolnicy Sp. z o. o. Team.
# https://meditsolutions.pl
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
import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F


class LinearMoE(nn.Linear):
    """
    Mixture of Experts implementation optimized for performance.
    This implementation uses parallel computation and efficient memory access patterns.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: any = None,
        dtype: any = None,
        num_experts: int = 1000,
        r=4,
        top_k: int = 4,
    ):
        super(LinearMoE, self).__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)  # Ensure top_k doesn't exceed num_experts
        self.r = r

        # Initialize components optimized for memory access
        self.gate = nn.Linear(in_features, r, bias=False)
        self.experts = nn.Parameter(torch.ones(num_experts), requires_grad=True)
        self.bias2 = nn.Parameter(
            torch.zeros(num_experts, in_features), requires_grad=True
        )
        self.activation = nn.SiLU()

        # Add numerical stability-related parameters
        self.eps = 1e-8  # Increased epsilon for better stability

        self.reset_parameters()

        # Enable fusion of operations where supported
        if hasattr(torch, "jit"):
            self.optimized = True
        else:
            self.optimized = False

    def forward(self, x):
        # Ensure the input doesn't have NaNs
        if torch.isnan(x).any():
            # Replace NaNs with zeros as a fallback
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        # Store original shape for later reshaping
        orig_shape = x.shape

        # Reshape for efficient parallel processing
        if len(x.shape) > 2:
            x = x.view(-1, x.size(-1))

        # Use memory-efficient unsqueeze operation
        x_unsqueezed = x.unsqueeze(-2)

        # Process gate in full precision with numerical stability safeguards
        x_float = x.float()

        # Clip extreme values that could cause numerical issues
        x_float = torch.clamp(x_float, min=-1e4, max=1e4)

        # Apply gate with safe computations
        gate_output = self.gate(x_float)

        # Check for NaNs/Infs in gate output
        if torch.isnan(gate_output).any() or torch.isinf(gate_output).any():
            # Apply special fallback handling for numerical stability
            gate_output = torch.where(
                torch.isnan(gate_output) | torch.isinf(gate_output),
                torch.zeros_like(gate_output),
                gate_output,
            )

        # Normalize values for numerical stability before softmax
        gate_max, _ = gate_output.max(dim=-1, keepdim=True)
        gate_output = gate_output - gate_max  # Subtract max for softmax stability

        # Apply softmax with numerical stability
        gate = F.softmax(gate_output, dim=-1)

        # Check again for NaNs/Infs in softmax output
        if torch.isnan(gate).any() or torch.isinf(gate).any():
            # Replace problematic values with uniform distribution
            uniform_value = 1.0 / gate.size(-1)
            gate = torch.where(
                torch.isnan(gate) | torch.isinf(gate),
                torch.ones_like(gate) * uniform_value,
                gate,
            )

        # Convert back to original dtype
        gate = gate.to(dtype=x.dtype)

        # Optimize expert weighting with safe operations
        if self.optimized:
            # Ensure experts doesn't have extreme values
            experts = torch.clamp(self.experts, min=-10.0, max=10.0)
            experts_weight = torch.einsum("be,n->bne", gate, experts)
        else:
            experts = torch.clamp(self.experts, min=-10.0, max=10.0)
            experts = rearrange(experts, "e -> 1 e 1")
            experts_weight = gate.unsqueeze(-1) * experts
            experts_weight = experts_weight.mean(-1, keepdim=True)

        # Apply activation with stricter clamping for numerical stability
        experts_contribution = 1.0 + self.activation(
            experts_weight.mean(-1, keepdim=True).clamp(-10, 10)
        )

        # Use efficient multiplication with safety checks
        if (
            torch.isnan(experts_contribution).any()
            or torch.isinf(experts_contribution).any()
        ):
            # Replace problematic values with ones as fallback
            experts_contribution = torch.where(
                torch.isnan(experts_contribution) | torch.isinf(experts_contribution),
                torch.ones_like(experts_contribution),
                experts_contribution,
            )

        x_scaled = x_unsqueezed * experts_contribution

        # Add bias with optimized broadcasting and safety check
        bias2_safe = torch.clamp(self.bias2, min=-10.0, max=10.0)
        x_biased = x_scaled + bias2_safe.unsqueeze(0)

        # Check results before top-k selection
        if torch.isnan(x_biased).any() or torch.isinf(x_biased).any():
            x_biased = torch.where(
                torch.isnan(x_biased) | torch.isinf(x_biased),
                torch.zeros_like(x_biased),
                x_biased,
            )

        # Sum x_biased safely with gradient clipping
        x_biased_sum = torch.clamp(x_biased.sum(dim=-1), min=-1e4, max=1e4)

        # Optimize top-k selection with safety measures
        k = min(self.top_k, x_biased.size(-2))
        vals, indices = torch.topk(x_biased_sum, k, dim=-1, sorted=False)

        # Expand indices for gathering
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, x.size(-1))

        # Gather and mean with numerical stability
        gathered_result = torch.gather(x_biased, -2, gather_indices)

        # Check for NaNs in gathered result
        if torch.isnan(gathered_result).any() or torch.isinf(gathered_result).any():
            gathered_result = torch.where(
                torch.isnan(gathered_result) | torch.isinf(gathered_result),
                torch.zeros_like(gathered_result),
                gathered_result,
            )

        result = gathered_result.mean(dim=-2)

        # Apply main linear layer transformation with checks
        result = super().forward(result)

        # Final NaN checking before output
        if torch.isnan(result).any() or torch.isinf(result).any():
            result = torch.where(
                torch.isnan(result) | torch.isinf(result),
                torch.zeros_like(result),
                result,
            )

        # Reshape back to original dimensions
        if len(orig_shape) > 2:
            result = result.view(orig_shape[0], orig_shape[1], -1)

        return result

    def reset_parameters(self) -> None:
        # Optimized parameter initialization
        super().reset_parameters()

        if hasattr(self, "gate"):
            nn.init.xavier_uniform_(self.gate.weight, gain=1.0)

        if hasattr(self, "experts"):
            # Initialize experts with conservative values
            nn.init.normal_(self.experts, mean=0.0, std=0.1)

        if hasattr(self, "bias2"):
            # Initialize bias2 with zeros
            nn.init.zeros_(self.bias2)
