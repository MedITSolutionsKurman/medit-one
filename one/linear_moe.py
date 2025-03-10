import torch
from torch import nn
import torch.nn.functional as F


class LinearMoE(nn.Linear):
    """
    Blazing fast Mixture of Experts implementation with the same functionality.
    Safety checks and multiple clamping operations have been reduced to optimize speed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: any = None,
        dtype: any = None,
        num_experts: int = 1000,
        r: int = 4,
        top_k: int = 4,
    ):
        super(LinearMoE, self).__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.num_experts = num_experts
        # r now represents the number of gating dimensions.
        self.r = r
        self.top_k = min(top_k, num_experts)  # Ensure top_k doesn't exceed num_experts

        # Gate maps input to r dimensions.
        self.gate = nn.Linear(in_features, r, bias=False)
        # Experts is a 1D tensor; we reshape it in forward for broadcasting.
        self.experts = nn.Parameter(torch.ones(num_experts), requires_grad=True)
        # One bias per expert.
        self.bias2 = nn.Parameter(
            torch.zeros(num_experts, in_features), requires_grad=True
        )
        self.activation = nn.SiLU()

        # Epsilon for numerical stability (could be used in conditional debug mode)
        self.eps = 1e-8

        self.reset_parameters()

    def forward(self, x):
        # Record the original shape and flatten if needed.
        orig_shape = x.shape
        if len(x.shape) > 2:
            x = x.reshape(-1, x.size(-1))

        # Clamp the input for safety (can be removed in production if inputs are trusted)
        x = x.clamp(-1e4, 1e4)

        # Compute the gate output and apply softmax.
        gate = F.softmax(self.gate(x), dim=-1)  # (B, r)

        # Reshape experts for broadcasting: (1, 1, num_experts)
        experts = self.experts.clamp(-10.0, 10.0).view(1, 1, self.num_experts)
        # Compute experts weight: (B, r, num_experts)
        experts_weight = gate.unsqueeze(-1) * experts

        # Average across the gating dimension to get per-expert scores: (B, num_experts)
        expert_scores = experts_weight.mean(dim=1)
        # Clamp scores and compute contributions using the activation.
        expert_scores = expert_scores.clamp(-10, 10)
        # Compute expert contributions and reshape to (B, num_experts, 1)
        expert_contribution = (1.0 + self.activation(expert_scores)).unsqueeze(-1)

        # Expand input to all experts: from (B, in_features) to (B, num_experts, in_features)
        x_expanded = x.unsqueeze(1).expand(-1, self.num_experts, -1)

        # Scale each expert's version of the input.
        x_scaled = x_expanded * expert_contribution

        # Add the corresponding expert bias.
        x_biased = x_scaled + self.bias2.unsqueeze(0)

        # Sum across the in_features dimension: result (B, num_experts)
        x_biased_sum = x_biased.sum(dim=-1)

        # Select top-k experts based on the summed values.
        k = min(self.top_k, x_biased_sum.size(-1))
        _, indices = torch.topk(x_biased_sum, k, dim=-1, sorted=False)

        # Gather the expert outputs corresponding to the top-k indices.
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
        gathered = torch.gather(x_biased, 1, gather_indices)  # (B, k, in_features)
        # Average the selected experts.
        result = gathered.mean(dim=1)  # (B, in_features)

        # Apply the main linear transformation.
        result = super().forward(result)

        # Reshape the result back to the original dimensions if necessary.
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
