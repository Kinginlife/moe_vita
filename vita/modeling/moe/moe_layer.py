"""
MoE Layer that combines multiple experts with routing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .expert import Expert
from .router import Router


class MoELayer(nn.Module):
    """Mixture of Experts layer."""

    def __init__(self, d_model, d_ffn, num_experts, router_dim=512, top_k=1, dropout=0.0, soft_temp=2.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ffn, dropout) for _ in range(num_experts)
        ])

        # Router network
        self.router = Router(d_model, num_experts, router_dim, soft_temp)

    def forward(self, x, task_id=None):
        """
        Args:
            x: [B, N, d_model] input features
            task_id: [B] optional task IDs for training supervision
        Returns:
            output: [B, N, d_model]
            routing_loss: scalar or None
        """
        B, N, D = x.shape

        # Get routing logits
        router_logits, routing_loss = self.router(x, task_id)  # [B, num_experts]

        # Use router to select experts (both training and inference)
        actual_k = min(self.top_k, self.num_experts)

        if actual_k == 1:
            # Top-1: batch by expert for efficiency
            expert_ids = router_logits.argmax(dim=-1)  # [B]
            output = torch.zeros_like(x)

            for expert_id in range(self.num_experts):
                mask = expert_ids == expert_id
                if mask.any():
                    batch_indices = mask.nonzero(as_tuple=True)[0]
                    expert_input = x[batch_indices]
                    expert_output = self.experts[expert_id](expert_input)
                    output[batch_indices] = expert_output.to(x.dtype)
        else:
            # Top-K: weighted combination
            topk_weights, topk_indices = torch.topk(router_logits, actual_k, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1)  # [B, actual_k]

            output = torch.zeros_like(x)
            for expert_id in range(self.num_experts):
                mask = (topk_indices == expert_id).any(dim=-1)
                if mask.any():
                    batch_indices = mask.nonzero(as_tuple=True)[0]
                    expert_input = x[batch_indices]
                    expert_output = self.experts[expert_id](expert_input).to(x.dtype)

                    for idx, b_idx in enumerate(batch_indices):
                        k_positions = (topk_indices[b_idx] == expert_id).nonzero(as_tuple=True)[0]
                        for k_pos in k_positions:
                            weight = topk_weights[b_idx, k_pos]
                            output[b_idx] += weight * expert_output[idx]

        return output, routing_loss

    def freeze_experts(self, expert_ids):
        """Freeze specified experts."""
        for expert_id in expert_ids:
            for param in self.experts[expert_id].parameters():
                param.requires_grad = False

    def add_expert(self, d_model, d_ffn, dropout=0.0, init_from=None, noise_scale=0.01):
        """Add a new expert, optionally initialized from existing expert."""
        new_expert = Expert(d_model, d_ffn, dropout)

        if init_from is not None and init_from < len(self.experts):
            # Copy weights from existing expert
            new_expert.load_state_dict(self.experts[init_from].state_dict())

        # Move to correct device before adding
        if len(self.experts) > 0:
            device = next(self.experts[0].parameters()).device
            new_expert = new_expert.to(device)

        self.experts.append(new_expert)
        self.num_experts += 1

        # Update router to handle new expert
        old_num_experts = self.router.num_experts
        self.router.num_experts = self.num_experts

        # Expand router output layer
        old_linear = self.router.network[-1]
        old_weight = old_linear.weight.data
        old_bias = old_linear.bias.data

        new_linear = nn.Linear(old_linear.in_features, self.num_experts)
        new_linear = new_linear.to(old_weight.device)

        # Copy old weights
        with torch.no_grad():
            new_linear.weight.data[:old_num_experts] = old_weight
            new_linear.bias.data[:old_num_experts] = old_bias

            # Initialize new expert's router weights from previous expert with small noise
            if init_from is not None and init_from < old_num_experts:
                new_linear.weight.data[old_num_experts] = old_weight[init_from] + torch.randn_like(old_weight[init_from]) * noise_scale
                new_linear.bias.data[old_num_experts] = old_bias[init_from] + torch.randn(1, device=old_bias.device).item() * noise_scale
            else:
                # Random initialization for new expert
                nn.init.xavier_uniform_(new_linear.weight.data[old_num_experts:])
                nn.init.zeros_(new_linear.bias.data[old_num_experts:])

        self.router.network[-1] = new_linear
