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

    def __init__(self, d_model, d_ffn, num_experts, router_dim=512, top_k=1, dropout=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ffn, dropout) for _ in range(num_experts)
        ])

        # Router network
        self.router = Router(d_model, num_experts, router_dim)

    def forward(self, x, task_id=None):
        """
        Args:
            x: [B, N, d_model] input features
            task_id: [B] optional task IDs for training
        Returns:
            output: [B, N, d_model]
            routing_loss: scalar or None
        """
        B, N, D = x.shape

        # Get routing logits
        router_logits, routing_loss = self.router(x, task_id)  # [B, num_experts]

        if self.training and task_id is not None:
            # Training: use task_id to select expert
            expert_outputs = []
            for i in range(B):
                expert_id = task_id[i].item()
                expert_out = self.experts[expert_id](x[i])  # [N, D]
                expert_outputs.append(expert_out)
            output = torch.stack(expert_outputs, dim=0)  # [B, N, D]
        else:
            # Inference: use router to select top-k experts
            actual_k = min(self.top_k, self.num_experts)

            if actual_k == 1:
                # Top-1: select single expert
                expert_ids = router_logits.argmax(dim=-1)  # [B]
                expert_outputs = []
                for i in range(B):
                    expert_id = expert_ids[i].item()
                    expert_out = self.experts[expert_id](x[i])
                    expert_outputs.append(expert_out)
                output = torch.stack(expert_outputs, dim=0)
            else:
                # Top-K: weighted combination
                topk_weights, topk_indices = torch.topk(router_logits, actual_k, dim=-1)
                topk_weights = F.softmax(topk_weights, dim=-1)  # [B, actual_k]

                output = torch.zeros_like(x)
                for i in range(B):
                    for k in range(actual_k):
                        expert_id = topk_indices[i, k].item()
                        weight = topk_weights[i, k]
                        output[i] += weight * self.experts[expert_id](x[i])

        return output, routing_loss

    def freeze_experts(self, expert_ids):
        """Freeze specified experts."""
        for expert_id in expert_ids:
            for param in self.experts[expert_id].parameters():
                param.requires_grad = False

    def add_expert(self, d_model, d_ffn, dropout=0.0, init_from=None):
        """Add a new expert, optionally initialized from existing expert."""
        new_expert = Expert(d_model, d_ffn, dropout)

        if init_from is not None and init_from < len(self.experts):
            # Copy weights from existing expert
            new_expert.load_state_dict(self.experts[init_from].state_dict())

        self.experts.append(new_expert)
        self.num_experts += 1

        # Update router to handle new expert
        old_num_experts = self.router.num_experts
        self.router.num_experts = self.num_experts

        # Expand router output layer
        old_weight = self.router.network[-1].weight.data
        old_bias = self.router.network[-1].bias.data

        new_linear = nn.Linear(self.router.network[-1].in_features, self.num_experts)
        new_linear.weight.data[:old_num_experts] = old_weight
        new_linear.bias.data[:old_num_experts] = old_bias

        self.router.network[-1] = new_linear
