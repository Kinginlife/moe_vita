"""
Router network for MoE.
Routes inputs to appropriate experts based on features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """Router network that learns to select experts."""

    def __init__(self, d_model, num_experts, router_dim=512):
        super().__init__()
        self.num_experts = num_experts
        self.network = nn.Sequential(
            nn.Linear(d_model, router_dim),
            nn.ReLU(),
            nn.Linear(router_dim, num_experts)
        )

    def forward(self, x, task_id=None):
        """
        Args:
            x: [B, N, d_model] or [B, d_model]
            task_id: [B] optional task IDs for training supervision
        Returns:
            router_logits: [B, num_experts]
            routing_loss: scalar or None
        """
        # Global pooling if input is [B, N, d_model]
        if x.dim() == 3:
            x = x.mean(dim=1)  # [B, d_model]

        router_logits = self.network(x)  # [B, num_experts]

        routing_loss = None
        if self.training and task_id is not None:
            routing_loss = F.cross_entropy(router_logits, task_id)

        return router_logits, routing_loss
