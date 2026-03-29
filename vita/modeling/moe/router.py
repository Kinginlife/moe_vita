"""
Router network for MoE.
Routes inputs to appropriate experts based on features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """Router network that learns to select experts."""

    def __init__(self, d_model, num_experts, router_dim=512, soft_temp=2.0):
        super().__init__()
        self.num_experts = num_experts
        self.soft_temp = soft_temp
        self.network = nn.Sequential(
            nn.Linear(d_model, router_dim),
            nn.ReLU(),
            nn.Linear(router_dim, num_experts)
        )
        self.old_expert_mask = 0  # Number of old experts to freeze
        self._register_gradient_hook()

    def _register_gradient_hook(self):
        """Register hook to zero out gradients for old expert weights."""
        def hook(grad):
            if self.old_expert_mask > 0:
                grad[:self.old_expert_mask] = 0
            return grad

        self.network[-1].weight.register_hook(hook)
        self.network[-1].bias.register_hook(hook)

    def forward(self, x, task_id=None):
        """
        Args:
            x: [B, N, d_model] or [N, B, d_model] (transformer format)
            task_id: scalar or [B] or [B, N] optional task IDs for training supervision
        Returns:
            router_logits: [B, N, num_experts] (query-level routing)
            routing_loss: scalar or None
        """
        # Handle transformer format [N, B, D] -> [B, N, D]
        if x.dim() == 3 and x.size(1) < x.size(0):
            x = x.transpose(0, 1)  # [N, B, D] -> [B, N, D]

        # Query-level routing: keep [B, N, D] shape
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)  # [B*N, D]
        router_logits = self.network(x_flat)  # [B*N, num_experts]
        router_logits = router_logits.reshape(B, N, self.num_experts)  # [B, N, num_experts]

        routing_loss = None
        if self.training:
            if task_id is not None:
                # Convert scalar task_id to tensor if needed
                if not isinstance(task_id, torch.Tensor):
                    task_id = torch.full((B,), task_id, dtype=torch.long, device=x.device)
                elif task_id.dim() == 0:
                    task_id = task_id.unsqueeze(0).expand(B)
                elif task_id.size(0) != B:
                    task_id = task_id[0].unsqueeze(0).expand(B)

                # Expand to [B, N] if needed
                if task_id.dim() == 1:
                    task_id = task_id.unsqueeze(1).expand(B, N)  # [B, N]

                # Soft supervision at query level
                target_probs = F.one_hot(task_id, self.num_experts).float()  # [B, N, num_experts]
                target_probs = target_probs * (1 - 0.1) + 0.1 / self.num_experts
                router_probs = F.log_softmax(router_logits / self.soft_temp, dim=-1)
                routing_loss = F.kl_div(router_probs.reshape(-1, self.num_experts),
                                       target_probs.reshape(-1, self.num_experts),
                                       reduction='batchmean')
            else:
                # Entropy regularization
                router_probs = F.softmax(router_logits, dim=-1)
                routing_loss = -(router_probs * torch.log(router_probs + 1e-8)).sum(dim=-1).mean() * 0.01

        return router_logits, routing_loss
