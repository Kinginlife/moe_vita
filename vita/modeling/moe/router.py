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

    def forward(self, x, task_id=None):
        """
        Args:
            x: [B, N, d_model] or [B, d_model] or [N, B, d_model] (transformer format)
            task_id: scalar or [B] optional task IDs for training supervision
        Returns:
            router_logits: [B, num_experts]
            routing_loss: scalar or None
        """
        # Handle transformer format [N, B, D] -> [B, N, D]
        if x.dim() == 3 and x.size(1) < x.size(0):
            x = x.transpose(0, 1)  # [N, B, D] -> [B, N, D]

        # Global pooling if input is [B, N, d_model]
        if x.dim() == 3:
            x = x.mean(dim=1)  # [B, d_model]

        router_logits = self.network(x)  # [B, num_experts]
        batch_size = router_logits.size(0)

        routing_loss = None
        if self.training:
            if task_id is not None:
                # Convert scalar task_id to tensor if needed
                if not isinstance(task_id, torch.Tensor):
                    task_id = torch.full((batch_size,), task_id, dtype=torch.long, device=x.device)
                elif task_id.dim() == 0:  # scalar tensor
                    task_id = task_id.unsqueeze(0).expand(batch_size)
                elif task_id.size(0) != batch_size:
                    # Expand if size mismatch
                    task_id = task_id[0].unsqueeze(0).expand(batch_size)

                # Soft supervision: encourage but don't force routing to task_id
                target_probs = F.one_hot(task_id, self.num_experts).float()
                # Add temperature to soften the target
                target_probs = target_probs * (1 - 0.1) + 0.1 / self.num_experts
                router_probs = F.log_softmax(router_logits / self.soft_temp, dim=-1)
                routing_loss = F.kl_div(router_probs, target_probs, reduction='batchmean')
            else:
                # Ensure router parameters always get gradients even without task_id
                # Use entropy regularization to encourage balanced expert usage
                router_probs = F.softmax(router_logits, dim=-1)
                routing_loss = -(router_probs * torch.log(router_probs + 1e-8)).sum(dim=-1).mean() * 0.01

        return router_logits, routing_loss
