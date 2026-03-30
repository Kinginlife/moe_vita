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
            nn.Linear(router_dim, num_experts, bias=False)
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
        

    def forward(self, x, routing_targets=None):
        """
        Args:
            x: [B, N, d_model] or [N, B, d_model] (transformer format)
            routing_targets: dict with keys:
                - 'target_expert_ids': [B, N] target expert for each query
                - 'valid_mask': [B, N] whether to supervise each query
                or None (inference mode)
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

        # Cosine similarity routing
        features = self.network[:-1](x_flat)  # [B*N, router_dim]
        last_weight = self.network[-1].weight  # [num_experts, router_dim]

        f_norm = F.normalize(features, p=2, dim=-1)
        w_norm = F.normalize(last_weight, p=2, dim=-1)

        router_logits = F.linear(f_norm, w_norm) * 20.0
        router_logits = router_logits.reshape(B, N, self.num_experts)  # [B, N, num_experts]

        routing_loss = None
        if self.training:
            routing_loss = self._compute_routing_loss(router_logits, routing_targets)

        return router_logits, routing_loss

    def _compute_routing_loss(self, router_logits, routing_targets):
        """
        Compute routing loss = supervised loss + entropy regularization

        Args:
            router_logits: [B, N, num_experts]
            routing_targets: dict or None
        """
        B, N, E = router_logits.shape

        supervised_loss = 0.0

        # 1. Supervised loss (only for positive samples)
        if routing_targets is not None:
            target_ids = routing_targets['target_expert_ids']  # [B, N]
            valid_mask = routing_targets['valid_mask']          # [B, N]

            if valid_mask.any():
                # Extract queries that need supervision
                valid_logits = router_logits[valid_mask]  # [M, E]
                valid_targets = target_ids[valid_mask]     # [M]

                # Generate soft labels (label smoothing)
                target_probs = F.one_hot(valid_targets, E).float()
                target_probs = target_probs * 0.9 + 0.1 / E  # [M, E]

                # KL divergence
                log_probs = F.log_softmax(valid_logits / self.soft_temp, dim=-1)
                supervised_loss = F.kl_div(
                    log_probs, target_probs, reduction='batchmean'
                )

        # 2. Entropy regularization (for all queries, including background)
        router_probs = F.softmax(router_logits, dim=-1)  # [B, N, E]
        entropy = -(router_probs * torch.log(router_probs + 1e-8)).sum(dim=-1)
        entropy_loss = -entropy.mean() * 0.01  # Negative: maximize entropy

        # 3. Total loss
        total_loss = supervised_loss + entropy_loss

        return total_loss
