"""
Router network for MoE.
Routes inputs to appropriate experts based on features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """Router network that learns to select experts with orthogonal gradient projection."""

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

        # Learnable temperature for cosine similarity scaling
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(5.0)))

        # Store projection matrix P = I - UU^T for OGP
        self.register_buffer('projection_matrix', None)

        self._register_gradient_hook()

    def _register_gradient_hook(self):
        """Register hook to apply OGP to new experts while keeping old expert routing intact."""
        def hook(grad):
            if grad is None:
                return grad

            if self.old_expert_mask > 0:
                # Scale down old expert gradients (not freeze completely)
                grad[:self.old_expert_mask] = grad[:self.old_expert_mask] * 0.1

                # Apply orthogonal projection to new experts
                if self.projection_matrix is not None and grad.size(0) > self.old_expert_mask:
                    grad_new = grad[self.old_expert_mask:].clone()  # [new_experts, router_dim]
                    proj_matrix = self.projection_matrix.to(grad.device)
                    grad_new = torch.matmul(grad_new, proj_matrix)
                    grad[self.old_expert_mask:] = grad_new
            return grad

        # Clear old hooks before registering new one (if they exist)
        if hasattr(self.network[-1].weight, '_backward_hooks') and self.network[-1].weight._backward_hooks is not None:
            self.network[-1].weight._backward_hooks.clear()
        self.network[-1].weight.register_hook(hook)

    def compute_projection_matrix(self, features, energy_threshold=0.7):
        """
        Compute orthogonal projection matrix P = I - UU^T from old task features.

        Args:
            features: [N, D] tensor of old task features (router input)
            energy_threshold: retain this fraction of energy (default 0.85, lower = less restrictive)
        """
        device = features.device
        D = features.shape[1]

        # Compute covariance matrix
        features_centered = features - features.mean(dim=0, keepdim=True)
        cov = torch.matmul(features_centered.T, features_centered) / (features.shape[0] - 1)

        # SVD: cov = U @ S @ U^T
        U, S, _ = torch.linalg.svd(cov)

        # Retain components that preserve energy_threshold of variance
        total_energy = S.sum()
        cumsum_energy = torch.cumsum(S, dim=0)
        k = (cumsum_energy / total_energy <= energy_threshold).sum() + 1
        k = min(k.item(), D)

        U_k = U[:, :k]  # [D, k]

        # Compute projection matrix P = I - U_k @ U_k^T
        I = torch.eye(D, device=device, dtype=features.dtype)
        P = I - torch.matmul(U_k, U_k.T)

        self.projection_matrix = P

    def set_projection_matrix(self, P):
        """Set projection matrix externally."""
        self.projection_matrix = P


    def forward(self, x):
        """
        Args:
            x: [B, N, d_model] or [N, B, d_model] (transformer format)
        Returns:
            router_logits: [B, N, num_experts] (query-level routing)
        """
        # Handle transformer format [N, B, D] -> [B, N, D]
        if x.dim() == 3 and x.size(1) < x.size(0):
            x = x.transpose(0, 1)  # [N, B, D] -> [B, N, D]

        # Query-level routing: keep [B, N, D] shape
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)  # [B*N, D]

        # Cosine similarity routing with learnable temperature
        features = self.network[:-1](x_flat)  # [B*N, router_dim]
        last_weight = self.network[-1].weight  # [num_experts, router_dim]

        f_norm = F.normalize(features, p=2, dim=-1)
        w_norm = F.normalize(last_weight, p=2, dim=-1)

        # Use learnable temperature (clamped to [1, 20])
        temperature = torch.clamp(torch.exp(self.log_temperature), 1.0, 20.0)
        router_logits = F.linear(f_norm, w_norm) * temperature
        router_logits = router_logits.reshape(B, N, self.num_experts)  # [B, N, num_experts]

        return router_logits

    @staticmethod
    def compute_routing_loss(router_logits, routing_targets, soft_temp=2.0, old_expert_mask=0):
        """
        Static method to compute routing loss (called from Criterion).

        Args:
            router_logits: [B, N, num_experts]
            routing_targets: dict with 'target_expert_ids' [B, N] and 'valid_mask' [B, N]
            soft_temp: temperature for soft routing
            old_expert_mask: number of old experts (for incremental learning)
        Returns:
            total_loss: scalar
        """
        B, N, E = router_logits.shape

        supervised_loss = torch.tensor(0.0, device=router_logits.device, dtype=router_logits.dtype)

        # 1. Supervised loss (only for positive samples)
        if routing_targets is not None:
            target_ids = routing_targets['target_expert_ids']
            valid_mask = routing_targets['valid_mask']
            num_valid = valid_mask.sum()

            if num_valid > 0:
                valid_logits = router_logits[valid_mask]
                valid_targets = target_ids[valid_mask]

                # Special handling for single expert (Task 0)
                if E == 1:
                    target_logits = torch.ones_like(valid_logits) * 5.0
                    supervised_loss = F.mse_loss(valid_logits, target_logits)
                else:
                    # Multi-expert: use cross-entropy
                    supervised_loss = F.cross_entropy(valid_logits / soft_temp, valid_targets)

            # 2. Negative supervision: prevent routing conflicts
            if E > 1 and old_expert_mask > 0 and num_valid > 0:
                valid_logits_all = router_logits[valid_mask]  # [num_valid, E]
                valid_targets_all = target_ids[valid_mask]  # [num_valid]

                # Prevent new task samples from routing to old experts
                new_expert_samples = valid_targets_all >= old_expert_mask
                num_new_samples = new_expert_samples.sum()
                if num_new_samples > 0:
                    old_expert_logits = valid_logits_all[new_expert_samples, :old_expert_mask]
                    suppress_new_to_old = -F.logsigmoid(-old_expert_logits).mean()
                    supervised_loss = supervised_loss + suppress_new_to_old * 0.5

                # Prevent old task samples from routing to new experts
                old_expert_samples = valid_targets_all < old_expert_mask
                num_old_samples = old_expert_samples.sum()
                if num_old_samples > 0:
                    new_expert_logits = valid_logits_all[old_expert_samples, old_expert_mask:]
                    suppress_old_to_new = -F.logsigmoid(-new_expert_logits).mean()
                    supervised_loss = supervised_loss + suppress_old_to_new * 0.5

        return supervised_loss
