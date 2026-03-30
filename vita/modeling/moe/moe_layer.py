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

    def forward(self, x):
        """
        Args:
            x: [N, B, d_model] (transformer format) or [B, N, d_model]
        Returns:
            output: same shape as x
            router_logits: [B, N, num_experts] for loss computation
        """
        # Handle transformer format [N, B, D] -> [B, N, D]
        is_transformer_format = False
        if x.dim() == 3 and x.size(1) < x.size(0):
            x = x.transpose(0, 1)  # [N, B, D] -> [B, N, D]
            is_transformer_format = True

        B, N, D = x.shape

        # Get query-level routing logits [B, N, num_experts]
        router_logits = self.router(x)

        # Use Top-1 routing for both training and inference (avoid train-test mismatch)
        actual_k = 1

        if actual_k == 1:
            # Top-1: select best expert per query
            expert_ids = router_logits.argmax(dim=-1)  # [B, N]
            output = torch.zeros_like(x)

            for expert_id in range(self.num_experts):
                mask = expert_ids == expert_id  # [B, N]
                if mask.any():
                    selected_x = x[mask]  # [num_selected, D]
                    expert_output = self.experts[expert_id](selected_x)
                    output[mask] = expert_output.to(x.dtype)
        else:
            # Top-K: weighted combination per query
            topk_weights, topk_indices = torch.topk(router_logits, actual_k, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1)  # [B, N, K]

            output = torch.zeros_like(x)
            for expert_id in range(self.num_experts):
                mask = (topk_indices == expert_id).any(dim=-1)  # [B, N]
                if mask.any():
                    selected_x = x[mask]
                    expert_output = self.experts[expert_id](selected_x).to(x.dtype)

                    # Apply weights
                    indices = mask.nonzero(as_tuple=False)
                    for idx, (b, n) in enumerate(indices):
                        k_positions = (topk_indices[b, n] == expert_id).nonzero(as_tuple=True)[0]
                        for k_pos in k_positions:
                            weight = topk_weights[b, n, k_pos]
                            output[b, n] += weight * expert_output[idx]

        # Convert back to transformer format if needed
        if is_transformer_format:
            output = output.transpose(0, 1)  # [B, N, D] -> [N, B, D]

        # Prevent DDP unused parameter error
        if self.training:
            dummy_add = 0.0
            for expert in self.experts:
                for param in expert.parameters():
                    if param.requires_grad:
                        dummy_add = dummy_add + param.sum() * 0.0
            output = output + dummy_add

        return output, router_logits

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
       

        new_linear = nn.Linear(old_linear.in_features, self.num_experts,bias=False)
        new_linear = new_linear.to(old_weight.device)

        # Copy old weights and FREEZE them
        with torch.no_grad():
            new_linear.weight.data[:old_num_experts] = old_weight

            # Initialize new expert's router weights with larger noise
            if init_from is not None and init_from < old_num_experts:
                new_linear.weight.data[old_num_experts] = old_weight[init_from] + torch.randn_like(old_weight[init_from]) * 0.5
            else:
                nn.init.xavier_uniform_(new_linear.weight.data[old_num_experts:])

            # Project new weights to orthogonal subspace
            if self.router.projection_matrix is not None:
                new_weights = new_linear.weight.data[old_num_experts:]  # [new_experts, D]
                proj_matrix = self.router.projection_matrix.to(new_weights.device)
                new_weights = torch.matmul(new_weights, proj_matrix)
                new_linear.weight.data[old_num_experts:] = new_weights

        self.router.network[-1] = new_linear

        # Update mask for orthogonal projection (freeze old experts)
        self.router.old_expert_mask = old_num_experts

        # Re-register gradient hook after replacing the layer
        self.router._register_gradient_hook()
