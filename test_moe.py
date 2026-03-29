"""
Test MoE module independently
"""
import torch
from vita.modeling.moe import MoELayer

# Test MoE layer
d_model = 256
d_ffn = 2048
num_experts = 1
batch_size = 2
seq_len = 100

moe = MoELayer(d_model, d_ffn, num_experts, router_dim=512, top_k=1)
moe.cuda()
moe.train()

# Test forward
x = torch.randn(batch_size, seq_len, d_model).cuda()
task_id = torch.zeros(batch_size, dtype=torch.long).cuda()

output, routing_loss = moe(x, task_id)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Routing loss: {routing_loss}")
print("✅ MoE module test passed!")
