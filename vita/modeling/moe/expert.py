"""
Expert network for MoE.
Each expert is a standard FFN: Linear -> ReLU -> Linear
"""
import torch
import torch.nn as nn


class Expert(nn.Module):
    """Single expert network (FFN)."""

    def __init__(self, d_model, d_ffn, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x
