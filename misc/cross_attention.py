"""keywords: attention, cross-attention, multiModal attention."""

import torch
import torch.nn as nn
from utilities import *


class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, embed_dim, n_heads=1, dropout=0.0):
        super().__init__()

        value_dim = dim1
        self.n_heads = n_heads
        self.att = nn.MultiheadAttention(
            embed_dim=embed_dim, vdim=value_dim, num_heads=n_heads, dropout=dropout
        )
        self.q = nn.Linear(dim1, embed_dim)
        self.k = nn.Linear(dim2, embed_dim)
        self.v = nn.Linear(dim2, value_dim)

    def forward(self, x1, x2):
        # Compute queries from x1
        blue_print("Shape of x1", x1.shape)
        blue_print("Shape of x2", x2.shape)
        q = self.q(x1)
        # Compute keys and values from x2
        k = self.k(x2)
        green_print("Shape Q_x1:", q.shape)
        green_print("Shape K_x2:", k.shape)
        v = self.v(x2)

        # Compute attention using x1 as queries and x2 as keys and values
        attn_output, attn_output_weights = self.att(q, k, v)

        return attn_output, attn_output_weights


# Example input tensors
n_embeddings_x1 = 1
n_embeddings_x2 = 1

dim1 = 1
dim2 = 1
embed_dim = 1

x1 = torch.randn(1, n_embeddings_x1, dim1)  # Embeddings whose attention we shall guide
x2 = torch.randn(1, n_embeddings_x2, dim2)  # vector that guides attention of x1

n_heads = 1  # number of attention heads
attention_layer = CrossAttention(dim1, dim2, embed_dim)

attn_output, attn_output_weights = attention_layer(x1, x2)
magenta_print("Attention Output:", attn_output.detach())
magenta_print("Shape:", attn_output.shape)
red_print("Attention Weights:", attn_output_weights)

n_params = 0
for p in attention_layer.parameters():
    n_params += len(p)

lime_print("Parameters:", n_params)
