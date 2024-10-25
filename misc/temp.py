# %%
import path_setup
import torch
import torch.nn as nn

# from einops import repeat


# class ViT(nn.Module):
#     def __init__(
#         self,
#         ch=3,
#         img_size=144,
#         patch_size=4,
#         emb_dim=32,
#         n_layers=6,
#         out_dim=37,
#         dropout=0.1,
#         heads=2,
#     ):
#         super(ViT, self).__init__()

#         # Attributes
#         self.channels = ch
#         self.height = img_size
#         self.width = img_size
#         self.patch_size = patch_size
#         self.n_layers = n_layers

#         # Patching
#         self.patch_embedding = PatchEmbedding(
#             in_channels=ch, patch_size=patch_size, emb_size=emb_dim
#         )
#         # Learnable params
#         num_patches = (img_size // patch_size) ** 2
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
#         self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))

#         # Transformer Encoder
#         self.layers = nn.ModuleList([])
#         for _ in range(n_layers):
#             transformer_block = nn.Sequential(
#                 ResidualAdd(
#                     PreNorm(emb_dim, Attention(emb_dim, n_heads=heads, dropout=dropout))
#                 ),
#                 ResidualAdd(
#                     PreNorm(emb_dim, FeedForward(emb_dim, emb_dim, dropout=dropout))
#                 ),
#             )
#             self.layers.append(transformer_block)

#         # Classification head
#         self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, out_dim))

#     def forward(self, img):
#         # Get patch embedding vectors
#         x = self.patch_embedding(img)
#         b, n, _ = x.shape

#         # Add cls token to inputs
#         cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
#         x = torch.cat([cls_tokens, x], dim=1)
#         x += self.pos_embedding[:, : (n + 1)]

#         # Transformer layers
#         for i in range(self.n_layers):
#             x = self.layers[i](x)

#         # Output based on classification token
#         return self.head(x[:, 0, :])

emb_dim = 32
torch.randn(1, 4, 4).shape
nn.Parameter(torch.randn(1, 1, 4)).shape
# %%
# CLS shape after repeat: torch.Size([1, 1, 32])
# The shape of the embedding is:
# torch.Size([1, 1296, 32])
from utilities import *

num_patches = 1296
dim_embeddings = 32
pos_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
cls_tokens = torch.randn(1, 1, dim_embeddings)


x = torch.randn(1, num_patches, dim_embeddings)
b, n, _ = x.shape
green_print("shape x before adding E_cls:", x.shape)

x = torch.cat([cls_tokens, x], dim=1)
green_print("shape x after adding E_cls:", x.shape)
teal_print(pos_embeddings[:, :(n + 1)].shape)
x1 = x + pos_embeddings[:, :(n + 1)]
x2 = x + pos_embeddings
x1 - x2

green_print("shape cls:", cls_tokens.shape)
magenta_print("shape positional embeddings array:", pos_embeddings.shape)

# %%
x = torch.randn(1)
silent_print(x.shape)
