"""This scripte implements Vision transformers (ViTs) following an online
example.

keywords: transformer, attention, ViT, visual transformers
"""

# %% [markdown]
# # Vision Transformers from scratch

# %% [markdown]
# - [ViT Blogpost by Francesco Zuppichini](https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632)
# - [D2L Tutorial ](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html)
# - [Brian Pulfer Medium Blogpost](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c)
# - [Lucidrains implementation Github ](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py)

# %% Image Patching
import torch
from torchvision.datasets import OxfordIIITPet
import matplotlib.pyplot as plt
from random import random
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import to_pil_image
from utilities import *
from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor

# %% Image Patching

# List of Transformations
to_tensor = [Resize((144, 144)), ToTensor()]


class Compose(object):
    """Class that applies multiple transformations in sequence."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        """Loop through the transformations in the list and apply them in sequence."""
        for t in self.transforms:
            image = t(image)
        return image, target


def show_images(images, num_samples=40, cols=8):
    """Plots some samples from the dataset"""
    plt.figure(figsize=(15, 15))
    idx = int(len(dataset) / num_samples)
    print(images)
    for i, img in enumerate(images):
        if i % idx == 0:
            plt.subplot(int(num_samples / cols) + 1, cols, int(i / idx) + 1)
            plt.imshow(to_pil_image(img[0]))


# 200 images for each pet
# Downloads images if they the expected folder is not found in 'root'
dataset = OxfordIIITPet(root="../data/", download=True, transforms=Compose(to_tensor))
show_images(dataset)

# %% Patch Images


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=8, emb_size=128):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            nn.Linear(patch_size * patch_size * in_channels, emb_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x


# Run a quick test
sample_datapoint = torch.unsqueeze(dataset[0][0], 0)
green_print("Initial shape: ", sample_datapoint.shape)
embedding = PatchEmbedding()(sample_datapoint)
green_print("Patches shape: ", embedding.shape)

# %% Model
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = torch.nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout
        )

    def forward(self, x):
        # Input x, x, x because we are using self-attention (no other modality is used to guide attention)
        attn_output, attn_output_weights = self.att(x, x, x)
        return attn_output


attention = Attention(dim=128, n_heads=4, dropout=0.0)(torch.ones((2, 5, 128)))
green_print("Attention output shape:", attention.shape)


# %%  The normalization step
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# %%
norm = PreNorm(128, Attention(dim=128, n_heads=4, dropout=0.0))
norm(torch.ones((1, 5, 128))).shape


# %%
class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )


ff = FeedForward(dim=128, hidden_dim=256)
ff(torch.ones((1, 5, 128))).shape


# %% Residual connections
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


# %%
residual_att = ResidualAdd(Attention(dim=128, n_heads=4, dropout=0.0))
residual_att(torch.ones((1, 5, 128))).shape

# %% Now bring all the components together
# - Not all parameters are like in the original implementation
# - Some Dropouts & Norms are missing

from einops import repeat


class ViT(nn.Module):
    def __init__(
        self,
        ch=3,
        img_size=144,
        patch_size=4,
        emb_dim=32,
        n_layers=6,
        out_dim=37,
        dropout=0.1,
        heads=2,
    ):
        super(ViT, self).__init__()

        # Attributes
        self.channels = ch
        self.height = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers

        # Patching
        self.patch_embedding = PatchEmbedding(
            in_channels=ch, patch_size=patch_size, emb_size=emb_dim
        )
        # Learnable params
        num_patches = (img_size // patch_size) ** 2
        # Positional embeddings are vectors that are added to the Token Embeddings to shift them to their respective positions.
        # They have the same size as the token embeddings.
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, emb_dim)
        )  # The '+1' is because we will add the CLS embedding, so there are num_patches + 1 embeddings in total
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))

        # Transformer Encoder
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            # Here I define what happens in each block
            transformer_block = nn.Sequential(
                ResidualAdd(
                    PreNorm(emb_dim, Attention(emb_dim, n_heads=heads, dropout=dropout))
                ),
                ResidualAdd(
                    PreNorm(emb_dim, FeedForward(emb_dim, emb_dim, dropout=dropout))
                ),
            )
            self.layers.append(transformer_block)

        # Classification head
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, out_dim))

    def forward(self, img):
        # Get patch embedding vectors
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        # Add cls token to inputs
        # Prepend the CLS (classification) embedding to the sequence of embeddings for that image
        # [emb1, emb2, ... embk] --> [cls_emb, emb1, emb2, ... embk]
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, : (n + 1)]

        # Transformer layers
        for i in range(self.n_layers):
            x = self.layers[i](x)

        # Output based on classification token
        return self.head(x[:, 0, :])


model = ViT()
print(model)
# Predict
model(torch.ones((1, 3, 144, 144)))


# %% [markdown]
# ## Training

# %%
from torch.utils.data import DataLoader
from torch.utils.data import random_split

train_split = int(0.8 * len(dataset))
train, test = random_split(dataset, [train_split, len(dataset) - train_split])

train_dataloader = DataLoader(train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test, batch_size=32, shuffle=True)
print("N.o. training images:", len(train_dataloader))
print("N.o. iterations per epoch:", len(train))

# %%
import torch.optim as optim
import numpy as np

device = "cuda"
model = ViT().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(200):
    epoch_losses = []
    model.train()
    for step, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    if epoch % 5 == 0:
        print(f">>> Epoch {epoch} train loss: ", np.mean(epoch_losses))
        epoch_losses = []
        # Something was strange when using this?
        # model.eval()
        for step, (inputs, labels) in enumerate(test_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_losses.append(loss.item())
        print(f">>> Epoch {epoch} test loss: ", np.mean(epoch_losses))

# %%
inputs, labels = next(iter(test_dataloader))
inputs, labels = inputs.to(device), labels.to(device)
outputs = model(inputs)

print("Predicted classes", outputs.argmax(-1))
print("Actual classes", labels)

# %% [markdown]
# This needs to train much longer :)
