# %%
import path_setup
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from utilities import *
import matplotlib.pyplot as plt
from utils_simulateData import simulate_data
from utilities import calc_acc
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


# %%


class SelfAttention2DPoints(nn.Module):
    """Simple implementation of the transformer architecture where """
    def __init__(self, embed_dim=1, n_heads=1, dropout=0.0, batch_first=True):
        super().__init__()

        self.n_heads = n_heads
        self.att = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.embed_dim = embed_dim
        self.seq_length = 2

        self.fc0 = nn.Linear(self.seq_length, self.seq_length)
        self.fc1 = nn.Linear(self.seq_length, self.seq_length)
        self.fc2 = nn.Linear(self.seq_length, 1)

    def forward(self, x1, x2):
        # ** Prepare Data **
        # Concatenate
        x = torch.cat([x1, x2], dim=-1)
        # Separate each sequence into its own coordinate
        # Batch size, sequence length, length of embeddings making up each sequence
        x = x.reshape(-1, self.seq_length, self.embed_dim)

        # x = F.sigmoid(self.fc0(x))
        # Compute attention using x1 as queries and x2 as keys and values
        x, attn_weights = self.att(x, x, x)
        # red_print("weights raw", attn_weights)

        attn_weights = [[w[0, 1], w[1, 1]] for w in attn_weights]
        self.attn_weights = torch.tensor(attn_weights)
        # magenta_print("weights", self.attn_weights)

        x = x.reshape(-1, 2)  # Reshape back to N by 2 shape

        # Finally, reduce to prediction and apply softmax
        output = F.sigmoid(self.fc2(x))
        return output


x1 = torch.tensor([[1.0], [2.0], [3.0]])
x2 = torch.tensor([[2.0], [1.5], [1.0]])
# model = SelfAttention()
# preds = model.forward(x1, x2)
X = torch.cat([x1, x2], dim=-1)
X.unsqueeze(0)
# Batch size, sequence length, length of embeddings making up each sequence
X = X.reshape(-1, 2, 1)

X.reshape(-1, 2)

X.shape

model = SelfAttention2DPoints()
model.forward(x1, x2)
green_print("Weights\n", model.attn_weights)

W = model.attn_weights
W = torch.tensor(W)
# Z = torch.rand(1, 3, 2)

# model = SelfAttention()
# model.forward(Z)
# model.attention_weights
# preds = model.forward(x1, x2)

# %%

x1, x2, C = simulate_data(n_samples=200, plot_data=True)
plt.show()

# Convert data to tensor format
X1 = torch.tensor(x1).unsqueeze(1).float()
X2 = torch.tensor(x2).unsqueeze(1).float()
labels = torch.tensor(C).unsqueeze(1).float()

X1_train, X1_test, X2_train, X2_test, labels_train, labels_test = train_test_split(
    X1, X2, labels, test_size=0.5
)

learning_rate = 0.05
num_epochs = 100
batch_size = 32
n_max_consecutive_failures = 20

loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
losses = []
weights = []

# Prepare Data for Training Loop
train_dataset = TensorDataset(X1, X2, labels)
dataset_loader = DataLoader(train_dataset, batch_size=batch_size)
consecutive_failures_to_improve = 0

for i in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    # Iterate over the batches
    for x1_batch, x2_batch, labels_batch in dataset_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        batch_predictions = model(x1_batch, x2_batch)

        # Compute the loss
        loss = loss_function(batch_predictions, labels_batch)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()
        weights.append(model.attn_weights)

    avg_epoch_loss = running_loss / len(dataset_loader)
    losses.append(avg_epoch_loss)

    if i >= 1:
        if losses[-1] > min(losses):
            consecutive_failures_to_improve += 1

        else:
            consecutive_failures_to_improve = 0

    if consecutive_failures_to_improve >= n_max_consecutive_failures:
        red_print(
            f"Failed to improve over {consecutive_failures_to_improve} Epochs: terminating training"
        )
        break

if True:
    plt.plot(losses)
    plt.xlabel("epochs")
    plt.show()

model(X1_test, X2_test)

acc = calc_acc(X1_test, X2_test, model=model, labels=labels_test)
green_print("Test Accuracy:", acc)

# Define the grid range
x1_range = np.linspace(-1, 3, 100)  # 100 points from 0 to 2
x2_range = np.linspace(-1, 3, 100)  # 100 points from 0 to 2
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)  # Create a meshgrid

# Flatten the grids to create input for the model
X1_flat = X1_grid.flatten()
X2_flat = X2_grid.flatten()

# Convert to PyTorch tensors and unsqueeze (turn into (N, 1) arrays)
X1_tensor = torch.tensor(X1_flat).float().unsqueeze(1)  # Shape: (N, 1)
X2_tensor = torch.tensor(X2_flat).float().unsqueeze(1)  # Shape: (N, 1)

# Get predictions from the model
with torch.no_grad():  # Disable gradient calculation
    model.eval()  # Set model to evaluation mode
    predictions = model(X1_tensor, X2_tensor)  # Model outputs logits or probabilities

# Reshape predictions to match the grid shape
predictions = predictions.numpy().reshape(X1_grid.shape)  # Reshape to (100, 100)
weights_x1 = (
    model.attn_weights[:, 0].numpy().reshape(X1_grid.shape)
)  # Reshape to (100, 100)
weights_x2 = (
    model.attn_weights[:, 1].numpy().reshape(X1_grid.shape)
)  # Reshape to (100, 100)

# Plotting the heatmap
plt.figure(figsize=(8, 6))
plt.imshow(
    predictions, extent=(-1, 3, -1, 3), origin="lower", cmap="viridis", alpha=0.8
)
plt.colorbar(label="Probability")
plt.title("Model Predictions Heatmap")
plt.xlabel("X1")
plt.ylabel("X2")
plt.scatter(
    X1_tensor.numpy(), X2_tensor.numpy(), s=1, color="white"
)  # Optional: Overlay the input points
plt.grid(False)

plt.plot(x1[C == 1], x2[C == 1], "ro", label="Class 1 (C=1)")
plt.plot(x1[C == 0], x2[C == 0], "bo", label="Class 0 (C=0)")
plt.legend()
plt.show()


# Weights
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
cax = ax[0].imshow(
    weights_x1, extent=(-1, 3, -1, 3), origin="lower", cmap="viridis", alpha=0.8
)
# fig.colorbar(cax, ax=ax[0], label="Probability")
ax[0].set_title("Model Predictions Heatmap")
ax[0].set_xlabel("X1")
ax[0].set_ylabel("X2")
ax[0].grid(True)

ax[0].plot(x1[C == 0], x2[C == 0], "bo", label="Class 0 (C=0)")
ax[0].plot(x1[C == 1], x2[C == 1], "ro", label="Class 1 (C=1)")
ax[0].legend()

cax = ax[1].imshow(
    weights_x2, extent=(-1, 3, -1, 3), origin="lower", cmap="viridis", alpha=0.8
)
# fig.colorbar(cax, ax=ax[1], label="Probability")
ax[1].set_title("Attention weight of X2")
ax[1].set_xlabel("X1")
ax[1].set_ylabel("X2")
ax[1].grid(True)

ax[1].plot(x1[C == 0], x2[C == 0], "bo", label="Class 0 (C=0)")
ax[1].plot(x1[C == 1], x2[C == 1], "ro", label="Class 1 (C=1)")

fig.colorbar(cax, ax=ax, orientation="vertical", label="Probability", fraction=0.05, pad=0.05)

# %% Plot weights


# %%

num_heads = 1
batch_size = 3
sequence_length = 2
embed_dim = 1

multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
green_print()

query = torch.rand(
    sequence_length, batch_size, embed_dim
)  # (sequence_length, batch_size, embed_dim)
key = torch.rand(sequence_length, batch_size, embed_dim)
value = torch.rand(sequence_length, batch_size, embed_dim)
magenta_print("Shape data", key.shape)
magenta_print("Data", key)

attn_output, attn_output_weights = multihead_attn(query, key, value)
red_print("Weights:", attn_output_weights)


# %%
num_heads = 1
batch_size = 3
sequence_length = 2
embed_dim = 1

multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
green_print()

query = torch.rand(
    batch_size, sequence_length, embed_dim
)  # (sequence_length, batch_size, embed_dim)
key = torch.rand(batch_size, sequence_length, embed_dim)
value = torch.rand(batch_size, sequence_length, embed_dim)
teal_print("data", key)
magenta_print("Shape data", key.shape)
magenta_print("Data", key)

attn_output, attn_output_weights = multihead_attn(query, key, value)
red_print("Weights:", attn_output_weights)
red_print("Output:", attn_output)
