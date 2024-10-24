"""
This script simulates two modalities of input data and applies a simple attention-based 
neural network to predict output labels. The model is trained on mini-batches using an 
Adam optimizer and Mean Squared Error loss. The goal is to test how attention mechanisms 
work in a multimodal learning setup and evaluate the model's performance by tracking the 
loss and correlation between predicted and true labels.
"""

# %%
import matplotlib.pyplot as plt
import path_setup
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from utilities import *


# Define the attention mechanism class
class SimpleAttentionTrueModel(nn.Module):
    def __init__(self, len_x1=10, len_x2=8, ndim_out=1):
        super(SimpleAttentionTrueModel, self).__init__()

        # Define linear transformations for queries, keys, and values
        self.W_Q = nn.Linear(len_x1, 5)
        self.W_K = nn.Linear(len_x2, 5)
        self.W_V = nn.Linear(len_x2, 5)
        self.fc1 = nn.Linear(5, 4)  # Final output layer
        self.fc2 = nn.Linear(4, ndim_out)  # Final output layer

    def forward(self, X1, X2):
        # Compute queries, keys, and values
        Q1 = self.W_Q(X1)  # Queries from modality 1
        K2 = self.W_K(X2)  # Keys from modality 2
        V2 = self.W_V(X2)  # Values from modality 2

        # Compute attention scores
        scores = torch.matmul(Q1, K2.T)  # Shape: (batch_size, batch_size)
        self.attention_weights = F.softmax(
            scores, dim=-1
        )  # Normalize to get attention weights

        # Compute the weighted sum of values
        output = torch.matmul(
            self.attention_weights, V2
        )  # Shape: (batch_size, output_dim)

        # Pass through the first fully connected layer with activation
        output = self.fc1(output)  # Shape: (batch_size, 4)
        output = F.relu(output)  # Apply ReLU activation

        # Pass through the second fully connected layer to get final output
        output = self.fc2(output)  # Shape: (batch_size, 1)

        return output


n_data = 300  # Number of observations
batch_size = 32
len_x1 = 7
len_x2 = 6
ndim_out = 1

# Random data for modality 1 and modality 2
X1 = torch.rand(n_data, len_x1)
X2 = torch.rand(n_data, len_x2)

# Simulate Ground Truth Labels
with torch.no_grad():  # No gradient tracking for this operation
    Y = SimpleAttentionTrueModel(
        len_x1=len_x1, len_x2=len_x2, ndim_out=ndim_out
    ).forward(X1, X2)
    Y = Y - Y.min()  # Avoid in-place operation
    Y = Y / Y.max()  # Normalize to range [0, 1]

    # Perturb X1 and X2 a bit
    X1 = X1 + torch.rand(n_data, len_x1) * 0.3
    X2 = X2 + torch.rand(n_data, len_x2) * 0.2

# Split into Training and Test set
X1_train, X1_test, X2_train, X2_test, Y_train, Y_test = train_test_split(
    X1, X2, Y, test_size=0.2
)


class SimpleAttentionTrainer:
    def __init__(
        self,
        model,
        X1,
        X2,
        Y,
        batch_size=32,
        learning_rate=0.03,
        num_epochs=50,
        plotLosses=True,
    ):
        # Prepare Dataset
        train_dataset = TensorDataset(X1, X2, Y)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.model = model
        # Define loss function and optimizer
        self.loss_function = F.mse_loss
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs

        # Utility variables
        self.losses = []
        self.plotLosses = plotLosses

    def train(self):
        for epoch in range(self.num_epochs):
            running_loss = 0.0

            # Iterate over batches
            for X1_batch, X2_batch, Y_batch in self.train_loader:
                # Zero the gradients
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.model(X1_batch, X2_batch)
                # Compute loss
                loss = self.loss_function(outputs, Y_batch)
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            n_epochs_in_batch = len(self.train_loader)
            self.losses.append(running_loss / n_epochs_in_batch)

        if self.plotLosses:
            plt.figure(figsize=(3, 3))
            plt.plot(self.losses)
            plt.title("Training Loss")
            plt.show()

        green_print(f"Training complete. Final Loss: {self.losses[-1]:.4f}")


model = SimpleAttentionTrueModel(len_x1=len_x1, len_x2=len_x2, ndim_out=ndim_out)
modelTrainer = SimpleAttentionTrainer(model=model, X1=X1_train, X2=X2_train, Y=Y_train)
modelTrainer.train()

y_pred_test = modelTrainer.model(X1_test, X2_test)

df = pd.DataFrame(
    {
        "y_pred": y_pred_test.detach().flatten(),
        "y_true": Y_test.detach().flatten(),
    }
)

magenta_print(
    f"Relative Standard Deviation: {df['y_pred'].std()/df['y_pred'].abs().mean():.3}"
)
plt.figure(figsize=(3, 3))
plt.plot(df["y_pred"], df["y_true"], "*")
plt.xlabel("y_pred")
plt.ylabel("y_true")
green_print(f"Test Correlation(y_hat, y_true): {df.corr().iloc[0, 1]:.3}")


# %%


# Define the attention mechanism class
class SimpleAttentionApproximator(nn.Module):
    def __init__(self, len_x1=10, len_x2=8, ndim_out=1):
        super(SimpleAttentionApproximator, self).__init__()

        # Define linear transformations for queries, keys, and values
        attention_out_dim = 3
        self.W_Q = nn.Linear(len_x1, attention_out_dim)
        self.W_K = nn.Linear(len_x2, attention_out_dim)
        self.W_V = nn.Linear(len_x2, attention_out_dim)
        self.fc1 = nn.Identity(
            attention_out_dim, attention_out_dim
        )  # Final output layer
        self.fc2 = nn.Linear(attention_out_dim, ndim_out)  # Final output layer

    def forward(self, X1, X2):
        # Compute queries, keys, and values
        Q1 = self.W_Q(X1)  # Queries from modality 1
        K2 = self.W_K(X2)  # Keys from modality 2
        V2 = self.W_V(X2)  # Values from modality 2

        # Compute attention scores
        scores = torch.matmul(Q1, K2.T)  # Shape: (batch_size, batch_size)
        self.attention_weights = F.softmax(
            scores, dim=-1
        )  # Normalize to get attention weights

        # Compute the weighted sum of values
        output = torch.matmul(
            self.attention_weights, V2
        )  # Shape: (batch_size, output_dim)

        # Pass through the first fully connected layer with activation
        output = self.fc1(output)  # Shape: (batch_size, 4)
        output = F.relu(output)  # Apply ReLU activation

        # Pass through the second fully connected layer to get final output
        output = self.fc2(output)  # Shape: (batch_size, 1)

        return output
