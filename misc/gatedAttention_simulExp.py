"""Here I implement the code from Gated Multimodal Units for Information Fusion.

keywords: Gated units, attention, Transformer, heatmap"""

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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        h_dim = 1
        len_xv = 1
        len_xt = 1
        width_Wz = len_xv + len_xt
        # Matrices used to map inputs to feature representations in shared space
        self.Wv = nn.Linear(len_xv, h_dim)
        self.Wt = nn.Linear(len_xt, h_dim)
        # Matrix used to map concatenated vector to weight z
        self.Wz = nn.Linear(width_Wz, h_dim)
        self.bn1 = nn.BatchNorm1d(num_features=2)

    def forward(self, xv, xt):
        hv = F.tanh(self.Wv(xv))
        ht = F.tanh(self.Wt(xt))
        x = torch.cat((xv, xt), dim=-1)
        x = self.bn1(x)
        z = F.sigmoid(self.Wz(x))
        h = hv * z + ht * (1 - z)
        h = F.sigmoid(h)
        return h


class SelfAttention2DPoints(nn.Module):
    """Simple implementation of the transformer architecture where"""

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

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(num_features=2)
        self.bn2 = nn.BatchNorm1d(num_features=2)
        # Fully Connected layers
        self.fc0 = nn.Linear(self.seq_length, self.seq_length)
        self.fc1 = nn.Linear(self.seq_length, 1)

    def forward(self, x1, x2):
        # ** Prepare Data **
        x = torch.cat([x1, x2], dim=-1)
        x = self.bn1(x)
        # Separate each sequence into its own coordinate
        # Batch size, sequence length, length of embeddings making up each sequence
        x = x.reshape(-1, self.seq_length, self.embed_dim)
        # Compute attention
        x, attn_weights = self.att(x, x, x)

        # Extract weights
        attn_weights = [[w[0, 1], w[1, 1]] for w in attn_weights]
        self.attn_weights = torch.tensor(attn_weights)

        x = x.reshape(-1, 2)  # Reshape back to N by 2 shape
        x = self.bn2(x)
        # Finally, reduce to prediction and apply softmax
        output = F.sigmoid(self.fc1(x))

        return output


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        # Matrices used to map inputs to feature representations in shared space
        self.layers = nn.ModuleDict(
            {
                "fc1": nn.Linear(2, 2),
                "fc2": nn.Linear(2, 1),
            }
        )
        self.bn1 = nn.BatchNorm1d(num_features=2)

    def forward(self, xv, xt):
        x = torch.concat((xv, xt), dim=-1)
        x = self.bn1(x)
        for layer in self.layers.values():
            x = F.sigmoid(layer(x))
        return x


def n_params(model):
    return sum(len(p) for p in model.parameters())


GatedModel = GatedAttention()

x1 = torch.tensor([[1.0], [2.0], [3.0]])
x2 = torch.tensor([[2.0], [1.5], [1.0]])
red_print("Concatted data:\n", torch.concat([x1, x2], dim=1))

FeedForwardModel = FeedForward()
FeedForwardModel(x1, x2)

SelfAttentionModel = SelfAttention2DPoints()
SelfAttentionModel.forward(x1, x2)

blue_print(f"N.o. params Benchmark: {n_params(FeedForwardModel)}")
blue_print(f"N.o. params SelfAttention: {n_params(SelfAttentionModel)}")
green_print(f"N.o. params GatedModel: {n_params(GatedModel)}")

# %% Function for simulating data
"""Here I simulate the same data as in the paper: 'Gated Multimodal Units for
Information Fusion'"""


def simulate_data(n_samples=200, plot_data=False):
    # Step 1: Simulate class label C with probability 0.5
    C = np.random.binomial(1, 0.5, n_samples)

    # Step 2: Simulate whether x1 or x2 is informative (M)
    M = np.random.binomial(1, 0.5, n_samples)

    # Step 3 & 4: Set informative distributions based on C
    sigma_e = 0.25
    informative_dist1 = np.where(
        C == 1,
        np.random.normal(0, sigma_e, n_samples),
        np.random.normal(1, sigma_e, n_samples),
    )
    informative_dist2 = np.where(
        C == 1,
        np.random.normal(2, sigma_e, n_samples),
        np.random.normal(1, sigma_e, n_samples),
    )

    # Step 5: Simulate the value of x1
    x1 = M * informative_dist1 + (1 - M) * np.random.normal(2, sigma_e, n_samples)

    # Step 6: Simulate the value of x2
    x2 = M * np.random.normal(0, sigma_e, n_samples) + (1 - M) * informative_dist2

    if plot_data:
        xx = np.linspace(-1, 3, 100)
        plt.plot(x1[C == 1], x2[C == 1], "r*")
        plt.plot(x1[C == 0], x2[C == 0], "g*")
        plt.plot(xx, 2 - xx)

    return x1, x2, C


# %% Define Model Trainer Class


def print_dict(d, c=None, header=None, n_digits=3, indent_values=True):
    """Function that prints the values in a dictionary"""
    if c is None:
        print_func = print
    else:
        print_func = lambda x: eval(f"{c}_print")(x)
    indent = ""
    if indent_values:
        indent = "  "
    if header:
        print_func(header)
    for key, value in d.items():
        if isinstance(value, int) or isinstance(value, str):
            print_func(f"{indent}{key}: {value}")
        else:
            print_func(f"{indent}{key}: {value:.{n_digits}}")


def calc_acc(X1, X2, labels, model):
    model.eval()
    predictions = (model(X1, X2) >= 0.5).float()
    matches = predictions == labels
    return (matches).float().mean()


def calc_sn(X1, X2, labels, model):
    model.eval()
    positive = (labels == 1).flatten()
    predictions = (model(X1[positive], X2[positive]) >= 0.5).float()
    matches = predictions == labels[positive]
    return (matches).float().mean()


def calc_sp(X1, X2, labels, model):
    model.eval()
    positive = (labels == 0).flatten()
    predictions = (model(X1[positive], X2[positive]) >= 0.5).float()
    matches = predictions == labels[positive]
    return (matches).float().mean()


def calc_acc_sn_sp(X1, X2, labels, model) -> list:
    return {
        "acc": calc_acc(X1, X2, labels, model),
        "sn": calc_sn(X1, X2, labels, model),
        "sp": calc_sp(X1, X2, labels, model),
    }


class ModelTrainer:
    def __init__(
        self,
        model,
        learning_rate=0.05,
        num_epochs=200,
        batch_size=32,
        n_max_consecutive_failures=10,
    ):
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.loss_function = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.losses = []
        self.n_max_consecutive_failures = n_max_consecutive_failures

    def train_model(self, X1, X2, labels, plot_losses=False):
        # Prepare Data for Training Loop
        train_dataset = TensorDataset(X1, X2, labels)
        dataset_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        consecutive_failures_to_improve = 0

        for i in range(self.num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0

            # Iterate over the batches
            for x1_batch, x2_batch, labels_batch in dataset_loader:
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # Forward pass
                batch_predictions = self.model(x1_batch, x2_batch)
                # Compute the loss
                loss = self.loss_function(batch_predictions, labels_batch)
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                # Accumulate loss
                running_loss += loss.item()

            avg_epoch_loss = running_loss / len(dataset_loader)
            self.losses.append(avg_epoch_loss)

            if i >= 1:
                if self.losses[-1] > min(self.losses):
                    consecutive_failures_to_improve += 1

                else:
                    consecutive_failures_to_improve = 0

            if consecutive_failures_to_improve >= self.n_max_consecutive_failures:
                red_print(
                    f"Failed to improve over {consecutive_failures_to_improve} Epochs: terminating training"
                )
                break

        if plot_losses:
            plt.plot(self.losses)
            plt.xlabel("epochs")
            plt.show()


# %% ** Prepare Data
x1, x2, C = simulate_data(n_samples=100)

# Convert data to tensor format
X1 = torch.tensor(x1).unsqueeze(1).float()
X2 = torch.tensor(x2).unsqueeze(1).float()
labels = torch.tensor(C).unsqueeze(1).float()

X1_train, X1_test, X2_train, X2_test, labels_train, labels_test = train_test_split(
    X1, X2, labels, test_size=0.25
)
lime_print(f"N.o. Training Obs: {len(X1_train)}")
lime_print(f"N.o. Test Obs: {len(X1_test)}")

if False:
    model = ModelTrainer(
        SelfAttention2DPoints(),
        n_max_consecutive_failures=20,
        num_epochs=50,
        learning_rate=0.01,
    )
    model.train_model(X1_train, X2_train, labels_train, plot_losses=True)
    green_print(calc_acc(X1_test, X2_test, labels_test, model.model))

# ** Training **
GatedModel = GatedAttention()
SelfAttentionModel = SelfAttention2DPoints()
FeedForwardModel = FeedForward()

GatedModelTrainer = ModelTrainer(GatedModel)
FeedForwardTrainer = ModelTrainer(SelfAttentionModel)
SelfAttentionTrainer = ModelTrainer(FeedForwardModel)

# Train models
GatedModelTrainer.train_model(X1_train, X2_train, labels_train)
SelfAttentionTrainer.train_model(X1_train, X2_train, labels_train)
FeedForwardTrainer.train_model(X1_train, X2_train, labels_train)

acc_Gated = calc_acc_sn_sp(X1_test, X2_test, labels_test, GatedModelTrainer.model)
acc_SelfAttention = calc_acc_sn_sp(
    X1_test, X2_test, labels_test, SelfAttentionTrainer.model
)
acc_FeedForward = calc_acc_sn_sp(
    X1_test, X2_test, labels_test, FeedForwardTrainer.model
)
calc_acc_sn_sp(X1_test, X2_test, labels_test, FeedForwardTrainer.model)
# positive = (labels_test == 0).flatten()
# predictions = (
#     SelfAttentionTrainer.model(X1_test[positive], X2_test[positive]) >= 0.5
# ).float()
# matches = predictions == labels_test[positive]
# green_print(matches.float().mean())

# Plot losses
plot_losses = True
if plot_losses:
    fig, ax = plt.subplots(1, 3, figsize=(6, 2))
    fig.suptitle("Loss")

    ax[0].plot(GatedModelTrainer.losses)
    ax[0].set_title("GatedModel")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylim([0, max(GatedModelTrainer.losses)])

    ax[1].plot(SelfAttentionTrainer.losses)
    ax[1].set_title("SelfAttention")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylim([0, max(SelfAttentionTrainer.losses)])

    ax[2].plot(FeedForwardTrainer.losses)
    ax[2].set_title("FeedForward")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylim([0, max(FeedForwardTrainer.losses)])
    plt.pause(0.1)

orange_print("** TEST ACCURACY **")
green_print("GatedModel")
print_dict(acc_Gated, "green", n_digits=5)
blue_print("AttentionModel")
print_dict(acc_SelfAttention, "blue", n_digits=5)
teal_print("FeedForward")
print_dict(acc_FeedForward, "teal", n_digits=5)

# Visualize predictions in heat map
import numpy as np
import torch
import matplotlib.pyplot as plt

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
    GatedModel.eval()  # Set model to evaluation mode
    scoresGated = GatedModelTrainer.model(X1_tensor, X2_tensor)
    scoresSelfAtt = SelfAttentionTrainer.model(X1_tensor, X2_tensor)
    scoresFeedForward = FeedForwardTrainer.model(X1_tensor, X2_tensor)

# Reshape scores to match the grid shape
scoresGated = scoresGated.numpy().reshape(X1_grid.shape)
scoresSelfAtt = scoresSelfAtt.numpy().reshape(X1_grid.shape)
scoresFeedForward = scoresFeedForward.numpy().reshape(X1_grid.shape)

# Gated Attention
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
cax = ax[0].imshow(
    scoresGated, extent=(-1, 3, -1, 3), origin="lower", cmap="viridis", alpha=0.8
)
ax[0].set_title("Gated Model")
ax[0].set_xlabel("X1")
ax[0].set_ylabel("X2")
ax[0].grid(True)
# Ground Truth
ax[0].plot(x1[C == 0], x2[C == 0], "b.", label="Class 0 (C=0)")
ax[0].plot(x1[C == 1], x2[C == 1], "r.", label="Class 1 (C=1)")
ax[0].legend()

# Self Attention Plot
cax = ax[1].imshow(
    scoresSelfAtt, extent=(-1, 3, -1, 3), origin="lower", cmap="viridis", alpha=0.8
)
ax[1].set_title("Self Attention")
ax[1].set_xlabel("X1")
ax[1].set_ylabel("X2")
ax[1].grid(True)
# Ground Truth
ax[1].plot(x1[C == 0], x2[C == 0], "b.", label="Class 0 (C=0)")
ax[1].plot(x1[C == 1], x2[C == 1], "r.", label="Class 1 (C=1)")

cax = ax[2].imshow(
    scoresFeedForward,
    extent=(-1, 3, -1, 3),
    origin="lower",
    cmap="viridis",
    alpha=0.8,
)
ax[2].set_title("FeedForward")
ax[2].set_xlabel("X1")
ax[2].set_ylabel("X2")
ax[2].grid(True)
# Ground Truth
ax[2].plot(x1[C == 0], x2[C == 0], "b.", label="Class 0 (C=0)")
ax[2].plot(x1[C == 1], x2[C == 1], "r.", label="Class 1 (C=1)")

fig.colorbar(
    cax, ax=ax, orientation="vertical", label="Probability", fraction=0.05, pad=0.05
)


# %% Compare Gated and FeedForward over multiple training sessions
run_comparison = False

if run_comparison:
    n_exp = 50
    accuracies = {"benchModel": np.zeros(n_exp), "gatedModel": np.zeros(n_exp)}
    n_samples = 200
    for i in range(n_exp):
        x1, x2, C = simulate_data(n_samples=150)
        # Convert data to tensor format
        X1 = torch.tensor(x1).unsqueeze(1).float()
        X2 = torch.tensor(x2).unsqueeze(1).float()
        labels = torch.tensor(C).unsqueeze(1).float()

        X1_train, X1_test, X2_train, X2_test, labels_train, labels_test = (
            train_test_split(X1, X2, labels, test_size=50)
        )

        FeedForwardTrainer = ModelTrainer(FeedForwardModel)
        GatedModelTrainer = ModelTrainer(GatedModel)

        FeedForwardTrainer.train_model(X1_train, X2_train, labels_train)
        GatedModelTrainer.train_model(X1_train, X2_train, labels_train)

        # pos_pred =
        acc_bench = calc_acc(X1_test, X2_test, labels_test, FeedForwardTrainer.model)
        acc_gated = calc_acc(X1_test, X2_test, labels_test, GatedModelTrainer.model)

        accuracies["benchModel"][i] = acc_bench
        accuracies["gatedModel"][i] = acc_gated

        blue_print(f"Acc. Benchmark: {acc_bench:.3}")
        green_print(f"Acc. Gated: {acc_gated:.3}")

    pink_print("Experiment Finnished")

# %% Analyze outcome of experiment
if run_comparison:
    acc_bench = accuracies["benchModel"].mean()
    acc_gated = accuracies["gatedModel"].mean()
    acc_diffs = ((accuracies["gatedModel"] - accuracies["benchModel"]) >= 0).mean()

    magenta_print(f"Accuracy better than benchmark in {100 * acc_diffs}% of trials")
    blue_print(f"Accuracy of benchmark {100 * acc_bench}% of trials")
    green_print(f"Accuracy of gated {100 * acc_gated}% of trials")

# Beats benchmark about 85% of the time for 75 training samples
