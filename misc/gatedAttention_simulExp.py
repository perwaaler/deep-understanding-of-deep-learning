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

    def forward(self, xv, xt):
        hv = F.tanh(self.Wv(xv))
        ht = F.tanh(self.Wt(xt))
        x_concatenated = torch.cat((xv, xt), dim=1)
        z = F.sigmoid(self.Wz(x_concatenated))
        h = hv * z + ht * (1 - z)
        h = F.sigmoid(h)
        return h


class BenchmarkMLP(nn.Module):
    def __init__(self):
        super(BenchmarkMLP, self).__init__()
        # Matrices used to map inputs to feature representations in shared space
        self.layers = nn.ModuleDict(
            {
                "fc1": nn.Linear(2, 2),
                "fc2": nn.Linear(2, 1),
            }
        )

    def forward(self, xv, xt):
        X = torch.concat((xv, xt), dim=1)
        for layer in self.layers.values():
            X = F.sigmoid(layer(X))
        return X


def n_params(model):
    return sum(len(p) for p in model.parameters())


gatedModel = GatedAttention()

xv = torch.tensor([[1.0], [2.0], [3.0]])  # Example batch of size 5 with 1 feature
xt = torch.tensor([[2.0], [1.5], [1.0]])  # Example batch of size 5 with 1 feature

benchMarkModel = BenchmarkMLP()
benchMarkModel(xv, xt)

blue_print(f"N.o. params Benchmark: {n_params(benchMarkModel)}")
green_print(f"N.o. params GatedModel: {n_params(gatedModel)}")
# %% Simulate data
"""Here I simulate the same data as in the paper: 'GATED MULTIMODAL UNITS FOR
INFORMATION FUSION'"""


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


# %% Train model
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

x1, x2, C = simulate_data(n_samples=150)

# Convert data to tensor format
X1 = torch.tensor(x1).unsqueeze(1).float()
X2 = torch.tensor(x2).unsqueeze(1).float()
labels = torch.tensor(C).unsqueeze(1).float()

X1_train, X1_test, X2_train, X2_test, labels_train, labels_test = train_test_split(
    X1, X2, labels, test_size=0.5
)
lime_print(f"N.o. Training Obs: {len(X1_train)}")
lime_print(f"N.o. Test Obs: {len(X1_test)}")


class ModelTrainer:
    def __init__(
        self,
        model,
        learning_rate=0.05,
        num_epochs=200,
        batch_size=32,
        n_max_consecutive_failures=4,
    ):
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.loss_function = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.losses = []
        self.n_max_consecutive_failures = n_max_consecutive_failures

    def train_model(self, X1, X2, labels):
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


def calc_acc(X1, X2, labels, model):
    predictions = (model(X1, X2) >= 0.5).float()
    return (predictions == labels).float().mean()


trainingGatedModel = ModelTrainer(gatedModel)
trainingBenchmarkModel = ModelTrainer(benchMarkModel)
trainingGatedModel.train_model(X1_train, X2_train, labels_train)
trainingBenchmarkModel.train_model(X1_train, X2_train, labels_train)

orange_print("TEST ACCURACY")
blue_print(
    f"Benchmark: {calc_acc(X1_test, X2_test, labels_test, trainingBenchmarkModel.model):.3}"
)
green_print(
    f"GatedModel: {calc_acc(X1_test, X2_test, labels_test, trainingGatedModel.model):.3}"
)

fig, ax = plt.subplots(1, 2, figsize=(5, 4))
ax[0].plot(trainingBenchmarkModel.losses)
ax[0].set_title("Benchmark Loss")
ax[0].set_xlabel("Epoch")
ax[1].plot(trainingGatedModel.losses)
ax[1].set_title("GatedModel Loss")
ax[1].set_xlabel("Epoch")


# %% Visualize predictions in heatmap
import numpy as np
import torch
import matplotlib.pyplot as plt

# Assuming `model` is your trained model
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
    gatedModel.eval()  # Set model to evaluation mode
    predictions = gatedModel(
        X1_tensor, X2_tensor
    )  # Model outputs logits or probabilities

# Reshape predictions to match the grid shape
predictions = predictions.numpy().reshape(X1_grid.shape)  # Reshape to (100, 100)

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


# %% Compare models over multiple training sessions
n_exp = 50
accuracies = {"benchModel": np.zeros(n_exp), "gatedModel": np.zeros(n_exp)}
n_samples = 200

for i in range(n_exp):
    x1, x2, C = simulate_data(n_samples=150)
    # Convert data to tensor format
    X1 = torch.tensor(x1).unsqueeze(1).float()
    X2 = torch.tensor(x2).unsqueeze(1).float()
    labels = torch.tensor(C).unsqueeze(1).float()

    X1_train, X1_test, X2_train, X2_test, labels_train, labels_test = train_test_split(
        X1, X2, labels, test_size=50
    )

    trainingBenchmarkModel = ModelTrainer(benchMarkModel)
    trainingGatedModel = ModelTrainer(gatedModel)

    trainingBenchmarkModel.train_model(X1_train, X2_train, labels_train)
    trainingGatedModel.train_model(X1_train, X2_train, labels_train)

    acc_bench = calc_acc(X1_test, X2_test, labels_test, trainingBenchmarkModel.model)
    acc_gated = calc_acc(X1_test, X2_test, labels_test, trainingGatedModel.model)

    accuracies["benchModel"][i] = acc_bench
    accuracies["gatedModel"][i] = acc_gated

    blue_print(f"Acc. Benchmark: {acc_bench:.3}")
    green_print(f"Acc. Gated: {acc_gated:.3}")

pink_print("Experiment Finnished")

# %% Analyze outcome of experiment

acc_bench = accuracies["benchModel"].mean()
acc_gated = accuracies["gatedModel"].mean()
acc_diffs = ((accuracies["gatedModel"] - accuracies["benchModel"]) >= 0).mean()

magenta_print(f"Accuracy better than benchmark in {100 * acc_diffs}% of trials")
blue_print(f"Accuracy of benchmark {100 * acc_bench}% of trials")
green_print(f"Accuracy of gated {100 * acc_gated}% of trials")

# Beats benchmark about 85% of the time for 75 training samples
