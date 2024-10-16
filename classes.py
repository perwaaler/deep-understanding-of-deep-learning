# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
import seaborn as sns
from utilities import *
import matplotlib.pyplot as plt


# %% Setting up MODEL TRAINING


# Define the model architecture
class AnnIris(nn.Module):  # Class names in CamelCase
    def __init__(self, n_hidden=64, batch_norm=True):
        super(AnnIris, self).__init__()
        # Model Architecture
        self.input_layer = nn.Linear(4, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.hidden_layer = nn.Linear(n_hidden, n_hidden)
        self.bn2 = nn.BatchNorm1d(n_hidden)
        self.output_layer = nn.Linear(n_hidden, 3)
        self.relu_activation = nn.ReLU()
        self.batch_norm = batch_norm

    def forward(self, x):
        x = self.input_layer(x)
        x = self.bn1(x) if self.batch_norm else x
        x = self.relu_activation(x)  # Activations layer 1

        x = self.hidden_layer(x)
        x = self.bn2(x) if self.batch_norm else x
        x = self.hidden_layer(x)
        x = self.relu_activation(x)  # Activations layer 2

        x = self.output_layer(x)  # Activations output layer

        return x

    def predict_label(self, x):
        return self.forward(x).argmax(axis=1)

    def test(self):
        self.eval()  # Set the model to evaluation mode
        n_input_layer = self.input_layer.in_features
        inputs_test = torch.randn(3, n_input_layer)
        green_print(f"Test input: {inputs_test}")
        test_preds = self.forward(inputs_test)
        green_print(f"Test input: {inputs_test}")
        blue_print(f"Test output: {test_preds}")
        return


model = AnnIris()


class ModelTrainer:
    def __init__(self, model, learning_rate=0.01, n_epochs=400, batch_size=64):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.loss_function = F.cross_entropy
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def calc_accuracy(self, scores, labels):
        matches = torch.argmax(scores, axis=1) == labels
        accuracy_pct = 100 * torch.mean(matches.float())
        return accuracy_pct

    def train_model(self, x_data, labels, epochs=None):
        epochs = epochs if epochs else self.n_epochs
        # Create a DataLoader to handle batches
        dataset = TensorDataset(x_data, labels)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.losses = torch.zeros(epochs)
        ongoing_accuracy = []  # Made consistent with snake_case

        # Training loop
        for epoch_index in range(epochs):  # Made more descriptive
            self.model.train()  # Some layers (like Dropout and BatchNorm) behave differently during training vs. evaluation.
            epoch_loss = 0

            for batch_data, batch_labels in data_loader:
                # Forward pass
                # Predict: run data 'forward' through the layers
                y_hat = self.model(batch_data)

                # Compute loss
                loss = self.loss_function(y_hat, batch_labels)
                epoch_loss += loss.item()

                # ** Backpropagation and optimization step **
                self.optimizer.zero_grad()  # Reset gradients
                loss.backward()  # Compute the gradient for the current loss
                self.optimizer.step()  # Update model parameters based on the gradients

            # Store average loss for this epoch
            self.losses[epoch_index] = epoch_loss / len(data_loader)
            y_hat = self.model(x_data)

            # Compute accuracy
            with torch.no_grad():  # Disable gradient computation for efficiency
                ongoing_accuracy.append(self.calc_accuracy(y_hat, labels))

        # Final forward pass for evaluation
        with torch.no_grad():
            predictions = self.model(x_data)
            total_accuracy = self.calc_accuracy(predictions, labels)

        return (
            self.model,
            total_accuracy,
            self.losses,
        )

    def simulate_training_data(self, n_samples=5, sigma_e=0.1):
        """Simulates a linear model with X * W = Y (normal random values)."""
        n_nodes_input = self.model.input_layer.in_features
        n_nodes_output = self.model.output_layer.out_features
        # Weights matrix that maps X to Y
        W = torch.randn(n_nodes_input, n_nodes_output)
        X = torch.randn(n_samples, n_nodes_input)
        noise = torch.randn(n_samples, n_nodes_output)
        Y = torch.matmul(X, W) + noise * sigma_e
        labels = Y.argmax(axis=1)
        return X, labels

    def test_training(self, n_samples=5, epochs=3, sigma_e=0.1):
        """Runs a training test run with simulated data."""
        orange_print("Test run with simulated data")
        data, labels = self.simulate_training_data(n_samples)
        green_print(
            f"First 3 test inputs:",
            f"{data[:3, :]}",
            f"First 3 test output: {labels[:3]}",
        )

        _, acc, losses = self.train_model(data, labels, epochs)
        blue_print(
            f"Test Accuracy: {acc:.3}%",
            f"First 3 Test losses {losses.detach()[:3]} ...",
        )
        ax = self.plot_losses()
        ax.set_title("Losses test training")

    def plot_losses(self):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(self.losses.detach())
        ax.set_title("Training Loss over Epochs")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.grid()
        return ax


torch.randn(10, 10)
ANNmodel = AnnIris(n_hidden=3)
# ANNmodel.input_layer.in_features
# ANNmodel.output_layer.out_features


modelTrainer = ModelTrainer(ANNmodel, learning_rate=0.1)
modelTrainer.test_training(n_samples=500, epochs=500, sigma_e=0.3)


# n = 5
# n_nodes_input = 4
# n_nodes_output = 3
# torch.randn(4, 7)

# # X = tf.random.normal(shape=(n_samples, input_dim))
# W = torch.randn(n_nodes_input, n_nodes_output)
# X = torch.randn(n, n_nodes_input)
# noise = torch.randn(n, n_nodes_output)
# Y = (torch.matmul(X, W) + noise).argmax(axis=1)
# magenta_print(Y)

# %% Train on Iris Dataset
iris = sns.load_dataset("iris")
predictor_columns = iris.columns[:4].to_list()


def prepare_data_for_analysis(iris_df):
    data = torch.tensor(iris_df[predictor_columns].values).float()
    # transform species to number
    labels = torch.zeros(len(data), dtype=torch.long)
    labels[iris_df.species == "setosa"] = 0
    labels[iris_df.species == "versicolor"] = 1
    labels[iris_df.species == "virginica"] = 2
    return data, labels


x_data, y_data = prepare_data_for_analysis(iris_df=iris)

# Example of using the class
model = AnnIris(n_hidden=64, batch_norm=True)  # Model with custom architecture
trainer = ModelTrainer(model, learning_rate=0.01, n_epochs=400, batch_size=2**7)

# Assuming x_data and labels are your input data
model, final_accuracy, loss_history = trainer.train_model(x_data, y_data)


# Plotting the losses after training
trainer.plot_losses()
