# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import seaborn as sns
from utilities import *
import matplotlib.pyplot as plt


# %% Setting up MODEL TRAINING


# Define the model architecture
class AnnIris(nn.Module):  # Class names in CamelCase
    def __init__(self, n_hidden, batch_norm=True):
        super(AnnIris, self).__init__()
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


class ModelTrainer:
    def __init__(self, model, learning_rate=0.01, n_epochs=400, batch_size=64):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def calc_accuracy(self, scores, labels):
        matches = torch.argmax(scores, axis=1) == labels
        accuracy_pct = 100 * torch.mean(matches.float())
        return accuracy_pct

    def train_model(self, x_data, labels):
        # Create a DataLoader to handle batches
        dataset = TensorDataset(x_data, labels)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.losses = torch.zeros(self.n_epochs)
        ongoing_accuracy = []  # Made consistent with snake_case

        # Training loop
        for epoch_index in range(self.n_epochs):  # Made more descriptive
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

    def plot_losses(self):
        plt.figure(figsize=(3, 3))
        plt.plot(self.losses.detach())
        plt.title("Training Loss over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid()
        plt.show()


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

# Plotting the losses after training
# trainer.plot_losses()

import matplotlib.pyplot as plt

# plt.plot(loss_history.detach())
