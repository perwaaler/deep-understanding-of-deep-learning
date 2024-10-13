# %%
import torch
import torch.nn as nn
import seaborn as sns
from utilities import *

# %%


class ANNiris(nn.Module):
    def __init__(self):
        super(ANNiris, self).__init__()

        # Define layers
        self.input_layer = nn.Linear(4, 64)  # input layer
        self.hidden_layer = nn.Linear(64, 64)  # hidden layer
        self.output_layer = nn.Linear(64, 3)  # output layer

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass through the layers
        x = self.relu(self.input_layer(x))  # Input layer + ReLU
        x = self.relu(self.hidden_layer(x))  # Hidden layer + ReLU
        x = self.output_layer(x)  # Output layer (no activation for CrossEntropyLoss)
        return x


model = ANNiris()


# %% Setting up MODEL TRAINING


import torch
import torch.nn as nn
import torch.optim as optim


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Define the model architecture
class AnnIris(nn.Module):  # Class names in CamelCase
    def __init__(self, n_hidden):
        super(AnnIris, self).__init__()
        self.input_layer = nn.Linear(4, n_hidden)
        self.hidden_layer = nn.Linear(n_hidden, n_hidden)
        self.output_layer = nn.Linear(n_hidden, 3)
        self.relu_activation = nn.ReLU()

    def forward(self, x):
        x = self.relu_activation(self.input_layer(x))
        x = self.relu_activation(self.hidden_layer(x))
        x = self.output_layer(x)
        return x


# Define the Trainer class
class ModelTrainer:
    def __init__(self, model, learning_rate=0.01, n_epochs=400):
        self.model = model  # Model is passed in during initialization
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        # Loss function and optimizer
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def calc_accuracy(self, scores, labels):
        matches = torch.argmax(scores, axis=1) == labels
        accuracy_pct = 100 * torch.mean(matches.float())
        return accuracy_pct

    def train_model(self, x_data, labels):
        self.losses = torch.zeros(self.n_epochs)
        ongoing_accuracy = []  # Made consistent with snake_case

        # Training loop
        for epoch_index in range(self.n_epochs):  # Made more descriptive
            # Forward pass
            self.model.train()  # Some layers (like Dropout and BatchNorm) behave differently during training vs. evaluation.
            # Predict: run data 'forward' through the layers
            y_hat = self.model(x_data)

            # Compute loss
            loss = self.loss_function(y_hat, labels)
            self.losses[epoch_index] = loss

            # ** Backpropagation and optimization step **
            # Reset gradients
            self.optimizer.zero_grad()
            # Compute the gradient for the current loss
            loss.backward()
            # Update model parameters based on the gradients
            self.optimizer.step()

            # Compute accuracy
            with torch.no_grad(): # Disable gradient computation for efficiency
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
model = ANNiris(n_hidden=64)  # Model with custom architecture
trainer = ModelTrainer(model, learning_rate=0.01, n_epochs=400)

# Assuming x_data and labels are your input data
model, final_accuracy, loss_history = trainer.train_model(x_data, y_data)


# Plotting the losses after training
trainer.plot_losses()

# Plotting the losses after training
# trainer.plot_losses()

import matplotlib.pyplot as plt

# plt.plot(loss_history.detach())
