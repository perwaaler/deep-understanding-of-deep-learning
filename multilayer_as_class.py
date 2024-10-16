# %%
# import libraries
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from utilities import *

# %%
# create data

nPerClust = 100
blur = 1

A = [1, 3]
B = [1, -2]

# generate data
a = [A[0] + np.random.randn(nPerClust) * blur, A[1] + np.random.randn(nPerClust) * blur]
b = [B[0] + np.random.randn(nPerClust) * blur, B[1] + np.random.randn(nPerClust) * blur]

# true labels
labels_np = np.vstack((np.zeros((nPerClust, 1)), np.ones((nPerClust, 1))))

# concatanate into a matrix
data_np = np.hstack((a, b)).T

# convert to a pytorch tensor
data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()

# show the data
fig = plt.figure(figsize=(5, 5))
plt.plot(data[np.where(labels == 0)[0], 0], data[np.where(labels == 0)[0], 1], "bs")
plt.plot(data[np.where(labels == 1)[0], 0], data[np.where(labels == 1)[0], 1], "ko")
plt.title("The qwerties!")
plt.xlabel("qwerty dimension 1")
plt.ylabel("qwerty dimension 2")
plt.show()

# %% [markdown]
# # Functions to build and train the model


# %%


class createANNmodel(nn.Module):

    def __init__(self):
        super().__init__()

        # Initialize empty dictionary
        self.input_layer = nn.Linear(2, 16)
        self.hidden_layer_1 = nn.Linear(16, 6)
        self.hidden_layer_2 = nn.Linear(6, 3)
        self.output_layer = nn.Linear(3, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer_1(x)
        x = F.relu(x)
        x = self.hidden_layer_2(x)
        x = F.relu(x)
        x = self.output_layer(x)
        x = F.sigmoid(x)
        return x

    def test_model(self):
        n_input = self.input_layer.in_features
        x_test = torch.randn(10, n_input)
        test_pred = self.forward(x_test)
        green_print(f"Test predictions: {self.forward(x_test)}")


model = createANNmodel()
model.test_model()


class trainModel:
    def __init__(self, model, epochs, learning_rate):
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def train_model(self, data, labels):
        self.losses = torch.zeros(self.epochs)

        for epoch in range(self.epochs):
            scores = self.model.forward(data)
            loss = self.loss_function(scores, labels)
            self.losses[epoch] = loss

            # backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            labels_pred = labels_pred > 0           
            
            # Compute accuracy
            with torch.no_grad(): # Disable gradient computation for efficiency
                accuracy = 100 * torch.mean((labels_pred == labels).float())
                accuracy.append(accuracy)

        # Final forward pass for evaluation
        with torch.no_grad():
            predictions = self.model(x_data)
            total_accuracy = self.calc_accuracy(predictions, labels)

        return (
            self.model,
            total_accuracy,
            self.losses,
        )
        
        def test():
            torch.rand()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# def createANNmodel(learningRate):

#     # model architecture
#     ANNclassify = nn.Sequential(
#         nn.Linear(2, 16),  # input layer
#         nn.ReLU(),  # activation unit
#         nn.Linear(16, 1),  # hidden layer
#         nn.ReLU(),  # activation unit
#         nn.Linear(1, 1),  # output unit
#         nn.Sigmoid(),  # final activation unit
#     )

#     # loss function
#     lossfun = nn.BCELoss()  # but better to use BCEWithLogitsLoss

#     # optimizer
#     optimizer = torch.optim.SGD(ANNclassify.parameters(), lr=learningRate)

#     # model output
#     return ANNclassify, lossfun, optimizer


# %%
# a function that trains the model

# a fixed parameter
numepochs = 1000


def trainTheModel(ANNmodel):

    # initialize losses
    losses = torch.zeros(numepochs)

    # loop over epochs
    for epochi in range(numepochs):

        # forward pass
        yHat = ANNmodel(data)

        # compute loss
        loss = lossfun(yHat, labels)
        losses[epochi] = loss

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # final forward pass
    predictions = ANNmodel(data)

    # compute the predictions and report accuracy
    # NOTE: Wasn't this ">0" previously?!?!
    totalacc = 100 * torch.mean(((predictions > 0.5) == labels).float())

    return losses, predictions, totalacc


ANNmodel, lossfun, optimizer = createANNmodel(0.01)
trainTheModel(ANNmodel)
# %%


# %% [markdown]
# # Test the new code by running it once

# %%
# create everything
ANNclassify, lossfun, optimizer = createANNmodel(0.01)

# run it
losses, predictions, totalacc = trainTheModel(ANNclassify)

# report accuracy
print("Final accuracy: %g%%" % totalacc)


# show the losses
plt.plot(losses.detach(), "o", markerfacecolor="w", linewidth=0.1)
plt.xlabel("Epoch"), plt.ylabel("Loss")
plt.show()

# %% [markdown]
# # Now for the real test (varying learning rates)

# %%
# learning rates
learningrates = np.linspace(0.001, 0.1, 50)

# initialize
accByLR = []
allLosses = np.zeros((len(learningrates), numepochs))


# the loop
for i, lr in enumerate(learningrates):

    # create and run the model
    ANNclassify, lossfun, optimizer = createANNmodel(lr)
    losses, predictions, totalacc = trainTheModel(ANNclassify)

    # store the results
    accByLR.append(totalacc)
    allLosses[i, :] = losses.detach()


# %%
# plot the results
fig, ax = plt.subplots(1, 2, figsize=(16, 4))

ax[0].plot(learningrates, accByLR, "s-")
ax[0].set_xlabel("Learning rate")
ax[0].set_ylabel("Accuracy")
ax[0].set_title("Accuracy by learning rate")

ax[1].plot(allLosses.T)
ax[1].set_title("Losses by learning rate")
ax[1].set_xlabel("Epoch number")
ax[1].set_ylabel("Loss")
plt.show()

# %%
accByLR

# %%
sum(torch.tensor(accByLR) > 70) / len(accByLR)

# %%


# %% [markdown]
# # Additional explorations

# %%
# 1) The code creates a model with 16 hidden units. Notice where the two "16"s appear when constructing the model.
#    Recreate the model using 32 hidden units. Does that help with the issue of models getting stuck in local minima?
#
# 2) Adjust the code to create two hidden layers. The first hidden layer should have 16 hidden units and the second
#    hidden layer shuold have 32 units. What needs to change in the code to make the numbers match to prevent coding errors?
#
