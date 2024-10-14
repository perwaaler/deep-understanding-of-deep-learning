# %%
# import libraries
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# for getting summary info on models
from torchsummary import summary

import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
from utilities import *

set_matplotlib_formats("png", "svg")  # Will render plots as PNG and SVG

# %% Import and process the data
# import dataset (comes with colab!)
data = pd.read_csv("Data/mnist_train_small.csv", delimiter=";").to_numpy()
# data = np.loadtxt(open("Data/mnist_train_small.csv", "rb"), delimiter=",")

# extract labels (number IDs) and remove from data
labels = data[:, 0]
data = data[:, 1:]

# normalize the data to a range of [0 1]
dataNorm = data / np.max(data)

# NEW: reshape to 2D!
dataNorm = dataNorm.reshape(dataNorm.shape[0], 1, 28, 28)

silent_print(f"Data shape: {dataNorm.shape}")

# %% Create train/test groups using DataLoader
# Step 1: convert to tensor
dataT = torch.tensor(dataNorm).float()
labelsT = torch.tensor(labels).long()  # Long can hold larger integers...

# Step 2: use scikitlearn to split the data
train_data, test_data, train_labels, test_labels = train_test_split(
    dataT, labelsT, test_size=0.1
)

# Step 3: convert into PyTorch Datasets
train_data = TensorDataset(train_data, train_labels)
test_data = TensorDataset(test_data, test_labels)

# Step 4: translate into dataloader objects (divide into batches)
# drop_last drops last incomplete dataset if datasize not divisible by batchsize
batchsize = 32
train_loader = DataLoader(
    train_data, batch_size=batchsize, shuffle=True, drop_last=True
)
test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])


silent_print(
    "Shape of the input train-dataset:", f"{train_loader.dataset.tensors[0].shape}"
)
silent_print(
    "Shape of the input test-dataset:", f"{test_loader.dataset.tensors[0].shape}"
)
test_loader

# %%first let's see how to shift a vectorized image

# grab one image data
tmp = test_loader.dataset.tensors[0][0, :]
# tmp = tmp.reshape(28,28) # reshape to 2D image

# shift the image (pytorch calls it "rolling")
tmpS = torch.roll(tmp, 8, dims=1)


# now show them both
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].imshow(torch.squeeze(tmp), cmap="gray")
ax[0].set_title("Original")

ax[1].imshow(torch.squeeze(tmpS), cmap="gray")
ax[1].set_title("Shifted (rolled)")

plt.show()

# %%
# check size (should be images X channels X width X height)
silent_print(f"Shape training data:", f"{train_loader.dataset.tensors[0].shape}")


# %% Create the DL model
# create a class for the model
def createTheMNISTNet(printtoggle=False):

    class mnistNet(nn.Module):
        def __init__(self, printtoggle):
            super().__init__()

            ### convolution layers
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=1)
            # size: np.floor( (28+2*1-5)/1 )+1 = 26/2 = 13 (/2 b/c maxpool)

            self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=1)
            # size: np.floor( (13+2*1-5)/1 )+1 = 11/2 = 5 (/2 b/c maxpool)

            # compute the number of units in FClayer (number of outputs of conv2)
            expectSize = (
                np.floor((5 + 2 * 0 - 1) / 1) + 1
            )  # fc1 layer has no padding or kernel, so set to 0/1
            expectSize = 20 * int(expectSize**2)

            ### fully-connected layer
            self.fc1 = nn.Linear(expectSize, 50)

            ### output layer
            self.out = nn.Linear(50, 10)

            # toggle for printing out tensor sizes during forward prop
            self.print = printtoggle

        # forward pass
        def forward(self, x):

            print(f"Input: {x.shape}") if self.print else None

            # convolution -> maxpool -> relu
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            print(f"Layer conv1/pool1: {x.shape}") if self.print else None

            # and again: convolution -> maxpool -> relu
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            print(f"Layer conv2/pool2: {x.shape}") if self.print else None

            # reshape for linear layer
            nUnits = x.shape.numel() / x.shape[0]
            x = x.view(-1, int(nUnits))
            if self.print:
                print(f"Vectorize: {x.shape}")

            # linear layers
            x = F.relu(self.fc1(x))
            if self.print:
                print(f"Layer fc1: {x.shape}")
            x = self.out(x)
            if self.print:
                print(f"Layer out: {x.shape}")

            return x

    # create the model instance
    net = mnistNet(printtoggle)

    # loss function
    lossfun = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    return net, lossfun, optimizer


# %%
# test the model with one batch
net, lossfun, optimizer = createTheMNISTNet(True)

X, y = next(iter(train_loader))
yHat = net(X)

# check sizes of model outputs and target variable
print(" ")
print(yHat.shape)
print(y.shape)

# now let's compute the loss
loss = lossfun(yHat, y)
print(" ")
print("Loss:")
print(loss)

# %%
# count the total number of parameters in the model
summary(net, (1, 28, 28))

# %% [markdown]
# # Create a function that trains the model

# %%
# a function that trains the model


def function2trainTheModel():

    # number of epochs
    numepochs = 10

    # create a new model
    net, lossfun, optimizer = createTheMNISTNet()

    # initialize losses
    losses = torch.zeros(numepochs)
    trainAcc = []
    testAcc = []

    # loop over epochs
    for epochi in range(numepochs):
        orange_print(f"Epoch: {epochi}")
        # loop over training data batches
        net.train()
        batchAcc = []
        batchLoss = []
        batch_counter = 0

        for X, y in train_loader:
            batch_counter += 1

            # forward pass and loss
            yHat = net(X)
            loss = lossfun(yHat, y)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss from this batch
            batchLoss.append(loss.item())

            # compute accuracy
            matches = torch.argmax(yHat, axis=1) == y  # booleans (false/true)
            matchesNumeric = matches.float()  # convert to numbers (0/1)
            accuracyPct = 100 * torch.mean(matchesNumeric)  # average and x100
            batchAcc.append(accuracyPct)  # add to list of accuracies

            if batch_counter % 100 == 0:
                silent_print(f"Batch {batch_counter}")
                silent_print(f"Accuracy: {accuracyPct}")
        # end of batch loop...

        # now that we've trained through the batches, get their average training accuracy
        trainAcc.append(np.mean(batchAcc))

        # and get average losses across the batches
        losses[epochi] = np.mean(batchLoss)

        # test accuracy
        net.eval()
        X, y = next(iter(test_loader))  # extract X,y from test dataloader
        with torch.no_grad():  # deactivates autograd
            yHat = net(X)

        # compare the following really long line of code to the training accuracy lines
        testAcc.append(100 * torch.mean((torch.argmax(yHat, axis=1) == y).float()))

    # end epochs

    # function output
    return trainAcc, testAcc, losses, net


# %% Run the model and show the results!
trainAcc, testAcc, losses, net = function2trainTheModel()

# %%
fig, ax = plt.subplots(1, 2, figsize=(16, 5))

ax[0].plot(losses, "s-")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].set_title("Model loss")

ax[1].plot(trainAcc, "s-", label="Train")
ax[1].plot(testAcc, "o-", label="Test")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy (%)")
ax[1].set_title(f"Final model test accuracy: {testAcc[-1]:.2f}%")
ax[1].legend()

plt.show()

# %%


# %% [markdown]
# # Additional explorations

# %%
# 1) Do we need both convolution layers in this model? Comment out the "conv2" layers in the mode definition. What else
#    needs to be changed in the code for this to work with one convolution layer? Once you get it working, how does the
#    accuracy compare between one and two conv layers? (hint: perhaps try adding some more training epochs)
#
#    Your observation here is actually the main reason why MNIST isn't very useful for evaluating developments in DL:
#    MNIST is way too easy! Very simple models do very well, so there is little room for variability. In fact, we'll
#    stop using MNIST pretty soon...
#
#    Final note about MNIST: You probably won't get much higher than 98% with this small dataset. These kinds of CNNs
#    can get >99% test-accuracy with the full dataset (60k samples instead of 18k).
#
