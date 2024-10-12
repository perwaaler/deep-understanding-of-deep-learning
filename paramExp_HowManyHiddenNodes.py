# %% [markdown]
# # COURSE: A deep understanding of deep learning
# ## SECTION: ANNs
# ### LECTURE: Multi-output ANN (iris dataset)
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/dudl/?couponCode=202207

# %%
# import libraries
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from IPython import display

from utilities import green_print, orange_print, silent_print, magenta_print, blue_print

display.set_matplotlib_formats("svg")

# %%
# import dataset (comes with seaborn)
import seaborn as sns

iris = sns.load_dataset("iris")

# check out the first few lines of data
iris.head()

predictor_columns = iris.columns[:4].to_list()
green_print(f"Predictors: {predictor_columns}")
# %%
# some plots to show the data
import seaborn as sns
import numpy as np

# sns.pairplot(iris, hue="species")
# plt.show()
green_print(f"N.o. samples: {len(iris)}")

n_test = int(len(iris) * 0.15)


orange_print("Create Training and test set")
np.random.seed(42)
index_test = iris.sample(n=n_test).index
index_train = iris.index.difference(index_test)

iris_test = iris.loc[index_test].reset_index(drop=True)
iris_train = iris.loc[index_train].reset_index(drop=True)
green_print(f"N training samples: {len(iris_train)}")
green_print(f"N test samples: {len(iris_test)}")

# %%
orange_print("Prepare the data")


def prepare_data_for_analysis(iris_df):
    data = torch.tensor(iris_df[predictor_columns].values).float()
    # transform species to number
    labels = torch.zeros(len(data), dtype=torch.long)
    labels[iris_df.species == "setosa"] = 0
    labels[iris_df.species == "versicolor"] = 1
    labels[iris_df.species == "virginica"] = 2
    return data, labels


x_train, labels_train = prepare_data_for_analysis(iris_df=iris_train)
x_test, labels_test = prepare_data_for_analysis(iris_df=iris_test)

green_print(f"N obs.: {len(labels_train)}")
green_print(f"N obs. C1: {(labels_train==0).sum().item()}")
green_print(f"N obs. C2: {(labels_train==1).sum().item()}")
green_print(f"N obs. C3: {(labels_train==2).sum().item()}")


# %%
# model architecture
ANNiris = nn.Sequential(
    nn.Linear(4, 64),  # input layer
    nn.ReLU(),  # activation
    nn.Linear(64, 64),  # hidden layer
    nn.ReLU(),  # activation
    nn.Linear(64, 3),  # output layer
)

# loss function
lossfun = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(ANNiris.parameters(), lr=0.01)

numepochs = 1000

# initialize losses
losses = torch.zeros(numepochs)
ongoingAcc = []

# loop over epochs
for epochi in range(numepochs):

    # forward pass
    yHat = ANNiris(x_train)

    # compute loss
    loss = lossfun(yHat, labels_train)
    losses[epochi] = loss

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # compute accuracy
    matches = torch.argmax(yHat, axis=1) == labels_train  # booleans (false/true)
    matchesNumeric = matches.float()  # convert to numbers (0/1)
    accuracyPct = 100 * torch.mean(matchesNumeric)  # average and x100
    ongoingAcc.append(accuracyPct)  # add to list of accuracies


# final forward pass
predictions = ANNiris(x_train)

predlabels = torch.argmax(predictions, axis=1)
totalacc = 100 * torch.mean((predlabels == labels_train).float())
plt.plot(losses.detach())

# %%


def train_model(n_hidden, x_data, labels, learning_rate=0.01, n_epochs=400):
    # model architecture
    ANNiris = nn.Sequential(
        nn.Linear(4, n_hidden),  # input layer
        nn.ReLU(),  # activation
        nn.Linear(n_hidden, n_hidden),  # hidden layer
        nn.ReLU(),  # activation
        nn.Linear(n_hidden, 3),  # output layer
    )

    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=ANNiris.parameters(), lr=learning_rate)

    # initialize losses
    losses = torch.zeros(n_epochs)
    ongoingAcc = []

    # loop over epochs
    for epochi in range(n_epochs):

        # forward pass
        yHat = ANNiris(x_data)

        # compute loss
        loss = lossfun(yHat, labels)
        losses[epochi] = loss

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute accuracy
        matches = torch.argmax(yHat, axis=1) == labels  # booleans (false/true)
        matchesNumeric = matches.float()  # convert to numbers (0/1)
        accuracyPct = 100 * torch.mean(matchesNumeric)  # average and x100
        ongoingAcc.append(accuracyPct)  # add to list of accuracies

    # final forward pass
    predictions = ANNiris(x_data)

    predlabels = torch.argmax(predictions, axis=1)
    totalacc = 100 * torch.mean((predlabels == labels).float()).item()
    return ANNiris, totalacc, losses


ANNiris, totalacc, losses = train_model(
    n_hidden=64,
    x_data=x_train,
    labels=labels_train,
    n_epochs=1000,
    learning_rate=0.01,
)
plt.plot(losses.detach())
green_print(f"Train Accuracy test run: {totalacc:.3}")

# %% Prepare for parametric experiment
x_vals = np.arange(0, 100, 3)
n_hidden_nodes = (x_vals + np.exp((x_vals - 1) / 15)).round().astype(int)

x_vals = np.arange(0, 100, 3)
n_hidden_nodes = (x_vals + np.exp((x_vals - 1) / 15)).round().astype(int)

plt.figure(figsize=(3, 3))
plt.plot(n_hidden_nodes)
plt.xlabel("experiment")
plt.ylabel("n_nodes")
green_print(f"Max number of nodes: {n_hidden_nodes[-1]}")
green_print(f"N nodes in experiment: {len(n_hidden_nodes)}")

# %% train in for-loop
accuracy_train = []
accuracy_test = []
loss_lists = []

for i, n_nodes in enumerate(n_hidden_nodes):
    ANNiris, totalacc, losses = train_model(
        n_hidden=n_nodes,
        x_data=x_train,
        labels=labels_train,
        learning_rate=0.03,
        n_epochs=1000,
    )
    accuracy_train.append(totalacc)
    loss_lists.append(losses)

    silent_print(f"N nodes: {n_nodes}")
    blue_print(f"Accuracy train: {accuracy_train[-1]:.3}")
    predictions_test = torch.argmax(ANNiris(x_test), axis=1)
    accuracy_test.append(
        torch.mean((predictions_test == labels_test).float()).item() * 100
    )
    magenta_print(f"Accuracy test: {accuracy_test[-1]:.3}")

    if i % 10 == 0:
        plt.figure(figsize=(3.5, 3.5))
        plt.plot(losses.detach())
        plt.ylabel("Training Loss")
        plt.pause(1)


# %%

fig, axs = plt.subplots(2, 1, figsize=(4, 4))

axs[0].plot(n_hidden_nodes, accuracy_train)
axs[0].plot(n_hidden_nodes, accuracy_test)
axs[0].legend(["train", "test"])

axs[1].plot(loss_lists[0].detach())
axs[1].set_title("Training loss example")

plt.tight_layout()
