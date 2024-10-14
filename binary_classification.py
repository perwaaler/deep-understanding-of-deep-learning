# %% [markdown]
# # COURSE: A deep understanding of deep learning
# ## SECTION: ANNs
# ### LECTURE: ANN for classifying qwerties
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/dudl/?couponCode=202207

# %%
# import libraries
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from IPython import display
from utilities import orange_print
from utilities import green_print
from utilities import silent_print

display.set_matplotlib_formats("svg")

# %%
# create data

nPerClust = 100
# Standard Deviation of random noise added
blur = 1

# Coordinates of the two clusters
clusters = {"coords": [[1, 1], [5, 1], [3, 4]], "colors": ["b", "g", "m"]}

n_clusters = len(clusters["coords"])
silent_print(f"n clusters: {n_clusters}")

# generate data
data_list = []
for coordinates in clusters["coords"]:
    x_cluster_i = coordinates[0] + np.random.randn(nPerClust) * blur
    y_cluster_i = coordinates[1] + np.random.randn(nPerClust) * blur
    data_array_cluster_i = np.array([x_cluster_i, y_cluster_i]).T
    data_list.append(data_array_cluster_i)

# Stack data horizontally
X_ready = np.vstack(data_list)
silent_print(f"shape of input array: {X_ready.shape}")

y_ready = []
for i, input_array in enumerate(data_list):
    n_i = len(input_array)
    y_ready.append(np.ones(n_i) * i)
y_ready = np.concatenate(y_ready)
silent_print(f"shape of label array: {y_ready.shape}")


# convert to a pytorch tensor
X_ready = torch.tensor(X_ready).float()
y_ready = torch.tensor(y_ready, dtype=torch.long)

# show the data
fig = plt.figure(figsize=(5, 5))
for i in range(n_clusters):
    cluster_idxs = np.where(y_ready == i)[0]
    plt.plot(
        X_ready[cluster_idxs, 0], X_ready[cluster_idxs, 1], f"{clusters['colors'][i]}o"
    )

plt.title("The qwerties!")
plt.xlabel("qwerty dimension 1")
plt.ylabel("qwerty dimension 2")
plt.show()

# %% BUILD THE MODEL
ANNclassify = nn.Sequential(
    nn.Linear(2, 3),  # input layer
    nn.ReLU(),  # activation unit
    nn.Linear(3, 3),  # output unit
)

# other model features
learningRate = 0.1
# loss function - binary cross entropy
lossfun = nn.CrossEntropyLoss()
# Note: You'll learn in the "Metaparameters" section that it's better to use
# BCEWithLogitsLoss, but this is OK for now.
optimizer = torch.optim.SGD(ANNclassify.parameters(), lr=learningRate)

# %%

# train the model
numepochs = 1000
losses = torch.zeros(numepochs)

for epochi in range(numepochs):

    # forward pass
    yHat = ANNclassify(X_ready)

    # compute loss
    loss = lossfun(yHat, y_ready)
    losses[epochi] = loss

    predClasses = torch.argmax(yHat, axis=1)
    accuracy = (predClasses == y_ready).float().mean().item()

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if accuracy > 0.9:
        green_print("Accuracy exceeds 90 - stopping training")
        break

green_print(f"Finnished training with accuracy {accuracy:.3}")

# %%
# show the losses

plt.plot(losses.detach(), "o", markerfacecolor="w", linewidth=0.1)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# %%
# compute the predictions

# manually compute losses
# final forward pass
predictions = ANNclassify(X_ready)

predlabels = torch.argmax(predictions, axis=1)

# find errors
misclassified = np.where(predlabels != y_ready)[0]

# total accuracy
totalacc = 100 - 100 * len(misclassified) / len(X_ready)

green_print("Final accuracy: %g%%" % totalacc)


# %%
# plot the labeled data
fig = plt.figure(figsize=(5, 5))

# Plot missclassified
plt.plot(
    X_ready[misclassified, 0],
    X_ready[misclassified, 1],
    "rx",
    markersize=12,
    markeredgewidth=3,
)

idx_class_predictions = [np.where(predlabels == i) for i in range(n_clusters)]

for i, index_ci in enumerate(idx_class_predictions):
    idx_class_i_predicted = np.where(predlabels == i)
    plt.plot(
        X_ready[idx_class_i_predicted, 0],
        X_ready[idx_class_i_predicted, 1],
        f"{clusters['colors'][i]}o",
    )

# plt.plot(X_ready[np.where(predlabels)[0], 0], X_ready[np.where(predlabels)[0], 1], "ko")

plt.legend(["Misclassified", "class 1", "class 2", "class 3"], bbox_to_anchor=(1, 1))
plt.title(f"{totalacc:.3}% correct")
plt.show()


# %% [markdown]
# # Additional explorations

# %%
# 1) It is common in DL to train the model for a specified number of epochs. But
#    you can also train until the model reaches a certain accuracy criterion.
#    Re-write the code so that the model continues training until it reaches 90%
#    accuracy. What would happen if the model falls into a local minimum and
#    never reaches 90% accuracy? Yikes! You can force-quit a process in
#    google-colab by clicking on the top-left 'play' button of a code cell.
#
# 2) It is intuitive that the model can reach 100% accuracy if the qwerties are
#    more separable. Modify the qwerty-generating code to get the model to have
#    100% classification accuracy.
#
