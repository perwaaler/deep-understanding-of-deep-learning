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

display.set_matplotlib_formats("svg")

# %% [markdown]
# # Import and process the data

# %%
# import dataset (comes with seaborn)
import seaborn as sns

iris = sns.load_dataset("iris")

# check out the first few lines of data
iris.head()

# %%
# some plots to show the data
import seaborn as sns
sns.pairplot(iris, hue="species")
plt.show()

# %%
# organize the data

# convert from pandas dataframe to tensor
x_train = torch.tensor(iris[iris.columns[0:4]].values).float()

# transform species to number
labels_train = torch.zeros(len(x_train), dtype=torch.long)
# labels[iris.species=='setosa'] = 0 # don't need!
labels_train[iris.species == "versicolor"] = 1
labels_train[iris.species == "virginica"] = 2

labels_train

# %% [markdown]
# # Create the ANN model

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

# %% [markdown]
# # Train the model

# %%
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

# %%
torch.argmax(yHat, axis=1)

# %% [markdown]
# # Visualize the results

# %%
# report accuracy
print("Final accuracy: %g%%" % totalacc)

fig, ax = plt.subplots(1, 2, figsize=(13, 4))

ax[0].plot(losses.detach())
ax[0].set_ylabel("Loss")
ax[0].set_xlabel("epoch")
ax[0].set_title("Losses")

ax[1].plot(ongoingAcc)
ax[1].set_ylabel("accuracy")
ax[1].set_xlabel("epoch")
ax[1].set_title("Accuracy")
plt.show()
# run training again to see whether this performance is consistent

# %%
# confirm that all model predictions sum to 1, but only when converted to softmax
sm = nn.Softmax(1)
torch.sum(sm(yHat), axis=1)

# %%
# plot the raw model outputs

fig = plt.figure(figsize=(10, 4))

plt.plot(yHat.detach(), "s-", markerfacecolor="w")
plt.xlabel("Stimulus number")
plt.ylabel("Probability")
plt.legend(["setosa", "versicolor", "virginica"])
plt.show()

# try it again without the softmax!

# %%


# %% [markdown]
# # Additional explorations

# %%
# 1) When the loss does not reach an asymptote, it's a good idea to train the model for more epochs. Increase the number of
#    epochs until the plot of the losses seems to hit a "floor" (that's a statistical term for being as small as possible).
#
# 2) We used a model with 64 hidden units. Modify the code to have 16 hidden units. How does this model perform? If there
#    is a decrease in accuracy, is that decrease distributed across all three iris types, or does the model learn some
#    iris types and not others?
#
# 3) Write code to compute three accuracy scores, one for each iris type. In real DL projects, category-specific accuracies
#    are often more informative than the aggregated accuracy.
#
