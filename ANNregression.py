# %%
# import libraries
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython import display

display.set_matplotlib_formats("svg")

# %%
# create data

N = 500
# generate 30 N(0,1) distributed random values
x = torch.randn(N, 1)
# generate the y values
y = x + torch.randn(N, 1) / 2

# and plot
# plt.plot(x, y, "o")
# plt.show()

# %%
# build model

ANNreg = nn.Sequential(
    nn.Linear(1, 6),  # input layer
    nn.ReLU(),  # activation function
    nn.Linear(6, 50),  # input layer
    nn.ReLU(),  # activation function
    nn.Linear(50, 30),  # input layer
    nn.ReLU(),  # input layer
    nn.Linear(30, 1),  # input layer
)

# ANNreg = nn.Sequential(nn.Linear(1, 20), nn.Sigmoid(), nn.Linear(20, 1))


# %%
# learning rate
learningRate = 0.15

# loss function
lossfun = nn.MSELoss()

# optimizer (the flavor of gradient descent to implement)
optimizer = torch.optim.SGD(ANNreg.parameters(), lr=learningRate)

# %%
# train the model
numepochs = 50
losses = torch.zeros(numepochs)


## Train the model!
for epochi in range(numepochs):

    # forward pass
    yHat = ANNreg(x)

    # compute loss
    loss = lossfun(yHat, y)

    # store for later visualization
    losses[epochi] = loss

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# %%
# show the losses

# manually compute losses
# final forward pass
predictions = ANNreg(x)

# final loss (MSE)
testloss = (predictions - y).pow(2).mean()

plt.plot(losses.detach(), "o", markerfacecolor="w", linewidth=0.1)
plt.plot(numepochs, testloss.detach(), "ro")
# plt.ylim([0,0.3])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Final loss = %g" % testloss.item())
plt.show()

# %%
testloss.item()

# %%
# plot the data
plt.plot(x, y, "bo", label="Real data")
# use detach to get only the numerical value stored in an object
plt.plot(x, predictions.detach(), "rs", label="Predictions")
plt.title(f"prediction-data r={np.corrcoef(y.T,predictions.detach().T)[0,1]:.2f}")
plt.legend()
plt.show()
print("?")

# %%
# 1) How much data is "enough"? Try different values of N and see how low the loss gets.
#    Do you still get low loss ("low" is subjective, but let's say loss<.25) with N=10? N=5?
#
# 2) Does your conclusion above depend on the amount of noise in the data? Try changing the noise level
#    by changing the division ("/2") when creating y as x+randn.
#
# 3) Notice that the model doesn't always work well. Put the original code (that is, N=30 and /2 noise)
#    into a function or a for-loop and repeat the training 100 times (each time using a fresh model instance).
#    Then count the number of times the model had a loss>.25.
