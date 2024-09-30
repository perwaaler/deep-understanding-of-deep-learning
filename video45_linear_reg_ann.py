# %% 
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython import display

display.set_matplotlib_formats("svg")


# %%
# create data
def train_and_test_model(slope=1, N=150, learningRate=0.1, numepochs=30):
    # generate 30 N(0,1) distributed random values
    x = torch.randn(N, 1)
    # generate the y values
    y = slope * x + torch.randn(N, 1) / 2

    # and plot
    plt.plot(x, y, "o")
    plt.show()

    # build model
    n1 = 4
    n2 = 4

    ANNreg = nn.Sequential(
        nn.Linear(1, n1),  # input layer
        nn.ReLU(),  # activation function
        nn.Linear(n1, n2),  # input layer
        nn.ReLU(),  # activation function
        nn.Linear(n2, 1),  # input layer
    )

    # loss function
    lossfun = nn.MSELoss()

    # optimizer (the flavor of gradient descent to implement)
    optimizer = torch.optim.SGD(ANNreg.parameters(), lr=learningRate)

    # train the model
    losses = torch.zeros(numepochs)

    ## Train the model!
    for epochi in range(numepochs):

        # forward pass
        yHat = ANNreg(x)

        # compute loss
        final_loss = lossfun(yHat, y)

        # store for later visualization
        losses[epochi] = final_loss

        # backprop
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

    # manually compute losses
    # final forward pass
    predictions = ANNreg(x)

    # final loss (MSE)
    testloss = (predictions - y).pow(2).mean()

    correlation = np.corrcoef(y.T, predictions.detach().T)[0, 1]
    final_loss = testloss.item()

    plt.plot(losses.detach(), "o", markerfacecolor="w", linewidth=0.1)
    plt.plot(numepochs, testloss.detach(), "ro")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Final loss = %g" % testloss.item())
    plt.show()

    # plot the data
    plt.plot(x, y, "bo", label="Real data")
    # use detach to get only the numerical value stored in an object
    plt.plot(x, predictions.detach(), "rs", label="Predictions")
    plt.title(f"prediction-data r={np.corrcoef(y.T,predictions.detach().T)[0,1]:.2f}")
    plt.legend()
    plt.show()

    return final_loss, correlation


slopes = np.linspace(-2, 2, 30)

losses = []
correlations = []

for i, slope in enumerate(slopes):
    print(f"Iteration {i} of 30")
    final_loss, correlation = train_and_test_model(slope=slope)
    losses.append(final_loss)
    correlations.append(correlation)

# %% Plot

plt.figure()
plt.plot(slopes, correlations)
plt.xlabel("Slopes")
plt.ylabel("correlation")
plt.show()

plt.figure()
plt.plot(slopes, losses)
plt.xlabel("Slopes")
plt.ylabel("loss")
plt.show()

# Explanation: as the slope
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
