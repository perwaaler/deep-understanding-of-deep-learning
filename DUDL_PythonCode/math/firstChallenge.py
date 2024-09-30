

from math import *
import numpy as np
import matplotlib.pyplot as plt

from IPython import display

display.set_matplotlib_formats("svg")

# function (as a function)
def fx(x):
    return np.cos(2 * pi * x) + x**2


# derivative function
def deriv(x):
    return -np.sin(2 * pi) * 2 * pi + 2 * x


# define a range for x
x = np.linspace(-2, 2, 2001)

# plotting
plt.plot(x, fx(x), x, deriv(x))
plt.xlim(x[[0, -1]])
plt.grid()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend(["y", "dy"])
plt.show()


# random starting point
# localmin = np.random.choice(x,1) # the old initial guess
localmin = 0
print(localmin)

# learning parameters
training_epochs = 100
learning_rate = 0.01

trainingData = np.zeros([training_epochs, 2])
# run through training
for i in range(training_epochs):
    grad = deriv(localmin)
    localmin = localmin - learning_rate * grad
    print(localmin)
    trainingData[i, :] = localmin, grad


plt.plot(trainingData[:, 0])
plt.ylim(np.array([-1, 1]))
plt.xlabel("iterations")
plt.show()

print(learning_rate)
np.abs(1)




from math import *
import numpy as np
import matplotlib.pyplot as plt

from IPython import display

display.set_matplotlib_formats("svg")

# function (as a function)
def fx(x):
    return np.cos(2 * pi * x) + x**2


# derivative function
def deriv(x):
    return -np.sin(2 * pi) * 2 * pi + 2 * x


# define a range for x
x = np.linspace(-2, 2, 2001)

# plotting
plt.plot(x, fx(x), x, deriv(x))
plt.xlim(x[[0, -1]])
plt.grid()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend(["y", "dy"])
plt.show()


# random starting point
# localmin = np.random.choice(x,1) # the old initial guess
localmin = 0
print(localmin)

# learning parameters
training_epochs = 100
learning_rate = 0.01

trainingData = np.zeros([training_epochs, 2])
# run through training
for i in range(training_epochs):
    grad = deriv(localmin)
    localmin = localmin - learning_rate * grad
    print(localmin)
    trainingData[i, :] = localmin, grad


plt.plot(trainingData[:, 0])
plt.ylim(np.array([-1, 1]))
plt.xlabel("iterations")
plt.show()

print(learning_rate)
np.abs(1)



from math import *
import numpy as np
import matplotlib.pyplot as plt

from IPython import display

display.set_matplotlib_formats("svg")

# function (as a function)
def fx(x):
    return np.cos(2 * pi * x) + x**2


# derivative function
def deriv(x):
    return -np.sin(2 * pi) * 2 * pi + 2 * x

# define a range for x
x = np.linspace(-2, 2, 2001)

# plotting
plt.plot(x, fx(x), x, deriv(x))
plt.xlim(x[[0, -1]])
plt.grid()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend(["y", "dy"])
plt.show()


# random starting point
# localmin = np.random.choice(x,1) # the old initial guess
localmin = 0
print(localmin)

# learning parameters
training_epochs = 100
learning_rate = 0.01

trainingData = np.zeros([training_epochs, 2])
# run through training
for i in range(training_epochs):
    grad = deriv(localmin)
    localmin = localmin - learning_rate * grad
    print(localmin)
    trainingData[i, :] = localmin, grad


plt.plot(trainingData[:, 0])
plt.ylim(np.array([-1, 1]))
plt.xlabel("iterations")
plt.show()

print(learning_rate)
np.abs(1)





