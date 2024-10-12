# %%

print("HI")
# %%

from tracemalloc import start
# import everything in the math library:
# Spare some change?

# SERIOUS FUDGE MAKEING!
from math import *
from matplotlib import *
import matplotlib as plt
from matplotlib.pyplot import plot
from numpy import *
import numpy as np
# import matplotlib.pyplot as plt
from pandas import DataFrame
from torch import rand

x = array([[1, 2, 3], [0, 0.5, 1.0], [-1, 10, 0.2]])
np.array(x)

# note that the columns of x are plotted columnwise
plot([1, 2, 3], x)
print(x.transpose)
transpose(x)

# %% ¤¤¤ Working with arrays ¤¤¤


# the range function returns integers in a vector:
range(1, 2+1)
x = np.linspace(start=1,stop=3,num=30)
# plot it:
plt.plot(x, x**2)

# same as
y = np.linspace(1, 2, 10)

# elementwise comparison:
np.mean(x==y)
# compare the whole arrays:
np.array_equal(x,y)

# what if they are of different lengths?
z = np.append(x,1)
np.array_equal(x,z)
# it just says they are not equal. Success!

# I can use the functions directly if I import everything from the library
z = exp(0.3*x**2)
plt.plot(x, exp(0.3*x**2))
plot(x,x)
# where do the functions intersect?
mean(1)
sum(2)
str(1)

# %% ¤¤¤ New Section Though ¤¤¤
