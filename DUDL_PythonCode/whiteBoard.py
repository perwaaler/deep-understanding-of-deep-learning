import numpy as np
import torch

# generate random sample from array
sample_space = [1,2,5,-1.2]
r = np.random.choice(sample_space)  
print(r)

np.random.seed(17)
print(np.random.randn(5))

randseed1 = np.random.RandomState(17)
randseed2 = np.random.RandomState(20210530)

print( randseed1.randn(5) ) # same sequence
print( randseed2.randn(5) ) # different from above, but same each time
print( randseed1.randn(5) ) # same as two up
print( randseed2.randn(5) ) # same as two up
print( np.random.randn(5) ) # different every time

