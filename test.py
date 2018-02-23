import numpy as np

A = np.random.rand(10,10)
B = A.sum(axis=0)
print(B)
A /= B
print(A)
print(A.sum(axis=0), A.sum(axis=1))