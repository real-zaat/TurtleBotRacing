import cvxpy as cp
import numpy as np
import scipy
### this is a test for some basic matrices
A = np.array([[3,0],[-1,1]])

v = cp.Variable(1)
z = cp.Variable(2)

obj = cp.Maximize(v)

problem = cp.Problem(obj, constraints=[
    z >= np.zeros(2),
    cp.sum(z) == 1,
    A @ z >= v
])

problem.solve()
print(f'Col Player - value: {problem.value} policy: {z.value}')
## row player

v = cp.Variable(1)
y = cp.Variable(2)

obj = cp.Minimize(v)
problem = cp.Problem(obj, constraints=[
    y >= np.zeros(2),
    cp.sum(y) == 1,
    A.T @ y <= v
])

problem.solve()
print(f'Row Player - value: {problem.value} policy: {y.value}')