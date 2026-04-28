import cvxpy as cp
import numpy as np

with open('simulation_results.txt', 'r') as infile, open('solved.txt', 'w') as outfile:
    for line in infile:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        data = eval(line)
        x = data[0]
        A = np.array([
            [data[1], data[2]],
            [data[3], data[4]]
        ])

        v = cp.Variable(1)
        z = cp.Variable(2)
        obj = cp.Maximize(v)
        problem = cp.Problem(obj, constraints=[
            z >= np.zeros(2),
            cp.sum(z) == 1,
            A @ z >= v
        ])
        problem.solve(verbose=False)

        val = problem.value
        z_opt = z.value

        v = cp.Variable(1)
        y = cp.Variable(2)
        obj = cp.Minimize(v)
        problem = cp.Problem(obj, constraints=[
            y >= np.zeros(2),
            cp.sum(y) == 1,
            A.T @ y <= v
        ])
        problem.solve(verbose=False)

        y_opt = y.value

        outfile.write(f"({x}, {val}, {y_opt[0]}, {y_opt[1]}, {z_opt[0]}, {z_opt[1]})\n")