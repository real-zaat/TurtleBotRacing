from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import numpy as np

A = np.array([[2, 30], [0, 8]], dtype=float)
B = np.array([[2, 0], [30, 8]], dtype=float)

def inner_objective(x, a, b):
    H = np.array([
    [0, 0, a[0,0]+b[0,0], a[0,1]+b[0,1], 0, 0],
    [0, 0, a[1,0]+b[1,0], a[1,1]+b[1,1], 0, 0],
    [a.T[0,0]+b.T[0,0], a.T[0,1]+b.T[0,1], 0, 0, 0, 0],
    [a.T[1,0]+b.T[1,0], a.T[1,1]+b.T[1,1], 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
    ])
    c = np.array([0,0,0,0, -1, -1])

    return (1/2) * (x.T @ H @ x) + (c@x)

objective = lambda x : inner_objective(x, A, B)

def inner_jac(x, a, b):
    H = np.array([
    [0, 0, a[0,0]+b[0,0], a[0,1]+b[0,1], 0, 0],
    [0, 0, a[1,0]+b[1,0], a[1,1]+b[1,1], 0, 0],
    [a.T[0,0]+b.T[0,0], a.T[0,1]+b.T[0,1], 0, 0, 0, 0],
    [a.T[1,0]+b.T[1,0], a.T[1,1]+b.T[1,1], 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
    ])

    return (1/2) * ((H + H.T) @ x) + np.array([0,0,0,0, -1, -1])

import time

t1 = time.time()

with open('simulation_results.txt', 'r') as infile, open('solved_bimatrix_pure.txt', 'w') as outfile:
    for line in infile:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        data = eval(line)
        x_val = data[0]

        distances = np.array([
            [data[1], data[2]],
            [data[3], data[4]]
        ])

        # Safety penalties
        ### Set to scale exponentially so we can maybe observe some cooler results
        ### This is our more aggressive player, where the penality takes longer to kick in
        A -= 5.0 / (np.abs(distances) + 0.1)
        ### This is our safer player, where the penality kicks in earlier
        B -= 8.0 / (np.abs(distances) + 0.1)

        if A.min() <= 0: A += abs(A.min()) + 1
        if B.min() <= 0: B += abs(B.min()) + 1

        objective = lambda x: inner_objective(x, A, B)
        jac = lambda x: inner_jac(x, A, B)

        r1 = np.hstack((np.zeros((2, 2)), -1 * A, np.ones((2, 1)), np.zeros((2, 1))))
        r2 = np.hstack((-1 * B.T, np.zeros((2, 3)), np.ones((2, 1))))

        A_ineq = np.vstack((r1, r2))
        b_ineq = np.zeros(4)
        A_eq = np.array([
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0]
        ])
        b_eq = np.ones(2)

        ineq_constraint = LinearConstraint(A_ineq, ub=b_ineq)
        eq_constraint = LinearConstraint(A_eq, lb=b_eq, ub=b_eq)
        zero_one_constraint = LinearConstraint(np.eye(6), lb=np.array([0, 0, 0, 0, -np.inf, -np.inf]),
                                               ub=np.array([1, 1, 1, 1, np.inf, np.inf]))

        out = 1
        counter = 0
        res = None
        ### utilizing a loop to hopefully ensure that we don't get stuck on a local min
        ### not super efficient but it will work, hopefully
        while np.abs(out) > 1e-3 and counter < 500:
            x0 = np.random.random(6)
            res = minimize(fun=objective, x0=x0, method='COBYQA',
                           constraints=[ineq_constraint, eq_constraint, zero_one_constraint])
            out = objective(res.x)
            counter += 1

        # Write format: (x, obj_val, p1_prob_a0, p1_prob_a1, p2_prob_a0, p2_prob_a1)
        if res is not None:
            outfile.write(
                f"({x_val}, {round(res.fun, 3)}, {round(res.x[0], 3)}, {round(res.x[1], 3)}, {round(res.x[2], 3)}, {round(res.x[3], 3)})\n")
print(f' Time taken to solve 1000 simulations: {time.time() - t1}')