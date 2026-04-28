import numpy as np
import sympy as sym
import cvxpy
import scipy
import math
# constants
### speeds for each turtlebots in m/s
v_s = 2/3 ## slower turtlebot velocity # REAL SPEED IS 0.1m/s
v_f = 1 ## faster turtlebot velocity # REAL SPEED IS 0.15m/s

def simulate_merge(x, case, dt=0.01):
    match case:
        ## both velocities set to slow
        case 0:
            v_behind, v_merge = v_s, v_s
        ## car in lane set to slow, car merging set to fast
        case 1:
            v_behind, v_merge = v_s, v_f
        ## car in lane set to fast, car merging set to slow
        case 2:
            v_behind, v_merge = v_f, v_s
        ## car in both lanes set to fast
        case _:
            v_behind, v_merge = v_f, v_f
   # calculate the x coordinate based on the distance between the
   # two turtlebots
    x0 = math.sqrt(x ** 2 - 1) # this will not work if dist < 1 which should be impossible
    # initial positions
    x_f, y_f = 0, 0
    x_m, y_m = x0, 1
    # 45deg velocity components
    vx_merge = v_merge * math.cos(math.pi / 4)
    vy_merge = -v_merge * math.sin(math.pi / 4)
    t = 0
    while y_m > 0:
        x_f += v_behind * dt
        x_m += vx_merge * dt
        y_m += vy_merge * dt
        t += dt

    # final distance
    dx = x_m - x_f
    dy = y_m - y_f
    dist = math.sqrt(dx ** 2 + dy ** 2)
    return dx


if __name__ == "__main__":
    results_file = "simulation_results.txt"

    with open(results_file, "w") as f:
        f.write("# (x, case0_dist, case1_dist, case2_dist, case3_dist)\n")

        # 1000 steps between 1.0 and 2.0
        for i in range(1000, 2001):
            test_x = i / 1000.0
            d0 = simulate_merge(test_x, 0)
            d1 = simulate_merge(test_x, 1)
            d2 = simulate_merge(test_x, 2)
            d3 = simulate_merge(test_x, 3)

            # Round x to 3 places and distances to 3
            res_tuple = (
                round(test_x, 5),
                round(d0, 5),
                round(d1, 5),
                round(d2, 5),
                round(d3, 5)
            )

            # Writing as a clean tuple per line
            f.write(f"{res_tuple}\n")