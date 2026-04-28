All of the following was setup for the following:

2 TurtleBot3s running a lane merge simulation
Both turtlebots are on Ubuntu 24.04 LTS running ROS2 Kilted

Use sim.py to simulate merging distances from an intial distance d with 4 unique speed policies
(slow+slow) (slow+fast) (fast+slow) (fast+fast)

Use the bimatrix solver to create nash equilibria based on that final distance and speed policies

Use NashNet.py to create weights based on these nash equilibria to allow our turtlebots to make a decision without having to solve a whole bimatrix game during the race.

Finally, run opponent_tracker2.py with these weights (adjust the constants accordingly based on your real physical distances)
