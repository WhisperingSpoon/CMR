"""Controllers and experiments for the simulated UR7e (Control Methods for Robotics, WMAI037-05).

Builds on the course-provided ``ur_simulation`` package (PyBullet wrapper + UR7e URDF).

Sub-modules
-----------
urdf_tools      URDF variants: bare arm, arm + payload, arm + peg (all attached to tool0)
sim             Headless plant wrapper around ``ur_simulation``'s UR7e class
dynamics        Rigid-body model in a separate PyBullet client (nominal or mis-specified)
trajectories    Quintic multi-waypoint and multi-sine joint trajectories
scene           Chamfered peg-in-hole fixture
controllers/    PD (baseline), computed torque, Cartesian impedance, Lagrangian-NN controller
learning/       Lagrangian neural network (Liu et al. 2024), PPO peg-in-hole environment
metrics         Tracking / contact metrics
"""
