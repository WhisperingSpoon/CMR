import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from ur_simulation.classic_control.robots.ur7e import UR7e
import time  



def create_scene():
    init_joint_angles = np.array([1.57, -1.7, 2.4, -1.57, -1.57, -1.57])
    robot = UR7e(
        block_gripper=True,
        neutral_joints=init_joint_angles,
    )
    robot.sim.create_plane(0)
    return robot


def inverse_dynamics(q, dq, q_des, dynamics, kp, kd):
    M = dynamics.mass_matrix
    C = dynamics.coriolis_vector
    g = dynamics.gravity_vector

    # Calculate the expected acceleration
    y = kp * (q_des - q) + kd * (-dq)

    # Convert desired acceleration into joint torques
    tau = (M @ y) + C + g
    return tau


def stiffness_control(robot, x_des, Kp, Kd, kq):
    """PD + gravity compensation in task space:
       tau = g(q) + J^T Kp (x_d - x) - J^T Kd J dq  (+ small joint damping)
    """
    dq = robot.robot_state.get_joint_velocities()
    x = robot.robot_state.get_end_effector_position()

    J = robot.robot_model.get_jacobian()[:3, :]        # linear part, 3x6
    g = robot.robot_model.get_dynamics().gravity_vector

    e_x = x_des - x          # task-space position error
    dx = J @ dq              # end-effector velocity

    tau = g + J.T @ (Kp @ e_x) - J.T @ (Kd @ dx)# - kq * dq
    return tau


def run(controller, target):
    robot = create_scene()

    desired_position = np.array(target)
    desired_orientation = robot.robot_state.get_end_effector_orientation()

    # Joint-space target (used by inverse dynamics, and for comparison plots)
    q_des = np.array(
        robot.robot_model.get_inverse_kinematics(
            position=desired_position,
            quaternion=desired_orientation,
        )
    )

    # Inverse dynamics gains (joint space)
    kp = np.array([10, 10, 10, 10, 10, 10])
    kd = np.array([10, 10, 10, 10, 10, 10])

    # Stiffness control gains (task space)
    Kp_x = np.diag([50.0, 50.0, 50.0])   # N/m  -> the arm's stiffness
    Kd_x = np.diag([10.0, 10.0, 10.0])      # N*s/m
    kq = 1.0                                # joint damping for the free wrist DOFs

    duration = 5.0
    times, joint_angles, ee_positions = [], [], []
    joint_velocities, commanded_torques, applied_torques = [], [], []
    step = 0
    try:
        while step * robot.sim.dt < duration:
            q = robot.robot_state.get_joint_angles()
            dq = robot.robot_state.get_joint_velocities()

            if controller == "stiffness":
                tau = stiffness_control(robot, desired_position, Kp_x, Kd_x, kq)
            else:
                dynamics = robot.robot_model.get_dynamics()
                tau = inverse_dynamics(q, dq, q_des, dynamics, kp, kd)

            times.append(step * robot.sim.dt)
            joint_angles.append(q)
            joint_velocities.append(dq)
            ee_positions.append(robot.robot_state.get_end_effector_position())
            commanded_torques.append(tau)
            applied_torques.append(np.clip(tau, -robot.arm_joint_forces, robot.arm_joint_forces))

            robot.control_torques(tau)
            robot.sim.step()
            time.sleep(robot.sim.dt)
            step += 1
    finally:
        actual_position = robot.robot_state.get_end_effector_position()
        print("Desired position:", desired_position)
        print("Actual position:", actual_position)
        print("Position error [m]:", np.linalg.norm(desired_position - actual_position))
        if controller == "stiffness":
            # Spring force the arm exerts at equilibrium (slides: F ≈ Kp (x_d - x_e))
            print("Estimated contact force [N]:", Kp_x @ (desired_position - actual_position))
        else:
            print("Joint error [rad]:", q_des - robot.robot_state.get_joint_angles())
        input("Press Enter to close the simulation...")
        robot.sim.close()

    times = np.array(times)
    joint_angles = np.array(joint_angles)
    ee_positions = np.array(ee_positions)
    desired_angles = np.tile(q_des, (len(times), 1))

    if controller == "stiffness":
        # Task-space plot: x, y, z of the end-effector vs. target
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i, (ax, name) in enumerate(zip(axes, ["x", "y", "z"])):
            ax.plot(times, np.full_like(times, desired_position[i]), color="#66c2a5", label=f"{name} (desired)")
            ax.plot(times, ee_positions[:, i], color="coral", label=f"{name} (actual)")
            ax.set_ylabel(f"{name} [m]")
            ax.legend()
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("t [s]")
    else:
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
        for i, ax in enumerate(axes.flat):
            ax.plot(times, desired_angles[:, i], color="#66c2a5", linewidth=1, label=f"q_{i} (desired)")
            ax.plot(times, joint_angles[:, i], color="coral", linewidth=1, label=f"joint_angles_{i} (actual)")
            ax.legend()
            ax.set_xlabel("t")
            ax.set_ylabel(f"q_{i}, joint_angles_{i}")
            ax.grid(True, alpha=0.3)
    fig.tight_layout()

    results = Path(__file__).resolve().parents[1] / "ground normal kp normal kd"
    results.mkdir(exist_ok=True)
    fig.savefig(results / f"{controller}.png", dpi=150)
    np.savez_compressed(results / f"{controller}.npz",
                        time=times, q=joint_angles, q_des=desired_angles,
                        ee=ee_positions, x_des=desired_position,
                        dq=np.array(joint_velocities),
                        tau_command=np.array(commanded_torques),
                        tau_applied=np.array(applied_torques))
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["inverse_dynamics", "stiffness"],
                        default="inverse_dynamics")
    parser.add_argument("--target", nargs=3, type=float,
                        default=[0.5, 0.25, 0.5], help="Target end-effector position in world coordinates.")
    args = parser.parse_args()
    run(args.controller, args.target)
