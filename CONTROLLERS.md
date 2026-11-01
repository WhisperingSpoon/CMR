# Control methods for the simulated UR7e

Four controllers (Lectures 2–3, WMAI037-05), two baselines, and two validation experiments,
built on the course `ur_simulation` package (PyBullet, 1 kHz torque control).

| | Free motion (E1) | Interaction: peg-in-hole (E2) |
|---|---|---|
| **Model-based** | Computed torque — `controllers/computed_torque.py` | Cartesian impedance — `controllers/impedance.py` |
| **Model-free** | Lagrangian NN + tracking law (Liu et al. 2024) — `learning/lnn.py`, `controllers/lnn_controller.py` | PPO policy — `learning/peg_env.py`, `experiments/train_ppo.py` |
| Baselines | independent-joint PD (no model) | stiff impedance (≈ position control, MB); scripted descent with joint PID (MF) |

## Install

```bash
pip install -r requirements.txt
pip install -e .
```
Python 3.10–3.12 gets a prebuilt pybullet wheel; on 3.13 pybullet compiles from source (~15 min).

## Running (laptop)

```bash
# --- E1: free motion ----------------------------------------------------------
python experiments/collect_lnn_data.py --n-traj 70 --duration 20     # ~1-2 min, data/lnn/
python experiments/train_lnn.py --epochs 200                          # or on Slurm (below)
python experiments/e1_free_motion.py --lnn results/lnn/lnn_best.pt    # results/e1/

# --- E2: peg-in-hole ----------------------------------------------------------
python experiments/train_ppo.py --timesteps 3000000 --n-envs 8        # or on Slurm (below)
python experiments/e2_peg_in_hole.py --ppo-model results/ppo/seed0/best_model.zip   # results/e2/

# --- figures (PDF/PNG) + LaTeX tables -----------------------------------------
python experiments/plot_results.py all                                # results/figures, results/tables
```
Watch any single run with `--gui`, e.g.
`python experiments/e2_peg_in_hole.py --controllers impedance --offsets 3 --trials 1 --gui`.
Controllers that do not need training can be run before training finishes:
`--controllers pd ct` (E1) and `--controllers impedance impedance_stiff pid_scripted` (E2).
Output folders can be redirected with `CMR_RESULTS=...` and `CMR_DATA=...`.

## Training on Slurm

```bash
bash slurm/setup_venv.sh            # once, on a login node (edit the module line first)
# edit slurm/env.sh: CMR_ROOT, CMR_VENV, module load ...
bash slurm/submit_training.sh       # LNN data (array) -> LNN training (GPU); PPO seeds 0-2 (CPU)
```
| Job | Resources | Notes |
|---|---|---|
| `collect_lnn.sbatch` | 10 × 1 CPU | 70 trajectories × 20 s, seeds 1000–1009 |
| `train_lnn.sbatch` | 1 GPU | env vars `EPOCHS`, `WIDTH`, `MAX_TRAIN_TRAJ`, `OUT` for ablations |
| `train_ppo.sbatch` | 3 × 16 CPU | one seed per array task; resubmitting resumes from the last checkpoint |
| `e2_peg.sbatch` | 8 × 1 CPU | optional sharded evaluation |

Copy back `results/lnn/lnn_best.pt` and `results/ppo/seed*/best_model.zip` (plus `loss_curve.csv`,
`lnn_eval*.{json,npz}`, `ppo/seed*/eval/` for the learning-curve figures).

## Experiment design

**E1 — free-motion tracking.** Plant: bare UR7e. Three test trajectories with fixed seeds that
are never used for training: `pick_place` (quintic segments through 6 task-space waypoints,
1 s each), `sine_in_dist` (multi-sine 0.05–0.5 Hz, same distribution as the LNN data) and
`sine_fast` (0.5–1.0 Hz, out of distribution). Two conditions: `nominal` and `payload`
(2 kg at the tool, unknown to both the CT model and the LNN). PD and LNN share identical
PD gains, so their difference isolates the learned feed-forward. Metrics: joint RMSE,
end-effector RMSE / max error, RMS torque, saturation.

**E2 — peg-in-hole.** 20 mm square peg, chamfered (4 mm, 45°) square hole, 0.5 mm clearance
per side, 30 mm deep. The controllers know only the *nominal* hole position; the true hole is
displaced by r ∈ {0,…,6} mm in a random direction (calibration / perception error). 30 trials
per r; trial seeds fix the error direction and the start pose, so every controller faces the
same scenes. PPO is trained with r ~ U(0, 5 mm) and randomised friction, so 6 mm tests
extrapolation. Metrics: success rate (tip within 5 mm of the bottom), insertion time, peak
contact force (1 kHz, contact solver ground truth), mean contact force while in contact.
Optional model-error study: `--model-mass-scale 1.2` on the impedance controller.

**LNN model quality** (in `train_lnn.py`, evaluation only): per-joint acceleration and
inverse-dynamics torque RMSE on held-out trajectories, and the learned gravity vector and
mass matrix against the true URDF model.

## What is taken from the literature / existing software

| Component | Source | Our contribution |
|---|---|---|
| Simulator, UR7e URDF, `UR7e`/`PyBullet` classes | course package `ur_simulation` (R. Moret); PyBullet (Coumans & Bai) | headless + custom-URDF options; fixed phantom 1 kg links (below) |
| Computed torque | Siciliano et al., *Robotics: Modelling, Planning and Control*, 2009, §8.5.2; Spong et al. 2006 | gain choice, experiments |
| Quintic trajectories, DLS inverse kinematics, quaternion error | Siciliano et al. 2009 (§4.2, §3.7, Eq. 3.91) | — |
| Cartesian impedance (operational-space form) | Hogan 1985; Khatib 1987; Ott, *Cartesian Impedance Control*, 2008 | insertion strategy, gains |
| Lagrangian NN (Cholesky mass matrix, energy-based EOM) | Lutter et al., DeLaN, ICLR 2019; Lutter & Peters, IJRR 2023; Cranmer et al. 2020 | — |
| Damping network, RK4 next-state loss, tracking law (Eq. 25), sinusoidal excitation data | Liu et al., *Adv. Intell. Syst.* 2024 | PyTorch re-implementation (orig. JAX); A(q)=I; normalised loss; w_q=0 (below) |
| PPO | Schulman et al. 2017; applied to manipulation in Shahid et al., *Auton. Robots* 2022 | — |
| PPO implementation | Stable-Baselines3 (Raffin et al., JMLR 2021), Gymnasium | peg-in-hole environment, observation / reward design |
| Autodiff / NN library | PyTorch (`torch.func`) | — |
| Peg-in-hole fixture, experiment protocol, metrics, Slurm pipeline | — | ours |

## Implementation notes worth reporting

* **Phantom mass in the URDF.** PyBullet gives every link without an `<inertial>` tag 1 kg and unit
  inertia. The course URDF has three such frames on the wrist (`ft_frame`, `flange`, `tool0`), i.e. a
  3 kg phantom payload at the tool. Generated URDFs (`urdf_tools.py`) give them negligible inertia.
* **Sampled-data gain limit.** Wrist-3 joint inertia is only 4.3·10⁻⁴ kg m². With 1 kHz explicit
  torque feedback, stability requires K_d·Δt / M_66 < 2, so wrist-3 derivative gains must stay below
  ~0.86 N m s/rad; all PD/PID gains satisfy K_d·Δt/M_ii ≤ 0.5. The same limit caps the yaw stiffness
  of the stiff impedance baseline.
* **IK.** PyBullet's `calculateInverseKinematics` met the position but not the yaw about the tool
  axis (≈30° error), which jams a square peg; a 6-D damped-least-squares IK is used instead.
* **Model separation.** Model-based controllers query a separate PyBullet client (`RigidBodyModel`)
  loaded from a possibly different URDF, never the plant, which makes model-error experiments possible.
  The model-free controllers use no dynamics model at all; the PPO agent and the scripted baseline use
  kinematics (IK) only to turn Cartesian targets into joint set-points for the joint PID servo.
* **LNN loss.** PyBullet integrates with semi-implicit Euler, so logged position increments differ from
  an RK4 prediction by ≈ a·Δt²/2 — the same size as the information in the position term of Liu et al.'s
  Eq. 8 at Δt = 1 ms. Training therefore uses the velocity term only (`--w-q 0`, default).
* The UR7e is fully actuated and torque driven, so Liu et al.'s input-matrix network A(q) is fixed to I.

## Code map

```
cmr_control/
  sim.py               Plant: headless UR7e, torque saturation, F/T sensor
  dynamics.py          RigidBodyModel: M, c+g, g, J, J'dq, FK, DLS-IK (separate client)
  urdf_tools.py        URDF variants (payload / peg), phantom-mass fix, mass scaling
  trajectories.py      quintic waypoints, multi-sine excitation
  scene.py             chamfered peg-in-hole fixture, contact force
  rotations.py         quaternion error
  controllers/         pd, computed_torque, impedance, lnn_controller, peg_strategies
  learning/            lnn (Lagrangian NN), peg_env (Gymnasium env)
experiments/           collect_lnn_data, train_lnn, e1_free_motion, train_ppo, e2_peg_in_hole, plot_results
slurm/                 env.sh, setup_venv.sh, *.sbatch, submit_training.sh
```
