"""Train the model-free peg-in-hole policy with PPO (experiment E2, model-free branch).

Method: PPO (Schulman et al. 2017) for continuous-action robotic manipulation, as discussed in
Lecture 3 (Shahid et al., Autonomous Robots 2022). Implementation: Stable-Baselines3
(Raffin et al., JMLR 2021) -- no algorithmic changes, only the environment is ours.

One pybullet instance per process (SubprocVecEnv); --n-envs defaults to $SLURM_CPUS_PER_TASK.
Checkpoints every --ckpt-every steps; --resume continues from the latest checkpoint.

Example
-------
  python experiments/train_ppo.py --timesteps 3000000 --n-envs 16 --seed 0
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, save_json  # noqa: E402


def make_env(rank, seed, max_offset):
    def _init():
        from stable_baselines3.common.monitor import Monitor

        from cmr_control.learning.peg_env import PegInHoleEnv

        env = PegInHoleEnv(max_offset=max_offset)
        env = Monitor(env, info_keywords=("is_success",))
        env.reset(seed=seed + 1000 * rank)
        return env

    return _init


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-offset", type=float, default=0.005, help="training hole-position error radius [m]")
    ap.add_argument("--n-steps", type=int, default=512, help="rollout length per env")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--ckpt-every", type=int, default=250_000)
    ap.add_argument("--eval-every", type=int, default=100_000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    torch.set_num_threads(1)
    out = args.out or RESULTS / "ppo" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.json", vars(args) | {"out": str(out)})

    env = SubprocVecEnv([make_env(i, args.seed, args.max_offset) for i in range(args.n_envs)], start_method="fork") \
        if args.n_envs > 1 else DummyVecEnv([make_env(0, args.seed, args.max_offset)])
    env = VecMonitor(env)
    # evaluation in a separate process (pybullet uses one physics client per process)
    eval_env = SubprocVecEnv([make_env(99, args.seed + 12345, args.max_offset)], start_method="fork")

    ckpts = sorted(glob.glob(str(out / "checkpoints" / "ppo_*_steps.zip")),
                   key=lambda p: int(Path(p).stem.split("_")[-2]))
    if args.resume and ckpts:
        print(f"resuming from {ckpts[-1]}")
        model = PPO.load(ckpts[-1], env=env, device="cpu")
        reset = False
    else:
        model = PPO(
            "MlpPolicy", env, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=10,
            learning_rate=args.lr, gamma=args.gamma, gae_lambda=0.95, clip_range=0.2,
            ent_coef=args.ent_coef, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-0.5),
            seed=args.seed, device="cpu", verbose=1,
        )
        reset = True
    model.set_logger(configure(str(out / "logs"), ["stdout", "csv"]))

    callbacks = [
        CheckpointCallback(save_freq=max(args.ckpt_every // args.n_envs, 1), save_path=str(out / "checkpoints"),
                           name_prefix="ppo"),
        EvalCallback(eval_env, n_eval_episodes=20, eval_freq=max(args.eval_every // args.n_envs, 1),
                     best_model_save_path=str(out), log_path=str(out / "eval"), deterministic=True),
    ]
    model.learn(total_timesteps=args.timesteps, callback=callbacks, reset_num_timesteps=reset, progress_bar=False)
    model.save(out / "final_model")
    env.close()
    eval_env.close()

    ev = np.load(out / "eval" / "evaluations.npz")
    if "successes" in ev:
        sr = ev["successes"].mean(1)
        print("eval success rate per evaluation:", np.round(sr, 2))


if __name__ == "__main__":
    main()
