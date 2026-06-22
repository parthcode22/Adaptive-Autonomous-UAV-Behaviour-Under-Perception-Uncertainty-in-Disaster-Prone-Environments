
"""
train_ppo.py — PPO Training Entry Point
============================================
SURE-UAV RL Navigation Agent

Wires together UAVNavigationEnv (uav_env.py) and our custom feature
extractor (ppo_network.py) into Stable-Baselines3's PPO trainer.

This trains against the MOCK environment (no Gazebo/PX4 yet) — useful
for validating the full pipeline and getting an initial policy, but
results here are not representative of real-world performance until
the mock environment is replaced with real simulation.
"""

from __future__ import annotations
import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from .uav_env import UAVNavigationEnv
from .ppo_network import get_policy_kwargs


class TrainingConfig:
    """All tunable training hyperparameters, centralized."""

    # --- Environment ---
    N_ENVS: int = 4

    # --- PPO hyperparameters (Stable-Baselines3 defaults are reasonable
    #     starting points; these are explicit here so they're visible
    #     and easy to tune later) ---
    LEARNING_RATE: float = 3e-4
    N_STEPS: int = 2048
    BATCH_SIZE: int = 64
    N_EPOCHS: int = 10
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_RANGE: float = 0.2
    ENT_COEF: float = 0.01  # entropy bonus — encourages exploration

    # --- Training duration ---
    TOTAL_TIMESTEPS: int = 500_000

    # --- Logging / checkpoints ---
    LOG_DIR: str = "./agent/logs"
    CHECKPOINT_DIR: str = "./agent/checkpoints"
    CHECKPOINT_FREQ: int = 25_000
    EVAL_FREQ: int = 10_000
    EVAL_EPISODES: int = 10


def make_env():
    """Factory function for a single UAVNavigationEnv instance, wrapped
    with Monitor for episode statistics tracking (reward, length)."""

    def _init():
        env = UAVNavigationEnv()
        env = Monitor(env)
        return env

    return _init


def train(config: TrainingConfig = TrainingConfig()):
    """
    Run PPO training against the mock UAVNavigationEnv.

    Returns
    -------
    model : the trained PPO model
    """
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    run_name = datetime.now().strftime("sure_uav_ppo_%Y%m%d_%H%M%S")

    # --- Vectorized training environment ---
    train_env = make_vec_env(make_env(), n_envs=config.N_ENVS)

    # --- Separate evaluation environment (single instance) ---
    eval_env = Monitor(UAVNavigationEnv())

    # --- Callbacks: periodic checkpointing + evaluation ---
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config.CHECKPOINT_FREQ // config.N_ENVS, 1),
        save_path=config.CHECKPOINT_DIR,
        name_prefix=run_name,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(
            config.CHECKPOINT_DIR,
            "best_model"
        ),
        log_path=config.LOG_DIR,
        eval_freq=max(config.EVAL_FREQ // config.N_ENVS, 1),
        n_eval_episodes=config.EVAL_EPISODES,
        deterministic=True,
    )

    # --- Build the PPO model with our custom feature extractor ---
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        policy_kwargs=get_policy_kwargs(),
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        ent_coef=config.ENT_COEF,
        tensorboard_log=config.LOG_DIR,
        verbose=1,
    )

    print(f"[train_ppo] Starting training run: {run_name}")
    print(f"[train_ppo] Total timesteps: {config.TOTAL_TIMESTEPS}")

    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name=run_name,
    )

    final_path = os.path.join(
        config.CHECKPOINT_DIR,
        f"{run_name}_final"
    )
    model.save(final_path)

    print(
        f"[train_ppo] Training complete. "
        f"Final model saved to: {final_path}"
    )

    return model


def evaluate_policy_detailed(model, n_episodes: int = 50):
    """
    Run n_episodes with the trained (or untrained) model and report
    success rate, crash rate, and timeout rate — the key metrics for
    judging this agent's real performance.
    """
    env = UAVNavigationEnv()

    successes = 0
    crashes = 0
    timeouts = 0
    episode_rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = env.step(
                action
            )

            ep_reward += reward
            done = terminated or truncated

        episode_rewards.append(ep_reward)

        breakdown = info["reward_breakdown"]

        if breakdown.crashed:
            crashes += 1
        elif terminated:   # terminated but not crashed -> goal reached
            successes += 1
        elif truncated:
            timeouts += 1

    success_rate = successes / n_episodes
    crash_rate = crashes / n_episodes
    timeout_rate = timeouts / n_episodes
    mean_reward = sum(episode_rewards) / len(episode_rewards)

    print(f"\n{'=' * 60}")
    print(f"Evaluation over {n_episodes} episodes:")
    print(f"  Success rate : {success_rate * 100:.1f}%")
    print(f"  Crash rate   : {crash_rate * 100:.1f}%")
    print(f"  Timeout rate : {timeout_rate * 100:.1f}%")
    print(f"  Mean reward  : {mean_reward:.2f}")
    print(f"{'=' * 60}")

    return {
        "success_rate": success_rate,
        "crash_rate": crash_rate,
        "timeout_rate": timeout_rate,
        "mean_reward": mean_reward,
    }


if __name__ == "__main__":
    config = TrainingConfig()
    model = train(config)

    print("\n[train_ppo] Running post-training evaluation...")
    evaluate_policy_detailed(model, n_episodes=50)

