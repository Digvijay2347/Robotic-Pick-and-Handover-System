"""Gymnasium Environments for Dual-Arm Manipulation (Gym-ALOHA / LeRobot Specification)."""

import gymnasium as gym
from envs.dual_ur10e_env import DualUR10eHandoverEnv

# Register Standard Visual Gym-ALOHA Environment
gym.register(
    id="DualUR10eHandover-v0",
    entry_point="envs.dual_ur10e_env:DualUR10eHandoverEnv",
    max_episode_steps=500,
    kwargs={"obs_type": "pixels"},
)

# Register Fast State-Based Environment for RL Benchmarking
gym.register(
    id="DualUR10eHandover-State-v0",
    entry_point="envs.dual_ur10e_env:DualUR10eHandoverEnv",
    max_episode_steps=500,
    kwargs={"obs_type": "state"},
)

__all__ = ["DualUR10eHandoverEnv"]
