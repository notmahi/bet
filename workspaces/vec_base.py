import logging
from collections import deque
from pathlib import Path
import os

import einops
import gym
import hydra
import numpy as np
import torch
import utils
import wandb
from omegaconf import OmegaConf

from workspaces.base import Workspace
from models.action_ae.generators.base import GeneratorDataParallel
from models.latent_generators.latent_generator import LatentGeneratorDataParallel

import envs


class VecWorkspace(Workspace):
    def __init__(self, cfg):
        if (
            cfg.experiment.plot_interactions
            or cfg.experiment.start_from_seen
            or cfg.experiment.record_video
            or cfg.experiment.enable_render
            or cfg.experiment.action_batch_size > 1
            or cfg.experiment.action_update_every > 1
        ):
            raise NotImplementedError(
                "This feature is not supported on the vectorized workspace."
            )
        super().__init__(cfg)

    def init_env(self):
        self.envs = gym.vector.make(
            self.cfg.env.gym_name,
            num_envs=self.cfg.experiment.num_envs,
            wrappers=[gym.wrappers.FlattenObservation]
            if self.cfg.experiment.flatten_obs
            else None,
            asynchronous=self.cfg.experiment.async_envs,
        )

    def run(self):
        all_returns = []
        for i in range(self.cfg.experiment.num_eval_eps):
            logging.info(f"==== Starting episode {i} ====")
            obs_traj, actions_traj, returns, done_at = self.run_single_episode()
            logs = {
                "episode": i,
                "mean_return": np.mean(returns),
                "std_return": np.std(returns),
                "avg_episode_length": np.mean(done_at),
                "std_episode_length": np.std(done_at),
            }
            wandb.log(logs)
            logging.info(logs)
            all_returns.append(returns)
        return all_returns, None

    def _prepare_obs(self, obs):
        obs = torch.from_numpy(obs).float().to(self.device)
        enc_obs = self.obs_encoding_net(obs)
        return enc_obs

    def run_single_episode(self):
        self.history = deque(maxlen=self.window_size)
        obs = self.envs.reset()
        done_at = np.zeros(self.cfg.experiment.num_envs)
        returns = np.zeros(self.cfg.experiment.num_envs)
        obs_traj = [obs]
        actions_traj = []
        for i in range(1, 1 + self.cfg.experiment.num_eval_steps):
            actions, latents = self._get_action(obs, sample=False, keep_last_bins=False)
            actions_traj.append(actions)
            obs, rewards, dones, infos = self.envs.step(actions)
            obs_traj.append(obs)
            # Returns accumulate reward until and evs is done.
            returns += rewards * (done_at == 0)
            # Mark the first step at which the env is done.
            done_at += i * dones * (done_at == 0)
        return obs_traj, actions_traj, returns, done_at
