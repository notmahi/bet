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
        all_obs_trajs = []
        all_action_trajs = []
        all_done_at = []
        if self.cfg.experiment.lazy_init_models:
            self._init_action_ae()
            self._init_obs_encoding_net()
            self._init_state_prior()
        for i in range(self.cfg.experiment.num_eval_eps):
            logging.info(f"==== Starting episode {i} ====")
            obs_trajs, action_trajs, returns, done_at = self.run_single_episode()
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
            all_obs_trajs.append(obs_trajs)
            all_action_trajs.append(action_trajs)
            all_done_at.append(done_at)
        self.envs.close()
        # Concatenate vectors.
        all_returns = np.concatenate(all_returns)
        all_obs_traj = np.concatenate(all_obs_trajs)
        all_actions_traj = np.concatenate(all_action_trajs)
        done_at = np.concatenate(all_done_at)
        wandb.log({
            "overall/mean_return": np.mean(all_returns),
            "overall/std_return": np.std(all_returns),
            "overall/avg_episode_length": np.mean(done_at),
            "overall/std_episode_length": np.std(done_at),
        })
        # Save trajectories.
        np.save(
            os.path.join(self.work_dir, f"obs_trajs.npy"),
            all_obs_traj
        )
        np.save(
            os.path.join(self.work_dir, f"action_trajs.npy"),
            all_actions_traj,
        )
        np.save(
            os.path.join(self.work_dir, f"done_at.npy"),
            done_at,
        )
        return all_returns, None

    def _prepare_obs(self, obs):
        obs = torch.from_numpy(obs).float().to(self.device)
        enc_obs = self.obs_encoding_net(obs)
        return enc_obs

    def run_single_episode(self):
        self.history = deque(maxlen=self.window_size)
        obs = self.envs.reset()     # (num_envs, obs_dim)
        done_at = np.zeros(self.cfg.experiment.num_envs)
        returns = np.zeros(self.cfg.experiment.num_envs)
        obs_trajs = []
        action_trajs = []

        for i in range(1, 1 + self.cfg.experiment.num_eval_steps):
            actions, _ = self._get_action(obs, sample=False, keep_last_bins=False)
            # action.shape = (num_envs, action_dim)
            obs_trajs.append(obs)
            action_trajs.append(actions)
            next_obs, rewards, dones, infos = self.envs.step(actions)
            # Returns accumulate reward until and evs is done.
            returns += rewards * (done_at == 0)
            # Mark the first step at which the env is done.
            done_at += i * dones * (done_at == 0)
            obs = next_obs

        # Save observation and action at final step to be consistent with the non-vectorized env.
        actions, _ = self._get_action(obs, sample=False, keep_last_bins=False)
        obs_trajs.append(obs)
        action_trajs.append(actions)
        # Concatenate vectors.
        # obs_traj has shape (num_eval_steps + 1, num_envs, obs_dim)
        # Should become (num_envs, num_eval_steps + 1, obs_dim)
        obs_trajs = np.stack(obs_trajs, axis=1)
        action_trajs = np.stack(action_trajs, axis=1)

        return obs_trajs, action_trajs, returns, done_at
