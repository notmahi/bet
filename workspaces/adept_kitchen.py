import adept_envs
import einops
import gym
import hydra
import joblib
import torch
import umap
import umap.plot
import wandb

import utils
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from workspaces import base
import envs


class AdeptKitchenWorkspace(base.Workspace):
    def _setup_plots(self):
        plt.ion()
        obs_mapper_path = (
            Path(self.cfg.env_vars.datasets.relay_kitchen) / "obs_mapper.pkl"
        )
        with (obs_mapper_path).open("rb") as f:
            obs_mapper = joblib.load(f)
        self.obs_mapper = obs_mapper
        # self.obs_ax = umap.plot.points(obs_mapper)
        self.obs_ax = plt.scatter(
            obs_mapper.embedding_[:, 0], obs_mapper.embedding_[:, 1], s=0.01, alpha=0.1
        )
        self.obs_sc = plt.scatter([0], [0], marker="X", c="orange")
        self._figure_1 = plt.gcf()

        self._figure_2 = plt.figure()
        action_mapper_path = (
            Path(self.cfg.env_vars.datasets.relay_kitchen) / "action_mapper.pkl"
        )
        with (action_mapper_path).open("rb") as f:
            action_mapper = joblib.load(f)
        self.action_mapper = action_mapper
        # self.action_ax = umap.plot.points(action_mapper)
        self.action_ax = plt.scatter(
            action_mapper.embedding_[:, 0],
            action_mapper.embedding_[:, 1],
            s=0.01,
            alpha=0.1,
        )
        self.action_sc = plt.scatter([0], [0], marker=".", c="orange", alpha=0.5)
        plt.draw()
        plt.pause(0.001)

    def _setup_starting_state(self):
        self.init_qpos = np.load(
            Path(self.cfg.env_vars.datasets.relay_kitchen) / "all_init_qpos.npy"
        )
        self.init_qvel = np.load(
            Path(self.cfg.env_vars.datasets.relay_kitchen) / "all_init_qvel.npy"
        )

    def _start_from_known(self):
        ind = np.random.randint(len(self.init_qpos))
        print(f"Starting from demo {ind}")
        qpos, qvel = self.init_qpos[ind], self.init_qvel[ind]
        self.env.set_state(qpos, qvel)
        obs, _, _, _ = self.env.step(np.zeros(self.cfg.env.action_dim))
        return obs

    def _plot_obs_and_actions(self, obs, chosen_action, done, all_actions=None):
        obs_embedding = self.obs_mapper.transform(
            einops.rearrange(obs, "(batch obs) -> batch obs", batch=1)
        )
        self.obs_sc.set_offsets(obs_embedding[:, :2])
        self.obs_sc.set_sizes([50])

        expanded_chosen_action = einops.rearrange(
            chosen_action, "(batch obs) -> batch obs", batch=1
        )
        action_embedding = self.action_mapper.transform(expanded_chosen_action)
        colors = ["orange"]
        sizes = [50]
        if all_actions is not None:
            all_action_embedding = self.action_mapper.transform(all_actions)
            action_embedding = np.concatenate([action_embedding, all_action_embedding])
            colors += ["green"] * len(all_actions)
            sizes += [10] * len(all_actions)
        else:
            all_action_embedding = action_embedding

        self.action_sc.set_offsets(all_action_embedding[:, :2])
        self.action_sc.set_color(colors)
        self.action_sc.set_sizes(sizes)

        self._figure_1.canvas.flush_events()
        self._figure_2.canvas.flush_events()
        self._figure_1.canvas.draw()
        self._figure_2.canvas.draw()

    def _report_result_upon_completion(self):
        print(
            "Complete tasks ", set(self.env.ALL_TASKS) - set(self.env.tasks_to_complete)
        )
        print("Incomplete tasks ", set(self.env.tasks_to_complete))
