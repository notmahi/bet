import einops
import gym
import hydra
import joblib
import torch
import wandb

import utils
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from workspaces import base
import envs


class BlockPushWorkspace(base.Workspace):
    def _setup_plots(self):
        pass

    def _setup_starting_state(self):
        pass

    def _start_from_known(self):
        pass

    def _plot_obs_and_actions(self, obs, chosen_action, done, all_actions=None):
        pass

    def _report_result_upon_completion(self):
        pass
