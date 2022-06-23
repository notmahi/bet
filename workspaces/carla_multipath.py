import matplotlib.pyplot as plt
import numpy as np
import hydra

import envs
from workspaces import base


class CarlaMultipathWorkspace(base.Workspace):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _setup_plots(self):
        pass

    def _setup_starting_state(self):
        self.known_seeds = list(range(100))

    def _start_from_known(self):
        obs = self.env.reset(
            seed=np.random.choice(self.known_seeds),
        )
        return obs

    def _plot_obs_and_actions(self, obs, chosen_action, done, all_actions=None):
        pass


class CarlaMultipathRepWorkspace(CarlaMultipathWorkspace):
    def __init__(self, cfg):
        super().__init__(cfg)
        # NOTE: bit of a hack; override snapshot and always use pretrained resnet18 as encoder.
        # During training, we precompute all observation embeddings, and use the identity encoder.
        # During evaluation, we use an actual pretrained resnet18 to encode the input RGB observations.
        self.obs_encoding_net = hydra.utils.instantiate(cfg.encoder).to(self.device)


class CarlaMultipathStateWorkspace(CarlaMultipathWorkspace):
    def __init__(self, cfg):
        super().__init__(cfg)
