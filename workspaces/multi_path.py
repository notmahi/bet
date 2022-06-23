import matplotlib.pyplot as plt
import numpy as np

import envs
from workspaces import base


class MultiPathWorkspace(base.Workspace):
    def _setup_plots(self):
        pass

    def _setup_starting_state(self):
        pass

    def _start_from_known(self):
        randomness = np.random.random()
        random_distnce = np.random.uniform(0, 5)
        if randomness <= 0.3:
            state = np.array([0, random_distnce]) + np.random.normal(0, 0.25)
        elif randomness >= 0.7:
            state = np.array([random_distnce, 0]) + np.random.normal(0, 0.25)
        else:
            state = np.random.normal(0, 0.25, size=(2,))
        return self.env.set_state(state)

    def _plot_obs_and_actions(self, obs, chosen_action, done, all_actions=None):
        if not hasattr(self, "obs_buffer"):
            self.obs_buffer = []
        if done:
            # Plot the whole thing
            obses = np.array(self.obs_buffer)
            plt.plot(obses[:, 0], obses[:, 1], "bo-", alpha=0.25)
            max_value = obses.max()
            plt.xlim([-2, max_value + 1])
            plt.ylim([-2, max_value + 1])
            plt.show(block=False)
            plt.pause(1)
            plt.close()
            self.obs_buffer = []
        else:
            self.obs_buffer.append(obs)
