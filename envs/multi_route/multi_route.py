import einops
import numpy as np
import matplotlib.pyplot as plt


BOUNDS = 10
MULTI_PATH_WAYPOINTS_1 = (
    (
        (0, 0),
        (BOUNDS, 0),
        (BOUNDS, BOUNDS),
        (BOUNDS, 2 * BOUNDS),
        (2 * BOUNDS, 2 * BOUNDS),
    ),
    (
        (0, 0),
        (BOUNDS, BOUNDS),
        (2 * BOUNDS, 2 * BOUNDS),
    ),
    (
        (0, 0),
        (0, BOUNDS),
        (BOUNDS, BOUNDS),
        (2 * BOUNDS, BOUNDS),
        (2 * BOUNDS, 2 * BOUNDS),
    ),
)

PATH_PROBS_1 = (0.3, 0.4, 0.3)
PATH_PROBS_2 = (0.5, 0.0, 0.5)


def get_cmap(n, name="rainbow"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def interpolate(point_1, point_2, num_intermediate, endpoint=True):
    t = np.linspace(0, 1, num_intermediate, endpoint=endpoint).reshape(-1, 1)
    interpolated_array = ((1 - t) * point_1.T) + (t * point_2.T)
    return interpolated_array


class PathGenerator:
    def __init__(self, waypoints, step_size, num_draws=10, noise_scale=0.25):
        """
        waypoints is a list of points where the
        """
        self.waypoints = waypoints
        self.step_size = step_size
        self._num_draws = num_draws
        self._noise_scale = noise_scale
        self.build_paths()

    def build_paths(self):
        self._paths = []
        for wp in self.waypoints:
            waypoints = np.array(wp)
            final_path = []
            for i in range(len(waypoints) - 1):
                point_1, point_2 = waypoints[i], waypoints[i + 1]
                path_length = np.linalg.norm(point_1 - point_2)
                path_num_steps = int(path_length / self.step_size)
                final_path.append(
                    interpolate(
                        point_1,
                        point_2,
                        path_num_steps,
                        endpoint=(i == (len(waypoints) - 2)),
                    )
                )
            final_path = np.concatenate(final_path)
            self._paths.append(final_path)

    def draw(self):
        # Draw the paths a little randomly each time.
        get_color = get_cmap(len(self._paths) + 1)
        for color, path in enumerate(self._paths):
            path_color = get_color(color)
            for _ in range(self._num_draws):
                random_noise = np.random.normal(
                    loc=0.0, scale=self._noise_scale, size=path.shape
                )
                random_path = path + random_noise

                plt.plot(
                    random_path[:, 0], random_path[:, 1], "o-", c=path_color, alpha=0.25
                )

        plt.show()

    def get_random_paths(self, num_paths, probabilities=None):
        if probabilities is None:
            probabilities = np.ones(len(self._paths))

        assert len(probabilities) == len(self._paths)
        # Normalize the probabilities
        probabilities = np.array(probabilities) / np.sum(probabilities)
        num_paths_on_each = (num_paths * probabilities).astype(int)

        paths_dataset = []
        for num_generated_path, path in zip(num_paths_on_each, self._paths):
            (path_len, dim) = path.shape
            random_noise = np.random.normal(
                loc=0.0,
                scale=self._noise_scale,
                size=(num_generated_path, path_len, dim),
            )
            random_paths = (
                einops.repeat(
                    path, "len dim -> batch len dim", batch=num_generated_path
                )
                + random_noise
            )
            paths_dataset.append(random_paths)

        return paths_dataset

    @staticmethod
    def get_obs_action_from_path(paths):
        """Given path of dimensions batch x len x dimension, convert it to two arrays of obs and actions."""
        obses = paths[:, :-1, :]
        # Find the velocity vector.
        actions = paths[:, 1:, :] - paths[:, :-1, :]
        return obses, actions

    def get_memoryless_dataset(self, num_paths, probabilities=None):
        all_paths = self.get_random_paths(num_paths, probabilities)
        all_obses, all_actions = [], []
        for path in all_paths:
            obses, actions = PathGenerator.get_obs_action_from_path(path)
            all_obses.append(
                einops.rearrange(obses, "batch len dim -> (batch len) dim")
            )
            all_actions.append(
                einops.rearrange(actions, "batch len dim -> (batch len) dim")
            )

        # This should have dimensions dataset_size (sum path lengths) x dim
        full_obs_dataset = np.concatenate(all_obses)
        # Calculate the next move.
        full_action_dataset = np.concatenate(all_actions)
        return (full_obs_dataset, full_action_dataset)

    def get_sequence_dataset(self, num_paths, probabilities=None):
        all_paths = self.get_random_paths(num_paths, probabilities)
        all_obses, all_actions = [], []
        max_length = -1
        total_trajectories = 0
        for path in all_paths:
            obses, actions = PathGenerator.get_obs_action_from_path(path)
            max_length = max(max_length, obses.shape[1])
            total_trajectories += len(obses)
            all_obses.append(obses)
            all_actions.append(actions)

        obs_array = np.zeros((total_trajectories, max_length, all_obses[-1].shape[-1]))
        action_array = np.zeros(
            (total_trajectories, max_length, all_actions[-1].shape[-1])
        )
        mask = np.zeros((total_trajectories, max_length))

        current_trajectory = 0
        for obses_nparray, actions_nparray in zip(all_obses, all_actions):
            assert len(obses_nparray) == len(actions_nparray)
            current_traj_length = obses_nparray.shape[1]
            obs_array[
                current_trajectory : current_trajectory + len(obses_nparray),
                :current_traj_length,
                :,
            ] = obses_nparray
            action_array[
                current_trajectory : current_trajectory + len(actions_nparray),
                :current_traj_length,
                :,
            ] = actions_nparray
            mask[
                current_trajectory : current_trajectory + len(actions_nparray),
                :current_traj_length,
            ] = 1
            current_trajectory += len(obses_nparray)

        return (obs_array, action_array, mask)
