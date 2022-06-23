import gym
import gym.spaces as spaces
import numpy as np

from envs.multi_route import multi_route


class MultiRouteEnvV1(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}
    NUM_ENV_STEPS = 50

    def __init__(self) -> None:
        super().__init__()
        self.obs_high_bound = 2 * multi_route.BOUNDS + 2
        self.obs_low_bound = -2
        self._dim = 2
        self._noise_scale = 0.25
        self._starting_point = np.zeros(self._dim)

        action_limit = 2  # double of the step size
        self.action_space = spaces.Box(
            -action_limit,
            action_limit,
            shape=self._starting_point.shape,
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(
            self.obs_low_bound,
            self.obs_high_bound,
            shape=self._starting_point.shape,
            dtype=np.float64,
        )

        self._target_bounds = spaces.Box(
            2 * multi_route.BOUNDS - 1,
            2 * multi_route.BOUNDS + 1,
            shape=self._starting_point.shape,
            dtype=np.float64,
        )

    def reset(self):
        self._state = self._starting_point + np.random.normal(
            0, self._noise_scale, size=self._starting_point.shape
        )
        return np.copy(self._state)

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        self._state += action
        reward = 0
        done = False

        if self._target_bounds.contains(self._state):
            reward = 1
            done = True
        if not self.observation_space.contains(self._state):
            reward = -1
            done = True

        return np.copy(self._state), reward, done, {}

    def render(self, *args, **kwargs):
        pass

    def set_state(self, state):
        err_msg = f"{state!r} ({type(state)}) invalid"
        assert self.observation_space.contains(state), err_msg
        self._state = np.copy(state)
        return self._state
