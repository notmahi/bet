import gym
import carla
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
from .agents.navigation.behavior_agent import BehaviorAgent
from .agents.navigation.global_route_planner import GlobalRoutePlanner
from .utils import serialize_control_to_dict, deserialize_control_from_dict


class BuiltinAgent:
    def __init__(
        self,
        env: gym.Env,
        route_points: Union[carla.Location, List[carla.Location]],
        seed: Optional[float] = None,
    ):
        self.env = env
        self.route_points = route_points
        if type(route_points) is carla.Location:
            self.route_points = [route_points]  # compatibility
        self.agent = BehaviorAgent(
            self.env.vehicle, behavior="normal", ignore_traffic_lights=True
        )
        self.agent.set_route_points(route_points)
        self.controller = self.agent.get_local_planner()._vehicle_controller
        self.controller.change_lateral_PID(
            {"K_P": 1.95, "K_I": 0.05, "K_D": 0, "dt": env.dt}
        )
        self.controller.change_longitudinal_PID(
            {"K_P": 0.2, "K_I": 0.005, "K_D": 0.002, "dt": env.dt}
        )
        self.start_frame = None
        self.latest_obs_frame = -1
        self.actions = []
        self.states: List[List[carla.SensorData]] = []

        if seed is None:
            self.seed = np.random.randint(0, 4294967295)  # 0..2^32-1
        else:
            self.seed = seed
        self.rng = np.random.RandomState(seed)

    def done(self):
        return self.agent.done()

    def step(self, obs: List, info: Dict):
        frame = info["frame"]
        if self.start_frame is None:
            self.start_frame = frame
        self.latest_obs_frame = max(self.latest_obs_frame, info["obs_frame"])
        if self.done():
            action = carla.VehicleControl()
        else:
            action = self.agent.run_step()
        if info["obs_frame"] >= self.start_frame:
            self.states.append(obs)
        self.actions.append(serialize_control_to_dict(frame - self.start_frame, action))
        return action

    def visualize_route(
        self,
        origin: carla.Location,
        destination: carla.Location,
        sampling_resolution: int = 2,
        marker: str = "O",
        color: Tuple[int, int, int] = (255, 255, 255),
    ):
        wmap = self.env.world.get_map()
        gp = GlobalRoutePlanner(wmap, sampling_resolution=sampling_resolution)
        plan = gp.trace_route(origin, destination)
        for waypoint, _ in plan:
            self.env.world.debug.draw_string(
                waypoint.transform.location,
                marker,
                draw_shadow=False,
                color=carla.Color(r=color[0], g=color[1], b=color[2]),
                life_time=5.0,
                persistent_lines=True,
            )

    def latest_frame_in_buffer(self):
        return self.latest_obs_frame

    def transpose(self, l: List[List[Any]]) -> List[List[Any]]:
        return [list(x) for x in zip(*l)]
