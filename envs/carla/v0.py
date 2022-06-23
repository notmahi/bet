import carla
import numpy as np
import gym
import subprocess
from collections import deque
from typing import Tuple, Optional, List
import socket
import time
import atexit
from .weather import get_weather_config
from .utils import (
    deserialize_control_from_arr,
    preproc_carla_img,
    add_noise_to_transform,
    add_noise_to_action,
    Town04_spawns,
    Town10_spawns,
    deserialize_transform,
)


class CarlaMultipathEnvBase(gym.Env):
    CARLA_BIN = "/opt/carla-simulator/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
    WORLD_NAME = "Town10HD_Opt"
    CARLA_ARGV = ["CarlaUE4", "--world", WORLD_NAME]
    TIMEOUT = 40
    R_CLOSE = 4.0
    # list of (spawn, waypoints); last waypoint is destination
    ROUTES: List[Tuple[carla.Transform, List[carla.Location]]] = []
    SPECTATOR_VIEWPOINT = carla.Transform(
        carla.Location(x=-49.796131, y=22.167667, z=129.373856),
        carla.Rotation(pitch=-82.899261, yaw=-12.960452, roll=-0.001222),
    )
    WEATHER = None  # None for random weather

    def __init__(
        self,
        dt: float = 0.05,
    ):
        super().__init__()
        self.dt = dt
        self.img_size = (224, 224)
        self.fov = 90
        self.spawns = [x[0] for x in self.ROUTES]
        self.active_routes = None

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(3, *self.img_size), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.frame = None
        self.rng = None
        self.server_proc = subprocess.Popen([self.CARLA_BIN, *self.CARLA_ARGV])
        atexit.register(self.server_proc.kill)
        start = time.time()
        while not self._is_open("localhost", 2000, timeout=1):
            if (time.time() - start) > self.TIMEOUT:
                raise socket.timeout("Failed to connect to CARLA.")
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(self.TIMEOUT)
        self.client.load_world(self.WORLD_NAME)
        self.world = self.client.get_world()
        self.spectator = self.world.get_spectator()
        self.vehicle = None
        self.sensors = []
        self._set_synchronous(True)

    def _is_open(self, host: str, port: int, timeout: float = 1.0):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((host, port))
            s.settimeout(timeout)
            s.shutdown(socket.SHUT_RDWR)
            return True
        except:
            return False

    def garbage_collect(self):
        if self.vehicle is not None:
            self.vehicle.destroy()
        if len(self.sensors) > 0:
            for elem in self.sensors:
                elem["sensor"].destroy()
        self.vehicle = None
        self.sensors = []

    def reset(
        self,
        reload: bool = False,
        return_info: bool = False,
        seed: Optional[int] = None,
        spawn_noise_std: float = 0.5,  # N(0, std) additive noise to spawn location
        action_exec_noise_std: float = 0.0,  # N(0, std) additive, but guaranteed throttle/brake axis does not flip sign
    ):
        if self.ROUTES is None:
            raise ValueError("No routes specified.")
        if reload:
            self.client.set_timeout(15)
            self.client.load_world(self.WORLD_NAME)
            self.world = self.client.get_world()
            self.spectator = self.world.get_spectator()
            self._set_synchronous(True)
        else:
            self.garbage_collect()
        self.vehicle = None
        self.sensors = []
        self.spawn_noise_std = spawn_noise_std
        self.action_exec_noise_std = action_exec_noise_std

        if seed is None:
            self.seed = np.random.randint(0, 4294967295)  # 0..2^32-1
        else:
            self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        self.spawn = self.rng.choice(self.spawns)
        self.active_routes = [x for x in self.ROUTES if x[0] == self.spawn]
        # NOTE: this assumes there's only 1 spawn
        self.active_routes_used = [False] * len(self.active_routes)
        # get weather config regardless of whether it's random
        # to ensure rng is consistent
        weather_config = get_weather_config(self.rng)
        if self.WEATHER is None:
            self.world.set_weather(weather_config)
        else:
            self.world.set_weather(self.WEATHER)
        vehicle_blueprint = self._get_vehicle_blueprint_by_id(1)
        self._spawn_vehicle(
            vehicle_blueprint,
            add_noise_to_transform(self.spawn, self.spawn_noise_std, self.rng),
        )
        self._spawn_camera(
            self.vehicle,
            self.img_size,
            self.fov,
        )

        # set spectator camera
        self.spectator.set_transform(self.SPECTATOR_VIEWPOINT)

        # make sure camera sensors receive the first image, and the car is on the ground
        for i in range(20):
            obs, _, _, info = self.step(np.array([0, 0]))

        if return_info:
            return obs, info
        else:
            return obs

    def step(self, action: np.ndarray):
        vel = self.vehicle.get_velocity().length() * 3.6
        # print(f"velocity: {vel:.2f}")
        if not (-1 <= action[0] <= 1):  # throttle/brake
            print("Throttle/brake action out of bounds: {}".format(action[0]))
        if not (-1 <= action[1] <= 1):  # steer
            print("Steer action out of bounds: {}".format(action[1]))
        action = np.clip(action, -1, 1)
        action = add_noise_to_action(action, self.action_exec_noise_std, self.rng)
        action = deserialize_control_from_arr(action)
        self.vehicle.apply_control(action)
        self.frame = self.world.tick()
        self._green_light_to_vehicle(self.vehicle)
        state = []  # observations
        obs_frame = -1
        for sensor in self.sensors:
            if len(sensor["buffer"]) > 0:
                elem = sensor["buffer"].popleft()
            else:
                elem = None
            state.append(elem)
        if len(state) == 1:
            state = state[0]
        if state is not None:
            obs_frame = state.frame
            state = preproc_carla_img(state, channel_first=True).copy()

        reward, routes_reached = self._calc_reward(self.vehicle, self.active_routes)
        for i in range(len(self.active_routes_used)):
            self.active_routes_used[i] = self.active_routes_used[i] or routes_reached[i]
        done = reward == 1
        info = {
            "frame": self.frame,
            "obs_frame": obs_frame,
            "routes_used": self.active_routes_used,
        }
        return state, reward, done, info

    def _calc_reward(self, vehicle, routes):
        dests = [x[1][-1] for x in routes]  # destinations
        ego_location = vehicle.get_location()
        routes_reached = [False] * len(routes)
        for i, route in enumerate(routes):
            # update whether ego vehicle has reached the waypoints
            # NOTE: assuming only destination overlaps between routes! Really should be counting
            for p in route[1][:-1]:  # not including the destination
                if p.distance(ego_location) < self.R_CLOSE:
                    routes_reached[i] = True
        # so _calc_reward really becomes _update_route_checkpoints
        # and done can be determined by checking if any destination flag is True
        best_distance = min([ego_location.distance(x) for x in dests])
        if best_distance < self.R_CLOSE:
            return 1, routes_reached
        else:
            return 0, routes_reached

    def render(self, mode):
        pass

    def set_state(self):
        pass

    def close(self):
        self._set_synchronous(False)
        self.garbage_collect()
        self.server_proc.kill()
        self.server_proc.wait()

    def _set_synchronous(self, synchronous: bool = True):
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous
        settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(settings)

    def _get_vehicle_blueprint_by_id(self, blueprint_id: int, randomize: bool = False):
        bps = self.world.get_blueprint_library().filter("vehicle.*")
        bps = sorted(bps, key=lambda bp: bp.id)
        bp = bps[blueprint_id]
        for attr in ["color", "driver_id"]:
            if bp.has_attribute(attr):
                vals = bp.get_attribute(attr).recommended_values
                if randomize:
                    val = self.rng.choice(vals)
                else:
                    val = vals[0]
                bp.set_attribute(attr, val)
        bp.set_attribute("role_name", "hero")
        return bp

    def _spawn_vehicle(
        self, vehicle_blueprint: carla.ActorBlueprint, spawn_point: carla.Transform
    ):
        self.vehicle = self.world.spawn_actor(vehicle_blueprint, spawn_point)
        light_state = (
            carla.VehicleLightState.NONE
            | carla.VehicleLightState.LowBeam
            | carla.VehicleLightState.Position
        )
        self.vehicle.set_light_state(carla.VehicleLightState(light_state))
        return self.vehicle

    def _register_sensor_buffer(self, sensor: carla.Sensor, buffer: deque):
        def callback(data):
            buffer.append(data)

        sensor.listen(callback)

    def _spawn_camera(
        self,
        vehicle: carla.Vehicle,
        size: Tuple[int, int] = (224, 224),
        fov: float = 90,
        transform: Optional[carla.Transform] = None,
    ):
        FRONT = carla.Transform(carla.Location(x=0.8, z=1.7))
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(size[0]))
        bp.set_attribute("image_size_y", str(size[1]))
        bp.set_attribute("fov", str(fov))
        bp.set_attribute("sensor_tick", str(self.dt))
        if transform is None:
            transform = FRONT
        camera = self.world.spawn_actor(bp, transform, attach_to=vehicle)
        buffer = deque()
        self._register_sensor_buffer(camera, buffer)
        self.sensors.append(
            {
                "sensor": camera,
                "buffer": buffer,
            }
        )
        return camera

    def _green_light_to_vehicle(self, vehicle: carla.Vehicle):
        if vehicle.is_at_traffic_light():
            traffic_light = vehicle.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                traffic_light.set_state(carla.TrafficLightState.Green)


class CarlaMultipathSplitV0(CarlaMultipathEnvBase):
    ROUTES = [
        (Town10_spawns[137], [Town10_spawns[96].location]),  # left
        (Town10_spawns[137], [Town10_spawns[111].location]),  # straight
        (Town10_spawns[137], [Town10_spawns[55].location]),  # right
    ]
    SPECTATOR_VIEWPOINT = carla.Transform(
        carla.Location(x=-49.796131, y=22.167667, z=129.373856),
        carla.Rotation(pitch=-82.899261, yaw=-12.960452, roll=-0.001222),
    )
    WEATHER = carla.WeatherParameters.ClearNoon


class CarlaMultipathTown04MergeV0(CarlaMultipathEnvBase):
    WORLD_NAME = "Town04_Opt"
    START_SPAWN = carla.Transform(
        carla.Location(270, -250, 0.28194),
        carla.Rotation(0, 179.6, 0),
    )
    ROUTES = [
        (
            START_SPAWN,
            [
                carla.Location(x=255.09, y=-223.53, z=0.01),  # for route detection
                Town04_spawns[354].location,
                Town04_spawns[359].location,
                Town04_spawns[160].location,
            ],
        ),  # left
        (
            START_SPAWN,
            [
                carla.Location(x=258.5, y=-270.2, z=0.01),  # for route detection
                Town04_spawns[369].location,
                Town04_spawns[361].location,
                Town04_spawns[160].location,
            ],
        ),  # right
    ]
    SPECTATOR_VIEWPOINT = carla.Transform(
        carla.Location(x=224.75, y=-270, z=111.8),
        carla.Rotation(
            pitch=-88, yaw=-179.9, roll=0.004
        ),  # gimbal lock seems to cause some weird drift during episodes
    )
    # copying the default weather for Town04, so that we don't generate random weather
    # but removing non-deterministic environment noise like clouds and wind
    WEATHER = carla.WeatherParameters(
        cloudiness=0.0,  # 10.0
        precipitation=0.0,
        precipitation_deposits=10.0,
        wind_intensity=0.0,  # 30.0
        sun_azimuth_angle=150.0,
        sun_altitude_angle=60.0,
        fog_density=40.0,
        fog_distance=60.0,
        fog_falloff=2.0,
        wetness=30.0,
        scattering_intensity=1.0,
        mie_scattering_scale=0.03,
        rayleigh_scattering_scale=0.0331,
    )


class CarlaMultipathTown10MergeV0(CarlaMultipathEnvBase):
    ROUTES = [
        (
            Town10_spawns[57],
            [
                carla.Location(x=26, y=41.9455, z=0.6),
                Town10_spawns[43].location,
                Town10_spawns[37].location,
                Town10_spawns[32].location,
            ],
        ),  # right, then left
        (
            Town10_spawns[57],
            [
                carla.Location(x=26, y=41.9455, z=0.6),
                carla.Location(x=28, y=24.61119, z=0.6),
                Town10_spawns[59].location,
                Town10_spawns[133].location,
            ],
        ),  # straight, then right
    ]
    SPECTATOR_VIEWPOINT = carla.Transform(
        carla.Location(x=75, y=47.07217025756836, z=126.378662109375),
        carla.Rotation(
            pitch=-88.99633026123047, yaw=-90.47611999511719, roll=-0.004094338044524193
        ),
    )
    WEATHER = carla.WeatherParameters.ClearNoon
    WEATHER.sun_altitude_angle = 90.0
