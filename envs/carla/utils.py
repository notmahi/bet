import carla
from typing import Tuple, Optional, Union
import numpy as np


def serialize_location(l: carla.Location) -> Tuple[float, float, float]:
    return l.x, l.y, l.z


def serialize_rotation(r: carla.Rotation) -> Tuple[float, float, float]:
    return r.pitch, r.yaw, r.roll


def serialize_transform(
    t: carla.Transform,
) -> Tuple[float, float, float, float, float, float]:
    return (*serialize_location(t.location), *serialize_rotation(t.rotation))


def deserialize_transform(
    t: Union[Tuple[float, float, float, float, float, float], np.ndarray],
) -> carla.Transform:
    return carla.Transform(
        carla.Location(x=t[0], y=t[1], z=t[2]),
        carla.Rotation(pitch=t[3], yaw=t[4], roll=t[5]),
    )


def serialize_control_to_dict(frame, control):
    keys = [
        "throttle",
        "steer",
        "brake",
        "hand_brake",
        "reverse",
        "manual_gear_shift",
        "gear",
    ]
    result = {k: getattr(control, k) for k in keys}
    result["frame"] = frame
    return result


def serialize_control_to_arr(control):
    return np.array([control.throttle - control.brake, control.steer])


def deserialize_control_from_arr(arr):
    if arr[0] > 0:
        throttle = arr[0]
        brake = 0
    else:
        throttle = 0
        brake = -arr[0]
    steer = arr[1]
    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)


def deserialize_control_from_dict(control_dict, tensor=False):
    keys = [
        "throttle",
        "steer",
        "brake",
        "hand_brake",
        "reverse",
        "manual_gear_shift",
        "gear",
    ]
    result = carla.VehicleControl()
    for k in keys:
        setattr(result, k, control_dict[k])
    return result


def preproc_carla_img(img, channel_first: bool = False) -> np.ndarray:
    img.convert(carla.ColorConverter.Raw)
    arr = np.frombuffer(img.raw_data, dtype="uint8")
    arr = arr.reshape((img.height, img.width, 4))
    arr = arr[:, :, :3]  # BGRA -> BGR
    arr = arr[:, :, ::-1]  # BGR -> RGB
    if channel_first:
        arr = arr.transpose(2, 0, 1)
    return arr.copy() / 255.0


def add_noise_to_transform(
    t: carla.Transform,
    noise_std: float,
    rng: Optional[np.random.RandomState] = None,
    keep_z: bool = True,
) -> carla.Transform:
    """
    Add noise to a transform.
    """
    if rng is None:
        rng = np.random
    noise = rng.normal(0, noise_std, size=6)
    if keep_z:
        noise[2] = 0
    serialized_t = np.array(serialize_transform(t))
    serialized_t += noise
    return deserialize_transform(serialized_t)


def add_noise_to_action(
    action: np.ndarray,
    noise_std: float,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Add additive noise to an action with N(0, noise_std).
    Throttle/brake axis is guaranteed to keep the same sign,
    i.e. sign(output[0]) == sign(action[0]), such that throttle never becomes brake and vice versa.
    """
    if rng is None:
        rng = np.random
    noise = rng.normal(0, noise_std, size=2)
    sign = np.sign(action[0])
    result = action + noise
    result[0] = sign * np.abs(result[0])
    result = np.clip(result, -1, 1)
    return result


# NOTE: this assumes that hydra will set the workdir to ./exp_local/{date}/{time}_{experiment_name}
Town10_spawns = np.load("../../../envs/carla/Town10_spawns.npy")
Town10_spawns = [deserialize_transform(t) for t in Town10_spawns]
Town04_spawns = np.load("../../../envs/carla/Town04_spawns.npy")
Town04_spawns = [deserialize_transform(t) for t in Town04_spawns]
