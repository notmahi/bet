import sys

sys.path.append("../..")

import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import envs
from envs.carla.builtin_agent import BuiltinAgent
import gym
import multiprocessing as mp
from typing import Dict
import torchvision
import carla
import einops


def save_trajectory_helper(
    agent: BuiltinAgent,
    path: os.PathLike,
    metadata: Dict,
    save_video: bool = False,
):
    trajectory_name = datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f")
    trajectory_dir = Path(path) / trajectory_name
    trajectory_dir.mkdir(parents=True)
    print("Saving trajectory to {}".format(trajectory_dir))
    with open(trajectory_dir / "actions.json", "w") as f:
        json.dump(agent.actions, f)
    with open(trajectory_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    buffer = agent.states
    sensor_name = "0"
    if save_video:
        video_path = trajectory_dir / (sensor_name + ".mp4")
        video_array = np.array(buffer)
        video_array = (video_array * 255).astype(np.uint8)
        video_array = einops.rearrange(video_array, "T C H W -> T H W C")
        torchvision.io.write_video(
            str(video_path),
            video_array=video_array,
            fps=1 / agent.env.dt,
        )
    else:
        for elem in buffer:
            if elem is not None:
                relative_frame = elem.frame - agent.start_frame
                save_path = trajectory_dir / sensor_name / f"{relative_frame:04d}.png"
                if relative_frame >= 0:
                    elem.save_to_disk(str(save_path))
    print("Saved trajectory to", trajectory_dir)
    return trajectory_dir


if __name__ == "__main__":
    env = gym.make("carla-multipath-town04-merge-v0")
    processes = []

    for i in range(100):
        obs, info = env.reset(
            return_info=True, seed=i, action_exec_noise_std=0.1, spawn_noise_std=0.5
        )
        print(env.world.get_weather())
        frame = info["frame"]
        done = False
        rng = np.random.RandomState(seed=i)
        route_choice = rng.randint(0, len(env.active_routes))
        # each route is a tuple (spawn, route_points)
        route_points = env.active_routes[route_choice][1]
        agent = BuiltinAgent(
            env,
            route_points,
            seed=i,
        )
        while not done:
            action = agent.step(obs, info)
            print(len(agent.states), len(agent.actions))
            action = np.array([action.throttle - action.brake, action.steer])
            action = np.clip(action, -1, 1)
            obs, reward, done, info = env.step(action)

        last_frame = info["frame"]
        while info["obs_frame"] < last_frame:
            action = agent.step(obs, info)
            action = np.array([action.throttle - action.brake, action.steer])
            action = np.clip(action, -1, 1)
            obs, reward, done, info = env.step(action)

        # async save
        p = mp.Process(
            target=save_trajectory_helper,
            args=(
                agent,
                "/path/to/dataset/save/dir",
                {"seed": i},
                True,
            ),
        )
        p.start()
        processes.append(p)

    env.close()
