import logging
import einops
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset
from pathlib import Path
import numpy as np
from envs.multi_route import multi_route
from utils import (
    shuffle_along_axis,
    transpose_batch_timestep,
    split_datasets,
    eval_mode,
)
from typing import Union, Callable, Optional
from tqdm import tqdm


class RelayKitchenTrajectoryDataset(TensorDataset):
    def __init__(self, data_directory, device="cpu"):
        data_directory = Path(data_directory)
        observations = np.load(data_directory / "observations_seq.npy")
        actions = np.load(data_directory / "actions_seq.npy")
        masks = np.load(data_directory / "existence_mask.npy")
        # The current values are in shape T x N x Dim, move to N x T x Dim
        observations, actions, masks = transpose_batch_timestep(
            observations, actions, masks
        )
        self.masks = masks
        super().__init__(
            torch.from_numpy(observations).to(device).float(),
            torch.from_numpy(actions).to(device).float(),
            torch.from_numpy(masks).to(device).float(),
        )
        self.actions = self.tensors[1]

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)


class CarlaMultipathTrajectoryDataset(Dataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        subset_fraction: float = 1.0,
        device="cpu",
        preprocess_to_float: bool = False,
    ):
        assert 0.0 < subset_fraction <= 1.0, "subset_fraction must be in (0, 1]"
        self.device = device
        self.data_directory = Path(data_directory)
        print("CARLA loading: started")
        self.seq_lengths_all = torch.load(self.data_directory / "seq_lengths.pth")
        self.observations_all = torch.load(self.data_directory / "all_observations.pth")
        self.actions_all = torch.load(self.data_directory / "all_actions_pm1.pth")
        print("CARLA loading: done")

        N = int(len(self.seq_lengths_all) * subset_fraction)
        self.seq_lengths = self.seq_lengths_all[:N]
        self.observations = self.observations_all[:N, :, :, :, :]
        self.actions = self.actions_all[:N, :, :]
        del self.seq_lengths_all, self.observations_all, self.actions_all

        self.preprocess_to_float = preprocess_to_float
        # NOTE: dividing by 255 might explode memory if image size is large
        if self.preprocess_to_float:
            self.observations = self.observations / 255.0

    def __len__(self):
        return len(self.seq_lengths)

    def __getitem__(self, idx):
        # observations: Tensor[N, T, C, H, W]
        # actions: Tensor[N, T, C]
        T = self.seq_lengths[idx]
        observations = self.observations[idx, :T, :, :, :]
        if not self.preprocess_to_float:
            observations = observations / 255.0
        actions = self.actions[idx, :T, :]
        mask = torch.ones(T, dtype=torch.float32)  # existence mask
        return (
            observations,
            actions,
            mask,
        )

    def get_seq_length(self, idx) -> int:
        return int(self.seq_lengths[idx])

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)


class CarlaMultipathStateTrajectoryDataset(Dataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        subset_fraction: float = 1.0,
        obs_noise_scale: float = 0.0,
        device="cpu",
    ):
        assert 0.0 < subset_fraction <= 1.0, "subset_fraction must be in (0, 1]"
        self.device = device
        self.data_directory = Path(data_directory)
        self.seq_lengths_all = torch.load(self.data_directory / "seq_lengths.pth")
        self.observations_all = torch.load(
            self.data_directory / "all_observations.pth"
        ).float()
        self.actions_all = torch.load(
            self.data_directory / "all_actions_pm1.pth"
        ).float()
        # normalize observations
        self.observations_mean = torch.load(
            self.data_directory / "observations_mean.pth"
        )
        self.observations_std = torch.load(self.data_directory / "observations_std.pth")
        # don't normalize the last col of affine (always 0,0,0,1)
        self.observations_std[12:16] = 1.0
        self.observations_all -= self.observations_mean
        self.observations_all /= self.observations_std
        self.obs_noise_scale = obs_noise_scale

        N = int(len(self.seq_lengths_all) * subset_fraction)
        self.seq_lengths = self.seq_lengths_all[:N]
        self.observations = self.observations_all[:N, :, :]
        self.actions = self.actions_all[:N, :, :]
        del self.seq_lengths_all, self.observations_all, self.actions_all

    def __len__(self):
        return len(self.seq_lengths)

    def __getitem__(self, idx):
        # observations: Tensor[batch seq state_dim(23)]
        # actions: Tensor[batch seq action_dim(2)]
        T = self.seq_lengths[idx]
        observations = self.observations[idx, :T, :]
        observations += torch.randn_like(observations) * self.obs_noise_scale
        actions = self.actions[idx, :T, :]
        mask = torch.ones(T, dtype=torch.float32)  # existence mask
        return (
            observations,
            actions,
            mask,
        )

    def get_seq_length(self, idx) -> int:
        return int(self.seq_lengths[idx])

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)


class PushTrajectoryDataset(TensorDataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        device="cpu",
    ):
        self.device = device
        self.data_directory = Path(data_directory)
        logging.info("Multimodal loading: started")
        self.observations = np.load(
            self.data_directory / "multimodal_push_observations.npy"
        )
        self.actions = np.load(self.data_directory / "multimodal_push_actions.npy")
        self.masks = np.load(self.data_directory / "multimodal_push_masks.npy")
        self.observations = torch.from_numpy(self.observations).to(device).float()
        self.actions = torch.from_numpy(self.actions).to(device).float()
        self.masks = torch.from_numpy(self.masks).to(device).float()
        logging.info("Multimodal loading: done")
        # The current values are in shape N x T x Dim, so all is good in the world.
        super().__init__(
            self.observations,
            self.actions,
            self.masks,
        )

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)


class MultiPathTrajectoryDataset(TensorDataset):
    def __init__(
        self,
        path_waypoints=multi_route.MULTI_PATH_WAYPOINTS_1,
        path_probs=multi_route.PATH_PROBS_1,
        num_samples=200_000,
        device="cpu",
    ):
        path_generator = multi_route.PathGenerator(
            waypoints=path_waypoints,
            step_size=1,
            num_draws=100,
            noise_scale=0.05,
        )
        observations, actions, length_mask = path_generator.get_sequence_dataset(
            num_paths=num_samples, probabilities=path_probs
        )
        self.length_mask = length_mask
        super().__init__(
            torch.from_numpy(observations).to(device).float(),
            torch.from_numpy(actions).to(device).float(),
            torch.from_numpy(length_mask).to(device).float(),
        )

    def get_seq_length(self, idx) -> int:
        return int(self.length_mask[idx].sum().item())


class GridTrajectoryDataset(TensorDataset):
    def __init__(
        self,
        grid_size=5,
        device="cpu",
        num_samples=1_000_000,
        top_prob=0.4,
        noise_scale=0.05,
        random_seed=42,
        scale_factor=1.0,
    ):
        rng = np.random.default_rng(random_seed)
        total_grid_size = grid_size * 2
        top_length = int(total_grid_size * top_prob)
        side_length = total_grid_size - top_length

        all_up_actions = np.concatenate(
            [np.ones((num_samples, top_length)), np.zeros((num_samples, side_length))],
            axis=-1,
        ).astype(float)
        all_up_actions = shuffle_along_axis(all_up_actions, axis=-1)
        all_side_actions = 1.0 - all_up_actions
        all_actions = np.stack([all_up_actions, all_side_actions], axis=-1)
        all_observations = np.cumsum(all_actions, axis=1)  # [N, T, 2]
        all_actions += rng.normal(scale=noise_scale, size=all_actions.shape)
        all_observations += rng.normal(scale=noise_scale, size=all_observations.shape)

        # Scale the actions to be between 0 and scale_factor
        all_observations, all_actions = (
            scale_factor * all_observations,
            scale_factor * all_actions,
        )
        # All cells are valid
        mask = np.ones(all_observations.shape[:-1])
        self.mask = mask

        super().__init__(
            torch.from_numpy(all_observations).to(device).float(),
            torch.from_numpy(all_actions).to(device).float(),
            torch.from_numpy(mask).to(device).float(),
        )

    def get_seq_length(self, idx) -> int:
        return int(self.mask[idx].sum().item())


class TrajectorySlicerDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        window: int,
        transform: Optional[Callable] = None,
    ):
        """
        Slice a trajectory dataset into unique (but overlapping) sequences of length `window`.

        dataset: a trajectory dataset that satisfies:
            dataset.get_seq_length(i) is implemented to return the length of sequence i
            dataset[i] = (observations, actions, mask)
            observations: Tensor[T, ...]
            actions: Tensor[T, ...]
            mask: Tensor[T]
                0: invalid
                1: valid
        window: int
            number of timesteps to include in each slice
        returns: a dataset of sequences of length `window`
        """
        self.dataset = dataset
        self.window = window
        self.transform = transform
        self.slices = []
        min_seq_length = np.inf
        for i in range(len(self.dataset)):  # type: ignore
            T = self._get_seq_length(i)  # avoid reading actual seq (slow)
            min_seq_length = min(T, min_seq_length)
            if T - window < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={window}")
            else:
                self.slices += [
                    (i, start, start + window) for start in range(T - window)
                ]  # slice indices follow convention [start, end)

            if min_seq_length < window:
                print(
                    f"Ignored short sequences. To include all, set window <= {min_seq_length}."
                )

    def _get_seq_length(self, idx: int) -> int:
        # Adding this convenience method to avoid reading the actual sequence
        # We retrieve the length in trajectory slicer just so we can use subsetting
        # and shuffling before we pass a dataset into TrajectorySlicerDataset
        return self.dataset.get_seq_length(idx)

    def _get_all_actions(self) -> torch.Tensor:
        return self.dataset.get_all_actions()

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        values = tuple(
            x[start:end] for x in self.dataset[i]
        )  # (observations, actions, mask)
        # optionally apply transform
        if self.transform is not None:
            values = self.transform(values)
        return values


class TrajectorySlicerSubset(TrajectorySlicerDataset):
    def _get_seq_length(self, idx: int) -> int:
        # self.dataset is a torch.dataset.Subset, so we need to use the parent dataset
        # to extract the true seq length.
        subset = self.dataset
        return subset.dataset.get_seq_length(subset.indices[idx])  # type: ignore

    def _get_all_actions(self) -> torch.Tensor:
        return self.dataset.dataset.get_all_actions()


class TrajectoryRepDataset(Dataset):
    def __init__(
        self,
        trajectory_dataset: Dataset,
        encoder: nn.Module,
        preprocess: Callable[[torch.Tensor], torch.Tensor] = None,
        postprocess: Callable[[torch.Tensor], torch.Tensor] = None,
        device: Union[torch.device, str] = "cuda",
        batch_size: Optional[int] = 128,
    ):
        """
        Given a trajectory dataset, encode its states into representations.
        Inputs:
            trajectory_dataset: a trajectory dataset that satisfies:
                dataset[i] = (observations, actions, mask)
                observations: Tensor[T, ...]
                actions: Tensor[T, ...]
                masks: Tensor[T]
                    0: invalid
                    1: valid
            encoder: a module that accepts observations and returns a representation
            device: encoder will be run on this device
            batch_size: if not None, will batch frames into batches of this size (to avoid OOM)
        """
        self.device = device
        encoder = encoder.to(device)  # not saving encoder to lower VRAM usage
        self.obs = []
        self.actions = []
        self.masks = []
        self.postprocess = postprocess
        with eval_mode(encoder, no_grad=True):
            for i in tqdm(range(len(trajectory_dataset))):
                obs, act, mask = trajectory_dataset[i]
                if preprocess is not None:
                    obs = preprocess(obs)
                if batch_size is not None:
                    obs_enc = []
                    for t in range(0, obs.shape[0], batch_size):
                        batch = obs[t : t + batch_size].to(self.device)
                        obs_enc.append(encoder(batch).cpu())
                    obs_enc = torch.cat(obs_enc, dim=0)
                else:
                    obs_enc = encoder(obs.to(self.device)).cpu()
                self.obs.append(obs_enc)
                self.actions.append(act)
                self.masks.append(mask)
        del encoder
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        obs = self.obs[idx]
        if self.postprocess is not None:
            obs = self.postprocess(obs)
        return (obs, self.actions[idx], self.masks[idx])

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        return torch.cat(self.actions, dim=0)


def get_relay_kitchen_train_val(
    data_directory,
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    window_size=10,
):

    relay_kitchen_trajectories = RelayKitchenTrajectoryDataset(data_directory)
    train_set, val_set = split_datasets(
        relay_kitchen_trajectories,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    # Convert to trajectory slices.
    train_trajectories = TrajectorySlicerSubset(train_set, window=window_size)
    val_trajectories = TrajectorySlicerSubset(val_set, window=window_size)
    return train_trajectories, val_trajectories


def get_push_train_val(
    data_directory,
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    window_size=10,
):
    push_trajectories = PushTrajectoryDataset(data_directory)
    train_set, val_set = split_datasets(
        push_trajectories,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    # Convert to trajectory slices.
    train_trajectories = TrajectorySlicerSubset(train_set, window=window_size)
    val_trajectories = TrajectorySlicerSubset(val_set, window=window_size)
    return train_trajectories, val_trajectories


def get_multiroute_dataset(
    train_fraction=0.9, random_seed=42, device="cpu", window_size=10
):
    train_set, val_set = split_datasets(
        MultiPathTrajectoryDataset(num_samples=20_000),
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    return (
        TrajectorySlicerSubset(train_set, window=window_size),
        TrajectorySlicerSubset(val_set, window=window_size),
    )


def get_grid_dataset(train_fraction=0.9, random_seed=42, device="cpu", window_size=10):
    train_set, val_set = split_datasets(
        GridTrajectoryDataset(random_seed=random_seed),
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    return (
        TrajectorySlicerSubset(train_set, window=window_size),
        TrajectorySlicerSubset(val_set, window=window_size),
    )


def get_carla_multipath_dataset(
    data_directory,
    subset_fraction=1.0,
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    window_size=10,
    preprocess_to_float: bool = False,
):
    train_set, val_set = split_datasets(
        CarlaMultipathTrajectoryDataset(
            data_directory,
            subset_fraction=subset_fraction,
            device=device,
            preprocess_to_float=preprocess_to_float,
        ),
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    return TrajectorySlicerSubset(
        train_set, window=window_size
    ), TrajectorySlicerSubset(val_set, window=window_size)


def get_carla_multipath_state_dataset(
    data_directory,
    subset_fraction=1.0,
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    window_size=10,
):
    train_set, val_set = split_datasets(
        CarlaMultipathStateTrajectoryDataset(
            data_directory,
            subset_fraction=subset_fraction,
            device=device,
        ),
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    return TrajectorySlicerSubset(
        train_set, window=window_size
    ), TrajectorySlicerSubset(val_set, window=window_size)


def get_carla_multipath_rep_dataset(
    data_directory,
    subset_fraction=1.0,
    train_fraction=0.9,
    random_seed=42,
    device="cuda",
    batch_size=None,
    window_size=10,
    encoder: nn.Module = nn.Identity,
    preprocess: Callable[[torch.Tensor], torch.Tensor] = None,
    postprocess: Callable[[torch.Tensor], torch.Tensor] = None,
):
    dataset = TrajectoryRepDataset(
        CarlaMultipathTrajectoryDataset(
            data_directory,
            subset_fraction=subset_fraction,
            device=device,
            preprocess_to_float=False,
        ),
        encoder,
        preprocess=preprocess,
        postprocess=postprocess,
        device=device,
        batch_size=batch_size,
    )
    train_set, val_set = split_datasets(
        dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    return TrajectorySlicerSubset(
        train_set, window=window_size
    ), TrajectorySlicerSubset(val_set, window=window_size)
