import logging
from collections import deque
from pathlib import Path
import os

import einops
import gym
from gym.wrappers import RecordVideo
import hydra
import numpy as np
import torch
from models.action_ae.generators.base import GeneratorDataParallel
from models.latent_generators.latent_generator import LatentGeneratorDataParallel
import utils
import wandb
from omegaconf import OmegaConf


class Workspace:
    def __init__(self, cfg):

        self.work_dir = os.path.join(
            Path.cwd(), f"snapshot_{cfg.experiment.cv_run_idx}"
        )
        os.makedirs(self.work_dir)
        print("Working directory: {}".format(self.work_dir))

        self.cfg = cfg
        self.device = (
            torch.device(cfg.experiment.device)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if self.cfg.experiment.data_parallel and self.device == torch.device("cpu"):
            raise ValueError("Data parallel is not supported on CPU")
        utils.set_seed_everywhere(cfg.experiment.seed)

        # Create the model
        self.action_ae = None
        self.obs_encoding_net = None
        self.state_prior = None
        if not self.cfg.experiment.lazy_init_models:
            self._init_action_ae()
            self._init_obs_encoding_net()
            self._init_state_prior()

        # Log in wandb
        wandb.init(
            dir=self.work_dir,
            project=cfg.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )
        wandb.config.update({"save_path": self.work_dir})
        # Add to eval config the config from the training model
        train_config_path = os.path.join(cfg.model.load_dir, ".hydra", "config.yaml")
        train_config = OmegaConf.load(train_config_path)
        wandb.config.update(
            {"train_config": OmegaConf.to_container(train_config, resolve=True)}
        )

        self.epoch = 0
        self.load_snapshot()

        # Define window-size to the one of the trained model by default, unless otherwise stated
        if cfg.experiment.window_size == "same":
            self.window_size = train_config.experiment.window_size
        else:
            self.window_size = cfg.experiment.window_size
        print(f"window_size: {self.window_size}")
        # Set up history archival.
        self.history = deque(maxlen=self.window_size)
        self.last_latents = None

        self.init_env()

    def init_env(self):
        self.env = gym.make(self.cfg.env.gym_name)
        if self.cfg.experiment.record_video:
            self.env = RecordVideo(
                self.env,
                video_folder=self.work_dir,
                episode_trigger=lambda x: x % 1 == 0,
            )

        if self.cfg.experiment.flatten_obs:
            self.env = gym.wrappers.FlattenObservation(self.env)

        if self.cfg.experiment.plot_interactions:
            self._setup_plots()

        if self.cfg.experiment.start_from_seen:
            self._setup_starting_state()

    def _init_action_ae(self):
        if self.action_ae is None:  # possibly already initialized from snapshot
            self.action_ae = hydra.utils.instantiate(
                self.cfg.action_interface.action_ae, _recursive_=False
            ).to(self.device)
            if self.cfg.experiment.data_parallel:
                self.action_ae = GeneratorDataParallel(self.action_ae)

    def _init_obs_encoding_net(self):
        if self.obs_encoding_net is None:  # possibly already initialized from snapshot
            self.obs_encoding_net = hydra.utils.instantiate(
                self.cfg.action_interface.encoder
            )
            self.obs_encoding_net = self.obs_encoding_net.to(self.device)
            if self.cfg.experiment.data_parallel:
                self.obs_encoding_net = torch.nn.DataParallel(self.obs_encoding_net)

    def _init_state_prior(self):
        if self.state_prior is None:  # possibly already initialized from snapshot
            self.state_prior = hydra.utils.instantiate(
                self.cfg.model,
                latent_dim=self.action_ae.latent_dim,
                vocab_size=self.action_ae.num_latents,
            ).to(self.device)
            if self.cfg.experiment.data_parallel:
                self.state_prior = LatentGeneratorDataParallel(self.state_prior)
            self.state_prior_optimizer = self.state_prior.get_optimizer(
                learning_rate=self.cfg.experiment.lr,
                weight_decay=self.cfg.experiment.weight_decay,
                betas=tuple(self.cfg.experiment.betas),
            )

    def _setup_plots(self):
        raise NotImplementedError

    def _setup_starting_state(self):
        raise NotImplementedError

    def _start_from_known(self):
        raise NotImplementedError

    def run_single_episode(self):
        obs_history = []
        action_history = []
        latent_history = []
        self.history = deque(maxlen=self.window_size)
        obs = self.env.reset()
        last_obs = obs
        if self.cfg.experiment.start_from_seen:
            obs = self._start_from_known()
        action, latents = self._get_action(obs, sample=True, keep_last_bins=False)
        done = False
        total_reward = 0
        obs_history.append(obs)
        action_history.append(action)
        latent_history.append(latents)
        for i in range(self.cfg.experiment.num_eval_steps):
            if self.cfg.experiment.plot_interactions:
                self._plot_obs_and_actions(obs, action, done)
            if done:
                self._report_result_upon_completion()
                break
            if self.cfg.experiment.enable_render:
                self.env.render(mode="human")
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if obs is None:
                obs = last_obs  # use cached observation in case of `None` observation
            else:
                last_obs = obs  # cache valid observation
            keep_last_bins = ((i + 1) % self.cfg.experiment.action_update_every) != 0
            action, latents = self._get_action(
                obs, sample=True, keep_last_bins=keep_last_bins
            )
            obs_history.append(obs)
            action_history.append(action)
            latent_history.append(latents)
        logging.info(f"Total reward: {total_reward}")
        logging.info(f"Final info: {info}")
        return total_reward, obs_history, action_history, latent_history, info

    def _report_result_upon_completion(self):
        pass

    def _plot_obs_and_actions(self, obs, chosen_action, done, all_actions=None):
        print(obs, chosen_action, done)
        raise NotImplementedError

    def _prepare_obs(self, obs):
        obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        enc_obs = self.obs_encoding_net(obs).squeeze(0)
        enc_obs = einops.repeat(
            enc_obs, "obs -> batch obs", batch=self.cfg.experiment.action_batch_size
        )
        return enc_obs

    def _get_action(self, obs, sample=False, keep_last_bins=False):
        with utils.eval_mode(
            self.action_ae, self.obs_encoding_net, self.state_prior, no_grad=True
        ):
            enc_obs = self._prepare_obs(obs)
            # Now, add to history. This automatically handles the case where
            # the history is full.
            self.history.append(enc_obs)
            if self.cfg.experiment.use_state_prior:
                enc_obs_seq = torch.stack(tuple(self.history), dim=0)  # type: ignore
                # Sample latents from the prior
                latents = self.state_prior.generate_latents(
                    enc_obs_seq,
                    torch.ones_like(enc_obs_seq).mean(dim=-1),
                )
                # For visualization, also get raw logits and offsets
                # placeholder_target = (
                #     torch.zeros_like(latents[0]),
                #     torch.zeros_like(latents[1]),
                # )
                # (
                #     logits_to_save,
                #     offsets_to_save,
                # ), _ = self.state_prior.get_latent_and_loss(enc_obs_seq, placeholder_target)
                logits_to_save, offsets_to_save = None, None

                offsets = None
                if type(latents) is tuple:
                    latents, offsets = latents

                if keep_last_bins and (self.last_latents is not None):
                    latents = self.last_latents
                else:
                    self.last_latents = latents

                # Take the final action latent
                if self.cfg.experiment.enable_offsets:
                    action_latents = (latents[:, -1:, :], offsets[:, -1:, :])
                else:
                    action_latents = latents[:, -1:, :]
            else:
                action_latents = self.action_ae.sample_latents(
                    num_latents=self.cfg.action_batch_size
                )
            actions = self.action_ae.decode_actions(
                latent_action_batch=action_latents,
                input_rep_batch=enc_obs,
            )
            actions = actions.cpu().numpy()
            if sample:
                sampled_action = np.random.randint(len(actions))
                actions = actions[sampled_action]
                # (seq==1, action_dim), since batch dim reduced by sampling
                actions = einops.rearrange(actions, "1 action_dim -> action_dim")
            else:
                # (batch, seq==1, action_dim)
                actions = einops.rearrange(
                    actions, "batch 1 action_dim -> batch action_dim"
                )
            return actions, (logits_to_save, offsets_to_save, action_latents)

    def run(self):
        rewards = []
        infos = []
        if self.cfg.experiment.lazy_init_models:
            self._init_action_ae()
            self._init_obs_encoding_net()
            self._init_state_prior()
        for i in range(self.cfg.experiment.num_eval_eps):
            reward, obses, actions, latents, info = self.run_single_episode()
            rewards.append(reward)
            infos.append(info)
            torch.save(obses, os.path.join(self.work_dir, f"obses_{i}.pth"))
            torch.save(actions, os.path.join(self.work_dir, f"actions_{i}.pth"))
            torch.save(latents, os.path.join(self.work_dir, f"latents_{i}.pth"))
        self.env.close()
        logging.info(rewards)
        logging.info(infos)
        return rewards, infos

    @property
    def snapshot(self):
        return (
            Path(self.cfg.model.load_dir or self.work_dir)
            / f"snapshot_{self.cfg.experiment.cv_run_idx}.pt"
        )

    def load_snapshot(self):
        keys_to_load = ["action_ae", "obs_encoding_net", "state_prior"]
        with self.snapshot.open("rb") as f:
            payload = torch.load(f, map_location=self.device)
        loaded_keys = []
        for k, v in payload.items():
            if k in keys_to_load:
                loaded_keys.append(k)
                self.__dict__[k] = v.to(self.device)

        if len(loaded_keys) != len(keys_to_load):
            raise ValueError(
                "Snapshot does not contain the following keys: "
                f"{set(keys_to_load) - set(loaded_keys)}"
            )
