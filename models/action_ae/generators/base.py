import abc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import utils
from typing import Optional, Dict, Tuple, Any, Union

from models.action_ae import AbstractActionAE


class AbstractGenerator(AbstractActionAE, utils.TrainWithLogger):
    def to(self, device: Union[str, torch.device]) -> None:
        self.device = device
        super().to(device)
        return self

    def fit_model(
        self,
        input_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        obs_encoding_net: Optional[nn.Module] = None,
    ) -> None:
        """
        Given a batch of input actions and states, fit the generator to the data.
        """
        # https://github.com/pytorch/pytorch/blob/master/torch/optim/optimizer.py#L280
        # check if optim already includes obs_encoding_net parameters;
        # if not, add them
        param_list = []
        for elem in self.optimizer.param_groups:
            param_list += elem["params"]
        param_set = set(param_list)
        if not param_set.issuperset(obs_encoding_net.parameters()):
            self.optimizer.add_param_group({"params": obs_encoding_net.parameters()})
        self.iterator = range(self.train_experiment.num_training_epochs)
        self.reset_log()
        for epoch in self.iterator:
            self.epoch = epoch
            self.train_epoch(input_dataloader, obs_encoding_net)
            self.flush_log(epoch)

            if ((self.epoch + 1) % self.train_cfg.experiment.eval_every) == 0:
                self.eval_epoch(eval_dataloader, obs_encoding_net)

            if ((self.epoch + 1) % self.train_cfg.experiment.save_every) == 0:
                self.save_snapshot()

    def eval_epoch(
        self, input_dataloader: DataLoader, obs_encoding_net: Optional[nn.Module] = None
    ) -> None:
        with utils.eval_mode(self, obs_encoding_net):
            for observations, action in input_dataloader:
                obs, act = observations.to(self.device), action.to(self.device)
                enc_obs = obs_encoding_net(obs)
                _, _, loss_components = self.calculate_encodings_and_loss(
                    act, enc_obs, return_all_losses=True
                )
                self.log_append("eval", len(observations), loss_components)

    def train_epoch(
        self, input_dataloader: DataLoader, obs_encoding_net: Optional[nn.Module] = None
    ) -> None:
        self.train()
        obs_encoding_net.train()
        for observations, action, mask in input_dataloader:
            self.optimizer.zero_grad(set_to_none=True)

            obs, act = observations.to(self.device), action.to(self.device)
            enc_obs = obs_encoding_net(obs)
            _, loss, loss_components = self.calculate_encodings_and_loss(
                act, enc_obs, return_all_losses=True
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.parameters(), self.train_cfg.experiment.grad_norm_clip
            )
            nn.utils.clip_grad_norm_(
                obs_encoding_net.parameters(), self.train_cfg.experiment.grad_norm_clip
            )
            self.optimizer.step()
            self.log_append("train", len(observations), loss_components)

    @property
    @abc.abstractmethod
    def optimizer(self) -> optim.Optimizer:
        """
        Returns the optimizer for the generator.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def latent_dim(self) -> int:
        """
        The dimensionality of the latent representation.
        """
        pass

    @abc.abstractmethod
    def calculate_encodings_and_loss(
        self,
        input_action: torch.Tensor,
        input_rep: Optional[torch.Tensor],
        return_all_losses: bool = False,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Given the observation representation and the action, calculate the model losses that should be optimized over.

        Inputs:
        input_action: Batch of the actions taken in the multimodal demonstrations.
        input_rep: Batch of the observations (or representations thereof) where the actions was taken.
        return_all_losses: return a dictionary with all the components of the losses, useful for logging purposes.

        Outputs:
        total_loss: The total loss calculated from the network and the algorithm, taking into account all of the components coming into the loss from the algorithm.
        loss_components: A dictionary mapping a loss component name to a scalar value which denotes the loss from the component, useful for logging purposes. Will be empty if return_all_losses is false.
        """
        return

    @abc.abstractmethod
    def sample_latents(self, num_latents: Optional[int] = None) -> torch.Tensor:
        """
        Sample possible latents from this generator class.

        Inputs:
        num_latents: Number of latents to sample. For generators with discrete latents, num_latents = None implies returning all latents, for continous classes, that should return an error.

        Returns:
        latents: a torch.tensor of the right dimensions filled with latents that can be fed directly to decode_actions.
        """
        return


class GeneratorDataParallel(nn.DataParallel):
    def encode_into_latent(self, *args, **kwargs):
        return self.module.encode_into_latent(*args, **kwargs)

    def calculate_encodings_and_loss(self, *args, **kwargs):
        return self.module.calculate_encodings_and_loss(*args, **kwargs)

    def decode_actions(self, *args, **kwargs):
        return self.module.decode_actions(*args, **kwargs)

    def sample_latents(self, *args, **kwargs):
        return self.module.sample_latents(*args, **kwargs)

    def fit_model(self, *args, **kwargs):
        return self.module.fit_model(*args, **kwargs)

    @property
    def latent_dim(self) -> int:
        return self.module.latent_dim

    @property
    def num_latents(self) -> int:
        return self.module.num_latents
