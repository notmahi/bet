import torch
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional, Tuple, Dict

from models.mlp import MLP
from models.action_ae.generators.base import AbstractGenerator
from typing import Optional, Any, Union

from omegaconf import DictConfig
import hydra


class IdentityGenerator(AbstractGenerator):
    def __init__(
        self,
        optimizer_cfg: Optional[DictConfig] = None,
        train_cfg: Optional[DictConfig] = None,
        output_mod: Optional[DictConfig] = None,
        input_dim: int = 1,
        output_dim: int = 1,
        *args,
        **kwargs,
    ):
        super(IdentityGenerator, self).__init__(*args, **kwargs)
        self.train_cfg = train_cfg
        self.input_dim = input_dim
        self.output_dim = output_dim

    def fit_model(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        Given a batch of input actions and states, fit the generator to the data.
        """
        return

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        pass

    def calculate_encodings_and_loss(
        self,
        act: torch.Tensor,
        obs_enc: torch.Tensor,
        return_all_losses: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        return (
            act,
            torch.zeros(1).to(act.device),
            OrderedDict(loss=torch.zeros(1)),
        )

    @property
    def latent_dim(self) -> int:
        return self.output_dim

    def decode_actions(
        self,
        latent_action_batch: Optional[torch.Tensor],
        input_rep_batch: Optional[torch.Tensor] = None,
    ):
        return latent_action_batch

    def sample_latents(self, num_latents: int):
        return torch.zeros(num_latents, 1)

    def encode_into_latent(
        self, input_action: torch.Tensor, input_rep: Optional[torch.Tensor]
    ) -> Any:
        return input_action

    @property
    def num_latents(self) -> Union[int, float]:
        return 1
