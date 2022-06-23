import abc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Tuple, Optional

from utils import SaveModule, TrainWithLogger

from models.action_ae import AbstractActionAE


class AbstractDiscretizer(AbstractActionAE, TrainWithLogger):
    """
    Abstract discretizer class that defines the interface for action discretization.
    """

    def fit_model(
        self,
        input_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        obs_encoding_net: Optional[nn.Module] = None,
    ) -> None:
        """
        Given a batch of input actions, fit the discretizer to the data.
        """
        all_action_tensors = []
        if hasattr(input_dataloader.dataset, "_get_all_actions"):
            all_action_tensors = (
                input_dataloader.dataset._get_all_actions()
            )  # N x T x action_dim
        else:
            for _, action, _ in input_dataloader:
                action_dim = action.shape[-1]
                all_action_tensors.append(action.view(-1, action_dim))
            all_action_tensors = torch.cat(all_action_tensors, dim=0)
        self.fit_discretizer(all_action_tensors)

    @abc.abstractmethod
    def fit_discretizer(self, input_actions: torch.Tensor) -> None:
        """
        Given a batch of input actions, fit the discretizer to the data.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def discretized_space(self) -> Tuple[int, int]:
        """
        The discretized space of the discretizer.

        Outputs:
        (num_tokens, token_dimension): The number of tokens and the dimension of the tokens per discretization.
        """
        raise NotImplementedError
