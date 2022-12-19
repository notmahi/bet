import logging
from collections import OrderedDict
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from hydra_utils import get_only_swept_params
from models.action_ae.generators.base import GeneratorDataParallel
from models.latent_generators.latent_generator import LatentGeneratorDataParallel
from omegaconf import OmegaConf

import utils
import wandb


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print("Saving to {}".format(self.work_dir))
        self.cfg = cfg
        self.device = (
            torch.device(cfg.experiment.device)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if self.cfg.experiment.data_parallel and self.device == torch.device("cpu"):
            raise ValueError("Data parallel is not supported on CPU")
        utils.set_seed_everywhere(cfg.experiment.seed)
        self.dataset = hydra.utils.call(
            cfg.env.dataset,
            train_fraction=cfg.experiment.train_fraction,
            random_seed=cfg.experiment.seed,
            device=self.device,
        )
        self.train_set, self.test_set = self.dataset
        self._setup_loaders()

        # Create the model.
        self.action_ae = None
        self.obs_encoding_net = None
        self.state_prior = None
        if not self.cfg.experiment.lazy_init_models:
            self._init_action_ae()
            self._init_obs_encoding_net()
            self._init_state_prior()

        self.log_components = OrderedDict()
        self.epoch = self.prior_epoch = 0

        self.save_training_latents = False
        self._training_latents = []

        # WandB initialization
        self.wandb_run = wandb.init(
            dir=str(self.work_dir),
            project=cfg.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )
        wandb.config.update({"save_path": self.work_dir})

    def _init_action_ae(self):
        if self.action_ae is None:  # possibly already initialized from snapshot
            self.action_ae = hydra.utils.instantiate(
                self.cfg.action_interface.action_ae.discretizer, _recursive_=False
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
                latent_dim=self.action_ae.latent_dim,  # defined to 1
                vocab_size=self.action_ae.num_latents,  # num bins
            ).to(self.device)
            if self.cfg.experiment.data_parallel:
                self.state_prior = LatentGeneratorDataParallel(self.state_prior)
            self.state_prior_optimizer = self.state_prior.get_optimizer(
                learning_rate=self.cfg.experiment.lr,
                weight_decay=self.cfg.experiment.weight_decay,
                betas=tuple(self.cfg.experiment.betas),
            )

    def _setup_loaders(self):
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.cfg.experiment.batch_size,
            shuffle=True,
            num_workers=self.cfg.experiment.num_workers,
            pin_memory=True,
        )

        self.test_loader = DataLoader(
            self.test_set,
            batch_size=self.cfg.experiment.batch_size,
            shuffle=False,
            num_workers=self.cfg.experiment.num_workers,
            pin_memory=True,
        )

        self.latent_collection_loader = DataLoader(
            self.train_set,
            batch_size=self.cfg.experiment.batch_size,
            shuffle=False,
            num_workers=self.cfg.experiment.num_workers,
            pin_memory=True,
        )

    def train_prior_one_epoch(self):
        self.state_prior.train()
        with utils.eval_mode(self.obs_encoding_net, self.action_ae):
            pbar = tqdm.tqdm(
                self.train_loader, desc=f"Training prior epoch {self.prior_epoch}"
            )
            for data in pbar:
                observations, action, mask = data
                self.state_prior_optimizer.zero_grad(set_to_none=True)
                obs, act = observations.to(self.device), action.to(self.device)
                enc_obs = self.obs_encoding_net(obs)
                latent = self.action_ae.encode_into_latent(act, enc_obs)
                _, loss, loss_components = self.state_prior.get_latent_and_loss(
                    obs_rep=enc_obs,
                    target_latents=latent,
                    return_loss_components=True,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.state_prior.parameters(), self.cfg.experiment.grad_norm_clip
                )
                self.state_prior_optimizer.step()
                self.log_append("prior_train", len(observations), loss_components)

    def eval_prior(self):
        with utils.eval_mode(
            self.obs_encoding_net, self.action_ae, self.state_prior, no_grad=True
        ):
            for observations, action, mask in self.test_loader:
                obs, act = observations.to(self.device), action.to(self.device)
                enc_obs = self.obs_encoding_net(obs)
                latent = self.action_ae.encode_into_latent(act, enc_obs)
                _, loss, loss_components = self.state_prior.get_latent_and_loss(
                    obs_rep=enc_obs,
                    target_latents=latent,
                    return_loss_components=True,
                )
                self.log_append("prior_eval", len(observations), loss_components)

    def run(self):
        snapshot = self.snapshot
        if snapshot.exists():
            print(f"Resuming: {snapshot}")
            self.load_snapshot()

        if self.cfg.experiment.lazy_init_models:
            self._init_obs_encoding_net()
            self._init_action_ae()
        self.action_ae.fit_model(
            self.train_loader,
            self.test_loader,
            self.obs_encoding_net,
        )
        if self.cfg.experiment.save_latents:
            self.save_latents()

        # Train the action prior model.
        if self.cfg.experiment.lazy_init_models:
            self._init_state_prior()
        self.state_prior_iterator = tqdm.trange(
            self.prior_epoch, self.cfg.experiment.num_prior_epochs
        )
        self.state_prior_iterator.set_description("Training prior: ")

        # Initialize the log.
        self.log_components = OrderedDict()

        # Save initialized model values.
        self.eval_prior()
        self.flush_log(epoch=0, iterator=self.state_prior_iterator)
        log.info(f"Initialization - Current model test loss: {self.current_test_loss}")
        self.best_model_test_loss = self.current_test_loss
        self.save_snapshot()
        log.info(
            f"Initialization - Saved new best model with test loss: {self.current_test_loss}"
        )

        for epoch in self.state_prior_iterator:
            # Train
            self.prior_epoch = epoch
            self.train_prior_one_epoch()
            # Report
            if ((self.prior_epoch + 1) % self.cfg.experiment.eval_prior_every) == 0:
                self.eval_prior()
                self.flush_log(
                    epoch=epoch + self.epoch + 1, iterator=self.state_prior_iterator
                )
            log.info(
                f"Epoch {self.prior_epoch} - Current model test loss: {self.current_test_loss}"
            )
            # self.prior_epoch += 1 TODO: Delete?
            if (
                (self.prior_epoch + 1) % self.cfg.experiment.save_prior_every
            ) == 0 and self.current_test_loss < self.best_model_test_loss:
                self.best_model_test_loss = self.current_test_loss
                self.save_snapshot()
                log.info(
                    f"Epoch {self.prior_epoch} - Saved new best model with test loss: {self.current_test_loss}"
                )

        # Expose DataParallel module class name for wandb tags.
        tag_func = (
            lambda m: m.module.__class__.__name__
            if self.cfg.experiment.data_parallel
            else m.__class__.__name__
        )
        tags = tuple(
            map(tag_func, [self.obs_encoding_net, self.action_ae, self.state_prior])
        )
        self.wandb_run.tags += tags
        self.wandb_run.finish()

    @property
    def snapshot(self):
        return self.work_dir / f"snapshot_{self.cfg.experiment.cv_run_idx}.pt"

    def save_snapshot(self):
        self._keys_to_save = [
            "action_ae",
            "obs_encoding_net",
            "epoch",
            "prior_epoch",
            "state_prior",
        ]
        payload = {k: self.__dict__[k] for k in self._keys_to_save}
        with self.snapshot.open("wb") as f:
            torch.save(payload, f)

    def save_latents(self):
        total_mse_loss = 0
        with utils.eval_mode(self.action_ae, self.obs_encoding_net, no_grad=True):
            for observations, action, mask in self.latent_collection_loader:
                obs, act = observations.to(self.device), action.to(self.device)
                enc_obs = self.obs_encoding_net(obs)
                latent = self.action_ae.encode_into_latent(act, enc_obs)
                reconstructed_action = self.action_ae.decode_actions(
                    latent,
                    enc_obs,
                )
                total_mse_loss += F.mse_loss(act, reconstructed_action, reduction="sum")
                if type(latent) == tuple:
                    # serialize into tensor; assumes last dim is latent dim
                    detached_latents = tuple(x.detach() for x in latent)
                    self._training_latents.append(torch.cat(detached_latents, dim=-1))
                else:
                    self._training_latents.append(latent.detach())
        self._training_latents_tensor = torch.cat(self._training_latents, dim=0)
        logging.info(f"Total MSE reconstruction loss: {total_mse_loss}")
        logging.info(
            f"Average MSE reconstruction loss: {total_mse_loss / len(self._training_latents_tensor)}"
        )
        torch.save(self._training_latents_tensor, self.work_dir / "latents.pt")

    def load_snapshot(self):
        with self.snapshot.open("rb") as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
        not_in_payload = set(self._keys_to_save) - set(payload.keys())
        if len(not_in_payload):
            logging.warning("Keys not found in snapshot: %s", not_in_payload)

    def log_append(self, log_key, length, loss_components):
        """Store info about the different losses components.

        It stores both the information from the training and the validation
        loss, in 3 components: loss in the bin chosen (class), loss in the
        offset and the total loss (sum of the previous two).
        """

        for key, value in loss_components.items():
            # all the `key_name`s used:
            # prior_eval/class, prior_eval/offset, prior_eval/total
            # or
            # prior_train/class, prior_train/offset, prior_train/total
            key_name = f"{log_key}/{key}"
            # set counter and sum to 0
            count, sum = self.log_components.get(key_name, (0, 0.0))
            # for each keyname, assign a tuple:
            # (total number of observations, sum of losses for all observations)
            self.log_components[key_name] = (
                count + length,
                sum + (length * value.detach().cpu().item()),
            )

    def flush_log(self, epoch, iterator):
        log_components = OrderedDict()
        iterator_log_component = OrderedDict()
        # Convert log components in average from tuple (count, sum)
        for key, value in self.log_components.items():
            # Get the average for each
            count, sum = value
            to_log = sum / count
            log_components[key] = to_log
            # Store current model loss
            if key == "prior_eval/total":
                self.current_test_loss = to_log
            # Set the iterator status
            log_key, name_key = key.split("/")
            iterator_log_name = f"{log_key[0]}{name_key[0]}".upper()
            iterator_log_component[iterator_log_name] = to_log
        postfix = ",".join(
            "{}:{:.2e}".format(key, iterator_log_component[key])
            for key in iterator_log_component.keys()
        )
        iterator.set_postfix_str(postfix)
        # Log to w&b
        wandb.log(log_components, step=epoch)
        # Zero the log components
        self.log_components = OrderedDict()


log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("get_only_swept_params", get_only_swept_params)


def run_cross_validation(cfg):
    """
    Run cross validation for the given config.
    cv stands for cross validation.
    """
    cv_losses = []
    for cv_run_idx in range(cfg.experiment.num_cv_runs):
        log.info(f"==== Starting cross-validation run: {cv_run_idx} ====")
        # Change cfg for new cross-validation run.
        cfg.experiment.cv_run_idx = cv_run_idx
        cfg.experiment.seed = cv_run_idx
        # Generate workspace (training).
        workspace = Workspace(cfg)
        workspace.run()
        cv_losses.append(workspace.best_model_test_loss)
        log.info(
            f"Cross-validation run: {cv_run_idx}. "
            f"Best model test loss: {workspace.best_model_test_loss}\n"
            f"Saved in {workspace.work_dir}"
        )
        log.info(f"==== End of run {cv_run_idx} ====\n")

    return cv_losses


@hydra.main(version_base="1.2", config_path="configs", config_name="config_train")
def main(cfg):
    """
    Runs training with the given configuration.
    Performs cross-validation with cfg.experiment.num_cv_runs.
    Saves the best model of each cross-validation run.
    Returns the average test loss of the best model over the cross-validation runs.
    """
    # Run training with multiple seeds and save the best model for each run.
    cv_losses = run_cross_validation(cfg)

    # Average performance across cross-validation runs on test set.
    objective = sum(cv_losses) / len(cv_losses)
    log.info(f"Mean test loss across cross-validation runs: {objective}")
    return objective


if __name__ == "__main__":
    main()
