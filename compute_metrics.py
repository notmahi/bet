import hydra
import json
import numpy as np
import utils.metrics
import wandb
import omegaconf
from pathlib import Path
from omegaconf import OmegaConf


def get_train_and_eval_log(snapshot_dir):

    local_path_to_wandb_config = Path("wandb", "latest-run", "files", "config.yaml")
    wandb_config_path = snapshot_dir / local_path_to_wandb_config

    wandb_config = OmegaConf.load(str(wandb_config_path))
    train_config = OmegaConf.to_container(
        wandb_config["train_config"]["value"], resolve=True
    )

    eval_config = {}
    for key, value in wandb_config.items():
        if key != "train_config":
            if type(value) != omegaconf.dictconfig.DictConfig:
                eval_config[key] = value
            else:
                eval_config[key] = OmegaConf.to_container(value, resolve=True)

    return train_config, eval_config


def get_environment_name(train_config):
    return train_config["env"]["name"]


def process_tags(tags):
    return tuple(tags.split(","))


def check_eval_vectorized(eval_config):
    return eval_config["experiment"]["value"]["vectorized_env"]


def get_snapshot_metrics(file_name, env_name, eval_was_vectorized):

    # Load the rollout
    rollout = np.load(file_name, allow_pickle=True)
    print(rollout.shape)

    # Compute metrics for blockpush
    if env_name == "blockpush":
        snapshot_metrics = utils.metrics.compute_blockpush_metrics(
            rollout, tolerance=0.05
        )

    # Compute metrics for kitchen
    elif env_name == "kitchen":
        (
            elements,
            mappings,
            timesteps,
        ) = utils.metrics.compute_kitchen_sequences(rollout, treshhold=0.3)
        # Compute the entropy
        count_seq, e = utils.metrics.compute_task_entropy(mappings)
        # Compute table 1 metrics
        table1_metrics = utils.metrics.compute_kitchen_metrics(elements)
        snapshot_metrics = (e, table1_metrics)

    return snapshot_metrics


@hydra.main(version_base="1.2", config_path="configs", config_name="config_metrics")
def main(cfg):

    # Should log in wandb the configs below
    # done Should relog the `train_config:`
    # done Should log the `eval_config:`
    # done Should log tags in wandb about which job ran e.g.
    # done can be gotten from   save_path: /opt/project/eval_runs/eval_blockpush/reproduction/ablation_focal_loss/0/snapshot_0
    # done ablation_focal_loss -> tags: ablation, focal, loss (just a split)

    # Should compute the metrics
    # should log the metrics to wandb
    # Should save the logs in a file. (serialize a dictionary.)

    # iterate over a folder, for each subfolder, load the rollout and compute the entropy.
    # use pathlib to iterate over the subfolders

    # Log in wandb
    # wandb.init(
    #     dir=self.work_dir,
    #     project=cfg.project,
    #     config=OmegaConf.to_container(cfg, resolve=True),
    #     reinit=True,
    # )
    # wandb.config.update({"save_path": self.work_dir})
    # # Add to eval config the config from the training model
    # train_config_path = os.path.join(cfg.model.load_dir, ".hydra", "config.yaml")
    # train_config = OmegaConf.load(train_config_path)
    # wandb.config.update(
    #     {"train_config": OmegaConf.to_container(train_config, resolve=True)}
    # )

    work_dir = Path.cwd()
    print(f"Saving to {work_dir}")

    load_dir = Path(cfg.load_dir)

    # Ensure that the dir provided is correct
    if len(list(load_dir.glob("*"))) == 0:
        raise Exception("The provided directory is empty")

    for subdir in load_dir.glob("*"):
        # if there is a snapshot folder (has snaphot in the name) print the full name
        if "snapshot_" in str(subdir):
            snapshot_dir = load_dir / subdir
            train_config, eval_config = get_train_and_eval_log(snapshot_dir)
            tags = process_tags(cfg.tags)

            wandb_run = wandb.init(
                dir=work_dir,
                project=cfg.project,
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=True,
            )
            wandb.config.update(
                {
                    "train_config": train_config,
                    "eval_config": eval_config,
                }
            )
            wandb_run.tags += tags

            env_name = get_environment_name(train_config)
            eval_was_vectorized = check_eval_vectorized(eval_config)

            for file_name in snapshot_dir.glob("*"):
                # load the observations (has obs in the name)
                if "obs" in str(file_name):

                    snapshot_metrics = get_snapshot_metrics(
                        file_name=str(file_name),
                        env_name=env_name,
                        eval_was_vectorized=eval_was_vectorized,
                    )
                    print(eval_was_vectorized)
                    # snapshot_metrics = {"test": 1}

                    # Store computed metrics
                    snapshot_name = snapshot_dir.stem

            with open(snapshot_name + "_metrics", "w") as f:
                json.dump(snapshot_metrics, f, indent=4)
            wandb.log(snapshot_metrics)
            wandb.finish()


if __name__ == "__main__":
    main()
