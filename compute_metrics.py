import hydra


@hydra.main(version_base="1.2", config_path="configs", config_name="FIX_ME")
def main(cfg):
    # Config should just take the folder of an evaluation job:
    # e.g. eval_runs/eval_blockpush/reproduction/ablation_focal_loss/0

    # SHould log in wandb the configs below
    # Should relog the `train_config:`
    # Should log the `eval_config:`
    # Should log tags in wandb about which job ran e.g.
    # can be gotten from   save_path: /opt/project/eval_runs/eval_blockpush/reproduction/ablation_focal_loss/0/snapshot_0
    # ablation_focal_loss -> tags: ablation, focal, loss (just a split)

    # Should compute the metrics
    # should log the metrics to wandb
    # Should save the logs in a file. (serialize a dictionary.)

    pass


if __name__ == "__main__":
    main()
