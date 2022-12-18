import logging
import joblib
import os
from pathlib import Path

import hydra
import numpy as np

from workspaces.vec_base import VecWorkspace


def run_evaluations(cfg):
    cv_run_idxs = process_cv_run_idxs(cfg)
    for cv_run_idx in cv_run_idxs:
        log.info(f"==== Starting evaluation for snapshot_{cv_run_idx} ====")
        snapshot_cfg = cfg
        snapshot_cfg.experiment.cv_run_idx = cv_run_idx
        run_snapshot(snapshot_cfg)
        log.info(f"==== End of evaluation for snapshot_{cv_run_idx} ====\n")


def process_cv_run_idxs(cfg):
    cv_run_idx = cfg.experiment.cv_run_idx
    if cv_run_idx == "all":
        list_snapshots = [
            filename
            for filename in os.listdir(cfg.model.load_dir)
            if "snapshot_" in filename
        ]
        cv_run_idx = sorted(
            [int(snapshot_name[9:-3]) for snapshot_name in list_snapshots]
        )
    elif type(cv_run_idx) is int:
        cv_run_idx = [cv_run_idx]
    else:
        raise Exception("Format of experiment.cv_run_idx not supported.")

    return cv_run_idx


def run_snapshot(cfg):
    if cfg.experiment.vectorized_env:
        # This workspace does not support all the features in the non-vectorized workspaces.
        # It's unique purpose is  generate rollouts quickly.
        workspace = VecWorkspace(cfg)
    else:
        # Needs _recursive_: False since we have more objects within that we are instantiating
        # without using nested instantiation from hydra
        workspace = hydra.utils.instantiate(
            cfg.env.workspace, cfg=cfg, _recursive_=False
        )
    rewards, infos = workspace.run()
    print(rewards)
    print(infos)
    print(f"Average reward: {np.mean(rewards)}")
    print(f"Std: {np.std(rewards)}")


log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs", config_name="config_eval")
def main(cfg):
    run_evaluations(cfg)


if __name__ == "__main__":
    main()
