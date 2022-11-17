# Behavior Transformers: Cloning k modes with one stone

[[Arxiv]](https://arxiv.org/abs/2206.11251) [[Code]](https://github.com/notmahi/bet) [[Data]](https://osf.io/983qz/) [[Project page and videos]](https://mahis.life/bet/)

Created by [Nur Muhammad (Mahi) Shafiullah](https://mahis.life), [Zichen Jeff Cui](https://jeffcui.com/), [Ariuntuya (Arty) Altanzaya](https://www.artys.page/), and [Lerrel Pinto](https://www.lerrelpinto.com/) at New York University.



https://user-images.githubusercontent.com/3000253/174540589-35c2d51b-3a5b-42d8-9f80-483b61df91a7.mp4



## Abstract
While behavior learning has made impressive progress in recent times, it lags behind computer vision and natural language processing due to its inability to leverage large, human generated datasets. Human behavior has a wide variance, multiple modes, and human demonstrations naturally don’t come with reward labels. These properties limit the applicability of current methods in Offline RL and Behavioral Cloning to learn from large, pre-collected datasets. In this work, we present Behavior Transformer (BeT), a new technique to model unlabeled demonstration data with multiple modes. BeT retrofits standard transformer architectures with action discretization coupled with a multi-task action correction inspired by offset prediction in object detection. This allows us to leverage the multi-modal modeling ability of modern transformers to predict multi-modal continuous actions. We experimentally evaluate BeT on a variety of robotic manipulation and self-driving behavior datasets. We show that BeT significantly improves over prior state-of-the-art work on solving demonstrated tasks while capturing the major modes present in the pre-collected datasets. Finally, through an extensive ablation study we further analyze the importance of every crucial component in BeT.

## Code release

In this repository, you can find the code to reproduce Behavior Transformer (BeT). The following assumes our current working directory is the root folder of this project repository; tested on Ubuntu 20.04 LTS (amd64).

## Getting started
### Setting up the project environments
- Install the project environment:
  ```
  conda env create --file=conda_env.yml
  ```
- Activate the environment:
  ```
  conda activate behavior-transformer
  ```
- Clone the Relay Policy Learning repo:
  ```
  git clone https://github.com/google-research/relay-policy-learning
  ```
- Install MuJoCo 2.1.0: https://github.com/openai/mujoco-py#install-mujoco
- Install CARLA server 0.9.13: https://carla.readthedocs.io/en/0.9.13/start_quickstart/#a-debian-carla-installation
- To enable logging, log in with a `wandb` account:
  ```
  wandb login
  ```
  Alternatively, to disable logging altogether, set the environment variable `WANDB_MODE`:
  ```
  export WANDB_MODE=disabled
  ```

### Getting the training datasets
Datasets used for training will be available at this OSF link: [https://osf.io/983qz/](https://osf.io/983qz/).
- Download and extract the datasets from the tar archive.
- Activate the conda environment with `conda activate behavior-transformer`.
- In the extracted folder, run `python3 process_carla.py carla` to preprocess the CARLA dataset into tensors.
- In `./config/env_vars/env_vars.yaml`, set the dataset paths to the unzipped directories.
  - `carla_multipath_town04_merge`: CARLA environment
  - `relay_kitchen`: Franka kitchen environment
  - `multimodal_push_fixed_target`: Block push environment

## Reproducing experiments
The following assumes our current working directory is the root folder of this project repository.

To reproduce the experiment results, the overall steps are:
1. Activate the conda environment with
   ```
   conda activate behavior-transformer
   ```
2. Train with `python3 train.py`. A model snapshot will be saved to `./exp_local/...`;
3. In the corresponding environment config, set the `load_dir` to the absolute path of the snapshot directory above;
4. Eval with `python3 run_on_env.py`.

See below for detailed steps for each environment.

### CARLA

- Train:
  ```
  python3 train.py --config-name=train_carla
  ```
  Snapshots will be saved to a new timestamped directory `./exp_local/{date}/{time}_carla_train`
- In `configs/env/carla_multipath_merge_town04_traj_rep.yaml`, set `load_dir` to the absolute path of the directory above.
- Evaluation:
  ```
  python3 run_on_env.py --config-name=eval_carla
  ```

### Franka kitchen

- Train:
  ```
  python3 train.py --config-name=train_kitchen
  ```
  Snapshots will be saved to a new timestamped directory `./exp_local/{date}/{time}_kitchen_train`
- In `configs/env/relay_kitchen_traj.yaml`, set `load_dir` to the absolute path of the directory above.
- Evaluation:
  ```
  export PYTHONPATH=$PYTHONPATH:$(pwd)/relay-policy-learning/adept_envs
  python3 run_on_env.py --config-name=eval_kitchen
  ```
  (Evaluation requires including the relay policy learning repo in `PYTHONPATH`.)

### Block push
Update (11/17/22): There was a small error in the published hyperparameters for Block push (namely `window_size` and `batch_size`). We have updated the hyperparameters in the config files to the parameters that replicate the results in the paper. We apologize for any inconvenience.
- Train:
  ```
  python3 train.py --config-name=train_blockpush
  ```
  Snapshots will be saved to a new timestamped directory `./exp_local/{date}/{time}_blockpush_train`
- In `configs/env/block_pushing_multimodal_fixed_target.yaml`, set `load_dir` to the absolute path of the directory above.
- Evaluation:
  ```
  ASSET_PATH=$(pwd) python3 run_on_env.py --config-name=eval_blockpush
  ```
  (Evaluation requires including this repository in `ASSET_PATH`.)
</details>

### Speeding up evaluation
- Rendering can be disabled for the kitchen and block pushing environments: set `enable_render: False` in `configs/eval_kitchen.yaml`, `configs/eval_blockpush.yaml`.
  
  (This option does not affect CARLA, as it requires rendering for RGB camera observations.)
- CARLA (Unreal Engine 4) renders on GPU 0 by default. If multiple GPUs are available, running the evaluated model on other GPUs can speed up evaluation: e.g. set `device: cuda:1` in `configs/eval_carla.yaml`.

## Acknowledgements
We are indebted to the following codebases and tools for making our lives significantly easier.
- [karpathy/MinGPT](https://github.com/karpathy/minGPT): MinGPT implementation and hyperparameters.
- [facebookresearch/hydra](https://github.com/facebookresearch/hydra): Configuration managements.
- [psf/black](https://github.com/psf/black): Linting.
